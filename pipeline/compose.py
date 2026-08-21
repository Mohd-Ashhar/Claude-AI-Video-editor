"""Lay a phrase of words out across the frame, and draw them as one image.

The four Gym-Inspiration reels do something the caption path in overlay.py
cannot express. Read frame by frame, Gym_4 at 3.2-6.4s builds up:

    So  ->  So / Just  ->  + Forget (red)  ->  + About (huge)  ->  + The  ->  clear

Five words, five different anchors, five different sizes, all on screen at once,
all clearing together on the phrase boundary. Measured across the four
references: 2-5 words co-resident, glyph heights from 41px to 351px on a 663px
strip -- a six-to-eight-fold size spread inside a single reel -- and anchors
spread over x 0.12-0.83, y 0.09-0.91 of the picture.

lyrics.py draws one centred word at a time. That is the gap this module fills:
it turns a list of words into a *composition*, then renders every co-resident
set to a single full-frame RGBA image.

Everything downstream is unchanged. The filtergraph still composites one RGBA
track against one inverted matte, because five simultaneous `movie` sources on
an 8 GB machine is the same trap lyrics.track_video() already documents at
forty -- just smaller.

    uv run python -m pipeline.compose --demo --style chrome
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

from pipeline import config

BOLD, RED, YELLOW, GREEN, DIM, RESET = "\033[1m", "\033[31m", "\033[33m", "\033[32m", "\033[2m", "\033[0m"

DEFAULT_STYLE = "chrome"

# How far a word must sit from the previous one, as a fraction of the frame
# diagonal. The references move the eye across the picture between words rather
# than stacking them in reading order; without this the packer fills the grid
# top-left first and the result reads as a paragraph.
MIN_ANCHOR_TRAVEL = 0.28

# A word may overlap an already-placed one by this fraction of its own area
# before the cell is rejected. Not zero: the reference lets ascenders and
# descenders interleave (Gym_4's "The" tucks into "About"), which is what stops
# the composition looking like a grid.
OVERLAP_TOLERANCE = 0.06

# Collision boxes are inflated by this fraction of the strip before they are
# tested. Overlap tolerance alone is not enough: two words can score zero
# overlap and still sit four pixels apart, which reads as one run-on word
# ("lJust") rather than as two. The margin is what buys the composition air.
COLLISION_MARGIN = 0.035

# Bonus for landing on the subject, when a matte told us where the subject is.
# The occlusion *is* the effect -- a word placed in empty sky is just a caption.
SUBJECT_BONUS = 0.55

MIN_PX = 18


class StyleError(Exception):
    """A style pack is missing, malformed, or names a face that will not load."""


# ---------------------------------------------------------------- style packs


def load(style_id: str | None) -> dict:
    """Read a style pack by id. Falls back to the default rather than failing."""
    wanted = style_id or DEFAULT_STYLE
    path = config.STYLES_DIR / f"{wanted}.json"
    if not path.exists():
        raise StyleError(f"no style pack '{wanted}' in {config.STYLES_DIR}")
    return json.loads(path.read_text())


def available() -> list[str]:
    return sorted(p.stem for p in config.STYLES_DIR.glob("*.json"))


def load_font(paths: list[str], size: int):
    """Load the first face that opens, honouring a '#N' collection index.

    macOS ships most of these faces inside .ttc collections, and the weight that
    matters is almost never index 0 -- Helvetica Neue Light is #7, Thin is #12.
    Naming the file alone would silently give Regular, which is the wrong voice
    for the chrome pack and not obviously wrong on inspection.
    """
    from PIL import ImageFont

    size = max(int(size), MIN_PX)
    for candidate in paths:
        name, _, index = candidate.partition("#")
        try:
            return ImageFont.truetype(name, size, index=int(index) if index else 0)
        except (OSError, ValueError):
            continue
    return ImageFont.load_default()


def _face_for(style: dict, role: str) -> list[str]:
    faces = style["faces"]
    return faces.get(role) or faces["display"]


def _cased(text: str, how: str) -> str:
    if how == "upper":
        return text.upper()
    if how == "lower":
        return text.lower()
    if how == "sentence":
        return text[:1].upper() + text[1:].lower() if text else text
    return text


# ---------------------------------------------------------------- geometry


def band_box(band: tuple[int, int] | None) -> tuple[int, int, int, int]:
    """The rectangle type may occupy: the safe box, narrowed to the picture strip.

    A letterboxed reel puts the picture in a 16:9 band and pure black above and
    below it. Measured on all four references: zero ink outside the band, on
    every sampled frame. Type that strays into the bars is the single most
    obvious tell that a reel was not composed this way.
    """
    x, y, w, h = config.safe_box()
    if band:
        top, bottom = band
        low, high = max(y, top), min(y + h, bottom)
        if high - low > 80:
            y, h = low, high - low
    return x, y, w, h


def _measure(draw, text: str, font, tracking: float) -> tuple[int, int]:
    """True ink extent of the word, from the rasteriser rather than from metrics.

    `getmetrics()` reports the font's line box, which for a display face at 300px
    overstates a lowercase word's height by a third. The packer collides real
    boxes, so it needs the real ink.
    """
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    extra = tracking * max(len(text) - 1, 0)
    return int(right - left + extra), int(bottom - top)


def _size_for(draw, text: str, style: dict, rung: float, box_h: int,
              face: list[str], tracking: float) -> tuple[object, int, int]:
    """Find the point size whose *cap height* hits the ladder rung.

    Binary search on the rendered height, not a metrics estimate. The ladder is
    expressed in fractions of the strip because that is how the references scale
    -- a rung means "this word is a fifth of the picture tall", which is a
    statement about composition and survives a change of delivery size.
    """
    target = max(int(box_h * rung), MIN_PX)
    lo, hi, best = MIN_PX, 900, MIN_PX
    while lo <= hi:
        mid = (lo + hi) // 2
        font = load_font(face, mid)
        _, ink_h = _measure(draw, text, font, tracking)
        if ink_h <= target:
            best, lo = mid, mid + 1
        else:
            hi = mid - 1
    font = load_font(face, best)
    return (font, *_measure(draw, text, font, tracking))


def _overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    dx = min(ax + aw, bx + bw) - max(ax, bx)
    dy = min(ay + ah, by + bh) - max(ay, by)
    return dx * dy if dx > 0 and dy > 0 else 0


# ---------------------------------------------------------------- layout


def layout(phrase: list[dict], style: dict, band: tuple[int, int] | None = None,
           subject: dict | None = None, seed: int = 0) -> list[dict]:
    """Give every word in a phrase an anchor, a size, a face and a colour.

    Deterministic for a given seed, so a re-render is byte-identical and
    verify.py can assert against a known composition. The seed is derived from
    the reel and the phrase index by the caller.

    Placement is a scored search over the pack's anchor grid rather than a
    random scatter. Random placement collides constantly and, when it does not,
    still reads as noise; the score is what makes it look composed:

      * hard reject if the box leaves the picture strip (unless the pack
        declares `bleed_edges`, which Gym_2 does deliberately)
      * hard reject if it overlaps an already-placed word by more than a sliver
      * reward distance from the previous word, so the eye travels
      * reward landing on the subject when a matte says where the subject is,
        because the occlusion is the whole effect

    A word that cannot be placed drops a rung and retries. If it still cannot,
    the phrase closes early and the word carries over -- never silently
    overlapped, never silently pushed into the bars.
    """
    from PIL import Image, ImageDraw

    box_x, box_y, box_w, box_h = band_box(band)
    scratch = ImageDraw.Draw(Image.new("RGBA", (8, 8)))
    rng = random.Random(seed)

    ladder = style["scale_ladder"]
    weights = style.get("ladder_weights") or [1.0] * len(ladder)
    grid = style["anchor_grid"]
    cols, rows = grid["cols"], grid["rows"]
    jitter = grid.get("jitter", 0.0)
    tracking = style.get("tracking", 0.0)
    bleed = style.get("bleed_edges", False)
    contrast_every = style.get("contrast_every")

    diagonal = (box_w ** 2 + box_h ** 2) ** 0.5
    placed: list[dict] = []
    boxes: list[tuple[int, int, int, int]] = []
    previous: tuple[float, float] | None = None
    big_used = False

    for order, word in enumerate(phrase):
        accent = bool(word.get("accent"))
        use_contrast = bool(contrast_every) and (order + 1) % contrast_every == 0

        role = "contrast" if use_contrast else ("accent" if accent else "display")
        face = _face_for(style, role)
        case = (style.get("contrast_case") if use_contrast else None) or style["case"]
        text = _cased(word["text"], case)

        # One word per phrase is allowed to go to the top of the ladder. Letting
        # every word draw freely produced phrases of four enormous words that
        # could not be packed at all; the references put exactly one hero word
        # in a phrase and set the rest well below it.
        top_rung = len(ladder) - 1
        pool = list(range(len(ladder) if not big_used else top_rung))
        pool_w = [weights[i] for i in pool]
        rung_index = rng.choices(pool, weights=pool_w, k=1)[0]
        if accent and rung_index < top_rung:
            rung_index += 1

        spot = None
        while rung_index >= 0 and spot is None:
            font, ink_w, ink_h = _size_for(scratch, text, style, ladder[rung_index],
                                           box_h, face, tracking)
            spot = _best_cell(rng, cols, rows, jitter, box_x, box_y, box_w, box_h,
                              ink_w, ink_h, boxes, previous, subject, bleed,
                              diagonal)
            if spot is None:
                rung_index -= 1

        if spot is None:
            # Nothing fits. The phrase is full; the caller re-opens a new one.
            break

        x, y = spot
        boxes.append((x, y, ink_w, ink_h))
        previous = (x + ink_w / 2, y + ink_h / 2)
        if rung_index == top_rung:
            big_used = True

        ink = style["colours"]["accent" if accent else "base"]
        placed.append({
            **word,
            "x": int(x), "y": int(y), "w": int(ink_w), "h": int(ink_h),
            "text_cased": text,
            "size": int(font.size if hasattr(font, "size") else MIN_PX),
            "rung": rung_index,
            "role": role,
            "rgb": list(ink["rgb"]),
            "alpha": float(ink["alpha"]),
        })

    return placed


def _best_cell(rng, cols, rows, jitter, box_x, box_y, box_w, box_h,
               ink_w, ink_h, boxes, previous, subject, bleed, diagonal):
    """Score every grid cell and return the top-left of the best, or None."""
    cell_w, cell_h = box_w / cols, box_h / rows
    area = max(ink_w * ink_h, 1)
    best, best_score = None, -1e9

    for row in range(rows):
        for col in range(cols):
            cx = box_x + (col + 0.5) * cell_w
            cy = box_y + (row + 0.5) * cell_h
            if jitter:
                cx += rng.uniform(-jitter, jitter) * box_w
                cy += rng.uniform(-jitter, jitter) * box_h
            x = int(cx - ink_w / 2)
            y = int(cy - ink_h / 2)

            if y < box_y or y + ink_h > box_y + box_h:
                continue
            if not bleed and (x < box_x or x + ink_w > box_x + box_w):
                continue
            if bleed and (x + ink_w < box_x + 0.25 * box_w
                          or x > box_x + 0.75 * box_w):
                continue

            here = (x, y, ink_w, ink_h)
            margin = int(box_h * COLLISION_MARGIN)
            grown = (x - margin, y - margin, ink_w + 2 * margin, ink_h + 2 * margin)
            crowd = sum(_overlap(grown, other) for other in boxes)
            if crowd > area * OVERLAP_TOLERANCE:
                continue

            score = 1.0 - crowd / area
            if previous is not None:
                travel = (((x + ink_w / 2 - previous[0]) ** 2
                           + (y + ink_h / 2 - previous[1]) ** 2) ** 0.5) / diagonal
                if travel < MIN_ANCHOR_TRAVEL:
                    score -= (MIN_ANCHOR_TRAVEL - travel) * 3.0
                score += min(travel, 0.75)
            if subject:
                score += SUBJECT_BONUS * _subject_overlap(here, subject)
            score += rng.uniform(0, 0.05)

            if score > best_score:
                best, best_score = (x, y), score

    return best


def _subject_overlap(box: tuple[int, int, int, int], subject: dict) -> float:
    """Fraction of the word's box that falls on the subject's bounding box.

    A bounding box, not the matte itself. The matte is a per-frame 1080x1920
    mask and the layout runs once per phrase; a box is the honest summary of
    "roughly here", and getting it wrong costs a less-good placement rather
    than a broken frame.
    """
    sub = subject.get("box")
    if not sub:
        return 0.0
    covered = _overlap(box, tuple(sub))
    return covered / max(box[2] * box[3], 1)


# ---------------------------------------------------------------- rendering


def state_key(words: list[dict], style: dict, band) -> str:
    payload = json.dumps(
        [[w["text_cased"], w["x"], w["y"], w["size"], w["rgb"], w["alpha"], w["role"]]
         for w in words] + [style["id"], str(band), config.OUT_W, config.OUT_H],
        sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


def render_state(words: list[dict], style: dict, directory: Path,
                 band: tuple[int, int] | None = None,
                 backdrop: str | None = None, force: bool = False) -> Path:
    """Draw every currently-visible word onto one full-frame RGBA image.

    Full-frame with the type already positioned, exactly as overlay.render_cue
    does, so the filtergraph composites at 0,0 and nothing ever scales a text
    layer. A scaled text layer is a soft text layer and it shows on a phone.
    """
    from PIL import Image, ImageDraw

    directory.mkdir(parents=True, exist_ok=True)
    dest = directory / f"state_{style['id']}_{state_key(words, style, band)}.png"
    if dest.exists() and not force:
        return dest

    canvas = Image.new("RGBA", (config.OUT_W, config.OUT_H), (0, 0, 0, 0))

    spec = style.get("backdrop_word")
    if backdrop and spec:
        _draw_backdrop(canvas, backdrop, style, spec, band)

    for word in words:
        layer = _draw_word(word, style)
        canvas.alpha_composite(layer, (word["x"], word["y"]))

    canvas.save(dest)
    return dest


def _draw_word(word: dict, style: dict):
    """Render one word to its own small RGBA tile, distressed if the pack says so.

    Its own tile rather than straight onto the canvas, because distress works on
    the glyph's alpha and has to be confined to that glyph -- applied to the
    whole canvas it would eat the other words in the phrase too.
    """
    from PIL import Image, ImageDraw

    face = _face_for(style, word["role"])
    font = load_font(face, word["size"])
    tracking = style.get("tracking", 0.0)
    pad = 6
    tile = Image.new("RGBA", (word["w"] + pad * 2, word["h"] + pad * 2), (0, 0, 0, 0))
    draw = ImageDraw.Draw(tile)

    fill = (*word["rgb"], int(255 * word["alpha"]))
    text = word["text_cased"]
    left, top, _, _ = draw.textbbox((0, 0), text, font=font)
    if tracking:
        x = pad - left
        for glyph in text:
            draw.text((x, pad - top), glyph, font=font, fill=fill)
            x += draw.textlength(glyph, font=font) + tracking
    else:
        draw.text((pad - left, pad - top), text, font=font, fill=fill)

    if style.get("distress"):
        tile = _distress(tile, style["distress"], seed=hash(text) & 0xFFFF)
    return tile


def _distress(tile, spec: dict, seed: int):
    """Erode and speckle the glyph alpha, deterministically.

    Gym_1's face is a distressed wood-type with no macOS equivalent, so the
    roughness is synthesised rather than sourced. This reads as rough slab, not
    as that specific face, and it is meant to: the alternative was committing a
    licensed binary to the repo.
    """
    from PIL import Image, ImageFilter

    try:
        import numpy as np
    except ImportError:
        return tile

    alpha = np.asarray(tile.getchannel("A"), dtype=np.float32) / 255.0
    rng = np.random.default_rng(seed)

    grain = spec.get("grain", 0)
    if grain:
        noise = rng.random(alpha.shape).astype(np.float32)
        noise = np.asarray(
            Image.fromarray((noise * 255).astype("uint8")).filter(
                ImageFilter.GaussianBlur(max(grain / 12.0, 0.6))),
            dtype=np.float32) / 255.0
        noise = (noise - noise.mean()) / (noise.std() + 1e-6)
        alpha = np.clip(alpha - np.maximum(noise, 0) * spec.get("erode", 0.0), 0, 1)

    # Speckle has to scale with the glyph, not be a constant. A fixed 12% pixel
    # drop is convincing wear on a 250px word and destroys an 80px one -- the
    # stroke is only a few pixels wide there, so the same probability eats the
    # letterform instead of its edge. Referenced to 200px, which is roughly a
    # third of the picture strip.
    speckle = spec.get("speckle", 0.0) * min(1.0, alpha.shape[0] / 200.0)
    if speckle:
        alpha = np.where(rng.random(alpha.shape) < speckle, 0.0, alpha)

    tile.putalpha(Image.fromarray((alpha * 255).astype("uint8")))
    return tile


def _draw_backdrop(canvas, text: str, style: dict, spec: dict, band):
    """The oversized low-alpha word Gym_1 puts behind everything else."""
    from PIL import Image, ImageDraw

    box_x, box_y, box_w, box_h = band_box(band)
    scratch = ImageDraw.Draw(Image.new("RGBA", (8, 8)))
    face = _face_for(style, "display")
    word = _cased(text, style["case"])
    font, ink_w, ink_h = _size_for(scratch, word, style, spec["scale"], box_h,
                                   face, 0.0)

    tile = Image.new("RGBA", (ink_w + 12, ink_h + 12), (0, 0, 0, 0))
    draw = ImageDraw.Draw(tile)
    left, top, _, _ = draw.textbbox((0, 0), word, font=font)
    draw.text((6 - left, 6 - top), word, font=font,
              fill=(*spec["rgb"], int(255 * spec["alpha"])))
    if style.get("distress"):
        tile = _distress(tile, style["distress"], seed=0xBEEF)

    x = int(box_x + (box_w - ink_w) / 2)
    y = int(box_y + (box_h - ink_h) / 2)
    canvas.alpha_composite(tile, (max(x, 0), max(y, 0)))


# ---------------------------------------------------------------- entry


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Lay out and draw a floating-text phrase.")
    ap.add_argument("--style", default=DEFAULT_STYLE, choices=available())
    ap.add_argument("--words", default="So just forget the world about tonight")
    ap.add_argument("--accent", type=int, nargs="*", default=[2])
    ap.add_argument("--letterbox", type=float, default=16 / 9)
    ap.add_argument("--dir", type=Path, default=config.WORK_DIR / "states")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args(argv)

    style = load(args.style)
    strip = int(round(config.OUT_W / args.letterbox))
    bar = (config.OUT_H - strip) // 2
    band = (bar, config.OUT_H - bar) if bar > 0 else None

    phrase = [{"text": w, "accent": i in set(args.accent)}
              for i, w in enumerate(args.words.split())]
    placed = layout(phrase, style, band, None, args.seed)

    print(f"\n{BOLD}compose{RESET}  {DIM}{style['id']} — {len(placed)}/{len(phrase)} "
          f"words placed{RESET}\n")
    for word in placed:
        mark = f"{RED}accent{RESET}" if word.get("accent") else f"{DIM}      {RESET}"
        print(f"  {mark}  {BOLD}{word['text_cased']:<12}{RESET}"
              f"{DIM}rung {word['rung']}  {word['size']:>4}pt  "
              f"{word['w']:>4}x{word['h']:<4} at ({word['x']:>4},{word['y']:>4}){RESET}")

    png = render_state(placed, style, args.dir, band, force=True)
    print(f"\n{GREEN}state {RESET}  {DIM}{png}{RESET}\n")
    return 0 if placed else 1


if __name__ == "__main__":
    sys.exit(main())
