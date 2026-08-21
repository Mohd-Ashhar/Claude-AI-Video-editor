"""Render text cues to transparent PNGs for the filtergraph to composite.

Pillow rather than drawtext, because this ffmpeg build has neither `drawtext`
nor `subtitles` nor `ass` (see config.OPTIONAL_FILTERS). That turned out to be a
better route anyway: laying out type in Python means the wrap, the fit and the
scrim can all be measured against the actual glyphs, which no drawtext
expression can do.

Each cue renders full-frame, with the text already positioned. The filtergraph
then only has to composite at 0,0 and animate an offset -- so nothing ever
scales a text layer, which is the difference between crisp type and mush.

    uv run python -m pipeline.overlay --demo
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

from pipeline import config

BOLD, RED, YELLOW, GREEN, DIM, RESET = "\033[1m", "\033[31m", "\033[33m", "\033[32m", "\033[2m", "\033[0m"

# Faces, in preference order per role. All present on macOS; the loader falls
# through to Pillow's default rather than failing, because a reel with ugly text
# is recoverable and a reel that would not render is not.
FACES = {
    "heavy": ["/System/Library/Fonts/Supplemental/Arial Black.ttf",
              "/System/Library/Fonts/Supplemental/Impact.ttf",
              "/System/Library/Fonts/Supplemental/Arial Bold.ttf"],
    "bold":  ["/System/Library/Fonts/Supplemental/Arial Bold.ttf",
              "/System/Library/Fonts/Supplemental/Arial.ttf"],
    # A serif against the grotesque, for the second layer of a stacked lyric.
    # The contrast between the two faces is the whole point of that treatment.
    "serif": ["/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf",
              "/System/Library/Fonts/Supplemental/Georgia Bold.ttf",
              "/System/Library/Fonts/NewYork.ttf"],
}

# Measured off the reference reel: the accent word sits around #AB2517-#B23226
# on screen. Kept slightly brighter here because ours is composited over graded
# footage rather than sampled from an already-delivered frame.
ACCENT_RGB = (204, 58, 42)

# Per-treatment layout. `centre` is the vertical midpoint as a fraction of the
# frame; all of them sit inside safe_box(), which is where the Instagram UI is
# not. `size` is a starting point -- the fitter shrinks from there until the
# wrapped block fits the safe width.
TREATMENTS = {
    "hook_title": {"face": "heavy", "size": 96, "centre": 0.38, "max_lines": 3,
                   "scrim": 0.55, "tracking": -1.0, "line_spacing": 1.06},
    "label":      {"face": "bold", "size": 58, "centre": 0.74, "max_lines": 2,
                   "scrim": 0.60, "tracking": 0.0, "line_spacing": 1.10},
    "cta":        {"face": "heavy", "size": 78, "centre": 0.46, "max_lines": 3,
                   "scrim": 0.60, "tracking": -0.5, "line_spacing": 1.08},

    # Lyric treatments: one word, grown to fill the frame rather than shrunk to
    # fit it, and deliberately faint. Measured off the reference reel, its type
    # sits at only 0.13-0.29 alpha -- it reads because it is enormous, not
    # because it is opaque, and a scrim would destroy the effect entirely.
    # Ours defaults higher because that reel is flat blue snow and a gym is not.
    "lyric":       {"face": "heavy", "size": 200, "centre": 0.44, "max_lines": 1,
                    "scrim": 0.0, "tracking": -2.0, "line_spacing": 1.0,
                    "fill": 0.92, "fill_height": 0.34,
                    "alpha": 0.62, "upper": True, "shadow": False},
    "lyric_accent": {"face": "heavy", "size": 200, "centre": 0.44, "max_lines": 1,
                     "scrim": 0.0, "tracking": -2.0, "line_spacing": 1.0,
                     "fill": 0.92, "fill_height": 0.34,
                     "alpha": 0.92, "upper": True, "shadow": False,
                     "colour": ACCENT_RGB},
    # The small serif word that sits on top of a big grotesque one.
    "lyric_inlay": {"face": "serif", "size": 110, "centre": 0.44, "max_lines": 1,
                    "scrim": 0.0, "tracking": 0.0, "line_spacing": 1.0,
                    "fill": 0.30, "fill_height": 0.14,
                    "alpha": 0.95, "upper": True, "shadow": False,
                    "colour": ACCENT_RGB},
}

TEXT_COLOUR = (255, 255, 255, 255)
SHADOW_COLOUR = (0, 0, 0, 190)
SHADOW_OFFSET = 4
SCRIM_PAD_X = 28
SCRIM_PAD_Y = 18
SCRIM_RADIUS = 18
MIN_SIZE = 30


def _load_font(paths: list[str], size: int):
    from PIL import ImageFont
    for candidate in paths:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _wrap(draw, text: str, font, max_width: int) -> list[str]:
    """Greedy word wrap measured on the real glyphs, not on a character count."""
    words, lines, current = text.split(), [], ""
    for word in words:
        trial = f"{current} {word}".strip()
        if draw.textlength(trial, font=font) <= max_width or not current:
            current = trial
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


MAX_SIZE = 460


def fit(draw, text: str, treatment: dict, max_width: int,
        max_height: int = config.OUT_H) -> tuple[list[str], object]:
    """Size the type to its treatment: grown to fill, or shrunk to fit.

    Two opposite behaviours, because two opposite jobs. A hook is a sentence that
    must fit inside a budget of lines, so it shrinks until it does. A lyric word
    is one word that should be as large as the frame allows -- the reference reel
    it is modelled on carries its type at 0.13-0.29 alpha and it still dominates
    the frame, which only works at that scale.
    """
    if treatment.get("fill"):
        return _fill(draw, text, treatment, max_width, max_height)

    size = treatment["size"]
    while size > MIN_SIZE:
        font = _load_font(FACES[treatment["face"]], size)
        lines = _wrap(draw, text, font, max_width)
        if len(lines) <= treatment["max_lines"]:
            return lines, font
        size -= 4

    font = _load_font(FACES[treatment["face"]], MIN_SIZE)
    return _wrap(draw, text, font, max_width), font


def _fill(draw, text: str, treatment: dict, max_width: int,
          max_height: int) -> tuple[list[str], object]:
    """Grow a single line until it fills the frame on whichever axis binds first.

    Both axes, not just width. Capping on width alone is right for a long word
    and badly wrong for a short one: measured in a 16:9 letterbox, filling 92% of
    the width with "LIT" produced type 524px tall in a 608px strip -- 86% of the
    picture, against roughly 14% in the reference this is modelled on. Three
    letters are as wide as seven only if they are two and a half times taller.

    Binary search rather than a linear walk: the range runs to 460px and a
    four-pixel step would cost a hundred font loads per cue, on every cue of a
    forty-word lyric track.
    """
    width_target = max_width * treatment["fill"]
    height_target = max_height * treatment.get("fill_height", 1.0)
    faces = FACES[treatment["face"]]
    lo, hi, best = MIN_SIZE, MAX_SIZE, MIN_SIZE

    while lo <= hi:
        mid = (lo + hi) // 2
        font = _load_font(faces, mid)
        ascent, descent = font.getmetrics() if hasattr(font, "getmetrics") else (mid, 0)
        if draw.textlength(text, font=font) <= width_target and \
                (ascent + descent) <= height_target:
            best, lo = mid, mid + 1
        else:
            hi = mid - 1

    font = _load_font(faces, best)
    # A single line, always. A lyric cue that wrapped would be two words, and two
    # words is a different cue -- the caller splits them, not the renderer.
    return [text], font


def cue_path(text: str, kind: str, directory: Path,
             band: tuple[int, int] | None = None) -> Path:
    """Content-addressed, so a re-render costs nothing and the name is shell-safe.

    The filename lands verbatim inside a filter_complex string, where a colon
    separates arguments and a comma separates filters. A hash cannot contain
    either; the text it came from very well might.
    """
    key = hashlib.sha1(
        f"{kind}|{text}|{config.OUT_W}x{config.OUT_H}|{band}".encode()).hexdigest()[:16]
    return directory / f"{kind}_{key}.png"


def render_cue(text: str, kind: str, directory: Path, force: bool = False,
               band: tuple[int, int] | None = None) -> Path:
    """Render one cue full-frame on transparency. Returns the PNG path.

    `band` narrows the vertical space the type may use, for a letterboxed reel
    where the picture is a strip and the rest of the frame is black. Without it
    the type is centred on the frame and sized against the safe box, which puts
    it partly in the bars.
    """
    from PIL import Image, ImageDraw

    if kind not in TREATMENTS:
        raise ValueError(f"unknown text treatment '{kind}'")
    text = " ".join(text.split())
    if not text:
        raise ValueError("empty text cue")
    if TREATMENTS[kind].get("upper"):
        text = text.upper()

    dest = cue_path(text, kind, directory, band)
    if dest.exists() and not force:
        return dest

    treatment = TREATMENTS[kind]
    box_x, box_y, box_w, box_h = config.safe_box()
    if band:
        top, bottom = band
        low, high = max(box_y, top), min(box_y + box_h, bottom)
        if high - low > 80:
            box_y, box_h = low, high - low

    canvas = Image.new("RGBA", (config.OUT_W, config.OUT_H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    # The scrim's padding has to come out of the width the text may occupy, or a
    # line that exactly fits the safe box pushes its own background outside it.
    inset = 0 if treatment.get("fill") else (SCRIM_PAD_X + SHADOW_OFFSET) * 2
    lines, font = fit(draw, text, treatment, box_w - inset, box_h)
    ascent, descent = font.getmetrics() if hasattr(font, "getmetrics") else (treatment["size"], 0)
    line_h = int((ascent + descent) * treatment["line_spacing"])
    block_h = line_h * len(lines)

    # Keep the block inside the safe interior even when it wrapped taller than
    # the nominal centre allows.
    centre_y = (int(box_y + box_h * treatment["centre"]) if band
                else int(config.OUT_H * treatment["centre"]))
    top = min(max(centre_y - block_h // 2, box_y), max(box_y + box_h - block_h, box_y))

    widths = [draw.textlength(line, font=font) for line in lines]
    widest = max(widths) if widths else 0

    # Centre on the safe interior, not on the frame. The Instagram UI is not
    # symmetrical -- 60px of margin on the left against 200px on the right for
    # the action rail -- so the interior's centre is x=470 while the frame's is
    # x=540. Centring on the frame put every hook and CTA scrim past the right
    # edge of the safe box; the label only passed because it was short enough to
    # get away with it.
    centre_x = box_x + box_w / 2

    if treatment["scrim"] > 0:
        left = int(centre_x - widest / 2) - SCRIM_PAD_X
        right = int(centre_x + widest / 2) + SCRIM_PAD_X
        draw.rounded_rectangle(
            [left, top - SCRIM_PAD_Y, right, top + block_h + SCRIM_PAD_Y],
            radius=SCRIM_RADIUS,
            fill=(0, 0, 0, int(255 * treatment["scrim"])),
        )

    # Colour and opacity are baked into the PNG rather than applied in the
    # filtergraph. A lyric layer is drawn at 0.62 alpha and then composited
    # *behind* a subject matte, and doing the two in different places would mean
    # two different alpha channels fighting over the same pixels.
    rgb = tuple(treatment.get("colour", TEXT_COLOUR[:3]))
    fill = (*rgb, int(255 * treatment.get("alpha", 1.0)))
    shadow = treatment.get("shadow", True)

    for index, line in enumerate(lines):
        x = centre_x - widths[index] / 2
        y = top + index * line_h
        if shadow:
            draw.text((x + SHADOW_OFFSET, y + SHADOW_OFFSET), line, font=font,
                      fill=SHADOW_COLOUR)
        draw.text((x, y), line, font=font, fill=fill)

    directory.mkdir(parents=True, exist_ok=True)
    canvas.save(dest)
    return dest


def within_safe_box(path: Path) -> bool:
    """Every non-transparent pixel sits inside the UI-clear interior.

    Checked on the rendered image rather than trusted from the layout maths,
    because the layout maths is exactly what would be wrong.
    """
    from PIL import Image

    box_x, box_y, box_w, box_h = config.safe_box()
    alpha = Image.open(path).getchannel("A")
    bounds = alpha.getbbox()
    if bounds is None:
        return True
    left, top, right, bottom = bounds
    return (left >= box_x and top >= box_y
            and right <= box_x + box_w and bottom <= box_y + box_h)


def cues_for_brief(brief: dict, directory: Path) -> dict[int, dict]:
    """Render every cue a brief calls for, keyed by shot index."""
    rendered: dict[int, dict] = {}
    for shot in brief.get("shots", []):
        cue = shot.get("text")
        if not cue:
            continue
        png = render_cue(cue["text"], cue["kind"], directory)
        rendered[shot["index"]] = {
            "type": "text",
            "png": str(png),
            "window": [cue["at"], round(cue["at"] + cue["duration"], 3)],
        }
    return rendered


# ---------------------------------------------------------------- entry


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Render text cues to PNGs.")
    ap.add_argument("--text", default=None)
    ap.add_argument("--kind", default="hook_title", choices=sorted(TREATMENTS))
    ap.add_argument("--dir", type=Path, default=config.WORK_DIR / "cues")
    ap.add_argument("--demo", action="store_true",
                    help="render one of each treatment, at a realistic length")
    args = ap.parse_args(argv)

    jobs = [(args.text, args.kind)] if args.text else []
    if args.demo or not jobs:
        jobs = [
            ("I have never lifted this before", "hook_title"),
            ("Nobody told me this place existed", "hook_title"),
            ("watch the knee", "label"),
            ("Save this before your next session", "cta"),
        ]

    print(f"\n{BOLD}overlay{RESET}  {DIM}{len(jobs)} cue(s) -> {args.dir}{RESET}\n")
    failures = 0
    for text, kind in jobs:
        try:
            path = render_cue(text, kind, args.dir, force=True)
        except (ValueError, ImportError) as exc:
            failures += 1
            print(f"  {RED}fail  {RESET}  {text[:40]}  {DIM}{exc}{RESET}")
            continue
        safe = within_safe_box(path)
        mark = GREEN if safe else RED
        print(f"  {mark}{'ok' if safe else 'OUTSIDE SAFE BOX'}{RESET}  "
              f"{kind:<11}{DIM}{path.name}  “{text[:38]}”{RESET}")
        failures += 0 if safe else 1

    print()
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
