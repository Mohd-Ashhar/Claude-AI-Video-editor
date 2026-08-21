"""Find the 2-8 second windows worth cutting with, inside a continuous take.

This is the stage the whole v2 update exists for. Cut detection cannot help
here: a 5-minute Pocket 3 clip is one continuous take with no cuts in it, so
scenes.py correctly returns a single 5-minute shot, and averaging any metric
across those five minutes describes nothing.

So instead of detecting boundaries, this searches for them. Every window of
every plausible length is scored against a dense per-frame measurement, the
best non-overlapping ones survive, and each survivor comes out shaped exactly
like a shot -- which is why vlm_tag, merge and the render spine needed no
structural change to work on moments instead of whole files.

    uv run python -m pipeline.moments --contact-sheet
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from pipeline import config, features, signals

BOLD, RED, YELLOW, GREEN, DIM, RESET = "\033[1m", "\033[31m", "\033[33m", "\033[32m", "\033[2m", "\033[0m"

# How much each term moves the window score. The means and the consistency
# penalty do the real work; the rest are nudges that break ties between windows
# whose quality is otherwise indistinguishable.
W_CONSISTENCY = 0.5     # subtracted std of frame score across the window
W_BOUNDARY = 0.12       # cutting where content or motion already changes
W_INTERNAL_CUT = 0.35   # a content change buried mid-window: it spans two things
W_DURATION_PRIOR = 0.05

# Above this, the movement is spread across the whole frame -- a pan, not a
# subject -- and its centroid says nothing about where to point the crop.
SUBJECT_MAX_SPREAD = 0.80


# ---------------------------------------------------------------- per-sample scoring


def _median_smooth(values: np.ndarray, kernel: int = 3) -> np.ndarray:
    """Kill single-sample outliers without moving real edges.

    One motion-blurred frame in an otherwise sharp second is a compression
    artefact or a footstep, not a reason to reject two seconds of footage.
    """
    if values.size < kernel or kernel < 3:
        return values
    pad = kernel // 2
    padded = np.pad(values, pad, mode="edge")
    stacked = np.lib.stride_tricks.sliding_window_view(padded, kernel)
    return np.median(stacked, axis=1)


def _moving_mean(values: np.ndarray, kernel: int) -> np.ndarray:
    if values.size < kernel or kernel < 2:
        return values
    pad = kernel // 2
    padded = np.pad(values, (pad, kernel - 1 - pad), mode="edge")
    return np.convolve(padded, np.ones(kernel) / kernel, mode="valid")


def smoothness_curve(dx: np.ndarray, dy: np.ndarray, magnitude: np.ndarray,
                     sample_fps: float) -> np.ndarray:
    """How steady the camera move is, per sample, in 0..1.

    Note this is deliberately *not* coherence. Coherence is spatial -- how much
    of the frame moves together -- and a rigidly shaking camera scores a perfect
    1.0 on it, because every pixel does shake in unison. Shake is temporal: the
    direction of travel keeps reversing. So this measures the angular change
    between consecutive motion vectors, the per-sample form of the jitter metric
    signals.py already scores per shot.

    Averaged over half a second, because one direction change is a deliberate
    whip and only sustained flipping is a defect.
    """
    angles = np.arctan2(dy, dx)
    delta = np.abs(np.diff(angles, prepend=angles[:1]))
    delta = np.minimum(delta, 2 * np.pi - delta)
    turning = np.where(magnitude >= config.MOTION_STILL, delta, 0.0)
    smoothed = _moving_mean(turning, max(int(round(0.5 * sample_fps)), 2))
    return np.clip(1.0 - smoothed / (np.pi / 2), 0.0, 1.0)


def interest_curve(magnitude: np.ndarray, smoothness: np.ndarray,
                   coherence: np.ndarray) -> np.ndarray:
    """How compelling the movement in this frame is.

    Saturating, not peaked: more motion is more engaging with diminishing
    returns. There is no "ideal" magnitude to miss, which matters because real
    shooting styles differ by 3-4x -- handheld gym footage and locked-off cooking
    close-ups cannot share one ideal, and picking either one wrecks the other.

    Whether the motion is *worth* watching is decided by the two factors it is
    multiplied by, not by its speed:

      smoothness -- controlled movement counts, shake does not. This is what
                    stops a violent frame scoring as the most interesting one.
      coherence  -- the frame moving as a whole (a pan, a push-in) counts for
                    more than parts of it moving independently.

    A dead-static frame still scores near zero, because it genuinely has nothing
    going on.
    """
    saturating = 1.0 - np.exp(-magnitude / config.MOTION_SATURATION)
    return saturating * smoothness * (0.5 + 0.5 * coherence)


def frame_scores(samples: list[dict], sample_fps: float) -> dict[str, np.ndarray]:
    """Score every sample in a file, 0..1, plus the components worth keeping.

    Sharpness is ranked within *this file* rather than across the batch: a
    5-minute take carries its own focus and exposure range, and what matters is
    which seconds of this clip are the sharp ones.
    """
    t = np.array([s["t"] for s in samples], dtype=float)
    sharp = _median_smooth(np.array([s["sharpness"] for s in samples], dtype=float))
    mag = _median_smooth(np.array([s["flow_mag"] for s in samples], dtype=float))
    coh = np.array([s["coherence"] for s in samples], dtype=float)
    novelty = np.array([s["novelty"] for s in samples], dtype=float)
    dx = np.array([s["flow_dx"] for s in samples], dtype=float)
    dy = np.array([s["flow_dy"] for s in samples], dtype=float)

    sharp_rank = np.asarray(signals.percentile_rank(sharp.tolist()), dtype=float)
    exposure = np.array([signals.exposure_score({
        "clip_high": s["clip_high"], "clip_low": s["clip_low"], "luma_mean": s["luma"],
    }) for s in samples], dtype=float)

    steadiness = smoothness_curve(dx, dy, mag, sample_fps)
    interest = interest_curve(mag, steadiness, coh)

    score = (0.35 * sharp_rank + 0.25 * exposure
             + 0.20 * steadiness + 0.20 * interest)

    return {
        "t": t, "score": score, "novelty": novelty,
        "magnitude": mag, "coherence": coh, "smoothness": steadiness,
        "dx": dx, "dy": dy,
        "motion_x": np.array([s.get("motion_x", 0.5) for s in samples], dtype=float),
        "motion_spread": np.array([s.get("motion_spread", 1.0) for s in samples], dtype=float),
    }


def boundary_affinity(field: dict) -> tuple[np.ndarray, np.ndarray]:
    """Per-sample 0..1 preference for starting and for ending a cut here.

    A cut that lands where something already changes -- the content shifts, or
    the camera starts or stops moving -- reads as intentional. A cut placed
    mid-glide reads like the editor ran out of footage.
    """
    change = np.clip(field["novelty"] / config.NOVELTY_CUT, 0.0, 1.0)

    moving = (field["magnitude"] >= config.MOTION_STILL).astype(float)
    onset = np.clip(np.diff(moving, prepend=moving[:1]), 0.0, 1.0)   # still -> moving
    settle = np.clip(-np.diff(moving, append=moving[-1:]), 0.0, 1.0)  # moving -> still

    start_affinity = np.maximum(change, onset)
    # Ending just before the content changes is the same instinct, one sample over.
    end_affinity = np.maximum(np.roll(change, -1), settle)
    end_affinity[-1] = settle[-1]
    return start_affinity, end_affinity


# ---------------------------------------------------------------- window search


def _windows_for_length(field: dict, length: int, hop: int,
                        start_affinity: np.ndarray, end_affinity: np.ndarray,
                        duration: float) -> list[dict]:
    """Score every window of exactly `length` samples, strided by `hop`."""
    score = field["score"]
    if score.size < length:
        return []

    view = np.lib.stride_tricks.sliding_window_view(score, length)[::hop]
    starts = np.arange(view.shape[0]) * hop

    means = view.mean(axis=1)
    stds = view.std(axis=1)
    mins = view.min(axis=1)

    # A content change buried inside the window means it spans two different
    # things; near the edges it is exactly the boundary we wanted.
    novelty_view = np.lib.stride_tricks.sliding_window_view(field["novelty"], length)[::hop]
    margin = max(length // 5, 1)
    interior = novelty_view[:, margin:length - margin] if length > 2 * margin else novelty_view[:, :0]
    internal = (interior.max(axis=1) / config.NOVELTY_CUT if interior.size
                else np.zeros(view.shape[0]))

    ends = starts + length - 1
    boundary = 0.5 * (start_affinity[starts] + end_affinity[ends])

    # Short shots hold attention; the prior is mild because the pacing arc in
    # sequence.py decides final length anyway.
    prior = 1.0 - min(abs(duration - 3.0) / 5.0, 1.0)

    total = (means
             - W_CONSISTENCY * stds
             + W_BOUNDARY * boundary
             - W_INTERNAL_CUT * np.clip(internal, 0.0, 1.0)
             + W_DURATION_PRIOR * prior)

    usable = mins >= config.FRAME_SCORE_FLOOR
    return [
        {"i0": int(s), "i1": int(e), "duration": duration,
         "score": float(v), "quality": float(m)}
        for s, e, v, m, ok in zip(starts, ends, total, means, usable) if ok
    ]


def search(field: dict, sample_fps: float, t0: float, t1: float,
           min_seconds: float | None = None,
           max_seconds: float | None = None) -> list[dict]:
    """Every plausible window inside [t0, t1), scored, best first.

    `min_seconds` defaults to config.MOMENT_MIN_SECONDS. Raising it is not a
    relaxed quality bar -- every window still has to clear FRAME_SCORE_FLOOR on
    every sample. It asks for the longest *good* window rather than the best
    short one, which is what a brief with long holds needs: the score rewards a
    high mean, and a short window almost always has a higher mean than a long
    one that is just as clean.
    """
    mask = (field["t"] >= t0) & (field["t"] < t1)
    if mask.sum() < 2:
        return []

    local = {k: v[mask] for k, v in field.items()}
    start_affinity, end_affinity = boundary_affinity(local)
    hop = max(int(round(config.MOMENT_HOP * sample_fps)), 1)

    found: list[dict] = []
    duration = config.MOMENT_MIN_SECONDS if min_seconds is None else min_seconds
    ceiling = config.MOMENT_MAX_SECONDS if max_seconds is None else max_seconds
    while duration <= ceiling + 1e-9:
        length = int(round(duration * sample_fps))
        if length >= 2:
            found += _windows_for_length(local, length, hop,
                                         start_affinity, end_affinity, duration)
        duration += config.MOMENT_DURATION_STEP

    for window in found:
        window["start"] = float(local["t"][window["i0"]])
        window["end"] = float(local["t"][window["i1"]])
        window["_field"] = local

    found.sort(key=lambda w: -w["score"])
    return found


def suppress(windows: list[dict], limit: int) -> list[dict]:
    """Greedy non-maximum suppression: best first, no overlaps, enforced gaps.

    Without the gap rule the top windows are all near-identical neighbours of
    the single best second, and the reel ends up cutting between six views of
    the same moment.
    """
    kept: list[dict] = []
    for window in windows:
        if len(kept) >= limit:
            break
        clash = any(
            window["start"] < k["end"] + config.MOMENT_MIN_GAP
            and k["start"] < window["end"] + config.MOMENT_MIN_GAP
            for k in kept
        )
        if not clash:
            kept.append(window)
    kept.sort(key=lambda w: w["start"])
    return kept


# ---------------------------------------------------------------- edge motion


def subject_position(field: dict, t0: float, t1: float) -> float | None:
    """Where the subject sits across the frame during this window, 0..1.

    None when there is nothing to go on: too little movement in the shot to
    locate anything, or movement so spread out that it is the camera panning
    rather than a subject moving. Both mean the crop should stay centred, and
    saying so explicitly beats returning a confident 0.5.

    The median across the window, not the mean: a moment where someone walks
    through the background for half a second should not drag the framing.
    """
    mask = (field["t"] >= t0) & (field["t"] <= t1)
    if not mask.any():
        return None

    magnitude = field["magnitude"][mask]
    # A quiet scene still contains movement, and following it frames the wrong
    # thing -- the hand rather than the pan. See config.SUBJECT_MIN_MOTION.
    if float(np.median(magnitude)) < config.SUBJECT_MIN_MOTION:
        return None

    usable = magnitude > max(config.MOTION_STILL, float(np.median(magnitude)) * 0.5)
    if usable.sum() < 3:
        return None

    localised = field["motion_spread"][mask][usable] < SUBJECT_MAX_SPREAD
    if localised.sum() < max(int(usable.sum() * 0.4), 2):
        return None

    return float(np.median(field["motion_x"][mask][usable][localised]))


def edge_motion(field: dict, t0: float, t1: float) -> dict:
    """Mean motion over a short span, as sequence.py's transition evidence.

    Recorded per moment rather than recomputed later so that choosing
    transitions never has to reopen the feature series -- and so a whip can only
    ever fire in the direction the camera actually moved.
    """
    mask = (field["t"] >= t0) & (field["t"] <= t1)
    if not mask.any():
        return {"dx": 0.0, "dy": 0.0, "magnitude": 0.0, "coherence": 1.0}
    return {
        "dx": round(float(field["dx"][mask].mean()), 4),
        "dy": round(float(field["dy"][mask].mean()), 4),
        "magnitude": round(float(field["magnitude"][mask].mean()), 4),
        "coherence": round(float(field["coherence"][mask].mean()), 4),
    }


# ---------------------------------------------------------------- build


def moments_for_take(take: dict, field: dict, sample_fps: float, index_base: int,
                     min_seconds: float | None = None,
                     max_seconds: float | None = None) -> list[dict]:
    span = float(take["end"]) - float(take["start"])
    density = max(int(span / config.MOMENT_SECONDS_PER_CANDIDATE), 1)
    limit = min(density, config.MAX_MOMENTS_PER_FILE)

    windows = search(field, sample_fps, float(take["start"]), float(take["end"]),
                     min_seconds, max_seconds)
    kept = suppress(windows, limit)

    out: list[dict] = []
    for offset, window in enumerate(kept):
        local = window.pop("_field")
        i0, i1 = window["i0"], window["i1"]
        start, end = window["start"], window["end"]

        peak_index = i0 + int(np.argmax(local["score"][i0:i1 + 1]))
        edge = config.MOMENT_EDGE_SECONDS

        out.append({
            "shot_id": f"{take['name']}__m{index_base + offset:02d}",
            "take_id": take["shot_id"],
            "source": take["source"],
            "proxy": take["proxy"],
            "name": take["name"],
            "start": round(start, 3),
            "end": round(end, 3),
            "duration": round(end - start, 3),
            "peak": round(float(local["t"][peak_index]), 3),
            "window_score": round(window["score"], 4),
            "subject_x": subject_position(local, start, end),
            "entry_flow": edge_motion(local, start, min(start + edge, end)),
            "exit_flow": edge_motion(local, max(end - edge, start), end),
            "fps": take["fps"],
            "width": take["width"],
            "height": take["height"],
            "max_crop_width": take["max_crop_width"],
        })
    return out


def build(takes: list[dict], features_dir: Path,
          min_seconds: float | None = None,
          max_seconds: float | None = None) -> list[dict]:
    by_name: dict[str, list[dict]] = {}
    for take in takes:
        by_name.setdefault(take["name"], []).append(take)

    moments: list[dict] = []
    for name, group in by_name.items():
        doc = features.load(name, features_dir)
        samples = doc["samples"]
        if len(samples) < 2:
            print(f"  {YELLOW}skip  {RESET}  {name:<28} too few samples to search")
            continue

        field = frame_scores(samples, doc["sample_fps"])
        found: list[dict] = []
        for take in sorted(group, key=lambda s: s["start"]):
            found += moments_for_take(take, field, doc["sample_fps"], len(found),
                                      min_seconds, max_seconds)

        found.sort(key=lambda m: -m["window_score"])
        found = found[:config.MAX_MOMENTS_PER_FILE]

        if not found:
            print(f"  {YELLOW}none  {RESET}  {name:<28} no window cleared the quality floor")
            continue

        # The density cap guarantees each take at least one candidate, which is
        # right for a continuous take and wrong for a stretch that is simply bad.
        best = found[0]["window_score"]
        kept = [m for m in found if m["window_score"] >= best - config.MOMENT_SCORE_DROP]
        if len(kept) < min(config.MOMENT_KEEP_FLOOR, len(found)):
            kept = found[:config.MOMENT_KEEP_FLOOR]
        dropped = len(found) - len(kept)

        kept.sort(key=lambda m: m["start"])
        note = f"  {DIM}best {best:.2f}" + (f", {dropped} weak dropped" if dropped else "") + RESET
        print(f"  {GREEN}search{RESET}  {name:<28} {len(kept)} moment(s) from "
              f"{len(group)} take(s){note}")
        moments += kept

    return moments


# ---------------------------------------------------------------- contact sheet


def contact_sheet(moments: list[dict], dest: Path, columns: int = 4, tile: int = 300) -> Path | None:
    """A labelled strip of every candidate's peak frame.

    The fastest way to judge whether the scorer agrees with you: one image
    instead of scrubbing forty windows. Tuning the weights against real footage
    is a two-minute loop with this and a guessing game without it.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print(f"  {YELLOW}Pillow not installed — skipping contact sheet{RESET}")
        return None

    if not moments:
        return None

    font = _load_font(15)
    rows = (len(moments) + columns - 1) // columns
    label_h = 34
    # Cells match the PROXY's shape (16:9), not the delivery shape. The frames
    # pasted in are whole source frames; sizing the cell 9:16 left each one
    # marooned in black with the caption sitting on top of it.
    cell_w, cell_h = tile, int(round(tile * 9 / 16))
    sheet = Image.new("RGB", (columns * cell_w, rows * (cell_h + label_h)), (18, 18, 20))
    draw = ImageDraw.Draw(sheet)

    # The 9:16 window the render will actually take, drawn on each frame so the
    # sheet shows what ends up on screen rather than what the camera saw.
    window_w = int(round(cell_h * config.OUT_W / config.OUT_H))

    with tempfile.TemporaryDirectory() as tmp:
        for index, moment in enumerate(moments):
            frame = Path(tmp) / f"{index:03d}.jpg"
            proc = subprocess.run(
                ["ffmpeg", "-y", "-v", "error", "-ss", f"{moment['peak']:.3f}",
                 "-i", moment["proxy"], "-frames:v", "1",
                 "-vf", f"scale={cell_w}:-2", str(frame)],
                capture_output=True, text=True,
            )
            col, row = index % columns, index // columns
            x, y = col * cell_w, row * (cell_h + label_h)

            if proc.returncode == 0 and frame.exists():
                thumb = Image.open(frame).convert("RGB")
                thumb.thumbnail((cell_w, cell_h))
                ox = x + (cell_w - thumb.width) // 2
                oy = y + (cell_h - thumb.height) // 2
                sheet.paste(thumb, (ox, oy))
                # Where the crop will actually land, subject-tracked or centred.
                position = moment.get("subject_x")
                if position is None:
                    offset = (thumb.width - window_w) / 2
                    colour = (150, 150, 158)
                else:
                    offset = position * thumb.width - window_w * 0.46
                    colour = (255, 210, 60)
                left = ox + int(round(min(max(offset, 0), thumb.width - window_w)))
                draw.rectangle([left, oy, left + window_w, oy + thumb.height - 1],
                               outline=colour, width=2)

            label = f"{index + 1}. {moment['shot_id']}"
            detail = (f"{moment['window_score']:.2f}   "
                      f"{moment['start']:.1f}-{moment['end']:.1f}s "
                      f"({moment['duration']:.1f}s)")
            draw.text((x + 6, y + cell_h + 4), label, fill=(235, 235, 240), font=font)
            draw.text((x + 6, y + cell_h + 19), detail, fill=(150, 150, 158), font=font)

    dest.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(dest, quality=88)
    return dest


def _load_font(size: int):
    from PIL import ImageFont
    for candidate in ("/System/Library/Fonts/Supplemental/Arial.ttf",
                      "/System/Library/Fonts/Helvetica.ttc",
                      "/System/Library/Fonts/Geneva.ttf"):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


# ---------------------------------------------------------------- entry


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Find cuttable moments inside each take.")
    ap.add_argument("--takes", type=Path, default=config.SHOTS_JSON)
    ap.add_argument("--features", type=Path, default=config.FEATURES_DIR)
    ap.add_argument("--out", type=Path, default=config.MOMENTS_JSON)
    ap.add_argument("--contact-sheet", action="store_true",
                    help="write a labelled strip of every candidate's peak frame")
    ap.add_argument("--sheet-path", type=Path, default=config.WORK_DIR / "contact_sheet.jpg")
    ap.add_argument("--min-seconds", type=float, default=config.MOMENT_MIN_SECONDS,
                    help="shortest window to search for. Raise it when the brief "
                         "asks for holds longer than the best short window "
                         "(cast.py reports this as 'nothing long enough'); the "
                         "quality floor still applies to every sample")
    ap.add_argument("--max-seconds", type=float, default=config.MOMENT_MAX_SECONDS,
                    help="longest window to search for")
    args = ap.parse_args(argv)

    if not args.takes.exists():
        print(f"{RED}no shots.json — run `python -m pipeline.scenes` first{RESET}",
              file=sys.stderr)
        return 1

    takes = json.loads(args.takes.read_text())
    footage = sum(float(t["duration"]) for t in takes)
    print(f"\n{BOLD}moments{RESET}  {DIM}{len(takes)} take(s) · {footage:.0f}s · "
          f"windows {args.min_seconds:g}-{args.max_seconds:g}s{RESET}\n")

    started = time.perf_counter()
    try:
        moments = build(takes, args.features, args.min_seconds, args.max_seconds)
    except FileNotFoundError as exc:
        print(f"{RED}{exc}{RESET}", file=sys.stderr)
        return 1

    if not moments:
        print(f"\n{RED}no usable moments found in {footage:.0f}s of footage{RESET}")
        print(f"{DIM}every window fell below the quality floor "
              f"({config.FRAME_SCORE_FLOOR}). Check focus, exposure and stability, "
              f"or lower config.FRAME_SCORE_FLOOR.{RESET}\n")
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(moments, indent=2))

    kept = sum(m["duration"] for m in moments)
    print(f"\n  {BOLD}{'moment':<26}{'window':>16}{'peak':>8}{'score':>8}{RESET}")
    for moment in sorted(moments, key=lambda m: -m["window_score"])[:12]:
        window = f"{moment['start']:.1f}-{moment['end']:.1f}s"
        print(f"  {moment['shot_id']:<26}{window:>16}{moment['peak']:>8.1f}"
              f"{moment['window_score']:>8.2f}")

    if args.contact_sheet:
        path = contact_sheet(moments, args.sheet_path)
        if path:
            print(f"\n  {GREEN}contact sheet{RESET}  {path}  {DIM}open it and check the picks{RESET}")

    print(f"\n{GREEN}{len(moments)} moments{RESET}  {DIM}{kept:.0f}s kept of {footage:.0f}s "
          f"({kept / footage * 100:.0f}%) · {time.perf_counter() - started:.1f}s "
          f"-> {args.out}{RESET}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
