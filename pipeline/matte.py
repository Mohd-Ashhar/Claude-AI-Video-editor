"""Segment the subject, so text can sit behind it instead of on top of it.

The single technique that separates a reference-grade lyric edit from a caption
pasted over footage: the words pass *behind* the person. Measured on the sample
reel, the letters are cut cleanly along the subject's silhouette -- the helmet
crops the "O" of ANOTHER along its own contour and the logo on it is untinted --
so it is a real matte, not a blend mode.

This runs U^2-Net (the 4.6 MB `u2netp` variant) on CPU via onnxruntime. The model
input is a fixed 320x320 regardless of source resolution, so cost is a flat
~180 ms per frame no matter what it is fed.

The honest part is the gate. u2netp finds a large subject well and a small distant
one not at all: measured across the sample reel it produced a usable matte on 12
of 18 sampled seconds, failing exactly where the snowboarder was a speck and
latching onto a snow plume instead. Rather than shipping a wrong matte, frames
that fail the gate are emitted black -- which means "nothing in front", and the
text simply draws over the top, which is what half the reference's own shots do.

    uv run python -m pipeline.matte --video out/reel_v1.mov
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from pipeline import config

BOLD, RED, YELLOW, GREEN, DIM, RESET = "\033[1m", "\033[31m", "\033[33m", "\033[32m", "\033[2m", "\033[0m"

MODEL = config.ASSETS_DIR / "models" / "u2netp.onnx"
MODEL_URL = ("https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx")
SIDE = 320
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Foreground area, as a fraction of frame, that counts as a real subject.
#
# The lower bound is the load-bearing one. A degenerate mask -- the model finding
# nothing and returning near-zero everywhere -- looks extremely "confident" by any
# bimodality measure, so crispness cannot be used to reject it. Area can: measured
# on the reference reel, every genuine subject covered 3.3-40.6% of frame while
# every failure sat at 0.5-2.6%.
MIN_FOREGROUND = 0.025
MAX_FOREGROUND = 0.55

# Softening applied to the matte edge, in output pixels. The model runs at 320px
# and is upscaled, so its silhouette is already soft; this stops the residual
# stair-stepping from reading as a hard cut-out.
FEATHER = 3


def ensure_model(path: Path = MODEL) -> Path:
    if path.exists():
        return path
    raise FileNotFoundError(
        f"no segmentation model at {path}.\n"
        f"  curl -L -o {path} {MODEL_URL}\n"
        f"  (4.6 MB — the small u2netp variant, CPU-only)")


def _session(path: Path):
    import onnxruntime as ort
    options = ort.SessionOptions()
    # One thread more than this gains nothing on an M2 and costs memory that the
    # renderer is about to want back.
    options.intra_op_num_threads = 4
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def infer(session, rgb: np.ndarray) -> np.ndarray:
    """Saliency map for one RGB frame, 0..1 at SIDExSIDE."""
    from PIL import Image

    small = np.asarray(Image.fromarray(rgb).resize((SIDE, SIDE), Image.BILINEAR),
                       dtype=np.float32) / 255.0
    x = ((small - MEAN) / STD).transpose(2, 0, 1)[None]
    out = session.run(None, {session.get_inputs()[0].name: x})[0][0, 0]
    span = float(out.max() - out.min())
    return (out - out.min()) / span if span > 1e-8 else np.zeros_like(out)


def usable(mask: np.ndarray,
           band: tuple[int, int] | None = None,
           height: int | None = None) -> tuple[bool, float]:
    """Whether this matte describes a real subject, and how much of picture it covers.

    Area, not confidence. A model that found nothing returns near-zero everywhere,
    which every sharpness or bimodality test scores as a perfect result.

    `band` is the picture strip of a letterboxed reel, and passing it is not
    optional there. The bounds were measured as a fraction of *picture*, and the
    matte runs on the rendered reel -- where a 16:9 strip is only 32% of a 9:16
    frame. Measured over the padded frame both bounds break: a mask covering the
    whole picture scores 0.32 and passes a 0.55 ceiling that is then unreachable,
    while a genuine 2.5% subject scores 0.79% and is rejected as a failure. On
    this footage that let masks covering 68-93% of the picture through, and the
    subject dodge they fed lifted the whole reel 10 points of luma.
    """
    if band and height:
        top = int(round(band[0] / height * mask.shape[0]))
        bottom = int(round(band[1] / height * mask.shape[0]))
        if bottom - top >= 1:
            mask = mask[top:bottom]
    area = float((mask > 0.5).mean())
    return MIN_FOREGROUND <= area <= MAX_FOREGROUND, area


def in_windows(when: float, windows: list[tuple[float, float]]) -> bool:
    return any(a - 1e-6 <= when <= b + 1e-6 for a, b in windows)


def build(video: Path, dest: Path, windows: list[tuple[float, float]] | None,
          width: int, height: int, fps: float, model: Path,
          band: tuple[int, int] | None = None) -> dict:
    """Write a greyscale matte video the same length as `video`.

    Streamed both ways through ffmpeg pipes: a 22s reel at 1080x1920 is 4 GB of
    raw frames, and this machine has 8 GB total.
    """
    from PIL import Image

    session = _session(model)
    reader = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", str(video),
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE, bufsize=10 ** 8)
    writer = subprocess.Popen(
        ["ffmpeg", "-y", "-v", "error",
         "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{width}x{height}",
         "-r", f"{fps}", "-i", "-",
         "-c:v", "ffv1", "-pix_fmt", "gray", str(dest)],
        stdin=subprocess.PIPE)

    frame_bytes = width * height * 3
    blank = np.zeros((height, width), dtype=np.uint8)
    index, scored, skipped, kept = 0, 0, 0, 0
    areas: list[float] = []
    # Where the subject was, per kept frame, in delivery pixels. compose.py uses
    # these to prefer anchors that land *on* the subject -- the occlusion is the
    # effect, and a word floating in empty ceiling is just a caption.
    boxes: list[dict] = []

    try:
        while True:
            raw = reader.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                break
            when = index / fps
            index += 1

            if windows is not None and not in_windows(when, windows):
                writer.stdin.write(blank.tobytes())
                skipped += 1
                continue

            frame = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)
            mask = infer(session, frame)
            scored += 1
            ok, area = usable(mask, band, height)
            areas.append(area)

            if not ok:
                # Black means "nothing in front of the text". Degrading to a plain
                # overlay is right; a wrong matte punches a hole in the words.
                writer.stdin.write(blank.tobytes())
                continue

            kept += 1
            rows = np.where(mask.max(axis=1) > 0.5)[0]
            cols = np.where(mask.max(axis=0) > 0.5)[0]
            if rows.size and cols.size:
                sy, sx = height / mask.shape[0], width / mask.shape[1]
                boxes.append({
                    "at": round(when, 3),
                    "box": [int(cols[0] * sx), int(rows[0] * sy),
                            int((cols[-1] - cols[0] + 1) * sx),
                            int((rows[-1] - rows[0] + 1) * sy)],
                })
            big = Image.fromarray((mask * 255).astype(np.uint8)).resize(
                (width, height), Image.BILINEAR)
            if FEATHER:
                from PIL import ImageFilter
                big = big.filter(ImageFilter.GaussianBlur(FEATHER))
            writer.stdin.write(np.asarray(big, dtype=np.uint8).tobytes())
    finally:
        if reader.stdout:
            reader.stdout.close()
        reader.wait()
        if writer.stdin:
            writer.stdin.close()
        writer.wait()

    return {
        "frames": index, "scored": scored, "skipped": skipped, "kept": kept,
        "median_area": round(float(np.median(areas)), 4) if areas else 0.0,
        "usable_fraction": round(kept / scored, 3) if scored else 0.0,
        "boxes": boxes,
    }


def subject_windows(brief_path: Path) -> list[tuple[float, float]]:
    """Spans of the reel where a person was actually asked for.

    u2netp is a *saliency* model, not a person detector, so on a shot with no
    subject it confidently returns whatever is brightest. Measured on a gym
    reel: on a shot of an empty doorway it produced a large soft blob over the
    glass and swallowed most of the word behind it, while on the very next shot
    -- a man walking toward camera -- it cut a clean silhouette.

    The brief already knows which shots are supposed to contain a person, so
    that is the gate. It costs nothing, and it converts a class of confident
    wrong answers into no answer at all.
    """
    doc = json.loads(brief_path.read_text())
    spans = []
    for shot in doc.get("shots", []):
        if shot.get("subject") == "none":
            continue
        spans.append((float(shot["start"]), float(shot["start"]) + float(shot["duration"])))
    return spans


def intersect(a: list[tuple[float, float]],
              b: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out = []
    for a0, a1 in a:
        for b0, b1 in b:
            lo, hi = max(a0, b0), min(a1, b1)
            if hi > lo:
                out.append((lo, hi))
    return sorted(out)


def windows_from_lyrics(path: Path, pad: float = 0.15) -> list[tuple[float, float]]:
    """Only the spans where a word is actually on screen need a matte.

    Roughly halves the cost, and the frames it skips are ones whose matte could
    not have changed a single pixel of output.
    """
    doc = json.loads(path.read_text())
    return [(max(w["at"] - pad, 0.0), w["at"] + w["duration"] + pad)
            for w in doc.get("words", [])]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Segment the subject into a matte video.")
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--lyrics", type=Path, default=config.WORK_DIR / "lyrics.json",
                    help="restrict work to the spans where words are on screen")
    ap.add_argument("--all-frames", action="store_true",
                    help="matte the whole reel, not just the lyric spans")
    ap.add_argument("--brief", type=Path, default=config.BRIEF_JSON,
                    help="skip shots the brief says have no subject")
    ap.add_argument("--model", type=Path, default=MODEL)
    args = ap.parse_args(argv)

    if not args.video.exists():
        print(f"{RED}no video at {args.video}{RESET}", file=sys.stderr)
        return 1
    try:
        model = ensure_model(args.model)
    except FileNotFoundError as exc:
        print(f"{RED}{exc}{RESET}", file=sys.stderr)
        return 1
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        print(f"{RED}uv pip install onnxruntime{RESET}", file=sys.stderr)
        return 1

    from pipeline import media
    info = media.probe(args.video)
    stream = media.video_stream(info)
    width, height = media.display_dimensions(stream)
    fps = media.parse_fps(stream) or config.OUT_FPS

    # The gate's bounds are fractions of picture, so on a letterboxed reel the
    # strip has to be found before any frame is scored. Same geometry the burn
    # confines type to, read from the same place.
    band = None
    if args.brief.exists():
        try:
            from pipeline.lyrics import letterbox_band
            band = letterbox_band(json.loads(args.brief.read_text()).get("letterbox"))
        except (json.JSONDecodeError, OSError):
            band = None

    windows = None
    if not args.all_frames and args.lyrics.exists():
        windows = windows_from_lyrics(args.lyrics)
        if args.brief.exists():
            subjects = subject_windows(args.brief)
            if subjects:
                windows = intersect(windows, subjects)

    key = hashlib.sha1(
        f"{args.video}|{args.video.stat().st_mtime_ns}|{windows}|{band}".encode()).hexdigest()[:12]
    dest = args.out or (config.WORK_DIR / f"matte_{key}.mkv")
    dest.parent.mkdir(parents=True, exist_ok=True)

    span = "lyric spans only" if windows else "every frame"
    print(f"\n{BOLD}matte{RESET}  {DIM}{args.video.name} · {width}x{height} @ {fps:.0f}fps "
          f"· {span}{RESET}\n")
    if windows:
        covered = sum(b - a for a, b in windows)
        print(f"  {DIM}{len(windows)} spans, {covered:.1f}s to segment{RESET}")

    started = time.perf_counter()
    stats = build(args.video, dest, windows, width, height, fps, model, band)
    elapsed = time.perf_counter() - started

    rate = stats["usable_fraction"]
    mark = GREEN if rate >= 0.6 else (YELLOW if rate >= 0.3 else RED)
    print(f"  {GREEN}frames{RESET}  {stats['frames']} total · {stats['scored']} segmented "
          f"· {stats['skipped']} skipped")
    print(f"  {mark}usable{RESET}  {stats['kept']}/{stats['scored']} "
          f"({rate * 100:.0f}%) · median subject area {stats['median_area'] * 100:.1f}% of "
          f"{'picture' if band else 'frame'}")

    if rate < 0.3 and stats["scored"]:
        print(f"\n  {YELLOW}most frames produced no usable matte{RESET}")
        print(f"  {DIM}u2netp needs the subject to occupy at least "
              f"{MIN_FOREGROUND * 100:.0f}% of frame. Shoot closer, or accept text "
              f"drawn over the top — which is what those frames will do.{RESET}")

    # The boxes go beside the matte, not inside it. compose.py runs long before
    # the filtergraph does and only needs "roughly where the subject was"; making
    # it decode a 22s greyscale video to find that out would be absurd.
    side = dest.with_suffix(".json")
    side.write_text(json.dumps(
        {k: v for k, v in stats.items() if k != "boxes"}
        | {"video": str(args.video), "fps": fps, "boxes": stats["boxes"]}, indent=2))
    print(f"  {GREEN}boxes {RESET}  {len(stats['boxes'])} subject boxes -> {side.name}")

    print(f"\n{GREEN}matte{RESET}  {DIM}{elapsed:.1f}s "
          f"({elapsed / max(stats['scored'], 1) * 1000:.0f}ms/frame) -> {dest}{RESET}\n")
    print(str(dest))
    return 0


if __name__ == "__main__":
    sys.exit(main())
