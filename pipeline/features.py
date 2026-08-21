"""One dense measurement pass per proxy, at config.SAMPLE_FPS.

Split out of signals.py because the input changed shape. When footage arrived as
many short clips, measuring once per shot was enough. Now a single file is a
continuous 30s-5min take, and one average across five minutes describes nothing:
a clip can be brilliant for four seconds and unusable for the other 296.

So this stage measures the whole file on a fixed grid and writes the raw series.
moments.py searches it for windows worth cutting with; signals.py aggregates it
per moment. Neither of them re-opens the video.

Cost, measured on this machine: 4.1 ms of optical flow plus 1.1 ms of Laplacian
per sample, so a 5-minute file lands around 15s at 8 Hz.

    uv run python -m pipeline.features
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

from pipeline import config

BOLD, RED, YELLOW, GREEN, DIM, RESET = "\033[1m", "\033[31m", "\033[33m", "\033[32m", "\033[2m", "\033[0m"

# Bins per channel for the novelty histogram. Coarse on purpose: this must fire
# on "the camera is now looking at something else", not on grain or a passing cloud.
HIST_BINS = 12


# ---------------------------------------------------------------- frame sampling


def sample_proxy(proxy: Path, sample_fps: float = config.SAMPLE_FPS) -> list[dict]:
    """Walk a proxy once, measuring every Nth frame.

    Sequential reading rather than seek-per-window: seeking a long proxy
    repeatedly is both slow and frame-inaccurate, and one pass serves every
    window the searcher will ever consider.
    """
    cap = cv2.VideoCapture(str(proxy))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open proxy: {proxy}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(int(round(fps / sample_fps)), 1)

    samples: list[dict] = []
    prev_small = None
    prev_hist = None
    index = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if index % stride:
            index += 1
            continue

        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        scale = config.FLOW_WIDTH / grey.shape[1]
        small = cv2.resize(grey, (config.FLOW_WIDTH, max(int(grey.shape[0] * scale), 1)))

        flow_mag = flow_dx = flow_dy = 0.0
        coherence = 1.0
        motion_x, motion_spread = 0.5, 1.0
        if prev_small is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev_small, small, None,
                pyr_scale=0.5, levels=2, winsize=15,
                iterations=2, poly_n=5, poly_sigma=1.1, flags=0,
            )
            flow_dx, flow_dy = float(flow[..., 0].mean()), float(flow[..., 1].mean())
            flow_mag = float(np.linalg.norm(flow, axis=2).mean())
            coherence = _coherence(flow_dx, flow_dy, flow_mag)
            motion_x, motion_spread = _motion_centre(flow)
        prev_small = small

        hist = _histogram(frame)
        novelty = float(np.abs(hist - prev_hist).sum() / 2.0) if prev_hist is not None else 0.0
        prev_hist = hist

        # One Laplacian serves two purposes: its variance is the standard cheap
        # focus measure, and its column profile says where the detail sits.
        laplacian = cv2.Laplacian(grey, cv2.CV_64F)
        detail_x, detail_spread = _profile_centre(np.abs(laplacian).sum(axis=0))

        total = grey.size
        samples.append({
            "t": round(index / fps, 3),
            "sharpness": round(float(laplacian.var()), 3),
            "detail_x": round(detail_x, 4),
            "detail_spread": round(detail_spread, 4),
            "luma": round(float(grey.mean()), 2),
            "clip_high": round(float((grey > config.CLIP_HIGH).sum() / total), 5),
            "clip_low": round(float((grey < config.CLIP_LOW).sum() / total), 5),
            "flow_mag": round(flow_mag, 4),
            "flow_dx": round(flow_dx, 4),
            "flow_dy": round(flow_dy, 4),
            "coherence": round(coherence, 4),
            "novelty": round(novelty, 4),
            "motion_x": round(motion_x, 4),
            "motion_spread": round(motion_spread, 4),
        })
        index += 1

    cap.release()
    return samples


def _coherence(dx: float, dy: float, magnitude: float) -> float:
    """How much of the frame's motion points the same way, in 0..1.

    The single most useful new signal in the moment finder, because raw
    magnitude conflates three completely different situations:

      high magnitude, high coherence -> a deliberate gimbal pan or push-in,
                                        which is what these reels are made of
      high magnitude, low coherence  -> shake, a knock, a stumble
      low magnitude,  low coherence  -> a subject moving inside a locked frame

    |mean of the vectors| over mean of |the vectors|: 1.0 when every pixel moves
    together, near 0 when directions cancel out.
    """
    if magnitude <= 1e-6:
        return 1.0  # a still frame is perfectly coherent, not chaotic
    return float(min(np.hypot(dx, dy) / magnitude, 1.0))


def _motion_centre(flow: np.ndarray) -> tuple[float, float]:
    """Where the subject is, horizontally: centroid and spread, both 0..1.

    A poor man's subject tracker, and the only one available until reframe.py
    exists. Knowing this is the difference between a 9:16 crop that holds the
    subject and one that centres on the wall behind them.

    Measured on *residual* flow -- each vector minus the frame's mean -- rather
    than raw magnitude. Raw magnitude only finds a subject when the camera is
    locked off: hand-held or gimbal footage moves every pixel at once, so the
    centroid tracks the camera and lands mid-frame regardless of where anyone is.
    Measured on real gym footage, raw magnitude located a subject in 1 window out
    of 15. Subtracting the global motion leaves exactly what moves independently
    of the camera, which is the subject.

    `spread` reports how scattered that residual is. Broad spread means no single
    thing is moving on its own -- parallax, a crowd, noise -- and the caller
    should leave the crop centred.
    """
    mean = flow.reshape(-1, 2).mean(axis=0)
    return _profile_centre(np.linalg.norm(flow - mean, axis=2).sum(axis=0))


def _profile_centre(columns: np.ndarray) -> tuple[float, float]:
    """Centroid and spread of a per-column weight profile, both 0..1.

    Only the strongest 30% of columns count. Every column carries some weight --
    residual noise, faint texture -- and including all of it drags every centroid
    back to the middle of the frame, which is the answer this exists to improve on.
    """
    total = float(columns.sum())
    width = columns.size
    if total <= 1e-6 or width < 2:
        return 0.5, 1.0

    threshold = float(np.percentile(columns, 70))
    weights = np.where(columns >= threshold, columns, 0.0)
    total = float(weights.sum())
    if total <= 1e-6:
        return 0.5, 1.0
    weights = weights / total

    positions = np.arange(width, dtype=float)
    centre = float((positions * weights).sum())
    variance = float((((positions - centre) ** 2) * weights).sum())
    uniform = width / np.sqrt(12.0)   # a flat distribution's std
    return centre / (width - 1), float(min(np.sqrt(variance) / uniform, 1.0))


def _histogram(frame: np.ndarray) -> np.ndarray:
    """Coarse normalised colour histogram, for frame-to-frame novelty."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [HIST_BINS] * 3,
                        [0, 180, 0, 256, 0, 256])
    total = hist.sum()
    return (hist / total).ravel() if total else hist.ravel()


# ---------------------------------------------------------------- audio


def audio_rms_series(proxy: Path) -> tuple[list[float], float]:
    """Coarse loudness envelope of the proxy's audio, and its rate in windows/sec.

    Extracted through ffmpeg to a low-rate mono wav rather than decoded
    in-process: librosa's backends are unreliable on AAC-in-MP4, and this costs
    a few hundred KB.
    """
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "a.wav"
        proc = subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", str(proxy),
             "-ac", "1", "-ar", "8000", "-c:a", "pcm_s16le", str(wav)],
            capture_output=True, text=True,
        )
        if proc.returncode != 0 or not wav.exists():
            return [], 0.0

        import soundfile as sf
        data, rate = sf.read(str(wav), dtype="float32")

    if data.ndim > 1:
        data = data.mean(axis=1)

    window = max(int(rate * 0.1), 1)  # 100 ms
    usable = (len(data) // window) * window
    if usable == 0:
        return [], 0.0
    frames = data[:usable].reshape(-1, window)
    return np.sqrt((frames ** 2).mean(axis=1)).round(6).tolist(), 10.0


# ---------------------------------------------------------------- io


def features_path(name: str, features_dir: Path) -> Path:
    return features_dir / f"{name}.json"


def load(name: str, features_dir: Path | None = None) -> dict:
    path = features_path(name, features_dir or config.FEATURES_DIR)
    if not path.exists():
        raise FileNotFoundError(
            f"no features for '{name}' — run `python -m pipeline.features` first")
    return json.loads(path.read_text())


def measure(entry: dict, features_dir: Path, sample_fps: float) -> dict:
    proxy = Path(entry["proxy"])
    samples = sample_proxy(proxy, sample_fps)
    rms, rms_fps = audio_rms_series(proxy)
    doc = {
        "version": 1,
        "name": entry["name"],
        "proxy": str(proxy),
        "source": entry["path"],
        "duration": entry["duration"],
        "sample_fps": sample_fps,
        "samples": samples,
        "audio_rms": rms,
        "audio_rms_fps": rms_fps,
    }
    features_dir.mkdir(parents=True, exist_ok=True)
    features_path(entry["name"], features_dir).write_text(json.dumps(doc))
    return doc


# ---------------------------------------------------------------- entry


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Dense per-frame measurement of every proxy.")
    ap.add_argument("--manifest", type=Path, default=config.PROXIES_DIR / "manifest.json")
    ap.add_argument("--out-dir", type=Path, default=config.FEATURES_DIR)
    ap.add_argument("--sample-fps", type=float, default=config.SAMPLE_FPS)
    ap.add_argument("--force", action="store_true", help="ignore cached feature files")
    args = ap.parse_args(argv)

    if not args.manifest.exists():
        print(f"{RED}no proxy manifest — run `python -m pipeline.proxy` first{RESET}",
              file=sys.stderr)
        return 1

    entries = json.loads(args.manifest.read_text())
    footage = sum(float(e["duration"]) for e in entries)
    print(f"\n{BOLD}features{RESET}  {DIM}{len(entries)} proxy file(s) · "
          f"{footage:.0f}s of footage · {args.sample_fps:g} samples/s{RESET}\n")

    started = time.perf_counter()
    for entry in entries:
        path = features_path(entry["name"], args.out_dir)
        if path.exists() and not args.force:
            samples = len(json.loads(path.read_text())["samples"])
            print(f"  {DIM}cached {RESET}  {entry['name']:<28} {samples} samples")
            continue

        clock = time.perf_counter()
        doc = measure(entry, args.out_dir, args.sample_fps)
        audio = "with audio" if doc["audio_rms"] else "no audio"
        print(f"  {GREEN}measure{RESET}  {entry['name']:<28} {len(doc['samples'])} samples, "
              f"{DIM}{audio} · {time.perf_counter() - clock:.1f}s{RESET}")

    elapsed = time.perf_counter() - started
    rate = f" · {footage / elapsed:.0f}x realtime" if elapsed > 0.05 else ""
    print(f"\n{GREEN}measured {len(entries)} file(s){RESET}  "
          f"{DIM}{elapsed:.1f}s{rate} -> {args.out_dir}{RESET}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
