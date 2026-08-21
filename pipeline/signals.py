"""Aggregate the dense feature series into one score per moment.

This replaces the CLIP + LAION aesthetic stage from the research plan. That model
is trained on still-image art ratings: it rewards sunset stills and punishes
motion blur, which is backwards for travel and fitness footage, and it costs 2 GB
of memory and 2.5 GB of disk to be wrong. Everything here is arithmetic over
numbers features.py already measured, and measures things that are actually true
of a frame.

The measurement pass itself moved to features.py when the input became long
continuous takes -- see that module for why. This stage no longer opens a video
at all; it reads work/features/*.json and work/moments.json.

What it cannot judge is content -- whether a shot is *interesting*. That is what
vlm_tag.py is for.

    uv run python -m pipeline.signals
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from pipeline import config, features

BOLD, RED, YELLOW, GREEN, DIM, RESET = "\033[1m", "\033[31m", "\033[33m", "\033[32m", "\033[2m", "\033[0m"


# ---------------------------------------------------------------- aggregation


def aggregate(shot: dict, samples: list[dict], rms: np.ndarray, rms_fps: float) -> dict:
    span = [s for s in samples if shot["start"] <= s["t"] < shot["end"]]
    if not span:
        span = samples[:1] or [{
            "sharpness": 0.0, "luma": 128.0, "clip_high": 0.0, "clip_low": 0.0,
            "flow_mag": 0.0, "flow_dx": 0.0, "flow_dy": 0.0, "coherence": 1.0,
        }]

    sharp = np.array([s["sharpness"] for s in span])
    luma = np.array([s["luma"] for s in span])
    hi = np.array([s["clip_high"] for s in span])
    lo = np.array([s["clip_low"] for s in span])
    mag = np.array([s["flow_mag"] for s in span])
    dx = np.array([s["flow_dx"] for s in span])
    dy = np.array([s["flow_dy"] for s in span])
    coh = np.array([s.get("coherence", 1.0) for s in span])

    # Jitter: how much the dominant motion direction flips between samples. The
    # Pocket 3's gimbal keeps this near zero, so a high value is a real defect
    # (a knock, a stumble) rather than intentional camera movement.
    if len(dx) > 2:
        angles = np.arctan2(dy, dx)
        delta = np.abs(np.diff(angles))
        delta = np.minimum(delta, 2 * np.pi - delta)
        moving = mag[1:] > 0.05
        jitter = float(delta[moving].mean()) if moving.any() else 0.0
    else:
        jitter = 0.0

    if rms.size and rms_fps:
        a0, a1 = int(shot["start"] * rms_fps), int(shot["end"] * rms_fps)
        window = rms[a0:a1]
        audio_rms = float(window.mean()) if window.size else 0.0
        audio_peak = float(window.max()) if window.size else 0.0
    else:
        audio_rms = audio_peak = 0.0

    return {
        "sharpness_raw": float(np.median(sharp)),
        "luma_mean": float(luma.mean()),
        "clip_high": float(hi.mean()),
        "clip_low": float(lo.mean()),
        "motion_raw": float(np.median(mag)),
        "motion_peak": float(mag.max()) if mag.size else 0.0,
        "coherence": float(coh.mean()),
        "jitter": jitter,
        "audio_rms": audio_rms,
        "audio_peak": audio_peak,
        "samples": len(span),
        # Kept for sequence.py, which trims a slot out of the middle of a moment.
        "series": [
            {"t": s["t"], "sharpness": round(s["sharpness"], 2),
             "motion": round(s["flow_mag"], 4)}
            for s in span
        ],
    }


def percentile_rank(values: list[float]) -> list[float]:
    """Rank each value in 0..1 against the batch.

    Absolute thresholds do not transfer between a bright beach clip and a dim gym,
    so quality is scored relative to the footage actually in hand.
    """
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return []
    if np.allclose(array, array[0]):
        return [0.5] * array.size
    order = array.argsort().argsort()
    return (order / max(array.size - 1, 1)).tolist()


def stability_score(jitter: float) -> float:
    """Absolute, not ranked.

    Sharpness only means something relative to the rest of the batch, but shake
    has a physical scale: the mean angular change between consecutive motion
    vectors approaches pi/2 when direction is random and sits near zero for smooth
    gimbal movement. Ranking this instead would hand a 0.0 to the least steady clip
    in a set of perfectly steady clips, and punish it in the composite score.
    """
    return float(max(0.0, 1.0 - min(jitter / (np.pi / 2), 1.0)))


def exposure_score(metrics: dict) -> float:
    """Penalise blown highlights, crushed shadows, and a badly-placed midtone.

    Clipping is the real defect and carries most of the weight. The midtone term
    is deliberately gentle and centred well below mid-grey: footage graded from
    D-Log M lands dark on purpose, and judging that against a neutral mid-grey
    marks down an entire look rather than any actual fault.
    """
    penalty = min(metrics["clip_high"] * 4.0, 1.0) * 0.6
    penalty += min(metrics["clip_low"] * 4.0, 1.0) * 0.4
    drift = abs(metrics["luma_mean"] - config.EXPOSURE_TARGET_LUMA) / config.EXPOSURE_TARGET_LUMA
    penalty += min(drift, 1.0) * config.EXPOSURE_DRIFT_WEIGHT
    return float(max(0.0, 1.0 - penalty))


def score_batch(shots: list[dict]) -> None:
    """Attach normalised 0..1 scores in place, ranked across the whole batch."""
    sharp_rank = percentile_rank([s["signals"]["sharpness_raw"] for s in shots])
    motion_rank = percentile_rank([s["signals"]["motion_raw"] for s in shots])

    for shot, sharpness, motion in zip(shots, sharp_rank, motion_rank):
        sig = shot["signals"]
        sig["sharpness_score"] = round(sharpness, 4)
        sig["motion_energy"] = round(motion, 4)
        sig["exposure_score"] = round(exposure_score(sig), 4)
        sig["stability_score"] = round(stability_score(sig["jitter"]), 4)
        sig["technical_score"] = round(
            0.4 * sig["sharpness_score"]
            + 0.35 * sig["exposure_score"]
            + 0.25 * sig["stability_score"],
            4,
        )


# ---------------------------------------------------------------- entry


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Score each moment from the dense features.")
    ap.add_argument("--shots", type=Path, default=config.MOMENTS_JSON,
                    help="moments.json (or shots.json to score whole takes)")
    ap.add_argument("--features", type=Path, default=config.FEATURES_DIR)
    ap.add_argument("--out", type=Path, default=config.SIGNALS_JSON)
    args = ap.parse_args(argv)

    if not args.shots.exists():
        print(f"{RED}no {args.shots.name} — run `python -m pipeline.moments` first{RESET}",
              file=sys.stderr)
        return 1

    shots = json.loads(args.shots.read_text())
    by_name: dict[str, list[dict]] = {}
    for shot in shots:
        by_name.setdefault(shot["name"], []).append(shot)

    print(f"\n{BOLD}signals{RESET}  {DIM}{len(shots)} moment(s) across "
          f"{len(by_name)} file(s){RESET}\n")
    started = time.perf_counter()

    for name, group in by_name.items():
        try:
            doc = features.load(name, args.features)
        except FileNotFoundError as exc:
            print(f"{RED}{exc}{RESET}", file=sys.stderr)
            return 1

        rms = np.asarray(doc.get("audio_rms", []), dtype=float)
        rms_fps = float(doc.get("audio_rms_fps", 0.0))
        for shot in group:
            shot["signals"] = aggregate(shot, doc["samples"], rms, rms_fps)
        print(f"  {GREEN}score {RESET}  {name:<28} {len(group)} moment(s)")

    score_batch(shots)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(shots, indent=2))

    print(f"\n  {BOLD}{'moment':<24}{'sharp':>7}{'expo':>7}{'stable':>8}{'motion':>8}{'score':>8}{RESET}")
    for shot in sorted(shots, key=lambda s: -s["signals"]["technical_score"])[:12]:
        sig = shot["signals"]
        print(f"  {shot['shot_id']:<24}{sig['sharpness_score']:>7.2f}{sig['exposure_score']:>7.2f}"
              f"{sig['stability_score']:>8.2f}{sig['motion_energy']:>8.2f}"
              f"{sig['technical_score']:>8.2f}")

    print(f"\n{GREEN}scored {len(shots)} moments{RESET}  "
          f"{DIM}{time.perf_counter() - started:.1f}s -> {args.out}{RESET}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
