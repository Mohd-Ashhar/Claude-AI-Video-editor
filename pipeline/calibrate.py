"""Report what your footage actually measures, and what the constants should be.

The motion thresholds in config.py are the only numbers in this system that
cannot be derived -- they depend on how you shoot. They were calibrated against
test footage, and test footage is not your footage: a handheld vlog and a
tripod-locked landscape produce completely different distributions, and a
threshold set for one silently mis-scores the other.

Run this once on a real project. It prints the distribution, says which of your
samples the current settings call "shake", and suggests values. It changes
nothing on its own -- you copy what you agree with into config.py.

    uv run python -m pipeline.calibrate
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from pipeline import config

BOLD, RED, YELLOW, GREEN, DIM, RESET = "\033[1m", "\033[31m", "\033[33m", "\033[32m", "\033[2m", "\033[0m"


def load_samples(features_dir: Path) -> tuple[np.ndarray, np.ndarray, list[tuple[str, int]]]:
    files = sorted(features_dir.glob("*.json"))
    magnitudes, coherences, counts = [], [], []
    for path in files:
        doc = json.loads(path.read_text())
        samples = doc["samples"]
        magnitudes.append(np.array([s["flow_mag"] for s in samples]))
        coherences.append(np.array([s.get("coherence", 1.0) for s in samples]))
        counts.append((doc["name"], len(samples)))
    if not magnitudes:
        return np.array([]), np.array([]), []
    return np.concatenate(magnitudes), np.concatenate(coherences), counts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Calibrate motion constants to your footage.")
    ap.add_argument("--features", type=Path, default=config.FEATURES_DIR)
    args = ap.parse_args(argv)

    magnitude, coherence, counts = load_samples(args.features)
    if magnitude.size == 0:
        print(f"{RED}no feature files in {args.features} — run "
              f"`python -m pipeline.features` first{RESET}", file=sys.stderr)
        return 1

    print(f"\n{BOLD}calibrate{RESET}  {DIM}{magnitude.size} samples from "
          f"{len(counts)} file(s){RESET}\n")

    print(f"  {BOLD}motion magnitude{RESET}")
    for percentile in (10, 25, 50, 75, 90, 95, 99):
        value = float(np.percentile(magnitude, percentile))
        print(f"    p{percentile:<3} {value:>7.3f}")

    still = magnitude < config.MOTION_STILL
    moving = magnitude[~still]
    print(f"\n  {DIM}{still.sum() / magnitude.size * 100:.0f}% of samples are below "
          f"MOTION_STILL ({config.MOTION_STILL}) — treated as locked off{RESET}")

    if moving.size < 10:
        print(f"\n  {YELLOW}almost nothing in this batch is moving; the motion "
              f"constants barely matter for this footage{RESET}\n")
        return 0

    median_move = float(np.median(moving))
    shake_floor = float(np.percentile(moving, 90))

    print(f"\n  {BOLD}current settings against this footage{RESET}")
    # Motion alone, before smoothness and coherence gate it -- this shows how
    # much of the movement in your footage the curve even registers.
    scored = 1.0 - np.exp(-magnitude / config.MOTION_SATURATION)
    print(f"    MOTION_SATURATION {config.MOTION_SATURATION}")
    print(f"    {(scored > 0.8).sum() / magnitude.size * 100:>5.0f}% of samples reach "
          f"0.8+ on the raw motion term")
    print(f"    {(scored < 0.1).sum() / magnitude.size * 100:>5.0f}% sit below 0.1 "
          f"{DIM}(barely moving){RESET}")

    whippable = (magnitude >= config.WHIP_MIN_MAGNITUDE) & (coherence >= config.WHIP_MIN_COHERENCE)
    print(f"    {whippable.sum() / magnitude.size * 100:>5.0f}% of samples could support "
          f"a whip {DIM}(mag >= {config.WHIP_MIN_MAGNITUDE}, coherent){RESET}")

    print(f"\n  {BOLD}suggested{RESET}  {DIM}copy into config.py if you agree{RESET}")
    print(f"    MOTION_SATURATION = {max(median_move / 2.2, 0.3):.2f}   "
          f"{DIM}# your median move lands around 0.8 on the motion term{RESET}")
    print(f"    WHIP_MIN_MAGNITUDE = {float(np.percentile(moving, 60)):.2f} "
          f"{DIM}# top 40% of your moving samples{RESET}")
    print(f"    FLASH_MIN_EXIT = {float(np.percentile(moving, 85)):.2f} "
          f"{DIM}# top 15%, i.e. genuinely fast exits{RESET}")

    if shake_floor > median_move * 3:
        print(f"\n    {YELLOW}note: your p90 ({shake_floor:.2f}) is far above your median "
              f"({median_move:.2f}). Motion that extreme is judged by smoothness, not "
              f"by magnitude, so no threshold change is needed for it.{RESET}")

    if whippable.sum() == 0:
        print(f"\n  {YELLOW}no sample in this batch clears the whip bar — directional "
              f"transitions can never fire on this footage until you lower it{RESET}")

    print(f"\n{GREEN}nothing was changed{RESET}  {DIM}this command only reports{RESET}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
