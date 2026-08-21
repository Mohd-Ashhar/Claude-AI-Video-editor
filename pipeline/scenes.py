"""Split each source into shots by detecting cuts on its proxy.

Runs on 480p, which is 20-40x cheaper than the source and yields identical cut
timecodes -- they are resolution-independent. Pocket 3 clips are often a single
continuous take, in which case the whole file becomes one shot; the detector
earns its place on longer takes and on anything already assembled.

    uv run python -m pipeline.scenes
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from scenedetect import AdaptiveDetector, detect

from pipeline import config

BOLD, RED, YELLOW, GREEN, DIM, RESET = "\033[1m", "\033[31m", "\033[33m", "\033[32m", "\033[2m", "\033[0m"

# Below this a "shot" is a detector artefact, not something you can cut with.
MIN_SHOT_SECONDS = 0.4


def detect_shots(proxy: Path, duration: float) -> list[tuple[float, float]]:
    """Return (start, end) pairs in seconds. Falls back to the whole clip."""
    try:
        scenes = detect(str(proxy), AdaptiveDetector(), show_progress=False)
    except Exception as exc:  # noqa: BLE001 - a detector failure must not lose the clip
        print(f"  {YELLOW}detector failed on {proxy.name} ({exc}); treating as one shot{RESET}")
        return [(0.0, duration)]

    if not scenes:
        return [(0.0, duration)]

    spans = [(s.get_seconds(), e.get_seconds()) for s, e in scenes]
    kept = [(s, e) for s, e in spans if e - s >= MIN_SHOT_SECONDS]
    return kept or [(0.0, duration)]


def build(entries: list[dict]) -> list[dict]:
    shots: list[dict] = []

    for entry in entries:
        proxy = Path(entry["proxy"])
        spans = detect_shots(proxy, float(entry["duration"]))

        for index, (start, end) in enumerate(spans):
            shots.append({
                "shot_id": f"{entry['name']}__{index:02d}",
                "source": entry["path"],
                "proxy": str(proxy),
                "name": entry["name"],
                "start": round(start, 3),
                "end": round(end, 3),
                "duration": round(end - start, 3),
                "fps": entry["fps"],
                "width": entry["width"],
                "height": entry["height"],
                "max_crop_width": entry["max_crop_width"],
            })

        label = "1 shot" if len(spans) == 1 else f"{len(spans)} shots"
        print(f"  {GREEN}cut   {RESET}  {entry['name']:<28} {label}")

    return shots


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Detect shot boundaries on proxies.")
    ap.add_argument("--manifest", type=Path, default=config.PROXIES_DIR / "manifest.json")
    ap.add_argument("--out", type=Path, default=config.WORK_DIR / "shots.json")
    args = ap.parse_args(argv)

    if not args.manifest.exists():
        print(f"{RED}no proxy manifest — run `python -m pipeline.proxy` first{RESET}", file=sys.stderr)
        return 1

    entries = json.loads(args.manifest.read_text())
    print(f"\n{BOLD}scenes{RESET}  {DIM}{len(entries)} proxy file(s){RESET}\n")

    started = time.perf_counter()
    shots = build(entries)
    args.out.write_text(json.dumps(shots, indent=2))

    total = sum(s["duration"] for s in shots)
    print(f"\n{GREEN}{len(shots)} shots{RESET}  {DIM}{total:.1f}s of usable footage · "
          f"{time.perf_counter() - started:.1f}s -> {args.out}{RESET}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
