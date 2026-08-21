"""Verify a rendered reel against the delivery spec before it ever reaches upload.

Checks the things that are cheap to get wrong and expensive to discover after
posting: frame size, pixel format, colour tagging (an untagged or mistagged file
washes out on Instagram), frame rate, duration, and integrated loudness.

    uv run python -m pipeline.qa out/reel.mp4
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from pipeline import config, media

BOLD, RED, YELLOW, GREEN, DIM, RESET = "\033[1m", "\033[31m", "\033[33m", "\033[32m", "\033[2m", "\033[0m"


class Report:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.warnings: list[str] = []

    def check(self, label: str, actual, expected, *, warn_only: bool = False) -> None:
        ok = actual == expected
        if ok:
            mark = f"{GREEN}OK  {RESET}"
        elif warn_only:
            mark = f"{YELLOW}WARN{RESET}"
            self.warnings.append(label)
        else:
            mark = f"{RED}FAIL{RESET}"
            self.failures.append(label)
        suffix = "" if ok else f"   {DIM}expected {expected}{RESET}"
        print(f"  {mark}  {label:<18} {actual}{suffix}")

    def note(self, label: str, value: str) -> None:
        print(f"  {DIM}····{RESET}  {label:<18} {value}")


def measure_loudness(path: Path) -> float | None:
    """Integrated LUFS via ebur128. Returns None when the file has no audio."""
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", str(path),
         "-af", "ebur128=framelog=quiet", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    if m := re.search(r"I:\s*(-?[\d.]+)\s*LUFS", proc.stderr):
        return float(m.group(1))
    return None


def audit(path: Path, expect_duration: float | None = None) -> Report:
    info = media.probe(path)
    stream = media.video_stream(info)
    report = Report()

    print(f"\n{BOLD}qa{RESET}  {DIM}{path}{RESET}\n")

    # The mov and mp4 muxers share a demuxer name, so the extension is what
    # actually decides the wrapper. Reported rather than asserted: both play
    # everywhere this delivers to, and the codec tag below is the part that
    # matters on Apple devices.
    report.note("container", f"{path.suffix.lstrip('.') or '?'}   "
                             f"{stream.get('codec_name', '?')} / "
                             f"{stream.get('codec_tag_string', '?')}")

    if stream.get("codec_name") == "hevc" and stream.get("codec_tag_string") != "hvc1":
        report.warnings.append(
            f"HEVC tagged '{stream.get('codec_tag_string')}' rather than hvc1 — "
            f"QuickTime and Photos will refuse it")

    width, height = media.display_dimensions(stream)
    report.check("resolution", f"{width}x{height}", f"{config.OUT_W}x{config.OUT_H}")
    report.check("pixel format", stream.get("pix_fmt"), "yuv420p")
    report.check("colour primaries", stream.get("color_primaries"), "bt709")
    report.check("colour transfer", stream.get("color_transfer"), "bt709")
    report.check("colour matrix", stream.get("color_space"), "bt709")

    fps = round(media.parse_fps(stream), 3)
    report.check("frame rate", fps, float(config.OUT_FPS))

    duration = media.duration_seconds(info)
    if expect_duration is not None:
        drift = abs(duration - expect_duration)
        ok = drift <= 0.15
        mark = f"{GREEN}OK  {RESET}" if ok else f"{RED}FAIL{RESET}"
        if not ok:
            report.failures.append("duration")
        print(f"  {mark}  {'duration':<18} {duration:.2f}s   "
              f"{DIM}timeline says {expect_duration:.2f}s (drift {drift:.3f}s){RESET}")
    else:
        report.note("duration", f"{duration:.2f}s")

    if not 3 <= duration <= 180:
        report.warnings.append("duration outside the 3-180s Reels range")

    audio = media.audio_stream(info)
    if audio is None:
        report.note("audio", "none")
        report.warnings.append("no audio track")
    else:
        lufs = measure_loudness(path)
        if lufs is None:
            report.note("loudness", "unmeasurable")
        else:
            ok = abs(lufs - config.TARGET_LUFS) <= 1.5
            mark = f"{GREEN}OK  {RESET}" if ok else f"{YELLOW}WARN{RESET}"
            if not ok:
                report.warnings.append("loudness")
            print(f"  {mark}  {'loudness':<18} {lufs:.1f} LUFS   "
                  f"{DIM}target {config.TARGET_LUFS}{RESET}")

    size_mb = path.stat().st_size / (1024 * 1024)
    report.note("file size", f"{size_mb:.1f} MB")
    report.note("faststart", "yes" if _has_faststart(path) else "no")

    return report


def _has_faststart(path: Path) -> bool:
    """moov before mdat means the file starts playing before it finishes downloading."""
    with path.open("rb") as fh:
        head = fh.read(4 * 1024 * 1024)
    moov, mdat = head.find(b"moov"), head.find(b"mdat")
    return moov != -1 and (mdat == -1 or moov < mdat)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Check a rendered reel against the delivery spec.")
    ap.add_argument("video", type=Path, nargs="?", default=config.out_path("reel"))
    ap.add_argument("--timeline", type=Path, default=None,
                    help="compare duration against this timeline")
    args = ap.parse_args(argv)

    if not args.video.exists():
        print(f"{RED}no such file: {args.video}{RESET}", file=sys.stderr)
        return 1

    expected = None
    if args.timeline and args.timeline.exists():
        from pipeline.render import segment_duration
        timeline = json.loads(args.timeline.read_text())
        expected = sum(segment_duration(seg) for seg in timeline["segments"])

    report = audit(args.video, expected)

    if report.failures:
        print(f"\n{RED}QA failed:{RESET} {', '.join(report.failures)}\n")
        return 1
    if report.warnings:
        print(f"\n{YELLOW}QA passed with warnings:{RESET} {', '.join(report.warnings)}\n")
        return 0
    print(f"\n{GREEN}QA passed{RESET}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
