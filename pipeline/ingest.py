"""Probe every source file, and refuse the ones that cannot make a sharp reel.

The gate that matters: a 9:16 crop takes only `height * 9/16` pixels of width
from a 16:9 source. From 4K that is 1215px, which downscales cleanly to 1080.
From a 1080p export it is 607px -- a 78% upscale that lands visibly soft and is
invisible on a laptop preview. Since colour grading happens upstream (Mimo ->
iPhone), a 1080p export from that app would silently ruin every reel, so this
check runs before anything else touches the footage.

    uv run python -m pipeline.ingest
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from pipeline import config, media

BOLD, RED, YELLOW, GREEN, DIM, RESET = "\033[1m", "\033[31m", "\033[33m", "\033[32m", "\033[2m", "\033[0m"


@dataclass
class Source:
    path: str
    name: str
    width: int
    height: int
    fps: float
    duration: float
    codec: str
    pix_fmt: str
    color_transfer: str
    color_primaries: str
    has_audio: bool
    max_crop_width: int
    status: str = "ok"
    reasons: list[str] = field(default_factory=list)


def max_crop_width(width: int, height: int) -> int:
    """Widest 9:16 region extractable from a WxH source, in source pixels.

    This is the number that decides sharpness. If it is below the delivery
    width, the render must upscale and detail is gone for good. It comes from
    config.crop_window -- the same geometry render.py crops to -- so the gate can
    never accept a source the renderer would then have to upscale.
    """
    return config.crop_window(width, height)[0]


def inspect(path: Path) -> Source:
    info = media.probe(path)
    stream = media.video_stream(info)
    width, height = media.display_dimensions(stream)

    src = Source(
        path=str(path),
        name=path.stem,
        width=width,
        height=height,
        fps=round(media.parse_fps(stream), 3),
        duration=round(media.duration_seconds(info), 3),
        codec=stream.get("codec_name", "?"),
        pix_fmt=stream.get("pix_fmt", "?"),
        color_transfer=stream.get("color_transfer", "unknown"),
        color_primaries=stream.get("color_primaries", "unknown"),
        has_audio=media.audio_stream(info) is not None,
        max_crop_width=max_crop_width(width, height),
    )

    if src.max_crop_width < config.OUT_W:
        upscale = (config.OUT_W / src.max_crop_width - 1) * 100 if src.max_crop_width else 0
        src.status = "rejected"
        src.reasons.append(
            f"a 9:16 crop yields only {src.max_crop_width}px of width from {width}x{height}; "
            f"filling {config.OUT_W}px needs a {upscale:.0f}% upscale. "
            f"Re-export this clip at 4K."
        )

    if src.color_transfer in config.HDR_TRANSFERS:
        src.status = "rejected"
        src.reasons.append(
            f"still tagged HDR ({src.color_transfer}); Instagram handles this badly. "
            f"This ffmpeg has no zscale, so normalise with: "
            f"-vf \"colorspace=all=bt709:iall=bt2020ncl:fast=1,format=yuv420p\""
        )

    if src.duration < 0.5:
        src.status = "rejected"
        src.reasons.append(f"only {src.duration:.2f}s long; too short to cut with")

    return src


def scan(inputs_dir: Path) -> list[Source]:
    files = sorted(
        p for p in inputs_dir.iterdir()
        if p.is_file() and p.suffix in config.VIDEO_EXTENSIONS and not p.name.startswith(".")
    )
    return [inspect(p) for p in files]


def report(sources: list[Source]) -> None:
    print(f"\n{BOLD}ingest{RESET}  {DIM}{len(sources)} file(s){RESET}\n")
    for src in sources:
        mark = f"{GREEN}OK  {RESET}" if src.status == "ok" else f"{RED}REJECT{RESET}"
        print(f"  {mark}  {src.name:<28} {src.width}x{src.height} @{src.fps:g}fps  "
              f"{src.duration:6.1f}s  {DIM}{src.codec} {src.pix_fmt} "
              f"crop-width {src.max_crop_width}px{RESET}")
        for reason in src.reasons:
            print(f"          {YELLOW}{reason}{RESET}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Probe and gate source footage.")
    ap.add_argument("--inputs", type=Path, default=config.INPUTS_DIR)
    ap.add_argument("--out", type=Path, default=config.SOURCES_JSON)
    ap.add_argument("--allow-partial", action="store_true",
                    help="continue with the accepted clips instead of halting on any rejection")
    args = ap.parse_args(argv)

    if not args.inputs.is_dir():
        print(f"{RED}no input directory: {args.inputs}{RESET}", file=sys.stderr)
        return 1

    sources = scan(args.inputs)
    if not sources:
        print(f"{RED}no video files in {args.inputs}{RESET}", file=sys.stderr)
        return 1

    report(sources)
    accepted = [s for s in sources if s.status == "ok"]
    rejected = [s for s in sources if s.status != "ok"]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    # The directory scanned is part of the result, not just provenance. auto.py
    # skips a stage whose output already exists, and without this it will happily
    # reuse an analysis of a *different* shoot: pointed at a new folder it rebuilt
    # the previous project's reel and reported success. Recorded here so the skip
    # can be conditional on the inputs actually matching.
    args.out.write_text(json.dumps(
        {"inputs": str(Path(args.inputs).resolve()),
         "accepted": [asdict(s) for s in accepted],
         "rejected": [asdict(s) for s in rejected]},
        indent=2,
    ))

    print(f"\n  {len(accepted)} accepted, {len(rejected)} rejected  {DIM}-> {args.out}{RESET}")

    if rejected and not args.allow_partial:
        print(f"\n{RED}halted — fix the rejected clips, or re-run with --allow-partial{RESET}\n")
        return 1

    print(f"\n{GREEN}ingest clear{RESET}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
