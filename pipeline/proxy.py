"""Generate 480p proxies once, so nothing downstream ever opens a 4K file.

This is what makes an 8 GB machine viable. Decode runs on the M2 media engine and
encode on the hardware H.264 encoder, so a proxy costs almost no memory and lands
at a few MB instead of a few hundred. Every analysis stage reads these; only the
final render touches the originals, and timecodes are source-relative seconds so
the two stay interchangeable.

    uv run python -m pipeline.proxy
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from pipeline import config, media

BOLD, RED, YELLOW, GREEN, DIM, RESET = "\033[1m", "\033[31m", "\033[33m", "\033[32m", "\033[2m", "\033[0m"


def proxy_path(source: Path, proxies_dir: Path) -> Path:
    return proxies_dir / f"{source.stem}.mp4"


def is_current(source: Path, proxy: Path) -> bool:
    """A proxy is current when it exists and post-dates its source."""
    return proxy.exists() and proxy.stat().st_mtime >= source.stat().st_mtime


def build_command(source: Path, dest: Path) -> list[str]:
    return [
        "ffmpeg", "-y", "-v", "error",
        "-hwaccel", "videotoolbox",
        "-i", str(source),
        "-vf", f"scale=-2:{config.PROXY_HEIGHT}",
        "-c:v", "h264_videotoolbox", "-b:v", config.PROXY_BITRATE,
        # Audio is kept: signals.py reads it for an energy curve, and it costs
        # almost nothing at this bitrate.
        "-c:a", "aac", "-b:a", "96k",
        "-movflags", "+faststart",
        str(dest),
    ]


def generate(sources: list[dict], proxies_dir: Path, force: bool = False) -> list[dict]:
    proxies_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []

    for src in sources:
        source = Path(src["path"])
        dest = proxy_path(source, proxies_dir)

        if is_current(source, dest) and not force:
            print(f"  {DIM}cached{RESET}  {dest.name}")
        else:
            start = time.perf_counter()
            media.run(build_command(source, dest), desc=f"proxy {source.name}")
            size_mb = dest.stat().st_size / (1024 * 1024)
            print(f"  {GREEN}proxy {RESET}  {dest.name:<28} {size_mb:5.1f} MB  "
                  f"{DIM}{time.perf_counter() - start:.1f}s{RESET}")

        results.append({**src, "proxy": str(dest)})

    return results


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build 480p analysis proxies.")
    ap.add_argument("--sources", type=Path, default=config.SOURCES_JSON)
    ap.add_argument("--out", type=Path, default=config.PROXIES_DIR)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    if not args.sources.exists():
        print(f"{RED}no sources.json — run `python -m pipeline.ingest` first{RESET}", file=sys.stderr)
        return 1

    accepted = json.loads(args.sources.read_text())["accepted"]
    if not accepted:
        print(f"{RED}no accepted sources to proxy{RESET}", file=sys.stderr)
        return 1

    print(f"\n{BOLD}proxy{RESET}  {DIM}{len(accepted)} source(s) -> {config.PROXY_HEIGHT}p{RESET}\n")
    started = time.perf_counter()
    results = generate(accepted, args.out, force=args.force)

    manifest = args.out / "manifest.json"
    manifest.write_text(json.dumps(results, indent=2))

    total_mb = sum(Path(r["proxy"]).stat().st_size for r in results) / (1024 * 1024)
    print(f"\n{GREEN}{len(results)} proxies{RESET}  {DIM}{total_mb:.1f} MB total · "
          f"{time.perf_counter() - started:.1f}s -> {manifest}{RESET}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
