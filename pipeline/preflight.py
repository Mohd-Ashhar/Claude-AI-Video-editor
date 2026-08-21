"""Blocking resource and toolchain guard. Runs first, every time.

Deliberately stdlib-only: this has to work even when the venv is half-installed
or broken, because its whole job is to fail early and legibly instead of letting
a stage thrash an 8 GB machine into swap or die mid-render on a missing filter.

Exit codes:
    0  safe to proceed
    1  resource pressure (RAM, disk)
    2  toolchain defect (missing ffmpeg filter/encoder, ffmpeg absent)
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys

from pipeline import config

BOLD, RED, YELLOW, GREEN, DIM, RESET = "\033[1m", "\033[31m", "\033[33m", "\033[32m", "\033[2m", "\033[0m"

GIB = 1024 ** 3


# ---------------------------------------------------------------- memory


def available_ram_gb() -> float:
    """Approximate macOS 'available' memory: free + inactive + speculative + purgeable."""
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    page_size = 4096
    if m := re.search(r"page size of (\d+) bytes", out):
        page_size = int(m.group(1))

    pages = {}
    for line in out.splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        if digits := re.search(r"(\d+)", value):
            pages[key.strip()] = int(digits.group(1))

    reclaimable = sum(
        pages.get(k, 0)
        for k in ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable")
    )
    return reclaimable * page_size / GIB


def swap_usage_gb() -> tuple[float, float]:
    """Return (used_gb, total_gb) of the macOS swap file."""
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    total = used = 0.0
    if m := re.search(r"total = ([\d.]+)M", out):
        total = float(m.group(1)) / 1024
    if m := re.search(r"used = ([\d.]+)M", out):
        used = float(m.group(1)) / 1024
    return used, total


def top_memory_consumers(n: int = 5) -> list[tuple[str, float]]:
    """Return the n processes holding the most resident memory, as (name, percent)."""
    out = subprocess.run(
        ["ps", "-Ao", "pmem,comm"], capture_output=True, text=True
    ).stdout.splitlines()[1:]

    rows: list[tuple[str, float]] = []
    for line in out:
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        try:
            pct = float(parts[0])
        except ValueError:
            continue
        # Collapse macOS app bundle paths down to the app name.
        name = parts[1].strip()
        if ".app/" in name:
            name = name.split(".app/")[0].split("/")[-1] + ".app"
        else:
            name = name.split("/")[-1]
        rows.append((name, pct))

    merged: dict[str, float] = {}
    for name, pct in rows:
        merged[name] = merged.get(name, 0.0) + pct
    return sorted(merged.items(), key=lambda kv: -kv[1])[:n]


# ---------------------------------------------------------------- toolchain


def ffmpeg_capabilities() -> tuple[set[str], set[str]]:
    """Return (filter_names, encoder_names) supported by the ffmpeg on PATH."""
    def names(flag: str, column: int) -> set[str]:
        out = subprocess.run(
            ["ffmpeg", "-hide_banner", flag], capture_output=True, text=True
        ).stdout
        found: set[str] = set()
        for line in out.splitlines():
            parts = line.split()
            if len(parts) > column and not line.startswith(("Filters", "Encoders", " ---", "-----")):
                found.add(parts[column])
        return found

    return names("-filters", 1), names("-encoders", 1)


# ---------------------------------------------------------------- report


def print_guidance(problems: list[str], toolchain_fault: bool) -> None:
    """Print the concrete remedy for each failure, not just the failure."""
    if "ram" in problems:
        print(f"\n{BOLD}Close these before running heavy stages:{RESET}")
        for name, pct in top_memory_consumers():
            print(f"    {pct:5.1f}%  {name}")
        print(f"\n  {DIM}Run the pipeline from Terminal with the editor and browser closed.{RESET}")

    if "disk" in problems:
        print(f"\n{BOLD}Free space:{RESET}")
        print(f"    rm -rf {config.WORK_DIR}/proxies/* {config.WORK_DIR}/segments/*")
        print(f"  {DIM}Or point REEL_INPUTS_DIR and TMPDIR at an external SSD.{RESET}")

    if toolchain_fault:
        print(f"\n{BOLD}Toolchain:{RESET}  brew reinstall ffmpeg   {DIM}(or install a static "
              f"build from evermeet.cx for libass/libzimg support){RESET}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Resource and toolchain guard.")
    ap.add_argument("--need-ram", type=float, default=config.MIN_FREE_RAM_GB,
                    help="GB of available RAM this stage requires")
    ap.add_argument("--need-disk", type=float, default=config.MIN_FREE_DISK_GB,
                    help="GB of free disk this stage requires")
    ap.add_argument("--stage", default=None, help="name of the calling stage, for the header")
    ap.add_argument("--warn-only", action="store_true",
                    help="report but always exit 0 (development use)")
    args = ap.parse_args(argv)

    label = f" · {args.stage}" if args.stage else ""
    print(f"\n{BOLD}preflight{RESET}  {DIM}reel-editor · M2 Air 8 GB{label}{RESET}\n")
    problems: list[str] = []
    toolchain_fault = False

    # -- ffmpeg present at all
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        print(f"  {RED}FAIL{RESET}  ffmpeg/ffprobe not on PATH")
        print(f"\n        Fix:  brew install ffmpeg\n")
        return 2

    # -- memory
    ram = available_ram_gb()
    swap_used, swap_total = swap_usage_gb()
    ok = ram >= args.need_ram
    print(f"  {GREEN + 'OK  ' if ok else RED + 'FAIL'}{RESET}  RAM available   {ram:5.1f} GB   "
          f"{DIM}(need {args.need_ram:.1f}){RESET}")
    if not ok:
        problems.append("ram")

    swap_pct = (swap_used / swap_total * 100) if swap_total else 0.0
    swap_tight = swap_pct > 80
    print(f"  {YELLOW + 'WARN' if swap_tight else GREEN + 'OK  '}{RESET}  swap in use     "
          f"{swap_used:5.1f} GB   {DIM}of {swap_total:.1f} GB ({swap_pct:.0f}%){RESET}")

    # -- disk
    free_disk = shutil.disk_usage(config.PROJECT_ROOT).free / GIB
    ok = free_disk >= args.need_disk
    print(f"  {GREEN + 'OK  ' if ok else RED + 'FAIL'}{RESET}  disk free       {free_disk:5.1f} GB   "
          f"{DIM}(need {args.need_disk:.0f}; 4K writes ~1 GB/min){RESET}")
    if not ok:
        problems.append("disk")

    # -- toolchain
    filters, encoders = ffmpeg_capabilities()
    missing_f = [f for f in config.REQUIRED_FILTERS if f not in filters]
    missing_e = [e for e in config.REQUIRED_ENCODERS if e not in encoders]
    if missing_f or missing_e:
        toolchain_fault = True
        print(f"  {RED}FAIL{RESET}  ffmpeg          missing {', '.join(missing_f + missing_e)}")
    else:
        print(f"  {GREEN}OK  {RESET}  ffmpeg          all {len(config.REQUIRED_FILTERS)} filters, "
              f"{len(config.REQUIRED_ENCODERS)} encoders present")

    absent = [f for f in config.OPTIONAL_FILTERS if f not in filters]
    if absent:
        print(f"  {DIM}note  unavailable in this build: {', '.join(absent)} "
              f"— stages route around these{RESET}")

    print_guidance(problems, toolchain_fault)

    if args.warn_only and (toolchain_fault or problems):
        print(f"\n{YELLOW}warn-only: proceeding despite "
              f"{' and '.join(problems + (['toolchain'] if toolchain_fault else []))}{RESET}\n")
        return 0
    if toolchain_fault:
        print(f"\n{RED}blocked — toolchain{RESET}\n")
        return 2
    if problems:
        print(f"\n{RED}blocked — {' and '.join(problems)} pressure{RESET}\n")
        return 1

    print(f"\n{GREEN}clear to proceed{RESET}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
