"""Run the whole chain: footage in, three drafts out.

Every stage is a separate subprocess that exits before the next one starts. That
is not tidiness -- it is the reason this works on 8 GB. A long-lived process
holding OpenCV, librosa and an HTTP client at once is the difference between
a pipeline that runs and one that swaps.

    uv run python -m pipeline.auto --music assets/track.wav

Stages are skipped when their output already exists, so re-running after
dropping in one new clip is cheap. Use --force to redo everything.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from pipeline import config

BOLD, RED, YELLOW, GREEN, DIM, RESET = "\033[1m", "\033[31m", "\033[33m", "\033[32m", "\033[2m", "\033[0m"


def run_stage(name: str, args: list[str], skip_if: Path | None, force: bool) -> float:
    """Run one stage in its own process. Returns elapsed seconds; raises on failure."""
    if skip_if and skip_if.exists() and not force:
        print(f"  {DIM}skip    {name:<12} {skip_if.name} already exists{RESET}")
        return 0.0

    print(f"  {BOLD}{name:<12}{RESET}{DIM}python -m pipeline.{name} "
          f"{' '.join(args)}{RESET}")
    started = time.perf_counter()
    proc = subprocess.run([sys.executable, "-m", f"pipeline.{name}", *args],
                          capture_output=True, text=True)
    elapsed = time.perf_counter() - started

    if proc.returncode != 0:
        print(f"\n{RED}{name} failed (exit {proc.returncode}){RESET}\n")
        sys.stdout.write(proc.stdout[-3000:])
        sys.stderr.write(proc.stderr[-2000:])
        raise SystemExit(proc.returncode)

    # Each stage already prints its own summary; surface just the last line.
    tail = [line for line in proc.stdout.splitlines() if line.strip()]
    if tail:
        print(f"            {DIM}{tail[-1].strip()[:110]}{RESET}")
    return elapsed


def stale_inputs(inputs: Path) -> bool:
    """Whether the cached analysis describes a different folder of footage.

    Every stage from ingest to merge is skipped when its output file exists,
    which is what makes re-running after dropping in one new clip cheap. It is
    also how a build pointed at a new shoot silently reused the last one: the
    outputs existed, so nine new clips were never looked at and the previous
    project's reel came out the other end reporting success.
    """
    if not config.SOURCES_JSON.exists():
        return False
    try:
        recorded = json.loads(config.SOURCES_JSON.read_text()).get("inputs")
    except (json.JSONDecodeError, OSError):
        return True
    return recorded != str(inputs.resolve())


def _brief_field(brief: Path | None, key: str):
    if not brief or not brief.exists():
        return None
    try:
        return json.loads(brief.read_text()).get(key)
    except (json.JSONDecodeError, OSError):
        return None


def _duration_of(path: Path) -> float:
    from pipeline import media
    return media.duration_seconds(media.probe(path))


def _latest_matte() -> Path | None:
    mattes = sorted(config.WORK_DIR.glob("matte_*.mkv"),
                    key=lambda p: p.stat().st_mtime)
    return mattes[-1] if mattes else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Footage in, three reel drafts out.")
    ap.add_argument("--inputs", type=Path, default=config.INPUTS_DIR)
    ap.add_argument("--music", type=Path, default=None,
                    help="reference track the cuts are locked to")
    ap.add_argument("--bpm", type=float, default=None, help="force a tempo")
    ap.add_argument("--target", type=float, default=config.TARGET_REEL_SECONDS)
    ap.add_argument("--variants", type=int, default=3)
    ap.add_argument("--force", action="store_true", help="re-run every stage")
    ap.add_argument("--no-render", action="store_true", help="stop after sequencing")
    ap.add_argument("--allow-partial", action="store_true",
                    help="continue past rejected source files")
    ap.add_argument("--skip-preflight", action="store_true",
                    help="run anyway on a loaded machine — expect swapping")
    ap.add_argument("--brief", type=Path, default=None,
                    help="build the story in this brief (from pipeline.brief)")
    ap.add_argument("--allow-gaps", action="store_true",
                    help="with --brief: build even if a required shot is missing")
    ap.add_argument("--sound", action="store_true",
                    help="also render the sound-design and mixed audio versions")
    ap.add_argument("--lyrics", default=None,
                    help="the lyric line to burn in; prefix a word with * to accent it")
    ap.add_argument("--lyrics-file", dest="lyrics_file", type=Path, default=None,
                    help="a lyric file for --lyrics: .lrc with timestamps places "
                         "every line exactly; plain text keeps its line breaks "
                         "as phrases")
    ap.add_argument("--style", default=None,
                    help="typographic identity for --lyrics (chrome, editorial, "
                         "marker, stencil); defaults to the brief's, then chrome")
    ap.add_argument("--no-matte", action="store_true",
                    help="draw lyric words on top instead of behind the subject "
                         "(skips ~4s of segmentation per second of reel)")
    args = ap.parse_args(argv)

    print(f"\n{BOLD}auto{RESET}  {DIM}{args.inputs}{RESET}\n")
    started = time.perf_counter()

    if stale_inputs(args.inputs) and not args.force:
        print(f"  {YELLOW}new footage{RESET}  work/ holds an analysis of a different "
              f"folder — re-analysing from scratch\n")
        args.force = True

    ingest_args = ["--inputs", str(args.inputs)]
    if args.allow_partial:
        ingest_args.append("--allow-partial")

    stages: list[tuple[str, list[str], Path | None]] = []
    if not args.skip_preflight:
        stages.append(
            ("preflight", ["--stage", "analysis", "--need-ram", "2.0", "--need-disk", "20"], None))
    stages += [
        ("ingest", ingest_args, config.SOURCES_JSON),
        ("proxy", [], config.PROXIES_DIR / "manifest.json"),
        ("scenes", [], config.SHOTS_JSON),
        ("features", [], None),
        ("moments", ["--contact-sheet"], config.MOMENTS_JSON),
        ("signals", [], config.SIGNALS_JSON),
        ("vlm_tag", [], config.TAGS_JSON),
        ("merge", [], config.CLIP_CARDS_JSON),
    ]

    # With a brief, the music map already exists -- pipeline.brief wrote it while
    # choosing the opening bar -- so re-running music here would overwrite an
    # alignment that was computed against the chosen blueprint.
    if args.music and not args.brief:
        music_args = ["--track", str(args.music), "--target", str(args.target)]
        if args.bpm:
            music_args += ["--bpm", str(args.bpm)]
        stages.append(("music", music_args, config.MUSIC_MAP_JSON))

    sequence_args = ["--target", str(args.target), "--variants", str(args.variants)]
    if args.music:
        sequence_args += ["--track", str(args.music)]

    if args.brief:
        cast_args = ["--brief", str(args.brief)]
        if args.allow_gaps:
            cast_args.append("--allow-gaps")
        stages.append(("cast", cast_args, None))
        sequence_args += ["--brief", str(args.brief)]

    stages.append(("sequence", sequence_args, None))

    timings: list[tuple[str, float]] = []
    for name, stage_args, skip_if in stages:
        try:
            timings.append((name, run_stage(name, stage_args, skip_if, args.force)))
        except SystemExit:
            # Casting exits non-zero on purpose when a required shot was never
            # filmed. That is the brief doing its job, not a crash, so it gets a
            # pointer to the reshoot list rather than a stack of ffmpeg output.
            if name != "cast":
                raise
            print(f"\n{YELLOW}a required shot has no footage{RESET}")
            print(f"{DIM}Read {config.OUT_DIR / 'coverage.md'} for what to reshoot, "
                  f"or re-run with --allow-gaps to build a shorter reel without "
                  f"it.{RESET}\n")
            return 2

    drafts: list[Path] = []
    if not args.no_render:
        print()
        for index in range(1, args.variants + 1):
            timeline = config.WORK_DIR / f"timeline_v{index}.json"
            if not timeline.exists():
                continue
            draft = config.out_path(f"reel_v{index}")
            # With lyrics to burn, the picture render is an *intermediate*: it
            # gets decoded again and composited onto. Encoding it to a delivery
            # codec here and then re-encoding after the burn cost two lossy
            # generations -- 20 Mb/s hardware, then 6.9 Mb/s x264 carrying the
            # first pass's artifacts. ProRes makes the burn the only lossy step.
            wants_text = bool(args.lyrics or args.lyrics_file)
            picture = (draft.with_name(f"{draft.stem}_picture.mov")
                       if wants_text else draft)
            render_args = ["--timeline", str(timeline), "--out", str(picture),
                           "--mute", "--skip-preflight"]
            render_args.append("--intermediate" if wants_text else "--draft")
            timings.append((f"render v{index}", run_stage(
                "render", render_args, None, True)))
            drafts.append(draft)

            if args.lyrics or args.lyrics_file:
                # Order matters here and it changed in v5. Words are timed first
                # (the matte only needs to segment the spans a word is actually
                # up), then the matte runs, and only then are the words *placed*
                # -- because the layout prefers anchors that land on the subject,
                # and it cannot prefer what it has not been told about yet.
                #
                # Timing is against the reel that was actually built, which is
                # routinely shorter than the brief when a slot went unfilled.
                lyric_args = ["--duration", f"{_duration_of(picture):.3f}"]
                if args.lyrics_file:
                    lyric_args += ["--lyrics-file", str(args.lyrics_file)]
                else:
                    lyric_args += ["--words", args.lyrics]
                if args.brief:
                    lyric_args += ["--brief", str(args.brief)]
                if args.style:
                    lyric_args += ["--style", args.style]
                timings.append((f"lyrics v{index}", run_stage(
                    "lyrics", lyric_args, None, True)))

                matte_path = None
                if not args.no_matte:
                    matte_args = ["--video", str(picture)]
                    if args.brief:
                        matte_args += ["--brief", str(args.brief)]
                        # Dodging needs the subject on every frame, not only
                        # where a word is up. Segmenting the lyric spans alone
                        # would make the subject brighten and dim as words came
                        # and went, which reads as a fault rather than a look.
                        if _brief_field(args.brief, "subject_dodge"):
                            matte_args.append("--all-frames")
                    timings.append((f"matte v{index}", run_stage(
                        "matte", matte_args, None, True)))
                    matte_path = _latest_matte()

                burn = ["--burn", str(picture), "--dest", str(draft)]
                if args.brief:
                    burn += ["--brief", str(args.brief)]
                if args.style:
                    burn += ["--style", args.style]
                if matte_path:
                    burn += ["--matte", str(matte_path)]
                timings.append((f"burn v{index}", run_stage("lyrics", burn, None, True)))

            if args.sound:
                sound_args = ["--timeline", str(timeline), "--video", str(draft)]
                if args.music:
                    sound_args += ["--music", str(args.music)]
                if args.brief:
                    sound_args += ["--brief", str(args.brief)]
                timings.append((f"sound v{index}", run_stage(
                    "sound", sound_args, None, True)))

    total = time.perf_counter() - started
    print(f"\n{BOLD}done{RESET}  {DIM}{total:.1f}s total{RESET}")
    slowest = sorted(timings, key=lambda item: -item[1])[:3]
    print(f"{DIM}slowest: " + ", ".join(f"{n} {s:.1f}s" for n, s in slowest if s > 0) + RESET)

    sheet = config.OUT_DIR / "sync.txt"
    if drafts:
        print(f"\n{GREEN}watch these on a phone, not the Mac{RESET}")
        for draft in drafts:
            preview = draft.with_name(f"{draft.stem}_preview{draft.suffix}")
            note = f"  {DIM}(+ {preview.name} to check the sync){RESET}" if preview.exists() else ""
            print(f"  {draft}{note}")
    if sheet.exists():
        print(f"\n{GREEN}then{RESET}  pick one, read {sheet} for the audio start point,")
        print(f"      and render it final:  {DIM}python -m pipeline.render "
              f"--timeline work/timeline_vN.json --mute{RESET}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
