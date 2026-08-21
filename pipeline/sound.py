"""Synthesise the whooshes and impacts that sell a cut, and mix them.

Half of why a transition reads as professional is audible, not visible. A whip
with no whoosh looks like a glitch; the same whip with one reads as intent.

Everything is synthesised from ffmpeg's own generators rather than sampled, so
there is no pack to license and nothing to ship. If `assets/sfx/<cue>.wav` exists
it is used instead -- real samples sound better, and this way they are an upgrade
rather than a dependency.

Three outputs, because the sound you are targeting comes from a screen recording
of Instagram and baking it in would be the wrong deliverable:

    reel.mov        silent            safest: add the trending sound in the app
    reel_sfx.mov    sound design only POST THIS. Instagram keeps the original
                    audio underneath the sound you add, so the whooshes survive
                    *and* the reel still registers as using the trending audio,
                    which is most of why you chose it.
    reel_mixed.mov  sfx + reference   a preview for checking sync, or the one to
                    post if the music is actually yours to use.

    uv run python -m pipeline.sound --timeline work/timeline_v1.json \
        --video out/reel.mov --music assets/song.wav
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pipeline import config, media, render

BOLD, RED, YELLOW, GREEN, DIM, RESET = "\033[1m", "\033[31m", "\033[33m", "\033[32m", "\033[2m", "\033[0m"

SR = 48000

# Levels, relative. Impacts sit loudest because they land on cuts the eye is
# already committed to; a whoosh that competes with the music reads as a mistake.
GAINS = {"impact": 0.85, "whoosh": 0.55, "riser": 0.40, "sub_drop": 0.70,
         "tick": 0.30}

# Cue lengths in seconds. A riser is the only long one, and it is placed to end
# on the drop rather than start there.
LENGTHS = {"impact": 0.30, "whoosh": 0.38, "riser": 1.60, "sub_drop": 0.55,
           "tick": 0.06}

# A whoosh peaks slightly before the cut, the way a real one does -- the sound
# arrives with the movement, not after it.
WHOOSH_LEAD = 0.14

# Cues with a hard attack, whose audible onset is their placement time. The rest
# swell: a whoosh fades in over about 95ms and a riser is near-silent for most of
# its length, so their onsets arrive well after `at` *by design*. Anything
# measuring cue timing has to know which is which, or it grades a riser as a
# late impact -- which is how the first version of that check read 496ms.
TRANSIENT_CUES = {"impact", "sub_drop", "tick"}
SWELL_CUES = {"whoosh", "riser"}


def _generator(cue: str) -> str:
    """A filter chain producing one cue, as an audio source.

    `aevalsrc` takes an expression of t, so a sweep is a formula. The frequency
    terms are inside a cumulative phase (f*t where f itself varies with t), which
    is why the swept cues use t squared rather than a literal frequency ramp.
    """
    length = LENGTHS[cue]

    if cue == "impact":
        # Sub thump plus a transient click, both decaying fast. Without the click
        # it is a hum; without the sub it is a tick.
        return (f"aevalsrc='exp(-22*t)*sin(2*PI*70*t)"
                f"+0.35*exp(-160*t)*sin(2*PI*1400*t)'"
                f":s={SR}:d={length}")

    if cue == "sub_drop":
        # Descending: 90 Hz down to about 35 Hz across the cue.
        return (f"aevalsrc='exp(-5*t)*sin(2*PI*(90*t-30*t*t))'"
                f":s={SR}:d={length}")

    if cue == "riser":
        # Ascending, and quiet until the end so it builds rather than announces.
        return (f"aevalsrc='(t/{length})*(t/{length})"
                f"*sin(2*PI*(300*t+400*t*t))':s={SR}:d={length}")

    if cue == "tick":
        return (f"anoisesrc=c=white:r={SR}:d={length},"
                f"highpass=f=3000,afade=t=out:st=0:d={length}")

    # whoosh: filtered noise with a fast in and a longer tail, band-limited so it
    # occupies the space above the music's body and below its air.
    return (f"anoisesrc=c=pink:r={SR}:d={length},"
            f"bandpass=f=1100:width_type=o:w=2.2,"
            f"afade=t=in:st=0:d={length * 0.25:.3f},"
            f"afade=t=out:st={length * 0.30:.3f}:d={length * 0.70:.3f}")


def sample_for(cue: str) -> Path | None:
    """A real sample for this cue, if one was dropped in assets/sfx."""
    candidate = config.ASSETS_DIR / "sfx" / f"{cue}.wav"
    return candidate if candidate.exists() else None


# ---------------------------------------------------------------- placement


def boundaries(timeline: dict) -> list[float]:
    """Cumulative cut times, from the same durations the renderer uses."""
    times, position = [], 0.0
    for segment in timeline["segments"]:
        times.append(round(position, 4))
        position += render.segment_duration(segment)
    return times


def plan_cues(timeline: dict, brief: dict | None = None) -> list[dict]:
    """What to put where, driven by the edit rather than sprinkled.

    A cue on every cut is the audio equivalent of a transition on every
    boundary -- the thing the visual budget exists to prevent. So: transitions
    get a whoosh because they are a movement, the payoff gets an impact and a sub
    because it is the arrival, and short stabs get a tick. Ordinary cuts get
    nothing, which is what makes the ones that do land.
    """
    cues: list[dict] = []
    starts = boundaries(timeline)
    segments = timeline["segments"]

    # Where the joins are. Reading `kind == "transition"` finds only the blends,
    # and v3 realises most of its vocabulary as a hard cut with accents -- so
    # that test found 2 of the 6 joins in a real reel and the audio landed on the
    # wrong two boundaries. The timeline records them explicitly instead.
    whooshed: set[int] = set()
    for join in timeline.get("joins") or []:
        index = int(join["segment"])
        if 0 <= index < len(starts):
            whooshed.add(index)
            at = starts[index]
            if join.get("blend"):
                # A blend's boundary is the middle of its own segment.
                at += render.segment_duration(segments[index]) / 2
            cues.append({"at": max(at - WHOOSH_LEAD, 0.0), "cue": "whoosh"})

    # Which segment holds the payoff, so the arrival gets the arrival sound.
    payoff_segment = None
    if brief:
        kept = (timeline.get("story") or {}).get("shots") or []
        payoff_ids = {s["id"] for s in brief["shots"] if s["role"] == "payoff"}
        shot_number = 0
        for index, segment in enumerate(segments):
            step = 2 if segment["kind"] == "transition" else 1
            if any(kept[n] in payoff_ids
                   for n in range(shot_number, min(shot_number + step, len(kept)))):
                payoff_segment = index
                break
            shot_number += step

    for index, segment in enumerate(segments):
        at = starts[index]
        duration = render.segment_duration(segment)

        if index == payoff_segment:
            cues.append({"at": at, "cue": "impact"})
            cues.append({"at": at, "cue": "sub_drop"})
            if at >= LENGTHS["riser"]:
                cues.append({"at": at - LENGTHS["riser"], "cue": "riser"})
            continue

        if index == 0 or index in whooshed:
            continue

        # Ordinary cuts get nothing. A cue on every cut is the audio equivalent
        # of a transition on every boundary -- it is what makes none of them land.
        if duration <= 0.9:
            cues.append({"at": at, "cue": "tick"})
        elif index % 2 == 0:
            cues.append({"at": at, "cue": "impact", "gain": 0.55})

    return sorted(cues, key=lambda c: c["at"])


# ---------------------------------------------------------------- mixing


def build_sfx_command(cues: list[dict], duration: float, dest: Path) -> list[str]:
    """Render the cue bed alone, as a WAV the length of the reel."""
    if not cues:
        return ["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
                "-i", f"anullsrc=r={SR}:cl=mono", "-t", f"{duration:.3f}", str(dest)]

    inputs: list[str] = []
    chains: list[str] = []
    labels: list[str] = []

    # Counted explicitly rather than derived from len(inputs): a sample costs two
    # argv entries and a generator costs four, so the stream index and the
    # argument count are unrelated.
    stream = 0
    for index, cue in enumerate(cues):
        name = cue["cue"]
        gain = cue.get("gain", GAINS[name])
        delay = int(round(cue["at"] * 1000))

        if sample := sample_for(name):
            inputs += ["-i", str(sample)]
        else:
            inputs += ["-f", "lavfi", "-i", _generator(name)]

        chains.append(f"[{stream}:a]aformat=sample_fmts=fltp:sample_rates={SR}:"
                      f"channel_layouts=mono,volume={gain:.3f},"
                      f"adelay=delays={delay}:all=1[c{index}]")
        labels.append(f"[c{index}]")
        stream += 1

    graph = ";".join(chains)
    # `apad` before the output `-t`, and it is load-bearing. amix ends when its
    # last input does, so a bed whose final cue lands at 18.2s is 18.2s long --
    # `-t` can only truncate, never pad. build_mix_command then muxes with
    # -shortest, which trims the *video* to the audio: a 23.27s reel shipped at
    # 18.17s with its last shot and last lyric line cut off, and qa.py caught it
    # only because it compares against the timeline.
    graph += (f";{''.join(labels)}amix=inputs={len(labels)}:normalize=0:"
              f"dropout_transition=0,alimiter=limit=0.9,apad[sfx]")

    return ["ffmpeg", "-y", "-v", "error", *inputs,
            "-filter_complex", graph, "-map", "[sfx]",
            "-t", f"{duration:.3f}", "-ar", str(SR), "-ac", "1", str(dest)]


def build_mix_command(video: Path, sfx: Path, music: Path | None,
                      start: float, duration: float, dest: Path,
                      sfx_only: bool) -> list[str]:
    """Mux the audio onto the video, loudness-normalised.

    When music is present it is ducked under the cues with sidechaincompress, so
    an impact cuts through instead of fighting the mix -- the same trick every
    trailer uses, and the reason a whoosh can be quiet and still read.
    """
    inputs = ["-i", str(video), "-i", str(sfx)]
    if music and not sfx_only:
        inputs += ["-ss", f"{start:.3f}", "-i", str(music)]

    if music and not sfx_only:
        graph = (
            f"[2:a]aformat=sample_fmts=fltp:sample_rates={SR}:channel_layouts=stereo,"
            f"atrim=0:{duration:.3f},asetpts=PTS-STARTPTS,volume=0.9[bed];"
            f"[1:a]aformat=sample_fmts=fltp:sample_rates={SR}:channel_layouts=stereo[cues];"
            f"[cues]asplit=2[cue_mix][cue_key];"
            f"[bed][cue_key]sidechaincompress=threshold=0.05:ratio=6:attack=5:"
            f"release=220[ducked];"
            f"[ducked][cue_mix]amix=inputs=2:normalize=0:dropout_transition=0,"
            f"loudnorm=I={config.TARGET_LUFS}:TP={config.TARGET_TRUE_PEAK}:"
            f"LRA={config.TARGET_LRA}[a]"
        )
    else:
        # Peak-limited, deliberately *not* normalised to the delivery loudness.
        # A cue bed is mostly silence, so an integrated target of -14 LUFS is met
        # by making the few impacts deafening -- measured, the sparse bed read
        # -21.6 LUFS and pushing it to -14 would have been wrong, not better.
        # This file is destined to sit *under* the sound added in the app, where
        # relative level is what matters and absolute loudness is not.
        graph = (
            f"[1:a]aformat=sample_fmts=fltp:sample_rates={SR}:channel_layouts=stereo,"
            f"volume=3.0,alimiter=limit={10 ** (config.TARGET_TRUE_PEAK / 20):.4f}[a]"
        )

    return ["ffmpeg", "-y", "-v", "error", *inputs,
            "-filter_complex", graph,
            "-map", "0:v", "-map", "[a]",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart", "-shortest", str(dest)]


def measure_loudness(path: Path) -> float | None:
    """Integrated loudness of a file, via loudnorm's analysis pass."""
    proc = media.run(
        ["ffmpeg", "-hide_banner", "-i", str(path), "-af",
         f"loudnorm=I={config.TARGET_LUFS}:TP={config.TARGET_TRUE_PEAK}:print_format=json",
         "-f", "null", "-"],
        desc="loudness", quiet=True)
    text = proc.stderr
    start = text.rfind("{")
    if start < 0:
        return None
    try:
        return float(json.loads(text[start:text.rfind("}") + 1])["input_i"])
    except (ValueError, KeyError):
        return None


# ---------------------------------------------------------------- entry


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Synthesise and mix sound design.")
    ap.add_argument("--timeline", type=Path, default=config.TIMELINE_JSON)
    ap.add_argument("--video", type=Path, required=True, help="the silent render")
    ap.add_argument("--music", type=Path, default=None, help="reference track")
    ap.add_argument("--brief", type=Path, default=config.BRIEF_JSON)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args(argv)

    for path in (args.timeline, args.video):
        if not path.exists():
            print(f"{RED}no {path}{RESET}", file=sys.stderr)
            return 1

    timeline = json.loads(args.timeline.read_text())
    brief = json.loads(args.brief.read_text()) if args.brief.exists() else None
    duration = sum(render.segment_duration(s) for s in timeline["segments"])

    cues = plan_cues(timeline, brief)
    counts: dict[str, int] = {}
    for cue in cues:
        counts[cue["cue"]] = counts.get(cue["cue"], 0) + 1

    print(f"\n{BOLD}sound{RESET}  {DIM}{len(cues)} cues over {duration:.2f}s · "
          f"{', '.join(f'{n}x{c}' for n, c in sorted(counts.items()))}{RESET}\n")

    directory = args.out_dir or args.video.parent
    directory.mkdir(parents=True, exist_ok=True)
    stem = args.video.stem
    sfx_wav = config.WORK_DIR / f"{stem}_sfx.wav"
    sfx_wav.parent.mkdir(parents=True, exist_ok=True)

    try:
        media.run(build_sfx_command(cues, duration, sfx_wav), desc="synthesise cues")
    except media.MediaError as exc:
        print(f"{RED}{exc}{RESET}", file=sys.stderr)
        return 1

    start = 0.0
    if brief:
        start = brief.get("music", {}).get("best_start", 0.0)
    elif timeline.get("sync"):
        start = timeline["sync"].get("best_start") or 0.0

    outputs: list[tuple[Path, bool]] = [(config.out_path(f"{stem}_sfx", directory), True)]
    if args.music and args.music.exists():
        outputs.append((config.out_path(f"{stem}_mixed", directory), False))

    for dest, sfx_only in outputs:
        try:
            media.run(build_mix_command(args.video, sfx_wav, args.music, start,
                                        duration, dest, sfx_only),
                      desc=f"mux {dest.name}")
        except media.MediaError as exc:
            print(f"  {RED}fail  {RESET}  {dest.name}  {DIM}{exc}{RESET}")
            continue

        lufs = measure_loudness(dest)
        detail = f"{lufs:.1f} LUFS" if lufs is not None else "loudness unmeasured"
        if sfx_only:
            # Judged on peak, not on integrated loudness — see build_mix_command.
            ok, label = True, "sound design only, peak-limited"
        else:
            ok = lufs is not None and abs(lufs - config.TARGET_LUFS) <= 1.5
            label = f"sfx + reference music, target {config.TARGET_LUFS:.0f}"
        mark = GREEN if ok else YELLOW
        print(f"  {mark}{'ok  ' if ok else 'warn'}{RESET}  {dest.name:<28}"
              f"{DIM}{detail} · {label}{RESET}")

    print(f"\n{GREEN}post{RESET}  {DIM}{config.out_path(f'{stem}_sfx', directory).name} "
          f"— Instagram keeps this audio under the sound you add, so the whooshes "
          f"survive and the reel still counts as using the trending audio{RESET}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
