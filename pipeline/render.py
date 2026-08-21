"""Render a timeline to a delivery MP4, one segment at a time.

Deliberately NOT one giant filtergraph. Each segment renders on its own to an
all-intra ProRes 422 LT intermediate via the hardware encoder, then the
intermediates are concatenated and encoded once. That buys three things this
machine needs: peak memory stays flat regardless of clip count, a failed render
resumes instead of restarting, and re-sequencing only re-renders the segments
whose content actually changed (cache key covers the spec and the source file).

    uv run python -m pipeline.render --timeline work/timeline.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

from pipeline import config, effects, media

BOLD, RED, YELLOW, GREEN, DIM, RESET = "\033[1m", "\033[31m", "\033[33m", "\033[32m", "\033[2m", "\033[0m"

# style name -> ffmpeg xfade transition. Named for what the picture does, not
# for the filter, because sequence.py picks these from measured camera motion:
# a "whip_left" fires only when the footage itself travels left. Every target
# below was confirmed present in this ffmpeg build's xfade.
TRANSITIONS = {
    "fade": "fade",
    "dissolve": "dissolve",
    "flash": "fadewhite",
    "fade_black": "fadeblack",
    "whip_left": "smoothleft",
    "whip_right": "smoothright",
    "whip_up": "smoothup",
    "whip_down": "smoothdown",
    "blur": "hblur",
    "wipe_up": "wipeup",
    "zoom_in": "zoomin",
    "squeeze": "squeezeh",
    "circle_open": "circleopen",
    "pixelize": "pixelize",
}


# ---------------------------------------------------------------- cache key


def _source_fingerprint(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise media.MediaError(f"source missing: {path}")
    stat = p.stat()
    return f"{p.resolve()}:{stat.st_size}:{int(stat.st_mtime)}"


def segment_key(seg: dict) -> str:
    """Stable hash over the segment spec plus the identity of every source it reads."""
    payload = json.dumps(seg, sort_keys=True)
    for clip in _clips_of(seg):
        payload += "|" + _source_fingerprint(clip["source"])
    payload += f"|{config.OUT_W}x{config.OUT_H}@{config.OUT_FPS}"
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


def _clips_of(seg: dict) -> list[dict]:
    """Clips a segment reads from.

    Tolerant of malformed input on purpose: validate_timeline calls this to
    report bad segments, so raising on an unknown kind or a missing side would
    crash the very check meant to describe the problem.
    """
    if seg.get("kind") == "shot":
        return [seg]
    return [seg[side] for side in ("a", "b") if isinstance(seg.get(side), dict)]


# ---------------------------------------------------------------- filtergraph


def clip_chain(clip: dict, label_in: str, label_out: str) -> str:
    """Filter chain taking one decoded source to a delivery-sized, delivery-rate stream.

    Crop happens first and at source resolution: the window is carved out of the
    full frame (2625px of pan latitude on a 4K source) and only then scaled down,
    so reframing costs no sharpness.

    **The window's shape follows the reel, not the delivery frame.** An ordinary
    reel carves 9:16, and a 16:9 source then has no vertical latitude at all, so
    subject framing is a horizontal decision. A *letterboxed* reel carves the
    strip's ratio instead and pads the result out to the frame. That is the
    difference between using a landscape shot and discarding it: on a 3840x2160
    source bound for a 16:9 strip, carving 9:16 first and drawing bars afterwards
    leaves 10% of the frame visible -- a 3.2x centre punch-in -- where carving
    16:9 leaves all of it. With mixed-orientation footage the two paths are not
    a preference; one of them is simply wrong.

    Everything past the crop is assembled by effects.py, which owns stage
    ordering: geometry before the downscale, retiming and look after it.
    """
    info = media.probe(clip["source"])
    stream = media.video_stream(info)
    width, height = media.display_dimensions(stream)

    strip_h = 0
    for spec in clip.get("effects") or []:
        if spec.get("type") == "letterbox":
            strip_h = config.strip_height(float(spec.get("ratio", 16 / 9)))
    aspect = (config.OUT_W / strip_h) if strip_h else None
    crop_w, crop_h = config.crop_window(width, height, aspect)

    ctx = effects.Context(
        source_w=width,
        source_h=height,
        crop_w=crop_w,
        crop_h=crop_h,
        strip_h=strip_h,
        duration=clip_duration(clip),
        source_span=float(clip["out"]) - float(clip["in"]),
        fps=config.OUT_FPS,
        subject_y=clip.get("subject_y"),
    )

    # The clip's own `speed` predates the effects engine and still works: it is a
    # constant retime, expressed once here rather than as an effect.
    speed = float(clip.get("speed", 1.0))
    if abs(speed - 1.0) > 1e-6:
        ctx.needs_pts_reset = True
        ctx.slowest_speed = min(ctx.slowest_speed, speed)
        clip = {**clip, "effects": [*(clip.get("effects") or []),
                                    {"type": "speed_up", "factor": speed}]}

    return effects.build_chain(clip, ctx, label_in, label_out)


def clip_duration(clip: dict) -> float:
    """Output duration of a clip after retiming, from every source of it.

    Both the `speed` field and any duration-changing effect count, because
    segment_frames() pins the encoder to this number: claim more than the
    filtergraph can produce and the segment renders short, silently, taking every
    later cut off the beat with it.
    """
    raw = float(clip["out"]) - float(clip["in"])
    return raw / float(clip.get("speed", 1.0)) * effects.duration_scale(clip)


def transition_duration(seg: dict) -> float:
    """Cross-fade length, clamped to what the two clips can actually sustain.

    xfade cannot fade for longer than its shorter input: asked to, it emits a
    segment of an entirely different length instead of failing, so an over-long
    duration silently desynchronises the render from the timeline. Clamping here
    -- and reading the clamped value from qa.py too -- keeps the two in agreement.
    """
    return min(
        float(seg.get("duration", 0.4)),
        clip_duration(seg["a"]),
        clip_duration(seg["b"]),
    )


def segment_duration(seg: dict) -> float:
    """Output duration of any segment. The one definition of timeline length."""
    if seg["kind"] == "shot":
        return clip_duration(seg)
    return clip_duration(seg["a"]) + clip_duration(seg["b"]) - transition_duration(seg)


def segment_frames(seg: dict) -> int:
    """How many frames this segment must contain.

    Pinned explicitly rather than left to `-t`, because a duration that is an
    exact frame multiple makes ffmpeg emit one frame too many -- the frame whose
    timestamp equals the cut point is inside the window by a rounding hair.
    Measured: two of ten segments ran a frame long, and those extra frames push
    every later cut off the beat.
    """
    return max(int(round(segment_duration(seg) * config.OUT_FPS)), 1)


def _input_args(clip: dict) -> list[str]:
    """Seek and duration as INPUT options.

    -t must limit the source read, not the output write. On the output side a
    speed-ramped shot would be trimmed after retiming and pull the wrong amount
    of source; on a transition input, omitting it entirely makes ffmpeg decode to
    end-of-file and the xfade runs long.
    """
    # Six decimals, not three: a frame is 33.3ms and millisecond rounding lands
    # close enough to a frame boundary to change the frame count of a segment.
    return [
        "-hwaccel", "videotoolbox",
        "-ss", f"{float(clip['in']):.6f}",
        "-t", f"{float(clip['out']) - float(clip['in']):.6f}",
        "-i", clip["source"],
    ]


def build_shot_command(seg: dict, dest: Path) -> list[str]:
    clip = seg
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        *_input_args(clip),
        "-an",
        "-filter_complex", clip_chain(clip, "0:v", "out"),
        "-map", "[out]",
        "-frames:v", str(segment_frames(seg)),
        *config.INTERMEDIATE_CODEC,
        *config.COLOR_TAGS,
        str(dest),
    ]
    return cmd


def build_transition_command(seg: dict, dest: Path) -> list[str]:
    a, b = seg["a"], seg["b"]
    style_name = seg.get("style", "fade")
    style = TRANSITIONS.get(style_name, "fade")
    dur = transition_duration(seg)
    offset = max(clip_duration(a) - dur, 0.0)

    # A style is a recipe, not just an xfade name: the accents that make a whip
    # look like a whip are ordinary effects applied to each side's own timeline.
    recipe = effects.transition_recipe(
        style_name, dur, clip_duration(a), clip_duration(b), a.get("exit_flow"))
    a = effects.merge_effects(a, recipe.a_effects)
    b = effects.merge_effects(b, recipe.b_effects)

    graph = ";".join([
        clip_chain(a, "0:v", "va"),
        clip_chain(b, "1:v", "vb"),
        f"[va][vb]xfade=transition={style}:duration={dur}:offset={offset:.3f}[out]",
    ])

    return [
        "ffmpeg", "-y", "-v", "error",
        *_input_args(a),
        *_input_args(b),
        "-an",
        "-filter_complex", graph,
        "-map", "[out]",
        "-frames:v", str(segment_frames(seg)),
        *config.INTERMEDIATE_CODEC,
        *config.COLOR_TAGS,
        str(dest),
    ]


# ---------------------------------------------------------------- validation


def validate_timeline(timeline: dict, schema_path: Path) -> list[str]:
    """Catch a malformed timeline before ffmpeg does.

    Without this a typo surfaces as a KeyError inside a filtergraph builder, or
    worse as a filter that ffmpeg accepts and renders wrongly. Structural checks
    come from the schema; the semantic ones below are things JSON Schema cannot
    express -- a clip whose out precedes its in, or a missing source file.
    """
    return _schema_errors(timeline, schema_path) + _semantic_errors(timeline)


def _schema_errors(timeline: dict, schema_path: Path) -> list[str]:
    if not schema_path.exists():
        return []
    try:
        import jsonschema
    except ImportError:
        return []
    schema = json.loads(schema_path.read_text())
    return [
        f"{'/'.join(str(p) for p in e.path) or '(root)'}: {e.message}"
        for e in jsonschema.Draft202012Validator(schema).iter_errors(timeline)
    ]


def _semantic_errors(timeline: dict) -> list[str]:
    problems: list[str] = []
    for index, seg in enumerate(timeline.get("segments", [])):
        for clip in _clips_of(seg):
            if float(clip["out"]) <= float(clip["in"]):
                problems.append(
                    f"segment {index}: out ({clip['out']}) must exceed in ({clip['in']})")
            if not Path(clip["source"]).exists():
                problems.append(f"segment {index}: source not found: {clip['source']}")
    return problems


def timeline_warnings(timeline: dict) -> list[str]:
    """Non-fatal corrections the render applies on the caller's behalf."""
    warnings: list[str] = []
    for index, seg in enumerate(timeline.get("segments", [])):
        if seg["kind"] != "transition":
            continue
        asked = float(seg.get("duration", 0.4))
        actual = transition_duration(seg)
        if actual < asked - 1e-6:
            warnings.append(
                f"segment {index}: {asked:.2f}s transition exceeds its shorter clip — "
                f"clamped to {actual:.2f}s"
            )
    return warnings


# ---------------------------------------------------------------- stages


def render_segments(timeline: dict, segments_dir: Path, force: bool = False) -> list[Path]:
    segments_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    reused = 0

    for index, seg in enumerate(timeline["segments"]):
        key = segment_key(seg)
        dest = segments_dir / f"{index:03d}_{seg['kind']}_{key}.mov"

        if dest.exists() and not force:
            paths.append(dest)
            reused += 1
            print(f"  {DIM}cached{RESET}  {dest.name}")
            continue

        # A stale render of this slot under a different key is now garbage.
        for old in segments_dir.glob(f"{index:03d}_*.mov"):
            old.unlink()

        builder = build_shot_command if seg["kind"] == "shot" else build_transition_command
        start = time.perf_counter()
        media.run(builder(seg, dest), desc=f"segment {index}")
        print(f"  {GREEN}render{RESET}  {dest.name}  {DIM}{time.perf_counter() - start:.1f}s{RESET}")
        paths.append(dest)

    # A timeline that lost segments leaves higher-index renders behind. They are
    # excluded from the concat list either way, but on a 4K workflow with limited
    # disk they are worth reclaiming rather than accumulating across edits.
    orphans = [
        old for old in segments_dir.glob("[0-9][0-9][0-9]_*.mov")
        if int(old.name[:3]) >= len(timeline["segments"])
    ]
    for old in orphans:
        old.unlink()
    if orphans:
        print(f"  {DIM}removed {len(orphans)} orphaned segment(s) from a longer edit{RESET}")

    if reused:
        print(f"  {DIM}{reused}/{len(paths)} segments reused from cache{RESET}")
    return paths


def concat_segments(paths: list[Path], dest: Path) -> Path:
    """Join the intermediates. All-intra and parameter-identical, so this is a stream copy."""
    listing = dest.parent / "concat.txt"
    listing.write_text("".join(f"file '{p.resolve()}'\n" for p in paths))
    media.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0",
         "-i", str(listing), "-c", "copy", str(dest)],
        desc="concat",
    )
    return dest


def final_encode(video: Path, dest: Path, encoder: str, audio: Path | None) -> Path:
    settings = config.FINAL_ENCODERS.get(encoder)
    if settings is None:
        raise SystemExit(f"unknown encoder '{encoder}'; choose from {list(config.FINAL_ENCODERS)}")

    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(video)]
    if audio:
        cmd += ["-i", str(audio), "-map", "0:v:0", "-map", "1:a:0", "-shortest",
                "-af", f"loudnorm=I={config.TARGET_LUFS}:TP={config.TARGET_TRUE_PEAK}:LRA={config.TARGET_LRA}",
                "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2"]
    else:
        cmd += ["-an"]

    cmd += [
        *settings,
        *config.COLOR_TAGS,
        "-pix_fmt", "yuv420p",
        "-r", str(config.OUT_FPS),
        "-movflags", "+faststart",
        str(dest),
    ]
    media.run(cmd, desc=f"final encode ({encoder})")
    return dest


# ---------------------------------------------------------------- entry


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Render a timeline to a delivery MP4.")
    ap.add_argument("--timeline", type=Path, default=config.TIMELINE_JSON)
    ap.add_argument("--out", type=Path, default=config.out_path("reel"),
                    help="the extension chooses the container (.mov or .mp4)")
    ap.add_argument("--encoder", default=config.FINAL_ENCODER, choices=list(config.FINAL_ENCODERS))
    ap.add_argument("--draft", action="store_true", help="shorthand for --encoder vt_h264")
    ap.add_argument("--intermediate", action="store_true",
                    help="write ProRes instead of a delivery codec, because "
                         "something still has to be composited onto this. Saves "
                         "a whole lossy generation when lyrics are burned in.")
    ap.add_argument("--mute", action="store_true",
                    help="write a silent upload file plus a _preview with the reference "
                         "track baked in (you add the real sound in the app)")
    ap.add_argument("--force", action="store_true", help="ignore the segment cache")
    ap.add_argument("--keep-intermediates", action="store_true")
    ap.add_argument("--skip-preflight", action="store_true")
    args = ap.parse_args(argv)

    # Measured, not guessed: segment rendering peaks well under a gigabyte because
    # only one segment is in flight at a time. The ML stages declare far more.
    if not args.skip_preflight:
        from pipeline import preflight
        code = preflight.main(["--stage", "render", "--need-ram", "1.0", "--need-disk", "10"])
        if code != 0:
            return code

    if not args.timeline.exists():
        print(f"{RED}no timeline at {args.timeline}{RESET}", file=sys.stderr)
        return 1

    timeline = json.loads(args.timeline.read_text())

    errors = validate_timeline(timeline, config.SCHEMAS_DIR / "timeline.schema.json")
    if errors:
        print(f"\n{RED}timeline is invalid:{RESET}")
        for problem in errors[:12]:
            print(f"  {problem}")
        return 1
    for warning in timeline_warnings(timeline):
        print(f"{YELLOW}warn{RESET}  {warning}")

    encoder = "prores" if args.intermediate else (
        "vt_h264" if args.draft else args.encoder)

    audio = timeline.get("audio", {}).get("music")
    audio_path = Path(audio) if audio else None
    if audio_path and not audio_path.exists():
        print(f"{YELLOW}music track missing ({audio_path}); rendering silent{RESET}")
        audio_path = None

    print(f"\n{BOLD}render{RESET}  {DIM}{len(timeline['segments'])} segments · {encoder}{RESET}\n")
    started = time.perf_counter()

    paths = render_segments(timeline, config.SEGMENTS_DIR, force=args.force)

    joined = config.SEGMENTS_DIR / "_joined.mov"
    concat_segments(paths, joined)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    if args.mute:
        # The upload file carries no audio, because the sound is added in the
        # app. The preview exists only so the sync can be checked first -- a
        # cut that drifts off the beat is obvious to watch and invisible to read.
        final_encode(joined, args.out, encoder, None)
        outputs.append(args.out)
        if audio_path:
            preview = args.out.with_name(f"{args.out.stem}_preview{args.out.suffix}")
            final_encode(joined, preview, encoder, audio_path)
            outputs.append(preview)
    else:
        final_encode(joined, args.out, encoder, audio_path)
        outputs.append(args.out)

    if not args.keep_intermediates:
        joined.unlink(missing_ok=True)

    print()
    for path in outputs:
        size_mb = path.stat().st_size / (1024 * 1024)
        role = "preview — check the sync, do not upload" if path is not outputs[0] and args.mute \
            else ("silent — upload this, add the sound in the app" if args.mute else "")
        print(f"{GREEN}rendered{RESET}  {path}  {DIM}{size_mb:.1f} MB"
              + (f" · {role}" if role else "") + RESET)
    print(f"{DIM}{time.perf_counter() - started:.1f}s total{RESET}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
