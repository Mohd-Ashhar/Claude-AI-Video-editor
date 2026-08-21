"""Run every gate the plan defines, in one command.

    python verify.py

Each check is a claim the pipeline makes about itself. They exist because each
one has failed at least once: the duration gate caught a transition that decoded
to end-of-file, the colour gate caught tags this ffmpeg build silently drops, and
the crop gate caught an odd-width crop on exactly the 4K footage this is built
for. Re-run after any change to the render or analysis path.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TEST_INPUTS = ROOT / "inputs" / "test"
PY = str(ROOT / ".venv" / "bin" / "python")

# Redirect every artifact into a scratch directory BEFORE importing the pipeline,
# so config resolves there and child stages inherit it. Verification would
# otherwise overwrite a real project's clip cards and wipe its segment cache,
# forcing a full re-render of work the user had already paid for.
_SCRATCH = Path(tempfile.mkdtemp(prefix="reel-verify-"))
os.environ["REEL_WORK_DIR"] = str(_SCRATCH / "work")
os.environ["REEL_OUT_DIR"] = str(_SCRATCH / "out")

from pipeline import config, effects, media, render  # noqa: E402 - must follow the env setup

BOLD, RED, GREEN, DIM, YELLOW, RESET = "\033[1m", "\033[31m", "\033[32m", "\033[2m", "\033[33m", "\033[0m"

CLIP_1 = str(TEST_INPUTS / "CLIP_4K_1.mp4")
CLIP_2 = str(TEST_INPUTS / "CLIP_4K_2.mp4")
CLIP_3 = str(TEST_INPUTS / "CLIP_4K_3.mp4")
RENDER = "pipeline.render"

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    mark = f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"
    print(f"  {mark}  {name}" + (f"   {DIM}{detail}{RESET}" if detail else ""))


def stage(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([PY, "-m", *args], capture_output=True, text=True, cwd=ROOT)


# ---------------------------------------------------------------- checks


def check_toolchain() -> None:
    from pipeline.preflight import ffmpeg_capabilities

    filters, encoders = ffmpeg_capabilities()
    missing = [f for f in config.REQUIRED_FILTERS if f not in filters]
    missing += [e for e in config.REQUIRED_ENCODERS if e not in encoders]
    check("toolchain has every required filter and encoder", not missing,
          f"missing {missing}" if missing else "")

    absent = [f for f in config.OPTIONAL_FILTERS if f not in filters]
    if absent:
        print(f"        {DIM}absent (stages route around these): {', '.join(absent)}{RESET}")


def check_crop_geometry() -> None:
    """The crop must be even and inside the frame at every resolution.

    Whether it can fill 1080 without upscaling is a separate claim: that is the
    ingest gate's job, and the two must agree — a source ingest accepts must
    never be one the renderer would have to upscale.
    """
    from pipeline.ingest import max_crop_width

    bad, disagreements = [], []
    for width, height in [(3840, 2160), (2704, 1520), (1920, 1082), (1080, 1920), (4096, 2160)]:
        crop_w, crop_h = config.crop_window(width, height)
        if crop_w % 2 or crop_h % 2:
            bad.append(f"{width}x{height} odd crop {crop_w}x{crop_h}")
        if crop_h > height or crop_w > width:
            bad.append(f"{width}x{height} crop {crop_w}x{crop_h} exceeds frame")
        if max_crop_width(width, height) != crop_w:
            disagreements.append(
                f"{width}x{height}: gate says {max_crop_width(width, height)}, render crops {crop_w}")

    check("crop window is even and inside the frame at every resolution", not bad, "; ".join(bad))
    check("ingest gate and render crop use the same geometry", not disagreements,
          "; ".join(disagreements))

    upscalers = [f"{w}x{h}" for w, h in [(2704, 1520), (1920, 1082)]
                 if config.crop_window(w, h)[0] >= config.OUT_W]
    check("sources too short for a sharp 9:16 crop are below the gate", not upscalers,
          f"{', '.join(upscalers)} would pass the gate but need upscaling" if upscalers else "")


def check_ingest_gate() -> None:
    """The gate that protects the whole chain: 1080p sources must be rejected."""
    proc = stage("pipeline.ingest", "--inputs", str(TEST_INPUTS), "--out",
                 str(config.WORK_DIR / "_verify_sources.json"))
    rejected_1080p = proc.returncode != 0 and "BAD_1080p" in proc.stdout
    check("ingest rejects a 1080p source", rejected_1080p,
          "" if rejected_1080p else "a 1080p clip was accepted — every reel would be soft")

    doc = json.loads((config.WORK_DIR / "_verify_sources.json").read_text())
    accepted_4k = len(doc["accepted"]) == 3
    check("ingest accepts the 4K sources", accepted_4k, f"{len(doc['accepted'])}/3 accepted")


def check_timeline_validation() -> None:
    schema = config.SCHEMAS_DIR / "timeline.schema.json"
    clip = {"source": CLIP_1, "in": 0.0, "out": 1.0}

    cases = {
        "reversed in/out": {"segments": [{**clip, "kind": "shot", "in": 1.0, "out": 0.5}]},
        "missing source file": {"segments": [{**clip, "kind": "shot", "source": "nope.mp4"}]},
        "unknown segment kind": {"segments": [{**clip, "kind": "shots"}]},
        "empty segment list": {"segments": []},
    }
    missed = [name for name, tl in cases.items()
              if not render.validate_timeline(tl, schema)]
    check("timeline validation rejects malformed edits", not missed,
          f"accepted: {', '.join(missed)}" if missed else f"{len(cases)} cases")

    good = {"segments": [{**clip, "kind": "shot"}]}
    check("timeline validation accepts a valid edit",
          not render.validate_timeline(good, schema))


def check_transition_clamp() -> None:
    """xfade silently changes segment length if asked to fade longer than its input."""
    seg = {
        "kind": "transition", "style": "fade", "duration": 1.0,
        "a": {"source": CLIP_1, "in": 0.0, "out": 0.5},
        "b": {"source": CLIP_2, "in": 0.0, "out": 0.8},
    }
    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp) / "t.mov"
        media.run(render.build_transition_command(seg, dest), desc="transition")
        actual = media.duration_seconds(media.probe(dest))
    expected = render.segment_duration(seg)
    drift = abs(actual - expected)
    check("over-long transition clamps and matches the predicted duration", drift <= 0.1,
          f"expected {expected:.2f}s, produced {actual:.2f}s")
    check("clamping is reported to the caller", bool(render.timeline_warnings({"segments": [seg]})))


def check_render_and_spec() -> None:
    """The Phase 0 gate: a hand-written timeline renders spec-compliant."""
    timeline_path = config.WORK_DIR / "_verify_timeline.json"
    out = config.OUT_DIR / "_verify.mp4"
    clip = CLIP_1
    timeline = {
        "version": 1, "fps": 30,
        "segments": [
            {"kind": "shot", "source": clip, "in": 0.2, "out": 1.2},
            {"kind": "transition", "style": "fade", "duration": 0.3,
             "a": {"source": clip, "in": 1.2, "out": 1.8},
             "b": {"source": CLIP_2, "in": 0.3, "out": 0.9}},
            {"kind": "shot", "source": CLIP_3,
             "in": 0.5, "out": 1.5, "crop_x": "600+200*t"},
        ],
    }
    timeline_path.write_text(json.dumps(timeline, indent=2))

    segments_dir = config.SEGMENTS_DIR
    shutil.rmtree(segments_dir, ignore_errors=True)

    proc = stage(RENDER, "--timeline", str(timeline_path), "--out", str(out),
                 "--skip-preflight")
    if proc.returncode != 0:
        check("hand-written timeline renders", False, proc.stdout.strip()[-160:])
        return
    check("hand-written timeline renders", True)

    info = media.probe(out)
    stream = media.video_stream(info)
    width, height = media.display_dimensions(stream)
    tags = (stream.get("color_primaries"), stream.get("color_transfer"), stream.get("color_space"))

    check("output is 1080x1920", (width, height) == (config.OUT_W, config.OUT_H),
          f"{width}x{height}")
    check("output carries all three BT.709 colour tags", tags == ("bt709",) * 3, str(tags))
    check("output is yuv420p", stream.get("pix_fmt") == "yuv420p", str(stream.get("pix_fmt")))

    expected = sum(render.segment_duration(s) for s in timeline["segments"])
    actual = media.duration_seconds(info)
    check("rendered duration matches the timeline", abs(actual - expected) <= 0.1,
          f"expected {expected:.2f}s, got {actual:.2f}s")

    # Selective re-render: change one clip, confirm the others are reused.
    timeline["segments"][2]["out"] = 1.8
    timeline_path.write_text(json.dumps(timeline, indent=2))
    proc = stage(RENDER, "--timeline", str(timeline_path), "--out", str(out),
                 "--skip-preflight")
    reused = proc.stdout.count("cached")
    check("changing one segment re-renders only that segment", reused == 2,
          f"{reused}/2 segments reused")

    # Shrinking the timeline must not leave renders behind.
    timeline["segments"] = timeline["segments"][:1]
    timeline_path.write_text(json.dumps(timeline, indent=2))
    stage(RENDER, "--timeline", str(timeline_path), "--out", str(out),
          "--skip-preflight")
    leftover = [p for p in segments_dir.glob("[0-9][0-9][0-9]_*.mov") if int(p.name[:3]) >= 1]
    check("shrinking the timeline reclaims orphaned segments", not leftover,
          f"{len(leftover)} left behind")


def check_analysis_chain() -> None:
    """Phase 1 end to end, ending in a schema-valid artifact."""
    for args in (
        ("pipeline.ingest", "--inputs", str(TEST_INPUTS), "--allow-partial"),
        ("pipeline.proxy",),
        ("pipeline.scenes",),
        ("pipeline.features",),
        ("pipeline.moments",),
        ("pipeline.signals",),
        ("pipeline.vlm_tag",),
        ("pipeline.merge",),
    ):
        proc = stage(*args)
        if proc.returncode != 0:
            # stderr too. A stage that raises prints its traceback there and
            # nothing to stdout, so reporting stdout alone showed the tail of
            # the last successful line and said nothing about the failure.
            detail = (proc.stderr.strip() or proc.stdout.strip())[-220:]
            check(f"analysis chain: {args[0]}", False, detail)
            return
    check("analysis chain runs end to end", True)

    doc = json.loads(config.CLIP_CARDS_JSON.read_text())
    check("clip cards are schema-valid and ranked",
          len(doc["cards"]) == 3 and doc["cards"] == sorted(
              doc["cards"], key=lambda c: -c["rank_score"]),
          f"{len(doc['cards'])} cards")

    # Stability is absolute, not batch-relative. Testing the function directly
    # rather than the batch: a real batch may legitimately contain a clip whose
    # motion is fully random, and that clip genuinely scores 0. What must never
    # happen is a score that depends on what else was in the batch.
    from pipeline.signals import stability_score
    absolute = (stability_score(0.0) == 1.0
                and stability_score(3.15) == 0.0
                and abs(stability_score(0.7854) - 0.5) < 0.01)
    check("stability scoring is absolute, not batch-ranked", absolute,
          "steady=1.0, random=0.0, independent of the batch")

    ranked = sorted(c["signals"]["stability_score"] for c in doc["cards"])
    check("stability scores are not a rank permutation",
          ranked != [0.0, 0.5, 1.0], f"{[round(v, 2) for v in ranked]}")

    # sequence.py trims around `peak` and reads edge motion to choose
    # transitions; a card missing either silently degrades both.
    carried = all(("peak" in c and "exit_flow" in c and "entry_flow" in c)
                  for c in doc["cards"])
    check("moment fields survive onto clip cards", carried, "peak, entry_flow, exit_flow")


# ---------------------------------------------------------------- moments


def _synthetic_clip(dest: Path) -> bool:
    """9s of 4K: 3s sharp and panning, 3s blurred, 3s shaken.

    One ffmpeg command rather than three concatenated files -- `crop` evaluates
    x per frame and `gblur` honours `enable`, so the whole thing is expressible
    as time conditions, and nothing depends on two encoders agreeing.
    """
    # The base must be wide enough for the whole pan AND the shake to stay
    # inside it. Get this wrong and crop silently clamps at the edge, the
    # picture stops moving, and the "shaken" third is really a static one --
    # which the finder then quite correctly scores as good footage.
    base = _SCRATCH / "base.png"
    if not base.exists():
        proc = subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
             "-i", "mandelbrot=size=5760x3240:rate=1", "-frames:v", "1", str(base)],
            capture_output=True, text=True)
        if proc.returncode != 0:
            return False

    pan_x = "if(lt(t,6), 100+170*t, 1200+60*sin(19*t))"   # max 1120, limit 1920
    pan_y = "if(lt(t,6), 400, 400+45*sin(26*t))"          # max 445,  limit 1080
    proc = subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-loop", "1", "-i", str(base), "-t", "9",
         "-vf", f"crop=3840:2160:'{pan_x}':'{pan_y}',"
                f"gblur=sigma=14:enable='between(t,3,6)',fps=30,format=yuv420p",
         "-c:v", "h264_videotoolbox", "-b:v", "30M", "-an", str(dest)],
        capture_output=True, text=True)
    return proc.returncode == 0 and dest.exists()


def check_moment_finder() -> None:
    """The claim this whole update rests on: it finds the good seconds."""
    inputs = _SCRATCH / "synth"
    inputs.mkdir(parents=True, exist_ok=True)
    clip = inputs / "SYNTH.mp4"

    if not _synthetic_clip(clip):
        check("synthetic test clip builds", False, "ffmpeg could not generate it")
        return

    work = _SCRATCH / "synth_work"
    env = {**os.environ, "REEL_WORK_DIR": str(work), "REEL_OUT_DIR": str(_SCRATCH / "synth_out")}
    for args in (("pipeline.ingest", "--inputs", str(inputs)),
                 ("pipeline.proxy",), ("pipeline.scenes",),
                 ("pipeline.features",), ("pipeline.moments",)):
        proc = subprocess.run([PY, "-m", *args], capture_output=True, text=True,
                              cwd=ROOT, env=env)
        if proc.returncode != 0:
            check(f"moment chain: {args[0]}", False, proc.stdout.strip()[-160:])
            return

    moments = json.loads((work / "moments.json").read_text())
    check("moments found in a continuous take", bool(moments), f"{len(moments)} found")
    if not moments:
        return

    # The clip is sharp 0-3s, blurred 3-6s, shaken 6-9s. Everything picked must
    # sit in the first third -- this is the discrimination the finder exists for.
    outside = [m for m in moments if m["start"] >= 3.0 or m["end"] > 3.6]
    check("picks land in the sharp, steady section, not the blurred or shaken ones",
          not outside,
          f"{len(moments)} picks, worst end {max(m['end'] for m in moments):.2f}s")

    lengths = [m["duration"] for m in moments]
    check("every window is a usable shot length",
          all(config.MOMENT_MIN_SECONDS - 0.2 <= d <= config.MOMENT_MAX_SECONDS + 0.2
              for d in lengths),
          f"{min(lengths):.1f}-{max(lengths):.1f}s")

    ordered = sorted(moments, key=lambda m: m["start"])
    overlaps = [(a["shot_id"], b["shot_id"]) for a, b in zip(ordered, ordered[1:])
                if b["start"] < a["end"] + config.MOMENT_MIN_GAP]
    check("no two moments overlap or crowd each other", not overlaps,
          f"min gap {config.MOMENT_MIN_GAP}s")

    peaks_inside = all(m["start"] <= m["peak"] <= m["end"] for m in moments)
    check("every peak lies inside its own window", peaks_inside)


# ---------------------------------------------------------------- sequencing


def _shot(dx: float, dy: float = 0.0, coherence: float = 0.95,
          stability: float = 0.9, klass: str = "landscape") -> dict:
    """A minimal card carrying only what transition selection reads."""
    flow = {"dx": dx, "dy": dy, "magnitude": abs(dx) + abs(dy), "coherence": coherence}
    return {"source": "x.mp4", "entry_flow": dict(flow), "exit_flow": dict(flow),
            "signals": {"stability_score": stability},
            "tags": {"content_class": klass}}


def _check_transition_rules(sequence) -> None:
    """Test the rule itself, not whichever transitions the test footage happens
    to produce. The assembled-timeline check above verified zero whips on this
    data, which proves nothing about the direction logic."""
    strong = config.WHIP_MIN_MAGNITUDE + 0.4

    left = sequence.propose_transition(_shot(-strong), _shot(-strong), False)
    right = sequence.propose_transition(_shot(strong), _shot(strong), False)
    check("matched leftward motion earns a whip_left",
          left is not None and left["style"] == "whip_left",
          left["style"] if left else "no transition proposed")
    check("matched rightward motion earns a whip_right",
          right is not None and right["style"] == "whip_right",
          right["style"] if right else "no transition proposed")

    opposed = sequence.propose_transition(_shot(-strong), _shot(strong), False)
    check("opposing motion earns no whip",
          opposed is None or not opposed["style"].startswith("whip"),
          opposed["style"] if opposed else "hard cut")

    weak = config.WHIP_MIN_MAGNITUDE - 0.2
    drifting = sequence.propose_transition(_shot(-weak), _shot(-weak), False)
    check("motion too weak to notice earns no whip",
          drifting is None or not drifting["style"].startswith("whip"),
          drifting["style"] if drifting else "hard cut")

    # Shake is spatially coherent, so coherence alone would wave this through.
    shaky = sequence.propose_transition(_shot(-strong, stability=0.1),
                                        _shot(-strong, stability=0.1), False)
    check("a shaky shot earns no whip despite coherent motion",
          shaky is None or not shaky["style"].startswith("whip"),
          shaky["style"] if shaky else "hard cut")

    still = config.MOTION_STILL / 2
    dissolve = sequence.propose_transition(_shot(still, klass="water"),
                                           _shot(still, klass="water"), False)
    check("two still shots of the same subject earn a dissolve",
          dissolve is not None and dissolve["style"] == "dissolve",
          dissolve["style"] if dissolve else "hard cut")

    # Never a *blend* between unrelated subjects: dissolving a pan of eggs into a
    # gym wall is the specific mistake this guards. A cut-style stab is fine and
    # is what v3 chooses, since two static shots back to back need something.
    differing = sequence.propose_transition(_shot(still, klass="water"),
                                            _shot(still, klass="food"), False)
    blended = differing is not None and differing["style"] in effects.BLEND_STYLES
    check("two still shots of different subjects are never blended",
          not blended, differing["style"] if differing else "hard cut")


def check_effects_engine() -> None:
    """Every registered effect must compile, and none may change the frame count.

    Compiled here rather than rendered: 26 renders would dominate the suite's
    runtime, and a spec that compiles to a valid graph is what the render then
    exercises anyway. The interaction bugs this suite does need to catch are the
    ones below.
    """
    from pipeline import effects

    defaults: dict[str, dict] = {
        "pan": {"pixels": 120}, "shake": {"amplitude": 6},
        "push_in": {"to": 1.1}, "pull_out": {"from": 1.1},
        "dutch": {"degrees": 3}, "spin": {"turns": 0.04},
        "speed_ramp": {"from": 0.8, "to": 1.4}, "speed_up": {"factor": 1.5},
        "motion_blur": {}, "echo_trail": {}, "strobe": {},
        "blur_ramp": {"sigma": 10}, "flash": {}, "exposure_pump": {},
        "contrast_punch": {}, "grade_shift": {}, "glitch": {},
        "chroma_bleed": {}, "grain": {}, "vignette_pulse": {},
        "sharpen": {}, "light_leak": {},
        "night_grade": {}, "negate_flash": {"at": 0.2, "frames": 3},
        "text": {"png": str(_text_cue())}, "letterbox": {},
    }
    missing = sorted(set(effects.REGISTRY) - set(defaults))
    check("every registered effect is covered here", not missing, str(missing))

    failed: list[str] = []
    for name, spec in defaults.items():
        if name not in effects.REGISTRY:
            continue
        clip = {"source": CLIP_1, "in": 0.0, "out": 2.0,
                "effects": [{"type": name, **spec}]}
        ctx = effects.Context(3840, 2160, 1214, 2160, duration=2.0, source_span=2.0)
        try:
            graph = effects.build_chain(clip, ctx, "0:v", "out")
            if "[out]" not in graph:
                failed.append(f"{name}: no output label")
        except Exception as exc:  # noqa: BLE001 - any failure is a failed check
            failed.append(f"{name}: {exc}")
    check("all effects compile to a filtergraph", not failed,
          f"{len(defaults)} effects" if not failed else "; ".join(failed[:3]))

    # Duration is pinned from the timeline, so anything that retimes must declare
    # it. A 1.6x speed_up rendered 19 frames against a claimed 30 before this.
    sped = {"kind": "shot", "source": CLIP_1, "in": 0.0, "out": 1.0,
            "effects": [{"type": "speed_up", "factor": 1.6}]}
    check("a retiming effect is reflected in the segment's frame count",
          render.segment_frames(sped) == 19,
          f"{render.segment_frames(sped)} frames for 1.0s at 1.6x")

    # An unknown effect must be refused before ffmpeg is invoked, not after.
    try:
        effects.plan_effects({"effects": [{"type": "nope"}]})
        rejected = False
    except effects.EffectError:
        rejected = True
    check("an unknown effect is rejected up front", rejected)

    # Two effects that both zoom must compose, not conflict: a style push-in on a
    # shot that also ends in a zoom_punch is an ordinary combination.
    both = {"source": CLIP_1, "in": 0.0, "out": 2.0, "effects": [
        {"type": "push_in", "to": 1.1}, {"type": "push_in", "to": 1.2}]}
    ctx = effects.Context(3840, 2160, 1214, 2160, duration=2.0, source_span=2.0)
    try:
        effects.build_chain(both, ctx, "0:v", "out")
        composed = ctx.zoom_expr is not None and ")*(" in ctx.zoom_expr
    except effects.EffectError:
        composed = False
    check("two zooms compose instead of erroring", composed)

    # `exposure` leaves the chain in float; grain after it saturated the frame.
    stack = {"source": CLIP_1, "in": 0.0, "out": 2.0, "effects": [
        {"type": "exposure_pump"}, {"type": "grain"}]}
    ctx = effects.Context(3840, 2160, 1214, 2160, duration=2.0, source_span=2.0)
    graph = effects.build_chain(stack, ctx, "0:v", "out")
    restored = graph.index(f"format={effects.WORKING_FORMAT}") < graph.index("noise=")
    check("the format is restored before grain follows an exposure change", restored)

    # Effects that compound rather than combine must collapse to one.
    merged = effects.merge_effects(
        {"effects": [{"type": "motion_blur", "frames": 3}]},
        [{"type": "motion_blur", "frames": 4}])
    blurs = [e for e in merged["effects"] if e["type"] == "motion_blur"]
    check("motion blur does not stack with itself",
          len(blurs) == 1 and blurs[0]["frames"] == 4,
          f"{len(blurs)} instance(s)")


def check_sequencing() -> None:
    """Assembly: on the beat, within the transition budget, and frame-exact."""
    from pipeline import sequence

    cards = json.loads(config.CLIP_CARDS_JSON.read_text())["cards"]
    beats = [round(i * 0.5, 3) for i in range(80)]   # a clean 120 BPM grid
    music = {"beats": beats, "beat_period": 0.5, "best_start": 0.0,
             "bar_seconds": 2.0, "tempo": 120.0, "track": "synthetic"}

    slots = sequence.plan_slots(music, 12.0, 6, "standard")
    check("slot boundaries land on the beat grid",
          all(any(abs(sum(s["duration"] for s in slots[:i]) - b) < 0.08 for b in beats)
              for i in range(1, len(slots) + 1)),
          f"{len(slots)} slots")

    # Every boundary must be a whole number of output frames, or the encoder
    # rounds each segment independently and the drift accumulates.
    frame = 1.0 / config.OUT_FPS
    off_grid = [s["duration"] for s in slots
                if abs(s["duration"] / frame - round(s["duration"] / frame)) > 1e-6]
    check("slot durations are whole frames", not off_grid,
          f"{config.OUT_FPS} fps")

    variant = sequence.assemble(cards, music, 12.0, min(6, len(cards)),
                                "standard", 0, None)
    check("an assembly is produced", bool(variant))
    if not variant:
        return

    segments = variant["timeline"]["segments"]
    total = sum(render.segment_duration(s) for s in segments)
    planned = sum(s["duration"] for s in variant["slots"])
    check("transitions do not change the reel's length",
          abs(total - planned) < 0.01,
          f"planned {planned:.3f}s, built {total:.3f}s")

    boundaries = max(len(variant["picked"]) - 1, 1)
    ratio = len(variant["transitions"]) / boundaries
    check("transitions stay inside their budget",
          ratio <= config.TRANSITION_BUDGET + 1e-9,
          f"{len(variant['transitions'])}/{boundaries} = {ratio:.0%} "
          f"(budget {config.TRANSITION_BUDGET:.0%})")

    adjacent = [i for i in variant["transitions"] if (i + 1) in variant["transitions"]]
    check("no two transitions are adjacent", not adjacent)

    _check_transition_rules(sequence)

    # A whip may only ever move the way the footage moved.
    wrong = []
    for index, proposal in variant["transitions"].items():
        if not proposal["style"].startswith("whip_"):
            continue
        dx = variant["picked"][index]["exit_flow"]["dx"]
        if (proposal["style"] == "whip_right") != (dx > 0):
            wrong.append(proposal["style"])
    check("whip direction matches the measured camera motion", not wrong,
          f"{len(variant['transitions'])} transitions checked")

    errors = render.validate_timeline(variant["timeline"],
                                      config.SCHEMAS_DIR / "timeline.schema.json")
    check("the assembled timeline is schema-valid", not errors,
          errors[0] if errors else "")


def check_frame_exact_render() -> None:
    """Every segment must render the exact number of frames the timeline claims."""
    timeline = {
        "version": 1, "fps": config.OUT_FPS,
        "segments": [
            # A duration that is an exact frame multiple: the case that used to
            # render one frame long, because the frame at the cut point counts
            # as inside the window.
            {"kind": "shot", "source": CLIP_1, "in": 0.0, "out": 1.5},
            {"kind": "shot", "source": CLIP_2, "in": 0.5, "out": 1.5333333},
        ],
    }
    path = config.WORK_DIR / "_verify_frames.json"
    path.write_text(json.dumps(timeline))

    out = config.OUT_DIR / "_verify_frames.mp4"
    proc = stage(RENDER, "--timeline", str(path), "--out", str(out),
                 "--draft", "--skip-preflight", "--force")
    if proc.returncode != 0:
        check("frame-exact render", False, proc.stdout.strip()[-160:])
        return

    expected = sum(render.segment_frames(s) for s in timeline["segments"])
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(out)],
        capture_output=True, text=True)
    actual = int(probe.stdout.strip() or 0)
    check("rendered frame count matches the timeline exactly", actual == expected,
          f"expected {expected}, got {actual}")


# ---------------------------------------------------------------- v4 fixtures


def _text_cue() -> Path:
    """A rendered caption, for the effects and safe-box checks."""
    from pipeline import overlay
    return overlay.render_cue("verify", "label", config.WORK_DIR / "cues")


def _synth_track(dest: Path, drop_at: float | None, duration: float = 44.0,
                 bpm: float = 120.0) -> Path:
    """A track with a known structure, so drop detection can be graded.

    Deliberately shaped like the material that broke the first two
    implementations: the build is a riser that reaches the same level as the drop
    it precedes, so loudness alone cannot separate them, and the intro is sparse
    enough that a beat tracker will not cover it.
    """
    import numpy as np
    import soundfile as sf

    sr = 22050
    beat, bar = 60.0 / bpm, 60.0 / bpm * 4
    y = np.zeros(int(duration * sr))
    rng = np.random.default_rng(11)

    def env(n: int, decay: float):
        return np.exp(-np.arange(n) / (sr * decay))

    def put(when: float, sig):
        i = int(when * sr)
        n = min(len(sig), len(y) - i)
        if n > 0:
            y[i:i + n] += sig[:n]

    def drums(start: float, bars: int, amp: float, hats: bool):
        for b in range(bars):
            for k in range(4):
                when, n = start + b * bar + k * beat, int(0.25 * sr)
                put(when, amp * 0.9 * np.sin(2 * np.pi * 60 * np.arange(n) / sr)
                    * env(n, 0.06))
                if k % 2:
                    put(when, amp * 0.5 * rng.standard_normal(n) * env(n, 0.03))
            if hats:
                for k in range(8):
                    n = int(0.05 * sr)
                    put(start + b * bar + k * beat / 2,
                        amp * 0.18 * rng.standard_normal(n) * env(n, 0.01))

    n = len(y)
    t = np.arange(n) / sr
    y += 0.08 * (np.sin(2 * np.pi * 220 * t) + np.sin(2 * np.pi * 330 * t))

    if drop_at is None:
        drums(0.0, int(duration / bar), 0.85, True)
    else:
        drums(0.0, int(drop_at / bar) - 4, 0.20, False)
        # The riser: loud, sustained, and almost beatless.
        rise_n = int(4 * bar * sr)
        rt = np.arange(rise_n) / sr
        freq = 200 * np.exp(np.log(20) * rt / rt[-1])
        put(drop_at - 4 * bar,
            0.45 * np.sin(2 * np.pi * np.cumsum(freq) / sr) * (rt / rt[-1]))
        drums(drop_at, int((duration - drop_at) / bar), 1.00, True)

    y = np.clip(y / (np.abs(y).max() + 1e-9) * 0.9, -1, 1)
    sf.write(str(dest), y.astype(np.float32), sr)
    return dest


def check_song_analysis() -> None:
    """The drop is the one measurement the whole shot list is built on."""
    from pipeline import music

    tracks = config.WORK_DIR / "tracks"
    tracks.mkdir(parents=True, exist_ok=True)

    structured = music.analyse(_synth_track(tracks / "drop.wav", 24.0), 22.0, None, None)
    flat = music.analyse(_synth_track(tracks / "flat.wav", None, 32.0), 22.0, None, None)

    check("tempo survives a half-time detection",
          abs(structured["tempo"] - 120.0) < 3.0, f"{structured['tempo']:.1f} BPM (truth 120)")

    detected = structured["drop_at"]
    bar = structured["bar_seconds"]
    check("the drop is found within one bar of the truth",
          detected is not None and abs(detected - 24.0) <= bar,
          f"{detected}s vs 24.0s (bar {bar:.2f}s)" if detected else "not found")

    check("a flat groove reports no drop", flat["drop_at"] is None,
          f"drop_at={flat['drop_at']}")

    check("the beat grid spans the whole track",
          structured["beats"] and structured["beats"][0] < 2.0
          and structured["beats"][-1] > structured["duration"] - 2.0,
          f"{structured['beats'][0]:.2f}s .. {structured['beats'][-1]:.2f}s "
          f"of {structured['duration']:.1f}s")

    check("strong onsets are a subset of all onsets, not all of them",
          0 < len(structured["onsets"]) < len(structured["beats"]) * 3,
          f"{len(structured['onsets'])} onsets, {len(structured['beats'])} beats")

    # The window must move so the drop lands where the story wants it.
    early = music.realign(structured, 22.0, 0.25)
    late = music.realign(structured, 22.0, 0.75)
    check("the opening bar moves to place the drop",
          late < early - 1.0,
          f"payoff at 0.25 -> start {early:.2f}s; at 0.75 -> start {late:.2f}s")

    check("sections cover the track without gaps",
          structured["sections"]
          and all(abs(a["end"] - b["start"]) < 1e-6
                  for a, b in zip(structured["sections"], structured["sections"][1:])),
          f"{len(structured['sections'])} spans")


def check_blueprints() -> None:
    """Every story template must be valid, buildable, and honest about itself."""
    import statistics

    from pipeline import brief as brief_stage

    blueprints = brief_stage.load_blueprints(ROOT / "blueprints")
    check("blueprints are present", len(blueprints) >= 6, f"{len(blueprints)} found")
    if not blueprints:
        return

    schema = json.loads((ROOT / "schemas" / "blueprint.schema.json").read_text())
    try:
        import jsonschema
        validator = jsonschema.Draft202012Validator(schema)
        invalid = [f"{b['id']}: {next(iter(validator.iter_errors(b))).message[:60]}"
                   for b in blueprints if list(validator.iter_errors(b))]
    except ImportError:
        invalid = []
    check("every blueprint is schema-valid", not invalid, "; ".join(invalid[:2]))

    spine = [b["id"] for b in blueprints
             if [s["role"] for s in b["shots"]][:2] != ["hook", "promise"]
             or not any(s["role"] == "payoff" for s in b["shots"])
             or b["shots"][-1]["role"] != "cta"]
    check("every blueprint follows the hook/promise..payoff/cta spine",
          not spine, str(spine[:3]))

    # The hook has to be inside the first two seconds at every plausible tempo,
    # which is what max_seconds plus enforce_caps() is for.
    late_hooks = []
    for blueprint in blueprints:
        for bpm in (90, 120, 150, 180):
            period = 60.0 / bpm
            multiples, _, _ = brief_stage.plan_beats(
                blueprint, period, blueprint.get("target_seconds", 22))
            if multiples[0] * period > 2.05:
                late_hooks.append(f"{blueprint['id']}@{bpm}")
    check("the hook ends inside the first 2 seconds at any tempo",
          not late_hooks, str(late_hooks[:3]))

    # A declared payoff position that disagrees with the shot weights is a
    # blueprint bug: it was true of eight of nine on the first pass.
    drifted = []
    for blueprint in blueprints:
        goal = blueprint.get("target_seconds", 22)
        derived = statistics.median([
            brief_stage.payoff_fraction(blueprint, 60.0 / bpm, goal)
            for bpm in (100, 120, 140, 160)])
        declared = blueprint["song"]["payoff_at"]
        if abs(declared - derived) > 0.08:
            drifted.append(f"{blueprint['id']} says {declared:.2f}, weights say {derived:.2f}")
    check("declared payoff position matches the shot weights", not drifted,
          "; ".join(drifted[:2]))

    words = [f"{b['id']} ({len(b['copy']['hook'].split())})" for b in blueprints
             if len(b["copy"]["hook"].split()) > brief_stage.HOOK_WORD_LIMIT]
    check("every template hook is inside the word limit", not words, str(words[:3]))


def check_brief_and_cast() -> None:
    """A brief must land on the grid exactly, and a gap must be reported as a gap."""
    from pipeline import brief as brief_stage
    from pipeline import cast as cast_stage
    from pipeline import music

    track = _synth_track(config.WORK_DIR / "tracks" / "brief.wav", 24.0)
    blueprints = brief_stage.load_blueprints(ROOT / "blueprints")
    blueprint = next(b for b in blueprints if b["id"] == "gym_pr_attempt")

    music_map = music.analyse(track, blueprint["target_seconds"], None, None)
    fraction = brief_stage.payoff_fraction(blueprint, music_map["beat_period"],
                                           blueprint["target_seconds"])
    music_map["best_start"] = music.realign(music_map, blueprint["target_seconds"], fraction)

    copy = dict(blueprint["copy"], source="template")
    doc = brief_stage.lay_out(blueprint, music_map, copy)

    errors = brief_stage.validate(doc, ROOT / "schemas" / "brief.schema.json")
    check("the brief is schema-valid", not errors, errors[0] if errors else "")

    # Tolerance rather than exactness: 119 frames at 30fps is 3.9666... which no
    # finite decimal represents. What matters is that the frame count is
    # unambiguous, and 1e-3 leaves three orders of magnitude of margin.
    off_grid = [s["id"] for s in doc["shots"]
                if abs(s["duration"] * config.OUT_FPS
                       - round(s["duration"] * config.OUT_FPS)) > 1e-3]
    check("every brief slot is a whole number of frames", not off_grid, str(off_grid[:3]))

    drift = abs(doc["total_seconds"] - sum(s["duration"] for s in doc["shots"]))
    check("brief durations sum to its stated total", drift < 1e-6, f"drift {drift:.6f}s")

    payoff = next(s for s in doc["shots"] if s["role"] == "payoff")
    offset = abs(payoff["track_time"] - music_map["drop_at"])
    check("the payoff cut lands on the drop", offset <= music_map["beat_period"] + 1e-3,
          f"{offset:.3f}s off (one beat is {music_map['beat_period']:.3f}s)")

    cues = [s for s in doc["shots"] if s.get("text")]
    overrun = [s["id"] for s in cues
               if s["text"]["at"] + s["text"]["duration"] > s["duration"] + 1e-6]
    check("no text cue outlasts the shot it sits on", not overrun, str(overrun[:3]))
    check("the hook carries text from the first frame",
          doc["shots"][0].get("text", {}).get("at") == 0.0,
          f"at {doc['shots'][0].get('text', {}).get('at')}")

    # Casting: a pool that cannot cover the payoff must say so rather than
    # substituting the least-bad candidate, which is the whole point of a brief.
    def card(shot_id: str, duration: float, motion: float,
             coherence: float = 0.9, tags: dict | None = None) -> dict:
        return {"shot_id": shot_id, "source": f"/tmp/{shot_id}.mov",
                "proxy": "", "start": 0.0, "end": duration, "duration": duration,
                "width": 3840, "height": 2160, "fps": 30.0, "max_crop_width": 1214,
                "rank_score": 0.7, "tags": tags,
                "signals": {"technical_score": 0.8, "motion_raw": motion},
                "entry_flow": {"dx": 0, "dy": 0, "magnitude": motion, "coherence": coherence},
                "exit_flow": {"dx": 0, "dy": 0, "magnitude": motion, "coherence": coherence}}

    def matching_card(shot: dict, index: int, duration: float) -> dict:
        """A moment that genuinely answers this requirement.

        Built per shot rather than as one uniform pool. A pool of identical
        high-motion cards is not an "adequate pool" -- it cannot satisfy a shot
        asking for stillness, and casting was right to report that slot unfilled.
        """
        low, high = config.MOTION_BANDS[shot["motion"]]
        motion = (low + min(high, low + 2.0)) / 2
        wanted = cast_stage.CAMERA_CLASSES.get(shot["camera"], "moving")
        if wanted == "locked":
            motion, coherence = min(motion, config.CAMERA_MOVING_MAGNITUDE - 0.2), 0.9
        elif wanted == "loose":
            motion, coherence = max(motion, config.CAMERA_MOVING_MAGNITUDE + 0.5), 0.3
        else:
            motion, coherence = max(motion, config.CAMERA_MOVING_MAGNITUDE + 0.5), 0.9
        return card(f"L{index}", duration, motion, coherence,
                    tags={"framing": shot["framing"],
                          "has_subject": shot["subject"] == "required",
                          "content_class": "workout", "hook_score": 7})

    longest = max(s["duration"] for s in doc["shots"])
    short_pool = [card(f"s{i}", longest - 0.5, 2.0) for i in range(len(doc["shots"]))]
    result = cast_stage.cast(short_pool, doc)
    payoff_slot = next(s for s in result["slots"] if s["role"] == "payoff")
    check("a slot no moment can fill is reported unfilled, not substituted",
          not payoff_slot["filled"] and not result["complete"],
          f"missing {result['missing']}")

    full_pool = [matching_card(shot, i, longest + 1.0)
                 for i, shot in enumerate(doc["shots"])]
    ok = cast_stage.cast(full_pool, doc)
    weakest = min(ok["slots"], key=lambda s: s["fit"])
    check("an adequate pool fills every required slot", ok["complete"],
          f"missing {ok['missing']}; weakest fit {weakest['fit']:.2f} ({weakest['id']})")

    # motion_raw is absolute, so a locked-off batch must not be read as moving.
    still_pool = [card(f"q{i}", longest + 1.0, 0.2) for i in range(len(doc["shots"]))]
    from pipeline import sequence
    check("a batch of locked-off shots is not read as moving",
          not any(sequence.is_moving(c) for c in still_pool),
          f"motion_raw 0.2 across {len(still_pool)} cards")


def check_text_overlay() -> None:
    """Text must stay inside the Instagram-safe interior and actually appear."""
    from pipeline import overlay

    cues = config.WORK_DIR / "cues"
    outside = []
    for text, kind in [("Go", "hook_title"),
                       ("I have never lifted this before", "hook_title"),
                       ("a deliberately overlong hook nobody should write but which "
                        "must still render inside the safe box", "hook_title"),
                       ("watch the knee", "label"),
                       ("Save this before your next session", "cta")]:
        path = overlay.render_cue(text, kind, cues, force=True)
        if not overlay.within_safe_box(path):
            outside.append(f"{kind}: {text[:24]}")
    check("every text treatment stays inside the safe box", not outside, str(outside[:2]))

    # The `movie` source must loop; without loop=0 a still PNG yields one frame
    # and the caption flashes for 1/30th of a second.
    png = overlay.render_cue("hold me", "label", cues)
    clip = {"source": CLIP_1, "in": 0.0, "out": 2.0,
            "effects": [{"type": "text", "png": str(png), "window": [0.0, 1.5]}]}
    ctx = effects.Context(3840, 2160, 1214, 2160, duration=2.0, source_span=2.0)
    graph = effects.build_chain(clip, ctx, "0:v", "out")
    check("the text source loops and rebuilds its timestamps",
          "loop=0" in graph and "setpts=N/(" in graph)

    # A path that would break the filtergraph must be refused before ffmpeg runs.
    bad = {"source": CLIP_1, "in": 0.0, "out": 1.0,
           "effects": [{"type": "text", "png": "/tmp/a:b,c.png"}]}
    try:
        effects.build_chain(bad, effects.Context(3840, 2160, 1214, 2160,
                                                duration=1.0, source_span=1.0),
                            "0:v", "out")
        refused = False
    except effects.EffectError:
        refused = True
    check("a filtergraph-unsafe text path is refused up front", refused)

    # And it must survive a real render, visibly.
    out = config.OUT_DIR / "_verify_text.mov"
    timeline = {"version": 1, "fps": config.OUT_FPS,
                "segments": [{"kind": "shot", **clip}]}
    path = config.WORK_DIR / "_verify_text.json"
    path.write_text(json.dumps(timeline))
    proc = stage(RENDER, "--timeline", str(path), "--out", str(out),
                 "--draft", "--skip-preflight", "--force", "--mute")
    if proc.returncode != 0:
        check("a reel with text renders", False, proc.stdout.strip()[-160:])
        return
    check("a reel with text renders", True)

    def bright(when: float) -> int:
        frame = config.WORK_DIR / f"_t{when}.png"
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{when}", "-i", str(out),
                        "-frames:v", "1", str(frame)], capture_output=True)
        from PIL import Image
        import numpy as np
        return int((np.asarray(Image.open(frame).convert("L")) > 235).sum())

    inside, after = bright(0.8), bright(1.9)
    check("the caption is on screen inside its window and gone after it",
          inside > 500 and after < inside / 4,
          f"{inside} bright px at 0.8s, {after} at 1.9s")


def check_sound_design() -> None:
    """Cues must land on the cuts, and the two audio outputs must differ correctly."""
    import tempfile as _tempfile

    from pipeline import music, sound

    # A short briefed-looking timeline with one recorded join.
    timeline = {
        "version": 1, "fps": config.OUT_FPS,
        "segments": [
            {"kind": "shot", "source": CLIP_1, "in": 0.0, "out": 2.0},
            {"kind": "shot", "source": CLIP_2, "in": 0.0, "out": 2.0},
            {"kind": "shot", "source": CLIP_3, "in": 0.0, "out": 2.0},
        ],
        "joins": [{"segment": 1, "style": "flash_cut", "blend": False}],
        "story": {"blueprint": "test", "shots": ["a", "b", "c"], "dropped": []},
    }
    brief = {"blueprint": "test", "title": "t",
             "shots": [{"id": "a", "role": "hook"}, {"id": "b", "role": "build"},
                       {"id": "c", "role": "payoff"}],
             "music": {"best_start": 0.0}}

    cues = sound.plan_cues(timeline, brief)
    check("cues are planned from the recorded joins, not from segment kinds",
          any(c["cue"] == "whoosh" for c in cues), f"{len(cues)} cues")
    check("the payoff gets the arrival sounds",
          {"impact", "sub_drop"} <= {c["cue"] for c in cues},
          str(sorted({c["cue"] for c in cues})))

    starts = sound.boundaries(timeline)
    check("no cue is placed outside the reel",
          all(-0.001 <= c["at"] <= starts[-1] + 3.0 for c in cues))

    path = config.WORK_DIR / "_verify_sound.json"
    path.write_text(json.dumps(timeline))
    silent = config.OUT_DIR / "_verify_sound.mov"
    proc = stage(RENDER, "--timeline", str(path), "--out", str(silent),
                 "--draft", "--skip-preflight", "--force", "--mute")
    if proc.returncode != 0:
        check("sound design renders", False, proc.stdout.strip()[-160:])
        return

    track = _synth_track(config.WORK_DIR / "tracks" / "sound.wav", 12.0, 30.0)
    proc = stage("pipeline.sound", "--timeline", str(path), "--video", str(silent),
                 "--music", str(track), "--brief", "/nonexistent",
                 "--out-dir", str(config.OUT_DIR))
    if proc.returncode != 0:
        check("sound design renders", False, proc.stdout.strip()[-200:])
        return
    check("sound design renders", True)

    sfx = config.out_path("_verify_sound_sfx", config.OUT_DIR)
    mixed = config.out_path("_verify_sound_mixed", config.OUT_DIR)
    check("both audio versions are written", sfx.exists() and mixed.exists(),
          f"{sfx.name}, {mixed.name}")
    check("the silent master still has no audio track",
          media.audio_stream(media.probe(silent)) is None)

    # The cue bed must be padded to the reel, not merely truncated to it. amix
    # ends when its last input does, so a bed whose final cue lands early is
    # short -- and build_mix_command muxes with -shortest, which then trims the
    # *video*. Measured on a real build: a 23.27s reel shipped at 18.17s with its
    # last shot and the last line of the lyric missing.
    wanted = sum(render.segment_duration(s) for s in timeline["segments"])
    for name, path_ in (("sfx", sfx), ("mixed", mixed)):
        if not path_.exists():
            continue
        got = media.duration_seconds(media.probe(path_))
        check(f"the {name} version is the full length of the reel",
              abs(got - wanted) <= 0.1,
              f"{got:.2f}s vs the timeline's {wanted:.2f}s")

    if not mixed.exists():
        return
    lufs = sound.measure_loudness(mixed)
    check("the mixed version hits the delivery loudness",
          lufs is not None and abs(lufs - config.TARGET_LUFS) <= 1.5,
          f"{lufs:.1f} LUFS (target {config.TARGET_LUFS})" if lufs else "unmeasured")

    if not sfx.exists():
        return
    import numpy as np
    import librosa

    with _tempfile.TemporaryDirectory() as tmp:
        wav = music.extract_audio(sfx, Path(tmp))
        y, sr = librosa.load(str(wav), sr=22050, mono=True)
    env = librosa.onset.onset_strength(y=y, sr=sr)
    times = librosa.times_like(env, sr=sr)
    peaks = times[librosa.onset.onset_detect(onset_envelope=env, sr=sr,
                                             units="frames", backtrack=False)]

    # Only the hard-attack cues are graded on placement. A whoosh fades in over
    # ~95ms and a riser is near-silent for most of its length, so their audible
    # onsets arrive after `at` by design; sound.TRANSIENT_CUES is the module's own
    # statement of which cues that applies to.
    plain = [c for c in cues if c["cue"] in sound.TRANSIENT_CUES]
    worst = max((float(np.min(np.abs(peaks - c["at"]))) for c in plain), default=0.0)
    check("transient cues land within 50ms of their cut",
          len(peaks) > 0 and plain and worst <= 0.050,
          f"worst {worst * 1000:.0f}ms across {len(plain)} cues")


def check_lyric_style() -> None:
    """The reference-reel techniques: letterbox, word track, and text behind a subject."""
    from pipeline import lyrics, matte, overlay

    # A COMPOSITE-stage effect that is a plain Filter must reach the graph.
    # `letterbox` compiled, validated, rendered and did nothing at all, because
    # the compiler collected only Combine ops from that stage.
    clip = {"source": CLIP_1, "in": 0.0, "out": 1.0,
            "effects": [{"type": "letterbox", "ratio": 16 / 9}]}
    ctx = effects.Context(3840, 2160, 1214, 2160, duration=1.0, source_span=1.0)
    graph = effects.build_chain(clip, ctx, "0:v", "out")
    check("a composite-stage Filter effect reaches the filtergraph",
          graph.count("drawbox") == 2, f"{graph.count('drawbox')} drawbox calls")

    # And it must actually black out the bars in a render.
    timeline = {"version": 1, "fps": config.OUT_FPS,
                "segments": [{"kind": "shot", **clip}]}
    path = config.WORK_DIR / "_verify_lb.json"
    path.write_text(json.dumps(timeline))
    out = config.OUT_DIR / "_verify_lb.mov"
    proc = stage(RENDER, "--timeline", str(path), "--out", str(out),
                 "--draft", "--skip-preflight", "--force", "--mute")
    if proc.returncode != 0:
        check("letterbox renders", False, proc.stdout.strip()[-160:])
        return
    frame = config.WORK_DIR / "_lb.png"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", "0.4", "-i", str(out),
                    "-frames:v", "1", str(frame)], capture_output=True)
    import numpy as np
    from PIL import Image
    arr = np.asarray(Image.open(frame).convert("L"))
    strip = int(round(config.OUT_W / (16 / 9)))
    bar = (config.OUT_H - strip) // 2
    check("the letterbox bars are actually black",
          arr[: bar - 4].max() < 24 and arr[config.OUT_H - bar + 4:].max() < 24,
          f"top max {arr[:bar - 4].max()}, bottom max {arr[config.OUT_H - bar + 4:].max()}")

    # Lyric type must stay inside the picture strip, not spill into the bars.
    band = lyrics.letterbox_band(16 / 9)
    cues = config.WORK_DIR / "cues"
    spill = []
    for word in ["LIT", "ANOTHER", "GO", "TONIGHT"]:
        png = overlay.render_cue(word, "lyric", cues, force=True, band=band)
        bounds = Image.open(png).getchannel("A").getbbox()
        if bounds and (bounds[1] < band[0] or bounds[3] > band[1]):
            spill.append(f"{word} {bounds[1]}-{bounds[3]}")
    check("lyric type stays inside the letterbox strip", not spill,
          f"strip {band[0]}-{band[1]}; " + (str(spill[:2]) if spill else "all inside"))

    # A short word must not become taller than a long one just because it is narrow.
    heights = {}
    for word in ["LIT", "ANOTHER"]:
        png = overlay.render_cue(word, "lyric", cues, force=True, band=band)
        bounds = Image.open(png).getchannel("A").getbbox()
        heights[word] = bounds[3] - bounds[1]
    check("a short lyric word is not sized taller than a long one",
          heights["LIT"] <= heights["ANOTHER"] * 1.3,
          f"LIT {heights['LIT']}px vs ANOTHER {heights['ANOTHER']}px")

    # Word placement must cover the reel, including its opening seconds.
    music = json.loads((config.WORK_DIR / "music_map.json").read_text()) \
        if (config.WORK_DIR / "music_map.json").exists() else None
    if music is None:
        from pipeline import music as music_mod
        music = music_mod.analyse(_synth_track(config.WORK_DIR / "tracks" / "lyr.wav",
                                               24.0), 20.0, None, None)
    words = lyrics.parse_words("one two *three four five six seven eight nine ten")
    placed, source = lyrics.place(words, music, music["best_start"], 20.0)
    check("every word is placed", len(placed) == len(words),
          f"{len(placed)}/{len(words)} on {source}")
    check("the lyric starts in the first second and reaches the end",
          placed and placed[0]["at"] <= 1.0 and placed[-1]["at"] >= 20.0 * 0.75,
          f"first {placed[0]['at']:.2f}s, last {placed[-1]['at']:.2f}s")
    overrun = [w["text"] for w in placed if w["at"] + w["duration"] > 20.0 + 1e-6]
    check("no word outlasts the reel", not overrun, str(overrun[:3]))
    check("accents survive placement",
          any(w["treatment"] == "lyric_accent" for w in placed))

    # The matte gate must reject a mask that found nothing, however confident.
    import numpy as np
    empty = np.zeros((320, 320), dtype=np.float32)
    full = np.ones((320, 320), dtype=np.float32)
    subject = np.zeros((320, 320), dtype=np.float32)
    subject[100:220, 120:200] = 1.0            # ~9% of frame
    check("the matte gate rejects an empty mask", not matte.usable(empty)[0])
    check("the matte gate rejects a whole-frame mask", not matte.usable(full)[0])
    check("the matte gate accepts a real subject", matte.usable(subject)[0],
          f"area {matte.usable(subject)[1] * 100:.1f}%")

    # The burn graph must end in a format x264 can actually encode. overlay in
    # RGBA leaves the chain 4:4:4 and the high profile refuses it outright.
    cmd = lyrics.burn_command(Path("a.mov"), Path("t.mov"), Path("m.mkv"), Path("o.mov"))
    graph = cmd[cmd.index("-filter_complex") + 1]
    check("the lyric burn ends in an encodable pixel format",
          "format=yuv420p[v]" in graph)
    check("the matte is applied by multiplying the text's own alpha",
          "alphaextract" in graph and "blend=all_mode=multiply" in graph)
    plain = lyrics.burn_command(Path("a.mov"), Path("t.mov"), None, Path("o.mov"))
    plain_graph = plain[plain.index("-filter_complex") + 1]
    check("without a matte the words simply draw on top",
          "alphaextract" not in plain_graph and "format=yuv420p[v]" in plain_graph)


def check_floating_text() -> None:
    """The Gym-Inspiration composition engine: style packs, layout, bursts, grade."""
    import numpy as np
    from PIL import Image

    import jsonschema

    from pipeline import compose, lyrics, media, sequence

    schema = json.loads((config.SCHEMAS_DIR / "style.schema.json").read_text())
    packs = sorted(config.STYLES_DIR.glob("*.json"))
    check("there are style packs to load", len(packs) >= 4, f"{len(packs)} found")

    bad_schema, bad_face, thin_ladder = [], [], []
    for path in packs:
        doc = json.loads(path.read_text())
        try:
            jsonschema.validate(doc, schema)
        except jsonschema.ValidationError as exc:
            bad_schema.append(f"{path.stem}: {exc.message[:60]}")
            continue
        # Every named face must actually open. A silent fall-through to Pillow's
        # bitmap default would render at 11px and look like a bug in the layout.
        for role, candidates in doc["faces"].items():
            font = compose.load_font(candidates, 64)
            if not hasattr(font, "getname"):
                bad_face.append(f"{path.stem}/{role}")
        if doc["scale_ladder"][-1] / doc["scale_ladder"][0] < 2.5:
            thin_ladder.append(path.stem)
    check("every style pack validates", not bad_schema, str(bad_schema[:2]))
    check("every face a pack names actually loads", not bad_face, str(bad_face[:3]))
    # The references carry a 6-8x size spread inside one reel. A ladder flatter
    # than this cannot produce the composition, whatever the packer does.
    check("every pack's ladder spans a real size range", not thin_ladder,
          str(thin_ladder))

    style = compose.load("chrome")
    band = lyrics.letterbox_band(16 / 9)
    box_x, box_y, box_w, box_h = compose.band_box(band)

    phrase = [{"text": t, "accent": t == "forget"}
              for t in ("so", "just", "forget", "about")]
    placed = compose.layout(phrase, style, band, None, seed=11)
    check("a phrase lays out completely", len(placed) == len(phrase),
          f"{len(placed)}/{len(phrase)}")

    # No two words may overlap by more than the tolerance that lets ascenders
    # interleave. This is the check the whole layout engine exists to pass.
    worst = 0.0
    for i, a in enumerate(placed):
        for b in placed[i + 1:]:
            area = max(a["w"] * a["h"], 1)
            worst = max(worst, compose._overlap(
                (a["x"], a["y"], a["w"], a["h"]),
                (b["x"], b["y"], b["w"], b["h"])) / area)
    check("no two words in a phrase overlap",
          worst <= compose.OVERLAP_TOLERANCE + 1e-6,
          f"worst overlap {worst * 100:.1f}% (tolerance "
          f"{compose.OVERLAP_TOLERANCE * 100:.0f}%)")

    outside = [w["text"] for w in placed
               if w["y"] < box_y or w["y"] + w["h"] > box_y + box_h
               or w["x"] < box_x or w["x"] + w["w"] > box_x + box_w]
    check("every word stays inside the safe box and the picture strip",
          not outside, f"strip rows {box_y}-{box_y + box_h}; out: {outside[:3]}")

    check("a phrase uses more than one rung of the ladder",
          len({w["rung"] for w in placed}) >= 2,
          f"rungs {sorted({w['rung'] for w in placed})}")

    again = compose.layout(phrase, style, band, None, seed=11)
    check("layout is deterministic for a fixed seed",
          [(w["x"], w["y"], w["size"]) for w in placed]
          == [(w["x"], w["y"], w["size"]) for w in again])
    other = compose.layout(phrase, style, band, None, seed=12)
    check("a different seed gives a different composition",
          [(w["x"], w["y"]) for w in placed] != [(w["x"], w["y"]) for w in other])

    # A subject box must actually pull the composition towards it.
    subject = {"box": [box_x, box_y, box_w // 3, box_h]}
    def _left_mass(words):
        return sum(w["x"] + w["w"] / 2 for w in words) / max(len(words), 1)
    pulled = compose.layout(phrase, style, band, subject, seed=11)
    check("a subject box pulls the words towards the subject",
          _left_mass(pulled) < _left_mass(placed),
          f"mean x {_left_mass(pulled):.0f} with subject vs "
          f"{_left_mass(placed):.0f} without")

    # A state image must contain every word that is visible in it, and nothing
    # in the letterbox bars.
    states = config.WORK_DIR / "_verify_states"
    png = compose.render_state(placed, style, states, band, force=True)
    alpha = np.asarray(Image.open(png).getchannel("A"))
    rows = np.where(alpha.max(axis=1) > 8)[0]
    check("a rendered state draws ink, and only inside the strip",
          rows.size and rows[0] >= band[0] and rows[-1] <= band[1],
          f"ink rows {rows[0]}-{rows[-1]} in band {band[0]}-{band[1]}"
          if rows.size else "no ink at all")
    ink = float((alpha > 8).mean())
    check("a four-word state covers a plausible amount of frame",
          0.005 < ink < 0.20, f"{ink * 100:.1f}% of frame")

    # Every pack must render without falling over, and produce visible ink.
    dead = []
    for path in packs:
        pack = compose.load(path.stem)
        words = compose.layout(phrase, pack, band, None, seed=5)
        if not words:
            dead.append(f"{path.stem}: nothing placed")
            continue
        image = compose.render_state(words, pack, states, band,
                                     backdrop="forget", force=True)
        if not (np.asarray(Image.open(image).getchannel("A")) > 8).any():
            dead.append(f"{path.stem}: blank")
    check("every style pack renders visible type", not dead, str(dead[:2]))

    # --- ink legibility ---
    #
    # Nothing in this file looked at `rgb` before. The packs' accent reds are
    # correct as measurements off the references and two of them are also close
    # to unreadable on a phone: chrome's #700D0D is luminance 34 at 0.62 alpha,
    # against a frame the grade takes to 44.
    def _relative_luma(rgb) -> float:
        return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]

    # Assert against what the 255 clamp can actually deliver, not against the
    # target. A saturated red cannot reach INK_LIFT_TO without desaturating --
    # chrome asks for a gain of 2.35, clips its red channel and arrives at 78.6.
    # Gating on the unreachable number would be a check that can only be passed
    # by weakening the colour.
    dim = []
    for path in packs:
        pack = compose.load(path.stem)
        for role in ("base", "accent"):
            raw = pack["colours"][role]["rgb"]
            lifted_ink = compose._adjust_ink_luminance(list(raw))
            if _relative_luma(lifted_ink) < compose.INK_MIN_LUMA:
                dim.append(f"{pack['id']}/{role} "
                           f"{_relative_luma(lifted_ink):.1f}")
            if any(not isinstance(c, int) or not 0 <= c <= 255
                   for c in lifted_ink):
                dim.append(f"{pack['id']}/{role} left the 0-255 integer range")
    check("every ink clears the legibility floor once lifted", not dim,
          str(dim) if dim
          else f"all 8 inks >= {compose.INK_MIN_LUMA:.0f} relative luma")

    # And the lift must survive into the placed word, which is what becomes a
    # PNG. Applying it anywhere the layout does not reach would be invisible
    # here and invisible in the reel.
    chrome_raw = compose.load("chrome")["colours"]["accent"]["rgb"]
    accented = [w for w in compose.layout(phrase, style, band, None, seed=11)
                if w.get("accent")]
    check("a dark accent is lifted before it reaches the PNG",
          bool(accented) and accented[0]["rgb"] != list(chrome_raw)
          and _relative_luma(accented[0]["rgb"]) > _relative_luma(chrome_raw),
          f"chrome accent {list(chrome_raw)} -> "
          f"{accented[0]['rgb'] if accented else 'nothing placed'}")

    # Phrase grouping: words share a group, and a group never exceeds the pack.
    times = [i * 0.4 for i in range(12)]
    groups = compose_groups = lyrics.group(times, 3, 5.0)
    check("phrase grouping never exceeds the pack's limit",
          all(len(g) <= 3 for g in groups), f"sizes {[len(g) for g in groups]}")
    check("phrase grouping covers every word",
          sorted(i for g in groups for i in g) == list(range(len(times))))
    turned = lyrics.group([0.0, 0.2, 0.4, 9.0], 8, 5.0)
    check("a phrase also closes when the music turns over",
          len(turned) == 2, f"{len(turned)} phrases from a 9s gap")

    # Co-residency is the whole point: the reel must have moments where more
    # than one word is on screen.
    if (config.WORK_DIR / "music_map.json").exists():
        music = json.loads((config.WORK_DIR / "music_map.json").read_text())
        line = lyrics.parse_words(
            "so just *forget about the world tonight we are coming for it all")
        timed, _ = lyrics.place(line, music, music.get("best_start", 0.0), 20.0, 4)
        peak = max(sum(1 for w in timed if w["at"] <= t < w["at"] + w["duration"])
                   for t in (w["at"] + 0.01 for w in timed))
        check("words are co-resident, not one at a time", peak >= 2,
              f"peak {peak} on screen at once (the references run 2-5)")
        marks = lyrics.states(timed, 20.0)
        check("the state timeline covers the whole reel",
              abs(sum(hold for hold, _ in marks) - 20.0) < 0.05,
              f"{len(marks)} states summing {sum(h for h, _ in marks):.3f}s")

    # Bursts: the cut count bends, the slot length never does.
    drift, out_of_range = [], []
    for seconds in (0.4, 0.5, 0.9, 1.2, 1.6, 1.8):
        for want in (3, 5, 8):
            lengths = sequence.burst_frames(seconds, want)
            expected = max(int(round(seconds * config.OUT_FPS)),
                           sequence.BURST_MIN_CUTS * sequence.BURST_MIN_FRAMES)
            if sum(lengths) != expected:
                drift.append(f"{seconds}s x{want}: {sum(lengths)}f != {expected}f")
            if not all(sequence.BURST_MIN_FRAMES <= n <= sequence.BURST_MAX_FRAMES
                       for n in lengths):
                out_of_range.append(f"{seconds}s x{want}: {lengths}")
    check("a burst always sums to its slot exactly", not drift, str(drift[:2]))
    check("every burst fragment is 2-7 frames", not out_of_range,
          str(out_of_range[:2]))
    check("a burst beats the ordinary slot floor",
          sequence.burst_frames(0.4, 6)[0] / config.OUT_FPS
          < sequence.MIN_SLOT_SECONDS,
          f"{sequence.burst_frames(0.4, 6)[0] / config.OUT_FPS:.3f}s vs floor "
          f"{sequence.MIN_SLOT_SECONDS}s")

    card = {"shot_id": "s", "source": str(CLIP_1), "start": 0.0, "end": 6.0,
            "peak": 3.0, "duration": 6.0}
    slot = {"index": 0, "duration": 1.2, "offset": 0.0, "on_downbeat": True}
    shot = {"id": "burst", "role": "build", "burst": {"cuts": 6, "negate": True}}
    cards, slots, shots, count = sequence.expand_bursts([card], [slot], [shot])
    check("a burst slot expands into rapid cuts", count == 1 and len(slots) > 1,
          f"{len(slots)} fragments")
    check("the expanded burst preserves the slot's total length",
          abs(sum(s["duration"] for s in slots) - 1.2) < 1e-6,
          f"{sum(s['duration'] for s in slots):.6f}s vs 1.200000s")
    check("the fragments draw from different points in the take",
          len({c["peak"] for c in cards}) == len(cards))
    check("exactly one fragment is inverted",
          sum(1 for s in slots if s.get("negate")) == 1)

    long_slot = {**slot, "duration": sequence.BURST_MAX_SECONDS + 0.5}
    _, kept, _, skipped = sequence.expand_bursts([card], [long_slot], [shot])
    check("a slot too long to burst is left whole",
          skipped == 0 and len(kept) == 1,
          f"{len(kept)} slots from a {long_slot['duration']:.2f}s request")

    # night_grade must compile, and must move the picture towards the measured
    # reference numbers rather than in some arbitrary direction.
    ctx = effects.Context(1080, 1920, 1080, 1920, duration=1.0, source_span=1.0)
    clip = {"source": CLIP_1, "in": 0.0, "out": 1.0,
            "effects": [{"type": "night_grade"}]}
    graph = effects.build_chain(clip, ctx, "0:v", "out")
    check("night_grade compiles into the filtergraph",
          "curves" in graph and "gamma_b" in graph)

    probe = config.WORK_DIR / "_verify_grade.png"
    plain = config.WORK_DIR / "_verify_plain.png"
    vf = ("eq=brightness=-0.10:contrast=1.22:saturation=0.82:"
          "gamma_b=1.100:gamma_r=0.900,curves=all='0/0 0.25/0.16 1/1'")
    for dest, chain in ((plain, "null"), (probe, vf)):
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", "1", "-i", str(CLIP_1),
                        "-frames:v", "1", "-vf", chain, str(dest)],
                       capture_output=True)
    before = np.asarray(Image.open(plain).convert("RGB"), dtype=np.float32)
    after = np.asarray(Image.open(probe).convert("RGB"), dtype=np.float32)
    cool_before = before[..., 2].mean() - before[..., 0].mean()
    cool_after = after[..., 2].mean() - after[..., 0].mean()
    check("night_grade darkens the picture", after.mean() < before.mean() * 0.75,
          f"luma {before.mean():.1f} -> {after.mean():.1f} "
          f"(the references sit at 34.7-39.9)")
    check("night_grade shifts the picture cool", cool_after > cool_before + 4,
          f"B-R {cool_before:+.1f} -> {cool_after:+.1f} "
          f"(the references sit at +9 to +13)")

    # The grade must be solved per clip, not fixed. A shoot whose clips span 45
    # points of luma cannot be served by one setting: the swept defaults, applied
    # unchanged to real footage, produced a 17.6 against the references' 34.7 and
    # a reel too dark to read the type on.
    tone_luma, tone_cool = media.measure_tone(CLIP_1, 0.0, 3.0)
    check("a clip's tone can be measured", 0 < tone_luma < 255,
          f"luma {tone_luma:.1f}, B-R {tone_cool:+.1f}")

    fitted = media.fit_grade(CLIP_1, 0.0, 3.0)
    # All three solved values, not two. Measuring with the default lift while
    # the solve had moved it reported a clip 5 luma darker than it renders.
    after = media.measure_tone(CLIP_1, 0.0, 3.0,
                               extra=media.grade_chain(fitted["brightness"],
                                                       fitted["cool"],
                                                       fitted["lift"]))
    # The band is the *target's*, not the references'. It used to be 34.0-40.5,
    # which was the references' own 34.7-39.9 with a margin. The frame target now
    # sits deliberately above that band -- the references were shot in rooms this
    # footage is not shot in, and reproducing their frame mean on non-studio gym
    # light shipped reels that could not be read on a phone. Widening this check
    # is the honest consequence of that decision, not a concession to it.
    check("the solved grade lands the clip on the frame target",
          41.0 <= after[0] <= 47.0,
          f"luma {tone_luma:.1f} -> {after[0]:.1f} "
          f"(target {media.GRADE_TARGET_LUMA:.1f}; references 34.7-39.9)")
    check("the solved grade lands the clip in the reference colour band",
          8.0 <= after[1] <= 14.0,
          f"B-R {tone_cool:+.1f} -> {after[1]:+.1f} (references +9 to +13)")

    # And it must move *with* the source, not sit still. Two clips of different
    # exposure must get different offsets, or the solve is not solving.
    # Only a mild darkening. Push it far enough that both versions bottom out
    # against the solver's brightness floor and the check stops testing anything
    # -- two saturated clips get near-identical parameters and should.
    dark = config.WORK_DIR / "_verify_dark.mov"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-t", "2", "-i", str(CLIP_1),
                    "-vf", "eq=brightness=-0.12", "-c:v", "libx264", "-crf", "20",
                    "-an", str(dark)], capture_output=True)
    if dark.exists():
        dark_fit = media.fit_grade(dark, 0.0, 2.0)
        saturated = (fitted["brightness"] <= media.GRADE_MIN_BRIGHTNESS + 1e-6
                     and dark_fit["brightness"] <= media.GRADE_MIN_BRIGHTNESS + 1e-6)
        check("a darker clip is graded differently from a bright one",
              saturated or dark_fit["brightness"] > fitted["brightness"] + 0.02,
              f"brightness {fitted['brightness']:+.3f} vs {dark_fit['brightness']:+.3f}"
              + (" (both at the floor)" if saturated else ""))
        dark_after = media.measure_tone(
            dark, 0.0, 2.0,
            extra=media.grade_chain(dark_fit["brightness"], dark_fit["cool"],
                                    dark_fit["lift"]))
        check("both exposures converge on the same look",
              abs(dark_after[0] - after[0]) < 5.0,
              f"{after[0]:.1f} and {dark_after[0]:.1f}")

    schema_card = json.loads(
        (config.SCHEMAS_DIR / "clip_cards.schema.json").read_text())

    def _find_card(node):
        if isinstance(node, dict):
            if node.get("type") == "object" and "shot_id" in (node.get("properties") or {}):
                return node
            for value in node.values():
                found = _find_card(value)
                if found:
                    return found
        return None

    bounds = _find_card(schema_card)["properties"]["grade"]["properties"]
    # Read from the schema rather than restated here. The solver's clamps were
    # widened twice during this work and the schema was not, so merge.py started
    # writing cards its own contract rejected -- and the failure surfaced two
    # stages later as a missing clip_cards.json.
    check("the grade offsets stay inside the range the schema allows",
          all(bounds[k]["minimum"] <= fitted[k] <= bounds[k]["maximum"]
              for k in ("brightness", "cool")),
          f"brightness {fitted['brightness']:+.3f} in "
          f"[{bounds['brightness']['minimum']}, {bounds['brightness']['maximum']}], "
          f"cool {fitted['cool']:+.3f} in "
          f"[{bounds['cool']['minimum']}, {bounds['cool']['maximum']}]")
    check("the solver's own clamps agree with the schema",
          bounds["brightness"]["minimum"] <= media.GRADE_MIN_BRIGHTNESS
          and media.GRADE_MAX_BRIGHTNESS <= bounds["brightness"]["maximum"]
          and bounds["cool"]["minimum"] <= media.GRADE_MIN_COOL
          and media.GRADE_MAX_COOL <= bounds["cool"]["maximum"],
          "a solver that can leave its own schema writes cards nothing will read")

    # --- exposure: the grade must target the subject, not just the frame ---
    check("the grade has a subject target, not only a frame target",
          media.GRADE_TARGET_SUBJECT > media.GRADE_TARGET_LUMA,
          f"subject {media.GRADE_TARGET_SUBJECT} against frame "
          f"{media.GRADE_TARGET_LUMA} — the references carry 53 against 35")
    # Both probes use lifts the solver can actually reach. 0.50 was below the
    # floor once GRADE_MIN_LIFT moved to 0.52, so the check was asserting on a
    # curve nothing would ever be rendered with.
    pivot = f"{media.GRADE_LIFT_PIVOT:.2f}"
    lifted = media.grade_chain(0.0, 0.1, lift=0.80)
    flat_curve = media.grade_chain(0.0, 0.1, lift=media.GRADE_MIN_LIFT)
    check("the grade curve has a mid control point that moves",
          f"{pivot}/0.800" in lifted
          and f"{pivot}/{media.GRADE_MIN_LIFT:.3f}" in flat_curve,
          "without one the subject falls with the room")

    # The fitter and the renderer must build the *same* curve. media.grade_chain
    # is what fit_grade measures against; effects.night_grade is what actually
    # renders. These were two hard-coded copies of the pivot agreeing by luck,
    # and the check above only ever exercised the fitter's -- so moving the
    # constant would have passed verification while silently rendering every clip
    # on a curve it was not solved for.
    night = effects.build_chain(
        {"source": CLIP_1, "in": 0.0, "out": 1.0,
         "effects": [{"type": "night_grade", "lift": 0.80}]}, ctx, "0:v", "o")
    check("the renderer and the fitter share one lift pivot",
          f"{pivot}/0.800" in night,
          f"pivot {pivot} from media.GRADE_LIFT_PIVOT must reach the filtergraph")

    # And the renderer must accept every value the solver is allowed to emit.
    # It did not: night_grade refused brightness outside -0.6..0.2 while
    # fit_grade clamped to -0.85..0.60 and clip_cards.schema.json permitted the
    # same, so a legitimately solved card could validate and then be refused at
    # graph-build time. A real build peaked at +0.1524 with config.GRADE_ARC
    # adding +0.0812 on top -- +0.2336, already past the old guard.
    reach = []
    for edge in (media.GRADE_MIN_BRIGHTNESS, media.GRADE_MAX_BRIGHTNESS):
        try:
            effects.build_chain(
                {"source": CLIP_1, "in": 0.0, "out": 1.0,
                 "effects": [{"type": "night_grade", "brightness": edge}]},
                ctx, "0:v", "o")
        except Exception as exc:  # noqa: BLE001 - a refusal is the failure
            reach.append(f"{edge:+.2f}: {exc}")
    check("night_grade accepts the whole range the solver can emit", not reach,
          str(reach) if reach
          else f"brightness {media.GRADE_MIN_BRIGHTNESS:+.2f}.."
               f"{media.GRADE_MAX_BRIGHTNESS:+.2f} compiles")

    # --- quality: one lossy generation, not two ---
    check("there is an intermediate encoder that is not a delivery codec",
          "prores" in config.FINAL_ENCODERS
          and "prores" in config.INTERMEDIATE_ENCODERS)
    check("the intermediate is not the default delivery encoder",
          config.FINAL_ENCODER not in config.INTERMEDIATE_ENCODERS,
          f"delivery is {config.FINAL_ENCODER}")
    auto_src = (ROOT / "pipeline" / "auto.py").read_text()
    check("a reel with burned-in text renders its picture as an intermediate",
          "--intermediate" in auto_src and "_picture" in auto_src,
          "otherwise the burn re-encodes an already-lossy draft")

    # --- the subject dodge ---
    dodged = lyrics.burn_command(Path("a.mov"), Path("t.mov"), Path("m.mkv"),
                                 Path("o.mov"), 0.06)
    dodge_graph = dodged[dodged.index("-filter_complex") + 1]
    check("a dodge lifts the subject through the matte, locally",
          "alphamerge[plita]" in dodge_graph
          and "eq=brightness=0.0600" in dodge_graph)
    check("the dodged burn still ends in an encodable pixel format",
          "format=yuv420p[v]" in dodge_graph)
    check("the dodged burn still puts the words behind the subject",
          "alphaextract" in dodge_graph and "blend=all_mode=multiply" in dodge_graph)
    check("the matte is split rather than decoded twice for the dodge",
          "split=2[m0][m1]" in dodge_graph)
    # maskedmerge requires all three inputs to share a pixel format and
    # negotiates down to the mask's, which is gray. It shipped a black-and-white
    # reel -- chroma spread 0.59 against the picture's own 16.9 -- while every
    # other measurement still passed, because nothing else was looking at colour.
    check("the dodge does not force the picture to the mask's format",
          "maskedmerge" not in dodge_graph
          and "alphamerge[plita]" in dodge_graph,
          "alphamerge + overlay keeps the picture in colour")

    # And prove it on real pixels, not just on the graph string.
    colour_src = config.WORK_DIR / "_verify_colour.mov"
    grey_mask = config.WORK_DIR / "_verify_mask.mkv"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
                    "-i", "testsrc2=size=320x568:rate=15:duration=1",
                    "-c:v", "libx264", "-crf", "18", str(colour_src)],
                   capture_output=True)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
                    "-i", "color=c=white:size=320x568:rate=15:duration=1",
                    "-vf", "format=gray", "-c:v", "ffv1", str(grey_mask)],
                   capture_output=True)
    if colour_src.exists() and grey_mask.exists():
        blank_track = config.WORK_DIR / "_verify_track.mov"
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
                        "-i", "color=c=black@0.0:size=320x568:rate=15:duration=1",
                        "-vf", "format=rgba", "-c:v", "qtrle", str(blank_track)],
                       capture_output=True)
        burned = config.WORK_DIR / "_verify_dodged.mov"
        cmd = lyrics.burn_command(colour_src, blank_track, grey_mask, burned, 0.05)
        run = subprocess.run(cmd, capture_output=True)
        if run.returncode == 0 and burned.exists():
            probe_cmd = ["ffmpeg", "-v", "error", "-i", str(burned), "-vf",
                         "fps=4,scale=64:64", "-f", "rawvideo",
                         "-pix_fmt", "rgb24", "-"]
            raw = subprocess.run(probe_cmd, capture_output=True).stdout
            frames = len(raw) // (64 * 64 * 3)
            if frames:
                arr = np.frombuffer(raw[:frames * 64 * 64 * 3],
                                    dtype=np.uint8).reshape(frames, 64, 64, 3)
                spread = float((arr.max(axis=3).astype("int16")
                                - arr.min(axis=3)).mean())
                check("a dodged burn comes out in colour", spread > 20.0,
                      f"chroma spread {spread:.1f} on a colour test pattern")
        else:
            check("a dodged burn renders", False,
                  run.stderr.decode()[-160:] if run.stderr else "no output")
    plain_burn = lyrics.burn_command(Path("a.mov"), Path("t.mov"), Path("m.mkv"),
                                     Path("o.mov"), 0.0)
    check("no dodge means no extra pass",
          "plita" not in plain_burn[plain_burn.index("-filter_complex") + 1])

    # --- lyrics that know their own timing ---
    lrc = config.WORK_DIR / "_verify.lrc"
    lrc.write_text("[00:00.50]We are gonna party like\n"
                   "[00:03.20]it is tonight\n"
                   "[00:06.00]So forget about the world\n")
    parsed = lyrics.parse_lyric_file(lrc)
    check("an LRC file's timestamps are read", len(parsed) == 3
          and abs(parsed[0]["at"] - 0.5) < 1e-6 and abs(parsed[2]["at"] - 6.0) < 1e-6,
          str([p["at"] for p in parsed]))
    plain_lyric = config.WORK_DIR / "_verify.txt"
    plain_lyric.write_text("first line here\nsecond line here\n")
    untimed = lyrics.parse_lyric_file(plain_lyric)
    check("a plain-text lyric keeps its lines and carries no times",
          len(untimed) == 2 and all(p["at"] is None for p in untimed))

    if (config.WORK_DIR / "music_map.json").exists():
        track = json.loads((config.WORK_DIR / "music_map.json").read_text())
        timed, source = lyrics.place_timed(parsed, track, 0.0, 12.0, 3)
        check("timed lyrics are placed on their own timestamps", source == "timed"
              and timed and abs(timed[0]["at"] - 0.5) < 0.35,
              f"first word at {timed[0]['at']:.2f}s for a line stamped 0.50s"
              if timed else "nothing placed")
        starts = {}
        for word in timed:
            starts.setdefault(word["phrase"], []).append(word["at"])
        check("no group holds more words than the pack allows",
              all(len(v) <= 3 for v in starts.values()),
              f"group sizes {[len(v) for v in starts.values()]}")
        check("a long line is split, not truncated",
              len(timed) == 13, f"{len(timed)} words from 13 written")
        check("groups run in the order the song does",
              [min(v) for v in starts.values()] == sorted(min(v) for v in starts.values()))

    # Vocal onsets: a denser grid than the mix, because words are not drums.
    if (config.WORK_DIR / "music_map.json").exists():
        mix = track.get("onsets") or []
        vocal = track.get("vocal_onsets") or []
        if vocal:
            check("the vocal grid is denser than the percussion grid",
                  len(vocal) > len(mix),
                  f"{len(vocal)} vocal onsets against {len(mix)} in the mix")

    # negate_flash must be brief. Held long it stops being an interrupt.
    check("negate_flash refuses to run long", _raises(
        lambda: effects.build_chain(
            {"source": CLIP_1, "in": 0.0, "out": 1.0,
             "effects": [{"type": "negate_flash", "frames": 20}]}, ctx, "0:v", "o")))

    # STYLE.md is the specification a future session reads instead of
    # re-deriving all of this. A document that quietly stops matching the code is
    # worse than no document, because it is believed. Every load-bearing constant
    # it states is checked against the code that owns it.
    style_doc = (ROOT / "STYLE.md")
    check("the styling specification exists", style_doc.exists(),
          "CLAUDE.md points every new session at it")
    if style_doc.exists():
        spec = style_doc.read_text()
        drift = []
        for label, value in (
                ("frame-luma target", f"{media.GRADE_TARGET_LUMA:.1f}"),
                ("subject-luma target", f"{media.GRADE_TARGET_SUBJECT:.1f}"),
                ("separation target", f"{media.GRADE_TARGET_SEPARATION:.2f}"),
                ("colour target", f"+{media.GRADE_TARGET_COOL:.1f}"),
                ("lift pivot", f"{media.GRADE_LIFT_PIVOT:.2f}"),
                ("cool floor", f"{media.GRADE_MIN_COOL:.2f}"),
                ("dodge ceiling", f"{media.DODGE_MAX:.2f}"),
                ("matte samples", str(media.MATTE_SAMPLES)),
                ("slot floor", f"{sequence.MIN_SLOT_SECONDS:.2f}"),
                ("burst ceiling", f"{sequence.BURST_MAX_SECONDS:.2f}"),
                ("subject centre", f"{config.SUBJECT_CENTRE_Y:.2f}"),
                ("strip height", str(config.strip_height(16 / 9))),
                ("delivery size", f"{config.OUT_W}\u00d7{config.OUT_H}"),
        ):
            if value not in spec:
                drift.append(f"{label} ({value})")
        check("the styling spec still states the code's own numbers", not drift,
              f"missing from STYLE.md: {drift}" if drift
              else f"{13} constants agree")

        # And every style pack it tabulates must exist, with the stated limits.
        pack_drift = []
        for path in sorted(config.STYLES_DIR.glob("*.json")):
            pack = json.loads(path.read_text())
            if f"`{pack['id']}`" not in spec:
                pack_drift.append(f"{pack['id']} undocumented")
                continue
            if str(pack["max_resident"]) not in spec:
                pack_drift.append(f"{pack['id']} resident count")
        check("every style pack is documented in the spec", not pack_drift,
              str(pack_drift))

    # The pipeline must carry the style and the grade through to the timeline.
    blueprints = {p.stem: json.loads(p.read_text())
                  for p in config.BLUEPRINTS_DIR.glob("gym_*.json")}
    styled = {k: v for k, v in blueprints.items() if v.get("style")}
    check("the gym blueprints declare a style pack", len(styled) >= 4,
          f"{sorted(styled)}")
    missing = [k for k, v in styled.items()
               if not (config.STYLES_DIR / f"{v['style']}.json").exists()]
    check("every blueprint's style pack exists", not missing, str(missing))
    ungraded = [k for k, v in styled.items() if v.get("grade") != "night"]
    check("every floating-text blueprint carries the grade", not ungraded,
          f"{ungraded} — the type has no scrim and needs the dark ground")
    captioned = [k for k, v in styled.items() if v.get("text_mode") != "lyrics"]
    check("a styled blueprint is in lyric mode, not caption mode", not captioned,
          str(captioned))

    # The integration, end to end: a brief with a burst must produce a timeline
    # where the grade is on every segment, the letterbox is on every segment, the
    # burst has expanded, exactly one fragment is inverted -- and the total length
    # is unchanged. Each of those was wired in a different file, and any one of
    # them failing silently would look like a styling opinion rather than a bug.
    def _card(i):
        return {"shot_id": f"s{i}", "source": str(CLIP_1), "start": 0.0, "end": 8.0,
                "peak": 4.0, "duration": 8.0, "rank_score": 0.5,
                "motion_raw": 0.4, "tags": {},
                "grade": {"brightness": -0.05 - i * 0.01, "cool": 0.12}}

    def _shot(index, sid, role, seconds, start, extra=None):
        return {"index": index, "id": sid, "role": role, "beats": 2,
                "duration": seconds, "start": start, "framing": "medium",
                "camera": "static", "subject": "required", "motion": "high",
                **(extra or {})}

    brief = {
        "blueprint": "gym_floating_text", "text_mode": "lyrics",
        "letterbox": 16 / 9, "style": "chrome", "grade": "night",
        "shots": [_shot(0, "a", "hook", 1.4, 0.0),
                  _shot(1, "burst", "build", 1.2, 1.4,
                        {"burst": {"cuts": 6, "negate": True}}),
                  _shot(2, "c", "payoff", 2.0, 2.6)],
    }
    cast = {"slots": [{"id": sh["id"], "filled": True, "shot_id": f"s{i}"}
                      for i, sh in enumerate(brief["shots"])]}
    built = sequence.assemble_briefed([_card(i) for i in range(3)], brief, cast,
                                      None, None)
    segments = built["timeline"]["segments"]
    kinds = [[e["type"] for e in seg.get("effects", [])] for seg in segments]
    wanted = sum(sh["duration"] for sh in brief["shots"])
    built_len = sum(seg["out"] - seg["in"] for seg in segments)

    check("a briefed burst expands in the real timeline", len(segments) == 8,
          f"{len(segments)} segments from 3 brief shots (one a 6-cut burst)")
    check("expanding a burst does not change the reel's length",
          abs(built_len - wanted) < 1e-6, f"{built_len:.6f}s vs {wanted:.6f}s")
    check("the grade lands on every segment",
          all("night_grade" in k for k in kinds),
          f"{sum('night_grade' in k for k in kinds)}/{len(kinds)}")
    solved = [seg for seg in segments
              for e in seg.get("effects", [])
              if e.get("type") == "night_grade" and "brightness" in e]
    check("a card's solved grade reaches the timeline",
          len(solved) == len(segments),
          f"{len(solved)}/{len(segments)} carry per-clip offsets")
    check("the letterbox lands on every segment",
          all("letterbox" in k for k in kinds),
          f"{sum('letterbox' in k for k in kinds)}/{len(kinds)}")
    check("the grade is applied before the bars are drawn",
          all(k.index("night_grade") < k.index("letterbox") for k in kinds),
          "otherwise the bars stop being black")
    check("exactly one segment of the burst is inverted",
          sum("negate_flash" in k for k in kinds) == 1,
          f"{sum('negate_flash' in k for k in kinds)} inverted")
    check("lyric mode leaves no per-shot caption on the timeline",
          not any("text" in k for k in kinds))

    # Mixed-orientation footage into a letterboxed reel. Carving 9:16 first and
    # drawing bars afterwards throws away 90% of a landscape frame; carving the
    # strip's own ratio keeps it. Both source shapes must land on the same
    # geometry, or a reel cut from a mixed shoot changes scale at every cut.
    strip = config.strip_height(16 / 9)
    check("a letterboxed reel has a strip to fit", strip == 608, f"{strip}px")
    land_w, land_h = config.crop_window(3840, 2160, config.OUT_W / strip)
    port_w, port_h = config.crop_window(1728, 3072, config.OUT_W / strip)
    check("a landscape source is used whole in a 16:9 strip",
          land_w / 3840 > 0.99 and land_h / 2160 > 0.99,
          f"crop {land_w}x{land_h} of 3840x2160 "
          f"({land_w * land_h / (3840 * 2160) * 100:.0f}% — 9:16 framing keeps 10%)")
    check("a portrait source is banded, not squashed, in a 16:9 strip",
          abs((port_w / port_h) - (config.OUT_W / strip)) < 0.02,
          f"crop {port_w}x{port_h} = {port_w / port_h:.3f}:1")

    graphs = {}
    for name, (sw, sh) in (("landscape", (3840, 2160)), ("portrait", (1728, 3072))):
        cw, ch = config.crop_window(sw, sh, config.OUT_W / strip)
        lb_ctx = effects.Context(sw, sh, cw, ch, duration=1.0, source_span=1.0,
                                 strip_h=strip)
        graphs[name] = effects.build_chain(
            {"source": CLIP_1, "in": 0.0, "out": 1.0,
             "effects": [{"type": "letterbox", "ratio": 16 / 9}]},
            lb_ctx, "0:v", "o")
    bar = (config.OUT_H - strip) // 2
    check("the strip is made by padding the picture, not by drawing over it",
          all(f"scale={config.OUT_W}:{strip}" in g
              and f"pad={config.OUT_W}:{config.OUT_H}:0:{bar}:black" in g
              and "drawbox" not in g for g in graphs.values()),
          "both orientations")
    check("mixed orientations land on identical delivery geometry",
          all(f"scale={config.OUT_W}:{strip}" in g for g in graphs.values()))

    # And the bars must still get drawn when the frame was *not* built as a strip
    # -- that is the 9:16-native path, and it is the one that already shipped.
    plain_ctx = effects.Context(3840, 2160, 1214, 2158, duration=1.0,
                               source_span=1.0)
    plain = effects.build_chain(
        {"source": CLIP_1, "in": 0.0, "out": 1.0,
         "effects": [{"type": "letterbox", "ratio": 16 / 9}]}, plain_ctx, "0:v", "o")
    check("a full-frame reel still gets its bars drawn",
          plain.count("drawbox") == 2, f"{plain.count('drawbox')} drawbox calls")

    # Cached analysis must not survive a change of footage. Pointed at a new
    # folder with work/ still holding the last shoot, every stage from ingest to
    # merge was skipped and the *previous* project's reel came out reporting
    # success -- a wrong answer delivered confidently, which is the worst kind.
    from pipeline import auto as auto_mod

    saved = config.SOURCES_JSON.read_text() if config.SOURCES_JSON.exists() else None
    try:
        config.SOURCES_JSON.write_text(json.dumps(
            {"inputs": str(Path("/somewhere/else").resolve()), "accepted": [],
             "rejected": []}))
        check("a change of input folder invalidates the cached analysis",
              auto_mod.stale_inputs(Path(".")),
              "otherwise a new shoot silently rebuilds the old reel")
        config.SOURCES_JSON.write_text(json.dumps(
            {"inputs": str(Path(".").resolve()), "accepted": [], "rejected": []}))
        check("the same input folder still reuses the analysis",
              not auto_mod.stale_inputs(Path(".")),
              "re-running after adding one clip must stay cheap")
        config.SOURCES_JSON.write_text("{ not json")
        check("an unreadable sources.json forces a re-analysis",
              auto_mod.stale_inputs(Path(".")))
    finally:
        if saved is None:
            config.SOURCES_JSON.unlink(missing_ok=True)
        else:
            config.SOURCES_JSON.write_text(saved)

    # A portrait band is placed on the subject, which sits below centre.
    port_ctx = effects.Context(1728, 3072, 1728, 972, duration=1.0,
                              source_span=1.0, strip_h=strip)
    _, crop_y = effects._resolve_crop(port_ctx)
    centred = (3072 - 972) / 2
    placed = 3072 * config.SUBJECT_CENTRE_Y - 972 / 2
    check("a portrait band is placed on the subject, not the frame's middle",
          f"{config.SUBJECT_CENTRE_Y:.3f}" in crop_y and placed > centred,
          f"band top {placed:.0f}px vs centred {centred:.0f}px "
          f"(measured subject centre y={config.SUBJECT_CENTRE_Y})")


def _raises(fn) -> bool:
    try:
        fn()
    except Exception:  # noqa: BLE001
        return True
    return False


# ---------------------------------------------------------------- entry


def main() -> int:
    if not TEST_INPUTS.is_dir():
        print(f"{RED}no test clips at {TEST_INPUTS}{RESET}", file=sys.stderr)
        return 1

    for directory in (config.WORK_DIR, config.PROXIES_DIR, config.SEGMENTS_DIR, config.OUT_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    print(f"\n{BOLD}verify{RESET}  {DIM}reel-editor gates · scratch {_SCRATCH}{RESET}\n")
    for name, fn in [
        ("toolchain", check_toolchain),
        ("crop geometry", check_crop_geometry),
        ("ingest gate", check_ingest_gate),
        ("timeline validation", check_timeline_validation),
        ("transition clamp", check_transition_clamp),
        ("render + delivery spec", check_render_and_spec),
        ("frame exactness", check_frame_exact_render),
        ("effects engine", check_effects_engine),
        ("analysis chain", check_analysis_chain),
        ("moment finder", check_moment_finder),
        ("sequencing", check_sequencing),
        ("song analysis", check_song_analysis),
        ("blueprints", check_blueprints),
        ("brief + casting", check_brief_and_cast),
        ("text overlay", check_text_overlay),
        ("sound design", check_sound_design),
        ("lyric style", check_lyric_style),
        ("floating text", check_floating_text),
    ]:
        print(f"{DIM}{name}{RESET}")
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - a crashed check is a failed check
            check(f"{name} raised", False, f"{type(exc).__name__}: {exc}")
        print()

    shutil.rmtree(_SCRATCH, ignore_errors=True)

    failed = [name for name, ok, _ in results if not ok]
    if failed:
        print(f"{RED}{len(failed)} of {len(results)} checks failed{RESET}")
        for name in failed:
            print(f"  - {name}")
        print()
        return 1

    print(f"{GREEN}all {len(results)} checks passed{RESET}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
