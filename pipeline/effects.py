"""Compile effect specs into ffmpeg filter chains.

The half of a video editor v2 never had. Before this, clip_chain() was
`crop -> scale -> fps -> format`: every shot came out a flat, static,
unmodified rectangle, which is exactly why the reels looked basic.

Two rules shape everything here.

**Stage ordering is not cosmetic.** Geometry runs before the downscale, at source
resolution, because cropping and zooming 4K costs no sharpness while doing it
afterwards would upscale. Everything else runs after, at 1080x1920, because
that is a quarter of the pixels and some of these filters are expensive.

**Output duration is sacred.** A segment's length is pinned by the timeline, and
the render pins its frame count to match. Speed effects therefore change how much
*source* is consumed, never how much output is produced -- see speed_ramp, where
the ramp is normalised so its endpoints land exactly on the slot. Get this wrong
and every cut after it drifts off the beat, silently.

Costs below are measured on this machine, per second of 1080x1920 output:
a plain render is 0.51s, `tmix` motion blur is 0.37s (free), and `minterpolate`
is 15.7s -- 30x realtime, which is why it is budgeted rather than offered freely.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable

from pipeline import config


class Stage(IntEnum):
    """Where in the chain an effect runs. Lower runs earlier."""
    GEOMETRY = 0     # at source resolution, before the downscale
    TEMPORAL = 1     # after the scale: retiming, frame synthesis, blending
    LOOK = 2         # grade, blur, glitch, grain
    COMPOSITE = 3    # overlays and second-stream blends


class Cost(IntEnum):
    FREE = 0         # at or below baseline render cost
    CHEAP = 1        # a small constant addition
    HEAVY = 2        # frame synthesis; must be budgeted


# ---------------------------------------------------------------- ops


@dataclass
class Filter:
    """A single-input filter appended to the main chain."""
    text: str


@dataclass
class Combine:
    """A second stream blended into the main one.

    `source` is a complete generator chain (e.g. a gradient) and `filter` is the
    two-input filter that merges it. The compiler allocates the labels, so an
    effect never has to know what else is in the graph.
    """
    source: str
    filter: str


Op = Filter | Combine


@dataclass
class Context:
    """Geometry the chain is being built around, mutable by GEOMETRY effects.

    Geometry effects do not append filters -- they contribute *terms* to the crop
    expression and a zoom factor to the scale. Two effects that both move the
    frame (a pan and a shake) then compose by addition instead of fighting over
    who gets to crop.
    """
    source_w: int
    source_h: int
    crop_w: int
    crop_h: int
    duration: float                       # output seconds
    source_span: float                    # source seconds consumed
    fps: int = config.OUT_FPS
    crop_x_terms: list[str] = field(default_factory=list)
    crop_y_terms: list[str] = field(default_factory=list)
    zoom_expr: str | None = None
    # Non-zero when the reel is letterboxed: the picture is scaled to
    # OUT_W x strip_h and padded out to OUT_H, rather than filling the frame.
    # Set by render.clip_chain from the clip's own `letterbox` effect, because
    # the crop window has to be carved to the strip's ratio and that decision
    # happens before any effect is built.
    strip_h: int = 0
    # Rotation needs far more zoom to hide its corners than a push-in ever does,
    # so it raises its own ceiling rather than letting every clip upscale that far.
    zoom_ceiling: float = config.MAX_ZOOM
    needs_pts_reset: bool = False
    slowest_speed: float = 1.0            # < 1.0 means frame synthesis is required
    # Where this clip's subject actually sits vertically, 0..1 of source height,
    # measured in merge.py. None falls back to config.SUBJECT_CENTRE_Y, which is
    # a median over nine landscape clips and is wrong for any framing outside the
    # 0.43-0.76 range it was measured from.
    subject_y: float | None = None


@dataclass
class Effect:
    name: str
    stage: Stage
    cost: Cost
    build: Callable[[dict, Context], list[Op]]


REGISTRY: dict[str, Effect] = {}


def effect(name: str, stage: Stage, cost: Cost = Cost.FREE):
    def wrap(fn: Callable[[dict, Context], list[Op]]):
        REGISTRY[name] = Effect(name, stage, cost, fn)
        return fn
    return wrap


class EffectError(ValueError):
    """A spec that cannot be compiled. Raised before ffmpeg ever runs."""


# ---------------------------------------------------------------- helpers


def _num(spec: dict, key: str, default: float) -> float:
    try:
        return float(spec.get(key, default))
    except (TypeError, ValueError) as exc:
        raise EffectError(f"{spec.get('type')}: '{key}' must be a number") from exc


def _window(spec: dict, ctx: Context) -> tuple[float, float]:
    """The [start, end] output-time span an effect applies over."""
    span = spec.get("window")
    if isinstance(span, (list, tuple)) and len(span) == 2:
        start, end = float(span[0]), float(span[1])
    else:
        start = _num(spec, "at", 0.0)
        end = start + _num(spec, "duration", ctx.duration)
    start = max(0.0, min(start, ctx.duration))
    end = max(start, min(end, ctx.duration))
    return start, end


def _gate(window: tuple[float, float], ctx: Context) -> str:
    """`enable=` clause, omitted when the effect covers the whole clip.

    Filters like gblur and exposure take a fixed value but honour `enable`, so
    this is how they are confined to part of a shot -- and, stacked, how they ramp.
    """
    start, end = window
    if start <= 1e-6 and end >= ctx.duration - 1e-6:
        return ""
    return f":enable='between(t,{start:.4f},{end:.4f})'"


def _ease(progress: str) -> str:
    """Smoothstep, so movement starts and stops instead of snapping."""
    return f"({progress})*({progress})*(3-2*({progress}))"


# The working format for the chain: 10-bit, matching the ProRes intermediate.
WORKING_FORMAT = "yuv422p10le"


def _restore_format() -> Filter:
    """Convert back to the working format after a filter that leaves float.

    `exposure` is the one in this vocabulary that does: ffmpeg silently
    auto-inserts a conversion to gbrpf32le for it, and every filter after it then
    operates on values in 0..1 rather than 0..1023. `noise` takes its strength in
    raw units, so a mild grain of 6 became total saturation -- measured, a frame
    went from mean 46 / detail 270 to mean 104 / detail 106795, which is why the
    reel's opening shot looked like static.

    Cheap insurance, and it keeps effect authors from having to know the order
    they will be composed in.
    """
    return Filter(f"format={WORKING_FORMAT}")


def _apply_zoom(ctx: Context, expr: str) -> None:
    """Compose a zoom with whatever zoom is already on the clip.

    Zooms multiply rather than conflict. Two of them legitimately co-occur -- a
    style's gentle push-in on a shot that also ends in a zoom_punch transition --
    and refusing the second meant the whole render died on a valid timeline.
    The product is clamped at scale time, so composition cannot quietly exceed
    MAX_ZOOM and start upscaling.
    """
    ctx.zoom_expr = expr if ctx.zoom_expr is None else f"({ctx.zoom_expr})*({expr})"


def _ramp(ctx: Context, window: tuple[float, float], start: float, end: float,
          eased: bool = True) -> str:
    """An expression going from `start` to `end` across `window`, holding outside."""
    a, b = window
    span = max(b - a, 1.0 / ctx.fps)
    progress = f"clip((t-{a:.4f})/{span:.4f},0,1)"
    shaped = _ease(progress) if eased else progress
    return f"({start:.5f}+({end - start:.5f})*{shaped})"


# ---------------------------------------------------------------- geometry


@effect("pan", Stage.GEOMETRY, Cost.FREE)
def _pan(spec: dict, ctx: Context) -> list[Op]:
    """Drift the 9:16 window across the frame.

    The cheapest way to make a locked-off shot feel handled, and it uses the
    2625 px of horizontal latitude a 4K source leaves after the crop. v2 could
    already express this and never once emitted it.
    """
    window = _window(spec, ctx)
    pixels = _num(spec, "pixels", 160.0)
    axis = spec.get("axis", "x")
    term = _ramp(ctx, window, 0.0, pixels)
    (ctx.crop_x_terms if axis == "x" else ctx.crop_y_terms).append(term)
    return []


@effect("shake", Stage.GEOMETRY, Cost.FREE)
def _shake(spec: dict, ctx: Context) -> list[Op]:
    """Add camera shake, deliberately.

    Counter-intuitive but standard in hype edits: a couple of frames of shake on
    an impact reads as energy. Two incommensurate frequencies on the two axes, so
    it never looks like a clean oscillation.
    """
    start, end = _window(spec, ctx)
    amplitude = _num(spec, "amplitude", 6.0)
    freq = _num(spec, "freq", 9.0)
    gate = f"between(t,{start:.4f},{end:.4f})"
    # Decay across the window: a shake that stops abruptly reads as a glitch.
    decay = f"(1-clip((t-{start:.4f})/{max(end - start, 1e-3):.4f},0,1))"
    ctx.crop_x_terms.append(
        f"({amplitude:.3f}*sin(2*PI*{freq:.3f}*t)*{decay}*{gate})")
    ctx.crop_y_terms.append(
        f"({amplitude * 0.7:.3f}*sin(2*PI*{freq * 1.37:.3f}*t+1.1)*{decay}*{gate})")
    return []


@effect("push_in", Stage.GEOMETRY, Cost.FREE)
def _push_in(spec: dict, ctx: Context) -> list[Op]:
    """Scale up over time. `pull_out` is the same effect reversed.

    Because `crop` fixes w/h at configuration time, a zoom cannot be a shrinking
    crop; it is a growing `scale` with `eval=frame` followed by a fixed crop back
    to delivery size. On 4K the 9:16 window is 1214 px wide, so zooming to
    1214/1080 = 1.124x is free of upscaling and anything beyond that trades a
    little sharpness for the punch.
    """
    window = _window(spec, ctx)
    start = _num(spec, "from", 1.0)
    end = _num(spec, "to", 1.10)
    limit = config.MAX_ZOOM
    if max(start, end) > limit + 1e-6:
        raise EffectError(
            f"push_in to {max(start, end):.2f}x exceeds MAX_ZOOM ({limit:.2f}); "
            f"beyond {ctx.crop_w / config.OUT_W:.3f}x this upscales")
    _apply_zoom(ctx, _ramp(ctx, window, start, end))
    return []


@effect("pull_out", Stage.GEOMETRY, Cost.FREE)
def _pull_out(spec: dict, ctx: Context) -> list[Op]:
    reversed_spec = dict(spec)
    reversed_spec.setdefault("from", 1.10)
    reversed_spec.setdefault("to", 1.0)
    return _push_in(reversed_spec, ctx)


def _cover_scale(radians: float) -> float:
    """Smallest scale that keeps a rotated frame's corners outside the frame.

    Rotating w x h by t gives a bounding box of (w·cos + h·sin) x (w·sin + h·cos),
    so covering the original needs the larger of the two ratios. For a portrait
    delivery frame the *width* term dominates by a long way -- h/w is 1.78 --
    which is why an earlier version of this used the height term, under-zoomed,
    and left visible black wedges in the corners of every tilted shot.
    """
    w, h = config.OUT_W, config.OUT_H
    sin, cos = abs(math.sin(radians)), abs(math.cos(radians))
    return max((w * cos + h * sin) / w, (w * sin + h * cos) / h)


def _max_rotation(zoom: float) -> float:
    """Largest rotation (radians) whose corners a given zoom can still hide.

    cos t + k sin t = R sin(t + phi) with k = h/w, so this inverts exactly rather
    than searching.
    """
    k = config.OUT_H / config.OUT_W
    radius = math.hypot(1.0, k)
    phase = math.atan2(1.0, k)
    if zoom >= radius:
        return math.pi / 2
    return max(math.asin(min(zoom / radius, 1.0)) - phase, 0.0)


@effect("dutch", Stage.GEOMETRY, Cost.CHEAP)
def _dutch(spec: dict, ctx: Context) -> list[Op]:
    """Tilt the horizon. Zooms just enough to keep the corners out of frame."""
    degrees = _num(spec, "degrees", 3.0)
    limit = math.degrees(_max_rotation(config.MAX_ZOOM))
    if abs(degrees) > limit:
        raise EffectError(
            f"dutch of {degrees:.1f}deg needs {_cover_scale(math.radians(degrees)):.2f}x "
            f"zoom to hide its corners, over MAX_ZOOM ({config.MAX_ZOOM}); "
            f"limit is {limit:.1f}deg")
    ctx.zoom_ceiling = max(ctx.zoom_ceiling, config.SPIN_MAX_ZOOM)
    _apply_zoom(ctx, f"{_cover_scale(math.radians(degrees)):.5f}")
    return [Filter(f"rotate={math.radians(degrees):.5f}:c=black:ow=iw:oh=ih")]


@effect("spin", Stage.GEOMETRY, Cost.CHEAP)
def _spin(spec: dict, ctx: Context) -> list[Op]:
    """Rotate through a window — the geometric half of a spin transition.

    Clamped rather than refused: a spin is brief and usually blurred, so the
    upscale needed to hide its corners is worth it, but only up to a point.
    """
    window = _window(spec, ctx)
    turns = _num(spec, "turns", 0.04)
    limit = _max_rotation(config.SPIN_MAX_ZOOM)
    radians = max(-limit, min(turns * 2 * math.pi, limit))

    angle = _ramp(ctx, window, 0.0, radians, eased=True)
    ctx.zoom_ceiling = max(ctx.zoom_ceiling, config.SPIN_MAX_ZOOM)
    _apply_zoom(ctx, f"{_cover_scale(radians):.5f}")
    return [Filter(f"rotate='{angle}':c=black:ow=iw:oh=ih")]


# ---------------------------------------------------------------- temporal


@effect("speed_ramp", Stage.TEMPORAL, Cost.FREE)
def _speed_ramp(spec: dict, ctx: Context) -> list[Op]:
    """Accelerate (or decelerate) through the shot, landing exactly on the slot.

    The signature effect of a modern edit, and the one with a trap in it. Output
    duration is fixed by the timeline, so the ramp must redistribute time inside
    that window rather than change it. With source span S and output duration D,
    integrating a linear speed profile gives a closed form:

        o(s) = D * log(1 + (r-1)*s/S) / log(r),   r = to/from

    which is exact, monotonic, and hits o(0)=0 and o(S)=D by construction. Only
    the *ratio* r matters -- the absolute values scale out in the normalisation,
    which is why the net speed still comes from `in`/`out` versus the slot.
    """
    start = _num(spec, "from", 0.6)
    end = _num(spec, "to", 1.8)
    if start <= 0 or end <= 0:
        raise EffectError("speed_ramp: from/to must be positive")

    ratio = end / start
    span, out = ctx.source_span, ctx.duration
    if abs(ratio - 1.0) < 1e-3:
        return []

    ctx.needs_pts_reset = True
    expr = (f"({out:.6f}*log(1+({ratio:.6f}-1)*clip(T,0,{span:.6f})/{span:.6f})"
            f"/log({ratio:.6f}))/TB")

    # Speed at the slow end, which decides whether frames must be synthesised.
    net = span / out if out > 0 else 1.0
    ctx.slowest_speed = min(ctx.slowest_speed,
                            net * math.log(ratio) / (ratio - 1.0))
    return [Filter(f"setpts='{expr}'")]


@effect("speed_up", Stage.TEMPORAL, Cost.FREE)
def _speed_up(spec: dict, ctx: Context) -> list[Op]:
    """Constant retime within the segment.

    Rarely needed directly: the clip's own `in`/`out` against its slot already
    sets the net speed. This exists for a deliberate stutter or a held-then-fast
    beat where the source range is fixed for other reasons.
    """
    factor = _num(spec, "factor", 1.5)
    if factor <= 0:
        raise EffectError("speed_up: factor must be positive")
    ctx.needs_pts_reset = True
    ctx.slowest_speed = min(ctx.slowest_speed, factor)
    return [Filter(f"setpts=PTS/{factor:.5f}")]


@effect("motion_blur", Stage.TEMPORAL, Cost.FREE)
def _motion_blur(spec: dict, ctx: Context) -> list[Op]:
    """Blend consecutive frames. Measured free (0.37s per second of output).

    This is what makes a hard cut between two moving shots read as intentional
    rather than jarring, and it is the cheapest professional-looking thing in the
    whole vocabulary.
    """
    frames = int(_num(spec, "frames", 3))
    if not 2 <= frames <= 8:
        raise EffectError("motion_blur: frames must be 2-8")

    # Weighted toward the current frame, not flat. A flat average over 3 frames
    # at 30fps smears 100ms of movement, where a real camera at a 180-degree
    # shutter gives 17ms -- six times less. Weighting keeps the frame legible
    # while still leaving a trail behind fast movement.
    weights = " ".join(f"{2 ** i}" for i in range(frames))
    return [Filter(f"tmix=frames={frames}:weights='{weights}'")]


@effect("echo_trail", Stage.TEMPORAL, Cost.FREE)
def _echo_trail(spec: dict, ctx: Context) -> list[Op]:
    """Weighted frame blend, so past frames linger behind the present one."""
    frames = int(_num(spec, "frames", 4))
    if not 2 <= frames <= 8:
        raise EffectError("echo_trail: frames must be 2-8")
    weights = " ".join(f"{max(0.15, 1.0 - i * 0.28):.2f}" for i in range(frames))
    return [Filter(f"tmix=frames={frames}:weights='{weights}'")]


@effect("strobe", Stage.TEMPORAL, Cost.FREE)
def _strobe(spec: dict, ctx: Context) -> list[Op]:
    """Hold every nth frame. Reads as a stutter locked to the beat."""
    every = int(_num(spec, "every", 3))
    if not 2 <= every <= 8:
        raise EffectError("strobe: every must be 2-8")
    # Duplicating the kept frame preserves the frame count, and therefore the slot.
    return [Filter(f"select='not(mod(n,{every}))',fps={ctx.fps}")]


# ---------------------------------------------------------------- look


@effect("blur_ramp", Stage.LOOK, Cost.CHEAP)
def _blur_ramp(spec: dict, ctx: Context) -> list[Op]:
    """Ramp blur up or down. The optical half of a whip pan.

    gblur takes a fixed sigma but honours `enable`, so a ramp is a stack of gated
    instances at rising strengths. Six steps over the 0.2s a whip lasts is well
    under one step per two frames -- indistinguishable from a smooth ramp.
    """
    start, end = _window(spec, ctx)
    peak = _num(spec, "sigma", 12.0)
    steps = max(int(_num(spec, "steps", 6)), 2)
    direction = spec.get("direction", "up")
    if direction not in ("up", "down"):
        raise EffectError("blur_ramp: direction must be 'up' or 'down'")

    slice_len = (end - start) / steps
    ops: list[Op] = []
    for index in range(steps):
        fraction = (index + 1) / steps
        if direction == "down":
            fraction = 1.0 - index / steps
        a = start + index * slice_len
        b = a + slice_len
        ops.append(Filter(
            f"gblur=sigma={peak * fraction:.3f}:steps=1"
            f":enable='between(t,{a:.4f},{b:.4f})'"))
    return ops


@effect("flash", Stage.LOOK, Cost.FREE)
def _flash(spec: dict, ctx: Context) -> list[Op]:
    """Blow the exposure for a couple of frames. Hardest on a downbeat."""
    at = _num(spec, "at", 0.0)
    frames = max(int(_num(spec, "frames", 2)), 1)
    strength = _num(spec, "strength", 0.9)
    end = at + frames / ctx.fps
    return [Filter(f"exposure=exposure={strength:.3f}"
                   f":enable='between(t,{at:.4f},{end:.4f})'"),
            _restore_format()]


@effect("exposure_pump", Stage.LOOK, Cost.FREE)
def _exposure_pump(spec: dict, ctx: Context) -> list[Op]:
    """A brief lift on the beat. Subtler than a flash, usable far more often."""
    window = _window(spec, ctx)
    strength = _num(spec, "strength", 0.22)
    return [Filter(f"exposure=exposure={strength:.3f}{_gate(window, ctx)}"),
            _restore_format()]


@effect("contrast_punch", Stage.LOOK, Cost.CHEAP)
def _contrast_punch(spec: dict, ctx: Context) -> list[Op]:
    """Snap contrast and saturation on a transient."""
    window = _window(spec, ctx)
    contrast = _num(spec, "contrast", 1.18)
    saturation = _num(spec, "saturation", 1.12)
    return [Filter(f"eq=contrast={contrast:.3f}:saturation={saturation:.3f}"
                   f"{_gate(window, ctx)}")]


@effect("grade_shift", Stage.LOOK, Cost.CHEAP)
def _grade_shift(spec: dict, ctx: Context) -> list[Op]:
    """Push the colour temperature and balance for a section."""
    window = _window(spec, ctx)
    kelvin = _num(spec, "temperature", 6500.0)
    ops: list[Op] = [Filter(
        f"colortemperature=temperature={kelvin:.0f}:mix={_num(spec, 'mix', 0.6):.2f}"
        f"{_gate(window, ctx)}")]
    lift = _num(spec, "shadows", 0.0)
    if abs(lift) > 1e-6:
        ops.append(Filter(f"colorbalance=rs={lift:.3f}:bs={-lift:.3f}"
                          f"{_gate(window, ctx)}"))
    return ops


@effect("glitch", Stage.LOOK, Cost.CHEAP)
def _glitch(spec: dict, ctx: Context) -> list[Op]:
    """RGB split plus noise for two or three frames.

    Deliberately brief. Held longer than about a tenth of a second it stops
    reading as an accent and starts reading as a broken file.
    """
    at = _num(spec, "at", 0.0)
    frames = max(int(_num(spec, "frames", 2)), 1)
    shift = int(_num(spec, "shift", 9))
    end = at + frames / ctx.fps
    gate = f":enable='between(t,{at:.4f},{end:.4f})'"
    return [
        Filter(f"rgbashift=rh={shift}:bh={-shift}:gv={shift // 2}{gate}"),
        Filter(f"noise=alls={int(_num(spec, 'noise', 22))}:allf=t+u{gate}"),
    ]


@effect("chroma_bleed", Stage.LOOK, Cost.CHEAP)
def _chroma_bleed(spec: dict, ctx: Context) -> list[Op]:
    """Displace the chroma planes. Cheaper and subtler than a full RGB split."""
    window = _window(spec, ctx)
    shift = int(_num(spec, "shift", 4))
    return [Filter(f"chromashift=cbh={shift}:crh={-shift}{_gate(window, ctx)}")]


@effect("grain", Stage.LOOK, Cost.CHEAP)
def _grain(spec: dict, ctx: Context) -> list[Op]:
    """Temporal grain. Ties disparate sources together and hides banding."""
    strength = int(_num(spec, "strength", 8))
    if not 1 <= strength <= 60:
        raise EffectError("grain: strength must be 1-60")
    return [Filter(f"noise=alls={strength}:allf=t")]


@effect("vignette_pulse", Stage.LOOK, Cost.CHEAP)
def _vignette_pulse(spec: dict, ctx: Context) -> list[Op]:
    """Darken the edges, optionally only on a beat."""
    window = _window(spec, ctx)
    angle = _num(spec, "angle", math.pi / 5)
    return [Filter(f"vignette=angle={angle:.4f}{_gate(window, ctx)}")]


@effect("night_grade", Stage.LOOK, Cost.CHEAP)
def _night_grade(spec: dict, ctx: Context) -> list[Op]:
    """The house look of the four Gym-Inspiration reels: dark, cool, crushed.

    Measured across all four, and it is not subtle: mean frame luma sits at
    34.7 / 35.6 / 39.9 / 38.5 out of 255, with blue running 9-13 points above
    red on the frame average. Ordinary well-lit gym footage lands two to three
    times brighter than that and reads neutral.

    This is not a mood filter -- it is load-bearing for the type. The floating
    text is thin, near-white and carries no scrim or shadow at all, which is
    only legible against a ground that has been taken down this far. Rendered
    over a bright commercial gym without it, the words wash out completely.
    Grade and type ship together or neither works.

    `gamma_g`/`gamma_b` rather than a `colorbalance` push: balance shifts the
    whole ramp including the blacks, which lifts the crush this effect just
    paid for. Gamma leaves the black point where it is.
    """
    window = _window(spec, ctx)
    # Calibrated, not guessed: swept against this user's own gym footage until
    # the output matched the references on both axes at once. These values take
    # a mean luma of 86.0 / B-R of -1.4 to 37.0 / +9.4, against a reference band
    # of 34.7-39.9 and +9 to +13. A source already shot dark will land darker --
    # this is a fixed grade, not an auto-exposure, and the numbers above are the
    # thing to re-measure if the footage changes character.
    luma = _num(spec, "brightness", -0.10)
    contrast = _num(spec, "contrast", 1.22)
    saturation = _num(spec, "saturation", 0.82)
    cool = _num(spec, "cool", 0.10)
    if not -0.6 <= luma <= 0.2:
        raise EffectError("night_grade: brightness must be -0.6..0.2")
    # The mid control point. Raising it lifts the subject out of the shadows
    # while the toe holds the room down -- which is the shape of the reference
    # look, and not something brightness can express. Measured on the four
    # references with a segmentation mask: subject 53-84 against frames of 35-40.
    lift = _num(spec, "lift", 0.62)
    if not 0.30 <= lift <= 0.95:
        raise EffectError("night_grade: lift must be 0.30..0.95")
    gate = _gate(window, ctx)
    return [
        Filter(f"eq=brightness={luma:.3f}:contrast={contrast:.3f}"
               f":saturation={saturation:.3f}"
               f":gamma_b={1.0 + cool:.3f}:gamma_r={1.0 - cool:.3f}{gate}"),
        # curves is the only filter in this build that can move the black point
        # and the mid-tones independently of the highlights the rim light needs.
        Filter(f"curves=all='0/0 0.20/{_num(spec, 'toe', 0.10):.3f} "
               f"0.55/{lift:.3f} 1/1'{gate}"),
    ]


@effect("negate_flash", Stage.LOOK, Cost.FREE)
def _negate_flash(spec: dict, ctx: Context) -> list[Op]:
    """Invert the frame for two or three frames, inside a burst.

    Gym_3 does this once, at 6.15s, for 0.07s -- measured R47 G107 B100 against
    a normal R24 G31 B40, i.e. a true photographic negative and not a colour
    push. It sits in the middle of that reel's seven-cut burst and is the single
    hardest pattern interrupt in any of the four references.

    Frames rather than seconds, like `glitch`, because the whole point is that
    it is over before the eye resolves it. Held for a third of a second it stops
    being an interrupt and starts being a mistake.
    """
    at = _num(spec, "at", 0.0)
    frames = max(int(_num(spec, "frames", 3)), 1)
    if frames > 8:
        raise EffectError("negate_flash: more than 8 frames reads as a broken file")
    end = at + frames / ctx.fps
    return [Filter(f"negate=enable='between(t,{at:.4f},{end:.4f})'")]


@effect("sharpen", Stage.LOOK, Cost.CHEAP)
def _sharpen(spec: dict, ctx: Context) -> list[Op]:
    """A little crispness back after the downscale."""
    amount = _num(spec, "amount", 0.6)
    return [Filter(f"unsharp=5:5:{amount:.3f}:5:5:0.0")]


# ---------------------------------------------------------------- composite


@effect("light_leak", Stage.COMPOSITE, Cost.CHEAP)
def _light_leak(spec: dict, ctx: Context) -> list[Op]:
    """Screen a warm gradient over the frame.

    Generated rather than sampled: `gradients` is a source filter, so this needs
    no asset and no extra input to the command.
    """
    window = _window(spec, ctx)
    colour = spec.get("colour", "0xFF9A3C")
    opacity = _num(spec, "opacity", 0.30)
    if not 0 < opacity <= 1:
        raise EffectError("light_leak: opacity must be within (0, 1]")
    source = (f"gradients=s={config.OUT_W}x{config.OUT_H}:r={ctx.fps}"
              f":d={ctx.duration + 0.5:.3f}:c0=0x000000:c1={colour}"
              f":nb_colors=2:type={spec.get('shape', 'radial')}:speed=0")
    return [Combine(source,
                    f"blend=all_mode=screen:all_opacity={opacity:.3f}"
                    f"{_gate(window, ctx)}")]


@effect("letterbox", Stage.COMPOSITE, Cost.FREE)
def _letterbox(spec: dict, ctx: Context) -> list[Op]:
    """Black bars top and bottom, leaving a wider strip of picture.

    The reference reel is a 16:9 strip inside the 9:16 frame -- measured at rows
    822-1485 of a 2556-tall recording, exactly 16:9, and constant for its whole
    duration. It is a real style choice and not an accident of a landscape
    upload: the type is composed inside the strip, not in the bars.

    Implemented by drawing the bars rather than by scaling the picture, so the
    subject stays the size it was shot and the crop keeps pointing where
    moments.py aimed it. Scaling to fit would shrink the subject by 44%.
    """
    ratio = _num(spec, "ratio", 16.0 / 9.0)
    if ratio <= 0:
        raise EffectError("letterbox: ratio must be positive")

    strip = config.strip_height(ratio)
    bar = max((config.OUT_H - strip) // 2, 0) if strip else 0
    if bar <= 0:
        return []
    # If the scale stage already padded to this strip, the bars exist and are
    # black. Drawing them again would be harmless but pointless -- and it would
    # hide a mismatch between the two, which is exactly the bug worth surfacing.
    if ctx.strip_h:
        if ctx.strip_h != strip:
            raise EffectError(
                f"letterbox: ratio {ratio:.3f} wants a {strip}px strip but the "
                f"frame was built at {ctx.strip_h}px")
        return []

    opacity = _num(spec, "opacity", 1.0)
    window = _window(spec, ctx)
    return [Filter(
        f"drawbox=x=0:y=0:w={config.OUT_W}:h={bar}:color=black@{opacity:.3f}:t=fill"
        f"{_gate(window, ctx)}"),
        Filter(
        f"drawbox=x=0:y={config.OUT_H - bar}:w={config.OUT_W}:h={bar}:"
        f"color=black@{opacity:.3f}:t=fill{_gate(window, ctx)}")]


TEXT_FADE_IN = 0.12
TEXT_FADE_OUT = 0.18
# How far the block travels into place. Small: a caption that flies in from
# off-screen reads as a template, while a few pixels of settle reads as a cut.
TEXT_RISE_PIXELS = 20.0
TEXT_RISE_SECONDS = 0.22

_SAFE_PNG = re.compile(r"^[A-Za-z0-9._/\-]+\.png$")


@effect("text", Stage.COMPOSITE, Cost.CHEAP)
def _text(spec: dict, ctx: Context) -> list[Op]:
    """Composite a pre-rendered caption, fading and settling into place.

    The PNG is full-frame with the type already positioned, so nothing here
    scales it -- a scaled text layer is a soft text layer, and at 1080 wide the
    softness is visible on a phone.

    Three details are load-bearing:

    * `loop=0` makes `movie` repeat forever. Without it a still PNG yields one
      frame and then EOF, and the caption appears for 1/30th of a second.
    * `setpts=N/(FPS*TB)` rebuilds timestamps from the frame counter. The looped
      frames otherwise all carry the original frame's PTS, which `overlay` reads
      as a single frame that never advances.
    * `fade` is used for the alpha ramp rather than a gated stack of opacity
      steps, because it is one of the few filters here that both animates
      smoothly and honours a start time.
    """
    path = str(spec.get("png") or "")
    if not path:
        raise EffectError("text: needs a rendered `png`")
    # The path is interpolated straight into filter_complex, where ':' separates
    # arguments and ',' separates filters. overlay.py hashes its filenames so
    # this always passes; a hand-written timeline might not.
    if not _SAFE_PNG.match(path):
        raise EffectError(f"text: unsafe path for a filtergraph: {path}")

    start, end = _window(spec, ctx)
    span = max(end - start, 0.1)
    fade_out_at = max(start, end - TEXT_FADE_OUT)

    source = (
        f"movie={path}:loop=0,"
        f"setpts=N/({ctx.fps}*TB),"
        f"format=rgba,"
        f"fade=t=in:alpha=1:st={start:.3f}:d={min(TEXT_FADE_IN, span / 2):.3f},"
        f"fade=t=out:alpha=1:st={fade_out_at:.3f}:d={TEXT_FADE_OUT:.3f}"
    )

    # Settle upward into position over the fade-in. `min` clamps the ramp so the
    # expression stays valid for the whole clip, not only inside the window.
    rise = (f"{TEXT_RISE_PIXELS:.1f}*(1-min(1,max(0,(t-{start:.3f}))"
            f"/{TEXT_RISE_SECONDS:.3f}))")

    return [Combine(source, f"overlay=x=0:y='{rise}':format=auto"
                            f"{_gate((start, end), ctx)}")]


# ---------------------------------------------------------------- transitions

# Styles that join with xfade, and therefore need a transition segment holding
# both clips. Everything else is a hard cut with accents on either side, which is
# both cheaper and closer to how professional edits actually get their density:
# the cut stays invisible and the punch sells it.
BLEND_STYLES = {
    "dissolve", "fade", "fade_black", "whip_left", "whip_right", "whip_up",
    "whip_down", "blur", "wipe_up", "zoom_in", "squeeze", "circle_open",
    "pixelize", "flash",
}


@dataclass
class Recipe:
    """How to realise a transition: accents on each side, plus an optional blend."""
    a_effects: list[dict] = field(default_factory=list)
    b_effects: list[dict] = field(default_factory=list)
    blend: bool = False


def transition_recipe(style: str, duration: float, dur_a: float, dur_b: float,
                      exit_flow: dict | None = None) -> Recipe:
    """Compose a transition from the effect vocabulary.

    v2's `whip_left` was a bare `xfade=smoothleft` -- a soft edge sliding across,
    which is not what a whip looks like. A whip is directional blur ramping up as
    the outgoing shot leaves, a fast positional slide, and blur ramping back down
    as the incoming shot settles. Built here from the same effects any clip can
    use, so a transition is not a special case in the renderer.

    Windows are in each clip's own output time: A's accent sits at its tail, B's
    at its head.
    """
    tail = max(dur_a - duration, 0.0)
    head = min(duration, dur_b)
    horizontal = True
    if exit_flow:
        horizontal = abs(exit_flow.get("dx", 0.0)) >= abs(exit_flow.get("dy", 0.0))
    direction = 1.0 if (exit_flow or {}).get("dx", 0.0) > 0 else -1.0

    if style in ("whip_left", "whip_right", "whip_up", "whip_down"):
        # Blur along the travel, and slide the frame the way the camera was going.
        slide = 260.0 * direction
        axis = "x" if horizontal else "y"
        return Recipe(
            a_effects=[
                {"type": "blur_ramp", "sigma": 26, "direction": "up",
                 "window": [tail, dur_a]},
                {"type": "pan", "pixels": slide, "axis": axis,
                 "window": [tail, dur_a]},
                {"type": "motion_blur", "frames": 3},
            ],
            b_effects=[
                {"type": "blur_ramp", "sigma": 26, "direction": "down",
                 "window": [0.0, head]},
                {"type": "pan", "pixels": -slide, "axis": axis,
                 "window": [0.0, head]},
                {"type": "motion_blur", "frames": 3},
            ],
            blend=True)

    if style == "flash_cut":
        return Recipe(
            a_effects=[{"type": "flash", "at": max(dur_a - 2 / config.OUT_FPS, 0.0),
                        "frames": 2, "strength": 0.75}],
            b_effects=[{"type": "flash", "at": 0.0, "frames": 2, "strength": 0.9},
                       {"type": "exposure_pump", "strength": 0.18,
                        "window": [0.0, head]}])

    if style == "glitch_cut":
        return Recipe(
            a_effects=[{"type": "glitch", "at": max(dur_a - 2 / config.OUT_FPS, 0.0),
                        "frames": 2, "shift": 11}],
            b_effects=[{"type": "glitch", "at": 0.0, "frames": 3, "shift": 14},
                       {"type": "chroma_bleed", "shift": 5, "window": [0.0, head]}])

    if style == "zoom_punch":
        return Recipe(
            a_effects=[{"type": "push_in", "from": 1.0, "to": 1.22,
                        "window": [tail, dur_a]},
                       {"type": "motion_blur", "frames": 3}],
            b_effects=[{"type": "pull_out", "from": 1.18, "to": 1.0,
                        "window": [0.0, min(head * 2.5, dur_b)]}])

    if style == "motion_match":
        # The invisible professional cut: no wipe, no flash, just matched blur so
        # two moving shots read as one continuous movement.
        return Recipe(
            a_effects=[{"type": "motion_blur", "frames": 4}],
            b_effects=[{"type": "motion_blur", "frames": 4}])

    if style == "impact_cut":
        return Recipe(
            a_effects=[{"type": "push_in", "from": 1.0, "to": 1.06,
                        "window": [tail, dur_a]}],
            b_effects=[{"type": "shake", "amplitude": 9, "freq": 11,
                        "window": [0.0, min(head * 2, dur_b)]},
                       {"type": "contrast_punch", "contrast": 1.22,
                        "window": [0.0, head]}])

    if style == "speed_ramp_cut":
        return Recipe(
            a_effects=[{"type": "speed_ramp", "from": 1.0, "to": 2.4},
                       {"type": "motion_blur", "frames": 3}],
            b_effects=[{"type": "speed_ramp", "from": 0.55, "to": 1.0}])

    if style == "light_leak_wipe":
        return Recipe(
            a_effects=[{"type": "light_leak", "opacity": 0.5,
                        "window": [tail, dur_a]}],
            b_effects=[{"type": "light_leak", "opacity": 0.45,
                        "window": [0.0, min(head * 2, dur_b)]}],
            blend=True)

    # Anything else is a plain blend with no accents (dissolve, fade, wipes).
    return Recipe(blend=style in BLEND_STYLES)


# Effects that must not stack. Two of these in one chain compound rather than
# combine: measured, a style's motion_blur plus a motion_match transition's own
# gave two chained `tmix` filters averaging ~230ms of movement, and the picture
# came out unreadably smeared. The strongest instance wins instead.
SINGLETON_EFFECTS = {"motion_blur", "echo_trail", "grain", "sharpen", "vignette_pulse"}


def merge_effects(clip: dict, extra: list[dict]) -> dict:
    """Return a copy of `clip` with `extra` effects appended, without stacking.

    Order is preserved for everything that legitimately layers -- two pans
    compose, two grades compose. Only SINGLETON_EFFECTS are collapsed.
    """
    if not extra:
        return clip

    merged: list[dict] = list(clip.get("effects") or [])
    for spec in extra:
        name = spec.get("type") if isinstance(spec, dict) else None
        if name in SINGLETON_EFFECTS:
            existing = next((e for e in merged if e.get("type") == name), None)
            if existing is not None:
                # Keep whichever asks for more, so an explicit heavy request is
                # not silently weakened by a style default arriving later.
                if float(spec.get("frames", spec.get("strength", 0)) or 0) > \
                        float(existing.get("frames", existing.get("strength", 0)) or 0):
                    merged[merged.index(existing)] = spec
                continue
        merged.append(spec)
    return {**clip, "effects": merged}


# ---------------------------------------------------------------- compilation


def _resolve_crop(ctx: Context) -> tuple[str, str]:
    """Base centring plus every geometry term, clamped inside the frame.

    Clamping matters: a shake or pan that runs past the edge does not error, it
    silently pins at the boundary and the movement stops dead. Measured on real
    footage during v2, a pan expression that overran its base image made the
    picture freeze while still appearing to be animated.
    """
    base_x = f"(iw-{ctx.crop_w})/2"
    # Vertically the band is placed on the subject, not on the frame's middle.
    # This is a no-op wherever the crop is already full height -- which is every
    # 9:16-framed reel -- and matters only when a portrait source is being
    # framed into a letterbox strip, where centring lands at waist height.
    centre_y = config.SUBJECT_CENTRE_Y if ctx.subject_y is None else ctx.subject_y
    base_y = (f"(ih*{centre_y:.3f}-{ctx.crop_h}/2)"
              if ctx.crop_h < ctx.source_h else f"(ih-{ctx.crop_h})/2")
    x = "+".join([base_x, *ctx.crop_x_terms])
    y = "+".join([base_y, *ctx.crop_y_terms])
    return (f"clip({x},0,iw-{ctx.crop_w})", f"clip({y},0,ih-{ctx.crop_h})")


def _scale_filters(ctx: Context) -> list[str]:
    """Downscale to delivery, honouring a zoom and a letterbox strip if requested.

    With a strip, the picture is scaled to its real height and *padded* out to
    the delivery frame. The pad is the letterbox: black above and below, made by
    the geometry rather than drawn over the top of it. That is what lets a 16:9
    source arrive whole instead of being punched into a 9:16 window first.
    """
    out_h = ctx.strip_h or config.OUT_H
    bar = (config.OUT_H - out_h) // 2

    if ctx.zoom_expr is None:
        chain = [f"scale={config.OUT_W}:{out_h}:flags=lanczos"]
        if bar > 0:
            chain.append(f"pad={config.OUT_W}:{config.OUT_H}:0:{bar}:black")
        return chain + ["setsar=1"]
    # Even dimensions: chroma-subsampled output cannot take an odd width, and
    # ffmpeg quietly adjusts the plane rather than failing if given one.
    # Clamped: composed zooms multiply, and the product must not drift past the
    # point where the crop has no headroom left and the frame starts upscaling.
    zoom = f"min({ctx.zoom_expr},{ctx.zoom_ceiling})"
    width = f"trunc({config.OUT_W}*{zoom}/2)*2"
    height = f"trunc({out_h}*{zoom}/2)*2"
    chain = [
        f"scale=w='{width}':h='{height}':eval=frame:flags=lanczos",
        f"crop={config.OUT_W}:{out_h}",
    ]
    if bar > 0:
        chain.append(f"pad={config.OUT_W}:{config.OUT_H}:0:{bar}:black")
    return chain + ["setsar=1"]


def _interpolation(ctx: Context) -> list[str]:
    """Synthesise frames when a ramp slows below real time.

    Skipped when nothing asked for slow motion, because at 15.7s per second of
    output this is 30x realtime -- the single most expensive thing available.
    """
    if ctx.slowest_speed >= config.SLOWMO_MIN_SPEED:
        return []
    target = min(int(math.ceil(ctx.fps / max(ctx.slowest_speed, 0.05))),
                 config.SLOWMO_MAX_FPS)
    return [f"minterpolate=fps={target}:mi_mode=mci:mc_mode=aobmc"]


def plan_effects(clip: dict) -> list[tuple[Effect, dict]]:
    """Validate and order a clip's effect specs. Raises before anything renders."""
    specs = clip.get("effects") or []
    if not isinstance(specs, list):
        raise EffectError("'effects' must be a list")

    resolved: list[tuple[Effect, dict]] = []
    for spec in specs:
        if not isinstance(spec, dict) or "type" not in spec:
            raise EffectError(f"each effect needs a 'type': {spec!r}")
        found = REGISTRY.get(spec["type"])
        if found is None:
            raise EffectError(
                f"unknown effect '{spec['type']}'; known: {', '.join(sorted(REGISTRY))}")
        resolved.append((found, spec))

    # Stable within a stage, so two effects at the same stage apply in the order
    # the author wrote them.
    return sorted(resolved, key=lambda pair: pair[0].stage)


def duration_scale(clip: dict) -> float:
    """How much an effect stack multiplies the clip's output duration.

    Almost everything here returns 1.0 by design -- `speed_ramp` normalises to
    its slot and `strobe` refills the frames it drops. `speed_up` genuinely does
    shorten its output, and render.clip_duration must know that or the segment
    gets its frame count pinned to a length the filtergraph cannot produce. That
    is not a cosmetic error: measured, a 1.6x speed_up asked for 30 frames and
    delivered 19, and every cut after it would have slipped.

    Prefer expressing pace through the clip's `in`/`out` against its slot, which
    keeps the reel beat-locked. This exists so that when something does change
    duration, the arithmetic is right rather than optimistic.
    """
    scale = 1.0
    for spec in clip.get("effects") or []:
        if not isinstance(spec, dict):
            continue
        if spec.get("type") == "speed_up":
            scale /= max(float(spec.get("factor", 1.5)), 1e-6)
    return scale


def clip_cost(clip: dict) -> int:
    """Total declared cost, for the budget check in render/director."""
    return sum(int(found.cost) for found, _ in plan_effects(clip))


def heavy_effects(clip: dict) -> list[str]:
    """Names of the effects in this clip that require frame synthesis."""
    names = [found.name for found, _ in plan_effects(clip) if found.cost is Cost.HEAVY]
    ctx = Context(0, 0, 1, 1, 1.0, 1.0)
    for found, spec in plan_effects(clip):
        if found.stage is Stage.TEMPORAL:
            try:
                found.build(spec, ctx)
            except EffectError:
                continue
    if ctx.slowest_speed < config.SLOWMO_MIN_SPEED:
        names.append("slow_motion")
    return names


def build_chain(clip: dict, ctx: Context, label_in: str, label_out: str) -> str:
    """The complete filtergraph for one clip, effects included.

    Returns a graph fragment. `Combine` effects allocate their own intermediate
    labels off `label_out`, so several clips can be compiled into one
    filter_complex without colliding.
    """
    planned = plan_effects(clip)
    staged: dict[Stage, list[Op]] = {stage: [] for stage in Stage}
    for found, spec in planned:
        staged[found.stage].extend(found.build(spec, ctx))

    crop_x, crop_y = _resolve_crop(ctx)
    # An explicit crop_x on the clip wins: that is the subject framing that
    # moments.py measured, and geometry effects offset from it.
    if "crop_x" in clip and not ctx.crop_x_terms:
        crop_x = str(clip["crop_x"])
    elif "crop_x" in clip:
        crop_x = f"clip({clip['crop_x']}+{'+'.join(ctx.crop_x_terms)},0,iw-{ctx.crop_w})"
    if "crop_y" in clip and not ctx.crop_y_terms:
        crop_y = str(clip["crop_y"])

    # Quoted, always. A comma inside an expression is otherwise read as a filter
    # separator -- `clip(x,0,iw-w)` becomes a filter literally named "0" -- and
    # v2 never tripped over it only because its crop expressions had no commas.
    chain: list[str] = [f"crop={ctx.crop_w}:{ctx.crop_h}:'{crop_x}':'{crop_y}'"]
    chain += [op.text for op in staged[Stage.GEOMETRY] if isinstance(op, Filter)]
    chain += _scale_filters(ctx)

    if ctx.needs_pts_reset:
        chain.append("setpts=PTS-STARTPTS")
    chain += _interpolation(ctx)
    chain += [op.text for op in staged[Stage.TEMPORAL] if isinstance(op, Filter)]
    chain.append(f"fps={ctx.fps}")

    chain += [op.text for op in staged[Stage.LOOK] if isinstance(op, Filter)]

    # A COMPOSITE effect may be a plain single-input Filter -- `letterbox` draws
    # two boxes and needs no second stream. Those were being dropped on the floor
    # here: the stage collected only Combine ops, so the effect compiled, landed
    # in the timeline, validated, rendered, and did nothing at all.
    chain += [op.text for op in staged[Stage.COMPOSITE] if isinstance(op, Filter)]

    # Combines need their own graph chains, so the main chain is closed here.
    composites = [op for op in staged[Stage.COMPOSITE] if isinstance(op, Combine)]
    if not composites:
        chain.append(f"format={WORKING_FORMAT}")
        return f"[{label_in}]" + ",".join(chain) + f"[{label_out}]"

    parts = [f"[{label_in}]" + ",".join(chain) + f"[{label_out}_m0]"]
    current = f"{label_out}_m0"
    for index, combine in enumerate(composites):
        aux = f"{label_out}_a{index}"
        nxt = f"{label_out}_m{index + 1}"
        parts.append(f"{combine.source}[{aux}]")
        parts.append(f"[{current}][{aux}]{combine.filter}[{nxt}]")
        current = nxt
    parts.append(f"[{current}]format={WORKING_FORMAT}[{label_out}]")
    return ";".join(parts)
