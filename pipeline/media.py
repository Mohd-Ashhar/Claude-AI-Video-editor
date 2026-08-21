"""Thin, honest wrappers around ffmpeg and ffprobe.

Every stage shells out through here so that failures surface with the actual
ffmpeg stderr rather than a bare non-zero exit code, and so that command
construction stays in one place.
"""

from __future__ import annotations

import json
import subprocess
from fractions import Fraction
from pathlib import Path


class MediaError(RuntimeError):
    """An ffmpeg/ffprobe invocation failed. Carries the stderr tail."""


def run(cmd: list[str], *, desc: str = "", quiet: bool = True) -> subprocess.CompletedProcess:
    """Run a command, raising MediaError with useful context on failure."""
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-15:])
        raise MediaError(f"{desc or cmd[0]} failed (exit {proc.returncode})\n\n{tail}")
    if not quiet and proc.stderr:
        print(proc.stderr)
    return proc


def probe(path: Path | str) -> dict:
    """Full ffprobe JSON for a media file."""
    proc = run(
        [
            "ffprobe", "-v", "error", "-print_format", "json",
            "-show_streams", "-show_format", str(path),
        ],
        desc=f"ffprobe {Path(path).name}",
    )
    return json.loads(proc.stdout)


def video_stream(info: dict) -> dict:
    """Return the first video stream from an ffprobe result."""
    for stream in info.get("streams", []):
        if stream.get("codec_type") == "video":
            return stream
    raise MediaError("no video stream found")


def audio_stream(info: dict) -> dict | None:
    for stream in info.get("streams", []):
        if stream.get("codec_type") == "audio":
            return stream
    return None


def parse_fps(stream: dict) -> float:
    """Frame rate as a float, preferring avg_frame_rate."""
    for key in ("avg_frame_rate", "r_frame_rate"):
        value = stream.get(key)
        if value and value != "0/0":
            try:
                return float(Fraction(value))
            except (ZeroDivisionError, ValueError):
                continue
    return 0.0


def display_dimensions(stream: dict) -> tuple[int, int]:
    """Width and height after applying any rotation side-data.

    Phones and gimbal cameras routinely store a rotated stream with a display
    matrix. The stored width/height are pre-rotation and lie about the framing,
    which would defeat the ingest height gate.
    """
    width, height = int(stream.get("width", 0)), int(stream.get("height", 0))

    rotation = 0
    for side in stream.get("side_data_list", []) or []:
        if "rotation" in side:
            rotation = int(side["rotation"])
            break
    if not rotation:
        tag = (stream.get("tags") or {}).get("rotate")
        if tag:
            rotation = int(tag)

    if abs(rotation) % 180 == 90:
        return height, width
    return width, height


def duration_seconds(info: dict) -> float:
    if fmt_dur := info.get("format", {}).get("duration"):
        return float(fmt_dur)
    stream = video_stream(info)
    if dur := stream.get("duration"):
        return float(dur)
    return 0.0


# ---------------------------------------------------------------- tone

# The reference source this grade was calibrated against, and the picture it was
# calibrated to produce. Both measured, both load-bearing: the calibration is
# only valid for footage that starts where the calibration clip started, and
# nine real clips arrive anywhere from luma 29.6 to 75.0.
GRADE_REF_LUMA = 86.0
GRADE_REF_COOL = -1.4
GRADE_BASE_BRIGHTNESS = -0.10
GRADE_BASE_COOL = 0.10
GRADE_TARGET_LUMA = 37.0
GRADE_TARGET_COOL = 11.0
# What the *subject* should read, not just the frame. Measured on the references
# with a segmentation mask: Gym_1 carries a frame mean of 35.0 and a subject at
# 53.0, Gym_4 40.5 against 84.1 -- the look is separation, a lit body against a
# crushed room, and not merely a dark picture. Grading to the frame mean alone
# reproduced the dark room and took the subject down with it (43.4, a ratio of
# 1.25 against the reference's 1.51), which is exactly the "too dark on my face"
# complaint. Gym_1 is the anchor because that is the style being copied.
# Raised from 53.0 after Reel-2 came back "too dark". 53.0 was Gym_1's subject,
# and Gym_1 is the darkest of the four references -- the measured range across
# them is 53.0-84.1, so the anchor was the bottom of it. What the complaint
# actually tracks is the *subject*, not the frame: Reel-2 delivered a frame of
# 42.1 (brighter than three of four references) with a subject of 53.1, and read
# as murky; Reel-1 delivered 41.2 with a subject of 73.2 and did not. A dark room
# is the look, a dark person is the fault. 66.0 sits mid-range and puts
# separation at 1.78, which is what Reel-1 measured.
GRADE_TARGET_SUBJECT = 66.0
# The mid control point of the curve, and the range it may take. Raising it
# lifts the body out of the shadows without lifting the room, because the toe
# below it stays crushed.
GRADE_BASE_LIFT = 0.62
GRADE_MIN_LIFT = 0.40
GRADE_MAX_LIFT = 0.86
GRADE_LIFT_PIVOT = 0.55
# Measured response: lift 0.52 -> 0.76 moved the subject 45.1 -> 56.8, so about
# 49 luma per unit of lift. Seeded only; the solve measures its own slope.
GRADE_LIFT_GAIN = 49.0
# The lift is solved against the subject/frame *ratio*, not the subject's
# absolute luma. Both controls move the frame mean, so targeting two absolute
# numbers had them fighting: brightness pulled the frame down, the lift pushed
# it back up, and a bright clip settled at 32.0 against a target of 37. A ratio
# barely moves when brightness does, which makes the two axes nearly
# independent. Gym_1 measures 53.0/35.0; Gym_4 84.1/40.5.
GRADE_TARGET_SEPARATION = GRADE_TARGET_SUBJECT / GRADE_TARGET_LUMA
# Seeded slope of separation against lift; the solve measures its own.
GRADE_LIFT_SEP_GAIN = 0.55
# Measured response of the graded picture to its own controls.
GRADE_COOL_GAIN = 81.6
# Luma gain is not constant -- the toe compresses the bottom of the range, so a
# dark picture moves less per unit brightness than a bright one. Measured across
# a brightness sweep: gain/output-luma came out 5.5 at luma 22 and 4.5 at 37.
GRADE_LUMA_GAIN_RATIO = 4.8
# Wide enough for a bright source. A gym shot under daylight measured luma 127
# against the references' 35 -- a 3.6x reduction -- and a +/-0.45 clamp simply
# could not reach it, leaving the clip 10 points bright and the solver reporting
# success. The bound exists to stop a runaway, not to express taste.
GRADE_MIN_BRIGHTNESS = -0.85
GRADE_MAX_BRIGHTNESS = 0.60
# Cool runs both ways. Clamped at zero it can only ever *add* blue, and a source
# that is already blue then has no correction available at all: a bright daylight
# clip graded out at B-R +25 against a target of +11 with the control pinned at
# its floor, and the solver correctly reported that it had done all it could. A
# negative value warms the picture, which is exactly what that clip needed.
GRADE_MIN_COOL = -0.34
GRADE_MAX_COOL = 0.34


def subject_masks(path, start: float = 0.0, span: float = 4.0,
                  count: int = 3) -> list:
    """Where the subject is, on a few frames, as boolean masks at 320x320.

    Derived from the *ungraded* source and reused for every trial grade. The mask
    is a property of the scene, not of the grade -- letting it re-run per trial
    made the measurement move with the thing being measured, and three of five
    candidate curves came back reporting a subject luma of zero because the model
    had simply stopped finding the subject in a picture it had just been shown
    differently.

    Returns an empty list when there is no usable subject, which is a normal
    answer: a detail shot of a barbell has none, and the grade then has only the
    frame mean to work with.
    """
    import numpy as np

    try:
        from pipeline import matte
        session = matte._session(matte.MODEL)
    except Exception:  # noqa: BLE001 - no model, no subject targeting
        return []

    out = []
    for index in range(max(count, 1)):
        when = start + span * (index + 0.5) / max(count, 1)
        proc = subprocess.run(
            ["ffmpeg", "-v", "error", "-ss", f"{when:.3f}", "-i", str(path),
             "-frames:v", "1", "-vf", "scale=320:320",
             "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True)
        if len(proc.stdout) < 320 * 320 * 3:
            continue
        frame = np.frombuffer(proc.stdout[:320 * 320 * 3],
                              dtype=np.uint8).reshape(320, 320, 3)
        mask = matte.infer(session, frame)
        usable, _ = matte.usable(mask)
        if usable and (mask > 0.5).sum() > 200:
            out.append((round(when, 3), mask > 0.5))
    return out


def measure_subject(path, masks: list, extra: str = "") -> float:
    """Mean luma inside the subject masks, under an optional grade."""
    import numpy as np

    if not masks:
        return 0.0
    chain = f"{extra}," if extra else ""
    values = []
    for when, mask in masks:
        proc = subprocess.run(
            ["ffmpeg", "-v", "error", "-ss", f"{when:.3f}", "-i", str(path),
             "-frames:v", "1", "-vf", f"{chain}scale=320:320",
             "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True)
        if len(proc.stdout) < 320 * 320 * 3:
            continue
        frame = np.frombuffer(proc.stdout[:320 * 320 * 3],
                              dtype=np.uint8).reshape(320, 320, 3).astype("float32")
        values.append(float(frame.mean(axis=2)[mask].mean()))
    return float(sum(values) / len(values)) if values else 0.0


def measure_tone(path, start: float = 0.0, span: float = 4.0,
                 extra: str = "") -> tuple[float, float]:
    """Mean luma and blue-minus-red of a span, both 0-255. Cheap, on purpose.

    Eight frames at 108x60. The numbers this feeds are a grade offset, not a
    measurement anyone reads, so precision past a fraction of a luma step buys
    nothing and a full-resolution decode would cost more than the grade does.
    """
    import numpy as np

    chain = f"{extra}," if extra else ""
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", f"{start:.3f}", "-t", f"{span:.3f}",
         "-i", str(path), "-vf", f"{chain}fps=2,scale=108:60",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True)
    frame = 108 * 60 * 3
    count = len(proc.stdout) // frame
    if count < 1:
        return GRADE_REF_LUMA, GRADE_REF_COOL
    arr = np.frombuffer(proc.stdout[:count * frame], dtype=np.uint8)
    arr = arr.reshape(count, 60, 108, 3).astype("float32")
    return float(arr.mean()), float(arr[..., 2].mean() - arr[..., 0].mean())


def grade_chain(brightness: float, cool: float, lift: float = GRADE_BASE_LIFT,
                contrast: float = 1.22, saturation: float = 0.82,
                toe: float = 0.10) -> str:
    """The night grade as a filter string. One definition, used to fit and to render.

    Four control points, not three. The old curve was a toe and nothing else --
    it crushed the shadows and left the rest of the range alone, so the subject
    fell with the room. Adding a mid point at 0.55 lets the body come up while
    the toe holds the background down, which is the whole shape of the reference
    look rather than a brightness setting.
    """
    return (f"eq=brightness={brightness:.4f}:contrast={contrast:.3f}"
            f":saturation={saturation:.3f}"
            f":gamma_b={1.0 + cool:.3f}:gamma_r={1.0 - cool:.3f},"
            f"curves=all='0/0 0.20/{toe:.3f} "
            f"{GRADE_LIFT_PIVOT:.2f}/{lift:.3f} 1/1'")


def fit_subject_y(path, start: float = 0.0, span: float = 4.0,
                  band: float = 1.0, count: int = 7) -> float | None:
    """Where to centre a letterbox band on this clip, 0..1 of source height.

    config.SUBJECT_CENTRE_Y is a median over nine landscape gym clips (range
    0.43-0.76) and is right for those. It is wrong for a framing outside that
    range, and the failure is not subtle: on a 1728x3072 portrait selfie the
    16:9 band keeps 32% of the height, so a band at 0.64 landed on the hands and
    cut the face off entirely for 23% of a reel.

    Measuring the subject's *centre* does not fix that on its own -- the same
    clip measured 0.615, barely different from the default -- because the
    subject is taller than the band can hold. When that is true the band cannot
    contain all of the subject and something has to be chosen, so this returns
    the top of the subject plus half a band: a portrait keeps its face, which is
    what a close or portrait slot is about. When the subject does fit, its own
    centre is used and nothing is cropped away.

    `band` is the band's height as a fraction of source height; at 1.0 the crop
    is full height, this cannot matter, and None is returned. None also means
    "no findable subject", which is a normal answer -- the caller falls back to
    config.SUBJECT_CENTRE_Y.
    """
    import numpy as np

    if band >= 1.0:
        return None

    masks = subject_masks(path, start, span, count=count)
    if not masks:
        return None

    tops, bottoms = [], []
    for _when, mask in masks:
        rows = np.where(np.asarray(mask).any(axis=1))[0]
        if rows.size:
            tops.append(rows.min() / 320.0)
            bottoms.append(rows.max() / 320.0)
    if not tops:
        return None

    top, bottom = float(np.median(tops)), float(np.median(bottoms))
    half = band / 2.0
    centre = (top + half) if (bottom - top) > band else ((top + bottom) / 2.0)
    # Never past the edges: a band hanging off the frame is clamped by the crop
    # expression anyway, but returning an honest number keeps the card readable.
    return round(min(max(centre, half), 1.0 - half), 4)


def fit_grade(path, start: float = 0.0, span: float = 4.0,
              rounds: int = 9, subject: bool = True) -> dict:
    """Find the grade offsets that land *this* clip on the reference look.

    By measurement, not by model, on three axes at once:

        brightness -> the frame's mean luma        (references 34.7-40.5)
        lift       -> the *subject's* mean luma    (references 53.0-84.1)
        cool       -> blue minus red               (references +9 to +13)

    The subject axis is the one that matters most and was missing longest. The
    reference look is separation -- a lit body against a crushed room, Gym_1 at
    35.0 frame against 53.0 subject -- and grading to the frame mean alone
    reproduced the room and took the body down with it, landing a subject at
    43.4 for a ratio of 1.25 against the reference's 1.51. That is visible as
    faces disappearing into shadow, and it is the correct diagnosis of it.

    Closed forms were tried and abandoned. A linear solve under-corrected dark
    clips to 25; a fixed-gain Newton step over-corrected them to 57; a damped
    version converged on gym footage and sent a bright daylight clip's colour to
    +29. `gamma_b` acts multiplicatively and the curve is not linear anywhere, so
    every axis runs a secant instead: two probes establish this clip's own
    response and each later step uses the slope it actually showed. Nothing is
    calibrated to any particular footage.

    Costs one ffmpeg probe per round, plus a handful up front to place the
    subject masks -- roughly a second per clip on a proxy.
    """
    luma, cool_now = measure_tone(path, start, span)
    masks = subject_masks(path, start, span) if subject else []

    brightness = GRADE_BASE_BRIGHTNESS + (GRADE_REF_LUMA - luma) / 255.0
    cool = GRADE_BASE_COOL + (GRADE_REF_COOL - cool_now) / GRADE_COOL_GAIN
    lift = GRADE_BASE_LIFT

    last: dict | None = None
    for _ in range(max(rounds, 1)):
        brightness = _clamp(brightness, GRADE_MIN_BRIGHTNESS, GRADE_MAX_BRIGHTNESS)
        cool = _clamp(cool, GRADE_MIN_COOL, GRADE_MAX_COOL)
        lift = _clamp(lift, GRADE_MIN_LIFT, GRADE_MAX_LIFT)
        chain = grade_chain(brightness, cool, lift)

        out_luma, out_cool = measure_tone(path, start, span, extra=chain)
        if out_luma <= 1.0:
            break
        out_subject = measure_subject(path, masks, chain) if masks else 0.0
        separation = (out_subject / out_luma) if (masks and out_luma > 1.0) else 0.0

        settled = (abs(GRADE_TARGET_LUMA - out_luma) < 0.7
                   and abs(GRADE_TARGET_COOL - out_cool) < 0.7
                   and (not masks
                        or abs(GRADE_TARGET_SEPARATION - separation) < 0.06))
        if settled:
            break

        # Secant where two points exist, seeded gain where they do not. The
        # slope guards matter: a near-zero slope means the control is saturated,
        # and dividing by it throws the solve into a clamp and leaves it there.
        luma_gain = max(GRADE_LUMA_GAIN_RATIO * out_luma, 40.0)
        cool_gain, lift_gain = GRADE_COOL_GAIN, GRADE_LIFT_SEP_GAIN
        if last is not None:
            if abs(brightness - last["b"]) > 1e-4:
                slope = (out_luma - last["luma"]) / (brightness - last["b"])
                if slope > 5.0:
                    luma_gain = slope
            if abs(cool - last["c"]) > 1e-4:
                slope = (out_cool - last["cool"]) / (cool - last["c"])
                if slope > 2.0:
                    cool_gain = slope
            if masks and abs(lift - last["l"]) > 1e-4:
                slope = (separation - last["sep"]) / (lift - last["l"])
                if slope > 0.05:
                    lift_gain = slope
        last = {"b": brightness, "luma": out_luma, "c": cool, "cool": out_cool,
                "l": lift, "subject": out_subject, "sep": separation}

        brightness += _clamp((GRADE_TARGET_LUMA - out_luma) / luma_gain, -0.35, 0.35)
        cool += _clamp((GRADE_TARGET_COOL - out_cool) / cool_gain, -0.15, 0.15)
        if masks:
            lift += _clamp((GRADE_TARGET_SEPARATION - separation) / lift_gain,
                           -0.12, 0.12)

    # Frame luma is the primary target and gets the last word. When the lift
    # saturates -- which it does on any clip whose subject and background share a
    # tonal range, a hand filling a bright frame being the honest example -- it
    # drags the frame mean up with it and the joint solve settles wherever the
    # two balance. A clip that cannot have the separation should still be dark.
    lift = _clamp(lift, GRADE_MIN_LIFT, GRADE_MAX_LIFT)
    brightness = _clamp(brightness, GRADE_MIN_BRIGHTNESS, GRADE_MAX_BRIGHTNESS)

    # Two explicit phases, because they are two different decisions and running
    # them in one loop let the first spend every round the second needed.
    #
    # Phase one: if brightness has bottomed out and the frame is *still* bright,
    # the two controls are pulling against each other. Give up the separation
    # rather than the darkness -- a clip that cannot have a lit subject should
    # still look like it belongs in this reel. The measured case is a hand
    # filling a daylit frame, where subject and background share a tonal range
    # and no curve can separate them anyway.
    for _ in range(6):
        if brightness > GRADE_MIN_BRIGHTNESS + 1e-6 or lift <= GRADE_MIN_LIFT + 1e-6:
            break
        out_luma, _ = measure_tone(path, start, span,
                                   extra=grade_chain(brightness, cool, lift))
        if out_luma <= GRADE_TARGET_LUMA + 0.7:
            break
        lift = _clamp(lift - 0.06, GRADE_MIN_LIFT, GRADE_MAX_LIFT)

    # Phase two: frame luma is the primary target and gets the last word.
    for _ in range(5):
        brightness = _clamp(brightness, GRADE_MIN_BRIGHTNESS, GRADE_MAX_BRIGHTNESS)
        out_luma, _ = measure_tone(path, start, span,
                                   extra=grade_chain(brightness, cool, lift))
        if out_luma <= 1.0 or abs(GRADE_TARGET_LUMA - out_luma) < 0.7:
            break
        gain = max(GRADE_LUMA_GAIN_RATIO * out_luma, 40.0)
        brightness += _clamp((GRADE_TARGET_LUMA - out_luma) / gain, -0.35, 0.35)

    return {"brightness": round(_clamp(brightness, GRADE_MIN_BRIGHTNESS,
                                       GRADE_MAX_BRIGHTNESS), 4),
            "cool": round(_clamp(cool, GRADE_MIN_COOL, GRADE_MAX_COOL), 4),
            "lift": round(lift, 4),
            "subject_seen": bool(masks)}


def _clamp(value: float, low: float, high: float) -> float:
    return max(min(value, high), low)


# ---------------------------------------------------------------- subject dodge

# What a lit subject reads at, and how far the dodge may go. A global curve can
# only take the subject as far as its tonal overlap with the room allows: on a
# clip whose subject started *darker* than the background (59.2 against 64.4)
# the curve's mid-lift saturated at a separation of 1.26 against the reference's
# 1.51. A dodge through the subject matte has no such limit, because it is not a
# tone operation at all -- it is a local one.
DODGE_MAX = 0.22
DODGE_PROBE = 0.10


# Twelve, not five. A reel is a dozen different shots and five samples of it is
# fitting to noise: the same file measured 40.9 subject luma at five samples and
# 66.3 at twelve, which is the difference between "needs a big dodge" and "needs
# none". The probes are 192x108 frame reads; a dozen costs almost nothing.
MATTE_SAMPLES = 12


def measure_through_matte(video, matte_video, samples: int = MATTE_SAMPLES,
                          extra: str = "", band: tuple[int, int] | None = None
                          ) -> tuple[float, float]:
    """Frame luma and subject luma of a rendered reel, using its own matte.

    Reads both files at the same timestamps and uses the matte as the mask, so
    this needs no segmentation of its own -- the expensive pass already ran.

    `band` is the picture strip of a letterboxed reel, and passing it is not
    optional there. Measured over the whole padded frame the black bars drag the
    mean down -- 15.5 against the picture's 36.5 -- which makes the subject look
    four times brighter than its surroundings and sends the dodge chasing a gap
    that does not exist. It over-fired to a subject of 84 against a reference of
    53 before the crop was added.
    """
    import numpy as np

    duration = duration_seconds(probe(video))
    frame_l, subject_l = [], []
    for index in range(max(samples, 1)):
        when = duration * (index + 0.5) / max(samples, 1)
        crop = ""
        if band:
            from pipeline import config
            crop = f"crop={config.OUT_W}:{band[1] - band[0]}:0:{band[0]},"
        chain = f"{extra}," if extra else ""
        pic = subprocess.run(
            ["ffmpeg", "-v", "error", "-ss", f"{when:.3f}", "-i", str(video),
             "-frames:v", "1", "-vf", f"{chain}{crop}scale=192:108",
             "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True)
        msk = subprocess.run(
            ["ffmpeg", "-v", "error", "-ss", f"{when:.3f}", "-i", str(matte_video),
             "-frames:v", "1", "-vf", f"{crop}scale=192:108",
             "-f", "rawvideo", "-pix_fmt", "gray", "-"], capture_output=True)
        if len(pic.stdout) < 192 * 108 * 3 or len(msk.stdout) < 192 * 108:
            continue
        rgb = np.frombuffer(pic.stdout[:192 * 108 * 3],
                            dtype=np.uint8).reshape(108, 192, 3).astype("float32")
        mask = np.frombuffer(msk.stdout[:192 * 108],
                             dtype=np.uint8).reshape(108, 192) > 127
        lum = rgb.mean(axis=2)
        frame_l.append(float(lum.mean()))
        if mask.sum() > 60:
            subject_l.append(float(lum[mask].mean()))
    return (float(np.mean(frame_l)) if frame_l else 0.0,
            float(np.mean(subject_l)) if subject_l else 0.0)


def fit_level(video, matte_video, band: tuple[int, int] | None = None,
              target: float = GRADE_TARGET_LUMA) -> float:
    """A single brightness trim for the finished picture.

    The per-clip grade solves on the whole proxy frame; the reel shows only the
    letterbox strip, and for a portrait source that strip is the middle band --
    a brighter part of the picture than the average it was solved against.
    Measured on a real build, clips that each solved to 37 concatenated to a
    strip reading 47.2.

    Rather than teach merge.py the delivery geometry it has no business knowing,
    the level is trimmed once here, where the actual pixels exist. The per-clip
    grade still does the work that has to be per clip -- separation and colour,
    which differ shot to shot -- and this only sets the overall level.
    """
    frame, _ = measure_through_matte(video, matte_video, band=band)
    if frame <= 1.0 or abs(frame - target) < 1.0:
        return 0.0
    trim = 0.0
    for _ in range(4):
        chain = f"eq=brightness={trim:.4f}" if trim else ""
        frame, _ = measure_through_matte(video, matte_video, extra=chain, band=band)
        if frame <= 1.0 or abs(frame - target) < 0.8:
            break
        gain = max(GRADE_LUMA_GAIN_RATIO * frame, 40.0)
        trim += _clamp((target - frame) / gain, -0.30, 0.30)
        trim = _clamp(trim, -0.60, 0.40)
    return round(trim, 4)


def fit_dodge(video, matte_video, target: float = GRADE_TARGET_SUBJECT,
              band: tuple[int, int] | None = None, level: float = 0.0) -> float:
    """How much to lift the subject so it reads against the room.

    Solved on the rendered reel rather than assumed, and solved *after* the
    per-clip grade rather than instead of it. Splitting it this way is what stops
    the two from fighting: the grade takes the subject as far as a global curve
    can, and this closes whatever gap is left. A fixed dodge on top of a grade
    that had already saturated its lift would have doubled the correction on
    exactly the clips that needed it least.

    Two probes and a secant, because the response is not linear either -- moving
    the control from 0 to 0.06 was worth 175 luma per unit and from 0.06 to 0.10
    about 290.
    """
    # Solved *after* the level trim, or the two fight: a dodge sized against an
    # untrimmed picture over-lifts once the trim takes the level down.
    base = f"eq=brightness={level:.4f}" if level else ""
    _, flat = measure_through_matte(video, matte_video, extra=base, band=band)
    if flat <= 0.0:
        return 0.0                       # no subject anywhere; nothing to dodge
    if flat >= target:
        return 0.0
    probe = (f"{base},"
             if base else "") + f"eq=brightness={DODGE_PROBE}"
    _, lifted = measure_through_matte(video, matte_video, extra=probe, band=band)
    gain = (lifted - flat) / DODGE_PROBE
    if gain <= 1.0:
        return 0.0
    return round(max(min((target - flat) / gain, DODGE_MAX), 0.0), 4)
