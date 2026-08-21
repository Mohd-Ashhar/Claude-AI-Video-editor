"""Shared constants and paths for the reel pipeline.

Every hard number the pipeline depends on lives here so that stages agree and
so that retargeting (external SSD, a different delivery spec) is a config edit
rather than a code change.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------- paths

# PROJECT_ROOT can be overridden so raw footage and intermediates can move to an
# external SSD without touching any stage. See plan: "a config change, not code".
PROJECT_ROOT = Path(os.environ.get("REEL_PROJECT_ROOT", Path(__file__).resolve().parent.parent))

INPUTS_DIR = Path(os.environ.get("REEL_INPUTS_DIR", PROJECT_ROOT / "inputs" / "raw"))
WORK_DIR = Path(os.environ.get("REEL_WORK_DIR", PROJECT_ROOT / "work"))
OUT_DIR = Path(os.environ.get("REEL_OUT_DIR", PROJECT_ROOT / "out"))
ASSETS_DIR = PROJECT_ROOT / "assets"
SCHEMAS_DIR = PROJECT_ROOT / "schemas"
BLUEPRINTS_DIR = PROJECT_ROOT / "blueprints"
STYLES_DIR = PROJECT_ROOT / "styles"

PROXIES_DIR = WORK_DIR / "proxies"
SEGMENTS_DIR = WORK_DIR / "segments"
FEATURES_DIR = WORK_DIR / "features"

SOURCES_JSON = WORK_DIR / "sources.json"
SHOTS_JSON = WORK_DIR / "shots.json"
MOMENTS_JSON = WORK_DIR / "moments.json"
SIGNALS_JSON = WORK_DIR / "signals.json"
TAGS_JSON = WORK_DIR / "tags.json"
CLIP_CARDS_JSON = WORK_DIR / "clip_cards.json"
MUSIC_MAP_JSON = WORK_DIR / "music_map.json"
BRIEF_JSON = WORK_DIR / "brief.json"
CAST_JSON = WORK_DIR / "cast.json"
LYRICS_JSON = WORK_DIR / "lyrics.json"
TIMELINE_JSON = WORK_DIR / "timeline.json"

# ---------------------------------------------------------------- delivery spec

OUT_W = 1080
OUT_H = 1920
OUT_FPS = 30

# Delivery container. QuickTime .mov is what the rest of this workflow speaks --
# the Pocket 3 exports it, the iPhone grade exports it, and it AirDrops back to
# Photos without a conversion step. The video and audio streams are byte-identical
# either way; only the wrapper differs, so this costs nothing in quality.
# Instagram accepts both.
OUT_CONTAINER = os.environ.get("REEL_OUT_CONTAINER", ".mov")


def out_path(name: str, directory: Path | None = None) -> Path:
    """Delivery path for `name` (no extension), in the configured container."""
    return (directory or OUT_DIR) / f"{name}{OUT_CONTAINER}"
TARGET_LUFS = -14.0
TARGET_TRUE_PEAK = -1.5
TARGET_LRA = 11.0

# Instagram Reels UI overlay, measured in delivery pixels. Subject framing and
# any on-screen text must stay inside the interior these leave behind.
SAFE_TOP = 250
SAFE_BOTTOM = 400
SAFE_RIGHT = 200
SAFE_LEFT = 60


def safe_box() -> tuple[int, int, int, int]:
    """Return (x, y, w, h) of the UI-clear interior in delivery pixels."""
    x = SAFE_LEFT
    y = SAFE_TOP
    return x, y, OUT_W - SAFE_LEFT - SAFE_RIGHT, OUT_H - SAFE_TOP - SAFE_BOTTOM


# ---------------------------------------------------------------- ingest gate

# A 9:16 crop from a 16:9 source uses height * 9/16 pixels of width. To fill
# OUT_W without upscaling, the source must be at least this tall:
#     height * 9/16 >= 1080  ->  height >= 1920
# A 1080p source yields only 607px of crop width and lands visibly soft.
MIN_SOURCE_HEIGHT = OUT_H


def crop_window(width: int, height: int,
                aspect: float | None = None) -> tuple[int, int]:
    """Largest `aspect` window inside a WxH frame, rounded to even dimensions.

    The single definition of this geometry: ingest gates on its width and render
    crops to it, so the two can never disagree about whether a source is sharp
    enough. Even dimensions matter because chroma-subsampled footage cannot be
    cropped to an odd width without ffmpeg quietly adjusting the plane -- and 4K
    lands on exactly that case, since 2160 * 9/16 is 1215.

    `aspect` defaults to the delivery frame's 9:16. A letterboxed reel passes the
    *strip's* ratio instead, and that is not a refinement -- it is the difference
    between using a landscape shot and throwing it away. Measured on a 3840x2160
    source destined for a 16:9 strip: cropping to 9:16 and then drawing bars
    leaves 1214x683 visible, **10% of the frame**, a 3.2x centre punch-in nobody
    asked for. Cropping to 16:9 instead leaves all of it.
    """
    ratio = aspect if aspect else OUT_W / OUT_H
    crop_w = int(min(width, height * ratio)) & ~1
    crop_h = min(int(crop_w / ratio) & ~1, height & ~1)
    return crop_w, crop_h


def strip_height(ratio: float) -> int:
    """Picture height for a reel letterboxed to `ratio`, or 0 if it fills the frame."""
    strip = int(round(OUT_W / ratio))
    strip -= strip % 2
    return strip if 0 < strip < OUT_H else 0


# Where the subject's centre actually sits, vertically, as a fraction of frame
# height. Measured on nine real gym clips: 0.43-0.76, median 0.64 -- below
# centre, not above. This only matters when a source has vertical latitude to
# give, which in practice means a portrait clip being framed into a 16:9 strip;
# centring that band puts it at waist height and cuts the head off.
SUBJECT_CENTRE_Y = 0.64

# The reel's tonal arc: shot luma as a ratio of the reel's own mean, against
# position through the reel. Measured over the four Gym-Inspiration references,
# 84 shots pooled into deciles of reel position.
#
# This exists because `media.fit_grade()` is a *normaliser*. It solves every clip
# to the same frame luma so that a shoot spanning 45 points of luma cuts
# together, and it is right to do that -- but nothing then put deliberate
# variation back, so the reel came out flat. Measured on a real build: per-shot
# luma spanned 0.86-1.09x of the reel mean against the references' 0.72-1.42x,
# and a viewer reads a constant mid-dark level with no bright relief as "too
# dark" even when the mean is *higher* than the references (42.1 against
# 32.3-38.2). The fix is not exposure, it is contrast across time.
#
# The shape is the references' own: open a little under, a bright peak across the
# first third, then a long tail below the mean. Applied on top of the solved
# grade, so normalisation still happens first and this only redistributes.
GRADE_ARC = (
    (0.05, 0.87), (0.15, 1.13), (0.25, 1.34), (0.35, 1.39), (0.45, 0.98),
    (0.55, 0.89), (0.65, 1.01), (0.75, 0.84), (0.85, 0.88), (0.95, 0.85),
)


def grade_arc(position: float) -> float:
    """Luma ratio for a shot at `position` (0..1) through the reel."""
    points = GRADE_ARC
    if position <= points[0][0]:
        return points[0][1]
    if position >= points[-1][0]:
        return points[-1][1]
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x0 <= position <= x1:
            span = x1 - x0
            return y0 + (y1 - y0) * ((position - x0) / span if span else 0.0)
    return 1.0


# Colour transfer characteristics that are NOT ready for delivery. This ffmpeg
# build has no zscale, so these must be tonemapped with the colorspace filter.
HDR_TRANSFERS = {"arib-std-b67", "smpte2084"}

VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".MP4", ".MOV", ".M4V"}

# ---------------------------------------------------------------- proxy

PROXY_HEIGHT = 480
PROXY_BITRATE = "2M"

# ---------------------------------------------------------------- features

# Samples per second of the dense analysis pass. Measured on this machine:
# 4.1 ms of optical flow + 1.1 ms of Laplacian per sample, so a 5-minute file
# costs ~15s at 8 Hz. 8 Hz also resolves a 2-second window to a quarter of a
# second, which is the finest cut placement moments.py can act on anyway.
SAMPLE_FPS = 8.0
FLOW_WIDTH = 320        # optical flow runs downscaled; magnitude is normalised out
CLIP_HIGH = 250         # luma above this counts as a blown highlight
CLIP_LOW = 5            # luma below this counts as a crushed shadow

# ---------------------------------------------------------------- moments

# A reel shot. Below 2s there is no time to read the frame; above 8s no reel
# slot will ever use the whole thing.
MOMENT_MIN_SECONDS = 2.0
MOMENT_MAX_SECONDS = 8.0
MOMENT_DURATION_STEP = 0.5
MOMENT_HOP = 0.25           # window search stride
MOMENT_MIN_GAP = 0.5        # dead space required between two accepted moments
MAX_MOMENTS_PER_FILE = 8
MOMENT_SECONDS_PER_CANDIDATE = 6.0   # density cap: one moment per this much take

# Drop moments scoring this far below the best window in the same file. An
# absolute margin rather than a ratio, because window scores already sit on a
# bounded scale: footage that is uniformly decent keeps everything, while a
# genuinely soft or shaky stretch is discarded instead of consuming a VLM call
# and a reel slot.
#
# Deliberately no padding floor above 1: if a file genuinely holds only one
# strong moment, emitting its weaker windows to fill slots would put footage on
# screen that was already judged bad. sequence.py handles a short candidate pool
# by building a shorter reel and saying so, which is the honest failure.
MOMENT_SCORE_DROP = 0.15
MOMENT_KEEP_FLOOR = 1

# How quickly motion becomes interesting. Interest saturates rather than peaking
# at a target magnitude: more movement is more engaging, with diminishing
# returns, and whether it is *good* movement is judged separately by smoothness.
#
# This replaced a bell curve centred on an "ideal" magnitude, which could not be
# made to work. Measured across real footage, shooting styles differ by 3-4x --
# gym handheld runs a median of 2.6-5.0 while locked-off cooking close-ups run
# 0.8-1.0 -- so any single ideal magnitude scores one of them badly. Tuned for
# gym footage it called the cooking clips lifeless; tuned for cooking it threw
# out 68% of the gym footage as too violent, and the reel came back full of
# empty rooms because an empty room holds still.
#
# The bell's real job was rejecting shake, and smoothness_curve() measures shake
# directly and far better, so nothing was lost by removing it.
MOTION_SATURATION = 1.5     # magnitude at which interest reaches ~0.49
MOTION_STILL = 0.12         # below this the frame is treated as locked off

# Motion required before the crop may be pointed at a moving subject.
#
# Well above MOTION_STILL, and for a different reason. A near-static scene still
# contains movement -- a hand cracking an egg -- and the tracker finds it
# correctly, then frames the hand while the pan that the shot is actually about
# sits outside the crop. Measured: cooking close-ups run a median of 0.8-1.0 and
# reframing them was wrong every time, while gym footage runs 2.6-5.0 and
# reframing was right. Below this line, centre the crop and accept it.
SUBJECT_MIN_MOTION = 1.6

# A window containing any sample this bad is unusable no matter how good its mean.
FRAME_SCORE_FLOOR = 0.22

# Mid-tone the exposure score treats as correct, and how hard it punishes drift.
#
# Not 118 (mid-grey). This footage is graded D-Log M -> Mimo -> iPhone and lands
# deliberately dark: measured means of 33-72 across real clips, with clip_low at
# essentially zero. Nothing is crushed -- the look is a choice -- yet a mid-grey
# target docked every clip 25-30% for having it. Clipping at either end is the
# real exposure defect and is scored separately, so drift now carries less weight
# than it used to.
EXPOSURE_TARGET_LUMA = 70.0
EXPOSURE_DRIFT_WEIGHT = 0.25

# Histogram distance that counts as the content genuinely changing. Absolute,
# not a within-take z-score, for the same reason stability_score is absolute:
# measured on this build, a real cut scores 0.23-0.28 while grain and lighting
# drift inside a continuous take never exceed 0.05. Normalising against local
# variance instead made grain read as a full-strength cut in a uniform take.
NOVELTY_CUT = 0.18

# Seconds at each end of a moment whose motion decides transitions downstream.
MOMENT_EDGE_SECONDS = 0.4

# ---------------------------------------------------------------- reel

TARGET_REEL_SECONDS = 22.0
MIN_REEL_SECONDS = 12.0
MAX_REEL_SECONDS = 35.0

# Fraction of boundaries allowed a *visible* transition. A transition on every
# boundary is still the clearest tell of an amateur edit, so this stays a hard
# cap -- but the accounting changed with v3. Most of the vocabulary now realises
# a transition as a hard cut with accents rather than a blend, and `motion_match`
# is invisible by design, so it is exempt. The budget therefore buys more
# density than the same number did in v2 while restraining the same thing.
TRANSITION_BUDGET = 0.40

# What a directional whip has to prove before it may fire. Set from the 60th
# percentile of moving samples in real footage, so roughly the faster half of
# real camera moves qualify -- the 25% budget then decides how many actually fire.
#
# WHIP_MIN_STABILITY is the one that stops the worst failure: shake is spatially
# coherent -- every pixel really does move together -- so coherence alone would
# happily whip out of a shot that was rejected for being shaky.
WHIP_MIN_MAGNITUDE = 3.10
WHIP_MIN_COHERENCE = 0.55
WHIP_MIN_STABILITY = 0.50
WHIP_MAX_ANGLE_DEGREES = 40.0
FLASH_MIN_EXIT = 5.50

# ---------------------------------------------------------------- effects

# Zoom beyond the crop's own headroom upscales. On 4K the 9:16 window is 1214 px
# wide against a 1080 px delivery, so 1.124x is free; past that a push-in trades
# a little sharpness for the punch, which is usually the right trade over the
# fifth of a second one lasts.
MAX_ZOOM = 1.35

# A spin needs far more zoom than a tilt to keep its corners out of a portrait
# frame, and gets its own ceiling because it lasts a fifth of a second and is
# usually blurred, so the upscale is invisible in a way a held shot's would not be.
SPIN_MAX_ZOOM = 1.60

# Below this speed a retimed shot judders and needs synthesised frames. That
# costs 15.7s per second of output -- 30x realtime, measured -- so it is capped
# per reel rather than offered freely.
SLOWMO_MIN_SPEED = 0.92
SLOWMO_MAX_FPS = 120
SLOWMO_BUDGET_PER_REEL = 3

# ---------------------------------------------------------------- casting

# Absolute flow-magnitude bands a blueprint's `motion` requirement is matched
# against. Measured across the first real project -- gym moments span 0.82-6.32
# and cooking 0.59-2.09 -- so these cover both shooting styles with the boundary
# between "low" and "medium" sitting in the gap between them.
#
# Absolute, and that is the whole point. signals.motion_energy is a percentile
# rank within the batch, so it always runs 0.00 to 1.00 whatever was shot; asking
# it for a high-motion shot in a batch of locked-off tripod shots returns a
# locked-off tripod shot. These bands read motion_raw instead.
MOTION_BANDS = {
    "still":  (0.00, 0.50),
    "low":    (0.50, 1.50),
    "medium": (1.50, 3.00),
    "high":   (3.00, 99.0),
}

# Telling a deliberate camera move from a shaky one. Coherence is the fraction of
# flow that moves together, so a pan scores high and a subject crossing a locked
# frame scores low -- measured 0.49-0.96 across the real project.
CAMERA_MOVING_MAGNITUDE = 1.20
CAMERA_COHERENT = 0.70

# Requirement fit below which a slot counts as unfilled. A must_have slot under
# this floor fails the build instead of accepting the least-bad candidate, which
# is exactly how v3 opened a reel on a wall of dumbbells.
CAST_FIT_FLOOR = 0.45

# ---------------------------------------------------------------- render

# ProRes 422 LT via the hardware encoder: all-intra, so concat is trivially
# safe, and fast enough that per-segment re-rendering stays cheap.
INTERMEDIATE_CODEC = ["-c:v", "prores_videotoolbox", "-profile:v", "1", "-pix_fmt", "yuv422p10le"]

# Set from the Phase 3 VMAF bake-off. "x264" is the conservative default.
FINAL_ENCODER = os.environ.get("REEL_FINAL_ENCODER", "x264")

# Colour tagging is fiddly and verified by experiment on this ffmpeg build:
# -color_primaries/-color_trc alone write only the matrix, leaving primaries and
# transfer unset. An untagged file washes out on Instagram, so each encoder gets
# the route that actually sticks -- x264 via its own VUI params, VideoToolbox via
# a metadata bitstream filter. qa.py fails the render if any of the three is lost.
_X264_VUI = "colorprim=bt709:transfer=bt709:colormatrix=bt709"
_H264_BSF = "h264_metadata=colour_primaries=1:transfer_characteristics=1:matrix_coefficients=1"
_HEVC_BSF = "hevc_metadata=colour_primaries=1:transfer_characteristics=1:matrix_coefficients=1"

FINAL_ENCODERS = {
    "x264": [
        "-c:v", "libx264", "-profile:v", "high", "-level", "4.2",
        "-preset", "slow", "-crf", "19",
        "-x264-params", f"keyint={OUT_FPS * 2}:min-keyint={OUT_FPS * 2}:scenecut=0:{_X264_VUI}",
    ],
    "vt_h264": [
        "-c:v", "h264_videotoolbox", "-b:v", "20M", "-maxrate", "24M", "-bufsize", "32M",
        "-bsf:v", _H264_BSF,
    ],
    "vt_hevc": [
        "-c:v", "hevc_videotoolbox", "-b:v", "20M", "-maxrate", "24M", "-bufsize", "32M",
        "-tag:v", "hvc1", "-bsf:v", _HEVC_BSF,
    ],
    # Not a delivery codec. This is what the picture render writes when something
    # still has to be composited onto it -- burned-in lyrics, most of the time.
    # Without it the chain was two lossy generations: a 20 Mb/s hardware draft,
    # re-encoded to 6.9 Mb/s x264, with the first pass's artifacts preserved and
    # a second pass laid on top. ProRes here makes the burn the only lossy step.
    "prores": [
        "-c:v", "prores_videotoolbox", "-profile:v", "3", "-pix_fmt", "yuv422p10le",
    ],
}

# Encoders whose output is an intermediate, not something to post.
INTERMEDIATE_ENCODERS = {"prores"}

COLOR_TAGS = [
    "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
    "-color_range", "tv",
]

# ---------------------------------------------------------------- preflight

MIN_FREE_RAM_GB = 3.0
MIN_FREE_DISK_GB = 40.0

# Filters and encoders every stage assumes. Checked once, up front, by name --
# this ffmpeg build is missing several things the research plan took for granted.
REQUIRED_FILTERS = ["scale", "crop", "xfade", "overlay", "colorspace", "loudnorm", "sidechaincompress"]
REQUIRED_ENCODERS = ["h264_videotoolbox", "prores_videotoolbox", "libx264", "aac"]

# Present in a full ffmpeg build, absent here. Stages must route around them.
OPTIONAL_FILTERS = ["subtitles", "ass", "drawtext", "zscale"]
