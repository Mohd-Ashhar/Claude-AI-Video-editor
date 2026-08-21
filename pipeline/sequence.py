"""Assemble ranked moments into a beat-locked reel.

The decisions this stage makes, in the order it makes them:

  1. how long each slot is        -- a pacing arc, quantised to the beat grid
  2. which moment fills each slot -- an assignment problem, solved optimally
  3. what plays next to what      -- a variety pass over the result
  4. where the cut lands          -- trimmed around each moment's peak instant
  5. how two shots are joined     -- a hard cut, unless the measured camera
                                     motion earns something else

Point 5 is the one worth defending. It is tempting to put a whip pan on every
boundary, and it is the clearest tell of an amateur edit. A transition here has
to be justified by what the camera actually did -- a whip only fires when clip A
genuinely exits panning the same way clip B enters -- and even then a hard budget
caps how many boundaries may be anything other than a straight cut.

    uv run python -m pipeline.sequence --variants 3
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Callable

import numpy as np

from pipeline import config, effects, media, render

BOLD, RED, YELLOW, GREEN, DIM, RESET = "\033[1m", "\033[31m", "\033[33m", "\033[32m", "\033[2m", "\033[0m"

BEATS_PER_BAR = 4

# Widened from [2,3,4,6,8]. v2's slots came out 1.50, 1.50, 1.50, 2.00, 2.00,
# 2.00, 3.00, 3.00, 2.00, 2.00, 1.50 -- a 2x range with no stabs and no holds,
# which is most of why the reels read as a slideshow. Professional edits swing
# from a third of a second to four.
ALLOWED_BEATS = [1, 2, 3, 4, 6, 8, 12]
MIN_SLOT_SECONDS = 0.30

# A burst is the one thing that legitimately breaks the floor above. Measured on
# the Gym-Inspiration reels: three of the four contain a machine-gun run of four
# to seven cuts at 0.07-0.23s each -- Gym_3 goes down to two frames -- and it is
# the hardest pattern interrupt any of them use. The floor exists so the *arc*
# never produces a slideshow; a burst is a deliberate single gesture, declared
# by the blueprint, and it is exempt.
BURST_MIN_FRAMES = 2
BURST_MAX_FRAMES = 7
BURST_MIN_CUTS = 3
BURST_MAX_CUTS = 8
# The longest slot a burst can cover: 8 cuts of 7 frames. Measured burst runs in
# the references total 1.3-1.6s, so this is the right neighbourhood. A slot
# longer than this cannot be a burst without becoming a thirteen-cut strobe, and
# the honest answer there is to leave it as an ordinary shot.
BURST_MAX_SECONDS = BURST_MAX_CUTS * BURST_MAX_FRAMES / config.OUT_FPS
MAX_SLOT_SECONDS = 4.0
TARGET_SLOT_SECONDS = 1.7

# Per-slot multipliers laid over the arc, cycled. A smooth arc alone produces
# neighbouring slots of near-identical length; a phrase pattern is what makes a
# reel feel like it has a rhythm rather than a duration.
PHRASE_PATTERNS = {
    "standard": [1.25, 0.55, 0.55, 1.10, 0.60, 1.45, 0.50, 0.50],
    "fast":     [0.85, 0.45, 0.45, 0.95, 0.40, 0.40, 1.15, 0.45],
    "breathe":  [1.50, 0.70, 1.30, 0.60, 1.70, 0.65, 0.65, 1.20],
}

# Assignment costs. CANNOT_FILL is large enough to never be chosen over any real
# option, but finite so the solver still returns a complete assignment when the
# candidate pool is too thin -- a short reel beats no reel.
CANNOT_FILL = 50.0
W_HOOK = 1.2
W_ROLE = 0.4
W_ENERGY = 0.3

# How much assignment cost may be traded for a transition opportunity. Small on
# purpose: a matching whip is worth a slightly weaker shot in that slot, and
# never worth a materially worse one.
MAX_TRANSITION_SWAP_COST = 0.35

WHIP_MAX_ANGLE = math.radians(config.WHIP_MAX_ANGLE_DEGREES)

# Cut-style accents carry a nominal duration too: it sizes the window their
# effects occupy on either side of the boundary, even though no frames are shared.
TRANSITION_SECONDS = {"whip": 0.20, "flash": 0.12, "dissolve": 0.40,
                      "punch": 0.22, "cut": 0.14}

ARCS = {
    # (floor, peak height, peak position, width)
    "standard": (0.80, 0.65, 0.60, 0.28),
    "fast":     (0.72, 0.38, 0.55, 0.32),
    "breathe":  (0.88, 0.85, 0.62, 0.24),
}


# ---------------------------------------------------------------- slot planning


def arc_weight(u: float, shape: str) -> float:
    """Relative slot length at position u (0..1) through the reel.

    Fast open, one longer shot to breathe around 60%, quick finish. A reel of
    uniform shot lengths reads as a slideshow no matter how good the footage.
    """
    floor, height, centre, width = ARCS[shape]
    return floor + height * math.exp(-(((u - centre) / width) ** 2))


def beat_time(beats: list[float], index: int) -> float:
    """Beat time by index, extrapolating past the end of the track."""
    if not beats:
        return 0.0
    if 0 <= index < len(beats):
        return beats[index]
    period = (beats[-1] - beats[0]) / max(len(beats) - 1, 1)
    if index >= len(beats):
        return beats[-1] + period * (index - len(beats) + 1)
    return beats[0] + period * index


def on_frame(seconds: float) -> float:
    """Snap a boundary to the output frame grid.

    Each segment renders independently to a whole number of frames, so a
    boundary that falls mid-frame is rounded by the encoder -- and those
    roundings accumulate. Measured before this snap: 0.112s of drift across a
    22s reel, which is three frames of slip by the final cut. Snapping the
    boundaries instead caps the error at half a frame and stops it growing.

    Module level rather than nested in plan_slots because brief.py lays a
    blueprint onto the same grid and has to snap it identically; two
    implementations of this would drift apart silently.
    """
    return round(seconds * config.OUT_FPS) / config.OUT_FPS


def allowed_multiples(period: float, ceiling: float | None = None) -> list[int]:
    """Beat multiples that land on a sane shot length at this tempo.

    A fixed set breaks at the extremes: 2 beats is 0.69s at 174 BPM (a flash,
    not a shot) and 8 beats is 5.3s at 90 BPM (an eternity in a reel).

    `ceiling` is the longest moment actually available. A slot longer than any
    moment on hand cannot be filled, and a slot that renders short does not just
    look wrong -- it shifts every cut after it off the beat.
    """
    limit = min(MAX_SLOT_SECONDS, ceiling) if ceiling else MAX_SLOT_SECONDS
    usable = [m for m in ALLOWED_BEATS if MIN_SLOT_SECONDS <= m * period <= limit]
    if usable:
        return usable
    # Nothing fits: fall back to the shortest multiple that a moment can cover.
    smallest = min(ALLOWED_BEATS)
    return [smallest] if not ceiling or smallest * period <= ceiling else [1]


def quantise(weights: list[float], target_beats: int, usable: list[int]) -> list[int]:
    """Turn arc weights into beat multiples summing to exactly target_beats.

    Exactness matters: the reel has to end on a downbeat, and a bar of drift is
    the difference between a loop that lands and one that stumbles.
    """
    total_weight = sum(weights)
    if total_weight <= 0:
        return [usable[0]] * len(weights)

    ideal = [w / total_weight * target_beats for w in weights]
    chosen = [min(usable, key=lambda m: abs(m - value)) for value in ideal]

    # Repair to the exact total, always moving the slot that minds least.
    guard = 0
    while sum(chosen) != target_beats and guard < 500:
        guard += 1
        up = sum(chosen) < target_beats
        best, best_penalty = None, None
        for i, current in enumerate(chosen):
            options = [m for m in usable if (m > current if up else m < current)]
            if not options:
                continue
            nxt = min(options) if up else max(options)
            penalty = abs(nxt - ideal[i]) - abs(current - ideal[i])
            if best_penalty is None or penalty < best_penalty:
                best, best_penalty = (i, nxt), penalty
        if best is None:
            break
        chosen[best[0]] = best[1]
    return chosen


def plan_slots(music: dict | None, target: float, count: int, shape: str,
               longest_moment: float | None = None) -> list[dict]:
    """Slot boundaries, taken from the beat grid itself rather than computed.

    Durations are differences of real beat times, so quantisation error cannot
    accumulate across the reel -- the last cut is as tightly on the beat as the
    first.
    """
    if music and music.get("beats"):
        beats = music["beats"]
        period = music["beat_period"] or (beats[-1] - beats[0]) / max(len(beats) - 1, 1)
        start = music.get("best_start", beats[0])
        origin = min(range(len(beats)), key=lambda i: abs(beats[i] - start))
    else:
        period = TARGET_SLOT_SECONDS / 4
        beats = [i * period for i in range(int(target / period) + 64)]
        origin = 0

    usable = allowed_multiples(period, longest_moment)
    target_beats = max(int(round(target / period)), BEATS_PER_BAR)
    target_beats = max(int(round(target_beats / BEATS_PER_BAR)) * BEATS_PER_BAR, BEATS_PER_BAR)

    count = max(1, min(count, target_beats // min(usable)))

    # With too few moments to fill the target, aim at what the slots can
    # actually hold instead of a length they cannot reach -- otherwise the
    # quantiser gives up mid-repair and the reel stops landing on whole bars.
    reachable = count * max(usable)
    if reachable < target_beats:
        target_beats = max((reachable // BEATS_PER_BAR) * BEATS_PER_BAR, min(usable) * count)

    pattern = PHRASE_PATTERNS.get(shape, PHRASE_PATTERNS["standard"])
    weights = [arc_weight(i / max(count - 1, 1), shape) * pattern[i % len(pattern)]
               for i in range(count)]
    multiples = quantise(weights, target_beats, usable)

    slots: list[dict] = []
    cumulative = 0
    for index, beats_long in enumerate(multiples):
        t0 = on_frame(beat_time(beats, origin + cumulative))
        t1 = on_frame(beat_time(beats, origin + cumulative + beats_long))
        cumulative += beats_long
        slots.append({
            "index": index,
            "beats": beats_long,
            # Six decimals: see the note in brief.lay_out. A frame-grid interval
            # is not always representable in four.
            "duration": round(t1 - t0, 6),
            "offset": round(t0 - on_frame(beat_time(beats, origin)), 6),
            "on_downbeat": (origin + cumulative - beats_long) % BEATS_PER_BAR
                           == origin % BEATS_PER_BAR,
            # Short slots want energy, long slots want something worth looking at.
            "energy": round(1.0 - min(beats_long / max(max(usable), 1), 1.0), 3),
        })
    return slots


# ---------------------------------------------------------------- assignment


def diversity_key(card: dict) -> str:
    """What must not repeat back to back.

    Content class when tagging ran; otherwise the source file, which is a
    weaker but still real proxy for "more of the same".
    """
    tags = card.get("tags") or {}
    return tags.get("content_class") or Path(card["source"]).stem


def slot_cost(card: dict, slot: dict, last: int) -> float:
    if card["duration"] < slot["duration"]:
        return CANNOT_FILL + (slot["duration"] - card["duration"])

    tags = card.get("tags") or {}
    cost = 1.0 - card["rank_score"]

    hook = (tags.get("hook_score", 5) - 1) / 9.0
    role = tags.get("reel_role")

    if slot["index"] == 0:
        # The opening shot is chosen on different terms from every other slot:
        # nothing else in the reel matters if this one does not stop the scroll.
        cost -= W_HOOK * hook
        if role == "opener":
            cost -= W_ROLE
        elif role in ("filler", "closer"):
            cost += W_ROLE
    elif slot["index"] == last:
        if role == "closer":
            cost -= W_ROLE
        elif role == "opener":
            cost += W_ROLE * 0.5
    else:
        if role in ("build", "peak"):
            cost -= W_ROLE * 0.5
        elif role == "filler":
            cost += W_ROLE * 0.5

    motion = card["signals"].get("motion_energy", 0.5)
    cost += W_ENERGY * abs(motion - slot["energy"])
    return cost


def assign(cards: list[dict], slots: list[dict]) -> list[int]:
    from scipy.optimize import linear_sum_assignment

    last = len(slots) - 1
    matrix = np.array([[slot_cost(card, slot, last) for slot in slots] for card in cards])
    rows, cols = linear_sum_assignment(matrix)

    chosen = [-1] * len(slots)
    for row, col in zip(rows, cols):
        chosen[col] = int(row)
    return chosen


def count_clashes(picked: list[dict]) -> int:
    """Adjacent pairs showing the same kind of thing."""
    return sum(1 for i in range(len(picked) - 1)
               if diversity_key(picked[i]) == diversity_key(picked[i + 1]))


def repair_variety(picked: list[dict], slots: list[dict],
                   rounds: int = 8) -> tuple[list[dict], int]:
    """Reorder shots so no two neighbours show the same kind of thing.

    Run after the optimal assignment rather than inside it: adjacency is a
    constraint *between* slots, which an assignment problem cannot express.

    The objective is the total number of clashing pairs, not the local one being
    looked at. A pass that only fixes the clash in front of it happily creates
    two more behind it, which is exactly what an earlier version of this did on
    footage where one source file supplied half the moments -- and half from one
    file is the normal case, not an edge case, when the input is a handful of
    long takes. Ties break on assignment cost, so variety never costs more
    quality than it has to.
    """
    fixed = 0
    last = len(slots) - 1

    for _ in range(rounds):
        current = count_clashes(picked)
        if current == 0:
            break

        best = None
        # Slot 0 is the hook and was chosen on different terms; leave it alone.
        for i in range(1, len(picked)):
            for j in range(i + 1, len(picked)):
                if picked[i]["duration"] < slots[j]["duration"]:
                    continue
                if picked[j]["duration"] < slots[i]["duration"]:
                    continue

                trial = picked[:]
                trial[i], trial[j] = trial[j], trial[i]
                after = count_clashes(trial)
                if after >= current:
                    continue

                delta = (slot_cost(picked[j], slots[i], last)
                         + slot_cost(picked[i], slots[j], last)
                         - slot_cost(picked[i], slots[i], last)
                         - slot_cost(picked[j], slots[j], last))
                if best is None or (after, delta) < best[0]:
                    best = ((after, delta), i, j)

        if best is None:
            break
        _, i, j = best
        picked[i], picked[j] = picked[j], picked[i]
        fixed += 1

    return picked, fixed


# ---------------------------------------------------------------- trimming


def trim(card: dict, duration: float) -> dict:
    """Take `duration` seconds out of a moment, centred on its strongest instant.

    Trimming from the start would be simpler and would routinely cut away the
    exact frame the window was selected for.
    """
    peak = card.get("peak", (card["start"] + card["end"]) / 2)
    latest = max(card["end"] - duration, card["start"])
    start = min(max(peak - duration / 2, card["start"]), latest)
    # Snap the in-point to the frame grid so the clip's length stays an exact
    # frame multiple; slot durations already are, and a mid-frame in-point would
    # put the rounding back that plan_slots just took out.
    start = round(start * config.OUT_FPS) / config.OUT_FPS
    clip = {"source": card["source"], "in": round(start, 6),
            "out": round(min(start + duration, card["end"]), 6)}

    crop_x = subject_crop_x(card)
    if crop_x is not None:
        clip["crop_x"] = crop_x
    if card.get("subject_y") is not None:
        clip["subject_y"] = card["subject_y"]
    return clip


def is_moving(card: dict) -> bool:
    """Whether the camera or subject is actually moving, in absolute terms.

    This used to read `motion_energy < 0.45`, and that was wrong for the same
    reason cast.py cannot use it: motion_energy is a percentile rank *within the
    batch*, so it always spans 0.00 to 1.00 whatever was shot. On a batch of
    locked-off tripod shots it declared the top half "already moving" and
    withheld the push-in they all needed; on a batch shot entirely handheld it
    declared the bottom half static and added a push-in to a moving camera,
    which is the one thing the function's own docstring says not to do.

    `motion_raw` is absolute flow magnitude, and the boundary is the same one
    cast.py matches requirements against.
    """
    signals = card.get("signals") or {}
    raw = signals.get("motion_raw")
    if raw is None:
        # Cards from before merge.py carried motion_raw. The edge measurements
        # are absolute too, so they are a correct if coarser answer.
        edges = [card.get("entry_flow"), card.get("exit_flow")]
        edges = [e for e in edges if e]
        raw = (sum(e.get("magnitude", 0.0) for e in edges) / len(edges)) if edges else 0.0
    return float(raw) >= config.MOTION_BANDS["medium"][0]


def style_effects(card: dict, slot: dict, index: int, total: int) -> list[dict]:
    """Give every shot deliberate movement and a consistent finish.

    The other half of why v2 looked basic: `crop_x` supported expressions and the
    sequencer never emitted a single one, so every shot sat locked off even where
    2625 px of pan latitude existed. Here nothing is static by default.

    The choice is driven by what was measured, not by variety for its own sake.
    A shot that is already moving gets motion blur and is otherwise left alone --
    adding a push-in to a moving camera reads as a mistake. A locked-off shot gets
    the movement it lacks.
    """
    fx: list[dict] = []
    duration = slot["duration"]
    stab = duration <= 0.9

    if not is_moving(card):
        # Locked off: give it a move. Alternated so consecutive shots do not all
        # push in, which reads as a zoom loop rather than an edit.
        if index % 2 == 0:
            fx.append({"type": "push_in", "from": 1.0,
                       "to": 1.18 if stab else 1.09})
        else:
            fx.append({"type": "pan", "pixels": 170 if index % 4 == 1 else -170})
    else:
        fx.append({"type": "motion_blur", "frames": 3})

    if stab:
        # Stabs are too short to read on their own; a punch makes them land.
        fx.append({"type": "contrast_punch", "contrast": 1.16, "saturation": 1.10,
                   "window": [0.0, min(0.12, duration)]})

    if index == 0:
        # The hook gets the strongest treatment: it decides whether the rest is seen.
        fx.append({"type": "exposure_pump", "strength": 0.16,
                   "window": [0.0, min(0.18, duration)]})

    # Grain last, over everything: it ties clips from different files together
    # and hides the banding this three-generation grading chain produces.
    fx.append({"type": "grain", "strength": 6})
    return fx


# Effect treatment per story role, as intensity multipliers. The hook and the
# payoff are the two shots that decide whether a reel is watched and whether it
# was worth watching, so they get the strongest handling; everything between them
# is deliberately quieter, because a reel where every shot shouts has no payoff.
ROLE_INTENSITY = {"hook": 1.30, "promise": 0.85, "build": 1.00,
                  "payoff": 1.45, "cta": 0.80}

FLAVOURS = {"standard": 1.00, "hype": 1.25, "calm": 0.75}


def role_effects(shot: dict, card: dict, index: int,
                 flavour: str = "standard") -> list[dict]:
    """The effect stack for a briefed shot, from its story role and its footage.

    Two inputs, both necessary. The role says how much to push -- a payoff earns
    treatment a build has not -- and the measurement says which direction to push
    in, because adding a push-in to a moving camera reads as a mistake no matter
    what the story wants there.
    """
    scale = ROLE_INTENSITY.get(shot["role"], 1.0) * FLAVOURS.get(flavour, 1.0)
    duration = shot["duration"]
    role = shot["role"]
    fx: list[dict] = []

    if is_moving(card):
        fx.append({"type": "motion_blur", "frames": 3})
    elif role == "cta":
        # Pull out on the closer: it releases the reel rather than driving into it.
        fx.append({"type": "pull_out", "from": 1.0 + 0.10 * scale, "to": 1.0})
    elif shot["camera"] in ("pan", "whip", "follow", "orbit"):
        # The brief asked for lateral movement and the footage has none, so put
        # it back with the crop rather than silently dropping the intent.
        fx.append({"type": "pan", "pixels": int(190 * scale) * (1 if index % 2 else -1)})
    else:
        fx.append({"type": "push_in", "from": 1.0,
                   "to": min(1.0 + 0.11 * scale, config.MAX_ZOOM)})

    if role == "hook":
        fx.append({"type": "exposure_pump", "strength": round(0.16 * scale, 3),
                   "window": [0.0, min(0.18, duration)]})
        fx.append({"type": "contrast_punch", "contrast": round(1.0 + 0.16 * scale, 3),
                   "saturation": round(1.0 + 0.10 * scale, 3),
                   "window": [0.0, min(0.14, duration)]})
    elif role == "payoff":
        fx.append({"type": "contrast_punch", "contrast": round(1.0 + 0.14 * scale, 3),
                   "saturation": round(1.0 + 0.12 * scale, 3),
                   "window": [0.0, min(0.20, duration)]})
        fx.append({"type": "vignette_pulse", "strength": round(0.30 * scale, 3),
                   "window": [0.0, min(0.45, duration)]})
    elif duration <= 0.9:
        fx.append({"type": "contrast_punch", "contrast": 1.16, "saturation": 1.10,
                   "window": [0.0, min(0.12, duration)]})

    fx.append({"type": "grain", "strength": 6})
    return fx


def subject_crop_x(card: dict) -> int | None:
    """Point the 9:16 window at the subject instead of the middle of the frame.

    A stand-in for reframe.py, which is not built. Centre-cropping a 16:9 frame
    throws away 2625 px of width and routinely cuts the subject in half, because
    people do not stand in the middle of the frame -- on this footage the crop
    missed the subject entirely on several shots.

    The subject is placed slightly left of centre rather than dead centre, since
    Instagram's action buttons cover roughly the right 200 px of the delivered
    frame.
    """
    position = card.get("subject_x")
    if position is None:
        return None

    width = int(card.get("width", 0))
    crop_w, _ = config.crop_window(width, int(card.get("height", 0)))
    if not width or crop_w >= width:
        return None

    # Where inside the crop the subject should land, in crop-relative terms.
    bias = 0.5 - (config.SAFE_RIGHT - config.SAFE_LEFT) / (2.0 * config.OUT_W)
    left = position * width - crop_w * bias
    return int(round(min(max(left, 0), width - crop_w)))


# ---------------------------------------------------------------- transitions


def _angle_between(a: dict, b: dict) -> float:
    return abs(math.atan2(a["dy"], a["dx"]) - math.atan2(b["dy"], b["dx"]))


def propose_transition(a: dict, b: dict, on_downbeat: bool) -> dict | None:
    """What, if anything, these two shots' measured motion justifies.

    Returns None for a hard cut, which is the right answer most of the time.
    """
    exit_flow = a.get("exit_flow")
    entry_flow = b.get("entry_flow")
    if not exit_flow or not entry_flow:
        return None

    angle = _angle_between(exit_flow, entry_flow)
    angle = min(angle, 2 * math.pi - angle)

    strong = (exit_flow["magnitude"] >= config.WHIP_MIN_MAGNITUDE
              and entry_flow["magnitude"] >= config.WHIP_MIN_MAGNITUDE)
    coherent = (exit_flow["coherence"] >= config.WHIP_MIN_COHERENCE
                and entry_flow["coherence"] >= config.WHIP_MIN_COHERENCE)
    # Shake is spatially coherent -- every pixel does move together -- so
    # coherence alone would happily whip out of a shot rejected for shaking.
    steady = (a["signals"].get("stability_score", 1.0) >= config.WHIP_MIN_STABILITY
              and b["signals"].get("stability_score", 1.0) >= config.WHIP_MIN_STABILITY)

    if strong and coherent and steady and angle <= WHIP_MAX_ANGLE:
        # dx is optical flow: positive means content travels right across the
        # frame, which is what a leftward camera pan looks like. The whip is
        # named for the direction the picture moves, so it matches the footage.
        horizontal = abs(exit_flow["dx"]) >= abs(exit_flow["dy"])
        if horizontal:
            style = "whip_right" if exit_flow["dx"] > 0 else "whip_left"
        else:
            style = "whip_up"
        strength = min(exit_flow["magnitude"], entry_flow["magnitude"]) \
            * min(exit_flow["coherence"], entry_flow["coherence"])
        return {"style": style, "duration": TRANSITION_SECONDS["whip"],
                "strength": round(float(strength), 4), "reason": "matched camera motion"}

    # Both moving, but not in agreement: matched motion blur across a plain cut.
    # The invisible professional join — no wipe, nothing to notice, and it reads
    # as one continuous movement instead of two shots bolted together.
    if strong and steady:
        return {"style": "motion_match", "duration": TRANSITION_SECONDS["cut"],
                "strength": round(min(exit_flow["magnitude"],
                                      entry_flow["magnitude"]) * 0.4, 4),
                "reason": "both shots moving — matched blur across the cut"}

    if on_downbeat and exit_flow["magnitude"] >= config.FLASH_MIN_EXIT:
        if entry_flow["magnitude"] <= config.MOTION_STILL * 2:
            return {"style": "flash_cut", "duration": TRANSITION_SECONDS["flash"],
                    "strength": round(float(exit_flow["magnitude"]) * 0.5, 4),
                    "reason": "motion into stillness on a downbeat"}
        return {"style": "impact_cut", "duration": TRANSITION_SECONDS["flash"],
                "strength": round(float(exit_flow["magnitude"]) * 0.45, 4),
                "reason": "hard landing on a downbeat"}

    if on_downbeat and abs(exit_flow["magnitude"] - entry_flow["magnitude"]) > 1.5:
        return {"style": "zoom_punch", "duration": TRANSITION_SECONDS["punch"],
                "strength": 0.5, "reason": "energy step change on a downbeat"}

    if exit_flow["magnitude"] <= config.MOTION_STILL \
            and entry_flow["magnitude"] <= config.MOTION_STILL:
        if diversity_key(a) == diversity_key(b):
            return {"style": "dissolve", "duration": TRANSITION_SECONDS["dissolve"],
                    "strength": 0.2, "reason": "two still shots of the same subject"}
        return {"style": "glitch_cut", "duration": TRANSITION_SECONDS["flash"],
                "strength": 0.25, "reason": "two static shots — a stab to break them up"}

    return None


def transition_value(picked: list[dict], slots: list[dict]) -> float:
    """Total strength of every transition this ordering makes possible."""
    return sum(
        proposal["strength"]
        for index in range(len(picked) - 1)
        for proposal in [propose_transition(picked[index], picked[index + 1],
                                            slots[index + 1]["on_downbeat"])]
        if proposal
    )


def encourage_transitions(picked: list[dict], slots: list[dict],
                          max_swaps: int = 2) -> tuple[list[dict], int]:
    """Reorder slightly so shots whose motion matches end up next to each other.

    Without this, transition selection is a passive observer: it reports which
    of the orderings it was handed happen to contain a matching pair, and the
    ordering was decided with no knowledge that matching pairs are valuable.
    Measured on the test set, three moments were individually eligible to start
    a whip and none ever landed adjacent to another, so the feature could not
    fire at all.

    Deliberately timid -- a couple of swaps, never worsening variety, and only
    for a bounded loss in assignment cost. A reel reordered until it is all
    whips is the amateur edit the budget exists to prevent.
    """
    made = 0
    ceiling = count_clashes(picked)
    last = len(slots) - 1

    for _ in range(max_swaps):
        current = transition_value(picked, slots)
        best = None

        # Slot 0 is the hook; it stays where it is.
        for a in range(1, len(picked)):
            for b in range(a + 1, len(picked)):
                if picked[a]["duration"] < slots[b]["duration"]:
                    continue
                if picked[b]["duration"] < slots[a]["duration"]:
                    continue

                trial = picked[:]
                trial[a], trial[b] = trial[b], trial[a]
                if count_clashes(trial) > ceiling:
                    continue

                gain = transition_value(trial, slots) - current
                if gain <= 1e-6:
                    continue

                delta = (slot_cost(picked[b], slots[a], last)
                         + slot_cost(picked[a], slots[b], last)
                         - slot_cost(picked[a], slots[a], last)
                         - slot_cost(picked[b], slots[b], last))
                if delta > MAX_TRANSITION_SWAP_COST:
                    continue
                if best is None or (-gain, delta) < best[0]:
                    best = ((-gain, delta), a, b)

        if best is None:
            break
        _, a, b = best
        picked[a], picked[b] = picked[b], picked[a]
        made += 1

    return picked, made


def choose_transitions(picked: list[dict], slots: list[dict]) -> dict[int, dict]:
    """Rank every justified transition, then keep only what the budget allows.

    Two rules beyond the budget: transitions may not be adjacent (a transition
    segment consumes both of its shots, so back-to-back ones would fight over
    the same clip), and each needs spare footage to extend into.
    """
    boundaries = len(picked) - 1
    if boundaries < 1:
        return {}

    allowance = max(int(round(boundaries * config.TRANSITION_BUDGET)), 1)

    candidates = []
    for index in range(boundaries):
        proposal = propose_transition(picked[index], picked[index + 1],
                                      slots[index + 1]["on_downbeat"])
        if proposal:
            candidates.append((index, proposal))

    candidates.sort(key=lambda item: -item[1]["strength"])

    accepted: dict[int, dict] = {}
    visible = 0
    for index, proposal in candidates:
        style = proposal["style"]
        blend = style in effects.BLEND_STYLES

        # Only blend transitions consume both of their clips, so only they can
        # collide with a neighbour. A cut-style accent touches one boundary and
        # nothing else, which is why they can sit back to back.
        if blend and ((index - 1) in accepted or (index + 1) in accepted):
            continue

        # motion_match is a plain hard cut with matched blur — there is nothing
        # to see, so it does not spend from a budget that exists to stop visible
        # transition spam.
        if style != "motion_match":
            if visible >= allowance:
                continue
            visible += 1

        accepted[index] = proposal
    return accepted


# ---------------------------------------------------------------- timeline


def _overlap_pair(a_card: dict, a_slot: dict, b_card: dict, b_slot: dict,
                  duration: float) -> tuple[dict, dict] | None:
    """Trim two clips that overlap by `duration` yet still occupy both slots.

    A transition segment outputs `A + B - duration`, so one side has to supply
    the overlap or the reel loses that much time at every transition and every
    later cut drifts off the beat. Either side will do, so try both before
    giving up -- a moment trimmed close to its own length has no room to extend,
    and refusing on the first failure throws away transitions the other side
    could have paid for.
    """
    for extend_a in (True, False):
        head = trim(a_card, a_slot["duration"] + (duration if extend_a else 0.0))
        tail = trim(b_card, b_slot["duration"] + (0.0 if extend_a else duration))
        wanted_a = a_slot["duration"] + (duration if extend_a else 0.0)
        wanted_b = b_slot["duration"] + (0.0 if extend_a else duration)
        # trim() clamps to the moment's bounds, so a short moment silently
        # returns less than asked. Confirm the extension actually happened.
        if (head["out"] - head["in"] >= wanted_a - 1e-3
                and tail["out"] - tail["in"] >= wanted_b - 1e-3):
            return head, tail
    return None


def build_timeline(picked: list[dict], slots: list[dict], transitions: dict[int, dict],
                   music: dict | None, track: Path | None,
                   dresser: Callable[[dict, dict, int, int], list[dict]] | None = None
                   ) -> tuple[dict, list[str], dict[int, dict]]:
    """Emit segments, preserving every slot boundary exactly.

    Returns the transitions actually applied, which is not always the set
    proposed: one may be dropped for want of footage, and reporting the proposal
    would describe a cut the viewer will not see.

    `dresser` supplies each shot's effect stack. It is a parameter rather than a
    hard call to style_effects so the briefed path can dress by story role
    instead -- applying both would compose two push-ins into one double zoom and
    stack two contrast punches, since neither is a singleton effect.
    """
    dress = dresser or style_effects
    segments: list[dict] = []
    notes: list[str] = []
    applied: dict[int, dict] = {}
    joins: list[dict] = []
    pending_accents: list[dict] = []
    index = 0

    while index < len(picked):
        card, slot = picked[index], slots[index]
        proposal = transitions.get(index)

        blend = proposal and proposal["style"] in effects.BLEND_STYLES

        if proposal and blend and index + 1 < len(picked):
            pair = _overlap_pair(card, slot, picked[index + 1], slots[index + 1],
                                 proposal["duration"])
            if pair is None:
                notes.append(f"boundary {index}: neither shot has footage to spare for a "
                             f"{proposal['style']} — cut instead")
            else:
                head, tail = pair
                # Both halves get dressed too. They were skipped entirely before,
                # so whenever a blend fired, two of the reel's shots quietly lost
                # their movement and their grain while every shot around them kept
                # theirs -- visible as a flat patch, and hard to attribute.
                head = effects.merge_effects(
                    effects.merge_effects(head, pending_accents),
                    dress(card, slot, index, len(picked)))
                tail = effects.merge_effects(
                    tail, dress(picked[index + 1], slots[index + 1],
                                index + 1, len(picked)))
                pending_accents = []
                segments.append({
                    "kind": "transition",
                    "style": proposal["style"],
                    "duration": proposal["duration"],
                    "a": {**head, "exit_flow": card.get("exit_flow")},
                    "b": tail,
                })
                applied[index] = proposal
                joins.append({"segment": len(segments) - 1, "style": proposal["style"],
                              "blend": True})
                index += 2
                continue

        shot = trim(card, slot["duration"])
        shot = effects.merge_effects(shot, pending_accents)
        pending_accents = []

        # A cut-style transition is a hard cut with accents on either side. No
        # blend, no shared segment, no footage borrowed -- which is why it never
        # has to be dropped for want of handles, and why the timeline length is
        # untouched. This is how professional edits get density without looking
        # like a transition pack.
        if proposal and not blend and index + 1 < len(picked):
            recipe = effects.transition_recipe(
                proposal["style"], proposal["duration"], slot["duration"],
                slots[index + 1]["duration"], card.get("exit_flow"))
            shot = effects.merge_effects(shot, recipe.a_effects)
            pending_accents = recipe.b_effects
            applied[index] = proposal
            # The join is the boundary *after* this segment, which is where the
            # next one begins -- that is the frame a whoosh or an impact lands on.
            joins.append({"segment": len(segments), "style": proposal["style"],
                          "blend": False})

        shot = effects.merge_effects(shot, dress(card, slot, index, len(picked)))
        segments.append({"kind": "shot", **shot})
        index += 1

    timeline = {
        "version": 1,
        "width": config.OUT_W,
        "height": config.OUT_H,
        "fps": config.OUT_FPS,
        "segments": segments,
        # Where the transitions ended up, by segment. Most of the vocabulary is
        # realised as a hard cut with accents rather than a blend, so a consumer
        # cannot find them by looking for `kind == "transition"` -- sound.py did
        # exactly that and placed two cues in a reel with six joins. Recorded
        # here so the audio can land on the same boundaries the picture uses.
        "joins": joins,
    }
    if track:
        timeline["audio"] = {"music": str(track)}
    if music:
        timeline["sync"] = {
            "track": music.get("track"),
            "tempo": music.get("tempo"),
            "best_start": music.get("best_start"),
            "bar_seconds": music.get("bar_seconds"),
        }
    return timeline, notes, applied


# ---------------------------------------------------------------- variants


def hook_strength(card: dict) -> float:
    """How well a card opens a reel, using the tag when there is one."""
    score = (card.get("tags") or {}).get("hook_score")
    return score / 10.0 if score else card["rank_score"]


def assemble(cards: list[dict], music: dict | None, target: float, count: int,
             shape: str, hook: int, track: Path | None) -> dict:
    """One complete assembly.

    `hook` picks the nth-best opening shot, which is what makes the variants
    genuinely different rather than three shuffles of the same edit: the first
    two seconds decide whether the rest is seen at all.
    """
    longest = max((card["duration"] for card in cards), default=None)
    slots = plan_slots(music, target, count, shape, longest)

    pool = list(cards)
    forced = None
    ordered = sorted(pool, key=hook_strength, reverse=True)
    if 0 <= hook < len(ordered):
        forced = ordered[hook]
        pool = [card for card in pool if card is not forced]

    body_slots = slots[1:] if forced else slots
    chosen = assign(pool, body_slots) if (body_slots and pool) else []

    picked: list[dict] = []
    used: list[dict] = []
    short = 0

    def take(card: dict | None, slot: dict) -> None:
        """Accept a pairing only if the moment can actually fill the slot.

        A moment shorter than its slot renders short, and every cut after it
        lands off the beat -- silently, because nothing else in the chain knows
        the slot was underfilled. Dropping the slot instead costs a shot and
        keeps the grid, and slot lengths are whole beat multiples so what remains
        is still beat-aligned.
        """
        nonlocal short
        if card is None:
            return
        if card["duration"] + 1e-6 < slot["duration"]:
            short += 1
            return
        picked.append(card)
        used.append(slot)

    if forced:
        take(forced, slots[0])
    for slot, card_index in zip(body_slots, chosen):
        take(pool[card_index] if card_index >= 0 else None, slot)

    if not picked:
        return {}

    picked, fixed = repair_variety(picked, used)
    picked, encouraged = encourage_transitions(picked, used)
    proposed = choose_transitions(picked, used)
    timeline, notes, transitions = build_timeline(picked, used, proposed, music, track)

    if short:
        notes.append(f"{short} slot(s) dropped — no moment was long enough to fill them")

    return {
        "timeline": timeline,
        "picked": picked,
        "slots": used,
        "transitions": transitions,
        "variety_fixes": fixed,
        "motion_swaps": encouraged,
        "notes": notes,
        "shape": shape,
        "hook": hook,
    }


def brief_slots(brief: dict) -> list[dict]:
    """Brief shots in the shape the rest of this module already speaks.

    The brief has already done the quantising, the frame snapping and the
    downbeat bookkeeping -- it laid itself over the same grid using the same
    helpers -- so this is a translation, not a second planning pass.
    """
    total = len(brief["shots"])
    return [{
        "index": shot["index"],
        "beats": shot["beats"],
        "duration": shot["duration"],
        "offset": shot["start"],
        "on_downbeat": bool(shot.get("on_downbeat")),
        "energy": round(1.0 - min(shot["beats"] / 12.0, 1.0), 3),
        "role": shot["role"],
        "is_last": shot["index"] == total - 1,
    } for shot in brief["shots"]]


def burst_frames(duration: float, cuts: int) -> list[int]:
    """Split a slot into sub-slots of whole frames that sum to it exactly.

    Whole frames and an exact sum, both non-negotiable. The render spine
    guarantees segment lengths are frame multiples and that their concat equals
    the timeline; a burst that came up a frame short would put back the
    sub-frame drift the whole pipeline exists to avoid.

    So the *cut count* bends, not the total. The requested count is clamped to
    what the slot can actually carry at 2-7 frames a piece -- a 1.6s slot cannot
    be four cuts without each running 12 frames, which is no longer a burst.
    Getting this backwards was a real bug: capping the frames instead of the
    cuts turned a 1.2s slot into 1.167s and a 1.6s slot into 0.933s, and every
    downstream length assertion would have been measuring a timeline that no
    longer matched the music.

    Lengths are deliberately uneven. An evenly divided burst reads as a strobe;
    the references cut unevenly inside the run.
    """
    total = max(int(round(duration * config.OUT_FPS)), BURST_MIN_CUTS * BURST_MIN_FRAMES)
    floor_cuts = -(-total // BURST_MAX_FRAMES)          # ceil
    ceil_cuts = total // BURST_MIN_FRAMES
    cuts = max(min(cuts, ceil_cuts, BURST_MAX_CUTS), floor_cuts, BURST_MIN_CUTS)
    cuts = min(cuts, ceil_cuts) or 1

    base, extra = divmod(total, cuts)
    lengths = [base] * cuts
    # Spend the remainder on alternating slots so the run has a shape rather
    # than one odd frame tacked onto the end.
    order = sorted(range(cuts), key=lambda i: (i % 2, i))
    for step in range(extra):
        lengths[order[step % cuts]] += 1
    return lengths


def expand_bursts(picked: list[dict], slots: list[dict], shots: list[dict]
                  ) -> tuple[list[dict], list[dict], list[dict], int]:
    """Turn any slot whose blueprint shot declares `burst` into a run of cuts.

    The sub-slots all draw from the same moment at staggered in-points, which is
    what the references do: a burst is one action seen in rapid fragments, not
    eight unrelated shots. Cards are shallow-copied with a shifted peak so
    trim() lands each fragment somewhere different inside the take.
    """
    out_cards: list[dict] = []
    out_slots: list[dict] = []
    out_shots: list[dict] = []
    count = 0

    for card, slot, shot in zip(picked, slots, shots):
        spec = shot.get("burst")
        if spec and slot["duration"] > BURST_MAX_SECONDS + 1e-6:
            # Too long to cut this hard. Left alone rather than quietly turned
            # into a thirteen-cut strobe -- a blueprint asking for a burst on a
            # three-second slot has asked for the wrong thing, and the note in
            # the timeline says so.
            spec = None
        if not spec:
            out_cards.append(card)
            out_slots.append(slot)
            out_shots.append(shot)
            continue

        lengths = burst_frames(slot["duration"], int(spec.get("cuts", 5)))
        cuts = len(lengths)
        span = card["end"] - card["start"]
        offset = slot.get("offset", 0.0)
        cursor = 0

        for step, frames in enumerate(lengths):
            share = card["start"] + span * (step + 0.5) / cuts
            fragment = {**card, "peak": share}
            out_cards.append(fragment)
            out_slots.append({
                **slot,
                "index": f"{slot['index']}b{step}",
                "duration": round(frames / config.OUT_FPS, 6),
                "offset": round(offset + cursor / config.OUT_FPS, 6),
                "on_downbeat": step == 0,
                "burst": True,
                # One inverted fragment, in the middle of the run. gym_3 puts it
                # at 6.15s of a burst that spans 6.03-7.33, i.e. one fragment in
                # from the front -- early enough to be part of the gesture rather
                # than the thing that ends it.
                "negate": bool(spec.get("negate")) and step == max(cuts // 3, 1),
            })
            out_shots.append({**shot, "burst_step": step, "text": None})
            cursor += frames
        count += 1

    return out_cards, out_slots, out_shots, count


def assemble_briefed(cards: list[dict], brief: dict, cast_result: dict,
                     music: dict | None, track: Path | None,
                     flavour: str = "standard",
                     cues: dict[int, dict] | None = None) -> dict:
    """Build the timeline the brief asked for, from the moments cast into it.

    Unlike assemble(), nothing here chooses an order or a length: the story did
    that before the footage existed. This resolves each slot to its cast moment,
    trims it, dresses it by role, and drops any slot that could not be filled --
    reporting the drop rather than substituting, because a payoff filled by the
    least-bad wide shot is exactly the failure the brief exists to prevent.
    """
    by_id = {card["shot_id"]: card for card in cards}
    slots_all = brief_slots(brief)
    cues = cues or {}

    picked: list[dict] = []
    used: list[dict] = []
    shots: list[dict] = []
    dropped: list[str] = []

    for entry, slot, shot in zip(cast_result["slots"], slots_all, brief["shots"]):
        card = by_id.get(entry.get("shot_id") or "")
        if not entry["filled"] or card is None:
            dropped.append(entry["id"])
            continue
        if card["duration"] + 1e-6 < slot["duration"]:
            dropped.append(entry["id"])
            continue
        picked.append(card)
        used.append(slot)
        shots.append(shot)

    if not picked:
        return {}

    # A lyric track and per-shot captions cannot share a frame. Measured on a
    # real build: the CTA card "Send this to someone who needs it" landed on top
    # of the word GET, and the hook block sat over another -- two text systems
    # competing for the same centre of frame. The reference reel this style comes
    # from carries lyric words and nothing else.
    lyric_mode = brief.get("text_mode") == "lyrics"
    letterbox = brief.get("letterbox")
    grade = brief.get("grade")

    picked, used, shots, bursts = expand_bursts(picked, used, shots)
    skipped_bursts = sum(1 for s in brief["shots"] if s.get("burst")) - bursts

    # Where each slot's midpoint falls through the reel, for the tonal arc.
    # Taken from the slot durations rather than the index: a burst is seven
    # sub-slots inside one slot's worth of time, and counting slots would put its
    # fragments across half the arc.
    _elapsed, _starts = 0.0, []
    for _slot in used:
        _starts.append(_elapsed)
        _elapsed += float(_slot["duration"])
    _reel = _elapsed or 1.0
    _midpoint = [(_starts[i] + float(used[i]["duration"]) / 2) / _reel
                 for i in range(len(used))]

    def dress(card: dict, slot: dict, index: int, total: int) -> list[dict]:
        """Dress by story role, and hang this shot's text cue on it."""
        shot = shots[index]
        stack = role_effects(shot, card, index, flavour)
        # Grade first in the stack, before anything composited on top of it.
        # Ordering inside a stage is preserved by the compiler, and a letterbox
        # bar that got graded would stop being black.
        if grade == "night":
            # The card's own solved offsets, when merge.py measured them. Without
            # them this falls back to the swept defaults, which are correct only
            # for footage that starts where the calibration clip started.
            solved = {k: v for k, v in (card.get("grade") or {}).items()
                      if k in ("brightness", "cool", "lift")}

            # The tonal arc, on top of the normalised grade. fit_grade() lands
            # every clip on the same luma so the shoot cuts together; this puts
            # the references' own variation back, or the reel reads flat. The
            # step is (ratio - 1) / gain and carries no luma term because
            # d(luma)/d(brightness) is proportional to luma -- so the same
            # brightness step is the same *ratio* on every clip, which is only
            # true because they were all normalised to one level first.
            ratio = config.grade_arc(_midpoint[index])
            if abs(ratio - 1.0) > 1e-3:
                base = solved.get("brightness", media.GRADE_BASE_BRIGHTNESS)
                solved["brightness"] = round(max(min(
                    base + (ratio - 1.0) / media.GRADE_LUMA_GAIN_RATIO,
                    media.GRADE_MAX_BRIGHTNESS), media.GRADE_MIN_BRIGHTNESS), 4)
            stack.insert(0, {"type": "night_grade", **solved})
        if slot.get("negate"):
            stack.append({"type": "negate_flash", "at": 0.0, "frames": 3})
        if letterbox:
            stack.append({"type": "letterbox", "ratio": float(letterbox)})
        if not lyric_mode and (cue := cues.get(shot["index"])):
            stack.append(cue)
        return stack

    # No variety repair and no reordering: the blueprint's order *is* the story.
    # repair_variety() exists to stop a ranked montage showing the same thing
    # twice, which is a problem a brief does not have.
    proposed = choose_transitions(picked, used)
    timeline, notes, transitions = build_timeline(picked, used, proposed,
                                                 music, track, dress)

    if dropped:
        notes.append(f"{len(dropped)} slot(s) dropped for want of footage: "
                     f"{', '.join(dropped)}")
    if bursts:
        notes.append(f"{bursts} burst slot(s) expanded into rapid cuts")
    if skipped_bursts > 0:
        notes.append(f"{skipped_bursts} burst slot(s) left whole — longer than "
                     f"{BURST_MAX_SECONDS:.2f}s is a strobe, not a burst")

    timeline["story"] = {
        "blueprint": brief["blueprint"],
        "flavour": flavour,
        "shots": [s["id"] for s in shots],
        "dropped": dropped,
    }

    return {
        "timeline": timeline,
        "picked": picked,
        "slots": used,
        "transitions": transitions,
        "variety_fixes": 0,
        "motion_swaps": 0,
        "notes": notes,
        "shape": brief["blueprint"],
        "hook": 0,
        "flavour": flavour,
        "dropped": dropped,
    }


def _shot_clips(timeline: dict) -> list[dict]:
    """Every clip in the timeline, in playback order.

    A transition segment holds two clips, so the flat list of segments is not the
    list of shots -- and the effect stacks have to line up with the brief's shots
    one for one or the payoff's treatment lands on the wrong picture.
    """
    clips: list[dict] = []
    for segment in timeline["segments"]:
        if segment["kind"] == "shot":
            clips.append(segment)
        else:
            clips.extend([segment["a"], segment["b"]])
    return clips


def sync_sheet(music: dict | None, timeline: dict, variants: list[dict]) -> str:
    lines = ["REEL SYNC SHEET", "=" * 52, ""]
    if music:
        start = music["best_start"]
        lines += [
            f"reference track : {music['track']}",
            f"tempo           : {music['tempo']:.1f} BPM  ({music['bar_seconds']:.3f}s per bar)",
            f"start the audio : {int(start // 60)}:{start % 60:06.3f}  "
            f"({start:.3f}s into the track)",
            "",
            "In Instagram: add the sound, open its trim control, and move the start",
            "to the timecode above. The reel opens on a downbeat and runs a whole",
            "number of bars, so a small offset reads as a choice rather than a mistake.",
        ]
    else:
        lines.append("no reference track — cuts follow a fixed pacing arc, not a beat grid")

    lines += ["", "variants", "-" * 52]
    for index, variant in enumerate(variants, start=1):
        total = sum(render.segment_duration(s) for s in variant["timeline"]["segments"])
        cuts = len(variant["picked"])
        lines.append(f"  v{index}  {total:6.2f}s  {cuts:2d} shots  "
                     f"{len(variant['transitions'])} transitions  ({variant['shape']} arc)")
    return "\n".join(lines) + "\n"


def edit_list(variant: dict) -> str:
    """A plain shot list, for finishing by hand in any NLE.

    Not FCPXML: the OTIO build here ships no FCP adapter, and shipping a
    hand-rolled XML importer that has never been opened in Resolve would be
    worse than shipping nothing. The timeline JSON stays the editable artifact.
    """
    lines = [f"{'#':<4}{'source':<34}{'in':>9}{'out':>9}{'len':>8}  join"]
    lines.append("-" * 72)
    position = 0
    for segment in variant["timeline"]["segments"]:
        clips = [segment] if segment["kind"] == "shot" else [segment["a"], segment["b"]]
        for offset, clip in enumerate(clips):
            join = "cut"
            if segment["kind"] == "transition" and offset == 1:
                join = f"{segment['style']} {segment['duration']:.2f}s"
            position += 1
            lines.append(
                f"{position:<4}{Path(clip['source']).name:<34}"
                f"{clip['in']:>9.3f}{clip['out']:>9.3f}"
                f"{clip['out'] - clip['in']:>8.3f}  {join}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- entry


def _write_variant(variant: dict, path: Path) -> list[str]:
    """Validate and write one timeline. Returns schema errors, empty on success."""
    errors = render.validate_timeline(variant["timeline"],
                                     config.SCHEMAS_DIR / "timeline.schema.json")
    if errors:
        return errors
    path.write_text(json.dumps(variant["timeline"], indent=2))
    variant["path"] = path
    return []


def _main_briefed(args, cards: list[dict], music: dict | None) -> int:
    """Build the story a brief describes, from the cast that filled it.

    The three variants differ in effect intensity rather than in structure or
    order, because with a brief those are decided: a variant that reordered the
    shots would be telling a different story, not offering the same one twice.
    """
    from pipeline import cast as cast_stage
    from pipeline import overlay

    if not args.brief.exists():
        print(f"{RED}no brief at {args.brief} — run `python -m pipeline.brief` "
              f"first{RESET}", file=sys.stderr)
        return 1

    brief = json.loads(args.brief.read_text())

    if args.cast.exists():
        cast_result = json.loads(args.cast.read_text())
    else:
        print(f"  {DIM}no cast.json — casting now{RESET}")
        cast_result = cast_stage.cast(cards, brief)

    cues = overlay.cues_for_brief(brief, config.WORK_DIR / "cues")

    print(f"\n{BOLD}sequence{RESET}  {DIM}{brief['blueprint']} · "
          f"{len(brief['shots'])} slots · {len(cues)} text cues · "
          f"{'beat-locked' if music else 'no music map'}{RESET}\n")

    if not cast_result["complete"]:
        print(f"  {YELLOW}building without {', '.join(cast_result['missing'])} — "
              f"required slot(s) had no footage{RESET}")

    written: list[dict] = []
    for index, flavour in enumerate(["standard", "hype", "calm"][:max(args.variants, 1)]):
        variant = assemble_briefed(cards, brief, cast_result, music,
                                   args.track, flavour, cues)
        if not variant:
            print(f"  {RED}{flavour}: nothing to build{RESET}")
            continue

        path = args.out_dir / f"timeline_v{index + 1}.json"
        if errors := _write_variant(variant, path):
            print(f"  {RED}{flavour} is invalid:{RESET}")
            for problem in errors[:6]:
                print(f"    {problem}")
            return 1
        written.append(variant)

        total = sum(render.segment_duration(s) for s in variant["timeline"]["segments"])
        print(f"  {GREEN}v{index + 1}    {RESET}  {total:5.2f}s · "
              f"{len(variant['picked'])} shots · {len(variant['transitions'])} transitions "
              f"{DIM}({flavour}) -> {path.name}{RESET}")
        for note in variant["notes"]:
            print(f"         {YELLOW}{note}{RESET}")

    if not written:
        print(f"\n{RED}nothing assembled{RESET}\n")
        return 1

    best = written[0]
    (args.out_dir / "timeline.json").write_text(json.dumps(best["timeline"], indent=2))

    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    (config.OUT_DIR / "sync.txt").write_text(sync_sheet(music, best["timeline"], written))
    (config.OUT_DIR / "edit_list.txt").write_text(edit_list(best))

    print(f"\n  {BOLD}{'slot':<18}{'role':<10}{'len':>7}  text{RESET}")
    kept = [s for s in brief["shots"]
            if s["id"] not in set(best["timeline"]["story"]["dropped"])]
    for shot, slot in zip(kept, best["slots"]):
        cue = (shot.get("text") or {}).get("text", "")
        print(f"  {shot['id']:<18}{shot['role']:<10}{slot['duration']:>7.2f}  "
              f"{DIM}{cue[:40]}{RESET}")

    print(f"\n{GREEN}{len(written)} variant(s){RESET}  {DIM}v1 copied to timeline.json · "
          f"start the sound at {brief['music']['best_start']:.2f}s{RESET}\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Assemble moments into beat-locked reels.")
    ap.add_argument("--cards", type=Path, default=config.CLIP_CARDS_JSON)
    ap.add_argument("--music", type=Path, default=config.MUSIC_MAP_JSON)
    ap.add_argument("--track", type=Path, default=None,
                    help="bake this audio into the render (default: silent, for in-app sound)")
    ap.add_argument("--out-dir", type=Path, default=config.WORK_DIR)
    ap.add_argument("--target", type=float, default=config.TARGET_REEL_SECONDS)
    ap.add_argument("--variants", type=int, default=3)
    ap.add_argument("--shots", type=int, default=None, help="force a slot count")
    ap.add_argument("--brief", type=Path, default=None,
                    help="build the story this brief describes instead of ranking")
    ap.add_argument("--cast", type=Path, default=config.CAST_JSON)
    args = ap.parse_args(argv)

    if not args.cards.exists():
        print(f"{RED}no clip_cards.json — run `python -m pipeline.merge` first{RESET}",
              file=sys.stderr)
        return 1

    cards = json.loads(args.cards.read_text())["cards"]
    music = json.loads(args.music.read_text()) if args.music.exists() else None
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.brief:
        return _main_briefed(args, cards, music)

    target = max(min(args.target, config.MAX_REEL_SECONDS), config.MIN_REEL_SECONDS)
    count = args.shots or max(int(round(target / TARGET_SLOT_SECONDS)), 3)

    print(f"\n{BOLD}sequence{RESET}  {DIM}{len(cards)} moments · target {target:.0f}s · "
          f"{'beat-locked' if music else 'no music map'}{RESET}\n")

    if len(cards) < count:
        print(f"  {YELLOW}only {len(cards)} moments for {count} slots — building a "
              f"shorter reel rather than reusing footage{RESET}")
        count = len(cards)

    shapes = ["standard", "fast", "breathe"]
    written: list[dict] = []

    for index in range(max(args.variants, 1)):
        variant = assemble(cards, music, target, count,
                           shapes[index % len(shapes)], index, args.track)
        if not variant:
            print(f"  {RED}variant {index + 1}: no assignable moments{RESET}")
            continue

        path = args.out_dir / f"timeline_v{index + 1}.json"
        errors = render.validate_timeline(variant["timeline"],
                                          config.SCHEMAS_DIR / "timeline.schema.json")
        if errors:
            print(f"  {RED}variant {index + 1} is invalid:{RESET}")
            for problem in errors[:6]:
                print(f"    {problem}")
            return 1

        path.write_text(json.dumps(variant["timeline"], indent=2))
        variant["path"] = path
        written.append(variant)

        total = sum(render.segment_duration(s) for s in variant["timeline"]["segments"])
        styles = ", ".join(sorted({t["style"] for t in variant["transitions"].values()})) or "all cuts"
        print(f"  {GREEN}v{index + 1}    {RESET}  {total:5.2f}s · {len(variant['picked'])} shots · "
              f"{len(variant['transitions'])}/{max(len(variant['picked']) - 1, 1)} transitions "
              f"{DIM}({styles}) · {variant['shape']} arc -> {path.name}{RESET}")
        for note in variant["notes"]:
            print(f"         {YELLOW}{note}{RESET}")

    if not written:
        print(f"\n{RED}nothing assembled{RESET}\n")
        return 1

    best = written[0]
    (args.out_dir / "timeline.json").write_text(json.dumps(best["timeline"], indent=2))

    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    (config.OUT_DIR / "sync.txt").write_text(sync_sheet(music, best["timeline"], written))
    (config.OUT_DIR / "edit_list.txt").write_text(edit_list(best))

    print(f"\n  {BOLD}{'slot':<6}{'moment':<26}{'len':>7}{'rank':>7}  join{RESET}")
    for position, (card, slot) in enumerate(zip(best["picked"], best["slots"])):
        proposal = best["transitions"].get(position)
        join = f"{proposal['style']} {DIM}({proposal['reason']}){RESET}" if proposal else f"{DIM}cut{RESET}"
        print(f"  {position:<6}{card['shot_id']:<26}{slot['duration']:>7.2f}"
              f"{card['rank_score']:>7.2f}  {join}")

    print(f"\n{GREEN}{len(written)} variant(s){RESET}  {DIM}v1 copied to timeline.json · "
          f"sync sheet and edit list in {config.OUT_DIR}{RESET}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
