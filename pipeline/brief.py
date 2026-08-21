"""Turn a song into a shot list, before a frame is shot.

This is the stage that inverts the pipeline. v3 consumed whatever footage
arrived and ranked it; the reel had no story because nothing ever decided what
the story was. Here the song comes first: it is measured, a story blueprint is
matched to its shape, and the blueprint is laid over the real beat grid so every
shot has a length in beats, a timecode and the words that will sit over it.

What comes out is a brief you shoot from. What goes back in is footage that was
always meant to fill these slots, which is the difference between editing and
salvage.

    uv run python -m pipeline.brief --track assets/song.mov
    uv run python -m pipeline.brief --track assets/song.mov --blueprint travel_reveal \
        --concept "the hike up to the lake" --copy work/copy.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from pipeline import config, music, sequence

BOLD, RED, YELLOW, GREEN, DIM, RESET = "\033[1m", "\033[31m", "\033[33m", "\033[32m", "\033[2m", "\033[0m"

# How much each song property may cost a blueprint's fit score. Tempo dominates
# because it is the one that decides whether the cutting rhythm is even
# buildable; a story that wants stabs cannot be told at 90 BPM. The drop penalty
# is largest of all and deliberately so -- a peak-driven story on a flat groove
# has nowhere to put its payoff, which is not a matter of taste.
FIT_WEIGHTS = {
    "tempo": 0.35,
    "aggression": 0.18,
    "vocal_density": 0.12,
    "energy_variance": 0.12,
}

# Everything starts here rather than at 1.0, so a story that actively exploits
# the track can score above a story that merely fails to clash with it. Without
# the headroom, a blueprint with no opinion about drops scored a perfect 1.00 on
# every track ever measured and drop-driven stories could only ever match it --
# so on a track with an obvious drop, the three top recommendations were the
# three blueprints that had no use for one.
BASE_FIT = 0.85

# Drop terms, signed. A peak-driven story on a track with a real arrival is the
# best case the system has; the same story on a flat groove has nowhere to put
# its payoff, which is a structural problem rather than a matter of taste.
DROP_BONUS = 0.20
NO_DROP_PENALTY = 0.60
# A story with no use for a drop, on a track built around one, wastes the single
# loudest event in the audio. Mild -- it is a missed opportunity, not a mistake.
UNUSED_DROP_PENALTY = 0.12
STRONG_DROP = 0.45

# Text timing. The hook goes up immediately -- published guidance is unusually
# consistent that the first frame should already carry it -- while labels are
# given a beat to breathe after the cut so they do not read as part of the
# previous shot.
HOOK_DELAY = 0.0
LABEL_DELAY = 0.20
CTA_DELAY = 0.10
MAX_HOOK_SECONDS = 2.2
MAX_LABEL_SECONDS = 1.8

# Six to eight words is the widely cited ceiling for a hook that can be read in
# the time it is on screen. Warned about rather than enforced: a five-word hook
# is fine, and a nine-word one is the author's call to make.
HOOK_WORD_LIMIT = 8


# ---------------------------------------------------------------- blueprints


def load_blueprints(directory: Path) -> list[dict]:
    return [json.loads(path.read_text())
            for path in sorted(directory.glob("*.json"))]


def _band_penalty(value: float, band: list[float] | None, weight: float) -> float:
    """Cost of sitting outside a preferred range, scaled by how far outside.

    A band is a preference, never a filter. Every blueprint stays choosable on
    every track -- the score orders the menu, it does not shorten it, because a
    deliberate mismatch is a legitimate creative choice and the system has no
    business overruling it.
    """
    if not band:
        return 0.0
    low, high = band
    if low <= value <= high:
        return 0.0
    span = max(high - low, 1e-6)
    distance = (low - value) if value < low else (value - high)
    return min(distance / span, 1.0) * weight


def _drop_term(wants: dict, has_drop: bool, sharpness: float) -> tuple[float, str | None]:
    """Signed contribution of the track's arrival, and how to say it."""
    if wants.get("needs_drop"):
        if not has_drop:
            return -NO_DROP_PENALTY, "needs a drop and this track has none"
        strength = min(sharpness / STRONG_DROP, 1.0)
        return DROP_BONUS * strength, "built around the drop this track has"

    if has_drop and sharpness >= STRONG_DROP:
        return -UNUSED_DROP_PENALTY, "this track has a strong drop the story never uses"
    return 0.0, None


def score_blueprint(blueprint: dict, profile: dict, has_drop: bool) -> tuple[float, list[str]]:
    """How well this story fits this song, and why, in words.

    The reasons matter as much as the number. This is a heuristic over five
    measurements and it will sometimes be wrong, so it has to show its working
    rather than hand down a ranking.

    Tempo and the drop carry the weight here because they are the two things
    that have actually been validated against known-structure audio. The
    aggression and vocal bands are informed guesses and are weighted like it --
    the same mistake as v2's MOTION_TARGET is available in this file, and the
    way not to make it is to keep unvalidated measurements cheap.
    """
    wants = blueprint.get("song", {})
    score, reasons = BASE_FIT, []

    tempo = profile.get("tempo", 0.0)
    band = wants.get("tempo")
    cost = _band_penalty(tempo, band, FIT_WEIGHTS["tempo"])
    score -= cost
    if band:
        reasons.append(f"{tempo:.0f} BPM {'sits in' if cost == 0 else 'is outside'} "
                       f"its {band[0]:.0f}-{band[1]:.0f} range")

    for key in ("aggression", "vocal_density", "energy_variance"):
        band = wants.get(key)
        if not band:
            continue
        value = profile.get(key, 0.0)
        cost = _band_penalty(value, band, FIT_WEIGHTS[key])
        score -= cost
        if cost > 0.02:
            direction = "low" if value < band[0] else "high"
            reasons.append(f"{key.replace('_', ' ')} {value:.2f} is {direction} for it")

    term, reason = _drop_term(wants, has_drop, profile.get("drop_sharpness", 0.0))
    score += term
    if reason:
        reasons.append(reason)

    return min(max(score, 0.0), 1.0), reasons


def rank_blueprints(blueprints: list[dict], music_map: dict,
                    pillar: str | None = None) -> list[tuple[float, dict, list[str]]]:
    profile = music_map.get("profile") or {"tempo": music_map.get("tempo", 0.0)}
    has_drop = music_map.get("drop_at") is not None

    scored = [(*score_blueprint(b, profile, has_drop), b)
              for b in blueprints if not pillar or b["pillar"] == pillar]
    return sorted(((score, blueprint, reasons) for score, reasons, blueprint in scored),
                  key=lambda row: -row[0])


# ---------------------------------------------------------------- layout


def enforce_caps(multiples: list[int], shots: list[dict], period: float,
                 usable: list[int]) -> tuple[list[int], bool]:
    """Hold shots to their declared maximum length, giving the beats elsewhere.

    The hook is the case this exists for. Published retention data is blunt
    about the first two seconds deciding everything, so a hook slot carries
    max_seconds and must keep it at any tempo -- but quantise() distributes
    against relative weights and knows nothing about absolute ceilings. Beats
    removed here are handed to the longest shot rather than dropped, because the
    reel has to stay a whole number of bars or it stops landing on the beat.
    """
    changed = False
    for index, shot in enumerate(shots):
        cap = shot.get("max_seconds")
        if not cap or multiples[index] * period <= cap:
            continue
        fits = [m for m in usable if m * period <= cap]
        if not fits:
            continue
        freed = multiples[index] - max(fits)
        multiples[index] = max(fits)
        changed = True

        # Park the freed beats on the longest slot, which is the one least
        # damaged by growing: it is already a hold rather than a stab.
        while freed > 0:
            target = max(range(len(multiples)),
                         key=lambda i: (multiples[i], -i) if i != index else (-1, 0))
            options = [m for m in usable if m > multiples[target]]
            if not options:
                break
            step = min(options) - multiples[target]
            if step > freed:
                break
            multiples[target] = min(options)
            freed -= step

    return multiples, changed


def plan_beats(blueprint: dict, period: float, goal: float) -> tuple[list[int], int, bool]:
    """Beat length of every shot, summing to a whole number of bars."""
    shots = blueprint["shots"]
    usable = sequence.allowed_multiples(period)

    target_beats = max(int(round(goal / period)), sequence.BEATS_PER_BAR)
    target_beats = max(round(target_beats / sequence.BEATS_PER_BAR)
                       * sequence.BEATS_PER_BAR, sequence.BEATS_PER_BAR)

    multiples = sequence.quantise([s["weight"] for s in shots], target_beats, usable)
    multiples, capped = enforce_caps(multiples, shots, period, usable)
    return multiples, target_beats, capped


def payoff_fraction(blueprint: dict, period: float, goal: float) -> float | None:
    """Where through the reel the payoff shot actually starts, 0..1.

    Derived from the quantised layout rather than read from the blueprint's
    declared `payoff_at`. Those two numbers are answers to the same question and
    the declared one was wrong: gym_pr_attempt claims 0.66, while its own shot
    weights put the payoff cut at 0.53. Aligning the drop to the declaration put
    it 2.49s inside the payoff shot instead of on the cut into it -- the reel was
    correct by the constant and wrong by the edit.

    `payoff_at` survives as documentation of intent, and a disagreement with
    this value is reported as a blueprint bug rather than silently resolved.
    """
    index = next((i for i, s in enumerate(blueprint["shots"])
                  if s["role"] == "payoff"), None)
    if index is None or period <= 0:
        return None

    multiples, target_beats, _ = plan_beats(blueprint, period, goal)
    if target_beats <= 0:
        return None
    return sum(multiples[:index]) / target_beats


def lay_out(blueprint: dict, music_map: dict, copy: dict,
            target: float | None = None) -> dict:
    """Put the blueprint on the grid: real durations, real timecodes, real text."""
    beats = music_map.get("beats") or []
    period = music_map.get("beat_period") or 0.0
    if not beats or period <= 0:
        raise ValueError("music map has no usable beat grid")

    shots = blueprint["shots"]
    goal = target or blueprint.get("target_seconds", config.TARGET_REEL_SECONDS)
    multiples, _, capped = plan_beats(blueprint, period, goal)

    start = music_map.get("best_start", beats[0])
    origin = min(range(len(beats)), key=lambda i: abs(beats[i] - start))
    reel_zero = sequence.on_frame(sequence.beat_time(beats, origin))

    labels = copy.get("labels") or []
    notes: list[str] = []
    if capped:
        notes.append("one or more shots were held to their maximum length; "
                     "the beats went to the longest slot")

    planned: list[dict] = []
    cumulative = 0
    for index, (shot, count) in enumerate(zip(shots, multiples)):
        t0 = sequence.on_frame(sequence.beat_time(beats, origin + cumulative))
        t1 = sequence.on_frame(sequence.beat_time(beats, origin + cumulative + count))
        cumulative += count

        entry = {
            "index": index,
            "id": shot["id"],
            "role": shot["role"],
            "beats": count,
            # Six decimals, not four. Both boundaries are already on the frame
            # grid, so their difference is an exact frame multiple -- but 119
            # frames at 30fps is 3.966666..., which four decimals turns into
            # 3.9667, and 3.9667 * 30 is 119.001. The frame count still rounds
            # correctly, so this never broke a render; it made the timeline's
            # arithmetic un-checkable, which is how it went unnoticed.
            "duration": round(t1 - t0, 6),
            "start": round(t0 - reel_zero, 6),
            "track_time": round(t0, 6),
            "section": music.section_at(music_map, t0),
            "on_downbeat": (origin + cumulative - count) % sequence.BEATS_PER_BAR
                           == origin % sequence.BEATS_PER_BAR,
            "must_have": bool(shot.get("must_have")),
            "what": shot["what"],
            "shoot": shot.get("shoot", ""),
            "framing": shot["framing"],
            "camera": shot["camera"],
            "subject": shot["subject"],
            "motion": shot["motion"],
        }

        # Carried through verbatim, because sequence.expand_bursts() reads it
        # off the *brief*, not the blueprint. Dropping it here did not fail
        # anything: the burst simply never happened, and the note that reports
        # a skipped burst counts the same key, so it stayed silent too. Three of
        # the four reference reels open their build on one -- see STYLE.md 8.
        if shot.get("burst"):
            entry["burst"] = shot["burst"]

        if cue := _text_cue(shot, copy, labels, entry["duration"]):
            entry["text"] = cue
        planned.append(entry)

    total = round(sum(s["duration"] for s in planned), 3)
    notes += _layout_notes(blueprint, planned, music_map, copy, reel_zero)

    return {
        "version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "blueprint": blueprint["id"],
        "pillar": blueprint["pillar"],
        "title": blueprint["title"],
        "premise": blueprint["premise"],
        "chronological": bool(blueprint.get("chronological")),
        "loop": bool(blueprint.get("loop")),
        "text_mode": blueprint.get("text_mode", "captions"),
        **({"style": blueprint["style"]} if blueprint.get("style") else {}),
        **({"grade": blueprint["grade"]} if blueprint.get("grade") else {}),
        **({"subject_dodge": True} if blueprint.get("subject_dodge") else {}),
        **({"letterbox": blueprint["letterbox"]} if blueprint.get("letterbox") else {}),
        "target_seconds": goal,
        "total_seconds": total,
        "music": {
            "track": music_map["track"],
            "tempo": music_map["tempo"],
            "beat_period": period,
            "bar_seconds": music_map.get("bar_seconds", period * 4),
            "best_start": round(reel_zero, 4),
            "drop_at": music_map.get("drop_at"),
            "payoff_at": blueprint.get("song", {}).get("payoff_at"),
        },
        "copy": copy,
        "shots": planned,
        "notes": notes,
    }


def _text_cue(shot: dict, copy: dict, labels: list[str], duration: float) -> dict | None:
    """Resolve a shot's text slot into an actual line with timing."""
    spec = shot.get("text")
    if not spec:
        return None

    kind = spec["kind"]
    if kind == "hook_title":
        text, delay, cap = copy.get("hook", ""), HOOK_DELAY, MAX_HOOK_SECONDS
    elif kind == "cta":
        text, delay, cap = copy.get("cta", ""), CTA_DELAY, duration
    else:
        index = spec.get("index", 0)
        if index >= len(labels):
            return None
        text, delay, cap = labels[index], LABEL_DELAY, MAX_LABEL_SECONDS

    if not text:
        return None

    delay = min(delay, max(duration - 0.2, 0.0))
    return {"kind": kind, "text": text, "at": round(delay, 3),
            "duration": round(min(cap, duration - delay), 3)}


def _layout_notes(blueprint: dict, planned: list[dict], music_map: dict,
                  copy: dict, reel_zero: float) -> list[str]:
    """Everything about this brief the author should know before shooting."""
    notes: list[str] = []

    hook = next((s for s in planned if s["role"] == "hook"), None)
    if hook and hook["start"] + hook["duration"] > 2.05:
        notes.append(f"the hook runs to {hook['start'] + hook['duration']:.2f}s — "
                     f"published guidance puts the whole hook inside the first 2s")

    words = len((copy.get("hook") or "").split())
    if words > HOOK_WORD_LIMIT:
        notes.append(f"hook text is {words} words; {HOOK_WORD_LIMIT} is the usual "
                     f"ceiling for something readable in two seconds")

    payoff = next((s for s in planned if s["role"] == "payoff"), None)
    drop = music_map.get("drop_at")
    if payoff and drop is not None:
        offset = payoff["track_time"] - drop
        if abs(offset) > music_map.get("bar_seconds", 2.0):
            notes.append(f"the payoff shot lands {offset:+.2f}s from the drop — "
                         f"more than a bar out")
    elif payoff and blueprint.get("song", {}).get("needs_drop"):
        notes.append("this story is built around a drop and the track has none; "
                     "the payoff will land on a beat but not on an arrival")

    # The pattern-interrupt rule: attention drifts when nothing changes for more
    # than about three seconds. A camera move or a moving subject *is* a change,
    # so only a long shot that is static in both senses counts -- the first
    # version of this check flagged every payoff in the library, which is a
    # signal that fires on the correct answer and is therefore useless.
    for shot in planned:
        static = shot["camera"] == "static" and shot["motion"] in ("still", "low")
        if shot["duration"] > 3.0 and static:
            notes.append(f"shot {shot['index'] + 1} ({shot['id']}) holds "
                         f"{shot['duration']:.2f}s with a locked camera and no "
                         f"movement — give it a push in, or cut it shorter")

    if music_map.get("duration", 0) < reel_zero + sum(s["duration"] for s in planned):
        notes.append("the reel is longer than the recorded audio — record more of "
                     "the sound, or the last cuts have no grid under them")

    return notes


# ---------------------------------------------------------------- output


def _timecode(seconds: float) -> str:
    return f"{int(seconds // 60)}:{seconds % 60:05.2f}"


def shotlist_markdown(brief: dict) -> str:
    music_info = brief["music"]
    lines = [
        f"# {brief['title']} — shot list",
        "",
        f"*{brief['premise']}*",
        "",
    ]
    if brief.get("concept"):
        lines += [f"**This reel:** {brief['concept']}", ""]

    lines += [
        f"- **Sound** `{Path(music_info['track']).name}` · {music_info['tempo']:.0f} BPM",
        f"- **Start the audio at** `{_timecode(music_info['best_start'])}` "
        f"({music_info['best_start']:.3f}s into the track)",
        f"- **Runtime** {brief['total_seconds']:.2f}s across {len(brief['shots'])} shots",
        f"- **Order** {'chronological — shoot and keep it in sequence' if brief['chronological'] else 'free — the editor may reorder'}",
    ]
    if brief.get("loop"):
        lines.append("- **Loops** — the last shot must match the first, framing for framing")
    if music_info.get("drop_at") is not None:
        lines.append(f"- **Drop** at `{_timecode(music_info['drop_at'])}` in the track")

    lines += ["", "## Camera", "",
              "| Setting | Use |", "|---|---|",
              "| 4K 16:9, not vertical | always — vertical throws away all reframing room |",
              "| 4K60 | anything marked for slow motion |",
              "| 4K30 | everything else |",
              "| D-Log M | always |",
              "", "## Shots", ""]

    for shot in brief["shots"]:
        must = " **· REQUIRED**" if shot["must_have"] else ""
        lines += [
            f"### {shot['index'] + 1}. {shot['id']} — {shot['role']}{must}",
            "",
            f"`{shot['duration']:.2f}s` · {shot['beats']} beats · "
            f"lands at {_timecode(shot['start'])} · {shot['section']}",
            "",
            f"**What** {shot['what']}",
            "",
            f"**Frame** {shot['framing']} · **Camera** {shot['camera']} · "
            f"**Subject** {shot['subject']} · **Motion** {shot['motion']}",
        ]
        if cue := shot.get("text"):
            lines += ["", f"**Text** “{cue['text']}” — appears {cue['at']:.1f}s in"]
        if shot.get("shoot"):
            lines += ["", f"**Shoot** {shot['shoot']}"]
        lines.append("")

    if brief["notes"]:
        lines += ["## Notes", ""] + [f"- {n}" for n in brief["notes"]] + [""]

    return "\n".join(lines)


def shotcard_html(brief: dict) -> str:
    """A shot list to read on a phone, at arm's length, in a gym or on a trail.

    Dark-first and high-contrast because that is the room it gets read in, and
    because the footage it describes is graded dark -- the page and the material
    should not disagree. Timecodes are tabular so the column of them can be
    scanned rather than read.

    Checkboxes are the only interaction, because ticking off shots is the only
    thing anyone does with a call sheet while holding a camera. They last for the
    session and make no claim to persist.
    """
    music_info = brief["music"]
    escape = _escape

    role_order = ["hook", "promise", "build", "payoff", "cta"]
    cards: list[str] = []
    for shot in brief["shots"]:
        cue = shot.get("text")
        rows = [
            ("frame", shot["framing"]),
            ("camera", shot["camera"]),
            ("subject", shot["subject"]),
            ("motion", f"{shot['motion']} · {shot['section']}"),
        ]
        meta = "".join(
            f'<div class="spec"><dt>{label}</dt><dd>{escape(value)}</dd></div>'
            for label, value in rows)

        text_block = ""
        if cue:
            text_block = (
                f'<p class="cue"><span class="cue-label">on screen</span>'
                f'<span class="cue-text">{escape(cue["text"])}</span>'
                f'<span class="cue-when">from {cue["at"]:.1f}s, for '
                f'{cue["duration"]:.1f}s</span></p>')

        shoot = (f'<p class="shoot">{escape(shot["shoot"])}</p>'
                 if shot.get("shoot") else "")
        required = ('<span class="req">required</span>'
                    if shot["must_have"] else "")
        slow = ('<span class="flag">4K60</span>'
                if "4K60" in (shot.get("shoot") or "") else "")

        cards.append(f"""
      <article class="shot" data-role="{shot['role']}">
        <label class="tick"><input type="checkbox"><span></span></label>
        <div class="num">{shot['index'] + 1}</div>
        <div class="body">
          <header>
            <span class="role r-{shot['role']}">{shot['role']}</span>
            {required}{slow}
            <span class="timing">{shot['duration']:.2f}s · {shot['beats']} beats
              · at {_timecode(shot['start'])}</span>
          </header>
          <p class="what">{escape(shot['what'])}</p>
          <dl class="specs">{meta}</dl>
          {text_block}
          {shoot}
        </div>
      </article>""")

    spine = " → ".join(
        f'<span class="r-{role}">{role}</span>' for role in role_order
        if any(s["role"] == role for s in brief["shots"]))

    notes = ""
    if brief.get("notes"):
        items = "".join(f"<li>{escape(n)}</li>" for n in brief["notes"])
        notes = f'<section class="notes"><h2>Before you shoot</h2><ul>{items}</ul></section>'

    concept = (f'<p class="concept">{escape(brief["concept"])}</p>'
               if brief.get("concept") else "")

    order_note = ("Shoot in order and keep it — this story is chronological."
                  if brief["chronological"]
                  else "Order is free; the edit may rearrange these.")
    loop_note = (" The last shot must match the first, framing for framing, so the "
                 "reel loops without a seam." if brief.get("loop") else "")

    drop = ""
    if music_info.get("drop_at") is not None:
        drop = (f'<div class="stat"><dt>drop</dt>'
                f'<dd>{_timecode(music_info["drop_at"])}</dd></div>')

    return f"""<title>{escape(brief['title'])} Shot List</title>
<style>
  :root {{
    --ground: #FAFAF8; --surface: #FFFFFF; --sunk: #EFEFEA;
    --ink: #14181A; --muted: #626C70; --line: #DCDCD5;
    --accent: #B26A12; --accent-soft: #F6E4CB;
    --time: #14706A;
    --hook: #B23A2E; --promise: #8A5A1E; --build: #4A6070;
    --payoff: #1E6E52; --cta: #6A4A8A;
  }}
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{
      --ground: #101315; --surface: #191E21; --sunk: #22282C;
      --ink: #E9ECEB; --muted: #93A0A5; --line: #2C3438;
      --accent: #F2A33C; --accent-soft: #3A2A12;
      --time: #6FC9BC;
      --hook: #F07A67; --promise: #E0A75A; --build: #8FAAB9;
      --payoff: #63C79E; --cta: #B79BD8;
    }}
  }}
  :root[data-theme="dark"] {{
    --ground: #101315; --surface: #191E21; --sunk: #22282C;
    --ink: #E9ECEB; --muted: #93A0A5; --line: #2C3438;
    --accent: #F2A33C; --accent-soft: #3A2A12;
    --time: #6FC9BC;
    --hook: #F07A67; --promise: #E0A75A; --build: #8FAAB9;
    --payoff: #63C79E; --cta: #B79BD8;
  }}

  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; background: var(--ground); color: var(--ink);
    font: 400 17px/1.55 ui-sans-serif, -apple-system, "Segoe UI", "Helvetica Neue", sans-serif;
    -webkit-text-size-adjust: 100%;
  }}
  .wrap {{ max-width: 44rem; margin: 0 auto; padding: 1.5rem 1.1rem 4rem; }}

  h1 {{
    font-size: clamp(1.9rem, 7vw, 2.6rem); line-height: 1.08; margin: 0 0 .35rem;
    letter-spacing: -.02em; text-wrap: balance;
  }}
  .premise {{ color: var(--muted); margin: 0 0 1rem; font-size: 1rem; }}
  .concept {{
    margin: 0 0 1.25rem; padding: .7rem .9rem; background: var(--sunk);
    border-left: 3px solid var(--accent); border-radius: 0 6px 6px 0;
    font-size: 1rem;
  }}

  .strip {{
    display: flex; flex-wrap: wrap; gap: .5rem 1.4rem;
    padding: .9rem 1rem; margin: 0 0 .75rem;
    background: var(--surface); border: 1px solid var(--line); border-radius: 10px;
    position: sticky; top: 0; z-index: 5;
  }}
  .stat {{ display: flex; flex-direction: column; gap: .1rem; }}
  .stat dt {{
    font-size: .68rem; text-transform: uppercase; letter-spacing: .09em;
    color: var(--muted);
  }}
  .stat dd {{
    margin: 0; font: 600 1.05rem/1.2 ui-monospace, "SF Mono", Menlo, monospace;
    font-variant-numeric: tabular-nums; color: var(--time);
  }}
  .stat.wide dd {{ color: var(--ink); font-size: .95rem; }}

  .spine {{
    font-size: .78rem; text-transform: uppercase; letter-spacing: .1em;
    color: var(--muted); margin: 0 0 1.5rem; padding-left: .2rem;
  }}
  .spine span {{ font-weight: 700; }}

  .progress {{
    display: flex; align-items: baseline; gap: .5rem; margin: 0 0 1rem;
    font-size: .85rem; color: var(--muted);
  }}
  .progress b {{
    font: 700 1.1rem/1 ui-monospace, Menlo, monospace; color: var(--ink);
    font-variant-numeric: tabular-nums;
  }}

  .shot {{
    display: grid; grid-template-columns: 2.4rem 1fr; gap: 0 .85rem;
    position: relative;
    padding: 1.1rem 1rem 1.15rem 1rem; margin-bottom: .7rem;
    background: var(--surface); border: 1px solid var(--line); border-radius: 10px;
  }}
  .shot:has(input:checked) {{ opacity: .48; }}
  .shot:has(input:checked) .what {{ text-decoration: line-through; }}

  .tick {{ position: absolute; top: .8rem; right: .8rem; cursor: pointer; }}
  .tick input {{ position: absolute; opacity: 0; width: 0; height: 0; }}
  .tick span {{
    display: block; width: 1.55rem; height: 1.55rem; border-radius: 5px;
    border: 2px solid var(--line); background: var(--ground);
  }}
  .tick input:checked + span {{
    background: var(--accent); border-color: var(--accent);
  }}
  .tick input:checked + span::after {{
    content: ""; display: block; width: .42rem; height: .8rem;
    margin: .18rem auto 0; border: solid var(--ground);
    border-width: 0 .18rem .18rem 0; transform: rotate(45deg);
  }}
  .tick input:focus-visible + span {{ outline: 2px solid var(--time); outline-offset: 2px; }}

  .num {{
    grid-row: 1 / span 2;
    font: 800 1.75rem/1 ui-monospace, Menlo, monospace;
    font-variant-numeric: tabular-nums;
    color: var(--muted); text-align: right; padding-top: .1rem;
  }}
  .body {{ min-width: 0; padding-right: 2rem; }}

  .shot header {{
    display: flex; flex-wrap: wrap; align-items: center; gap: .4rem .6rem;
    margin-bottom: .45rem;
  }}
  .role {{
    font: 700 .68rem/1 ui-sans-serif, sans-serif; text-transform: uppercase;
    letter-spacing: .1em; padding: .3rem .45rem; border-radius: 4px;
    background: var(--sunk);
  }}
  .r-hook {{ color: var(--hook); }}
  .r-promise {{ color: var(--promise); }}
  .r-build {{ color: var(--build); }}
  .r-payoff {{ color: var(--payoff); }}
  .r-cta {{ color: var(--cta); }}
  .req, .flag {{
    font: 700 .66rem/1 ui-sans-serif, sans-serif; text-transform: uppercase;
    letter-spacing: .09em; padding: .3rem .45rem; border-radius: 4px;
    color: var(--accent); background: var(--accent-soft);
  }}
  .timing {{
    margin-left: auto; font: 500 .8rem/1 ui-monospace, Menlo, monospace;
    font-variant-numeric: tabular-nums; color: var(--time); white-space: nowrap;
  }}

  .what {{ margin: 0 0 .6rem; font-size: 1.06rem; }}

  .specs {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(7rem, 1fr));
    gap: .45rem .8rem; margin: 0 0 .6rem;
  }}
  .spec dt {{
    font-size: .64rem; text-transform: uppercase; letter-spacing: .09em;
    color: var(--muted);
  }}
  .spec dd {{ margin: 0; font-size: .9rem; font-weight: 600; }}

  .cue {{
    margin: 0 0 .55rem; padding: .55rem .7rem; background: var(--sunk);
    border-radius: 7px; display: flex; flex-direction: column; gap: .15rem;
  }}
  .cue-label {{
    font-size: .62rem; text-transform: uppercase; letter-spacing: .1em;
    color: var(--muted);
  }}
  .cue-text {{ font-weight: 700; font-size: 1.02rem; }}
  .cue-when {{
    font: 500 .74rem/1 ui-monospace, Menlo, monospace; color: var(--muted);
    font-variant-numeric: tabular-nums;
  }}

  .shoot {{
    margin: 0; font-size: .92rem; color: var(--muted);
    border-top: 1px solid var(--line); padding-top: .5rem;
  }}

  .notes {{
    margin-top: 2rem; padding: 1rem 1.1rem; background: var(--sunk);
    border-radius: 10px;
  }}
  .notes h2 {{
    margin: 0 0 .5rem; font-size: .74rem; text-transform: uppercase;
    letter-spacing: .1em; color: var(--muted);
  }}
  .notes ul {{ margin: 0; padding-left: 1.1rem; }}
  .notes li {{ margin-bottom: .35rem; font-size: .93rem; }}

  .rig {{ margin-top: 2rem; border-top: 1px solid var(--line); padding-top: 1rem; }}
  .rig h2 {{
    margin: 0 0 .6rem; font-size: .74rem; text-transform: uppercase;
    letter-spacing: .1em; color: var(--muted);
  }}
  .rig dl {{ margin: 0; display: grid; gap: .4rem; }}
  .rig div {{ display: flex; gap: .7rem; font-size: .92rem; }}
  .rig dt {{ font-weight: 700; min-width: 9rem; }}
  .rig dd {{ margin: 0; color: var(--muted); }}

  @media (prefers-reduced-motion: reduce) {{ * {{ transition: none !important; }} }}
</style>

<div class="wrap">
  <h1>{escape(brief['title'])}</h1>
  <p class="premise">{escape(brief['premise'])}</p>
  {concept}

  <dl class="strip">
    <div class="stat"><dt>start audio</dt><dd>{_timecode(music_info['best_start'])}</dd></div>
    <div class="stat"><dt>tempo</dt><dd>{music_info['tempo']:.0f}</dd></div>
    {drop}
    <div class="stat"><dt>runtime</dt><dd>{brief['total_seconds']:.1f}s</dd></div>
    <div class="stat"><dt>shots</dt><dd>{len(brief['shots'])}</dd></div>
    <div class="stat wide"><dt>sound</dt>
      <dd>{escape(Path(music_info['track']).name)}</dd></div>
  </dl>
  <p class="spine">{spine}</p>

  <p class="progress"><b><span id="done">0</span>/{len(brief['shots'])}</b>
    shot &mdash; {escape(order_note + loop_note)}</p>

  {''.join(cards)}

  {notes}

  <section class="rig">
    <h2>Camera, every time</h2>
    <dl>
      <div><dt>4K 16:9</dt><dd>never vertical — vertical throws away all the
        reframing room the edit needs</dd></div>
      <div><dt>4K60</dt><dd>anything you want slowed down. There is no clean slow
        motion without it</dd></div>
      <div><dt>4K30</dt><dd>everything else</dd></div>
      <div><dt>D-Log M</dt><dd>always</dd></div>
      <div><dt>Hold longer</dt><dd>every shot above needs at least a second more
        than its listed length, or the cut has nothing to trim into</dd></div>
    </dl>
  </section>
</div>

<script>
  const boxes = Array.from(document.querySelectorAll('.tick input'));
  const done = document.getElementById('done');
  const update = () => {{ done.textContent = boxes.filter(b => b.checked).length; }};
  boxes.forEach(b => b.addEventListener('change', update));
  update();
</script>
"""


def _escape(value: str) -> str:
    return (str(value).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def copy_text(brief: dict) -> str:
    copy = brief["copy"]
    lines = [f"COPY — {brief['title']}", "=" * 52, "",
             f"source: {copy['source']}", "",
             "HOOK (on screen, first 2 seconds)", f"  {copy['hook']}", ""]
    if copy.get("labels"):
        lines.append("LABELS (in order)")
        lines += [f"  {index + 1}. {line}" for index, line in enumerate(copy["labels"])]
        lines.append("")
    lines += ["CTA (end card)", f"  {copy['cta']}", "",
              "Paste the hook as the first line of your caption too — it is the",
              "line the algorithm reads and the one that shows in the feed.", ""]
    return "\n".join(lines)


def validate(doc: dict, schema_path: Path) -> list[str]:
    try:
        import jsonschema
    except ImportError:
        return ["jsonschema not installed — skipped validation"]
    schema = json.loads(schema_path.read_text())
    validator = jsonschema.Draft202012Validator(schema)
    return [f"{'/'.join(str(p) for p in e.path) or '(root)'}: {e.message}"
            for e in sorted(validator.iter_errors(doc), key=lambda e: list(e.path))]


# ---------------------------------------------------------------- entry


def _print_ranking(ranked: list[tuple[float, dict, list[str]]], limit: int) -> None:
    print(f"\n{BOLD}best fits{RESET}  {DIM}heuristic over five measurements — "
          f"a low score is a warning, not a veto{RESET}\n")
    for score, blueprint, reasons in ranked[:limit]:
        mark = GREEN if score >= 0.85 else (YELLOW if score >= 0.6 else RED)
        print(f"  {mark}{score:.2f}{RESET}  {BOLD}{blueprint['id']:<28}{RESET}"
              f"{DIM}{blueprint['title']}{RESET}")
        print(f"        {DIM}{blueprint['premise']}{RESET}")
        for reason in reasons:
            print(f"        {DIM}· {reason}{RESET}")
        print()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Turn a song into a shot list.")
    ap.add_argument("--track", type=Path, required=True,
                    help="the sound, as audio or video — a screen recording is fine")
    ap.add_argument("--blueprint", default=None, help="skip the ranking and use this one")
    ap.add_argument("--pillar", default=None, choices=["fitness", "travel", "lifestyle"])
    ap.add_argument("--concept", default=None, help="what this reel is about, in a sentence")
    ap.add_argument("--copy", type=Path, default=None,
                    help="JSON with hook/labels/cta, overriding the template lines")
    ap.add_argument("--target", type=float, default=None, help="override the reel length")
    ap.add_argument("--bpm", type=float, default=None)
    ap.add_argument("--blueprints", type=Path, default=config.BLUEPRINTS_DIR)
    ap.add_argument("--out", type=Path, default=config.BRIEF_JSON)
    ap.add_argument("--music-out", type=Path, default=config.MUSIC_MAP_JSON)
    ap.add_argument("--top", type=int, default=3)
    args = ap.parse_args(argv)

    if not args.track.exists():
        print(f"{RED}no track at {args.track}{RESET}", file=sys.stderr)
        print(f"{DIM}Play the sound on its own page in Instagram, screen-record it, "
              f"and AirDrop the file here.{RESET}", file=sys.stderr)
        return 1

    blueprints = load_blueprints(args.blueprints)
    if not blueprints:
        print(f"{RED}no blueprints in {args.blueprints}{RESET}", file=sys.stderr)
        return 1

    chosen = None
    if args.blueprint:
        chosen = next((b for b in blueprints if b["id"] == args.blueprint), None)
        if not chosen:
            print(f"{RED}no blueprint '{args.blueprint}'{RESET}", file=sys.stderr)
            print(f"{DIM}have: {', '.join(b['id'] for b in blueprints)}{RESET}",
                  file=sys.stderr)
            return 1

    # The reel window is placed so the drop lands where the story wants its
    # payoff, so the blueprint has to be known before the track is analysed.
    # Without one, fall back to the plain highest-energy window.
    payoff_at = (chosen or {}).get("song", {}).get("payoff_at")
    target = args.target or (chosen or {}).get("target_seconds") or config.TARGET_REEL_SECONDS

    print(f"\n{BOLD}brief{RESET}  {DIM}{args.track.name}{RESET}\n")
    try:
        music_map = music.analyse(args.track, target, args.bpm, payoff_at)
    except ImportError:
        print(f"{RED}pip install librosa soundfile{RESET}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 — surface the real cause, whatever it is
        print(f"{RED}{exc}{RESET}", file=sys.stderr)
        return 1

    if music_map["tempo_source"] == "failed":
        print(f"{RED}no pulse found — re-run with --bpm{RESET}", file=sys.stderr)
        return 1

    profile = music_map["profile"]
    drop = music_map["drop_at"]
    print(f"  {GREEN}track {RESET}  {music_map['tempo']:.0f} BPM · "
          f"aggression {profile['aggression']:.2f} · vocal {profile['vocal_density']:.2f} · "
          f"{'drop at ' + _timecode(drop) if drop is not None else 'no drop — flat groove'}")

    ranked = rank_blueprints(blueprints, music_map, args.pillar)
    if not chosen:
        _print_ranking(ranked, args.top)
        print(f"{YELLOW}pick one and re-run with --blueprint <id>{RESET}")
        print(f"{DIM}add --concept \"what this reel is about\" to tailor the copy{RESET}\n")
        args.music_out.parent.mkdir(parents=True, exist_ok=True)
        args.music_out.write_text(json.dumps(music_map))
        return 0

    score, reasons = score_blueprint(chosen, profile, drop is not None)
    mark = GREEN if score >= 0.85 else (YELLOW if score >= 0.6 else RED)
    print(f"  {mark}fit   {RESET}  {score:.2f}  {DIM}{'; '.join(reasons)}{RESET}")

    # Second alignment pass. The first used the blueprint's declared payoff
    # position because nothing better was known yet; now the grid exists, the
    # real one can be derived from the quantised layout and the opening bar
    # re-chosen so the drop lands on the cut into the payoff.
    fraction = payoff_fraction(chosen, music_map["beat_period"], target)
    if fraction is not None and drop is not None:
        music_map["best_start"] = round(music.realign(music_map, target, fraction), 4)
        declared = chosen.get("song", {}).get("payoff_at")
        if declared is not None and abs(declared - fraction) > 0.08:
            print(f"  {YELLOW}note  {RESET}  {DIM}{chosen['id']} declares "
                  f"payoff_at {declared:.2f} but its shot weights put the payoff "
                  f"cut at {fraction:.2f} — aligning to the weights{RESET}")

    copy = dict(chosen["copy"])
    copy["source"] = "template"
    if args.copy:
        supplied = json.loads(args.copy.read_text())
        copy.update({k: v for k, v in supplied.items() if k in ("hook", "labels", "cta")})
        copy["source"] = "supplied"

    try:
        brief = lay_out(chosen, music_map, copy, args.target)
    except ValueError as exc:
        print(f"{RED}{exc}{RESET}", file=sys.stderr)
        return 1

    if args.concept:
        brief["concept"] = args.concept

    problems = validate(brief, config.SCHEMAS_DIR / "brief.schema.json")
    if problems:
        print(f"\n  {RED}brief failed validation:{RESET}")
        for problem in problems[:8]:
            print(f"    {problem}")
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(brief, indent=2))
    args.music_out.write_text(json.dumps(music_map))

    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    (config.OUT_DIR / "shotlist.md").write_text(shotlist_markdown(brief))
    (config.OUT_DIR / "shotlist.html").write_text(shotcard_html(brief))
    (config.OUT_DIR / "copy.txt").write_text(copy_text(brief))

    print(f"\n  {BOLD}{'#':<4}{'shot':<18}{'role':<10}{'len':>7}{'at':>8}  "
          f"{'frame':<8}{'camera':<12}text{RESET}")
    for shot in brief["shots"]:
        cue = (shot.get("text") or {}).get("text", "")
        must = f"{RED}*{RESET}" if shot["must_have"] else " "
        print(f"  {shot['index'] + 1:<3}{must}{shot['id']:<18}{shot['role']:<10}"
              f"{shot['duration']:>7.2f}{shot['start']:>8.2f}  "
              f"{shot['framing']:<8}{shot['camera']:<12}{DIM}{cue[:28]}{RESET}")

    for note in brief["notes"]:
        print(f"\n  {YELLOW}note{RESET}  {note}")

    print(f"\n{GREEN}{brief['total_seconds']:.2f}s · {len(brief['shots'])} shots{RESET}  "
          f"{DIM}{RED}*{RESET}{DIM} = required. -> {args.out}, "
          f"{config.OUT_DIR / 'shotlist.md'}, {config.OUT_DIR / 'copy.txt'}{RESET}")
    print(f"\n{GREEN}start the sound at{RESET}  {_timecode(brief['music']['best_start'])} "
          f"{DIM}in Instagram's audio trim{RESET}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
