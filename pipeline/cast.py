"""Fit the footage that was shot to the shots that were asked for.

sequence.assign() answers "which of these moments is best, and in what order".
That is the right question when the footage is all there is. Once a brief exists
the question changes to "which moment is the payoff, and did anyone shoot one",
and the difference is not cosmetic: ranking cannot report a gap, so a reel with
no payoff still comes out looking complete.

This stage matches on requirement fit rather than quality, and its most useful
output is the list of slots it could *not* fill.

    uv run python -m pipeline.cast
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from pipeline import config

BOLD, RED, YELLOW, GREEN, DIM, RESET = "\033[1m", "\033[31m", "\033[33m", "\033[32m", "\033[2m", "\033[0m"

# How the fit score is composed. Motion and camera lead because they are
# measured directly from the footage; framing and subject depend on the vision
# tags, which are optional, so they cannot be allowed to dominate a score that
# has to work without them.
WEIGHTS = {"motion": 0.30, "camera": 0.25, "framing": 0.20, "subject": 0.15,
           "quality": 0.10}

CANNOT_FILL = 50.0

# Framing is a five-point scale, so being one step out (close for medium) is a
# near miss and two steps out (detail for wide) is a different shot entirely.
FRAMING_ORDER = ["detail", "close", "medium", "wide"]

# Blueprint camera moves, grouped by what can actually be measured. Optical flow
# gives magnitude, direction and coherence -- enough to separate a locked frame
# from a deliberate move from a wobble, and *not* enough to tell a push-in from
# an orbit. Pretending otherwise would produce confident wrong answers, so the
# three groups here are the honest resolution of the measurement.
CAMERA_CLASSES = {
    "static": "locked",
    "push_in": "moving", "pull_out": "moving", "pan": "moving",
    "orbit": "moving", "follow": "moving", "tilt_reveal": "moving",
    "whip": "moving",
    "handheld": "loose",
}


# ---------------------------------------------------------------- measurement


def flow_of(card: dict) -> tuple[float, float]:
    """Mean magnitude and coherence across the moment's two measured edges."""
    edges = [card.get("entry_flow"), card.get("exit_flow")]
    edges = [e for e in edges if e]
    if not edges:
        return 0.0, 1.0
    magnitude = sum(float(e.get("magnitude", 0.0)) for e in edges) / len(edges)
    coherence = sum(float(e.get("coherence", 1.0)) for e in edges) / len(edges)
    return magnitude, coherence


def motion_of(card: dict) -> float:
    """Absolute movement in the shot, in flow units.

    Prefers `motion_raw`, which is the median over the whole moment. Falls back
    to the edge measurements for cards built before merge.py carried it.
    """
    raw = (card.get("signals") or {}).get("motion_raw")
    if raw is not None:
        return float(raw)
    return flow_of(card)[0]


def camera_class(card: dict) -> str:
    """locked | moving | loose — what the camera was doing, as far as flow shows."""
    magnitude, coherence = flow_of(card)
    if magnitude < config.CAMERA_MOVING_MAGNITUDE:
        return "locked"
    return "moving" if coherence >= config.CAMERA_COHERENT else "loose"


def framing_of(card: dict) -> str | None:
    """The vision tag, when tagging ran. None means unknown, not 'medium'."""
    return (card.get("tags") or {}).get("framing")


# ---------------------------------------------------------------- fit


def motion_fit(card: dict, want: str) -> float:
    low, high = config.MOTION_BANDS.get(want, (0.0, 99.0))
    value = motion_of(card)
    if low <= value < high:
        return 1.0
    span = max(high - low, 0.5)
    distance = (low - value) if value < low else (value - high)
    return max(1.0 - distance / (span * 2.0), 0.0)


def camera_fit(card: dict, want: str) -> float:
    wanted = CAMERA_CLASSES.get(want, "moving")
    actual = camera_class(card)
    if wanted == actual:
        return 1.0
    # A deliberate move where a locked frame was asked for is recoverable in the
    # edit -- the crop can be stabilised, or the move used. The reverse is not:
    # nothing puts movement into a locked-off shot that never had any.
    if wanted == "locked" and actual == "moving":
        return 0.55
    if wanted == "moving" and actual == "loose":
        return 0.45
    return 0.25


def framing_fit(card: dict, want: str) -> float:
    """Distance on the detail-to-wide scale, or neutral when untagged.

    Returns 0.6 rather than 0 for an unknown framing. Scoring it as a failure
    would make every slot look unfillable whenever tagging was skipped, which
    turns a missing *measurement* into a reported missing *shot* -- a coverage
    report that cries wolf is worse than no coverage report.
    """
    actual = framing_of(card)
    if actual is None:
        return 0.6
    if actual == want:
        return 1.0
    if want == "pov" or actual == "pov":
        return 0.3
    try:
        gap = abs(FRAMING_ORDER.index(actual) - FRAMING_ORDER.index(want))
    except ValueError:
        return 0.5
    return max(1.0 - gap * 0.35, 0.0)


def subject_fit(card: dict, want: str) -> float:
    tags = card.get("tags") or {}
    has_tag = "has_subject" in tags
    present = bool(tags.get("has_subject")) if has_tag else card.get("subject_x") is not None

    if want == "optional":
        return 1.0
    if want == "required":
        return 1.0 if present else (0.35 if not has_tag else 0.1)
    return 0.35 if present else 1.0


def fit_score(card: dict, shot: dict) -> float:
    """0..1, how well this moment answers this shot requirement."""
    parts = {
        "motion": motion_fit(card, shot["motion"]),
        "camera": camera_fit(card, shot["camera"]),
        "framing": framing_fit(card, shot["framing"]),
        "subject": subject_fit(card, shot["subject"]),
        "quality": (card.get("signals") or {}).get("technical_score", 0.5),
    }
    return sum(parts[k] * w for k, w in WEIGHTS.items())


def fit_detail(card: dict, shot: dict) -> list[str]:
    """Which axes let a pairing down, for the coverage report."""
    problems = []
    if motion_fit(card, shot["motion"]) < 0.7:
        problems.append(f"wanted {shot['motion']} motion, measured {motion_of(card):.2f}")
    if camera_fit(card, shot["camera"]) < 0.7:
        problems.append(f"wanted {shot['camera']}, camera looks {camera_class(card)}")
    # Only report framing and subject when there was something to check. Without
    # tags every slot would carry the same two lines, which buries the one or two
    # that are about this shot specifically -- the report already says once, at
    # the top, that tagging did not run.
    if (actual := framing_of(card)) and framing_fit(card, shot["framing"]) < 0.7:
        problems.append(f"wanted {shot['framing']} framing, this is {actual}")
    if "has_subject" in (card.get("tags") or {}) and subject_fit(card, shot["subject"]) < 0.7:
        problems.append(f"wanted a subject and there is none in frame")
    return problems


# ---------------------------------------------------------------- assignment


def chronology_index(cards: list[dict]) -> dict[str, float]:
    """Position of each moment in shooting order, 0..1.

    Ordered by source file then timecode. File order stands in for capture order,
    which is right for a card dumped in one go and wrong if files were renamed --
    so this only ever softens the cost, never constrains the assignment.
    """
    ordered = sorted(cards, key=lambda c: (str(c["source"]), float(c["start"])))
    last = max(len(ordered) - 1, 1)
    return {c["shot_id"]: index / last for index, c in enumerate(ordered)}


def build_matrix(cards: list[dict], shots: list[dict], chronological: bool,
                 order: dict[str, float]) -> np.ndarray:
    matrix = np.zeros((len(cards), len(shots)))
    last = max(len(shots) - 1, 1)

    for row, card in enumerate(cards):
        for col, shot in enumerate(shots):
            if card["duration"] + 1e-6 < shot["duration"]:
                matrix[row][col] = CANNOT_FILL + (shot["duration"] - card["duration"])
                continue

            cost = 1.0 - fit_score(card, shot)
            if chronological:
                # A soft nudge, not a rule. Process content breaks when reordered,
                # but a hard constraint would override a genuinely better match on
                # the strength of a filename.
                cost += 0.35 * abs(order.get(card["shot_id"], 0.5) - col / last)
            matrix[row][col] = cost

    return matrix


def cast(cards: list[dict], brief: dict) -> dict:
    from scipy.optimize import linear_sum_assignment

    shots = brief["shots"]
    order = chronology_index(cards) if brief.get("chronological") else {}
    matrix = build_matrix(cards, shots, bool(brief.get("chronological")), order)

    rows, cols = linear_sum_assignment(matrix)
    picked: dict[int, int] = {int(c): int(r) for r, c in zip(rows, cols)}

    slots: list[dict] = []
    for index, shot in enumerate(shots):
        row = picked.get(index)
        card = cards[row] if row is not None else None

        entry = {
            "index": index,
            "id": shot["id"],
            "role": shot["role"],
            "duration": shot["duration"],
            "must_have": bool(shot.get("must_have")),
            "filled": False,
            "shot_id": None,
            "fit": 0.0,
            "problems": [],
        }

        if card is None:
            entry["problems"] = ["no moment available"]
        elif card["duration"] + 1e-6 < shot["duration"]:
            entry["problems"] = [f"nothing long enough — needs {shot['duration']:.2f}s, "
                                 f"best candidate is {card['duration']:.2f}s"]
        else:
            score = fit_score(card, shot)
            entry.update({
                "shot_id": card["shot_id"],
                "source": card["source"],
                "fit": round(score, 4),
                "filled": score >= config.CAST_FIT_FLOOR,
                "problems": fit_detail(card, shot),
            })
        slots.append(entry)

    missing = [s for s in slots if s["must_have"] and not s["filled"]]
    return {
        "version": 1,
        "brief": brief["blueprint"],
        "tagged": any(c.get("tags") for c in cards),
        "slots": slots,
        "complete": not missing,
        "missing": [s["id"] for s in missing],
    }


# ---------------------------------------------------------------- report


def coverage_report(result: dict, brief: dict) -> str:
    lines = [f"# Coverage — {brief['title']}", "",
             f"Blueprint `{result['brief']}` · {len(result['slots'])} shots", ""]

    if not result["tagged"]:
        lines += ["> Content tagging did not run, so framing and subject could not "
                  "be checked. Fit scores below are from motion, camera and "
                  "technical quality alone.", ""]

    lines += ["| # | shot | role | need | fit | filled with | notes |",
              "|---|---|---|---|---|---|---|"]
    for slot in result["slots"]:
        mark = "ok" if slot["filled"] else ("**MISSING**" if slot["must_have"] else "weak")
        source = Path(slot["source"]).name if slot.get("source") else "—"
        notes = "; ".join(slot["problems"]) or ""
        lines.append(f"| {slot['index'] + 1} | {slot['id']} | {slot['role']} | "
                     f"{slot['duration']:.2f}s | {slot['fit']:.2f} {mark} | {source} | {notes} |")

    gaps = [s for s in result["slots"] if not s["filled"]]
    if gaps:
        lines += ["", "## Reshoot list", ""]
        for slot in gaps:
            shot = brief["shots"][slot["index"]]
            required = " **(required)**" if slot["must_have"] else ""
            lines += [f"### {slot['index'] + 1}. {slot['id']}{required}", "",
                      f"{shot['what']}", "",
                      f"`{shot['duration']:.2f}s` · {shot['framing']} · {shot['camera']} · "
                      f"{shot['motion']} motion · subject {shot['subject']}", ""]
            if shot.get("shoot"):
                lines += [f"{shot['shoot']}", ""]
    else:
        lines += ["", "Every slot filled.", ""]

    return "\n".join(lines)


# ---------------------------------------------------------------- entry


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Fit moments to a brief's shot slots.")
    ap.add_argument("--cards", type=Path, default=config.CLIP_CARDS_JSON)
    ap.add_argument("--brief", type=Path, default=config.BRIEF_JSON)
    ap.add_argument("--out", type=Path, default=config.CAST_JSON)
    ap.add_argument("--report", type=Path, default=None)
    ap.add_argument("--allow-gaps", action="store_true",
                    help="build anyway, dropping the slots that could not be filled")
    args = ap.parse_args(argv)

    for path, hint in ((args.cards, "python -m pipeline.merge"),
                       (args.brief, "python -m pipeline.brief --track ...")):
        if not path.exists():
            print(f"{RED}no {path.name} — run `{hint}` first{RESET}", file=sys.stderr)
            return 1

    cards = json.loads(args.cards.read_text())["cards"]
    brief = json.loads(args.brief.read_text())

    print(f"\n{BOLD}cast{RESET}  {DIM}{len(cards)} moments -> {len(brief['shots'])} "
          f"slots of {brief['blueprint']}{RESET}\n")

    if not cards:
        print(f"{RED}no moments to cast{RESET}", file=sys.stderr)
        return 1

    result = cast(cards, brief)

    report = args.report or (config.OUT_DIR / "coverage.md")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    report.write_text(coverage_report(result, brief))

    print(f"  {BOLD}{'#':<4}{'shot':<18}{'role':<10}{'fit':>6}  filled with{RESET}")
    for slot in result["slots"]:
        if slot["filled"]:
            mark, colour = "ok  ", GREEN
        elif slot["must_have"]:
            mark, colour = "MISS", RED
        else:
            mark, colour = "weak", YELLOW
        source = slot["shot_id"] or "—"
        print(f"  {slot['index'] + 1:<4}{slot['id']:<18}{slot['role']:<10}"
              f"{colour}{slot['fit']:>6.2f}{RESET}  {mark}  {DIM}{source}{RESET}")
        for problem in slot["problems"]:
            print(f"        {DIM}· {problem}{RESET}")

    if not result["tagged"]:
        print(f"\n  {YELLOW}no content tags — framing and subject were not checked{RESET}")
        print(f"  {DIM}set ANTHROPIC_API_KEY and re-run vlm_tag for a real coverage "
              f"report{RESET}")

    if result["complete"]:
        print(f"\n{GREEN}every required slot filled{RESET}  {DIM}-> {args.out}, {report}{RESET}\n")
        return 0

    print(f"\n{RED}{len(result['missing'])} required slot(s) unfilled: "
          f"{', '.join(result['missing'])}{RESET}")
    print(f"{DIM}The reshoot list is in {report}. Build anyway with --allow-gaps, "
          f"which drops those slots and shortens the reel.{RESET}\n")
    return 0 if args.allow_gaps else 2


if __name__ == "__main__":
    sys.exit(main())
