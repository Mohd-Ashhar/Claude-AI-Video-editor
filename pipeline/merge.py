"""Combine signals and tags into clip_cards.json, validated against the schema.

The single artifact every later stage reads. Sequencing, reframing and rendering
all work from these cards rather than re-deriving anything, which is what keeps
the stages independently runnable.

Ranking weights are an input, not a constant: the orchestrator sets them from the
project brief (a fitness reel wants motion, a landscape reel wants technical
polish), and they are recorded in the output so any ranking can be explained.

    uv run python -m pipeline.merge
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from pipeline import config, media

BOLD, RED, YELLOW, GREEN, DIM, RESET = "\033[1m", "\033[31m", "\033[33m", "\033[32m", "\033[2m", "\033[0m"

DEFAULT_WEIGHTS = {"technical": 0.5, "hook": 0.3, "motion": 0.2}

SIGNAL_FIELDS = ("technical_score", "sharpness_score", "exposure_score",
                 "stability_score", "motion_energy", "motion_raw", "jitter",
                 "audio_rms", "series")


def rank_score(signals: dict, tags: dict | None, weights: dict) -> float:
    """Blend technical quality, content appeal, and motion into one 0..1 score.

    When tagging was skipped the hook term has no source, so its weight is
    redistributed across the remaining terms rather than scored as zero — an
    untagged batch should rank on what was actually measured.
    """
    parts = {
        "technical": signals["technical_score"],
        "motion": signals["motion_energy"],
    }
    if tags and "hook_score" in tags:
        parts["hook"] = (tags["hook_score"] - 1) / 9.0

    active = {k: w for k, w in weights.items() if k in parts}
    total = sum(active.values())
    if total <= 0:
        return 0.0
    return round(sum(parts[k] * w for k, w in active.items()) / total, 4)


def build(shots: list[dict], tags: dict, weights: dict,
          grade: bool = True) -> dict:
    cards = []
    for shot in shots:
        signals = shot["signals"]
        card_tags = tags.get(shot["shot_id"])
        card = {
            "shot_id": shot["shot_id"],
            "source": shot["source"],
            "proxy": shot["proxy"],
            "start": shot["start"],
            "end": shot["end"],
            "duration": shot["duration"],
            "width": shot["width"],
            "height": shot["height"],
            "fps": shot["fps"],
            "max_crop_width": shot["max_crop_width"],
            "signals": {k: signals[k] for k in SIGNAL_FIELDS if k in signals},
            "tags": card_tags,
            "rank_score": rank_score(signals, card_tags, weights),
        }
        # Present when the card came from moments.py rather than whole takes.
        # sequence.py trims each slot around `peak` and chooses transitions from
        # the edge motion, so these must survive onto the card.
        for field in ("take_id", "peak", "window_score", "subject_x",
                      "entry_flow", "exit_flow"):
            if field in shot:
                card[field] = shot[field]

        # Grade offsets are a property of this clip, not of the reel, so they are
        # solved once here and travel on the card. Measured on the proxy over the
        # moment's own window: two clips from the same shoot can differ by 45
        # points of luma, and a single fixed grade cannot serve both.
        if grade:
            card["grade"] = media.fit_grade(
                shot["proxy"], shot["start"], min(shot["duration"], 4.0))

            # Where a letterbox band should sit on this clip. Only ever consulted
            # when the crop is shorter than the source, which means a letterboxed
            # reel; a 9:16 reel crops full height and effects.py ignores this.
            # Measured for the same reason the grade is: config.SUBJECT_CENTRE_Y
            # is a median over nine landscape clips and put the band on the hands
            # of a portrait selfie, cutting the face off entirely.
            strip = config.strip_height(16 / 9)
            _, band_h = config.crop_window(shot["width"], shot["height"],
                                           config.OUT_W / strip if strip else None)
            band = band_h / max(shot["height"], 1)
            if band < 0.995:
                centre = media.fit_subject_y(
                    shot["proxy"], shot["start"], min(shot["duration"], 4.0),
                    band=band)
                if centre is not None:
                    card["subject_y"] = centre
        cards.append(card)

    cards.sort(key=lambda c: -c["rank_score"])
    return {
        "version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "weights": weights,
        "cards": cards,
    }


def validate(doc: dict, schema_path: Path) -> list[str]:
    try:
        import jsonschema
    except ImportError:
        return ["jsonschema not installed — skipped validation"]

    schema = json.loads(schema_path.read_text())
    validator = jsonschema.Draft202012Validator(schema)
    return [f"{'/'.join(str(key) for key in e.path) or '(root)'}: {e.message}"
            for e in sorted(validator.iter_errors(doc), key=lambda e: list(e.path))]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build validated clip cards.")
    ap.add_argument("--signals", type=Path, default=config.SIGNALS_JSON)
    ap.add_argument("--tags", type=Path, default=config.TAGS_JSON)
    ap.add_argument("--out", type=Path, default=config.CLIP_CARDS_JSON)
    ap.add_argument("--schema", type=Path,
                    default=config.SCHEMAS_DIR / "clip_cards.schema.json")
    ap.add_argument("--no-grade", action="store_true",
                    help="skip the per-clip grade solve (about a second a clip)")
    for name, value in DEFAULT_WEIGHTS.items():
        ap.add_argument(f"--w-{name}", type=float, default=value,
                        help=f"weight for {name} (default {value})")
    args = ap.parse_args(argv)

    if not args.signals.exists():
        print(f"{RED}no signals.json — run `python -m pipeline.signals` first{RESET}",
              file=sys.stderr)
        return 1

    shots = json.loads(args.signals.read_text())
    tags = json.loads(args.tags.read_text()) if args.tags.exists() else {}
    weights = {name: getattr(args, f"w_{name}") for name in DEFAULT_WEIGHTS}

    print(f"\n{BOLD}merge{RESET}  {DIM}{len(shots)} shots, {len(tags)} tagged · "
          f"weights {weights}{RESET}\n")

    doc = build(shots, tags, weights, grade=not args.no_grade)

    problems = validate(doc, args.schema)
    if problems:
        print(f"{RED}schema validation failed:{RESET}")
        for problem in problems[:10]:
            print(f"    {problem}")
        return 2

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2))

    print(f"  {BOLD}{'rank':<6}{'shot':<26}{'score':>7}{'tech':>8}{'motion':>8}"
          f"  class{RESET}")
    for index, card in enumerate(doc["cards"], start=1):
        klass = (card.get("tags") or {}).get("content_class", "—")
        print(f"  {index:<6}{card['shot_id']:<26}{card['rank_score']:>7.2f}"
              f"{card['signals']['technical_score']:>8.2f}"
              f"{card['signals']['motion_energy']:>8.2f}  {DIM}{klass}{RESET}")

    if not tags:
        print(f"\n  {YELLOW}no content tags — ranked on technical signals only{RESET}")

    if not args.no_grade:
        seen = sum(1 for c in doc["cards"] if (c.get("grade") or {}).get("subject_seen"))
        mark = GREEN if seen >= len(doc["cards"]) * 0.6 else YELLOW
        print(f"\n  {mark}grade{RESET}  {seen}/{len(doc['cards'])} clips had a "
              f"findable subject to expose for "
              f"{DIM}(the rest are graded on the frame alone){RESET}")

    print(f"\n{GREEN}{len(doc['cards'])} clip cards{RESET}  "
          f"{DIM}schema-valid -> {args.out}{RESET}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
