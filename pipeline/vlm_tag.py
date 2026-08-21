"""Tag each shot's content by sending two sampled frames to a vision model.

This is the one stage that needs a model, and the one judgement signals.py
cannot make: what is actually in the shot, and whether it is interesting. Two
frames per shot at 768px is a few hundred tokens of image each -- cents per reel.

Frames go out at reduced resolution deliberately. The tagging task is scene
classification, not fine detail, so the high-resolution vision tier would cost
several times more tokens for no gain in the answer.

Results are cached per shot on a content hash, so re-running after a sequencing
change costs nothing.

    uv run python -m pipeline.vlm_tag
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from pipeline import config

BOLD, RED, YELLOW, GREEN, DIM, RESET = "\033[1m", "\033[31m", "\033[33m", "\033[32m", "\033[2m", "\033[0m"

# Enough for scene, subject and setting; well below the high-resolution tier.
FRAME_WIDTH = 768
FRAME_QUALITY = 4  # ffmpeg -q:v, 2 (best) to 31

CONTENT_CLASSES = [
    "landscape", "cityscape", "water", "mountain", "forest", "beach", "sky",
    "food", "architecture", "interior", "street", "vehicle", "wildlife",
    "portrait", "group", "action", "workout", "detail", "transition", "other",
]

REEL_ROLES = ["opener", "build", "peak", "closer", "filler"]

SYSTEM_PROMPT = """You tag short video clips for a travel and fitness reel editor.

You will see two frames from one 2-8 second moment, chosen from inside a longer
take. The second frame is that moment's strongest instant. Describe what the
shot contains, not what it could be edited into. Judge the frames you are given.

Guidance on the harder fields:
- content_class: the single dominant subject. A person running on a beach is
  "action", not "beach" -- the moving subject wins over the setting.
- hook_score: how well this shot would hold a viewer in the first second of a
  reel. Reward striking motion, scale, colour, and a clear subject. Do not
  reward technical polish; a sharp but static shot of nothing is a low hook.
- reel_role: where this belongs in a 20-second reel. "opener" stops the scroll
  on its own. "build" carries momentum in the middle. "peak" is the payoff
  moment the reel is built toward. "closer" resolves or lands. "filler" is
  competent but unremarkable -- use it honestly, most shots are filler.
- has_subject: true only if a person or animal is a deliberate subject of the
  frame, not incidental background.

Do not guess at camera movement. You are seeing stills, and the editor measures
motion directly from the footage."""


def _schema_model():
    """Defined lazily so the module imports without pydantic installed."""
    from typing import Literal

    from pydantic import BaseModel, Field

    # A Literal becomes a JSON Schema enum, which structured outputs enforce
    # server-side. sequence.py's variety rule compares these values directly, so
    # a free-text class that drifted ("waterfall" vs "water") would silently stop
    # two adjacent shots from being recognised as similar.
    class ShotTags(BaseModel):
        content_class: Literal[tuple(CONTENT_CLASSES)] = Field(  # type: ignore[valid-type]
            description="The single dominant subject of the shot")
        subject: str = Field(description="The main subject in three words or fewer")
        setting: str = Field(description="Where this was shot, in three words or fewer")
        description: str = Field(description="One plain sentence describing the shot")
        # Numeric bounds are stripped from the schema by the SDK and enforced
        # client-side, so an out-of-range score raises rather than corrupting a rank.
        hook_score: int = Field(ge=1, le=10, description="1-10, how well this opens a reel")
        reel_role: Literal[tuple(REEL_ROLES)] = Field(  # type: ignore[valid-type]
            description="Where this shot belongs in a 20-second reel")
        has_subject: bool = Field(description="Is a person or animal a deliberate subject")
        keywords: list[str] = Field(description="Three to six lowercase keywords")

    return ShotTags


# ---------------------------------------------------------------- frames


def extract_frames(proxy: Path, start: float, end: float, dest_dir: Path,
                   peak: float | None = None) -> list[Path]:
    """Pull two frames: the moment's strongest instant, and one earlier for context.

    The peak is the argmax of moments.py's per-frame score, so the model judges
    the frame the reel will actually be built around rather than an arbitrary
    sample. The first and last frames of a window are the most likely to be
    mid-cut or motion-blurred, so the context frame stays well inside it.
    """
    span = end - start
    if peak is not None and start <= peak <= end:
        stamps = sorted({round(start + span * 0.3, 3), round(peak, 3)})
    else:
        stamps = [start + span * 0.33, start + span * 0.66]
    frames: list[Path] = []

    for index, when in enumerate(stamps):
        out = dest_dir / f"f{index}.jpg"
        proc = subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-ss", f"{when:.3f}", "-i", str(proxy),
             "-frames:v", "1", "-vf", f"scale={FRAME_WIDTH}:-2",
             "-q:v", str(FRAME_QUALITY), str(out)],
            capture_output=True, text=True,
        )
        if proc.returncode == 0 and out.exists():
            frames.append(out)

    return frames


def encode(path: Path) -> dict:
    data = base64.standard_b64encode(path.read_bytes()).decode()
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}}


# ---------------------------------------------------------------- tagging


def cache_key(shot: dict) -> str:
    """Identity of what was actually sent to the model.

    Includes the peak because it selects one of the two frames: a re-run of
    moments.py that moves the peak must re-tag rather than serve a cached answer
    about a frame that is no longer being shown.
    """
    payload = (f"{shot['shot_id']}|{shot['start']}|{shot['end']}|{shot['proxy']}"
               f"|{shot.get('peak')}|{len(REEL_ROLES)}")
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


def tag_shot(client, model: str, shot: dict, schema) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        frames = extract_frames(Path(shot["proxy"]), shot["start"], shot["end"],
                                Path(tmp), shot.get("peak"))
        if not frames:
            raise RuntimeError(f"could not extract frames from {shot['shot_id']}")

        content: list[dict] = [encode(f) for f in frames]
        content.append({
            "type": "text",
            "text": f"These two frames are from one {shot['duration']:.1f}s shot. Tag it.",
        })

        response = client.messages.parse(
            model=model,
            max_tokens=2048,
            # Cached: the system prompt is identical on every shot, so only the
            # frames are billed at full rate after the first call.
            system=[{"type": "text", "text": SYSTEM_PROMPT,
                     "cache_control": {"type": "ephemeral"}}],
            # Classification with a fixed schema — no reason to spend depth here.
            output_config={"effort": "low"},
            messages=[{"role": "user", "content": content}],
            output_format=schema,
        )

    if response.stop_reason == "refusal":
        raise RuntimeError(f"declined: {getattr(response.stop_details, 'category', 'unknown')}")
    if response.parsed_output is None:
        raise RuntimeError(f"no structured output (stop_reason={response.stop_reason})")

    return response.parsed_output.model_dump()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Tag shot content with a vision model.")
    ap.add_argument("--shots", type=Path, default=config.SIGNALS_JSON)
    ap.add_argument("--out", type=Path, default=config.TAGS_JSON)
    ap.add_argument("--cache", type=Path, default=config.WORK_DIR / "tag_cache.json")
    ap.add_argument("--model", default="claude-opus-5",
                    help="use claude-haiku-4-5 for high-volume batches")
    ap.add_argument("--limit", type=int, default=None, help="tag only the first N shots")
    args = ap.parse_args(argv)

    if not args.shots.exists():
        print(f"{RED}no signals.json — run `python -m pipeline.signals` first{RESET}", file=sys.stderr)
        return 1

    shots = json.loads(args.shots.read_text())

    # No key is a soft stop: merge.py treats tags as optional, so the rest of
    # Phase 1 still produces usable clip cards without them.
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        print(f"\n{YELLOW}no ANTHROPIC_API_KEY set — skipping content tagging{RESET}")
        print(f"  {DIM}clip cards will carry technical signals only. To enable:{RESET}")
        print(f"  {DIM}export ANTHROPIC_API_KEY=sk-ant-...{RESET}\n")
        args.out.write_text(json.dumps({}, indent=2))
        return 0

    try:
        import anthropic
    except ImportError:
        print(f"{RED}pip install anthropic pydantic{RESET}", file=sys.stderr)
        return 1

    client = anthropic.Anthropic()
    schema = _schema_model()
    cache = json.loads(args.cache.read_text()) if args.cache.exists() else {}

    todo = shots[: args.limit] if args.limit else shots
    print(f"\n{BOLD}vlm_tag{RESET}  {DIM}{len(todo)} shots · {args.model}{RESET}\n")
    started = time.perf_counter()

    tags: dict[str, dict] = {}
    failures = 0

    for shot in todo:
        key = cache_key(shot)
        if key in cache:
            tags[shot["shot_id"]] = cache[key]
            print(f"  {DIM}cached{RESET}  {shot['shot_id']}")
            continue

        try:
            result = tag_shot(client, args.model, shot, schema)
        except Exception as exc:  # noqa: BLE001 - one bad shot must not lose the batch
            failures += 1
            print(f"  {RED}fail  {RESET}  {shot['shot_id']}  {DIM}{exc}{RESET}")
            continue

        cache[key] = result
        tags[shot["shot_id"]] = result
        print(f"  {GREEN}tag   {RESET}  {shot['shot_id']:<24} {result['content_class']:<12} "
              f"{result['reel_role']:<8} hook {result['hook_score']:>2}  "
              f"{DIM}{result['description'][:44]}{RESET}")

    args.cache.write_text(json.dumps(cache, indent=2))
    args.out.write_text(json.dumps(tags, indent=2))

    mark = RED if failures else GREEN
    print(f"\n{mark}{len(tags)} tagged{RESET}, {failures} failed  "
          f"{DIM}{time.perf_counter() - started:.1f}s -> {args.out}{RESET}\n")
    return 1 if failures and not tags else 0


if __name__ == "__main__":
    sys.exit(main())
