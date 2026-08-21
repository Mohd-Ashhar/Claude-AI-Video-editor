# Reel Editor — working notes

Turns Osmo Pocket 3 4K footage into 9:16 Instagram Reels on a **MacBook Air M2,
8 GB, fanless**. `README.md` is what it does, `RUNBOOK.md` is how to shoot for it.
This file is the things that are expensive to rediscover.

> **Anything about the look — grade, type, layout, cutting, lyrics, framing —
> read [`STYLE.md`](STYLE.md) first.** It is the measured specification for the
> gym reel style: every number in it came off the reference reels or off real
> footage. Do not change a styling constant without reading it, and do not argue
> with it — re-measure instead. `.claude/commands/analyse-reference.md` has the
> method.

Run everything through the venv: `.venv/bin/python -m pipeline.<stage>`, or
`uv run python -m …`. There is no `.venv/bin/pip` — the venv is uv-managed, so
installs are `VIRTUAL_ENV=.venv uv pip install <pkg>`.

**The gate is `python verify.py`.** 176 checks, ~2 minutes. Nothing is done until
it passes. Every check exists because something was silently wrong.

---

## The three rules the design follows

1. **Structure is data, taste is the model, everything measurable stays
   measured.** Blueprints, style packs and schemas are JSON. Copy and casting
   tie-breaks are the only places a model belongs. Anything with a number
   attached gets measured and asserted.
2. **One process per stage, and it exits before the next one starts.** Not
   tidiness — a long-lived process holding OpenCV, librosa, onnxruntime and
   Pillow at once is the difference between a pipeline that runs and one that
   swaps. `auto.py` shells out per stage on purpose.
3. **Report the gap, never fill it quietly.** A `must_have` slot with no footage
   fails the build and prints a reshoot list. A word that will not fit is
   dropped and reported. A matte that found nothing emits black and degrades to
   a plain overlay. Substituting the least-bad thing is how v3 produced a wall
   of dumbbells.

---

## Architecture in one screen

```
song ─> music.py ──> brief.py ──> shotlist          (before you shoot)
                        │
   YOU SHOOT ───────────┘
        │
  ingest → proxy → scenes → features → moments → signals → vlm_tag → merge
        │                                                      └─ solves each
        │                                                         clip's grade
  cast.py  (requirement fit; coverage.md; fails on a missing must_have)
        │
  sequence.py  (slots, bursts, transitions, effect stacks) ──> timeline.json
        │
  render.py  (segment → concat spine, frame-exact; ProRes when text follows)
        │
  matte.py   (u2netp subject masks + subject boxes)      ─┐
  lyrics.py  (word times, phrase grouping)                ├─> compose.py
  compose.py (anchors, sizes, colours -> state PNGs)     ─┘
        │
  lyrics.py burn  (level trim · subject dodge · text behind subject · one encode)
        │
   out/reel_v{1,2,3}.mov
```

**Stage order is load-bearing.** render → **matte** → place → **compose** → burn.
The layout prefers anchors that land on the subject, so the matte has to exist
before the words are positioned. Getting it backwards costs nothing visible — the
layout just silently stops preferring anything.

---

## Filtergraph traps

Every one of these cost real debugging time, and every one failed *silently*.

| Trap | Symptom | Fix |
|---|---|---|
| `overlay` in RGBA leaves the chain 4:4:4 | x264 high profile never opens; "Nothing was written into output file" | append `format=yuv420p` |
| `movie=x.png` emits one frame then EOF | caption appears for 1/30s | `loop=0` **and** `setpts=N/(FPS*TB)` |
| `exposure` hands on `gbrpf32le` | the next filter refuses it, error points elsewhere | convert format explicitly |
| a COMPOSITE-stage effect that is a plain `Filter`, not a `Combine` | compiles, validates, renders, **does nothing** | `build_chain` must collect both |
| **`maskedmerge` negotiates every input down to the mask's format** | a `gray` mask silently delivers a **black-and-white reel** | `alphamerge` + `overlay` instead |

The lesson from the last one: when adding a filter that touches the picture
globally, assert **chroma**, not only luma. Nothing else was looking at colour.

This build has **no `drawtext`, no `subtitles`, no `ass`, no `zscale`**. All type
is rendered in Pillow and composited as a PNG — which turned out better, because
layout measured on real glyphs beats any drawtext expression.

---

## Signals: absolute vs relative

`signals.motion_energy` is a **percentile rank within the batch**. It always
spans 0.00–1.00 whatever was shot, so a threshold on it means nothing. On tripod
footage it called the top half "already moving"; on all-handheld footage it
called the bottom half static. **Use `motion_raw` for any threshold.**

Motion thresholds are footage-dependent (gym and cooking differ 3–4×). Run
`pipeline/calibrate.py` before trusting them on new material.

---

## Caching

**`auto.py` skips any stage whose output already exists.** That is what makes
re-running after dropping in one clip cheap, and it is how a build pointed at a
new folder silently rebuilt the *previous* project and reported success.
`sources.json` records the directory it scanned; `auto.stale_inputs()` forces a
re-analysis when it changes. Segments cache separately — clear `work/segments`
when a grade or effect constant moves, or you will measure an old render.

---

## Where things live

| Path | What |
|---|---|
| `STYLE.md` | ★ the measured styling specification — read before touching the look |
| `pipeline/` | one module per stage, each runnable as `-m` |
| `blueprints/*.json` | 14 story templates; four `gym_*` ones carry the v5 style |
| `styles/*.json` | 4 typographic identities: `chrome`, `editorial`, `marker`, `stencil` |
| `schemas/*.json` | the contracts; everything crossing a stage boundary validates |
| `Gym-Inspiration/`, `Sample/` | reference reels, kept for re-measurement |
| `Gym-Input/` | the user's own footage, by reel |
| `verify.py` | the gate |
| `.claude/commands/` | `new-reel` (the interaction contract), `analyse-reference` (the forensic method) |

Keep the solver clamps and the schema ranges in step. They diverged once and
`merge.py` began writing cards its own contract rejected — surfacing two stages
later as a missing `clip_cards.json`.
