---
description: Start a reel — song first, then blueprint, then a shot list to shoot from
---

The interaction contract for making a reel. It is song-first: the track decides
the structure, and the shot list exists before any footage does. Follow it in
order and do not skip to editing.

## 1. Ask for the sound

> "What sound are you targeting?"

They screen-record it from Instagram (open the **sound's own page**, so it comes
clean rather than over someone's voiceover) and drop the file in `assets/`. Any
container — an iOS screen recording arrives as AAC-in-.mov and `music.py`
normalises it through ffmpeg before librosa touches it.

## 2. Measure it and report back

```
uv run python -m pipeline.brief --track assets/<file> --pillar fitness --concept "..."
```

Come back with: tempo, where the drop is, what kind of track it is, and the
**top three blueprints it fits, with the reason each scored**. Say plainly that
the recommendation is a heuristic.

For the floating-text look, the four gym blueprints are:

| Blueprint | Style | Feel |
|---|---|---|
| `gym_floating_text` | `chrome` | light sans, sentence case — the default |
| `gym_editorial_cut` | `editorial` | Didone serif against bold grotesque |
| `gym_marker_cut` | `marker` | brush caps, hot red, hardest cutting, 16s |
| `gym_stencil_cut` | `stencil` | distressed slab, oversized red backdrop word |

## 3. They pick one, and say what the reel is about in a sentence

## 4. Hand over the shot list

`out/shotlist.md` for the repo; publish the phone version as an Artifact — one
card per shot, readable at arm's length in a gym. Header carries the song, BPM,
the exact in-app trim timecode, total runtime, and the camera settings from
`RUNBOOK.md` Part 2.

Tell them the two constraints that decide whether the style works at all:
- **The subject must fill at least 2.5% of frame** for words to sit behind it.
  Measured 87–95% usable on real gym footage, far lower on wides.
- **Empty, dark space on one side of frame is not wasted** — it is where the
  words go. Shots that fill the frame edge to edge leave the layout nowhere.

## 5. They shoot, and drop the files in `inputs/raw/`

## 6. Build it

```
uv run python -m pipeline.auto --music assets/<file> --brief work/brief.json \
    --lyrics "so just *forget about the world tonight" --style chrome
```

Prefix a word with `*` to accent it. Drop `--style` to use the blueprint's.
`--no-matte` skips the ~4s-per-second segmentation and draws words on top.

## 7. Report the coverage honestly

Read `out/coverage.md` and tell them **what to reshoot**, not just what worked.
Then hand over the three variants and the in-app trim timecode from
`out/sync.txt`.

## What not to do

- Do not fill a `must_have` slot with the least-bad wide shot. The build failing
  is the brief doing its job.
- Do not add captions to a lyric-mode reel. Two text systems fight for the same
  centre of frame; `text_mode: "lyrics"` suppresses per-shot cues for this reason.
- Do not report a reel as done without running `python verify.py`.
