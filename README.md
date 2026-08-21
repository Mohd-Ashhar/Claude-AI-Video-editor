# Reel Editor

Story-driven 9:16 reels from Osmo Pocket 3 footage, on a MacBook Air M2 (8 GB, fanless).

**The song comes first.** You give it the sound you are targeting; it measures the
track, matches a story blueprint to the track's shape, and hands back a shot list
with a timecode and an on-screen line for every shot. You shoot that list. It then
casts your footage into the story, dresses it, burns in the text, and tells you
what you failed to shoot.

Full plan and rationale: `~/.claude/plans/kindly-analyse-reel-editor-plan-witty-taco.md`
Step-by-step operating guide: `RUNBOOK.md`

## The loop

```
  the sound you want   ──►  brief   ──►  SHOT LIST  ──►  you shoot
                                                              │
   reel + text + sfx   ◄──  render  ◄──  cast  ◄──  analysis  ◄┘
        │                                  │
        └── out/reel_sfx.mov               └── out/coverage.md — what to reshoot
```

## Status

| Stage | Module | State |
|---|---|---|
| Resource + toolchain guard | `pipeline/preflight.py` | done |
| **Song analysis** (tempo · onsets · sections · drop) | `pipeline/music.py` | done |
| **Story blueprints** (9, as data) | `blueprints/*.json` | done |
| **Brief + shot list** | `pipeline/brief.py` | done |
| Probe and gate sources | `pipeline/ingest.py` | done |
| 480p proxies | `pipeline/proxy.py` | done |
| Shot detection (takes) | `pipeline/scenes.py` | done |
| Dense per-frame measurement | `pipeline/features.py` | done |
| Moment finding | `pipeline/moments.py` | done |
| Quality + motion scoring | `pipeline/signals.py` | done |
| Content tagging (vision model) | `pipeline/vlm_tag.py` | done |
| Validated clip cards | `pipeline/merge.py` | done |
| **Casting + coverage report** | `pipeline/cast.py` | done |
| Assembly + transitions | `pipeline/sequence.py` | done |
| Effects engine (23 effects) | `pipeline/effects.py` | done |
| **Kinetic text** | `pipeline/overlay.py` | done |
| Segment → concat render spine | `pipeline/render.py` | done |
| **Sound design + mix** | `pipeline/sound.py` | done |
| Delivery spec audit | `pipeline/qa.py` | done |
| One-command driver | `pipeline/auto.py` | done |
| Threshold calibration report | `pipeline/calibrate.py` | done |
| Subject framing (residual-flow) | in `features.py` / `sequence.py` | interim |
| Subject-tracking reframe (YOLO) | `pipeline/reframe.py` | not started |
| LLM director (copy + casting taste) | `pipeline/director.py` | not started |

## Run it

### 1. Before you shoot

```bash
source .venv/bin/activate

# What does this track want to be? (screen-record the sound from Instagram;
# audio or video, any container)
python -m pipeline.brief --track assets/song.mov --pillar fitness

# Pick one of the recommendations, and get the shot list
python -m pipeline.brief --track assets/song.mov --blueprint gym_pr_attempt \
    --concept "first 140kg deadlift" --copy work/copy.json
```

That writes `work/brief.json`, plus `out/shotlist.md`, `out/shotlist.html` (open
this on your phone while shooting) and `out/copy.txt`.

### 2. After you shoot

```bash
python -m pipeline.auto --inputs inputs/raw --music assets/song.mov \
    --brief work/brief.json --sound
```

It stops with a reshoot list if a required shot is missing. `--allow-gaps` builds
a shorter reel without it.

Or stage by stage — each is a separate process that exits, which is what keeps
peak memory flat on 8 GB:

```bash
python -m pipeline.preflight                  # is this machine safe to work on?
python -m pipeline.ingest --inputs inputs/raw # probe + gate footage
python -m pipeline.proxy                      # 480p proxies
python -m pipeline.scenes                     # hard cuts -> takes
python -m pipeline.features                   # dense 8 Hz measurement
python -m pipeline.moments --contact-sheet    # -> the 2-8s windows worth using
python -m pipeline.signals                    # score each moment
python -m pipeline.vlm_tag                    # content tags (needs ANTHROPIC_API_KEY)
python -m pipeline.merge                      # -> work/clip_cards.json
python -m pipeline.cast --brief work/brief.json   # -> out/coverage.md
python -m pipeline.sequence --brief work/brief.json --variants 3
python -m pipeline.render --timeline work/timeline_v1.json --draft --mute
python -m pipeline.sound --timeline work/timeline_v1.json --video out/reel_v1.mov \
    --music assets/song.mov
python -m pipeline.qa out/reel_v1.mov --timeline work/timeline_v1.json
```

Without `--brief`, `sequence.py` falls back to the v3 rule engine: a ranked
montage with no story. Better than nothing, and the reason the brief exists.

### Which file to post

| File | Contents | Use |
|---|---|---|
| `out/reel_vN.mov` | silent | safest — add the trending sound in the app |
| `out/reel_vN_sfx.mov` | sound design only | **post this.** Instagram keeps this audio under the sound you add, so the whooshes survive *and* the reel still registers as using the trending audio |
| `out/reel_vN_mixed.mov` | sfx + reference music, −14 LUFS | preview the sync, or post if the music is yours to use |

## The floating-text style

> The complete, measured specification for this look lives in
> [`STYLE.md`](STYLE.md) — reference measurements, the four style packs, the
> grade targets, layout rules, lyric timing, framing and what to shoot.
> `verify.py` asserts that it still states the code's own numbers.

Four style packs, measured off the reels in `Gym-Inspiration/`. They are a
different text model from the caption path: words **accumulate** two to five at a
time at their own anchors and sizes, hold together, and clear as a group.

| Pack | Reference | Faces | Feel |
|---|---|---|---|
| `chrome` | Gym_4 | Helvetica Neue Light | sentence case, cool near-white, deep blood-red accent |
| `editorial` | Gym_2 | Didot + Helvetica Neue Bold | two voices alternating, serif and grotesque |
| `marker` | gym_3 | Marker Felt | brush caps, `#F12109`, hardest cutting |
| `stencil` | Gym_1 | Rockwell + synthesised distress | slab caps with an oversized red word behind |

```bash
uv run python -m pipeline.auto --music assets/song.mov --brief work/brief.json \
    --lyrics "so just *forget about the world tonight" --style chrome
```

Four gym blueprints ship with it: `gym_floating_text` (chrome, 22s),
`gym_editorial_cut` (24s), `gym_marker_cut` (16s), `gym_stencil_cut` (24s).

Three things this depends on, none of them optional:

- **A 16:9 letterbox.** Every reference is exactly 1180×663 inside 1180×2556,
  with zero ink in the bars. `letterbox: 1.778` draws the bars rather than
  scaling the picture, so the subject stays the size it was shot.
- **`night_grade`.** The type has no scrim, stroke or shadow — it is legible only
  because the ground is graded to a mean luma near 38 of 255, which is where all
  four references sit. Over bright footage the words wash out entirely.
- **A subject matte.** `matte.py` runs u2netp on CPU (~105s for a 22s reel) so
  words sit *behind* the lifter. It needs the subject to fill at least 2.5% of
  frame; measured 87–95% usable on real gym footage. Where it fails, the words
  simply draw on top. `--no-matte` skips it.

One more thing the references do and this now does: a **burst** — a run of four
to seven cuts at 0.07–0.23s, declared by the blueprint. It is the only thing
allowed past the 0.30s slot floor.

Install the segmentation runtime once:

```bash
VIRTUAL_ENV=.venv uv pip install onnxruntime
```

## Verify

```bash
python verify.py
```

Runs all 85 gates in a scratch directory, so it never touches a real project's
clip cards or segment cache. Run it after any change to the analysis, assembly or
render path. Every check exists because something actually failed it — see
*Bugs these gates caught*.

## Where the story comes from

`blueprints/*.json` are nine story templates, three per pillar, all sharing one
spine:

```
HOOK (≤2.0s)  →  PROMISE (by 3.0s)  →  BUILD  →  PAYOFF (on the drop)  →  CTA / LOOP
```

They are data, not code, so disagreeing with one is an edit rather than a patch.
Each declares what it needs from a track — a tempo band, whether it needs a drop,
where the payoff falls — and `brief.py` scores all nine against the measured
profile and shows its working. A low score is a warning, never a veto.

The rules encoded in them come from published best-practice guidance on
short-form retention, not from measurements of your account. That makes them a
strong prior and nothing more: after ten posted reels, your own retention curves
are worth more, and the blueprints should be edited to match them.

## What the song analysis has to get right

Everything downstream trusts one number: where the drop is. Two formulations were
tried and the obvious one was wrong.

- **Per-bar energy buckets** put bar length — and therefore the whole section map
  — at the mercy of tempo detection. On a 48s test track with a drop at 24.0s and
  a half-time detection, they found no drop at all. The curve now runs on a fixed
  one-second grid, which no beat tracker can break, and the answer is snapped to
  the nearest downbeat afterwards.
- **Maximising the energy step** finds where the *riser* starts, not where the
  track arrives: it answered 19.5s for that same 24.0s drop. A drop is now the
  first point that *holds* at its plateau, and loudness alone cannot find it —
  a white-noise riser reaches the same level as the drop it builds to. What
  separates them is that a build thins the beat out, so the curve multiplies
  loudness by transient density.

Two more guards earn their place: normalising a flat loop manufactures structure
out of noise (a deliberately uniform groove reported a drop at 8.5s until a
relative-spread floor was added — structured tracks measure 0.70–0.76, flat ones
0.02–0.04), and beat grids are extended across the whole track because trackers
only report where they were confident, which cost the reel its choice of opening
bar.

## Absolute versus relative, three times over

The same bug appeared in three places and is worth naming once. `motion_energy` is
a **percentile rank within the batch** — its minimum is 0.00 and its maximum 1.00
on every batch ever scored. It answers "the most movement here", never "a lot of
movement".

- `cast.py` matches a blueprint's `motion` requirement against **`motion_raw`**,
  the absolute flow magnitude, with bands measured on real footage (gym moments
  span 0.82–6.32, cooking 0.59–2.09).
- `sequence.is_moving()` used to read `motion_energy < 0.45` to decide whether a
  shot needed a push-in. On an all-tripod batch it declared the top half "already
  moving" and withheld the move they all needed; on an all-handheld batch it added
  a push-in to a moving camera, which its own docstring forbids.
- `moments.py` had it as a within-take novelty z-score, where film grain became a
  full-strength content change.

## Finding moments inside a continuous take

Cut detection cannot help here. A 5-minute Pocket 3 clip is one continuous take,
so `scenes.py` correctly reports a single 5-minute shot — and averaging any
metric across those five minutes describes nothing about a clip that is brilliant
for four seconds and unusable for the other 296.

So `moments.py` searches instead of detecting. Every window of every plausible
length is scored against a dense per-frame measurement, and the best
non-overlapping ones survive. The per-frame score is:

```
0.35 · sharpness (ranked within the file)
0.25 · exposure
0.20 · steadiness
0.20 · interest
```

Three of those deserve explanation:

- **Sharpness is ranked within the file**, not across the batch. A five-minute
  take carries its own focus and exposure range; what matters is which seconds of
  *this* clip are the sharp ones.
- **Interest saturates rather than peaking.** More movement is more engaging with
  diminishing returns, gated by how *controlled* it is. There is deliberately no
  ideal magnitude to miss: measured on real footage, handheld gym clips run a
  median of 2.6–5.0 while locked-off cooking close-ups run 0.8–1.0, so any single
  ideal scores one of them badly. An earlier bell curve tuned for one threw out
  68% of the other as "too violent", and the reel came back full of empty rooms —
  because an empty room holds still.
- **Steadiness is temporal, not spatial.** See below.

Each surviving moment records its **peak** (the strongest instant, which slots are
later trimmed around) and its **entry and exit motion** (which decides transitions).

`--contact-sheet` writes a labelled strip of every candidate's peak frame. It is
the fastest way to check whether the scorer agrees with you: one image instead of
scrubbing forty windows.

## Two motion signals that look alike and are not

**Coherence** is spatial — how much of the frame moves together. It separates a
deliberate camera move (the whole frame travels) from a subject moving inside a
locked frame. It does **not** detect shake: a rigidly shaking camera scores a
perfect 1.0, because every pixel really does shake in unison.

**Smoothness** is temporal — how much the direction of travel reverses between
samples. That is what shake actually is, and it is the per-sample form of the
jitter metric `signals.py` scores per shot.

Using coherence where smoothness was needed made the finder rate a shaken clip as
steady. Both signals are now used, for the two different jobs they can each do.

## Pointing the crop at the subject

Centre-cropping a 16:9 frame to 9:16 throws away 2625 px of width, and people do
not stand in the middle of the frame. Until `reframe.py` exists, the crop is aimed
using **residual optical flow** — every flow vector minus the frame's mean, so
what remains is whatever moves independently of the camera.

Raw flow magnitude does not work: hand-held footage moves every pixel at once, so
the centroid tracks the camera and lands mid-frame regardless. Measured on real
gym footage, raw magnitude located a subject in 1 window out of 15; residual flow
found 11 of 15, including the shots where centre-cropping cut the subject in half.

It declines rather than guesses in two cases, both real:

- **Quiet scenes.** It correctly finds the only moving thing — a hand cracking an
  egg — and frames that, leaving the pan the shot is about outside the crop. Below
  `SUBJECT_MIN_MOTION` the crop stays centred.
- **Whole-frame motion.** A pan moves everything, so the centroid means nothing.

The subject sits slightly left of centre, clear of Instagram's action buttons. On
the contact sheet, a yellow box means reframed and grey means centred.

## How transitions are chosen

Every boundary is a **hard cut** unless the measured camera motion earns something
else. A whip only fires when clip A genuinely exits panning the same way clip B
enters — matched within 40°, both coherent, both from shots that scored as steady.
A budget then caps non-cut boundaries at 25%.

| Condition (all measured) | Result |
|---|---|
| Exit and entry motion agree in direction, strongly and coherently | `whip_left` / `whip_right` / `whip_up` |
| Strong motion into a near-static shot, on a downbeat | `flash` |
| Two near-static shots of the same subject | `dissolve` |
| Everything else | **hard cut** |

Constant transitions are the clearest tell of an amateur edit, which is why this
is evidence-driven rather than decorative. `sequence.py` will also make a couple
of small reorderings to *create* matching-motion adjacencies — otherwise transition
selection is a passive observer of an ordering chosen without any knowledge that
matching pairs are valuable.

## Music and the in-app sound

You add the trending audio in Instagram, so the reference track is a grid, not a
deliverable. `music.py` finds the tempo, beat grid, downbeats, and `best_start` —
the downbeat opening the highest-energy stretch, i.e. the drop rather than the intro.

The render produces a **silent file to upload** and a **`_preview` with the track
baked in** so you can check the sync first, plus `out/sync.txt` telling you where
to set the audio start in the app.

Instagram's in-app audio trim is coarse, so alignment may land a beat off. The
mitigation is structural rather than clever: the reel starts on a downbeat and
runs a whole number of bars, so a small offset reads as a choice.

## Calibrate before you trust the motion thresholds

The motion constants are the only numbers here that cannot be derived — they
depend on how you shoot. They were calibrated against test footage, and test
footage is not your footage.

```bash
python -m pipeline.calibrate
```

It reports your actual distribution and suggests values. It changes nothing; you
copy what you agree with into `config.py`.

## Two things this build works around

Verified on this machine:

1. **This ffmpeg has no `libass`, `freetype`, or `zimg`** — so no `subtitles`,
   `ass`, `drawtext` or `zscale` filters. On-screen text will go through
   Pillow-rendered PNGs and the `overlay` filter. HDR normalisation uses
   `colorspace`, not `zscale`.
2. **Colour tags do not stick from `-color_primaries`/`-color_trc` alone.** x264
   needs `colorprim/transfer/colormatrix` in `-x264-params`; VideoToolbox needs the
   `h264_metadata`/`hevc_metadata` bitstream filter. Both are wired into
   `config.FINAL_ENCODERS`, and `qa.py` fails the render if any tag is lost — an
   untagged file washes out badly on Instagram.

Also: `crop` evaluates `w`/`h` once at configuration, not per frame, so a zoom
cannot be done with a time-varying crop size. `crop_x`/`crop_y` *are* per-frame
expressions, which is why panning within a shot works today.

## Why the render is built this way

Each segment renders alone to an all-intra ProRes 422 LT intermediate (hardware
encoded), then everything is concatenated and encoded once. Peak memory stays flat
regardless of clip count, a failed render resumes, and re-sequencing re-renders only
the segments whose spec or source actually changed. Cropping happens at source
resolution before the downscale, so reframing costs no sharpness.

## Bugs these gates caught

Each of these was live in the code and silent — none raised an error:

| Bug | Symptom | Fix |
|---|---|---|
| Transition inputs had no `-t` | Both inputs decoded to end-of-file; a 4.80s timeline rendered 7.70s | Duration is an **input** option |
| Colour tags dropped | `-color_primaries`/`-color_trc` write only the matrix on this ffmpeg | x264 VUI params; `h264_metadata` bitstream filter for VideoToolbox |
| Odd crop width | 4K gives `2160 × 9/16 = 1215` — odd, silently adjusted on chroma-subsampled footage | `config.crop_window()`, rounded even, shared by ingest and render |
| Over-long transition | xfade asked to fade longer than its shorter clip emits a different length | `transition_duration()` clamps and warns |
| Validation crashed on bad input | `_clips_of` raised `KeyError` on the exact input it exists to report | Tolerant of malformed segments by design |
| Stability was batch-ranked | Handed 0.00 to the least steady clip in a set of steady clips | Absolute scale |
| Verify clobbered real work | Wrote into `work/` and wiped the segment cache | Runs in a scratch dir |
| Shake scored as steady | Coherence is spatial; a rigidly shaking camera scores 1.0 on it | Separate temporal `smoothness` signal |
| Novelty was z-scored per take | Inside a uniform take, grain became a full-strength "content change" | Absolute scale — a real cut measures 0.23–0.28, grain never exceeds 0.05 |
| Motion target sat between the modes | `MOTION_TARGET = 1.1` fell in the gap between real pans (0.2–1.0) and shake (2.4–5.0), rewarding motion on the edge of shake | Calibrated to 0.65 against ~1000 measured samples |
| Cuts drifted off the beat | Each segment rounds to whole frames independently; 0.112s accumulated across 22s | Slot boundaries snap to the frame grid; `-frames:v` pins each segment |
| Segments ran one frame long | A duration that is an exact frame multiple includes the frame at the cut point | Explicit `segment_frames()` rather than trusting `-t` |
| A short moment broke the beat lock | A moment shorter than its slot rendered short, shifting every later cut | Slots capped to the longest available moment; underfilled slots dropped |
| Reported transitions were never built | The summary listed proposals, including ones dropped for want of footage | `build_timeline` returns what it actually applied |
| Whips could never fire | Every eligible moment existed, but ordering never placed two adjacent | A bounded reordering pass that creates matching-motion adjacencies |
| Motion scoring was tuned on synthetic clips | Real footage moves ~5x more; 68% of it scored as "too violent" and the reel filled with empty rooms | The bell curve was removed — interest saturates, and shake is judged by smoothness |
| Exposure penalised the grade, not the exposure | A mid-grey target docked D-Log M footage 25–30% for being deliberately dark, with zero shadow clipping | `EXPOSURE_TARGET_LUMA` 70, softer drift weight; clipping carries the score |
| Subject tracking followed the camera | Raw flow moves with the whole frame, so the centroid sat mid-frame — 1 hit in 15 | Residual flow (minus the global mean) — 11 in 15 |

## The ingest gate

A 9:16 crop takes `height × 9/16` pixels of width. From 4K that is 1215 px, which
downscales cleanly to 1080. From a 1080p export it is 607 px — a 78% upscale that
looks fine on a laptop and soft on a phone. Since grading happens upstream (D-Log M →
Mimo → iPhone), **verify that your iPhone export writes 4K.** Sources whose crop width
falls below 1080 are rejected rather than silently upscaled.

## Timeline format

```json
{
  "fps": 30,
  "audio": {"music": "assets/track.wav"},
  "segments": [
    {"kind": "shot", "source": "…", "in": 0.2, "out": 1.4,
     "crop_x": "600+300*t", "speed": 1.0},
    {"kind": "transition", "style": "whip_left", "duration": 0.2,
     "a": {"source": "…", "in": 1.4, "out": 2.2},
     "b": {"source": "…", "in": 0.3, "out": 1.1}}
  ]
}
```

`sequence.py` writes exactly this format, so anything it produces can be hand-edited.
`crop_x`/`crop_y` accept a number or any ffmpeg expression (`t` is segment time).
Transition styles: `fade`, `dissolve`, `flash`, `fade_black`, `whip_left`,
`whip_right`, `whip_up`, `whip_down`, `blur`, `wipe_up`, `zoom_in`, `squeeze`,
`circle_open`, `pixelize`.
