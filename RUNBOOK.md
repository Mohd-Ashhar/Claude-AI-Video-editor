# Reel Editor — Runbook

How to go from a trending sound to a posted reel, on a MacBook Air M2 with 8 GB.

**The order matters and it is not the obvious one.** The song comes first, then the
shot list, then the shoot, then the edit. Shooting first and asking for a reel
afterwards still works, but you get a ranked montage with no story — which is
exactly the thing the shot list exists to stop.

**What it still does not do:** reframe the 9:16 crop around a moving subject. That
is `reframe.py`, and it is not built. You can pan by hand today — see Part 7.

---

## Part 0 — The sound, first

Everything is planned from the track, so this is the step that cannot be skipped.

**Getting the audio.** On your phone: open the sound's *own page* in Instagram —
not a reel that uses it, or you will record someone's voiceover over the top —
start an iOS screen recording, let the sound play for 30 seconds, stop, and AirDrop
the video to the Mac. iOS screen recording captures in-app audio, so this is clean.

Drop it in `assets/`. Audio or video, any container: the pipeline extracts and
normalises the track through ffmpeg before anything touches it.

```bash
python -m pipeline.brief --track assets/song.mov --pillar fitness
```

That prints what the track actually is — tempo, how aggressive, how vocal, where
the drop is — and the three story blueprints that fit it best, with reasons. The
score is a heuristic over five measurements and it says so; a low score is a
warning, not a veto.

```bash
python -m pipeline.brief --track assets/song.mov --blueprint gym_pr_attempt \
    --concept "first 140kg deadlift"
```

Now you have:

| File | What it is |
|---|---|
| `out/shotlist.html` | **open this on your phone.** One card per shot, with a checkbox |
| `out/shotlist.md` | the same list, for the repo |
| `out/copy.txt` | the hook, the labels and the CTA, for your caption |
| `work/brief.json` | the contract the edit is built against |

### Writing your own copy

The template lines are placeholders. To use your own, write a small JSON file and
pass `--copy`:

```json
{
  "hook": "I have never lifted this before",
  "labels": ["last attempt: failed", "one shot at it", "no spotter"],
  "cta": "Follow for the next one"
}
```

Keep the hook to **eight words or fewer** — the brief warns you if it is longer,
because it has to be readable in the two seconds it is on screen.

### The nine blueprints

| Pillar | Blueprint | Needs from the track |
|---|---|---|
| fitness | `gym_pr_attempt` | a real drop, ≥120 BPM |
| fitness | `gym_form_breakdown` | steady, vocal-light |
| fitness | `gym_motivation` | aggressive, high variance |
| travel | `travel_reveal` | a real drop |
| travel | `travel_day_in_place` | flat or gentle — survives no drop |
| travel | `travel_transition_showcase` | hard, regular transients |
| lifestyle | `lifestyle_routine` | steady, mid-tempo (recipes live here) |
| lifestyle | `lifestyle_pov_day` | vocal-led |
| lifestyle | `lifestyle_quote_card` | slow, emotional |

---

## Part 1 — One-time setup

```bash
cd ~/Desktop/AI-Vid-Edit
source .venv/bin/activate
```

Add to your `~/.zshrc` if you want content tagging (optional — everything works
without it, ranked on measured signals alone):

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

Keep intermediates on the project disk so they are visible and cleanable:

```bash
export TMPDIR=~/Desktop/AI-Vid-Edit/work/tmp
```

When the external SSD arrives, this is a config change and not a code change:

```bash
export REEL_INPUTS_DIR=/Volumes/SSD/project/raw
export REEL_WORK_DIR=/Volumes/SSD/project/work
```

Confirm the toolchain once:

```bash
python verify.py        # 43 checks, ~35s
```

---

## Part 2 — Shooting

| Setting | Use | Why |
|---|---|---|
| **4K 16:9, not 3K vertical** | always | Vertical is sharper per pixel and throws away all reframing room. 4K 16:9 gives a 1214 px-wide 9:16 window *and* 2625 px of horizontal latitude to pan within. |
| **4K60** | anything you plan to slow down | Retimed 30 fps judders. There is no slow motion without it. |
| **4K30** | everything else | Renders 1:1 to a 30 fps timeline with no conversion. |
| **D-Log M** | always | Your grading chain expects it. |

Shoot deliberately long takes. The system is built to search *inside* a continuous
take, so a steady 40-second push through a market is more useful than eight
snatched 5-second clips.

Give it movement it can use: a slow, confident pan or push-in scores highest.
Dead-static shots and handheld wobble both score low, by design.

### Shooting the list

Work down `out/shotlist.html` and tick shots off. Four things decide whether the
edit can use what you bring back:

1. **Hold every shot at least a second longer than its listed length.** A moment
   exactly as long as its slot renders short, and every cut after it lands off the
   beat. This is the single most common reason a slot comes back unfilled — on the
   first real project, the payoff needed 3.97s and the longest moment in six files
   was 3.60s.
2. **Shoot the shots marked `required`.** Casting refuses to fill those with the
   least-bad alternative; it stops and tells you what to reshoot instead.
3. **Match the `camera` field.** The system can tell a locked frame from a
   deliberate move from a wobble, and no finer than that — it does not pretend to
   distinguish a push-in from an orbit. So "static" versus "moving" is what it
   actually checks, and getting that one right matters.
4. **4K60 for anything marked for slow motion.** There is no clean slow motion
   from 30 fps footage, only interpolation, and it is capped at three uses a reel
   because it costs 30× realtime.

Shoot extra beyond the list. Casting picks the best moment for each slot from
everything you bring, so more coverage is never wasted.

---

## Part 3 — Grading (unchanged)

D-Log M → recover colour in DJI Mimo → grade on iPhone 15 → export.

> **Check this once, before a real shoot: your iPhone export must be 4K.**
>
> A 1080p export yields a 607 px-wide crop, which needs a 78% upscale to fill the
> frame. It looks fine on the laptop and soft on a phone. `ingest` rejects it, but
> finding out after a shoot day is expensive.

This chain is three lossy generations before Instagram re-encodes. It holds up for
most content; it shows on skies and sunsets as banding. Export at maximum quality.

---

## Part 4 — Make a reel

Copy the graded exports in, and run one command:

```bash
cp ~/Downloads/graded/*.mov inputs/raw/

python -m pipeline.auto --inputs inputs/raw --music assets/song.mov \
    --brief work/brief.json --sound
```

That runs everything: gate → proxies → takes → dense measurement → moment finding
→ scoring → tagging → **casting** → assembly → drafts → **sound design**.

Roughly **2 minutes for 10 minutes of footage**, plus about 20 seconds per draft.

If a required shot is missing, it stops with exit code 2 and points at
`out/coverage.md`. That is the brief working, not a crash — read the reshoot list,
or pass `--allow-gaps` to build a shorter reel without that shot.

| Flag | Effect |
|---|---|
| `--brief work/brief.json` | build the story. Without it you get a ranked montage |
| `--allow-gaps` | build even if a required shot was never filmed |
| `--sound` | also write the sound-design and mixed audio versions |
| `--target 25` | reel length in seconds (the blueprint's own target wins) |
| `--variants 3` | with a brief, these differ in effect intensity, not in order |
| `--bpm 120` | force a tempo when the track has no clear pulse |
| `--no-render` | stop after sequencing; just write the timelines |
| `--force` | redo every stage instead of reusing cached output |
| `--allow-partial` | continue past rejected source files |

With `--brief`, the music stage is skipped: `brief.py` already wrote the music map,
and re-running it would overwrite the opening bar it chose to put the drop under
your payoff.

### The variants

With a brief, the story, the order and the length are all decided before any
footage exists, so the three variants differ in **effect intensity** — `standard`,
`hype`, `calm`. A variant that reordered the shots would be telling a different
story, not offering the same one twice.

---

## Part 4b — Which file to post

Four files come out per variant, and the interesting one is not the obvious one.

| File | Contents | Use |
|---|---|---|
| `out/reel_vN.mov` | silent | safest. Add the trending sound in the app |
| `out/reel_vN_sfx.mov` | whooshes and impacts, no music | **post this** |
| `out/reel_vN_mixed.mov` | sfx + the reference track, −14 LUFS | check the sync, or post if the music is yours to use |
| `out/reel_vN_preview.mov` | reference track baked in | check the sync |

**Why the sfx version.** Instagram keeps a reel's original audio underneath the
sound you add, at a volume you control. So posting the sfx-only file and adding the
trending audio in-app gets you both: the whooshes and impacts survive, *and* the
reel still registers as using that sound — which is most of the reason you chose a
trending one. Baking the music in loses that, and it is a screen recording of
Instagram's own audio, so it is not the file to publish.

Set the audio start point from `out/sync.txt` — that is the timecode to dial into
Instagram's audio trim, and it is exact rather than computed because the analysis
ran on the same recording you are trimming.

Output is QuickTime `.mov`, which AirDrops straight into Photos. Set
`REEL_OUT_CONTAINER=.mp4` if you ever want the other wrapper; the streams inside
are identical, so it makes no difference to quality.

---

## Part 5 — Check the picks

Before trusting the assembly, look at what it found:

```bash
open work/contact_sheet.jpg
```

One labelled frame per candidate moment, with its score and timecode. If the top
picks are not the ones you would have chosen, the fix is Part 9 (calibration),
not hand-editing.

---

## Part 6 — Choose a variant

```bash
open out/reel_v1.mov out/reel_v2.mov out/reel_v3.mov
```

**Watch them on a phone, not the Mac.** Framing and safe-zone problems are
invisible on a laptop and obvious on the device people actually use.

The three differ in their opening shot, their ordering, and their pacing arc
(`standard`, `fast`, `breathe`). The hook is the biggest difference and the one
that matters most — the first two seconds decide whether the rest is seen.

Then render your pick at final quality:

```bash
python -m pipeline.render --timeline work/timeline_v2.json --out out/reel.mov --mute
python -m pipeline.qa out/reel.mov --timeline work/timeline_v2.json
```

---

## Part 7 — Adjusting by hand

`sequence.py` writes exactly the format you would write yourself, so anything it
produces can be edited. Open `work/timeline_v2.json`:

```json
{
  "kind": "shot", "source": "inputs/raw/GYM_01.mov",
  "in": 15.733, "out": 18.133, "crop_x": "600+300*t"
}
```

| Field | Meaning |
|---|---|
| `in` / `out` | source seconds. Keep the length identical or the beat lock breaks. |
| `crop_x` | horizontal position of the 9:16 window, `0`–`2626` on 4K. A number, or an expression in `t` (segment time). |
| `speed` | `2.0` is double speed. Only sensible on 4K60 source. |

`crop_x` is the cheapest way to make a reel feel edited rather than assembled:

- `"600+300*t"` — pans right at 300 px/second
- `"1300-200*t"` — pans left
- `1313` — centred and locked (the default)

Re-render after editing. Only the segments you changed re-render; the rest come
from cache, so it takes seconds.

> **If you change a clip's length, change it by whole beats.** The reel is built on
> a beat grid and starts on a downbeat; an arbitrary trim puts every later cut off
> the beat.

---

## Part 8 — Post it

```bash
cat out/sync.txt
```

It gives you the tempo and the exact point in the track the reel starts on.

1. Upload **`out/reel.mov`** (the silent one).
2. Add the trending sound in the app.
3. Open the sound's trim control and set its start to the timecode in `sync.txt`.
4. Post.

The app's audio trim is coarse, so you may land a beat off. The reel is built to
survive that: it starts on a downbeat and runs a whole number of bars, so a small
offset reads as a stylistic choice rather than a mistake.

---

## Part 9 — Calibrating to your footage

The motion thresholds are the only numbers in the system that cannot be derived —
they depend on how you shoot. They are calibrated against test footage, and test
footage is not your footage.

Run this once, after your first real project:

```bash
python -m pipeline.calibrate
```

It prints your actual motion distribution, shows how the current settings score it,
and suggests values. **It changes nothing** — you copy what you agree with into
`pipeline/config.py`.

Read it for two things:

- If **almost everything scores below 0.1 on the interest curve**, `MOTION_TARGET`
  is wrong for how you shoot, and good moments are being rejected as static.
- If **no sample clears the whip bar**, directional transitions can never fire and
  every boundary will be a hard cut.

---

## Part 10 — Between projects

```bash
rm -rf work/proxies work/segments work/features
```

Keep `work/clip_cards.json` and the timelines if you might revisit the edit; they
are small. The proxies and segments are the bulk and regenerate cheaply.

Pocket 3 4K writes about 1 GB per minute, and `preflight` refuses to start below
40 GB free. One project at a time until the SSD arrives.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `blocked — ram pressure` | Under 3 GB free | Close Chrome and VS Code; run from Terminal. `--skip-preflight` overrides, expect swapping. |
| `REJECT … crop-width 607px` | A 1080p export | Re-export at 4K. Do not upscale. |
| `no usable moments found` | Everything fell below the quality floor | Check focus and stability, or lower `FRAME_SCORE_FLOOR`. |
| `only N moments for M slots` | Not enough good footage | It builds a shorter reel rather than reusing shots. Shoot more, or lower `--target`. |
| `slot(s) dropped — no moment long enough` | Moments shorter than the pacing arc wants | Normal with short takes. Shoot longer, or lower `--target`. |
| Every boundary is a hard cut | No matching motion, or thresholds too high | Expected on static footage. Check Part 9. |
| `no pulse found in this track` | The beat tracker failed | Re-run with `--bpm 120`. |
| `music` seems to hang | librosa compiles its kernels on first use | Wait a few seconds. Only happens once per session. |
| Cuts drift off the beat | A hand-edited clip length | Change lengths by whole beats only. |
| Colour looks washed out | A lost colour tag | `qa.py` fails on this. Re-render; do not upload. |
| Something is wrong and you cannot tell what | — | `python verify.py` separates "the tool is broken" from "my input is wrong". |

---

## What is coming

**`reframe.py`** — YOLO11n on the Neural Engine plus ByteTrack on the proxies,
smoothing a subject path into the `crop_x` expressions you currently write by hand.
This is where the 2625 px of pan latitude in your 4K footage starts paying off.

**Phase 3** — an audio bed with ducked nat sound, Pillow-rendered text overlays
(this ffmpeg has no libass), a chosen cover frame, and the VMAF encoder bake-off
that settles x264 against VideoToolbox for good.


---

## Shooting for the floating-text style

The style puts words *inside* the frame with the subject, so what you shoot
decides whether it can work at all. Four things, in order of how often they go
wrong.

**Leave empty space.** Dark, uncluttered area on one side of frame is where the
words go. A shot that fills the frame edge to edge leaves the layout nowhere to
put anything, and it will drop words rather than overlap them.

**Get the subject big enough.** Words sit behind the lifter only where the
segmentation finds a subject filling at least 2.5% of frame. Measured on real
gym footage: 87–95% of frames usable on mediums and closes, far lower on wides.
A distant figure gets no occlusion and the effect collapses to a caption.

**Backlight.** Every reference is lit with a hard rim and almost nothing else.
The grade takes the picture down to a mean luma near 38 of 255, and a rim on the
shoulders is what survives that. Flat overhead gym light does not.

**Shoot the burst as one take.** The burst slot is cut into four to seven
fragments of two to seven frames each, drawn from different points in a single
moment. Every fragment needs visible motion — a still frame inside a burst reads
as a dropped frame. One long continuous take, not seven short ones.

And one for the payoff: **shoot it at 4K60**. The references decelerate into
their payoff and hold, rather than speeding into it — 17 decelerating ramps
against 2 accelerating across the four. That ramp needs the extra frames.
