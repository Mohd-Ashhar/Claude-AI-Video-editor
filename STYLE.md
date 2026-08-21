# The gym reel style — measured

Everything here was measured off the four reels in `Gym-Inspiration/` and off
real footage in `Gym-Input/`. **These numbers are the specification.** If output
stops matching them, re-measure — do not re-argue. The method for measuring a
new reference is in `.claude/commands/analyse-reference.md`.

Last measured: 2026-08-21. Gate: `python verify.py` (180 checks).

---

## 1. What the four references share

| Property | Gym_1 | Gym_2 | gym_3 | Gym_4 |
|---|---|---|---|---|
| Duration | 32.4s | 27.4s | 15.7s | 28.1s |
| Picture strip | 1180×663 inside 1180×2556 — **exactly 16:9** | same | same | same |
| Ink outside the strip | **0 px** | 0 px | 0 px | 0 px |
| Cuts | 25 | 25 | 23 | 21 |
| Shot length min / med / max | 0.36 / 1.10 / 3.56s | 0.20 / 1.00 / 2.30s | **0.07** / 0.24 / 2.34s | 0.14 / 1.47 / 3.47s |
| Burst run | 6.27→7.73s | 16.27→17.87s | 6.03→7.33s | 9.70→11.30s |
| Beat-grid fit | 180.4 BPM, 20/25 within 80ms | 193.8, 16/25 | 196.0, 12/23 | 199.8, 17/21 |
| Text changes on a cut | 56% | 67% | 92% | 60% |
| Frame luma /255 | 34.7 | 35.6 | 39.9 | 38.5 |
| **Subject luma** | **53.0** | — | — | **84.1** |
| **Separation** | **1.51×** | — | — | **2.08×** |
| Colour cast (B−R) | +13 | +9 | +12 | +12 |
| Chroma spread | 26.9 | — | — | — |
| Speed ramps | 7 dec / 0 acc | 3 / 1 | 4 / 0 | 3 / 1 |
| Occlusion matte | yes | yes | yes | yes |

**Words co-resident:** 2–5, each at its own anchor and size, all clearing
together. **Glyph heights 41→351px on a 663px strip — a 6–8× spread inside one
reel.** Anchors spread x 0.12–0.83, y 0.09–0.91.

---

## 2. The five things that define the look

1. **A 16:9 strip.** 1080×608 picture, black above and below, zero ink in the
   bars. Made by padding the picture, never by drawing bars over a 9:16 crop.
2. **Text as composition, not sequence.** Words accumulate and clear as a group.
   Gym_4 at 3.2–6.4s: `So` → `So/Just` → `+Forget` (red) → `+About` (huge) →
   `+The` → clear.
3. **Separation.** A lit subject against a crushed room. Not merely a dark
   picture — this is the part that reads as "cinematic" and the part that
   grading to the frame mean destroys.
4. **Occlusion.** Words sit *behind* the subject, through a real segmentation
   matte.
5. **Deceleration.** 17 decelerating ramps against 2 accelerating across the
   four. They slow *into* a hold; they do not speed out of one.

---

## 3. Framing

Delivery is **1080×1920 @ 30fps**, `.mov`. A letterboxed reel carves the
**strip's** ratio (16:9) and pads out to the frame.

| Source | Crop for a 16:9 strip | Kept |
|---|---|---|
| 3840×2160 (landscape) | 3836×2158 | **100%** |
| 1728×3072 (portrait) | 1728×972 | 32% — unavoidable |

Carving 9:16 first and drawing bars afterwards leaves **10% of a landscape
frame** — a 3.2× centre punch-in. With mixed-orientation footage one of these
paths is simply wrong.

Portrait sources get their band placed at `SUBJECT_CENTRE_Y = 0.64` — measured
across nine real clips (range 0.43–0.76). Centring lands at waist height.

---

## 4. Grade — three targets, solved per clip

`media.fit_grade()` runs a **secant against each clip's own measured response**.
No assumed gains: three closed forms were tried and all three failed, because
`gamma_b` acts multiplicatively and the curve is not linear anywhere.

| Axis | Control | Target | Reference range |
|---|---|---|---|
| Frame luma | `brightness` | 44.0 — **above the references, deliberately** | 34.7–40.5 |
| **Subject luma** | `lift` (curve mid-point at 0.48) | 74.0, solved as a **ratio** of 1.68 | 53.0–84.1 |
| Colour | `cool` (`gamma_b`/`gamma_r`) | B−R +11.0 | +9 to +13 |

Clamps: `brightness` `-0.85`..`0.60`, `cool` **`-0.34`**..`0.34`, `lift` `0.52`..`0.86`.

**The frame target is the one number here that is a choice, not a measurement.**
The references sit at 34.7–39.9 and were shot in rooms this footage is not shot
in; reproducing their frame mean under non-studio gym light reproduced the
darkness without the light that made it read, and shipped reels that could not be
read on a phone. Everything else in this file still tracks a measurement. This
one tracks a decision, and `verify.py`'s luma band was widened to 41.0–47.0 to
match it rather than the references.

- **`cool` must be allowed negative.** A source that grades out too blue needs
  warming; a zero floor leaves the solver pinned.
- **`lift` targets the ratio, not absolute subject luma.** Both controls move the
  frame mean, so two absolute targets had them fighting.
- **When both saturate, frame darkness wins.** A clip that cannot have a lit
  subject should still look like it belongs. The `lift` floor is what that phase
  spends; at `0.40` it spent too much — 2 of 9 cards in a real build sat pinned
  there, i.e. rendered as dark as the solver could make them. `0.52` keeps the
  give-up over a narrower range.
- **Colour is solved last, alone.** The joint loop's `cool` secant is polluted by
  the `brightness` and `lift` steps taken in the same round: the slope it reads
  is the response to everything that moved, so the axis stops while still wrong.
  Measured — a daylight clip settled at `cool -0.1447` for B−R **+15.2** against
  a target of +11.0, nowhere near its `-0.34` floor, and 14 rounds did no better
  than 9. A third phase re-solves `cool` with the other two frozen: +15.2 → +10.3
  and +9.3 → +11.3, frame luma unmoved. ~5 extra probes per clip.
- Solved on the **proxy**, at the moment's own window, stored on the clip card.
  One shoot spanned luma 29.6–75.0 and B−R −12.5 to +4.0; a fixed grade put the
  reel at 17.6 against the reference's 34.7.

**Two burn-time trims**, because the per-clip grade cannot see the finished reel:

- **Level** (`media.fit_level`) — the grade measures the whole proxy frame; the
  reel shows only the strip. Measured against the old target, clips that each
  solved to 37 concatenated to a strip reading 47; the trim closes that gap
  against whatever the frame target currently is.
- **Dodge** (`media.fit_dodge`) — a local lift through the matte, for subjects a
  global curve cannot reach. Some start *darker* than their background (59.2 vs
  64.4 measured); no tone curve can invert that. Max 0.22. Needs an **all-frames
  matte** or the subject pulses as words come and go.
- **Dodge floor.** Below a subject of 65 the dodge is floored at 0.08 rather than
  allowed to return nothing, because the global curve routinely runs out first:
  on the daylight test clip the new grade takes the frame to 43.5 and the subject
  only to 47.3, with separation moving 1.07 → 1.09. Subject and background share
  a tonal range there and no curve can separate them. Floored only when a subject
  was actually *found* — `fit_dodge` returns 0.0 for "already bright enough",
  "nothing in the matte" and "the probe says it cannot be lifted", and only the
  first means no lift is needed. The other two get reported, not filled.

Both sample **12 frames** — at five the same file measured subject luma 40.9 and
66.3, which is the difference between "needs a big dodge" and "needs none".

**Achieved on real footage, before this change:** frame 41.9 · subject 58.4 ·
separation 1.39 · chroma 16.4, against Gym_1's 35.0 · 53.0 · 1.51 · 26.9. **Not
yet re-measured against the new targets** — do that with
`media.measure_through_matte(reel, matte, band=(656, 1264))` on a fresh build and
replace this line. Until then it is history, not a current reading.

**The subject target went 53.0 → 66.0 → 74.0**, twice raised against the same
complaint. 53.0 is Gym_1's subject and Gym_1 is the darkest of the four, so the
anchor was the bottom of a 53.0–84.1 range; 66.0 put it mid-range. A reel came
back "too dark" measuring frame 42.1 against a subject of 53.1 — the frame was
*brighter* than three of the four references, so the fault was the subject, not
the exposure. A dark room is the look; a dark person is the fault. 74.0 sits in
the upper half of the reference range. Note what this last step is and is not: it
is not a re-measurement of the references, it is an admission that the references
were shot under light this footage does not have.

---

## 4a. The tonal arc — `config.GRADE_ARC`

`fit_grade()` is a **normaliser**: it lands every clip on the same luma so a
shoot spanning 45 points cuts together. Nothing then put deliberate variation
back, and the reel came out flat — per-shot luma spanned 0.86–1.09× of the reel
mean against the references' 0.72–1.42×. A constant mid-dark level with no
bright relief reads as "too dark" even when the mean is higher than the
reference. **The fix is contrast across time, not exposure.**

Measured over the four references, 84 shots pooled into deciles of reel
position (shot luma ÷ reel mean):

| position | 0–10 | 10–20 | 20–30 | 30–40 | 40–50 | 50–60 | 60–70 | 70–80 | 80–90 | 90–100 |
|---|---|---|---|---|---|---|---|---|---|---|
| ratio | 0.87 | 1.13 | 1.34 | 1.39 | 0.98 | 0.89 | 1.01 | 0.84 | 0.88 | 0.85 |

Open a little under, a **bright peak across the first third**, then a long tail
below the mean. Applied in `sequence.dress()` on top of the solved grade, as
`(ratio - 1) / GRADE_LUMA_GAIN_RATIO` — no luma term, because d(luma)/d(brightness)
is proportional to luma, so one brightness step is the same *ratio* on every
clip. That is only true because they were all normalised to one level first.

The burn-time level trim shifts the whole reel equally, so it holds the mean on
target without flattening the arc.

---

## 5. Style packs — `styles/*.json`

Four typographic identities, one per reference. Data, so a fifth is a JSON edit.

| Pack | Reference | Faces (macOS, `.ttc#index`) | Case | Base | Accent | Resident | Ladder (of strip height) |
|---|---|---|---|---|---|---|---|
| `chrome` | Gym_4 | `HelveticaNeue.ttc#7` (Light), `#12` (Thin) | sentence | `#DBDDEA` @0.90 | `#700D0D` @0.62 | 4 | .09 .13 .19 .30 .45 |
| `editorial` | Gym_2 | `Didot.ttc#0` + `HelveticaNeue.ttc#1` as a second voice | sentence / CAPS | `#F6F7FA` @0.92 | `#961A16` @0.78 | 3 | .10 .15 .22 .34 .52 |
| `marker` | gym_3 | `MarkerFelt.ttc#1` | CAPS | `#FFFFFF` @1.0 | `#F12109` @1.0 | 3 | .11 .16 .24 .38 |
| `stencil` | Gym_1 | `Rockwell.ttc#2` + synthesised distress | CAPS | `#F3F3F5` @0.95 | `#A3201A` @0.88 | 3 | .09 .12 .17 .26 |

- `.ttc` collections **need the index**. Naming the file alone silently gives
  Regular, which is the wrong voice and not obviously wrong on inspection.
- `editorial` alternates a second face every 5th word and allows words to **bleed
  past the frame edge** — the reference does it deliberately.
- `stencil` draws an **oversized backdrop word** at half the strip height, 0.26
  alpha, every 4th phrase. Its distress is synthesised (seeded alpha erosion),
  stroke-aware — a fixed pixel drop is convincing wear at 250px and destroys an
  80px word.
- `marker`'s `#F12109` is measured on an **eroded glyph core**; the raw mask
  reads a muddy `#BE281E` because anti-aliased edges drag toward the background.

### The accents above are what was measured. They are not all what renders.

`compose._adjust_ink_luminance()` lifts any ink whose Rec.709 relative luminance
falls under **60** toward **80**, floor ×1.4, clamped at 255. The packs keep the
measured value on disk; the layout lifts it on the way past, so this is one rule
in one place and `state_key()` (which hashes `rgb`) cannot serve a stale PNG.

| Pack | Measured | L | Renders as | L |
|---|---|---|---|---|
| `chrome` | `#700D0D` | 34.0 | **`#FF1F1F`** | 78.6 (red channel clips) |
| `editorial` | `#961A16` | 52.1 | **`#E62822`** | 80.0 |
| `stencil` | `#A3201A` | 59.4 | **`#E42D24`** | 83.3 (the ×1.4 floor, not the target) |
| `marker` | `#F12109` | 75.5 | unchanged | 75.5 |

Why: `#700D0D` at 0.62 alpha against a frame the grade now takes to 44 reads as
depth on a laptop in a dark room and as nothing at all on a phone at arm's
length. Three consequences worth knowing before touching this:

- **`marker`'s red is now the dimmest accent in the system**, at 75.5 against
  78.6–83.3. The "only fully saturated colour" note above still holds on
  saturation; it no longer holds on brightness.
- **`stencil` is in scope by 0.6 of a luminance point.** A re-measure would flip
  it out entirely.
- **Every `base` ink (221–255) and `stencil`'s `backdrop_word` are untouched.**
  The backdrop never enters `placed` — it is a ghost behind the type, not copy.

An earlier pass targeted 110 and took all three reds to 87–100, above `marker`
and near-identical to each other. 80 was chosen because only `chrome` clips
there, so `editorial` and `stencil` keep their measured hue and saturation
exactly — a scalar gain on all three channels is a pure value change.

---

## 6. Layout — `pipeline/compose.py`

Deterministic, seeded on `(reel, phrase)`. Per word: draw a rung from the pack's
ladder (weighted; one hero word per phrase), rasterise for the true ink box, then
score every cell of the anchor grid:

- **hard reject** if the box leaves `safe_box() ∩ strip` (unless `bleed_edges`)
- **hard reject** on overlap past 6% of the word's own area, with boxes inflated
  by 3.5% of the strip first — zero overlap still reads as one run-on word
- **reward** distance from the previous word, so the eye travels
- **reward** landing on the subject when a matte says where it is — the occlusion
  *is* the effect

No cell → drop a rung and retry → close the phrase. Never silently overlap,
never silently push into the bars.

Rendering: all co-resident words onto **one** full-frame RGBA canvas, one PNG per
state change (~80 for 40 words). Five simultaneous `movie` sources would be ~8 MB
each plus filter state on an 8 GB machine.

---

## 7. Lyrics

**Supply the real ones.** `--lyrics-file` takes:

- **`.lrc`** — per-line timestamps, exact. The line *is* the phrase, so the
  accumulate-then-clear grouping comes free. Lines longer than the pack holds are
  **split into consecutive groups**, never truncated.
- **plain text** — lines are phrases, positions inferred.
- **`--lyrics "..."`** — a bare string, everything inferred. `*` prefixes an
  accent word.

Word placement uses **`vocal_onsets`**: the onset detector run on the harmonic
component, band-limited 200–4000 Hz. Measured on a real track — **72 vocal onsets
(2.23/s) against 20 in the full mix (0.62/s)**. The mix is percussion; words
snapped to it land consistently off the voice.

**LRC timestamps are absolute in the song** and the reel starts at `best_start`,
so a lyric written from zero falls entirely outside the window. The tool prints
the window it covers.

Word rate in the references: ~1.1–2.1 per second. Text is **loosely coupled** to
cuts (56–92% coincide) — both tracks quantise to the same grid; neither is slaved
to the other.

---

## 8. Cutting

`MIN_SLOT_SECONDS = 0.30` is the arc's floor and stays. A **burst** is the only
exemption — a declared single gesture, not a pacing decision.

- 3–8 cuts of **2–7 frames** each, slot capped at **1.87s**
- the **cut count bends; the slot length never does**. Capping frames instead of
  cuts turned a 1.2s slot into 1.167s and every downstream length assertion would
  then have been measuring a timeline that no longer matched the music
- one fragment may be **inverted** (`negate_flash`) — gym_3 does it once, 0.07s

Blueprints hold the burst slot short with `max_seconds: 1.80`, or the weight arc
allocates past the ceiling and sequencing correctly refuses to burst it.

---

## 9. Quality

**One lossy generation, never two.** With text to burn, `render --intermediate`
writes ProRes and the burn is the only delivery encode.

| | Before | After |
|---|---|---|
| Picture render | vt_h264 16.7 Mb/s | ProRes 217 Mb/s |
| Delivered | x264 6.9 Mb/s (gen 2) | x264 9.1 Mb/s (gen 1) |

**`maskedmerge` negotiates all three inputs down to the mask's pixel format.**
With a `gray` matte that is a **black-and-white reel** — chroma 0.59 against the
picture's own 16.9 — while every other check still passes. Use `alphamerge` +
`overlay`, which is what the text composite already does.

---

## 10. Blueprints

| Blueprint | Pack | Target | Feel |
|---|---|---|---|
| `gym_floating_text` | `chrome` | 22s | the archetype |
| `gym_editorial_cut` | `editorial` | 24s | serif against grotesque, long holds |
| `gym_marker_cut` | `marker` | 16s | brush caps, hardest cutting, one invert |
| `gym_stencil_cut` | `stencil` | 24s | distressed slab, oversized red backdrop |

All four carry `text_mode: "lyrics"`, `letterbox: 1.778`, `grade: "night"`,
`subject_dodge: true`, and empty `copy` — lyrics and captions fight for the same
centre of frame.

---

## 11. What to shoot

In order of how often it goes wrong:

1. **Leave empty, dark space** on one side of frame. That is where the words go.
   A shot filled edge to edge leaves the layout nowhere and it will drop words.
2. **Get the subject big enough** — at least 2.5% of frame for the matte to find
   it. Measured 87–95% usable on mediums and closes, far lower on wides.
3. **Backlight.** Every reference is a hard rim and almost nothing else. The
   grade takes the picture to luma ~44 and a rim is what survives that. Flat
   overhead gym light gives separation ~1.1 and the grade can only reach ~1.4
   from there; a rim gets 1.5+.
4. **Shoot the burst as one continuous take**, not six short ones. Every fragment
   needs visible motion — a still frame inside a burst reads as a dropped frame.
5. **Payoff at 4K60** so it can be ramped down into a hold.

Deliver 16:9 4K30 where possible: it maps to the strip whole.

---

## 12. Known limits

- **`matte.py` is a saliency model, not a person detector.** Gated on the brief's
  `subject` field and on foreground **area** (2.5–55%), not confidence — a model
  that found nothing returns near-zero everywhere, which scores as maximally
  crisp. On 9 real clips, 3 of 5 sampled gave clean separation; 2 had mask
  failures and fell back to frame-only grading.
- **That area is a fraction of *picture*, and the matte runs on the rendered,
  letterboxed reel.** Measured over the padded 1080×1920 frame instead, both
  bounds break at once: a 16:9 strip is only 31.7% of a 9:16 frame, so a mask
  covering the **whole picture** scores 0.32 and passes a 0.55 ceiling that is
  then unreachable, while a genuine 2.5%-of-picture subject scores 0.79% and is
  thrown away as a failure. Measured on Reel-3: masks covering 68–93% of the
  picture passed the gate, and `median_area` reported a healthy-looking 7.1%
  "of frame" the whole time. `usable()` takes the band and crops to it first;
  the same reel then read 22.3% of picture with 69% of frames kept.
  **A gate whose ceiling cannot be reached reports success, not a problem.**
- **The layout is collision-free, not tasteful.** Expect to tune the anchor
  scoring. It is seeded and asserted so tuning is safe.
- **Synthesised distress is not Gym_1's face.** It reads as rough slab. Accepted.
- **`chrome` is thin near-white type with no scrim.** It is legible only because
  of the grade. The two ship together or neither works.
- **Text-event measurements are a floor.** The detector misses words drawn over
  bright footage.
