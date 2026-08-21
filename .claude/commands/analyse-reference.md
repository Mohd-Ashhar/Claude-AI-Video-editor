---
description: Reverse-engineer a reference reel into measurements, not impressions
---

How to take a screen-recorded reel apart. Every step here replaced a guess that
turned out wrong, so run the measurements rather than describing what you see.

Work in the scratchpad. Assume a 1180×2556 iOS screen recording at ~60fps.

## 1. Find the picture inside the recording

A per-row **temporal variance** profile across 6 frames spread through the file.
Rows that change over time are picture; the rest is chrome and bars. Take the
largest contiguous run.

The four Gym-Inspiration reels and the earlier Sample reel all came back as rows
822–1484 = 1180×663 = **exactly 16:9**. That is a house style, not an accident.

Then check for ink *outside* the strip (count pixels >90 in the bars, excluding
the right rail). All five references: **zero**. If a reel puts type in the bars,
that is a real difference and worth knowing.

## 2. Crop to the strip before anything else

```
ffmpeg -i ref.mp4 -vf "crop=1180:663:0:822" -c:v libx264 -crf 16 strip.mp4
```

Note libx264 needs even dimensions — 663 becomes 662. Account for it when you
reshape raw frames, or you get a confusing `cannot reshape array` error.

**Mask out the right rail (x > 1020) in every subsequent measurement.** The
like/comment icons and the word "Likes" are baked into the recording and they
will show up as text blobs, glyph heights and anchor positions. This silently
polluted a whole run: every reel reported a median text anchor at x = 0.92.

## 3. Cuts

Do **not** use ffmpeg `scdet` — kinetic text triggers scene scores on its own.
Do not use histogram Bhattacharyya either on dark monochrome footage; at
threshold 0.35 it found 1 cut in a 25-cut reel.

Use **mean absolute frame difference** on a 96×54 greyscale downscale, with an
**adaptive** threshold of `max(6 × median, 9.0)` and a 0.2s debounce.

## 4. Tempo

librosa's tempo/beat tracker disagreed with the actual cuts (103 BPM, 2 of 16
within 80ms). **Brute-force fit period and phase directly to the cut times** —
sweep 60–200 BPM at 0.2 and phase at 0.01 — then report how many cuts land
within 80ms. Cross-check against onsets.

## 5. Typography

Two masks: near-white (`min channel > 200` and `max−min < 30`) and saturated red
(`R > 70` and `R − max(G,B) > 40`).

For anchors and sizes, add a **temporal-stability** filter (`|Δluma| < 6` between
frames) — type holds still while footage moves. Then `binary_closing` with a
wide horizontal structure so letters merge into words, `ndimage.label`, and
reject blobs with aspect > 16 (gym light bars) or area < 500.

Report the counts as a **floor**: this misses words drawn over bright footage.
Say so rather than quoting them as truth.

For colour, sample the **eroded core** of the glyph, not the whole mask —
anti-aliased edges drag every measurement towards the background. Gym_3's red is
`#F12109` on the core and a muddy `#BE281E` on the raw mask.

## 6. Matte or blend mode?

The decisive test: find a frame where type crosses a **bright** limb. If the
letters vanish there, it is a segmentation matte. If they survive or tint, it is
a blend mode. A dark subject proves nothing either way — a `darken` blend and a
matte look identical over a dark shirt.

Zoom to at least 3× with `flags=neighbor` before deciding. An early measurement
counted dark pixels inside the text strip and concluded "occluded"; those pixels
were the gaps between letters.

## 7. Grade

Mean per-channel values on a downscale, over the whole file. Report luma out of
255 and **B − R**. The four references: luma 34.7–39.9, B − R +9 to +13. Raw
footage lands 2–3× brighter and neutral, so this is always a real, deliberate
grade and never a camera profile.

## 8. Speed ramps

Per shot, compare mean frame-difference in the second half against the first.
`> 1.6` is accelerating, `< 0.62` decelerating. Across the four references:
**17 decelerating, 2 accelerating.** Do not assume ramps go the exciting way.

## 9. Write it down as a table of numbers

The output of this exercise is a measured table, not adjectives. Put it in the
blueprint's `notes` and in `CLAUDE.md`, so the next disagreement is a
measurement to re-run rather than an argument to have.
