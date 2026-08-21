"""Place words on the beat, and burn them behind the subject.

The text model the rest of this pipeline uses is one cue per shot: a hook, a
label, a CTA. The reference reel does something structurally different -- about
forty words across nineteen seconds, roughly 2.1 per second, changing on the
*vocal* and not on the cut. Words outnumber shots better than two to one and are
completely decoupled from them.

So a lyric track is its own timeline. Words land on the strong onsets music.py
already measures, which is what makes them feel sung rather than scheduled, and
the accent words -- the stressed ones -- come up in red against a contrasting
face, exactly as the reference does with OVER, SHY, CONTROL, NO and ALRIGHT.

Words are grouped into *phrases*, and a phrase is what clears. The four
Gym-Inspiration reels do not replace one word with the next: they accumulate two
to five words at their own anchors and sizes, hold them together, and then wipe
the whole group. Placement of that composition is compose.py's job; this module
decides only when each word arrives and which group it belongs to.

    uv run python -m pipeline.lyrics --words "feel your all over me" --accent 3
    uv run python -m pipeline.lyrics --burn out/reel_v1.mov
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

from pipeline import compose, config, media

BOLD, RED, YELLOW, GREEN, DIM, RESET = "\033[1m", "\033[31m", "\033[33m", "\033[32m", "\033[2m", "\033[0m"

# A word holds until just before the next one. The gap is what makes the change
# read as a cut rather than a crossfade; the reference has no visible dissolve
# between words at all.
WORD_GAP = 0.04
MIN_WORD_SECONDS = 0.18
# No upper bound any more. A word holds until its phrase clears, which the
# references let run two to three seconds; capping it would empty the frame
# mid-group and break the accumulation the whole style is built on.

# The first word should arrive almost immediately -- the reference reel opens on
# one at 0.5s -- and the last should land before the reel runs out.
LEAD_IN = 0.35
TAIL_OUT = 0.60

# Marker for an accent word when the lyric is given as one string.
ACCENT_PREFIX = "*"

# A phrase also closes when the music turns over, not only when it is full.
# Measured on the references, groups run roughly 2-3s, which at 90-100 BPM is
# about two bars.
PHRASE_BARS = 2.0
# Blank between one phrase clearing and the next word arriving. Small, but not
# zero -- without it the change reads as a re-layout of the same group rather
# than as a new thought.
PHRASE_GAP = 0.06


def parse_words(text: str, accents: list[int] | None = None) -> list[dict]:
    """Split a lyric into words, honouring `*` or explicit indices for accents."""
    raw = [w for w in text.split() if w.strip()]
    marked = set(accents or [])
    words = []
    for index, word in enumerate(raw):
        accent = word.startswith(ACCENT_PREFIX) or index in marked
        words.append({"text": word.lstrip(ACCENT_PREFIX), "accent": accent})
    return words


LRC_LINE = re.compile(r"^((?:\[\d+:\d+(?:[.:]\d+)?\])+)\s*(.*)$")
LRC_STAMP = re.compile(r"\[(\d+):(\d+)(?:[.:](\d+))?\]")


def parse_lyric_file(path: Path) -> list[dict]:
    """Read a lyric file: LRC with timestamps, or plain text, one line per phrase.

    Timed lyrics are the difference between words that look sung and words that
    are sung. Everything else here places them by inference -- evenly spaced,
    snapped to the nearest transient -- which is a good guess and still only a
    guess. An `.lrc` says exactly when each line lands, and the phrase structure
    comes free with it: a line *is* a phrase, which is what the accumulate-then-
    clear composition wants anyway.

    Plain text is accepted too and treated as untimed phrases, because a lyric
    copied out of a browser is what will usually be to hand.
    """
    lines: list[dict] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        text = raw.strip()
        if not text or text.startswith("#"):
            continue
        match = LRC_LINE.match(text)
        if not match:
            lines.append({"text": text, "at": None})
            continue
        body = match.group(2).strip()
        if not body:
            continue
        # One line may carry several stamps -- LRC's way of repeating a chorus.
        for stamp in LRC_STAMP.finditer(match.group(1)):
            minutes, seconds, fraction = stamp.groups()
            at = int(minutes) * 60 + int(seconds)
            if fraction:
                at += int(fraction) / (10 ** len(fraction))
            lines.append({"text": body, "at": float(at)})
    lines.sort(key=lambda item: (item["at"] is None, item["at"] or 0.0))
    return lines


def place_timed(lines: list[dict], music: dict, start: float, duration: float,
                max_resident: int = 4) -> tuple[list[dict], str]:
    """Place words from lyrics that already know when they happen.

    A line's own timestamp is the truth and is never moved. Only the words
    *inside* a line are inferred, and they are snapped to vocal onsets rather
    than to the full-mix grid -- the mix is dominated by percussion, and a word
    snapped to the kick lands consistently off the voice.

    Timestamps are absolute in the track; the reel starts at `best_start`, so
    lines before it or past the end are dropped rather than squeezed in.
    """
    grid = np.asarray(music.get("vocal_onsets") or music.get("onsets") or [],
                      dtype=float)
    timed = [line for line in lines if line["at"] is not None]
    if not timed:
        return [], "none"

    placed: list[dict] = []
    for index, line in enumerate(timed):
        line_at = line["at"] - start
        following = timed[index + 1]["at"] - start if index + 1 < len(timed) else duration
        line_end = min(following - PHRASE_GAP, duration)
        if line_end <= 0 or line_at >= duration:
            continue
        line_at = max(line_at, 0.0)
        words = parse_words(line["text"])
        if not words:
            continue

        span = max(line_end - line_at, MIN_WORD_SECONDS)
        ideal = [line_at + span * i / max(len(words), 1) for i in range(len(words))]
        inside = np.sort(grid[(grid >= line_at + start) & (grid < line_end + start)] - start)

        times: list[float] = []
        for want in ideal:
            at = float(want)
            if inside.size:
                free = inside[inside > (times[-1] + MIN_WORD_SECONDS if times else -1e9)]
                if free.size:
                    nearest = float(free[np.argmin(np.abs(free - want))])
                    # A snap may only nudge a word, never re-order the line.
                    if abs(nearest - want) <= span / max(len(words), 1):
                        at = nearest
            if times:
                at = max(at, times[-1] + MIN_WORD_SECONDS)
            times.append(min(at, line_end - 0.01))

        # A written line can be longer than the frame can hold. The style pack's
        # `max_resident` is a composition limit -- four words at these sizes is
        # already a full frame -- so a long line is split into consecutive
        # groups that each clear in turn, rather than having its tail dropped.
        # Dropping words out of a real lyric would be the wrong trade every time.
        size = max(int(max_resident), 1)
        chunks = [list(range(i, min(i + size, len(words))))
                  for i in range(0, len(words), size)]
        for chunk_no, members in enumerate(chunks):
            # Each group holds until the next group starts, and the last until
            # the line itself clears.
            if chunk_no + 1 < len(chunks):
                chunk_end = times[chunks[chunk_no + 1][0]] - PHRASE_GAP
            else:
                chunk_end = line_end
            for slot, position in enumerate(members):
                at = times[position]
                placed.append({
                    "index": len(placed),
                    "phrase": f"{index}.{chunk_no}",
                    "slot": slot,
                    "text": words[position]["text"],
                    "accent": bool(words[position]["accent"]),
                    "at": round(at, 3),
                    "duration": round(max(chunk_end - at, MIN_WORD_SECONDS), 3),
                    "treatment": ("lyric_accent" if words[position]["accent"]
                                  else "lyric"),
                })

    return [w for w in placed if w["at"] < duration - 0.05], "timed"


def group(times: list[float], max_resident: int, window: float) -> list[list[int]]:
    """Split word indices into phrases: full, or the music turned over.

    Whichever comes first. `max_resident` is the pack's own limit on how many
    words may share the frame; `window` is two bars. A phrase that only ever
    closed when full would run past the section change on a sparse line, and one
    that only closed on the bar would pile eight words into a busy one.
    """
    if not times:
        return []
    phrases: list[list[int]] = [[0]]
    anchor = times[0]
    for index in range(1, len(times)):
        full = len(phrases[-1]) >= max(max_resident, 1)
        turned = times[index] - anchor >= window
        if full or turned:
            phrases.append([index])
            anchor = times[index]
        else:
            phrases[-1].append(index)
    return phrases


def place(words: list[dict], music: dict, start: float, duration: float,
          max_resident: int = 4) -> tuple[list[dict], str]:
    """Give every word a time, using the track's own transients as the grid.

    Onsets rather than beats. A beat grid is regular by construction, so words
    placed on it arrive like a metronome; onsets are where the track actually
    hits, which is where a singer lands. Falls back to the beat grid, then to
    even spacing, so this still produces something on a track with no clear pulse.
    """
    if not words:
        return [], "none"

    # Vocal onsets first. Words belong on syllables; the full-mix grid is mostly
    # percussion, and on a real track it ran 0.62 events a second against the
    # vocal grid's 2.23 -- too sparse to place a lyric on at all.
    grid = np.asarray(music.get("vocal_onsets") or [], dtype=float)
    source = "vocal onsets"
    if grid.size < 2:
        grid = np.asarray(music.get("onsets") or [], dtype=float)
        source = "onsets"
    if grid.size < 2:
        grid = np.asarray(music.get("beats") or [], dtype=float)
        source = "beats"

    inside = np.sort(grid[(grid >= start) & (grid <= start + duration)] - start)
    if inside.size < 2:
        inside = np.asarray([])
        source = "even"

    # Snap evenly-spaced ideal times onto the nearest real transient, rather than
    # spreading the words across the onsets that happen to exist.
    #
    # Spreading over onset *indices* was the obvious approach and it fails on any
    # track with a sparse intro: measured on a test track whose first four
    # seconds are quiet, the first word landed at 3.99s and the reel opened on
    # silence. Ideal times cover the whole duration by construction; snapping only
    # moves each one to the nearest hit, so the lyric stays sung *and* stays put.
    lead = min(LEAD_IN, duration * 0.05)
    span = max(duration - lead - TAIL_OUT, 0.1)
    ideal = lead + np.arange(len(words)) * (span / max(len(words) - 1, 1))

    times: list[float] = []
    for want in ideal:
        if inside.size:
            free = inside[inside > (times[-1] + MIN_WORD_SECONDS if times else -1)]
            at = float(free[np.argmin(np.abs(free - want))]) if free.size else float(want)
        else:
            at = float(want)
        # Never let snapping drag a word more than half a slot from where the
        # even spread wanted it, or a dense cluster of onsets swallows the line.
        if abs(at - want) > span / max(len(words), 1):
            at = float(want)
        at = max(at, times[-1] + MIN_WORD_SECONDS) if times else at
        # A snap may land past the end of the reel; the word still has to appear.
        times.append(min(at, duration - MIN_WORD_SECONDS))

    # Group into phrases before assigning holds. A word's hold is not "until the
    # next word" any more -- it is "until this phrase clears", because the whole
    # group goes at once.
    bar = float(music.get("bar_seconds") or (music.get("beat_period", 0.5) * 4))
    phrases = group(times, max_resident, bar * PHRASE_BARS)

    placed: list[dict] = []
    for phrase_index, member_indices in enumerate(phrases):
        nxt = phrases[phrase_index + 1][0] if phrase_index + 1 < len(phrases) else None
        ends = (times[nxt] - PHRASE_GAP) if nxt is not None else duration
        ends = min(ends, duration)
        for slot, index in enumerate(member_indices):
            at = times[index]
            hold = max(ends - at, MIN_WORD_SECONDS)
            hold = min(hold, max(duration - at, MIN_WORD_SECONDS))
            placed.append({
                "index": index,
                "phrase": phrase_index,
                "slot": slot,
                "text": words[index]["text"],
                "accent": bool(words[index]["accent"]),
                "at": round(at, 3),
                "duration": round(hold, 3),
                "treatment": "lyric_accent" if words[index]["accent"] else "lyric",
            })

    return [w for w in placed if w["at"] < duration - 0.05], source


def letterbox_band(ratio: float | None) -> tuple[int, int] | None:
    """Top and bottom of the picture strip, for a letterboxed reel."""
    if not ratio:
        return None
    strip = int(round(config.OUT_W / float(ratio)))
    if strip >= config.OUT_H:
        return None
    bar = (config.OUT_H - strip) // 2
    return bar, config.OUT_H - bar


def states(words: list[dict], duration: float) -> list[tuple[float, list[dict]]]:
    """Every interval over which the visible set of words is constant.

    Words overlap now, so a per-word timeline no longer describes the screen. The
    screen changes at the union of every arrival and every expiry, and between
    two of those instants the composition is one fixed image. Forty words make
    roughly eighty states, which is the same order of magnitude as forty cues --
    the concat pass and its memory cost are unchanged.
    """
    marks = sorted({0.0, duration}
                   | {w["at"] for w in words}
                   | {min(w["at"] + w["duration"], duration) for w in words})
    out: list[tuple[float, list[dict]]] = []
    for index in range(len(marks) - 1):
        lo, hi = marks[index], marks[index + 1]
        if hi - lo < 1.0 / config.OUT_FPS:
            continue
        mid = (lo + hi) / 2
        visible = [w for w in words if w["at"] <= mid < w["at"] + w["duration"]]
        out.append((hi - lo, visible))
    return out


def track_video(words: list[dict], duration: float, cues: Path, dest: Path,
                style: dict, band: tuple[int, int] | None = None) -> Path:
    """Render the whole word track to one RGBA video.

    One video rather than one overlay per word. Five simultaneous `movie` sources
    would each hold a 1080x1920 RGBA frame -- about 8 MB apiece plus filter state
    -- on a machine with 8 GB, and forty would be 330 MB before a single pixel is
    composited. Concatenating stills instead costs one pass and almost no memory,
    and it is why co-resident words were affordable at all.
    """
    from PIL import Image

    cues.mkdir(parents=True, exist_ok=True)
    blank = cues / "_blank.png"
    if not blank.exists():
        Image.new("RGBA", (config.OUT_W, config.OUT_H), (0, 0, 0, 0)).save(blank)

    entries: list[tuple[Path, float]] = []
    for hold, visible in states(words, duration):
        if not visible:
            entries.append((blank, hold))
            continue
        backdrop = visible[0].get("backdrop")
        entries.append((compose.render_state(visible, style, cues, band, backdrop),
                        hold))
    if not entries:
        entries.append((blank, duration))

    listing = dest.with_suffix(".txt")
    lines = []
    for path, hold in entries:
        lines.append(f"file '{path}'")
        lines.append(f"duration {max(hold, 1.0 / config.OUT_FPS):.4f}")
    # The concat demuxer ignores the final entry's duration unless the file is
    # repeated, which otherwise drops the last state a frame short.
    lines.append(f"file '{entries[-1][0]}'")
    listing.write_text("\n".join(lines) + "\n")

    media.run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0",
               "-i", str(listing),
               "-vf", f"fps={config.OUT_FPS},format=rgba",
               "-t", f"{duration:.3f}",
               "-c:v", "qtrle", str(dest)],
              desc="lyric track")
    return dest


def burn_command(video: Path, track: Path, matte: Path | None, dest: Path,
                 dodge: float = 0.0, level: float = 0.0) -> list[str]:
    """Composite the word track over the reel, behind the subject if there is one.

    The occlusion is done by multiplying the text's own alpha by the inverse of
    the matte, rather than by `maskedmerge` over three streams. Same result, and
    it keeps the base video as a single stream for the text composite.

    `dodge` lifts the subject before the text lands on it, through the same
    matte. That one *does* want `maskedmerge`: the whole point is a local
    adjustment, which is not something a tone curve can express. The reference
    look is separation -- Gym_1 carries its subject at 53 against a frame of 35 --
    and a clip whose subject starts darker than its room cannot be graded there
    globally at any setting.
    """
    if matte is None:
        graph = ("[1:v]format=rgba[t];"
                 "[0:v][t]overlay=x=0:y=0:format=auto,format=yuv420p[v]")
        inputs = ["-i", str(video), "-i", str(track)]
    elif dodge > 0.0:
        graph = (
            # Two uses of the matte, so it is split rather than decoded twice.
            "[2:v]format=gray,split=2[m0][m1];"
            f"[0:v]{('eq=brightness=%.4f,' % level) if level else ''}split=2[p0][p1];"
            # alphamerge + overlay, not maskedmerge. maskedmerge requires all
            # three inputs to share a pixel format and negotiates down to the
            # mask's -- which is gray. The reel came out of the burn in black
            # and white: chroma spread 0.59 against the picture's own 16.9 and
            # the reference's 26.9, with every other measurement still passing.
            # This is the same idiom the text composite below already uses.
            f"[p1]eq=brightness={dodge:.4f},format=rgba[plit];"
            "[plit][m0]alphamerge[plita];"
            "[p0][plita]overlay=x=0:y=0:format=auto[base];"
            "[1:v]format=rgba,split=2[t0][t1];"
            "[t1]alphaextract[ta];"
            "[m1]negate[mi];"
            "[ta][mi]blend=all_mode=multiply[a2];"
            "[t0][a2]alphamerge[t2];"
            "[base][t2]overlay=x=0:y=0:format=auto,format=yuv420p[v]"
        )
        inputs = ["-i", str(video), "-i", str(track), "-i", str(matte)]
        return ["ffmpeg", "-y", "-v", "error", *inputs,
                "-filter_complex", graph, "-map", "[v]", "-map", "0:a?",
                *config.FINAL_ENCODERS[config.FINAL_ENCODER], *config.COLOR_TAGS,
                "-c:a", "copy", "-movflags", "+faststart", str(dest)]
    else:
        graph = (
            "[1:v]format=rgba,split=2[t0][t1];"
            # The letterforms' own alpha, as a greyscale image.
            "[t1]alphaextract[ta];"
            # White in the matte is subject; inverted it becomes "may draw here".
            "[2:v]format=gray,negate[mi];"
            "[ta][mi]blend=all_mode=multiply[a2];"
            "[t0][a2]alphamerge[t2];"
            # The trailing conversion is load-bearing. `overlay` in RGBA mode
            # leaves the chain 4:4:4, and x264's high profile refuses it outright
            # -- the encoder never opens and nothing is written. Same family as
            # the `exposure` float-format trap: a filter hands on a format the
            # next stage cannot take, and the failure surfaces nowhere near it.
            "[0:v][t2]overlay=x=0:y=0:format=auto,format=yuv420p[v]"
        )
        inputs = ["-i", str(video), "-i", str(track), "-i", str(matte)]

    return ["ffmpeg", "-y", "-v", "error", *inputs,
            "-filter_complex", graph, "-map", "[v]",
            *(["-map", "0:a?"]),
            *config.FINAL_ENCODERS[config.FINAL_ENCODER], *config.COLOR_TAGS,
            "-c:a", "copy", "-movflags", "+faststart", str(dest)]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Place lyric words and burn them in.")
    ap.add_argument("--words", default=None,
                    help="the lyric line; prefix a word with * to accent it")
    ap.add_argument("--lyrics-file", dest="lyrics_file", type=Path, default=None,
                    help="a lyric file: .lrc with timestamps (exact), or plain "
                         "text with one line per phrase")
    ap.add_argument("--accent", type=int, nargs="*", default=None,
                    help="extra word indices to accent (0-based)")
    ap.add_argument("--music", type=Path, default=config.MUSIC_MAP_JSON)
    ap.add_argument("--brief", type=Path, default=config.BRIEF_JSON)
    ap.add_argument("--duration", type=float, default=None)
    ap.add_argument("--out", type=Path, default=config.WORK_DIR / "lyrics.json")
    ap.add_argument("--burn", type=Path, default=None,
                    help="a rendered reel to composite the words onto")
    ap.add_argument("--matte", type=Path, default=None)
    ap.add_argument("--style", default=None, choices=compose.available(),
                    help="typographic identity; defaults to the brief's, then chrome")
    ap.add_argument("--dest", type=Path, default=None)
    args = ap.parse_args(argv)

    if args.burn:
        return _burn(args)

    if not args.words and not args.lyrics_file:
        print(f"{RED}--words or --lyrics-file is required{RESET}", file=sys.stderr)
        return 1
    if not args.music.exists():
        print(f"{RED}no music map — run pipeline.brief or pipeline.music first{RESET}",
              file=sys.stderr)
        return 1

    music = json.loads(args.music.read_text())
    start = music.get("best_start", 0.0)
    duration = args.duration
    if duration is None and args.brief.exists():
        duration = json.loads(args.brief.read_text()).get("total_seconds")
    duration = duration or config.TARGET_REEL_SECONDS

    style = compose.load(args.style or _brief_field(args.brief, "style"))
    if args.lyrics_file:
        lines = parse_lyric_file(args.lyrics_file)
        if any(line["at"] is not None for line in lines):
            placed, source = place_timed(lines, music, start, duration,
                                         style["max_resident"])
        else:
            # Untimed lines are still phrases -- the poet's grouping beats a
            # count of four -- but their positions have to be inferred.
            words = parse_words(" ".join(line["text"] for line in lines),
                                args.accent)
            placed, source = place(words, music, start, duration,
                                   style["max_resident"])
    else:
        words = parse_words(args.words, args.accent)
        placed, source = place(words, music, start, duration, style["max_resident"])
    if args.lyrics_file and source == "timed":
        # LRC timestamps are absolute in the song and the reel starts partway in,
        # so the window that matters is [best_start, best_start + duration].
        # Saying so is the difference between "the lyric is wrong" and "you are
        # looking at the wrong part of the song".
        print(f"  {DIM}reel covers {start:.2f}s-{start + duration:.2f}s of the "
              f"track; lyric lines outside that are not used{RESET}")
    if not placed:
        print(f"{RED}no words landed inside the reel{RESET}", file=sys.stderr)
        print(f"{DIM}The reel plays {start:.2f}s-{start + duration:.2f}s of the "
              f"track. LRC timestamps are absolute in the song, so a lyric "
              f"written from zero will fall entirely before this window.{RESET}",
              file=sys.stderr)
        return 1

    doc = {"version": 1, "source": source, "start": start, "style": style["id"],
           "max_resident": style["max_resident"],
           "duration": round(duration, 3), "words": placed}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2))

    groups = len({w["phrase"] for w in placed})
    print(f"\n{BOLD}lyrics{RESET}  {DIM}{len(placed)} words in {groups} phrases over "
          f"{duration:.2f}s · placed on {source} · style {style['id']}{RESET}\n")
    current = -1
    for word in placed:
        if word["phrase"] != current:
            current = word["phrase"]
            print(f"  {DIM}--- phrase {current}{RESET}")
        mark = f"{RED}accent{RESET}" if word["accent"] else f"{DIM}      {RESET}"
        print(f"  {word['at']:>6.2f}s  holds {word['duration']:>5.2f}s  {mark}  "
              f"{BOLD}{word['text']}{RESET}")

    rate = len(placed) / duration if duration else 0
    print(f"\n{GREEN}{rate:.1f} words/second{RESET}  "
          f"{DIM}(the references run 2.1) -> {args.out}{RESET}\n")
    return 0


def _brief_field(brief_path: Path, key: str):
    if not brief_path.exists():
        return None
    return json.loads(brief_path.read_text()).get(key)


def subject_for(boxes: list[dict], lo: float, hi: float) -> dict | None:
    """One representative subject box for a phrase's window.

    The median edge of every measured box in the span, not the union. A union
    grows to the whole frame the moment the subject walks across it, at which
    point "prefer anchors on the subject" degrades to "prefer anywhere" and the
    bonus stops meaning anything.
    """
    inside = [b["box"] for b in boxes if lo <= b["at"] <= hi]
    if not inside:
        return None
    columns = list(zip(*inside))
    mid = [sorted(c)[len(c) // 2] for c in columns]
    return {"box": mid}


def compose_words(words: list[dict], style: dict, band, boxes: list[dict],
                  seed_base: str) -> list[dict]:
    """Lay every phrase out, and return the words that found a place.

    A phrase whose last words could not be packed loses them rather than
    overlapping them. That is deliberate and it is reported: a lyric line too
    long for the frame is a line to shorten, not a thing to silently cram.
    """
    placed: list[dict] = []
    # Phrase ids are opaque labels, not indices -- timed lyrics number them
    # "line.chunk" so a long line's groups stay distinguishable. Ordered by when
    # they actually start, which is the only ordering that means anything.
    order = sorted({w["phrase"] for w in words},
                   key=lambda pid: min(w["at"] for w in words if w["phrase"] == pid))
    for position, index in enumerate(order):
        members = [w for w in words if w["phrase"] == index]
        lo = min(w["at"] for w in members)
        hi = max(w["at"] + w["duration"] for w in members)
        seed = int(hashlib.sha1(f"{seed_base}|{index}".encode()).hexdigest()[:8], 16)
        laid = compose.layout(members, style, band, subject_for(boxes, lo, hi), seed)
        # The oversized backdrop word is an accent, not a bed. Gym_1 shows it a
        # handful of times across 32s; drawn behind every phrase it stops being a
        # gesture and becomes a red wash the type has to fight. `every` was in
        # the pack from the start and simply was not being read.
        spec = style.get("backdrop_word") or {}
        show = spec and position % max(int(spec.get("every", 1)), 1) == 0
        for word in laid:
            word["backdrop"] = members[0]["text"] if show else None
        placed.extend(laid)
    return placed


def _burn(args) -> int:
    if not args.out.exists():
        print(f"{RED}no lyrics.json — run without --burn first{RESET}", file=sys.stderr)
        return 1
    doc = json.loads(args.out.read_text())
    if not doc["words"]:
        print(f"{YELLOW}no words to burn{RESET}")
        return 0

    info = media.probe(args.burn)
    duration = media.duration_seconds(info)

    # Re-place against the reel that actually exists. Words are timed from the
    # brief's target length, and the built reel is routinely shorter -- a slot
    # the footage could not fill is dropped, which is the whole point of the
    # coverage report. Measured: a 20.0s brief rendered 16.02s and the last four
    # words were timed past the end of the file, so they simply never appeared.
    style = compose.load(args.style or _brief_field(args.brief, "style")
                         or doc.get("style"))

    # Two independent reasons to re-place. The reel being shorter than the brief
    # is the old one. The new one is a *style* change: max_resident decides how
    # many words share a phrase, so burning a 3-word pack over a doc grouped at
    # 4 would hand compose.layout() a phrase it must then truncate -- losing a
    # word to a setting the user only changed the look with.
    stale_length = abs(duration - doc.get("duration", duration)) > 0.15
    stale_grouping = doc.get("max_resident") != style["max_resident"]
    if (stale_length or stale_grouping) and args.music.exists():
        music = json.loads(args.music.read_text())
        words = [{"text": w["text"], "accent": w["accent"]} for w in doc["words"]]
        replaced, source = place(words, music, doc.get("start", 0.0), duration,
                                 style["max_resident"])
        why = "the reel is shorter than the brief" if stale_length else \
              f"style {style['id']} groups {style['max_resident']} to a phrase"
        print(f"  {YELLOW}retimed{RESET}  {why} — words re-placed on {source}")
        doc["words"] = replaced
        doc["duration"] = round(duration, 3)
        doc["max_resident"] = style["max_resident"]
        doc["style"] = style["id"]
        args.out.write_text(json.dumps(doc, indent=2))
    cues = config.WORK_DIR / "states"
    track = config.WORK_DIR / "lyric_track.mov"

    print(f"\n{BOLD}lyrics{RESET}  {DIM}burning {len(doc['words'])} words into "
          f"{args.burn.name} · style {style['id']}{RESET}\n")
    band = letterbox_band(_brief_field(args.brief, "letterbox"))
    if band:
        print(f"  {GREEN}band  {RESET}  letterboxed — type confined to rows "
              f"{band[0]}-{band[1]}")

    boxes = []
    if args.matte:
        side = args.matte.with_suffix(".json")
        if side.exists():
            boxes = json.loads(side.read_text()).get("boxes", [])
    if boxes:
        print(f"  {GREEN}subject{RESET} {len(boxes)} boxes — anchors will prefer "
              f"landing on the subject")

    laid = compose_words(doc["words"], style, band, boxes, args.burn.stem)
    dropped = len(doc["words"]) - len(laid)
    if dropped:
        print(f"  {YELLOW}packed{RESET}  {len(laid)}/{len(doc['words'])} words fit; "
              f"{dropped} had nowhere to go {DIM}(shorten the line, or raise "
              f"max_resident in styles/{style['id']}.json){RESET}")
    doc["words"] = laid
    args.out.write_text(json.dumps(doc, indent=2))

    peak = max((sum(1 for w in laid if w["at"] <= t < w["at"] + w["duration"])
                for t in (w["at"] + 0.01 for w in laid)), default=0)
    print(f"  {GREEN}layout{RESET}  {len({w['phrase'] for w in laid})} phrases · "
          f"peak {peak} words on screen at once "
          f"{DIM}(the references run 2-5){RESET}")

    track_video(doc["words"], duration, cues, track, style, band)
    print(f"  {GREEN}track {RESET}  {track.name} {DIM}({track.stat().st_size / 1e6:.0f} MB, "
          f"RGBA){RESET}")

    matte = args.matte if (args.matte and args.matte.exists()) else None
    dodge = 0.0
    level = 0.0
    if matte:
        print(f"  {GREEN}matte {RESET}  {matte.name} {DIM}— words will sit behind "
              f"the subject{RESET}")
        if _brief_field(args.brief, "subject_dodge") is not False:
            frame_l, subject_l = media.measure_through_matte(args.burn, matte,
                                                             band=band)
            level = media.fit_level(args.burn, matte, band=band)
            if abs(level) > 1e-4:
                print(f"  {GREEN}level {RESET}  picture reads {frame_l:.0f}; "
                      f"trimming {level:+.3f} to reach "
                      f"{media.GRADE_TARGET_LUMA:.0f} "
                      f"{DIM}(the strip is not what the per-clip grade "
                      f"measured){RESET}")
            dodge = media.fit_dodge(args.burn, matte, band=band, level=level)
            if dodge > 0:
                print(f"  {GREEN}dodge {RESET}  subject reads {subject_l:.0f} against "
                      f"a frame of {frame_l:.0f} — lifting it +{dodge:.3f} "
                      f"{DIM}(the references carry 53 against 35){RESET}")
            else:
                print(f"  {DIM}dodge   subject already reads {subject_l:.0f}; "
                      f"no lift needed{RESET}")
    else:
        print(f"  {YELLOW}matte {RESET}  none — words draw over the top "
              f"{DIM}(run pipeline.matte for occlusion){RESET}")

    dest = args.dest or config.out_path(f"{args.burn.stem}_lyric", args.burn.parent)
    try:
        media.run(burn_command(args.burn, track, matte, dest, dodge, level),
                  desc="burn lyrics")
    except media.MediaError as exc:
        print(f"{RED}{exc}{RESET}", file=sys.stderr)
        return 1

    print(f"\n{GREEN}burned{RESET}  {DIM}{dest}{RESET}\n")
    print(str(dest))
    return 0


if __name__ == "__main__":
    sys.exit(main())
