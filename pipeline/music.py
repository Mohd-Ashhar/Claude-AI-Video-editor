"""Read a reference track's pulse and its shape, so a reel can be planned on it.

You add the trending sound inside Instagram, so this track is never the
deliverable -- it is the grid. sequence.py quantises every cut to the beats
found here, and the reel starts on a downbeat and runs a whole number of bars,
which is what makes the in-app alignment forgiving: the app's audio trim is
coarse, and a reel that starts mid-bar has no way to hide a small offset.

v4 measures three more things, because a shot list has to be planned before a
frame is shot and the song is the only thing that exists at that point:

  onsets    where the transients are, so a cut can land on a snare rather than
            on a metronome beat that nobody can hear
  sections  intro / build / drop / sustain / outro, per bar. This is what lets
            a shot list say "shot 6 is your payoff and it lands at 0:11.8"
  profile   four numbers describing what kind of track this is, which is what
            brief.py scores story blueprints against

    uv run python -m pipeline.music --track assets/track.wav
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from pipeline import config, media

BOLD, RED, YELLOW, GREEN, DIM, RESET = "\033[1m", "\033[31m", "\033[33m", "\033[32m", "\033[2m", "\033[0m"

SR = 22050
BEATS_PER_BAR = 4
ARC_FPS = 10.0   # energy arc resolution

# Tempo range reels actually live in. Anything outside it is a metrical-level
# error rather than a real tempo, and is octave-corrected back into range.
TEMPO_RANGE = (85.0, 190.0)

# Resolution of the energy curve, in seconds, and the span compared either side
# of a candidate drop. One second is fine enough to place a payoff on the right
# bar and coarse enough that a single loud transient cannot masquerade as an
# arrival.
ENERGY_RESOLUTION = 1.0
DROP_WINDOW = 3.0

# How close to the track's 90th-percentile level counts as "at the plateau".
# Loose enough to survive real dynamics inside a drop, tight enough that the
# tail of a build does not qualify.
PLATEAU_FRACTION = 0.88

# Relative 5th-to-95th-percentile spread below which a track is treated as
# having no structure to find. See _normalise().
FLAT_SPREAD = 0.20

# Normalised energy above this counts as "the loud part of the track"; below the
# lower figure it is quiet enough to be an intro or an outro. Two thresholds
# rather than one so a section cannot flicker between labels around a single line.
LOUD = 0.60
QUIET = 0.35

# A step in normalised energy that counts as a real drop rather than the track
# simply getting on with itself. Below this the track is treated as having no
# drop at all, which is a legitimate shape -- plenty of trending sounds are a
# flat groove -- and blueprints that need a payoff moment score badly against it.
DROP_MIN_STEP = 0.22

# HPSS is the one expensive measurement here, and its answer (how percussive and
# how vocal the track is) does not change materially across a song. Measured on
# a 60s excerpt from the middle rather than the whole file.
PROFILE_SECONDS = 60.0

VOCAL_BAND_HZ = (200.0, 4000.0)


def _timecode(seconds: float) -> str:
    return f"{int(seconds // 60):d}:{seconds % 60:06.3f}"


# ---------------------------------------------------------------- input


def extract_audio(path: Path, dest_dir: Path) -> Path:
    """Normalise any audio *or video* file to a mono 22050 Hz WAV.

    Not optional. The intended source for this stage is an iOS screen recording
    of the sound playing inside Instagram, which arrives as AAC in a .mov --
    librosa's loader on this build opens neither the container nor the codec, so
    without this step the whole song-first workflow fails at its first input.
    Passing a plain .wav through costs one fast copy and keeps one code path.
    """
    dest = dest_dir / "track.wav"
    media.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(path),
         "-vn", "-ac", "1", "-ar", str(SR), "-c:a", "pcm_s16le", str(dest)],
        desc=f"extract audio from {path.name}",
    )
    if not dest.exists() or dest.stat().st_size < 1024:
        raise media.MediaError(f"{path.name} produced no usable audio")
    return dest


# ---------------------------------------------------------------- analysis


def octave_correct(tempo: float, beats: np.ndarray) -> tuple[float, np.ndarray]:
    """Pull a half- or double-time detection back into the range reels live in.

    Beat trackers pick a metrical level, not a tempo, and the level they pick is
    the single most common thing they get wrong. Measured here on a synthetic
    120 BPM track with kick-on-every-beat and snare on 2 and 4 -- about as plain
    as dance music gets -- librosa returned 60.1 BPM, and every bar-length
    measurement downstream inherited the error at 2x.

    Halving or doubling does not move any beat that was already found; it only
    changes the subdivision, by interpolating midpoints or dropping alternates.
    So this is cheap to be wrong about and expensive to skip.
    """
    while tempo > 0 and tempo < TEMPO_RANGE[0] and tempo * 2 <= TEMPO_RANGE[1]:
        if beats.size >= 2:
            midpoints = (beats[:-1] + beats[1:]) / 2.0
            beats = np.sort(np.concatenate([beats, midpoints]))
        tempo *= 2.0

    while tempo > TEMPO_RANGE[1]:
        beats = beats[::2]
        tempo /= 2.0

    return tempo, beats


def beat_grid(y: np.ndarray, sr: int, forced_bpm: float | None) -> tuple[float, np.ndarray, str]:
    """Beat times in seconds, with an honest fallback.

    Measured on this build: librosa's tracker can return tempo=0 and no beats at
    all on material it cannot find a pulse in. That must not crash the pipeline
    or, worse, silently produce an empty grid that sequence.py then treats as
    "no music" -- hence an explicit source string in the output.
    """
    import librosa

    if forced_bpm:
        period = 60.0 / forced_bpm
        return forced_bpm, np.arange(0.0, len(y) / sr, period), "forced"

    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units="time")
    tempo = float(np.atleast_1d(tempo)[0])

    if tempo <= 0 or len(beats) < 4:
        return 0.0, np.asarray([]), "failed"

    corrected, beats = octave_correct(tempo, np.asarray(beats, dtype=float))
    source = "detected" if abs(corrected - tempo) < 1e-6 else "octave-corrected"
    return corrected, beats, source


def extend_grid(beats: np.ndarray, period: float, duration: float) -> np.ndarray:
    """Continue the detected beat grid across the whole track.

    Beat trackers only report where they were confident, which is not the same
    as where the pulse is. Measured on a 48s test track with a sparse intro,
    librosa returned 45 beats of a possible 96 and the first was at 18.5s -- so
    every stage that picks a position from the grid could only pick one in the
    second half. That silently defeated drop alignment: the reel could not start
    early enough to put the drop two thirds in, and there was no error, just a
    reel that opened in the wrong place.

    The grid is a single fixed tempo by construction, so extending it is
    arithmetic rather than inference.
    """
    if beats.size == 0 or period <= 0:
        return beats

    before = np.arange(beats[0] - period, -period / 2, -period)[::-1]
    after = np.arange(beats[-1] + period, duration + period, period)
    grid = np.concatenate([before[before >= 0], beats, after[after <= duration]])
    return np.round(grid, 6)


def downbeat_phase(beats: np.ndarray, onset_env: np.ndarray, times: np.ndarray) -> int:
    """Which of the four beat positions carries the most attack.

    A reel that starts on beat 3 of a bar feels wrong in a way most people
    cannot name, so the opening cut is placed on a downbeat rather than just
    any beat.
    """
    if beats.size < BEATS_PER_BAR or onset_env.size == 0:
        return 0
    strength = np.interp(beats, times, onset_env)
    return int(np.argmax([strength[phase::BEATS_PER_BAR].mean()
                          for phase in range(BEATS_PER_BAR)]))


def onset_peaks(onset_env: np.ndarray, times: np.ndarray,
                percentile: float = 70.0) -> np.ndarray:
    """Times of the transients strong enough to be worth cutting on.

    Every onset is not a cut point. A hi-hat ticking sixteenths produces dozens
    per bar and cutting on them reads as noise, so only onsets above the given
    percentile of envelope strength survive -- roughly, the hits you would clap
    along to rather than every event the detector can find.
    """
    import librosa

    if onset_env.size == 0:
        return np.asarray([])

    frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=SR,
                                        units="frames", backtrack=False)
    if frames.size == 0:
        return np.asarray([])

    strength = onset_env[frames]
    keep = strength >= np.percentile(strength, percentile)
    return np.asarray(times[frames[keep]], dtype=float)


HOP_LENGTH = 512


def vocal_onsets(y: np.ndarray, sr: int = SR,
                 percentile: float = 62.0) -> np.ndarray:
    """Times where a *sung* syllable starts, not where a drum hits.

    The full-mix onset envelope is dominated by percussion, which is the right
    grid to cut picture on and the wrong one to place words on: a word snapped
    to the kick lands consistently early or late against the vocal, and the
    result reads as karaoke rather than as lyric.

    So this runs the onset detector on the harmonic component only, band-limited
    to roughly where a voice sits (200-4000 Hz). It is not vocal separation and
    does not claim to be -- a sustained synth in that band will contribute. It is
    reliably better than the full mix, which is all it needs to be.
    """
    import librosa

    if y.size < sr // 2:
        return np.asarray([])

    harmonic = librosa.effects.harmonic(y, margin=3.0)
    spectrum = np.abs(librosa.stft(harmonic, n_fft=2048, hop_length=HOP_LENGTH))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    band = (freqs >= 200.0) & (freqs <= 4000.0)
    env = librosa.onset.onset_strength(S=librosa.amplitude_to_db(spectrum[band]),
                                       sr=sr, hop_length=HOP_LENGTH)
    if env.size == 0:
        return np.asarray([])
    times = librosa.frames_to_time(np.arange(env.size), sr=sr, hop_length=HOP_LENGTH)
    frames = librosa.onset.onset_detect(onset_envelope=env, sr=sr,
                                        units="frames", backtrack=False)
    if frames.size == 0:
        return np.asarray([])
    keep = env[frames] >= np.percentile(env[frames], percentile)
    return np.asarray(times[frames[keep]], dtype=float)


def _bucket(values: np.ndarray, fps: float) -> np.ndarray:
    """Average onto the fixed one-second analysis grid."""
    span = max(int(round(fps * ENERGY_RESOLUTION)), 1)
    buckets = values.size // span
    if buckets < 1:
        return np.asarray([])
    return values[: buckets * span].reshape(buckets, span).mean(axis=1)


def _normalise(values: np.ndarray) -> np.ndarray:
    """Scale to 0..1 against this track's own 5th/95th percentiles.

    Comparative rather than absolute because every judgement downstream asks "is
    this loud *for this song*", and trending sounds arrive at wildly different
    mastering levels -- especially when the source is a phone screen recording
    rather than a distribution master.

    The flatness guard is the load-bearing half. Normalising stretches whatever
    variation exists to fill 0..1, so a uniform loop comes out looking like it
    has structure and the drop finder duly finds one: measured, a deliberately
    flat 32s groove reported a drop at 8.5s. Relative spread separates the cases
    cleanly -- the three structured test tracks measure 0.70-0.76, the two flat
    ones 0.02-0.04 -- so anything under FLAT_SPREAD is reported as no structure
    at all rather than amplified into false structure.
    """
    if values.size == 0:
        return values
    low, high = np.percentile(values, 5), np.percentile(values, 95)
    if high <= 0 or (high - low) / high < FLAT_SPREAD:
        return np.zeros_like(values)
    return np.clip((values - low) / (high - low), 0.0, 1.0)


def impact_curve(rms: np.ndarray, rms_fps: float,
                 onset_env: np.ndarray, onset_fps: float) -> np.ndarray:
    """How hard the track is hitting, per second, 0..1.

    Loudness alone is not enough, and the failure is specific. Measured on a 48s
    synthetic with a riser through bars 9-12 and a drop at 24.0s, an RMS-only
    curve put the drop at 20.5s: a white-noise riser reaches the same level as
    the drop it is building to, so by loudness they are indistinguishable. What
    separates them is that a build *thins the beat out* -- so this multiplies
    loudness by transient density, and a loud stretch with no attacks in it
    scores low.

    The grid is seconds, deliberately not bars. Bars were the obvious unit and
    they are the wrong one: bar length comes from tempo detection, so a
    half-time detection halves the resolution of the section map and the drop
    vanishes into a bucket -- measured, per-bar buckets found no drop at all on
    a track that has an obvious one. Seconds are tempo-independent, so the
    measurement the whole shot list hangs on cannot be broken by a beat tracker.
    """
    loudness = _normalise(_bucket(rms, rms_fps))
    density = _normalise(_bucket(onset_env, onset_fps))
    if loudness.size == 0 or density.size == 0:
        return np.asarray([])

    span = min(loudness.size, density.size)
    return np.sqrt(loudness[:span] * density[:span])


def find_drop(energy: np.ndarray) -> tuple[float | None, float]:
    """When the track arrives at its plateau, in seconds, and how hard.

    Two formulations were tried. Maximising the energy *step* is the obvious one
    and it is wrong: measured on a 48s synthetic whose build ramps for 8 bars
    into a drop at 24.0s, it answered 19.5s -- it finds where the riser starts,
    which is the loudest *change*, not where the track arrives.

    So the drop is defined as the first point that holds at its plateau: every
    second of the next DROP_WINDOW is within PLATEAU_FRACTION of the track's
    90th-percentile level. `min`, not `mean`, because a mean over the window
    straddles the boundary and reports the answer a second or two early.

    The step is then measured against *everything before* the candidate rather
    than a local window. Its only job is to reject a track that was already loud
    -- one that opens at full level has no drop, however loud it is -- and a
    local comparison cannot tell that case from a real one when the build is
    long.
    """
    reach = int(round(DROP_WINDOW / ENERGY_RESOLUTION))
    if energy.size < reach + 2:
        return None, 0.0

    plateau = float(np.percentile(energy, 90))
    target = max(plateau * PLATEAU_FRACTION, LOUD)

    best_step = 0.0
    for index in range(1, energy.size - reach + 1):
        window = energy[index:index + reach]
        if float(window.min()) < target:
            continue
        step = float(window.mean() - energy[:index].mean())
        best_step = max(best_step, step)
        if step >= DROP_MIN_STEP:
            return round(index * ENERGY_RESOLUTION, 4), round(step, 4)

    return None, round(best_step, 4)


def snap_to_downbeat(when: float, downbeats: np.ndarray) -> float:
    """Move a measured time onto the nearest downbeat.

    The energy curve is deliberately tempo-independent, so its answer lands
    wherever it lands. A payoff has to sit on a downbeat to feel deliberate, so
    the two measurements are reconciled here rather than by compromising either.
    """
    if downbeats.size == 0:
        return when
    return float(downbeats[int(np.argmin(np.abs(downbeats - when)))])


def label_sections(energy: np.ndarray, drop_at: float | None) -> list[dict]:
    """Merged spans of intro | build | drop | sustain | outro.

    Coarse on purpose. The consumer is a shot list, which can only act on "this
    is where the payoff goes" and "this is where the reel opens" -- a finer
    segmentation would be more impressive and no more useful.
    """
    if energy.size == 0:
        return []

    drop_index = None if drop_at is None else int(round(drop_at / ENERGY_RESOLUTION))

    labels: list[str] = []
    for index, level in enumerate(energy):
        if drop_index is not None and index >= drop_index:
            labels.append("drop" if level >= LOUD else "sustain")
        elif level < QUIET:
            labels.append("intro")
        elif drop_index is not None:
            labels.append("build")
        else:
            labels.append("sustain")

    # "drop" is the arrival, not everything after it: once the level has dipped
    # out of the loud band, later loud stretches are the track sustaining rather
    # than a second drop. Without this the label covers most of the song.
    if drop_index is not None:
        settled = False
        for index in range(drop_index, len(labels)):
            if labels[index] != "drop":
                settled = True
            elif settled:
                labels[index] = "sustain"

    # A quiet tail is an outro regardless of what came before it.
    for index in range(len(labels) - 1, -1, -1):
        if energy[index] >= QUIET:
            break
        labels[index] = "outro"

    spans: list[dict] = []
    levels: list[list[float]] = []
    for index, label in enumerate(labels):
        when = index * ENERGY_RESOLUTION
        if spans and spans[-1]["label"] == label:
            spans[-1]["end"] = round(when + ENERGY_RESOLUTION, 4)
            levels[-1].append(float(energy[index]))
            continue
        spans.append({"label": label, "start": round(when, 4),
                      "end": round(when + ENERGY_RESOLUTION, 4)})
        levels.append([float(energy[index])])

    for span, values in zip(spans, levels):
        span["energy"] = round(sum(values) / len(values), 4)
    return spans


def track_profile(y: np.ndarray, sr: int, energy: np.ndarray,
                  tempo: float, drop_step: float) -> dict:
    """Four numbers describing what kind of track this is.

    brief.py scores story blueprints against these. They are deliberately blunt:
    the question being answered is "would a peak-driven story work on this
    song", not "what genre is it", and a blunt answer that is right most of the
    time beats a precise one that is confidently wrong.
    """
    import librosa

    # HPSS is the expensive call here. A minute from the middle is
    # representative and bounds the cost regardless of track length.
    span = int(PROFILE_SECONDS * sr)
    if y.size > span:
        centre = y.size // 2
        excerpt = y[max(centre - span // 2, 0):][:span]
    else:
        excerpt = y

    harmonic, percussive = librosa.effects.hpss(excerpt)
    harmonic_energy = float(np.mean(np.abs(harmonic))) + 1e-9
    percussive_energy = float(np.mean(np.abs(percussive))) + 1e-9
    percussive_ratio = percussive_energy / (harmonic_energy + percussive_energy)

    centroid = float(np.mean(librosa.feature.spectral_centroid(y=excerpt, sr=sr)))
    brightness = min(centroid / 4000.0, 1.0)

    # Vocals are harmonic content in the band a voice occupies. Not a vocal
    # detector -- a lead synth scores the same -- but the planning question is
    # "is there a melodic line to hang text cues on", and for that they are
    # equivalent.
    spectrum = np.abs(librosa.stft(harmonic, n_fft=2048))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    band = (freqs >= VOCAL_BAND_HZ[0]) & (freqs <= VOCAL_BAND_HZ[1])
    total = float(spectrum.sum()) + 1e-9
    vocal_density = float(spectrum[band].sum()) / total * (1.0 - percussive_ratio) * 2.0

    return {
        "tempo": round(tempo, 2),
        "aggression": round(percussive_ratio * 0.6 + brightness * 0.4, 4),
        "vocal_density": round(min(vocal_density, 1.0), 4),
        "energy_variance": round(float(np.std(energy)) if energy.size else 0.0, 4),
        "drop_sharpness": round(drop_step, 4),
        "percussive_ratio": round(percussive_ratio, 4),
        "brightness": round(brightness, 4),
    }


def pick_start(downbeats: np.ndarray, rms: np.ndarray, rms_fps: float,
               target: float, duration: float,
               drop_at: float | None = None,
               payoff_position: float | None = None) -> float:
    """The downbeat to open the reel on.

    Two jobs, in priority order. Tracks are built to arrive somewhere, and
    starting at 0:00 usually means opening on an intro -- the worst place to
    spend the two seconds that decide whether anyone watches the rest. So the
    default is the highest-energy window.

    When a drop was found *and* the story wants its payoff at a particular point
    (a PR attempt lands on the drop about two thirds in), the window is placed so
    the drop falls there. That is the whole mechanism by which a song decides
    where a shot goes, so it outranks raw energy -- but only when both facts are
    known, otherwise this degrades to exactly the v3 behaviour.
    """
    if downbeats.size == 0:
        return 0.0
    if rms.size == 0:
        return float(downbeats[0])

    aligning = drop_at is not None and payoff_position is not None and target > 0

    best, best_score = float(downbeats[0]), -1e9
    for start in downbeats:
        if start + target > duration + 1e-6:
            break
        a, b = int(start * rms_fps), int((start + target) * rms_fps)
        window = rms[a:b]
        if not window.size:
            continue

        score = float(window.mean()) / (float(rms.max()) + 1e-9)
        if aligning:
            position = (drop_at - start) / target
            if not 0.0 <= position <= 1.0:
                continue
            score -= 2.0 * abs(position - payoff_position)

        if score > best_score:
            best, best_score = float(start), score

    return best


def analyse(path: Path, target: float, forced_bpm: float | None,
            payoff_position: float | None = None) -> dict:
    """Everything a reel can be planned from, measured once.

    `path` may be any audio or video file; it is normalised through ffmpeg
    first. The output is a superset of the v3 music map, so timelines and
    sequencing built against the old shape keep working unchanged.
    """
    import librosa

    with tempfile.TemporaryDirectory() as tmp:
        wav = extract_audio(path, Path(tmp))
        y, sr = librosa.load(str(wav), sr=SR, mono=True)

    duration = len(y) / sr

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    times = librosa.times_like(onset_env, sr=sr)

    tempo, beats, source = beat_grid(y, sr, forced_bpm)
    phase = downbeat_phase(beats, onset_env, times) if source != "forced" else 0
    period = 60.0 / tempo if tempo > 0 else 0.0
    bar_seconds = period * BEATS_PER_BAR

    # Phase is measured on the detected beats, then the grid is extended, so a
    # sparse intro cannot cost the reel its choice of opening bar.
    downbeats = beats[phase::BEATS_PER_BAR] if beats.size else np.asarray([])
    beats = extend_grid(beats, period, duration)
    downbeats = extend_grid(downbeats, bar_seconds, duration)

    hop = max(int(sr / ARC_FPS), 1)
    rms = librosa.feature.rms(y=y, frame_length=hop * 2, hop_length=hop)[0]

    onset_fps = 1.0 / float(times[1] - times[0]) if times.size > 1 else ARC_FPS
    energy = impact_curve(rms, ARC_FPS, onset_env, onset_fps)
    measured_drop, drop_step = find_drop(energy)
    sections = label_sections(energy, measured_drop)
    drop_at = (snap_to_downbeat(measured_drop, downbeats)
               if measured_drop is not None else None)

    best_start = pick_start(downbeats, rms, ARC_FPS, target, duration,
                            drop_at, payoff_position)

    return {
        "version": 2,
        "track": str(path),
        "duration": round(duration, 3),
        "tempo": round(tempo, 2),
        "tempo_source": source,
        "beats_per_bar": BEATS_PER_BAR,
        "beat_period": round(60.0 / tempo, 5) if tempo > 0 else 0.0,
        "bar_seconds": round(bar_seconds, 5),
        "beats": np.round(beats, 4).tolist(),
        "downbeats": np.round(downbeats, 4).tolist(),
        "onsets": np.round(onset_peaks(onset_env, times), 4).tolist(),
        # A second grid, for words rather than cuts. See vocal_onsets().
        "vocal_onsets": np.round(vocal_onsets(y, sr), 4).tolist(),
        "best_start": round(best_start, 4),
        "energy_arc": np.round(rms, 5).tolist(),
        "energy_arc_fps": ARC_FPS,
        "sections": sections,
        "drop_at": round(drop_at, 4) if drop_at is not None else None,
        "profile": track_profile(y, sr, energy, tempo, drop_step),
    }


def realign(music_map: dict, target: float, payoff_position: float | None) -> float:
    """Recompute the opening downbeat for a different payoff position.

    Cheap on purpose: everything pick_start needs is already in the map, since
    `energy_arc` is the RMS series it scores on. The caller only learns where
    the payoff actually falls after quantising the blueprint onto the grid --
    which needs the grid -- so alignment has to be a second pass, and a second
    pass that re-ran the analysis would cost four seconds to answer a question
    that is pure arithmetic.
    """
    downbeats = np.asarray(music_map.get("downbeats") or [], dtype=float)
    rms = np.asarray(music_map.get("energy_arc") or [], dtype=float)
    return pick_start(downbeats, rms, music_map.get("energy_arc_fps", ARC_FPS),
                      target, music_map.get("duration", 0.0),
                      music_map.get("drop_at"), payoff_position)


def section_at(music: dict, when: float) -> str:
    """The section label covering a time, for reporting and blueprint fitting."""
    for section in music.get("sections") or []:
        if section["start"] <= when < section["end"]:
            return section["label"]
    return "sustain"


# ---------------------------------------------------------------- entry


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Extract the beat grid from a reference track.")
    ap.add_argument("--track", type=Path, default=config.ASSETS_DIR / "music.wav")
    ap.add_argument("--out", type=Path, default=config.MUSIC_MAP_JSON)
    ap.add_argument("--target", type=float, default=config.TARGET_REEL_SECONDS,
                    help="reel length the best_start search optimises for")
    ap.add_argument("--bpm", type=float, default=None,
                    help="force a fixed grid when the tracker cannot find a pulse")
    ap.add_argument("--payoff-at", type=float, default=None,
                    help="0..1 position in the reel where the drop should land")
    args = ap.parse_args(argv)

    if not args.track.exists():
        print(f"{RED}no track at {args.track}{RESET}", file=sys.stderr)
        print(f"{DIM}Screen-record the trending sound from Instagram and point --track "
              f"at the file. Audio or video, any container — the cuts are built on "
              f"its grid.{RESET}", file=sys.stderr)
        return 1

    print(f"\n{BOLD}music{RESET}  {DIM}{args.track.name}{RESET}")
    print(f"  {DIM}librosa compiles its analysis kernels on first use — the first "
          f"run takes a few seconds longer than it looks like it should{RESET}\n")

    started = time.perf_counter()
    try:
        doc = analyse(args.track, args.target, args.bpm, args.payoff_at)
    except ImportError:
        print(f"{RED}pip install librosa soundfile{RESET}", file=sys.stderr)
        return 1
    except media.MediaError as exc:
        print(f"{RED}{exc}{RESET}", file=sys.stderr)
        return 1

    if doc["tempo_source"] == "failed":
        print(f"  {RED}no pulse found in this track{RESET}")
        print(f"  {DIM}librosa returned no usable beats. Re-run with an explicit "
              f"tempo, e.g. --bpm 120, or use a track with a clearer beat.{RESET}\n")
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc))

    label = "forced" if doc["tempo_source"] == "forced" else "detected"
    print(f"  {GREEN}tempo {RESET}  {doc['tempo']:.1f} BPM {DIM}({label}) · "
          f"beat {doc['beat_period']:.3f}s · bar {doc['bar_seconds']:.3f}s{RESET}")
    print(f"  {GREEN}grid  {RESET}  {len(doc['beats'])} beats, "
          f"{len(doc['downbeats'])} downbeats, {len(doc['onsets'])} strong onsets "
          f"over {doc['duration']:.1f}s")

    profile = doc["profile"]
    if doc["drop_at"] is not None:
        print(f"  {GREEN}drop  {RESET}  {_timecode(doc['drop_at'])} "
              f"{DIM}— energy step {profile['drop_sharpness']:.2f}{RESET}")
    else:
        print(f"  {YELLOW}drop  {RESET}  none found "
              f"{DIM}— flat groove (best step {profile['drop_sharpness']:.2f}, "
              f"needs {DROP_MIN_STEP}). Peak-driven stories will score badly.{RESET}")

    if doc["sections"]:
        shape = "  ".join(f"{s['label']} {s['start']:.0f}-{s['end']:.0f}s"
                          for s in doc["sections"])
        print(f"  {GREEN}shape {RESET}  {DIM}{shape}{RESET}")

    print(f"  {GREEN}kind  {RESET}  aggression {profile['aggression']:.2f} · "
          f"vocal {profile['vocal_density']:.2f} · "
          f"variance {profile['energy_variance']:.2f}")
    print(f"  {GREEN}start {RESET}  {_timecode(doc['best_start'])} "
          f"{DIM}— best {args.target:.0f}s window, on a downbeat{RESET}")

    print(f"\n{GREEN}music map{RESET}  {DIM}{time.perf_counter() - started:.1f}s "
          f"-> {args.out}{RESET}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
