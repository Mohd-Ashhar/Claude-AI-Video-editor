# AI Reel Editor — Build Plan for MacBook Air M2 (8-core CPU / 10-core GPU / 8 GB)

A hardware-specific rewrite of the original blueprint. The architecture, JSON contracts, and agent design are unchanged — only the model layer, the media handling, and the execution discipline are adapted. Everything below assumes the base M2 Air: fanless, 8 GB unified memory, likely 256 GB SSD.

---

## 1. What you actually have to spend

| Resource | Total | Realistically available |
|---|---|---|
| Unified memory | 8 GB | **~4.5 GB** (macOS holds 3–3.5 GB; Claude Code's Node process is another ~0.5–1 GB) |
| GPU | 10-core, Metal / MPS | Shares the same 8 GB — GPU allocation *subtracts* from CPU headroom |
| Neural Engine | 16-core | Free compute for CoreML models — use it |
| Media engine | H.264 + HEVC + ProRes, encode & decode | The single biggest advantage this machine has |
| CPU | 4 performance + 4 efficiency | Fanless: sustained x264 encoding throttles after ~3–5 min |
| Disk | 256 GB typical | 4K source + intermediates will fill it fast |

**Two rules follow from this and they drive the whole plan:**

1. **One model in memory at a time, each in its own process.** Python does not reliably return freed torch/MPS memory to the OS. Every analysis stage runs as a separate `uv run python -m pipeline.X` invocation that exits cleanly. Never import two model stacks in one script.
2. **Never let the pipeline touch full-resolution media until the final render.** All analysis and all editing decisions happen on 480p proxies. This is a standard pro workflow and it is what makes an 8 GB machine viable.

---

## 2. Architecture (unchanged spine, new model layer)

```
              ┌──────────────────────────────────────────────┐
              │        CLAUDE CODE (single orchestrator)      │
              │  reads JSON artifacts, makes taste calls,     │
              │  invokes deterministic tools one at a time    │
              └───────────────┬──────────────────────────────┘
                              │
   ┌──────────┬───────────┬───┴────────┬───────────┬──────────────┐
   ▼          ▼           ▼            ▼           ▼              ▼
 PROXY     ANALYZE      MUSIC      SEQUENCE     RENDER          QA
 (VT hw)   (local +     (librosa)  (scipy DP)   (x264/VT)      (ffprobe)
           cloud VLM)
   │          │           │            │           │              │
   └──────────┴───────────┴────────────┴───────────┴──────────────┘
        proxies/*.mp4 · clip_cards.json · editing_dna.json
        music_map.json · timeline.otio · out.mp4

Decision timecodes are resolution-independent → proxies analyse, originals render.
```

---

## 3. Component substitutions

| Stage | Original blueprint | M2 Air 8 GB version | Why |
|---|---|---|---|
| Scene detection | PySceneDetect on source | PySceneDetect on **480p proxy** | 20–40× cheaper, identical cut timecodes |
| Scene semantics | Qwen3-VL-8B local | **Gemini 2.5 Flash / Claude API**, 1–2 sampled frames per shot | 8B model needs 6 GB at 4-bit and will swap |
| Aesthetic score | pyiqa (MUSIQ, Q-Align, CLIP-IQA) | **open_clip ViT-L/14 + LAION aesthetic MLP** (~0.9 GB fp16), optional cloud re-rank of top 10 | Q-Align is 7B — impossible here |
| Technical quality | learned NR-IQA | **Laplacian variance, histogram clipping, flow-jitter** (OpenCV, CPU) | Free, deterministic, good enough |
| Detection / tracking | YOLO11m/x + SAM 2 | **YOLO11n exported to CoreML** (Neural Engine) + ByteTrack | Runs off-GPU, leaves memory for everything else |
| Beat + structure | allin1 (NATTEN) | **librosa** (beats, onsets, tempo) + optional **beat_this** for downbeats | NATTEN has no working MPS path |
| Stem separation | Demucs v4 htdemucs | **Skip by default**; if needed, `--segment 7 -d cpu` or Replicate | Minutes per track on CPU, heavy swap |
| ASR | WhisperX (faster-whisper) | **whisper.cpp + Metal**, `large-v3-turbo` Q5_0 (~550 MB) | Purpose-built for Apple Silicon |
| Captions | Remotion | **ASS / libass karaoke** via FFmpeg `subtitles=` | Remotion spawns headless Chrome, 2–4 GB per worker |
| Speed ramps | RIFE / FILM interpolation | **FFmpeg `setpts` on 60 fps source** | Shoot 60 fps and retime — looks better than interpolation anyway |
| Upscale | SeedVR2 / Real-ESRGAN | **Removed** — shoot at delivery resolution | Diffusion upscaling is out of reach |
| Draft encode | libx264 | **`h264_videotoolbox`** | Hardware, near-instant, negligible RAM |
| Final encode | libx264 slow CRF 19 | **Unchanged** — x264 slow CRF 19 | 30 s of 1080×1920 is within thermal budget |

---

## 4. Install order

Run these in sequence. Each block is a checkpoint — verify before moving on.

### 4.1 Base tools

```bash
# Homebrew (skip if present)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

brew install ffmpeg uv cmake git-lfs

# Verify the encoders and filters you depend on exist
ffmpeg -hide_banner -encoders | grep -E "videotoolbox|libx264"
ffmpeg -hide_banner -filters  | grep -E "lut3d|subtitles|zscale|tonemap|loudnorm"
```

If `zscale` is missing, your ffmpeg lacks libzimg — fall back to the `colorspace` filter for HDR→SDR (recipe in §7.3).

### 4.2 Project + Python environment

```bash
mkdir -p ~/reel-editor && cd ~/reel-editor
uv venv --python 3.12
source .venv/bin/activate

# Core (light, no torch) — install first, get Phase 0 working on this alone
uv pip install scenedetect[opencv-headless] opentimelineio numpy scipy \
               librosa soundfile pillow jsonschema rich typer
```

`opencv-headless` rather than full OpenCV: no GUI dependencies, smaller install, less memory.

### 4.3 Torch layer (only when you reach Phase 1)

```bash
uv pip install torch torchvision open_clip_torch
```

~2.5 GB of disk. Always export this before any torch script, so unimplemented MPS ops degrade to CPU instead of crashing:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

Put it in `.envrc` or the top of every runner script.

Aesthetic head — download the LAION v2 linear predictor weights (`sac+logos+ava1-l14-linearMSE.pth`) into `models/`. It is a small MLP on top of CLIP ViT-L/14 image embeddings, output scale ~1–10.

### 4.4 whisper.cpp with Metal

```bash
cd ~/tools && git clone https://github.com/ggml-org/whisper.cpp && cd whisper.cpp
cmake -B build && cmake --build build -j --config Release
sh ./models/download-ggml-model.sh large-v3-turbo-q5_0
./build/bin/whisper-cli -m models/ggml-large-v3-turbo-q5_0.bin -f samples/jfk.wav
```

Metal is enabled by default on macOS builds. Use `-ml` / word-level output flags for karaoke timing; if word timestamps are unstable, fall back to `base.en` with segment-level timing and split words evenly.

### 4.5 YOLO11n → CoreML (only when you reach Phase 3)

```bash
uv pip install ultralytics coremltools
yolo export model=yolo11n.pt format=coreml nms=True imgsz=640
```

Note Ultralytics is **AGPL-3.0** — buy an Enterprise licence if this ships commercially.

### 4.6 Cloud keys

```bash
export GEMINI_API_KEY=...        # or ANTHROPIC_API_KEY for VLM tagging
```

---

## 5. Memory budget — what may run concurrently

| Process | Peak | Concurrent with anything else? |
|---|---|---|
| FFmpeg proxy generation (VideoToolbox) | ~300 MB | Yes |
| PySceneDetect on 480p | ~400 MB | Yes |
| CLIP ViT-L/14 fp16 on MPS | ~1.5–2 GB | **No** |
| YOLO11n CoreML | ~300 MB | Yes |
| whisper.cpp turbo Q5_0 | ~1.2 GB | **No** |
| librosa analysis | ~500 MB | Yes |
| FFmpeg final render (x264 + filters) | ~800 MB | **No** |
| Claude Code (Node) | 0.5–1 GB | Always present |

**Enforcement pattern** — the orchestrator runs stages as isolated subprocesses:

```bash
uv run python -m pipeline.proxy      --in inputs/raw --out work/proxies
uv run python -m pipeline.scenes     --in work/proxies --out work/shots.json
uv run python -m pipeline.aesthetic  --in work/proxies --out work/aes.json   # CLIP only
uv run python -m pipeline.detect     --in work/proxies --out work/det.json   # CoreML only
uv run python -m pipeline.vlm_tag    --in work/proxies --out work/tags.json  # cloud only
uv run python -m pipeline.merge      --out work/clip_cards.json
```

Each exits before the next starts. Memory is genuinely reclaimed. This is the difference between the pipeline running and the machine swapping to a halt.

---

## 6. The proxy-first workflow

Generate once, analyse forever:

```bash
# 480p H.264 proxies — hardware encoded, tiny, decode-cheap
ffmpeg -i inputs/raw/CLIP.mp4 \
  -vf "scale=-2:480" -c:v h264_videotoolbox -b:v 2M \
  -c:a aac -b:a 96k -movflags +faststart work/proxies/CLIP.mp4
```

A 5-minute 4K batch converts in roughly a minute and lands at a few hundred MB instead of tens of GB. All timecodes in `clip_cards.json` and `timeline.otio` refer to source-relative seconds, so the final render pulls from `inputs/raw/` unchanged.

**Disk discipline for a 256 GB machine:** keep `inputs/raw/` on an external SSD, symlink it into the project, and set `TMPDIR` to the external drive before long renders. Delete `work/proxies/` between projects.

---

## 7. FFmpeg recipes tuned for M2

### 7.1 Draft render (iterate on this)

```bash
ffmpeg -i in.mp4 -vf "scale=1080:1920:force_original_aspect_ratio=increase,\
crop=1080:1920,setsar=1,format=yuv420p" \
  -c:v h264_videotoolbox -b:v 12M -maxrate 16M -bufsize 24M \
  -c:a aac -b:a 192k -ar 48000 -movflags +faststart -r 30 draft.mp4
```

Seconds, not minutes. Use for every iteration.

### 7.2 Hero render (final deliverable only)

```bash
ffmpeg -i in.mp4 \
 -vf "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,\
lut3d=luts/look.cube,\
eq=contrast=1.08:saturation=1.12,\
unsharp=5:5:0.6:5:5:0.0,\
subtitles=work/captions.ass,\
setsar=1,format=yuv420p" \
 -c:v libx264 -profile:v high -level 4.2 -preset slow -crf 19 \
 -x264-params "keyint=60:min-keyint=60:scenecut=0" \
 -colorspace bt709 -color_primaries bt709 -color_trc bt709 \
 -af "loudnorm=I=-14:TP=-1.5:LRA=11" \
 -c:a aac -b:a 192k -ar 48000 -ac 2 -movflags +faststart -r 30 out.mp4
```

### 7.3 iPhone HDR → SDR (do this first if shooting on iPhone)

iPhone records HLG HDR by default. Uploaded untagged, it washes out badly on Instagram. Normalise at proxy time and again at render:

```bash
# preferred (needs libzimg)
-vf "zscale=t=linear:npl=100,tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p"

# fallback (no zimg)
-vf "colorspace=all=bt709:iall=bt2020ncl:fast=1,format=yuv420p"
```

Alternatively, set the iPhone to record SDR (Settings → Camera → Formats → HDR Video off) and skip this entirely. For a fitness/travel reel workflow that is the pragmatic choice.

### 7.4 Thermal batching

The Air has no fan. Chain hero renders with a cooldown rather than running them back to back:

```bash
for f in work/final/*.mp4; do render "$f"; sleep 90; done
```

Expect the second sustained x264 encode to be noticeably slower than the first. Drafts on VideoToolbox do not have this problem.

---

## 8. Phased roadmap

### Phase 0 — Render spine (no ML at all)
Install §4.1–4.2. Build `pipeline/render.py`: walk an OTIO timeline, emit an FFmpeg filtergraph, produce a spec-compliant 1080×1920 BT.709 MP4. Hand-author a timeline JSON with three clips and a crossfade.
**Gate:** a hand-written timeline renders correctly and passes `ffprobe` spec checks. Do not proceed until this is solid — it is half the value of the system.

### Phase 1 — Proxy + analysis
Build `proxy.py`, `scenes.py` (PySceneDetect adaptive), `aesthetic.py` (CLIP + LAION head), `defects.py` (Laplacian, histogram, flow jitter), `vlm_tag.py` (cloud). Merge into `clip_cards.json` against a JSON Schema.
**Gate:** ranking on a 20-clip test set matches your own top-5 picks at least 3 out of 5 times.

### Phase 2 — Music + sequencing
`music.py` (librosa: tempo, beat frames, onset envelope, RMS energy curve; optional `beat_this` for downbeats). `sequence.py`: beat-quantised slot generation → cost matrix → `scipy.optimize.linear_sum_assignment` → DP ordering with variety and arc constraints → `timeline.otio`.
**Gate:** ≥80% of cuts land within ±80 ms of a beat; no two adjacent shots share a content class.

### Phase 3 — Look and framing
`color.py` (LUT + eq + selective sky/greens/water + HDR handling). `reframe.py` (YOLO11n CoreML + ByteTrack on proxies → smoothed crop path via Savitzky-Golay → per-shot `crop` expressions applied at full res). `transitions.py` (xfade, flash, whip via crop+blur, SFX mixing).
**Gate:** subject stays inside the 9:16 crop in >95% of frames on a walking-shot test clip.

### Phase 4 — Captions and QA
whisper.cpp → word timings → ASS generator with `\k` karaoke tags, styled from the Editing DNA typography block. `qa.py`: ffprobe spec check, duration, file size, loudness verification, colour tag verification.
**Gate:** QA passes automatically on three consecutive renders without manual fixes.

### Phase 5 — Claude Code orchestration
CLAUDE.md contracts, the ten skills, three read-only subagents, `settings.json` permissions. End-to-end: drop files in `inputs/`, get `out.mp4`.
**Gate:** a full reel from raw footage with no manual intervention.

---

## 9. Claude Code configuration

### 9.1 `CLAUDE.md` (the contract that keeps it deterministic)

```markdown
# Reel Editor — Operating Rules

## Hardware constraints (non-negotiable)
- Machine: MacBook Air M2, 8 GB unified memory, fanless.
- Run ONE model-loading process at a time. Never chain two torch stages in one script.
- Always analyse proxies in work/proxies/, never inputs/raw/.
- Draft renders use h264_videotoolbox. Only the final deliverable uses libx264.
- Always `export PYTORCH_ENABLE_MPS_FALLBACK=1` before torch scripts.

## Determinism
- If a step has a measurable ground truth (cuts, beats, scores, colour math, encoding),
  call the deterministic tool in pipeline/. Do not reason about it.
- Use the LLM only for: interpreting the style prompt, setting ranking weights,
  the final hook/sequence taste call, and QA triage.
- Every tool reads and writes named JSON validated against schemas/.

## Never
- Never load a VLM locally. Frame semantics go to the cloud API.
- Never install Remotion or spawn headless Chrome.
- Never run two renders concurrently.
```

### 9.2 Subagents (read-only, context isolation only)

`.claude/agents/clip-analyst.md`:

```markdown
---
name: clip-analyst
description: Scores and tags raw clips from proxies, returns clip_cards.json. Read-only.
tools: Read, Glob, Grep, Bash
model: sonnet
---
Run the pipeline stages in strict sequence as separate subprocesses:
proxy → scenes → aesthetic → detect → vlm_tag → merge.
Validate against schemas/clip_cards.schema.json.
Return ONLY the output path and a five-line summary. Never edit source media.
Never run two stages concurrently — this machine has 8 GB.
```

Same shape for `film-analyst` (inspiration DNA) and `qa-reviewer`.

### 9.3 MCP

Keep it minimal. Filesystem and GitHub only. Skip the community FFmpeg MCP servers — your own skills with typed contracts are safer and testable, and every extra Node process costs memory you do not have.

```json
{ "mcpServers": {
  "filesystem": {"command":"npx","args":["-y","@modelcontextprotocol/server-filesystem","./"]}
}}
```

---

## 10. Expected performance

For a 30-second reel from 20 clips (~5 minutes of 4K source). Estimates — measure on your own machine and adjust.

| Stage | Time |
|---|---|
| Proxy generation | 1–2 min |
| Scene detection | 15–30 s |
| Aesthetic scoring (~200 frames) | 30–60 s |
| Detection / tracking | 30–60 s |
| Cloud VLM tagging | 30–60 s |
| Music analysis | 10 s |
| Sequencing | <5 s |
| Captions (whisper.cpp) | 10–20 s |
| Draft render | 10–20 s |
| Hero render (x264 slow) | 2–4 min |
| **Total** | **~7–11 min** |

Iteration after the first pass is much faster — analysis artifacts are cached, so a re-sequence plus draft render is well under a minute.

---

## 11. Cloud offload cost

Per reel you send roughly 40–80 sampled frames at low resolution to a VLM for tagging and DNA extraction. On a fast, cheap tier (Gemini Flash class) this is cents per reel; on a frontier model it is still well under a dollar. Verify current pricing before you commit to a batch run — rates change.

If you want to remove the cloud dependency entirely, the fallback is CLIP/SigLIP zero-shot classification against a fixed tag vocabulary (waterfall, mountain, sunset, beach, city, gym, POV, drone…). It runs locally in the CLIP process you already have, costs nothing, and is meaningfully worse at narrative and framing judgement — but perfectly adequate for content tagging.

---

## 12. What changes if you upgrade

Triggers worth watching, in priority order:

1. **16 GB machine** → local Qwen2.5-VL-7B at 4-bit becomes viable; drop the cloud VLM dependency.
2. **24 GB (M-series Pro/Max)** → the original blueprint runs as written: Qwen3-VL-8B, pyiqa MUSIQ/CLIP-IQA, allin1, RIFE, Demucs. Also unlocks ProRes-heavy workflows and much faster sustained encoding (active cooling).
3. **External GPU box or a rented instance** → only worth it for diffusion upscaling (SeedVR2), which is the one thing genuinely impossible here.

Nothing in Phases 0–5 needs rewriting for any of these. The substitutions in §3 are configuration, not architecture — which is the point of keeping the JSON contracts fixed.

---

## Caveats

- Timing figures in §10 and memory figures in §5 are estimates for planning, not benchmarks. Measure early.
- `beat_this` and other recent beat trackers should be verified working on MPS before you depend on them; librosa alone is the safe baseline and covers beats, onsets, and tempo.
- `madmom` installs poorly on Python 3.12 / Apple Silicon. Do not put it on the critical path.
- Ultralytics YOLO is AGPL-3.0; Remotion needs a company licence; several IQA repos are non-commercial. Check licences before shipping commercially.
- Platform specs (file size caps, codec acceptance) drift. Re-verify before a major publish run. Ship SDR BT.709 — HDR remains badly handled on Instagram and TikTok.
- Homebrew's ffmpeg build occasionally ships without libzimg. The §7.3 fallback exists for that case.
