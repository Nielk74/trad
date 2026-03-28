# Real-Time Voice Translator — Design Spec
**Date:** 2026-03-28
**Status:** Approved

---

## Overview

A local real-time voice translator with a browser-based frontend. The user holds a button to speak in one language, releases it, and immediately hears the translation in the target language — with transcript and translation text also displayed. No external APIs or tokens required. All inference runs locally.

---

## Goals

- Fully offline — no API tokens, no cloud calls
- Support English, Chinese (Mandarin), French, German, Spanish, Italian, Portuguese, Dutch, Hindi, Arabic, Vietnamese
- Three hardware tiers: High (GPU), Medium (CPU laptop), Small (phone/ARM)
- Single command to start (`python app.py --config high|medium|small`)
- Vietnamese auto-fallback in High tier (Voxtral doesn't support VI; Medium models used transparently, indicated by a 🔄 icon)

---

## Language Support Matrix

| Language   | Code | High STT | High TTS | Med/Small |
|------------|------|----------|----------|-----------|
| English    | en   | ✅        | ✅        | ✅         |
| Chinese    | zh   | ✅        | ✅        | ✅         |
| French     | fr   | ✅        | ✅        | ✅         |
| German     | de   | ✅        | ✅        | ✅         |
| Spanish    | es   | ✅        | ✅        | ✅         |
| Italian    | it   | ✅        | ✅        | ✅         |
| Portuguese | pt   | ✅        | ✅        | ✅         |
| Dutch      | nl   | ✅        | ✅        | ✅         |
| Hindi      | hi   | ✅        | ✅        | ✅         |
| Arabic     | ar   | ✅        | ✅        | ✅         |
| Vietnamese | vi   | 🔄 fallback | 🔄 fallback | ✅      |

🔄 = High tier falls back to Medium models; UI shows fallback icon with tooltip.

---

## Hardware Tiers

### High — GPU (≥16GB VRAM, AMD ROCm or NVIDIA CUDA)

| Component   | Model                                      | Notes                            |
|-------------|--------------------------------------------|----------------------------------|
| STT         | `mistralai/Voxtral-Mini-4B-Realtime-2602`  | Streaming, <500ms latency        |
| Translation | `tencent/HY-MT1.5-7B` GGUF                | WMT25 winner, 33 languages       |
| TTS         | `mistralai/Voxtral-4B-TTS-2603` via vLLM  | 9 languages, preset voices       |
| VI fallback | Medium STT + TTS models                    | Auto-activated, shown with 🔄    |

Two processes: FastAPI + `vllm serve`. Start with `start.sh`.

### Medium — CPU laptop (≥8GB RAM, no GPU)

| Component   | Model                                        | Notes                              |
|-------------|----------------------------------------------|------------------------------------|
| STT         | `faster-whisper large-v3-turbo`              | EN/ZH/FR/VI, ~1.5GB, 6× faster    |
| Translation | `tencent/HY-MT1.5-1.8B` GPTQ-Int4           | ~900MB, same quality as 7B         |
| TTS         | `Kokoro-82M` ONNX int8 (EN/ZH/FR)           | ~80MB, Apache 2.0                  |
|             | Piper via `sherpa-onnx` (VI)                 | ~22–60MB per voice                 |

Single Python process.

### Small — Phone / Termux / ARM (≥2GB RAM)

| Component   | Model                                   | Notes                              |
|-------------|-----------------------------------------|------------------------------------|
| STT         | `faster-whisper small`                  | EN/ZH/FR/VI, ~466MB, real-time ARM |
| Translation | `Helsinki-NLP/opus-mt` per pair         | ~300MB/pair, CC-BY-4.0, CPU-only   |
| TTS         | Piper via `sherpa-onnx`                 | EN/ZH/FR/VI, 22–120MB/voice        |

Single Python process.

---

## Architecture

```
Browser (HTML/JS)
    │  ① hold button → MediaRecorder captures audio (WAV)
    │  ② release     → POST /transcribe
    │  ③             ← { text, fallback }
    │  ④             → POST /translate
    │  ⑤             ← { translation }
    │  ⑥             → POST /synthesize
    │  ⑦             ← { audio_b64, fallback }
    │  ⑧             play audio, display transcript + translation
    ▼
FastAPI (app.py)
    ├── GET  /languages   → list of supported languages with tier support flags
    ├── GET  /status      → active config, loaded models, fallback_active list
    ├── GET  /health      → per-model load status (for UI loading spinner)
    ├── POST /transcribe  → STT
    ├── POST /translate   → translation
    └── POST /synthesize  → TTS
    └── GET  /            → serves frontend/index.html

vLLM server (High tier only, localhost:8000)
    └── mistralai/Voxtral-4B-TTS-2603
```

Models are loaded lazily (on first use). The `/health` endpoint reflects load state.

---

## API Contract

### POST /transcribe
```json
Request:  { "audio": "<base64 WAV>", "lang": "fr" }
Response: { "text": "Bonjour tout le monde", "fallback": false }
```

### POST /translate
```json
Request:  { "text": "Bonjour tout le monde", "source": "fr", "target": "zh" }
Response: { "translation": "大家好" }
```

### POST /synthesize
```json
Request:  { "text": "大家好", "lang": "zh" }
Response: { "audio": "<base64 WAV>", "fallback": false }
```

### GET /languages
```json
Response: {
  "languages": [
    { "code": "fr", "name": "French", "flag": "🇫🇷", "tiers": ["high","medium","small"] },
    { "code": "vi", "name": "Vietnamese", "flag": "🇻🇳", "tiers": ["medium","small"], "fallback_in_high": true }
  ]
}
```

### GET /status
```json
Response: {
  "config": "high",
  "models_loaded": ["stt_voxtral", "translation_hymt7b", "tts_voxtral"],
  "fallback_active": ["vi_stt", "vi_tts"]
}
```

### POST /config
```json
Request:  { "config": "medium" }
Response: { "config": "medium", "status": "reloading" }
```
Triggers model unload + reload for the new tier. The UI polls `/health` until all models are ready.

**Constraints:**
- Audio input max 30s — requests exceeding this return HTTP 400
- If vLLM is unreachable at startup (High tier), tier auto-downgrades to Medium with a console warning

---

## Frontend

Single HTML file (`frontend/index.html`) with inline CSS and JS. No build step, no framework.

**Layout:**
```
┌─────────────────────────────────────────┐
│  🌐 Real-Time Translator    ⚙ [High ▾]  │
├─────────────────────────────────────────┤
│   Language A              Language B    │
│  [French    ▾]           [Chinese   ▾]  │
│                                         │
│  ┌──────────────┐    ┌──────────────┐   │
│  │  🎤 Hold to  │    │  🎤 Hold to  │   │
│  │    speak     │    │    speak     │   │
│  └──────────────┘    └──────────────┘   │
│                                         │
│  ┌─────────────────────────────────┐    │
│  │ transcript text                 │    │
│  │ → translated text          🔊   │    │
│  └─────────────────────────────────┘    │
│  [🔄 Fallback mode — tooltip on hover]  │
└─────────────────────────────────────────┘
```

**Behaviour:**
- Hold button: starts recording via `MediaRecorder` API
- Release button: stops recording, immediately kicks off transcribe → translate → synthesize pipeline
- Spinner shown during processing
- Audio plays automatically on completion
- 🔊 button replays the last audio
- 🔄 icon appears in status bar when any fallback model is active; tooltip lists which components are in fallback
- Config selector (top-right) calls the backend to switch tier at runtime (reloads models)
- Language dropdowns populated from `GET /languages`; languages unavailable in current tier are greyed out with a tooltip

---

## Project Structure

```
trad/
├── app.py                        # FastAPI entry point, routes
├── config.py                     # Tier definitions, model IDs, fallback rules
├── models/
│   ├── stt.py                    # STT abstraction (Voxtral / faster-whisper)
│   ├── translation.py            # Translation abstraction (HY-MT / opus-mt)
│   └── tts.py                    # TTS abstraction (Voxtral-TTS / Kokoro / Piper)
├── frontend/
│   └── index.html                # Single-file frontend
├── tests/
│   ├── conftest.py               # --config pytest flag, model fixture init
│   ├── download_fixtures.py      # Downloads audio samples, caches in tests/fixtures/audio/
│   ├── fixtures/
│   │   └── audio/                # Cached .wav clips (gitignored)
│   └── test_e2e.py               # E2E tests
├── start.sh                      # Starts vLLM (if High) + FastAPI
└── requirements/
    ├── requirements-high.txt
    ├── requirements-medium.txt
    └── requirements-small.txt
```

---

## Testing

### Fixture Audio Sources

Downloaded once by `tests/download_fixtures.py`, cached in `tests/fixtures/audio/` (gitignored):

| Language | Source | Clip | Ground-truth text |
|----------|--------|------|-------------------|
| EN | LibriSpeech (CC-BY-4.0) | 5s clip | known sentence |
| ZH | Mozilla Common Voice 17 (CC0) | 5s clip | known sentence |
| FR | Mozilla Common Voice 17 (CC0) | 5s clip | known sentence |
| VI | Mozilla Common Voice 17 (CC0) | 5s clip | known sentence |

Downloaded from `mozilla-foundation/common_voice_17_0` on HuggingFace and LibriSpeech via OpenSLR.

### E2E Test Suite (`tests/test_e2e.py`)

```
test_transcribe_en         audio clip → STT → assert expected words in transcript
test_transcribe_zh         audio clip → STT → assert ZH characters in output
test_transcribe_fr         audio clip → STT → assert expected words
test_transcribe_vi         audio clip → STT → assert VI text (fallback in High)
test_translate_fr_to_zh    known phrase → translate → assert ZH non-empty + plausible
test_translate_zh_to_en    known phrase → translate → assert EN non-empty
test_translate_en_to_vi    known phrase → translate → assert VI non-empty
test_synthesize_en         "Hello world" → TTS → valid WAV, duration > 0.5s
test_synthesize_zh         "大家好" → TTS → valid WAV
test_synthesize_vi         VI text → TTS → valid WAV (fallback in High)
test_full_pipeline_fr_zh   FR audio → STT → translate → TTS → all steps pass
test_full_pipeline_en_vi   EN audio → STT → translate → TTS → fallback flag set in High
test_fallback_indicator    High tier + VI → response includes fallback: true
```

### Running Tests

```bash
# Default: Small tier (fast, no GPU needed)
pytest tests/

# Medium tier
pytest tests/ --config=medium

# High tier (requires GPU + vLLM running)
pytest tests/ --config=high
```

`conftest.py` reads `--config`, initialises the correct model set, and exposes it as a session-scoped fixture. All tests use Small models by default to keep CI fast.

---

## Startup

```bash
# Small / Medium
python app.py --config small
python app.py --config medium

# High (two steps, or use start.sh)
vllm serve mistralai/Voxtral-4B-TTS-2603 --omni &
python app.py --config high

# Or one command
./start.sh --config high
```

`start.sh` handles launching vLLM in the background for High, then starts FastAPI, and shuts down vLLM on exit (trap SIGINT).

---

## Out of Scope

- Speaker diarisation / multi-party conversations
- Saving conversation history
- Custom voice cloning
- Translation memory or glossary injection
- Mobile app packaging (APK/PWA) — the Small tier targets Termux browser, not a native app
