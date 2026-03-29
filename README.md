# Real-Time Voice Translator

Fully offline, real-time voice translation supporting 11 languages. Speak in one language, hear the translation in another — no API keys, no cloud.

Three hardware tiers let you run on anything from a Raspberry Pi to a GPU workstation.

## Demo

1. Generate a self-signed TLS certificate (required for microphone access):
   ```bash
   openssl req -x509 -newkey rsa:2048 -keyout key.pem -out cert.pem -days 3650 -nodes -subj "/CN=localhost"
   ```
2. Open the web UI at `https://localhost:3003` (accept the browser's self-signed cert warning)
3. Select source and target languages
4. Hold **Record**, speak, release
5. The transcript and translation appear instantly; synthesized audio plays automatically
6. Enable **Clone my voice** to hear the translation spoken in your own voice

## Hardware Tiers

| | **Small** | **Medium** | **High** |
|---|---|---|---|
| **Target hardware** | Raspberry Pi 4, low-end laptop | Modern laptop / desktop (no GPU) | GPU workstation |
| **RAM** | 4 GB | 8 GB | 16 GB + VRAM |
| **GPU** | Not required | Not required | NVIDIA / AMD required |
| **Disk (models)** | ~500 MB | ~3 GB | ~20 GB |
| **STT** | Whisper Small (CPU int8) | Whisper large-v3-turbo (CPU int8) | Voxtral-Mini-4B (GPU fp16) |
| **Translation** | Helsinki-NLP opus-mt (CPU) | HY-MT1.5-1.8B GGUF (CPU) | HY-MT1.5-7B GGUF (GPU) |
| **TTS** | Piper via sherpa-onnx (CPU) | Kokoro-82M ONNX (CPU) | Voxtral-4B-TTS via vLLM (GPU) |
| **Voice cloning** | OuteTTS 0.3-1B (CPU, isolated venv) | Qwen3-TTS 1.7B (GPU, isolated venv) | Qwen3-TTS 1.7B (GPU, isolated venv) |

> **High tier** requires a running vLLM instance (see [High tier setup](#high-tier-setup)).

## Language Support

All 11 languages are available in every tier. The difference is *which model* handles each language and whether a fallback is used.

| Language | Code | Small | Medium | High |
|---|---|---|---|---|
| English | `en` | Whisper + opus-mt + Piper | Whisper + HY-MT + Kokoro | Voxtral native |
| French | `fr` | Whisper + opus-mt + Piper | Whisper + HY-MT + Kokoro | Voxtral native |
| Chinese | `zh` | Whisper + opus-mt + MeloTTS | Whisper + HY-MT + Kokoro | Voxtral native |
| German | `de` | Whisper + opus-mt + Piper | Whisper + HY-MT + Piper | Voxtral native |
| Spanish | `es` | Whisper + opus-mt + Piper | Whisper + HY-MT + Kokoro | Voxtral native |
| Italian | `it` | Whisper + opus-mt + Piper | Whisper + HY-MT + Kokoro | Voxtral native |
| Portuguese | `pt` | Whisper + opus-mt + Piper | Whisper + HY-MT + Kokoro | Voxtral native |
| Dutch | `nl` | Whisper + opus-mt + Piper | Whisper + HY-MT + Piper | Voxtral native |
| Hindi | `hi` | Whisper + opus-mt + Piper | Whisper + HY-MT + Kokoro | Voxtral native |
| Arabic | `ar` | Whisper + opus-mt + Piper | Whisper + HY-MT + Piper | Voxtral native |
| Vietnamese | `vi` | Whisper + opus-mt + Piper | Whisper + HY-MT + Piper | Whisper fallback + Piper fallback* |

*Vietnamese is not natively supported by Voxtral. In the high tier, STT falls back to Whisper large-v3-turbo and TTS falls back to Piper. The API response includes `"fallback": true` when this occurs.

### Voice cloning

Enable the **Clone my voice** checkbox in the UI to synthesize the translation in your own voice. The recording you just made is used as the reference sample — no separate upload needed.

| Tier | Model | Released | Languages supported |
|---|---|---|---|
| Small | [OuteTTS 0.3-1B](https://github.com/edwko/OuteTTS) (CPU, isolated venv) | 2025 | en, zh, de, fr, ja, ko |
| Medium | [Qwen3-TTS 1.7B](https://github.com/QwenLM/Qwen3-TTS) (GPU) | Jan 2026 | en, zh, fr, de, es, it, pt, ru, ja, ko |
| High | [Qwen3-TTS 1.7B](https://github.com/QwenLM/Qwen3-TTS) (GPU) | Jan 2026 | en, zh, fr, de, es, it, pt, ru, ja, ko |

If the target language is not supported by the cloning model (e.g. Vietnamese, Arabic on small tier), synthesis falls back to the standard voice automatically.

Voice cloning models run in **isolated venvs** to avoid dependency conflicts with the main app. Set them up once before use:

```bash
# Both tiers
python workers/setup_clone_venvs.py

# Or individually
python workers/setup_clone_venvs.py --tier small
python workers/setup_clone_venvs.py --tier medium
```

Workers launch automatically on the first clone request and stay alive. If a worker venv is not set up, the checkbox falls back to standard TTS silently.

### Translation pairs

All pairs between the 11 languages are supported. Non-English pairs are pivoted through English in the small tier (e.g. `zh → fr` = `zh → en → fr`). Medium and high tiers use a single multilingual model that translates directly.

## Quick Start

### Prerequisites

- Python 3.10+
- `ffmpeg` (required for audio decoding):
  ```bash
  # Debian/Ubuntu
  sudo apt install ffmpeg
  # macOS
  brew install ffmpeg
  ```
- `espeak-ng` (required for Piper TTS phonemization):
  ```bash
  # Debian/Ubuntu
  sudo apt install espeak-ng
  # macOS
  brew install espeak-ng
  ```
- A self-signed TLS certificate (required for microphone access in browsers):
  ```bash
  openssl req -x509 -newkey rsa:2048 -keyout key.pem -out cert.pem -days 3650 -nodes -subj "/CN=localhost"
  ```
  The app auto-detects `key.pem` / `cert.pem` in the working directory and serves HTTPS automatically.

### Install

```bash
python -m venv .venv && source .venv/bin/activate

# Small tier (recommended to start)
pip install -r requirements/requirements-small.txt

# Medium tier
pip install -r requirements/requirements-medium.txt

# High tier
pip install -r requirements/requirements-high.txt
```

### Run

```bash
# Small tier — no GPU needed, ~500 MB models, all 11 languages
.venv/bin/python app.py --config small

# Medium tier — better quality, ~3 GB models
.venv/bin/python app.py --config medium

# High tier — best quality, requires GPU + vLLM running
.venv/bin/python app.py --config high

# Custom host/port
.venv/bin/python app.py --config small --host 0.0.0.0 --port 3003
```

Or use the convenience script:

```bash
./start.sh [small|medium|high] [host] [port]
# e.g.
./start.sh medium
```

Models are downloaded automatically on first use and cached in `models_cache/`.

Open `https://localhost:3003` in your browser (accept the self-signed certificate warning).

### High tier setup

The high tier uses vLLM to serve Voxtral TTS. Start it before launching the app:

```bash
# NVIDIA
vllm serve mistralai/Voxtral-4B-TTS-2603

# AMD ROCm — see requirements/requirements-high.txt for ROCm install notes
vllm serve mistralai/Voxtral-4B-TTS-2603
```

Then in a separate terminal:

```bash
python app.py --config high
```

## API

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Web UI |
| `GET` | `/health` | Model load status |
| `GET` | `/status` | Config, models loaded, active fallbacks |
| `GET` | `/languages` | Supported languages with per-tier info |
| `POST` | `/transcribe` | Audio → text (base64 WAV/webm input) |
| `POST` | `/translate` | Text → translated text |
| `POST` | `/synthesize` | Text → audio (base64 WAV output) |

### Example

```bash
# Transcribe
curl -sk -X POST https://localhost:3003/transcribe \
  -H "Content-Type: application/json" \
  -d '{"audio": "<base64-wav>", "lang": "fr"}'
# → {"text": "Bonjour le monde", "fallback": false}

# Translate
curl -sk -X POST https://localhost:3003/translate \
  -H "Content-Type: application/json" \
  -d '{"text": "Bonjour le monde", "source": "fr", "target": "en"}'
# → {"translation": "Hello world"}

# Synthesize (standard)
curl -sk -X POST https://localhost:3003/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "lang": "en"}'
# → {"audio": "<base64-wav>", "fallback": false}

# Synthesize with voice cloning (pass reference audio as base64 WAV)
curl -sk -X POST https://localhost:3003/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "lang": "en", "reference_audio": "<base64-wav>"}'
# → {"audio": "<base64-wav>", "fallback": false}
```

## Tests

```bash
# Download audio fixtures once (requires internet, ~30 MB from FLEURS dataset)
python tests/download_fixtures.py

# Run e2e tests (small tier, no GPU needed)
pytest tests/test_e2e.py -v

# Run with medium or high tier
pytest tests/test_e2e.py -v --config=medium
pytest tests/test_e2e.py -v --config=high   # requires vLLM running

# Run all tests
pytest tests/ -v
```

Expected result: **16 passed, 2 skipped** (the two skipped tests require `--config=high` with vLLM running).

### Playwright browser tests

Requires the app to be running:

```bash
# Install playwright (once)
pip install playwright pytest-playwright
playwright install chromium

# Run
pytest tests/test_playwright.py -v
```

48 tests covering page load, API endpoints, language selects, tier switching, mobile touch recording (with mocked MediaRecorder), network request monitoring, accessibility, and error handling.

## Project Structure

```
app.py                  FastAPI application and API routes
config.py               Model configs, language definitions, download URLs
audio_utils.py          Audio preprocessing (resampling, format conversion)
models/
  stt.py                Speech-to-text (faster-whisper / Voxtral)
  translation.py        Translation (opus-mt / HY-MT1.5 GGUF)
  tts.py                Text-to-speech (Piper / Kokoro / Voxtral-TTS)
frontend/
  index.html            Single-page web UI
requirements/
  requirements-small.txt
  requirements-medium.txt
  requirements-high.txt
tests/
  test_e2e.py           End-to-end pipeline tests
  test_playwright.py    Browser / UI tests (Playwright)
  test_stt.py           STT unit tests
  test_translation.py   Translation unit tests
  test_tts.py           TTS unit tests
  test_audio_utils.py   Audio utility tests
  download_fixtures.py  Downloads FLEURS audio clips for tests
```
