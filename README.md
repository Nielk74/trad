# Real-Time Voice Translator

Fully offline, real-time voice translation supporting 11 languages. Speak in one language, hear the translation in another — no API keys, no cloud.

Three hardware tiers let you run on anything from a Raspberry Pi to a GPU workstation.

## Demo

1. Open the web UI at `http://localhost:8080`
2. Select source and target languages
3. Press **Record**, speak, release
4. The transcript and translation appear instantly; synthesized audio plays automatically

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

### Translation pairs

All pairs between the 11 languages are supported. Non-English pairs are pivoted through English in the small tier (e.g. `zh → fr` = `zh → en → fr`). Medium and high tiers use a single multilingual model that translates directly.

## Quick Start

### Prerequisites

- Python 3.10+
- `espeak-ng` (required for Piper TTS phonemization):
  ```bash
  # Debian/Ubuntu
  sudo apt install espeak-ng
  # macOS
  brew install espeak-ng
  ```

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
python app.py --config small

# Medium tier — better quality, ~3 GB models
python app.py --config medium

# High tier — best quality, requires GPU + vLLM running
python app.py --config high

# Custom host/port
python app.py --config small --host 0.0.0.0 --port 8080
```

Models are downloaded automatically on first use and cached in `models_cache/`.

Open `http://localhost:8080` in your browser.

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
curl -s -X POST http://localhost:8080/transcribe \
  -H "Content-Type: application/json" \
  -d '{"audio": "<base64-wav>", "lang": "fr"}'
# → {"text": "Bonjour le monde", "fallback": false}

# Translate
curl -s -X POST http://localhost:8080/translate \
  -H "Content-Type: application/json" \
  -d '{"text": "Bonjour le monde", "source": "fr", "target": "en"}'
# → {"translation": "Hello world"}

# Synthesize
curl -s -X POST http://localhost:8080/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "lang": "en"}'
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
  test_stt.py           STT unit tests
  test_translation.py   Translation unit tests
  test_tts.py           TTS unit tests
  test_audio_utils.py   Audio utility tests
  download_fixtures.py  Downloads FLEURS audio clips for tests
```
