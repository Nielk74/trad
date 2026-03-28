# Real-Time Voice Translator — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fully local real-time voice translator with a browser UI, FastAPI backend, and three hardware tiers (High/GPU, Medium/CPU, Small/ARM).

**Architecture:** FastAPI serves the API and static frontend. A shared model abstraction layer (`models/`) loads tier-appropriate models lazily. High tier spawns a separate vLLM process for Voxtral-TTS; Medium and Small run entirely in one Python process. Vietnamese auto-falls back to Medium models when running High tier.

**Tech Stack:** Python 3.11+, FastAPI, uvicorn, faster-whisper, transformers, llama-cpp-python, kokoro-onnx, sherpa-onnx, vLLM (High TTS only), pydub, soundfile, pytest, HuggingFace Hub

---

## File Map

| File | Responsibility |
|------|---------------|
| `config.py` | Language defs, model IDs per tier, fallback rules, voice maps |
| `audio_utils.py` | WAV/webm conversion, resample to 16kHz mono PCM |
| `models/__init__.py` | Empty |
| `models/stt.py` | STT: faster-whisper (small/large-v3-turbo) or Voxtral-Mini |
| `models/translation.py` | Translation: opus-mt (Small) or HY-MT1.5 GGUF (Med/High) |
| `models/tts.py` | TTS: Piper/sherpa-onnx (Small), Kokoro-82M (Medium), vLLM client (High) |
| `app.py` | FastAPI app, all routes, model lifecycle |
| `frontend/index.html` | Single-page UI (inline CSS+JS, no build step) |
| `tests/__init__.py` | Empty |
| `tests/conftest.py` | `--config` flag, session-scoped model fixtures |
| `tests/download_fixtures.py` | Downloads FLEURS audio clips, caches in `tests/fixtures/audio/` |
| `tests/test_stt.py` | STT unit tests |
| `tests/test_translation.py` | Translation unit tests |
| `tests/test_tts.py` | TTS unit tests |
| `tests/test_e2e.py` | Full pipeline E2E tests |
| `start.sh` | Starts vLLM (High only) then FastAPI; traps SIGINT to clean up |
| `requirements/requirements-high.txt` | GPU dependencies |
| `requirements/requirements-medium.txt` | CPU laptop dependencies |
| `requirements/requirements-small.txt` | ARM/phone dependencies |
| `.gitignore` | Ignore models cache, fixtures, venv |

---

## Task 1: Git Init + Project Scaffold

**Files:**
- Create: `.gitignore`
- Create: `requirements/requirements-high.txt`
- Create: `requirements/requirements-medium.txt`
- Create: `requirements/requirements-small.txt`
- Create: `models/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/fixtures/audio/.gitkeep`

- [ ] **Step 1: Initialize git repo**

```bash
cd /data/data/com.termux/files/home/projects/trad
git init
```

- [ ] **Step 2: Create .gitignore**

```
# Python
__pycache__/
*.pyc
.venv/
venv/
*.egg-info/

# Models cache
~/.cache/huggingface/
models_cache/
*.gguf
*.onnx
*.bin

# Test fixtures (downloaded at test time)
tests/fixtures/audio/*.wav
tests/fixtures/audio/*.flac
tests/fixtures/audio/*.mp3

# IDE
.idea/
.vscode/

# OS
.DS_Store
```

- [ ] **Step 3: Create requirements-small.txt**

```
fastapi>=0.110.0
uvicorn>=0.27.0
python-multipart>=0.0.9
faster-whisper>=1.1.0
transformers>=4.35.0
sentencepiece>=0.1.99
sacremoses>=0.1.0
torch>=2.1.0
torchaudio>=2.1.0
sherpa-onnx>=1.10.0
pydub>=0.25.1
soundfile>=0.12.1
numpy>=1.26.0
huggingface_hub>=0.21.0
pytest>=7.4.0
pytest-asyncio>=0.23.0
httpx>=0.26.0
```

- [ ] **Step 4: Create requirements-medium.txt**

```
fastapi>=0.110.0
uvicorn>=0.27.0
python-multipart>=0.0.9
faster-whisper>=1.1.0
llama-cpp-python>=0.2.90
huggingface_hub>=0.21.0
kokoro-onnx>=0.4.0
sherpa-onnx>=1.10.0
pydub>=0.25.1
soundfile>=0.12.1
numpy>=1.26.0
pytest>=7.4.0
pytest-asyncio>=0.23.0
httpx>=0.26.0
```

- [ ] **Step 5: Create requirements-high.txt**

```
# AMD ROCm: install torch first:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.1
# AMD ROCm: install llama-cpp-python with:
# CMAKE_ARGS="-DGGML_HIPBLAS=on" pip install llama-cpp-python
# Then install vLLM with ROCm support per: https://docs.vllm.ai/en/latest/getting_started/amd-installation.html

fastapi>=0.110.0
uvicorn>=0.27.0
python-multipart>=0.0.9
transformers>=5.2.0
torch>=2.1.0
torchaudio>=2.1.0
faster-whisper>=1.1.0
llama-cpp-python>=0.2.90
huggingface_hub>=0.21.0
httpx>=0.26.0
sherpa-onnx>=1.10.0
pydub>=0.25.1
soundfile>=0.12.1
numpy>=1.26.0
pytest>=7.4.0
pytest-asyncio>=0.23.0
```

- [ ] **Step 6: Create empty init files and fixture placeholder**

```bash
mkdir -p models tests/fixtures/audio frontend docs/superpowers/specs docs/superpowers/plans requirements
touch models/__init__.py tests/__init__.py tests/fixtures/audio/.gitkeep
```

- [ ] **Step 7: Create tests/conftest.py now (needed by all unit tests)**

```python
# tests/conftest.py
import pytest
from models.stt import STTModel
from models.translation import TranslationModel
from models.tts import TTSModel


def pytest_addoption(parser):
    parser.addoption(
        "--config",
        action="store",
        default="small",
        choices=["high", "medium", "small"],
        help="Model tier to use for tests",
    )


@pytest.fixture(scope="session")
def config(request):
    return request.config.getoption("--config")


@pytest.fixture(scope="session")
def stt_model(config):
    model = STTModel(config)
    model.load()
    return model


@pytest.fixture(scope="session")
def translation_model(config):
    model = TranslationModel(config)
    model.load()
    return model


@pytest.fixture(scope="session")
def tts_model(config):
    model = TTSModel(config)
    model.load()
    return model
```

- [ ] **Step 8: First commit**

```bash
git add .
git commit -m "chore: project scaffold, requirements, gitignore, pytest conftest"
```

---

## Task 2: config.py

**Files:**
- Create: `config.py`

- [ ] **Step 1: Write config.py**

```python
# config.py
from typing import Optional

LANGUAGES = [
    {"code": "en", "name": "English",    "flag": "🇬🇧", "voxtral_stt": True,  "voxtral_tts": True},
    {"code": "zh", "name": "Chinese",    "flag": "🇨🇳", "voxtral_stt": True,  "voxtral_tts": True},
    {"code": "fr", "name": "French",     "flag": "🇫🇷", "voxtral_stt": True,  "voxtral_tts": True},
    {"code": "de", "name": "German",     "flag": "🇩🇪", "voxtral_stt": True,  "voxtral_tts": True},
    {"code": "es", "name": "Spanish",    "flag": "🇪🇸", "voxtral_stt": True,  "voxtral_tts": True},
    {"code": "it", "name": "Italian",    "flag": "🇮🇹", "voxtral_stt": True,  "voxtral_tts": True},
    {"code": "pt", "name": "Portuguese", "flag": "🇵🇹", "voxtral_stt": True,  "voxtral_tts": True},
    {"code": "nl", "name": "Dutch",      "flag": "🇳🇱", "voxtral_stt": True,  "voxtral_tts": True},
    {"code": "hi", "name": "Hindi",      "flag": "🇮🇳", "voxtral_stt": True,  "voxtral_tts": True},
    {"code": "ar", "name": "Arabic",     "flag": "🇸🇦", "voxtral_stt": True,  "voxtral_tts": True},
    {"code": "vi", "name": "Vietnamese", "flag": "🇻🇳", "voxtral_stt": False, "voxtral_tts": False},
]

# Languages natively supported by Voxtral (STT and TTS)
VOXTRAL_LANGS = {l["code"] for l in LANGUAGES if l["voxtral_stt"]}

# Voxtral TTS voice per language (preset voices from Voxtral-4B-TTS-2603)
VOXTRAL_TTS_VOICES: dict[str, str] = {
    "en": "casual_male",
    "fr": "Angele",
    "de": "casual_male",
    "es": "Gustavo",
    "it": "casual_male",
    "pt": "Gustavo",
    "nl": "casual_male",
    "hi": "Sanchit",
    "ar": "casual_male",
    "zh": "casual_male",
}

# Kokoro-82M voice per language (Medium tier EN/ZH/FR; others fall back to Piper)
KOKORO_VOICES: dict[str, Optional[str]] = {
    "en": "af_heart",
    "zh": "zf_xiaobei",
    "fr": "ff_siwis",
    "de": None,
    "es": "ef_dora",
    "it": "if_sara",
    "pt": "pf_dora",
    "nl": None,
    "hi": "hf_alpha",
    "ar": None,
    "vi": None,
}

# Piper sherpa-onnx model download info (used for VI in all tiers, and all langs in Small)
# Source: https://k2-fsa.github.io/sherpa/onnx/tts/pretrained_models/vits.html
PIPER_MODELS: dict[str, dict] = {
    "vi": {
        "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-vi_VN-vivos-x_low.tar.bz2",
        "dir": "vits-piper-vi_VN-vivos-x_low",
        "model": "vi_VN-vivos-x_low.onnx",
        "tokens": "tokens.txt",
        "espeak_lang": "vi",
    },
    "en": {
        "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-lessac-medium.tar.bz2",
        "dir": "vits-piper-en_US-lessac-medium",
        "model": "en_US-lessac-medium.onnx",
        "tokens": "tokens.txt",
        "espeak_lang": "en-us",
    },
    "fr": {
        "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-fr_FR-mls-medium.tar.bz2",
        "dir": "vits-piper-fr_FR-mls-medium",
        "model": "fr_FR-mls-medium.onnx",
        "tokens": "tokens.txt",
        "espeak_lang": "fr",
    },
    "zh": {
        "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-melo-tts-zh_en.tar.bz2",
        "dir": "vits-melo-tts-zh_en",
        "model": "model.onnx",
        "tokens": "tokens.txt",
        "espeak_lang": "zh",
    },
    "de": {
        "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-de_DE-mls-medium.tar.bz2",
        "dir": "vits-piper-de_DE-mls-medium",
        "model": "de_DE-mls-medium.onnx",
        "tokens": "tokens.txt",
        "espeak_lang": "de",
    },
    "es": {
        "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-es_ES-mls_10246-medium.tar.bz2",
        "dir": "vits-piper-es_ES-mls_10246-medium",
        "model": "es_ES-mls_10246-medium.onnx",
        "tokens": "tokens.txt",
        "espeak_lang": "es",
    },
    "it": {
        "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-it_IT-riccardo-x_low.tar.bz2",
        "dir": "vits-piper-it_IT-riccardo-x_low",
        "model": "it_IT-riccardo-x_low.onnx",
        "tokens": "tokens.txt",
        "espeak_lang": "it",
    },
    "pt": {
        "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-pt_BR-faber-medium.tar.bz2",
        "dir": "vits-piper-pt_BR-faber-medium",
        "model": "pt_BR-faber-medium.onnx",
        "tokens": "tokens.txt",
        "espeak_lang": "pt",
    },
    "nl": {
        "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-nl_NL-mls-medium.tar.bz2",
        "dir": "vits-piper-nl_NL-mls-medium",
        "model": "nl_NL-mls-medium.onnx",
        "tokens": "tokens.txt",
        "espeak_lang": "nl",
    },
    "hi": {
        "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-hi_IN-deep-medium.tar.bz2",
        "dir": "vits-piper-hi_IN-deep-medium",
        "model": "hi_IN-deep-medium.onnx",
        "tokens": "tokens.txt",
        "espeak_lang": "hi",
    },
    "ar": {
        "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-ar_JO-kareem-medium.tar.bz2",
        "dir": "vits-piper-ar_JO-kareem-medium",
        "model": "ar_JO-kareem-medium.onnx",
        "tokens": "tokens.txt",
        "espeak_lang": "ar",
    },
}

# Helsinki-NLP OPUS-MT model IDs (Small tier translation)
# Non-EN pairs go through EN as pivot (e.g. zh→fr = zh→en→fr)
OPUS_MT_MODELS: dict[tuple[str, str], str] = {
    ("en", "zh"): "Helsinki-NLP/opus-mt-en-zh",
    ("zh", "en"): "Helsinki-NLP/opus-mt-zh-en",
    ("en", "fr"): "Helsinki-NLP/opus-mt-tc-big-en-fr",
    ("fr", "en"): "Helsinki-NLP/opus-mt-tc-big-fr-en",
    ("en", "de"): "Helsinki-NLP/opus-mt-en-de",
    ("de", "en"): "Helsinki-NLP/opus-mt-de-en",
    ("en", "es"): "Helsinki-NLP/opus-mt-en-es",
    ("es", "en"): "Helsinki-NLP/opus-mt-es-en",
    ("en", "it"): "Helsinki-NLP/opus-mt-en-it",
    ("it", "en"): "Helsinki-NLP/opus-mt-it-en",
    ("en", "pt"): "Helsinki-NLP/opus-mt-en-pt",
    ("pt", "en"): "Helsinki-NLP/opus-mt-pt-en",
    ("en", "nl"): "Helsinki-NLP/opus-mt-en-nl",
    ("nl", "en"): "Helsinki-NLP/opus-mt-nl-en",
    ("en", "hi"): "Helsinki-NLP/opus-mt-en-hi",
    ("hi", "en"): "Helsinki-NLP/opus-mt-hi-en",
    ("en", "ar"): "Helsinki-NLP/opus-mt-en-ar",
    ("ar", "en"): "Helsinki-NLP/opus-mt-ar-en",
    ("en", "vi"): "Helsinki-NLP/opus-mt-en-vi",
    ("vi", "en"): "Helsinki-NLP/opus-mt-vi-en",
}

TIER_CONFIGS: dict[str, dict] = {
    "high": {
        "stt_model": "mistralai/Voxtral-Mini-4B-Realtime-2602",
        "stt_fallback_model": "large-v3-turbo",   # faster-whisper model size
        "translation_model_repo": "tencent/HY-MT1.5-7B-GGUF",
        "translation_model_file": "HY-MT1.5-7B-Q4_K_M.gguf",
        "tts_vllm_url": "http://localhost:8000",
        "tts_vllm_model": "mistralai/Voxtral-4B-TTS-2603",
    },
    "medium": {
        "stt_model": "large-v3-turbo",             # faster-whisper model size
        "translation_model_repo": "tencent/HY-MT1.5-1.8B-GGUF",
        "translation_model_file": "HY-MT1.5-1.8B-Q4_K_M.gguf",
        "tts_kokoro_model": "NeuML/kokoro-int8-onnx",
    },
    "small": {
        "stt_model": "small",                      # faster-whisper model size
        "translation_model": "opus-mt",
        "tts_model": "piper",
    },
}

# Directory where downloaded models/voices are cached
MODELS_CACHE_DIR = "models_cache"
```

- [ ] **Step 2: Commit**

```bash
git add config.py
git commit -m "feat: add config with language definitions and tier model IDs"
```

---

## Task 3: audio_utils.py

**Files:**
- Create: `audio_utils.py`
- Create: `tests/test_audio_utils.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_audio_utils.py
import io
import numpy as np
import soundfile as sf
import pytest
from audio_utils import to_pcm16k


def make_wav_bytes(duration_s: float = 1.0, sample_rate: int = 44100) -> bytes:
    """Generate a simple sine wave WAV in memory."""
    samples = np.sin(2 * np.pi * 440 * np.arange(int(sample_rate * duration_s)) / sample_rate)
    buf = io.BytesIO()
    sf.write(buf, samples.astype(np.float32), sample_rate, format="WAV")
    return buf.getvalue()


def test_to_pcm16k_returns_numpy_array():
    wav = make_wav_bytes(duration_s=1.0, sample_rate=44100)
    arr, sr = to_pcm16k(wav)
    assert isinstance(arr, np.ndarray)
    assert sr == 16000


def test_to_pcm16k_resamples_to_16khz():
    wav = make_wav_bytes(duration_s=1.0, sample_rate=44100)
    arr, sr = to_pcm16k(wav)
    assert sr == 16000
    assert len(arr) == pytest.approx(16000, abs=200)


def test_to_pcm16k_mono():
    """Stereo input must be converted to mono."""
    samples = np.zeros((44100, 2), dtype=np.float32)
    buf = io.BytesIO()
    sf.write(buf, samples, 44100, format="WAV")
    arr, sr = to_pcm16k(buf.getvalue())
    assert arr.ndim == 1


def test_to_pcm16k_rejects_audio_over_30s():
    from audio_utils import AudioTooLongError
    wav = make_wav_bytes(duration_s=31.0, sample_rate=16000)
    with pytest.raises(AudioTooLongError):
        to_pcm16k(wav)
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/test_audio_utils.py -v
```
Expected: `ModuleNotFoundError: No module named 'audio_utils'`

- [ ] **Step 3: Implement audio_utils.py**

```python
# audio_utils.py
import io
import numpy as np
import soundfile as sf

try:
    import librosa
    _HAS_LIBROSA = True
except ImportError:
    _HAS_LIBROSA = False

MAX_AUDIO_SECONDS = 30


class AudioTooLongError(ValueError):
    pass


def to_pcm16k(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    """
    Convert raw audio bytes (WAV or webm/ogg via pydub fallback) to
    a 16kHz mono float32 numpy array.

    Returns (samples, sample_rate=16000).
    Raises AudioTooLongError if audio exceeds MAX_AUDIO_SECONDS.
    """
    buf = io.BytesIO(audio_bytes)
    try:
        samples, sr = sf.read(buf, dtype="float32", always_2d=False)
    except Exception:
        # fallback: try pydub (handles webm/ogg from browsers)
        from pydub import AudioSegment
        buf.seek(0)
        seg = AudioSegment.from_file(buf)
        seg = seg.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        raw = np.frombuffer(seg.raw_data, dtype=np.int16).astype(np.float32) / 32768.0
        duration = len(raw) / 16000
        if duration > MAX_AUDIO_SECONDS:
            raise AudioTooLongError(f"Audio is {duration:.1f}s; max is {MAX_AUDIO_SECONDS}s")
        return raw, 16000

    # to mono
    if samples.ndim == 2:
        samples = samples.mean(axis=1)

    duration = len(samples) / sr
    if duration > MAX_AUDIO_SECONDS:
        raise AudioTooLongError(f"Audio is {duration:.1f}s; max is {MAX_AUDIO_SECONDS}s")

    # resample to 16kHz
    if sr != 16000:
        if _HAS_LIBROSA:
            samples = librosa.resample(samples, orig_sr=sr, target_sr=16000)
        else:
            # simple integer ratio resample via numpy (good enough for testing)
            ratio = 16000 / sr
            new_len = int(len(samples) * ratio)
            samples = np.interp(
                np.linspace(0, len(samples) - 1, new_len),
                np.arange(len(samples)),
                samples,
            ).astype(np.float32)

    return samples.astype(np.float32), 16000


def samples_to_wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    """Convert a float32 numpy array to WAV bytes."""
    buf = io.BytesIO()
    sf.write(buf, samples, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()
```

- [ ] **Step 4: Install dependencies and run tests**

```bash
pip install soundfile pydub numpy
pytest tests/test_audio_utils.py -v
```
Expected: all 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add audio_utils.py tests/test_audio_utils.py
git commit -m "feat: audio_utils — WAV/webm to 16kHz mono PCM, 30s limit"
```

---

## Task 4: STT Module (TDD — Small tier first)

**Files:**
- Create: `models/stt.py`
- Create: `tests/test_stt.py`

The STT module exposes a single class `STTModel` with a `transcribe(audio_bytes, lang) -> (text, is_fallback)` method. Small tier uses faster-whisper. Medium uses faster-whisper large-v3-turbo. High uses Voxtral-Mini (with faster-whisper fallback for VI).

- [ ] **Step 1: Write failing tests for Small tier STT interface**

```python
# tests/test_stt.py
import numpy as np
import io
import soundfile as sf
import pytest


def make_silent_wav(duration_s: float = 1.0) -> bytes:
    samples = np.zeros(int(16000 * duration_s), dtype=np.float32)
    buf = io.BytesIO()
    sf.write(buf, samples, 16000, format="WAV")
    return buf.getvalue()


def test_stt_small_transcribe_returns_str(stt_model):
    wav = make_silent_wav()
    text, is_fallback = stt_model.transcribe(wav, lang="en")
    assert isinstance(text, str)
    assert isinstance(is_fallback, bool)


def test_stt_small_no_fallback_for_supported_lang(stt_model, config):
    wav = make_silent_wav()
    _, is_fallback = stt_model.transcribe(wav, lang="en")
    if config == "small":
        assert is_fallback is False


def test_stt_high_vi_triggers_fallback(stt_model, config):
    if config != "high":
        pytest.skip("Only relevant for high tier")
    wav = make_silent_wav()
    _, is_fallback = stt_model.transcribe(wav, lang="vi")
    assert is_fallback is True


def test_stt_rejects_audio_over_30s(stt_model):
    from audio_utils import AudioTooLongError
    samples = np.zeros(int(16000 * 31), dtype=np.float32)
    buf = io.BytesIO()
    sf.write(buf, samples, 16000, format="WAV")
    with pytest.raises(AudioTooLongError):
        stt_model.transcribe(buf.getvalue(), lang="en")
```

- [ ] **Step 2: Run — verify they fail**

```bash
pytest tests/test_stt.py -v
```
Expected: FAIL — `stt_model` fixture not found (conftest not written yet; this is expected, proceed)

- [ ] **Step 3: Implement models/stt.py**

```python
# models/stt.py
from __future__ import annotations
import logging
import numpy as np
from audio_utils import to_pcm16k, AudioTooLongError
from config import VOXTRAL_LANGS, TIER_CONFIGS

logger = logging.getLogger(__name__)


class STTModel:
    def __init__(self, config: str):
        self.config = config
        self._primary: object = None
        self._fallback: object = None   # faster-whisper, used for VI in high tier
        self._loaded = False

    def load(self):
        if self._loaded:
            return
        cfg = TIER_CONFIGS[self.config]
        if self.config == "small":
            self._primary = self._load_faster_whisper(cfg["stt_model"])
        elif self.config == "medium":
            self._primary = self._load_faster_whisper(cfg["stt_model"])
        elif self.config == "high":
            self._primary = self._load_voxtral(cfg["stt_model"])
            self._fallback = self._load_faster_whisper(cfg["stt_fallback_model"])
        self._loaded = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transcribe(self, audio_bytes: bytes, lang: str) -> tuple[str, bool]:
        """
        Returns (transcript_text, is_fallback).
        is_fallback=True when high-tier falls back to faster-whisper for VI.
        """
        self.load()
        arr, _sr = to_pcm16k(audio_bytes)  # raises AudioTooLongError if > 30s

        needs_fallback = self.config == "high" and lang not in VOXTRAL_LANGS
        if needs_fallback:
            text = self._transcribe_faster_whisper(self._fallback, arr, lang)
            return text, True

        if self.config in ("small", "medium"):
            text = self._transcribe_faster_whisper(self._primary, arr, lang)
            return text, False

        # high tier, supported language
        text = self._transcribe_voxtral(self._primary, arr, lang)
        return text, False

    def is_loaded(self) -> bool:
        return self._loaded

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    def _load_faster_whisper(self, model_size: str):
        from faster_whisper import WhisperModel
        device = "cpu"
        compute = "int8"
        if self.config == "high":
            device = "auto"
            compute = "float16"
        logger.info("Loading faster-whisper %s (device=%s)", model_size, device)
        return WhisperModel(model_size, device=device, compute_type=compute)

    def _load_voxtral(self, model_id: str):
        # Voxtral-Mini-4B-Realtime-2602 uses the transformers ASR pipeline.
        # Verify exact pipeline class against model card if issues arise.
        from transformers import pipeline
        import torch
        device = 0 if torch.cuda.is_available() else -1
        logger.info("Loading Voxtral STT %s", model_id)
        return pipeline(
            "automatic-speech-recognition",
            model=model_id,
            device=device,
            torch_dtype=torch.float16,
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _transcribe_faster_whisper(self, model, arr: np.ndarray, lang: str) -> str:
        segments, _info = model.transcribe(arr, language=lang, beam_size=5)
        return " ".join(s.text.strip() for s in segments).strip()

    def _transcribe_voxtral(self, pipe, arr: np.ndarray, lang: str) -> str:
        result = pipe({"array": arr, "sampling_rate": 16000}, generate_kwargs={"language": lang})
        return result["text"].strip()
```

- [ ] **Step 4: Commit**

```bash
git add models/stt.py tests/test_stt.py
git commit -m "feat: STT module — faster-whisper (small/medium) and Voxtral-Mini (high) with VI fallback"
```

---

## Task 5: Translation Module (TDD — Small tier first)

**Files:**
- Create: `models/translation.py`
- Create: `tests/test_translation.py`

Small tier uses Helsinki-NLP opus-mt. Non-EN pairs pivot through English (e.g. fr→zh = fr→en→zh). Medium and High use HY-MT1.5 GGUF via llama-cpp-python.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_translation.py
import pytest


def test_translate_returns_string(translation_model):
    result = translation_model.translate("Hello world", source="en", target="fr")
    assert isinstance(result, str)
    assert len(result) > 0


def test_translate_en_to_fr(translation_model):
    result = translation_model.translate("Hello", source="en", target="fr")
    # "Bonjour" or similar — just check non-empty non-identical
    assert result.strip() != ""
    assert result.strip().lower() != "hello"


def test_translate_same_lang_returns_input(translation_model):
    result = translation_model.translate("Bonjour", source="fr", target="fr")
    assert result == "Bonjour"


def test_translate_pivot_non_en_pair(translation_model, config):
    """zh→fr must work even if no direct model exists (pivot through EN)."""
    if config != "small":
        pytest.skip("Pivot only needed for small tier opus-mt")
    result = translation_model.translate("你好", source="zh", target="fr")
    assert isinstance(result, str)
    assert len(result) > 0
```

- [ ] **Step 2: Run — verify they fail**

```bash
pytest tests/test_translation.py -v
```
Expected: FAIL — `translation_model` fixture not found

- [ ] **Step 3: Implement models/translation.py**

```python
# models/translation.py
from __future__ import annotations
import logging
from config import OPUS_MT_MODELS, TIER_CONFIGS, MODELS_CACHE_DIR
import os

logger = logging.getLogger(__name__)


class TranslationModel:
    def __init__(self, config: str):
        self.config = config
        self._models: dict = {}   # opus-mt: (src,tgt)->pipeline; gguf: single Llama
        self._loaded = False

    def load(self):
        if self._loaded:
            return
        if self.config == "small":
            # Lazy: models loaded on first use per pair
            pass
        else:
            self._load_gguf()
        self._loaded = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def translate(self, text: str, source: str, target: str) -> str:
        self.load()
        if source == target:
            return text
        if self.config == "small":
            return self._translate_opusmt(text, source, target)
        return self._translate_gguf(text, source, target)

    def is_loaded(self) -> bool:
        return self._loaded

    # ------------------------------------------------------------------
    # Small tier — Helsinki-NLP opus-mt
    # ------------------------------------------------------------------

    def _get_opusmt_pipeline(self, src: str, tgt: str):
        key = (src, tgt)
        if key not in self._models:
            model_id = OPUS_MT_MODELS.get(key)
            if model_id is None:
                raise ValueError(f"No opus-mt model for {src}→{tgt}")
            from transformers import pipeline
            logger.info("Loading opus-mt %s→%s (%s)", src, tgt, model_id)
            self._models[key] = pipeline("translation", model=model_id)
        return self._models[key]

    def _translate_opusmt(self, text: str, source: str, target: str) -> str:
        direct_key = (source, target)
        if direct_key in OPUS_MT_MODELS:
            pipe = self._get_opusmt_pipeline(source, target)
            return pipe(text, max_length=512)[0]["translation_text"]

        # Pivot through English
        if source != "en":
            pipe_to_en = self._get_opusmt_pipeline(source, "en")
            text = pipe_to_en(text, max_length=512)[0]["translation_text"]
        if target != "en":
            pipe_from_en = self._get_opusmt_pipeline("en", target)
            text = pipe_from_en(text, max_length=512)[0]["translation_text"]
        return text

    # ------------------------------------------------------------------
    # Medium / High tier — HY-MT1.5 GGUF via llama-cpp-python
    # ------------------------------------------------------------------

    def _load_gguf(self):
        from llama_cpp import Llama
        from huggingface_hub import hf_hub_download
        cfg = TIER_CONFIGS[self.config]
        model_path = hf_hub_download(
            repo_id=cfg["translation_model_repo"],
            filename=cfg["translation_model_file"],
            cache_dir=MODELS_CACHE_DIR,
        )
        logger.info("Loading HY-MT1.5 GGUF from %s", model_path)
        # n_gpu_layers=-1 offloads all layers to GPU (High); 0 = CPU (Medium)
        n_gpu = -1 if self.config == "high" else 0
        self._models["gguf"] = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_gpu_layers=n_gpu,
            verbose=False,
        )

    # HY-MT1.5 prompt format (ChatML; verify against model card if output is wrong)
    _LANG_NAMES = {
        "en": "English", "zh": "Chinese", "fr": "French", "de": "German",
        "es": "Spanish", "it": "Italian", "pt": "Portuguese", "nl": "Dutch",
        "hi": "Hindi", "ar": "Arabic", "vi": "Vietnamese",
    }

    def _translate_gguf(self, text: str, source: str, target: str) -> str:
        src_name = self._LANG_NAMES.get(source, source)
        tgt_name = self._LANG_NAMES.get(target, target)
        prompt = (
            f"<|im_start|>system\nYou are a professional translation engine.<|im_end|>\n"
            f"<|im_start|>user\n"
            f"Translate the following {src_name} text into {tgt_name}. "
            f"Output only the translation, nothing else:\n{text}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        llm = self._models["gguf"]
        output = llm(prompt, max_tokens=512, stop=["<|im_end|>"], echo=False)
        return output["choices"][0]["text"].strip()
```

- [ ] **Step 4: Commit**

```bash
git add models/translation.py tests/test_translation.py
git commit -m "feat: translation module — opus-mt (small) + HY-MT1.5 GGUF (medium/high)"
```

---

## Task 6: TTS Module (TDD — Small tier first)

**Files:**
- Create: `models/tts.py`
- Create: `tests/test_tts.py`

Small tier uses Piper via sherpa-onnx for all languages. Medium uses Kokoro-82M for EN/ZH/FR/ES/IT/PT/HI, Piper for the rest. High uses Voxtral-TTS via vLLM HTTP, Piper for VI.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_tts.py
import pytest
import io
import soundfile as sf


def test_tts_returns_wav_bytes(tts_model):
    audio_bytes, is_fallback = tts_model.synthesize("Hello world", lang="en")
    assert isinstance(audio_bytes, bytes)
    assert len(audio_bytes) > 0
    assert isinstance(is_fallback, bool)


def test_tts_output_is_valid_wav(tts_model):
    audio_bytes, _ = tts_model.synthesize("Hello world", lang="en")
    arr, sr = sf.read(io.BytesIO(audio_bytes))
    assert sr > 0
    assert len(arr) > 0


def test_tts_output_has_minimum_duration(tts_model):
    audio_bytes, _ = tts_model.synthesize("Hello world", lang="en")
    arr, sr = sf.read(io.BytesIO(audio_bytes))
    duration = len(arr) / sr
    assert duration > 0.3


def test_tts_vi_no_fallback_in_small(tts_model, config):
    if config != "small":
        pytest.skip("Only for small tier")
    audio_bytes, is_fallback = tts_model.synthesize("Xin chào", lang="vi")
    assert isinstance(audio_bytes, bytes)
    assert is_fallback is False


def test_tts_high_vi_triggers_fallback(tts_model, config):
    if config != "high":
        pytest.skip("Only for high tier")
    audio_bytes, is_fallback = tts_model.synthesize("Xin chào", lang="vi")
    assert isinstance(audio_bytes, bytes)
    assert is_fallback is True
```

- [ ] **Step 2: Run — verify they fail**

```bash
pytest tests/test_tts.py -v
```
Expected: FAIL — `tts_model` fixture not found

- [ ] **Step 3: Implement models/tts.py**

```python
# models/tts.py
from __future__ import annotations
import io
import logging
import os
import tarfile
import urllib.request
import numpy as np
import soundfile as sf
from config import (
    VOXTRAL_LANGS, VOXTRAL_TTS_VOICES, KOKORO_VOICES,
    PIPER_MODELS, TIER_CONFIGS, MODELS_CACHE_DIR,
)

logger = logging.getLogger(__name__)


class TTSModel:
    def __init__(self, config: str):
        self.config = config
        self._kokoro = None
        self._piper: dict[str, object] = {}   # lang -> sherpa_onnx.OfflineTts
        self._loaded = False

    def load(self):
        if self._loaded:
            return
        if self.config == "medium":
            self._load_kokoro()
        # Piper loaded lazily per language on first use
        self._loaded = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def synthesize(self, text: str, lang: str) -> tuple[bytes, bool]:
        """Returns (wav_bytes, is_fallback)."""
        self.load()

        if self.config == "high":
            if lang not in VOXTRAL_LANGS:
                wav = self._synthesize_piper(text, lang)
                return wav, True
            return self._synthesize_vllm(text, lang), False

        if self.config == "medium":
            voice = KOKORO_VOICES.get(lang)
            if voice:
                return self._synthesize_kokoro(text, lang, voice), False
            return self._synthesize_piper(text, lang), False

        # small — always piper
        return self._synthesize_piper(text, lang), False

    def is_loaded(self) -> bool:
        return self._loaded

    # ------------------------------------------------------------------
    # Kokoro-82M (Medium tier)
    # ------------------------------------------------------------------

    def _load_kokoro(self):
        from kokoro_onnx import Kokoro
        from huggingface_hub import snapshot_download
        model_dir = snapshot_download(
            repo_id="NeuML/kokoro-int8-onnx",
            cache_dir=MODELS_CACHE_DIR,
        )
        onnx_path = os.path.join(model_dir, "kokoro-int8.onnx")
        voices_path = os.path.join(model_dir, "voices.bin")
        logger.info("Loading Kokoro-82M from %s", model_dir)
        self._kokoro = Kokoro(onnx_path, voices_path)

    def _synthesize_kokoro(self, text: str, lang: str, voice: str) -> bytes:
        # Map lang code to Kokoro lang string
        lang_map = {
            "en": "en-us", "zh": "zh", "fr": "fr-fr",
            "es": "es", "it": "it", "pt": "pt-br",
            "hi": "hi", "ar": "ar",
        }
        kokoro_lang = lang_map.get(lang, "en-us")
        samples, sr = self._kokoro.create(text, voice=voice, speed=1.0, lang=kokoro_lang)
        return self._to_wav_bytes(np.array(samples), sr)

    # ------------------------------------------------------------------
    # Piper via sherpa-onnx (Small tier + VI fallback)
    # ------------------------------------------------------------------

    def _get_piper(self, lang: str):
        if lang not in self._piper:
            self._piper[lang] = self._load_piper(lang)
        return self._piper[lang]

    def _load_piper(self, lang: str):
        import sherpa_onnx
        info = PIPER_MODELS.get(lang)
        if info is None:
            raise ValueError(f"No Piper model configured for language '{lang}'")

        model_dir = self._download_piper_model(lang, info)
        model_path = os.path.join(model_dir, info["model"])
        tokens_path = os.path.join(model_dir, info["tokens"])
        # espeak-ng data is bundled with sherpa-onnx
        data_dir = sherpa_onnx.get_default_data_dir()

        config = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                    model=model_path,
                    lexicon="",
                    tokens=tokens_path,
                    data_dir=data_dir,
                ),
                provider="cpu",
                num_threads=2,
            ),
        )
        logger.info("Loading Piper TTS for lang=%s", lang)
        return sherpa_onnx.OfflineTts(config)

    def _download_piper_model(self, lang: str, info: dict) -> str:
        cache_dir = os.path.join(MODELS_CACHE_DIR, "piper")
        os.makedirs(cache_dir, exist_ok=True)
        model_dir = os.path.join(cache_dir, info["dir"])
        if os.path.isdir(model_dir):
            return model_dir
        archive_path = os.path.join(cache_dir, info["dir"] + ".tar.bz2")
        if not os.path.exists(archive_path):
            logger.info("Downloading Piper model for %s from %s", lang, info["url"])
            urllib.request.urlretrieve(info["url"], archive_path)
        with tarfile.open(archive_path, "r:bz2") as tar:
            tar.extractall(cache_dir)
        return model_dir

    def _synthesize_piper(self, text: str, lang: str) -> bytes:
        tts = self._get_piper(lang)
        audio = tts.generate(text, sid=0, speed=1.0)
        return self._to_wav_bytes(np.array(audio.samples), audio.sample_rate)

    # ------------------------------------------------------------------
    # Voxtral-TTS via vLLM HTTP (High tier)
    # ------------------------------------------------------------------

    def _synthesize_vllm(self, text: str, lang: str) -> bytes:
        import httpx
        cfg = TIER_CONFIGS["high"]
        voice = VOXTRAL_TTS_VOICES.get(lang, "casual_male")
        payload = {
            "input": text,
            "model": cfg["tts_vllm_model"],
            "response_format": "wav",
            "voice": voice,
        }
        resp = httpx.post(
            f"{cfg['tts_vllm_url']}/v1/audio/speech",
            json=payload,
            timeout=120.0,
        )
        resp.raise_for_status()
        return resp.content

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _to_wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
        buf = io.BytesIO()
        sf.write(buf, samples, sample_rate, format="WAV", subtype="PCM_16")
        return buf.getvalue()
```

- [ ] **Step 4: Commit**

```bash
git add models/tts.py tests/test_tts.py
git commit -m "feat: TTS module — Piper (small/VI fallback), Kokoro-82M (medium), Voxtral-TTS (high)"
```

---

## Task 7: FastAPI App

**Files:**
- Create: `app.py`

- [ ] **Step 1: Write app.py**

```python
# app.py
import argparse
import base64
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from audio_utils import AudioTooLongError
from config import LANGUAGES, VOXTRAL_LANGS
from models.stt import STTModel
from models.translation import TranslationModel
from models.tts import TTSModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--config", choices=["high", "medium", "small"], default="small")
parser.add_argument("--host", default="0.0.0.0")
parser.add_argument("--port", type=int, default=8080)
args, _ = parser.parse_known_args()

CURRENT_CONFIG = {"value": args.config}

# -----------------------------------------------------------------------
# Model singletons
# -----------------------------------------------------------------------
stt = STTModel(args.config)
translation = TranslationModel(args.config)
tts = TTSModel(args.config)


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    # nothing to clean up — models are cleaned up by GC


app = FastAPI(lifespan=lifespan)

# -----------------------------------------------------------------------
# Request / Response schemas
# -----------------------------------------------------------------------

class TranscribeRequest(BaseModel):
    audio: str    # base64-encoded WAV or webm
    lang: str


class TranslateRequest(BaseModel):
    text: str
    source: str
    target: str


class SynthesizeRequest(BaseModel):
    text: str
    lang: str


class ConfigRequest(BaseModel):
    config: str


# -----------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------

@app.get("/languages")
def get_languages():
    result = []
    for lang in LANGUAGES:
        tiers = ["small", "medium"]
        if lang["voxtral_stt"] and lang["voxtral_tts"]:
            tiers.append("high")
        result.append({
            "code": lang["code"],
            "name": lang["name"],
            "flag": lang["flag"],
            "tiers": tiers,
            "fallback_in_high": not (lang["voxtral_stt"] and lang["voxtral_tts"]),
        })
    return {"languages": result}


@app.get("/status")
def get_status():
    fallback_active = []
    if CURRENT_CONFIG["value"] == "high":
        fallback_active = [
            f"{l['code']}_stt" for l in LANGUAGES if not l["voxtral_stt"]
        ] + [
            f"{l['code']}_tts" for l in LANGUAGES if not l["voxtral_tts"]
        ]
    return {
        "config": CURRENT_CONFIG["value"],
        "models_loaded": {
            "stt": stt.is_loaded(),
            "translation": translation.is_loaded(),
            "tts": tts.is_loaded(),
        },
        "fallback_active": fallback_active,
    }


@app.get("/health")
def health():
    return {
        "stt": stt.is_loaded(),
        "translation": translation.is_loaded(),
        "tts": tts.is_loaded(),
    }


@app.post("/transcribe")
def transcribe(req: TranscribeRequest):
    try:
        audio_bytes = base64.b64decode(req.audio)
        text, is_fallback = stt.transcribe(audio_bytes, req.lang)
        return {"text": text, "fallback": is_fallback}
    except AudioTooLongError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Transcription error")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/translate")
def translate(req: TranslateRequest):
    try:
        result = translation.translate(req.text, source=req.source, target=req.target)
        return {"translation": result}
    except Exception as e:
        logger.exception("Translation error")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/synthesize")
def synthesize(req: SynthesizeRequest):
    try:
        wav_bytes, is_fallback = tts.synthesize(req.text, req.lang)
        audio_b64 = base64.b64encode(wav_bytes).decode()
        return {"audio": audio_b64, "fallback": is_fallback}
    except Exception as e:
        logger.exception("TTS error")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/config")
def set_config(req: ConfigRequest):
    global stt, translation, tts
    if req.config not in ("high", "medium", "small"):
        raise HTTPException(status_code=400, detail="Invalid config")
    CURRENT_CONFIG["value"] = req.config
    stt = STTModel(req.config)
    translation = TranslationModel(req.config)
    tts = TTSModel(req.config)
    return {"config": req.config, "status": "reloading"}


@app.get("/")
def index():
    return FileResponse("frontend/index.html")


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)
```

- [ ] **Step 2: Verify the app starts (Small tier, no models downloaded yet)**

```bash
python app.py --config small --port 8080
```
Expected: `INFO: Application startup complete.` with no errors.
Press Ctrl+C to stop.

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat: FastAPI app with transcribe/translate/synthesize/config routes"
```

---

## Task 8: Test Fixture Downloader

**Files:**
- Create: `tests/download_fixtures.py`

- [ ] **Step 1: Write download_fixtures.py**

```python
# tests/download_fixtures.py
"""
Downloads short audio clips from FLEURS (Apache 2.0 / CC-BY-4.0) for E2E tests.
Run once: python tests/download_fixtures.py
Clips are cached in tests/fixtures/audio/.
"""
import io
import os
import soundfile as sf
import numpy as np

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "audio")

# FLEURS config names and expected ground-truth text fragments
CLIPS = [
    {
        "lang": "en",
        "fleurs_config": "en_us",
        "filename": "en_sample.wav",
        # A known sentence from FLEURS English test set index 0
        "expected_words": ["the", "a"],   # partial match — lowercase words expected in transcript
    },
    {
        "lang": "zh",
        "fleurs_config": "cmn_hans_cn",
        "filename": "zh_sample.wav",
        "expected_words": ["的", "了"],   # common Chinese particles
    },
    {
        "lang": "fr",
        "fleurs_config": "fr_fr",
        "filename": "fr_sample.wav",
        "expected_words": ["le", "la", "de"],
    },
    {
        "lang": "vi",
        "fleurs_config": "vi_vn",
        "filename": "vi_sample.wav",
        "expected_words": ["và", "của", "là"],
    },
]


def download_all():
    os.makedirs(FIXTURES_DIR, exist_ok=True)
    try:
        from datasets import load_dataset
    except ImportError:
        print("Install: pip install datasets")
        return

    for clip in CLIPS:
        out_path = os.path.join(FIXTURES_DIR, clip["filename"])
        if os.path.exists(out_path):
            print(f"  already exists: {clip['filename']}")
            continue
        print(f"  downloading {clip['lang']} ({clip['fleurs_config']})...")
        ds = load_dataset(
            "google/fleurs",
            clip["fleurs_config"],
            split="test",
            streaming=True,
            trust_remote_code=True,
        )
        sample = next(iter(ds))
        arr = np.array(sample["audio"]["array"], dtype=np.float32)
        sr = sample["audio"]["sampling_rate"]
        # trim to 8 seconds max
        arr = arr[: sr * 8]
        sf.write(out_path, arr, sr, format="WAV")
        print(f"  saved {clip['filename']} ({len(arr)/sr:.1f}s)")


def get_fixture_path(lang: str) -> str:
    for clip in CLIPS:
        if clip["lang"] == lang:
            path = os.path.join(FIXTURES_DIR, clip["filename"])
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Fixture for '{lang}' not found. Run: python tests/download_fixtures.py"
                )
            return path
    raise ValueError(f"No fixture configured for lang='{lang}'")


def get_expected_words(lang: str) -> list[str]:
    for clip in CLIPS:
        if clip["lang"] == lang:
            return clip["expected_words"]
    return []


if __name__ == "__main__":
    print("Downloading FLEURS audio fixtures...")
    download_all()
    print("Done.")
```

- [ ] **Step 2: Download fixtures**

```bash
pip install datasets
python tests/download_fixtures.py
```
Expected output:
```
Downloading FLEURS audio fixtures...
  downloading en (en_us)...
  saved en_sample.wav (8.0s)
  downloading zh (cmn_hans_cn)...
  ...
Done.
```

- [ ] **Step 3: Commit**

```bash
git add tests/download_fixtures.py
git commit -m "test: fixture downloader (FLEURS) with --config flag support"
```

---

## Task 9: E2E Tests

**Files:**
- Create: `tests/test_e2e.py`

- [ ] **Step 1: Write test_e2e.py**

```python
# tests/test_e2e.py
"""
End-to-end tests: audio clip → STT → translate → TTS.
Run with Small models by default (fast, no GPU):
    pytest tests/test_e2e.py
Run with Medium:
    pytest tests/test_e2e.py --config=medium
Run with High (GPU + vLLM must be running):
    pytest tests/test_e2e.py --config=high
"""
import io
import pytest
import soundfile as sf
from tests.download_fixtures import get_fixture_path, get_expected_words


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def load_wav_bytes(lang: str) -> bytes:
    path = get_fixture_path(lang)
    with open(path, "rb") as f:
        return f.read()


# -----------------------------------------------------------------------
# STT tests
# -----------------------------------------------------------------------

@pytest.mark.parametrize("lang", ["en", "fr"])
def test_transcribe_latin_lang(stt_model, lang):
    wav = load_wav_bytes(lang)
    text, is_fallback = stt_model.transcribe(wav, lang=lang)
    assert isinstance(text, str) and len(text) > 0
    words = get_expected_words(lang)
    assert any(w in text.lower() for w in words), f"None of {words} found in: '{text}'"


def test_transcribe_zh(stt_model):
    wav = load_wav_bytes("zh")
    text, _ = stt_model.transcribe(wav, lang="zh")
    assert isinstance(text, str) and len(text) > 0
    # Chinese output should contain Chinese characters
    assert any("\u4e00" <= c <= "\u9fff" for c in text), f"No CJK chars in: '{text}'"


def test_transcribe_vi(stt_model, config):
    wav = load_wav_bytes("vi")
    text, is_fallback = stt_model.transcribe(wav, lang="vi")
    assert isinstance(text, str) and len(text) > 0
    if config == "high":
        assert is_fallback is True


# -----------------------------------------------------------------------
# Translation tests
# -----------------------------------------------------------------------

def test_translate_en_to_fr(translation_model):
    result = translation_model.translate("Good morning", source="en", target="fr")
    assert len(result) > 0
    assert result.lower() != "good morning"


def test_translate_fr_to_en(translation_model):
    result = translation_model.translate("Bonjour le monde", source="fr", target="en")
    assert "hello" in result.lower() or "good" in result.lower() or "world" in result.lower()


def test_translate_en_to_zh(translation_model):
    result = translation_model.translate("Hello", source="en", target="zh")
    assert any("\u4e00" <= c <= "\u9fff" for c in result), f"No CJK in: '{result}'"


def test_translate_en_to_vi(translation_model):
    result = translation_model.translate("Thank you", source="en", target="vi")
    assert len(result) > 0
    assert result.lower() != "thank you"


def test_translate_zh_to_fr(translation_model, config):
    """Non-EN pair — pivots through EN in small tier."""
    result = translation_model.translate("你好", source="zh", target="fr")
    assert len(result) > 0


# -----------------------------------------------------------------------
# TTS tests
# -----------------------------------------------------------------------

@pytest.mark.parametrize("lang,text", [
    ("en", "Hello world"),
    ("fr", "Bonjour le monde"),
    ("zh", "你好世界"),
    ("vi", "Xin chào thế giới"),
])
def test_synthesize_produces_audio(tts_model, lang, text):
    wav_bytes, is_fallback = tts_model.synthesize(text, lang=lang)
    assert isinstance(wav_bytes, bytes) and len(wav_bytes) > 100
    arr, sr = sf.read(io.BytesIO(wav_bytes))
    duration = len(arr) / sr
    assert duration > 0.3, f"Audio too short: {duration:.2f}s"


def test_synthesize_vi_fallback_in_high(tts_model, config):
    if config != "high":
        pytest.skip("Only relevant for high tier")
    _, is_fallback = tts_model.synthesize("Xin chào", lang="vi")
    assert is_fallback is True


def test_synthesize_vi_no_fallback_in_small(tts_model, config):
    if config != "small":
        pytest.skip("Only relevant for small tier")
    _, is_fallback = tts_model.synthesize("Xin chào", lang="vi")
    assert is_fallback is False


# -----------------------------------------------------------------------
# Full pipeline tests
# -----------------------------------------------------------------------

def test_full_pipeline_fr_to_zh(stt_model, translation_model, tts_model):
    """French audio → transcript → Chinese → synthesized Chinese audio."""
    wav = load_wav_bytes("fr")
    text, _ = stt_model.transcribe(wav, lang="fr")
    assert len(text) > 0

    translation = translation_model.translate(text, source="fr", target="zh")
    assert len(translation) > 0

    audio_bytes, _ = tts_model.synthesize(translation, lang="zh")
    arr, sr = sf.read(io.BytesIO(audio_bytes))
    assert len(arr) / sr > 0.3


def test_full_pipeline_en_to_vi(stt_model, translation_model, tts_model, config):
    """English audio → transcript → Vietnamese → synthesized Vietnamese audio.
    In high tier, both STT and TTS for VI must trigger fallback=True."""
    wav = load_wav_bytes("en")
    text, _ = stt_model.transcribe(wav, lang="en")
    assert len(text) > 0

    vi_text = translation_model.translate(text, source="en", target="vi")
    assert len(vi_text) > 0

    audio_bytes, is_fallback = tts_model.synthesize(vi_text, lang="vi")
    arr, sr = sf.read(io.BytesIO(audio_bytes))
    assert len(arr) / sr > 0.3

    if config == "high":
        assert is_fallback is True


def test_fallback_indicator_in_status(config):
    """GET /status returns fallback_active list for VI when config=high."""
    import httpx
    # This test requires the app to be running. Skip if not.
    try:
        resp = httpx.get("http://localhost:8080/status", timeout=2.0)
    except Exception:
        pytest.skip("App not running — start with: python app.py --config high")

    data = resp.json()
    assert data["config"] == config
    if config == "high":
        assert "vi_stt" in data["fallback_active"]
        assert "vi_tts" in data["fallback_active"]
```

- [ ] **Step 2: Run E2E tests (Small tier — downloads models on first run)**

```bash
pytest tests/test_e2e.py --config=small -v
```
Expected: all tests PASS (first run will download ~500MB faster-whisper small + opus-mt models)

- [ ] **Step 3: Commit**

```bash
git add tests/test_e2e.py
git commit -m "test: E2E tests — full pipeline for all tiers, fallback assertions"
```

---

## Task 10: Frontend

**Files:**
- Create: `frontend/index.html`

- [ ] **Step 1: Write frontend/index.html**

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Real-Time Translator</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: system-ui, sans-serif; background: #0f1117; color: #e2e8f0; min-height: 100vh; display: flex; flex-direction: column; }

  header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 1rem 1.5rem; border-bottom: 1px solid #2d3748;
  }
  header h1 { font-size: 1.1rem; font-weight: 600; }
  .tier-selector { display: flex; align-items: center; gap: 0.5rem; }
  .tier-selector select {
    background: #1a202c; border: 1px solid #4a5568; color: #e2e8f0;
    padding: 0.3rem 0.6rem; border-radius: 6px; font-size: 0.85rem; cursor: pointer;
  }

  main { flex: 1; display: flex; flex-direction: column; gap: 1.5rem; padding: 1.5rem; max-width: 800px; width: 100%; margin: 0 auto; }

  .lang-row { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
  .lang-col { display: flex; flex-direction: column; gap: 0.75rem; }
  .lang-col label { font-size: 0.8rem; color: #a0aec0; text-transform: uppercase; letter-spacing: 0.05em; }
  .lang-col select {
    background: #1a202c; border: 1px solid #4a5568; color: #e2e8f0;
    padding: 0.6rem 0.8rem; border-radius: 8px; font-size: 0.95rem; cursor: pointer;
  }

  .record-btn {
    background: #2d3748; border: 2px solid #4a5568; color: #e2e8f0;
    border-radius: 12px; padding: 2rem 1rem; font-size: 1rem; cursor: pointer;
    transition: all 0.15s; user-select: none; display: flex; flex-direction: column;
    align-items: center; gap: 0.5rem; -webkit-tap-highlight-color: transparent;
  }
  .record-btn:hover { border-color: #63b3ed; }
  .record-btn.recording { background: #742a2a; border-color: #fc8181; animation: pulse 1s infinite; }
  @keyframes pulse { 0%,100% { opacity:1; } 50% { opacity:0.7; } }
  .record-btn .mic-icon { font-size: 2rem; }
  .record-btn .btn-label { font-size: 0.85rem; color: #a0aec0; }

  .output-box {
    background: #1a202c; border: 1px solid #2d3748; border-radius: 10px;
    padding: 1rem 1.25rem; min-height: 5rem; position: relative;
  }
  .output-source { color: #a0aec0; font-size: 0.9rem; margin-bottom: 0.5rem; }
  .output-translation {
    color: #e2e8f0; font-size: 1rem; display: flex;
    align-items: flex-start; justify-content: space-between; gap: 1rem;
  }
  .replay-btn {
    background: none; border: none; color: #63b3ed; font-size: 1.1rem;
    cursor: pointer; flex-shrink: 0; padding: 0.2rem;
  }
  .replay-btn:disabled { color: #4a5568; cursor: default; }
  .spinner { display: inline-block; width: 18px; height: 18px; border: 2px solid #4a5568; border-top-color: #63b3ed; border-radius: 50%; animation: spin 0.7s linear infinite; }
  @keyframes spin { to { transform: rotate(360deg); } }

  .status-bar {
    font-size: 0.8rem; color: #718096; display: flex; align-items: center; gap: 0.5rem;
    padding: 0.5rem 0;
  }
  .fallback-badge {
    display: inline-flex; align-items: center; gap: 0.25rem;
    background: #2d3748; border: 1px solid #4a5568; border-radius: 4px;
    padding: 0.1rem 0.4rem; cursor: help; font-size: 0.78rem;
  }
</style>
</head>
<body>

<header>
  <h1>🌐 Real-Time Translator</h1>
  <div class="tier-selector">
    <span style="font-size:0.8rem;color:#a0aec0">⚙</span>
    <select id="tierSelect" onchange="setTier(this.value)">
      <option value="small">Small</option>
      <option value="medium">Medium</option>
      <option value="high">High</option>
    </select>
  </div>
</header>

<main>
  <div class="lang-row">
    <div class="lang-col">
      <label>Language A</label>
      <select id="langA"></select>
    </div>
    <div class="lang-col">
      <label>Language B</label>
      <select id="langB"></select>
    </div>
  </div>

  <div class="lang-row">
    <div class="lang-col">
      <button class="record-btn" id="btnA"
        onmousedown="startRecord('A')" onmouseup="stopRecord('A')"
        ontouchstart="startRecord('A')" ontouchend="stopRecord('A')">
        <span class="mic-icon">🎤</span>
        <span class="btn-label">Hold to speak</span>
      </button>
    </div>
    <div class="lang-col">
      <button class="record-btn" id="btnB"
        onmousedown="startRecord('B')" onmouseup="stopRecord('B')"
        ontouchstart="startRecord('B')" ontouchend="stopRecord('B')">
        <span class="mic-icon">🎤</span>
        <span class="btn-label">Hold to speak</span>
      </button>
    </div>
  </div>

  <div class="output-box">
    <div class="output-source" id="sourceText">—</div>
    <div class="output-translation">
      <span id="translationText">—</span>
      <button class="replay-btn" id="replayBtn" onclick="replayAudio()" disabled title="Replay">🔊</button>
    </div>
  </div>

  <div class="status-bar" id="statusBar">
    <span id="statusText">Ready</span>
  </div>
</main>

<audio id="audioPlayer" hidden></audio>

<script>
const API = '';   // same origin
let languages = [];
let currentConfig = 'small';
let recorder = null;
let audioChunks = [];
let lastAudioB64 = null;

// -----------------------------------------------------------------------
// Init
// -----------------------------------------------------------------------
async function init() {
  await loadStatus();
  await loadLanguages();
}

async function loadStatus() {
  const r = await fetch(`${API}/status`);
  const d = await r.json();
  currentConfig = d.config;
  document.getElementById('tierSelect').value = currentConfig;
  updateFallbackBadge(d.fallback_active || []);
}

async function loadLanguages() {
  const r = await fetch(`${API}/languages`);
  const d = await r.json();
  languages = d.languages;
  populateSelect('langA', 'en');
  populateSelect('langB', 'fr');
}

function populateSelect(id, defaultCode) {
  const sel = document.getElementById(id);
  sel.innerHTML = '';
  languages.forEach(l => {
    const opt = document.createElement('option');
    opt.value = l.code;
    opt.textContent = `${l.flag} ${l.name}`;
    if (l.code === defaultCode) opt.selected = true;
    sel.appendChild(opt);
  });
}

// -----------------------------------------------------------------------
// Tier switching
// -----------------------------------------------------------------------
async function setTier(value) {
  setStatus('Switching tier…');
  await fetch(`${API}/config`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({config: value}),
  });
  currentConfig = value;
  await pollUntilReady();
  await loadStatus();
  setStatus('Ready');
}

async function pollUntilReady() {
  for (let i = 0; i < 60; i++) {
    await new Promise(r => setTimeout(r, 1000));
    try {
      const r = await fetch(`${API}/health`);
      const d = await r.json();
      if (Object.values(d).every(Boolean)) return;
    } catch {}
  }
}

// -----------------------------------------------------------------------
// Recording
// -----------------------------------------------------------------------
async function startRecord(side) {
  if (recorder) return;
  const stream = await navigator.mediaDevices.getUserMedia({audio: true});
  const mimeType = MediaRecorder.isTypeSupported('audio/wav') ? 'audio/wav' : 'audio/webm';
  recorder = new MediaRecorder(stream, {mimeType});
  audioChunks = [];
  recorder.ondataavailable = e => audioChunks.push(e.data);
  recorder.start();

  const btn = document.getElementById('btn' + side);
  btn.classList.add('recording');
  btn.querySelector('.btn-label').textContent = 'Recording…';
  btn._side = side;
}

async function stopRecord(side) {
  if (!recorder) return;
  recorder.onstop = () => handleAudio(side);
  recorder.stop();
  recorder.stream.getTracks().forEach(t => t.stop());
  recorder = null;

  const btn = document.getElementById('btn' + side);
  btn.classList.remove('recording');
  btn.querySelector('.btn-label').textContent = 'Hold to speak';
}

async function handleAudio(side) {
  const blob = new Blob(audioChunks);
  const arrayBuf = await blob.arrayBuffer();
  const b64 = btoa(String.fromCharCode(...new Uint8Array(arrayBuf)));

  const srcLang = document.getElementById('lang' + side).value;
  const tgtLang = document.getElementById(side === 'A' ? 'langB' : 'langA').value;

  setStatus('<span class="spinner"></span> Transcribing…');
  document.getElementById('sourceText').textContent = '…';
  document.getElementById('translationText').textContent = '…';
  document.getElementById('replayBtn').disabled = true;

  try {
    // 1. Transcribe
    const tRes = await fetch(`${API}/transcribe`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({audio: b64, lang: srcLang}),
    }).then(r => r.json());

    document.getElementById('sourceText').textContent = tRes.text;
    let fallback = tRes.fallback;

    // 2. Translate
    setStatus('<span class="spinner"></span> Translating…');
    const trRes = await fetch(`${API}/translate`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({text: tRes.text, source: srcLang, target: tgtLang}),
    }).then(r => r.json());

    document.getElementById('translationText').textContent = trRes.translation;

    // 3. Synthesize
    setStatus('<span class="spinner"></span> Synthesizing…');
    const sRes = await fetch(`${API}/synthesize`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({text: trRes.translation, lang: tgtLang}),
    }).then(r => r.json());

    fallback = fallback || sRes.fallback;
    lastAudioB64 = sRes.audio;
    playAudio(sRes.audio);
    document.getElementById('replayBtn').disabled = false;

    setStatus(fallback ? '🔄 Fallback mode active' : 'Ready', fallback);
  } catch (e) {
    setStatus('⚠ Error: ' + e.message);
  }
}

// -----------------------------------------------------------------------
// Audio playback
// -----------------------------------------------------------------------
function playAudio(b64) {
  const bytes = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
  const blob = new Blob([bytes], {type: 'audio/wav'});
  const url = URL.createObjectURL(blob);
  const player = document.getElementById('audioPlayer');
  player.src = url;
  player.play();
}

function replayAudio() {
  if (lastAudioB64) playAudio(lastAudioB64);
}

// -----------------------------------------------------------------------
// Status bar
// -----------------------------------------------------------------------
function setStatus(html, isFallback = false) {
  const bar = document.getElementById('statusBar');
  let content = `<span id="statusText">${html}</span>`;
  if (isFallback) {
    content += ` <span class="fallback-badge" title="Vietnamese STT and TTS are handled by Medium-tier models in the High tier configuration.">🔄 Fallback</span>`;
  }
  bar.innerHTML = content;
}

function updateFallbackBadge(fallbackActive) {
  if (fallbackActive.length > 0) {
    setStatus('Ready', true);
  }
}

init();
</script>
</body>
</html>
```

- [ ] **Step 2: Verify in browser**

```bash
python app.py --config small
```
Open `http://localhost:8080` — confirm the UI loads, language dropdowns are populated, and the tier selector shows "Small".

- [ ] **Step 3: Commit**

```bash
git add frontend/index.html
git commit -m "feat: single-page frontend with hold-to-speak, fallback indicator, tier switcher"
```

---

## Task 11: Launch Script + GitHub Repo

**Files:**
- Create: `start.sh`

- [ ] **Step 1: Write start.sh**

```bash
#!/usr/bin/env bash
set -e

CONFIG=${1:-small}
HOST=${2:-0.0.0.0}
PORT=${3:-8080}

VLLM_PID=""

cleanup() {
  if [ -n "$VLLM_PID" ]; then
    echo "Stopping vLLM (pid $VLLM_PID)..."
    kill "$VLLM_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

if [ "$CONFIG" = "high" ]; then
  echo "Starting vLLM for Voxtral-4B-TTS-2603..."
  vllm serve mistralai/Voxtral-4B-TTS-2603 --omni --host 127.0.0.1 --port 8000 &
  VLLM_PID=$!

  echo "Waiting for vLLM to be ready..."
  for i in $(seq 1 60); do
    if curl -sf http://127.0.0.1:8000/health > /dev/null 2>&1; then
      echo "vLLM is ready."
      break
    fi
    sleep 2
  done
fi

echo "Starting FastAPI (config=$CONFIG, port=$PORT)..."
python app.py --config "$CONFIG" --host "$HOST" --port "$PORT"
```

- [ ] **Step 2: Make executable**

```bash
chmod +x start.sh
```

- [ ] **Step 3: Commit**

```bash
git add start.sh
git commit -m "feat: start.sh — launches vLLM (high tier) + FastAPI, cleans up on exit"
```

- [ ] **Step 4: Create GitHub repo and push**

```bash
gh repo create trad --public --description "Local real-time voice translator — Voxtral STT/TTS, HY-MT1.5 translation, no API tokens" --source . --push
```

Expected: repo created at `https://github.com/<your-username>/trad`

---

## Running the Project

```bash
# Small tier (phone / Termux)
pip install -r requirements/requirements-small.txt
python tests/download_fixtures.py   # one-time fixture download
pytest tests/ --config=small        # run tests
./start.sh small                     # start app → open http://localhost:8080

# Medium tier (CPU laptop)
pip install -r requirements/requirements-medium.txt
pytest tests/ --config=medium
./start.sh medium

# High tier (AMD GPU — install ROCm torch + llama-cpp first, see requirements-high.txt comments)
pip install -r requirements/requirements-high.txt
pytest tests/ --config=high         # requires vLLM running
./start.sh high
```
