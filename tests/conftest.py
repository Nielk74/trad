# tests/conftest.py
import io
import numpy as np
import pytest
import soundfile as sf
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


@pytest.fixture(scope="session")
def reference_audio_wav() -> bytes:
    """Synthetic 3-second 16kHz mono WAV — used as reference audio for voice cloning tests."""
    sr = 16000
    duration = 3
    t = np.linspace(0, duration, sr * duration, endpoint=False)
    # Simple voiced-speech-like signal: mix of harmonics
    signal = (
        0.4 * np.sin(2 * np.pi * 180 * t) +
        0.3 * np.sin(2 * np.pi * 360 * t) +
        0.2 * np.sin(2 * np.pi * 540 * t) +
        0.1 * np.random.default_rng(42).standard_normal(len(t)) * 0.05
    ).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, signal, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()
