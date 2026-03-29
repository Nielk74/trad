# tests/test_tts.py
import pytest
import io
import soundfile as sf
from config import VOICE_CLONE_CONFIGS


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


# -----------------------------------------------------------------------
# Voice cloning tests
# -----------------------------------------------------------------------

def test_voice_clone_worker_venv_exists(config):
    """The isolated clone worker venv must exist before voice cloning can work."""
    import os
    venv_tier = "small" if config == "small" else "medium"
    python = os.path.join("venvs", f"clone_{venv_tier}", "bin", "python")
    if not os.path.exists(python):
        pytest.skip(
            f"Clone worker venv not set up for tier={config}. "
            f"Run: python workers/setup_clone_venvs.py --tier {venv_tier}"
        )


def test_voice_clone_supported_langs_configured(config):
    """Every tier has a non-empty supported_langs set in VOICE_CLONE_CONFIGS."""
    cfg = VOICE_CLONE_CONFIGS.get(config)
    assert cfg is not None, f"No voice clone config for tier={config}"
    assert len(cfg["supported_langs"]) > 0


def test_voice_clone_returns_valid_wav(tts_model, config, reference_audio_wav):
    """synthesize() with reference_audio + reference_text returns valid WAV."""
    supported = VOICE_CLONE_CONFIGS[config]["supported_langs"]
    if "en" not in supported:
        pytest.skip(f"English not supported for cloning on tier={config}")
    audio_bytes, is_fallback = tts_model.synthesize(
        "Hello world",
        lang="en",
        reference_audio=reference_audio_wav,
        reference_text="Hello world",
    )
    assert isinstance(audio_bytes, bytes) and len(audio_bytes) > 100
    arr, sr = sf.read(io.BytesIO(audio_bytes))
    assert len(arr) / sr > 0.3, f"Cloned audio too short: {len(arr)/sr:.2f}s"
    assert is_fallback is False


def test_voice_clone_without_ref_text(tts_model, config, reference_audio_wav):
    """synthesize() with reference_audio but no ref_text falls back to x_vector mode and still returns audio."""
    supported = VOICE_CLONE_CONFIGS[config]["supported_langs"]
    if "en" not in supported:
        pytest.skip(f"English not supported for cloning on tier={config}")
    audio_bytes, _ = tts_model.synthesize(
        "Hello world",
        lang="en",
        reference_audio=reference_audio_wav,
        reference_text=None,
    )
    assert isinstance(audio_bytes, bytes) and len(audio_bytes) > 100
    arr, sr = sf.read(io.BytesIO(audio_bytes))
    assert len(arr) / sr > 0.3


def test_voice_clone_unsupported_lang_falls_back_to_standard(tts_model, config, reference_audio_wav):
    """synthesize() with reference_audio for an unsupported clone language returns audio anyway (standard TTS)."""
    supported = VOICE_CLONE_CONFIGS[config]["supported_langs"]
    # Vietnamese is unsupported for cloning on all tiers
    if "vi" in supported:
        pytest.skip("Vietnamese is supported for cloning on this tier — test not applicable")
    audio_bytes, _ = tts_model.synthesize(
        "Xin chào thế giới",
        lang="vi",
        reference_audio=reference_audio_wav,
        reference_text="Hello world",
    )
    assert isinstance(audio_bytes, bytes) and len(audio_bytes) > 100
    arr, sr = sf.read(io.BytesIO(audio_bytes))
    assert len(arr) / sr > 0.3


def test_voice_clone_ref_audio_is_bytes_not_path(tts_model, config, reference_audio_wav):
    """reference_audio must be bytes — passing a string path should raise TypeError."""
    supported = VOICE_CLONE_CONFIGS[config]["supported_langs"]
    if "en" not in supported:
        pytest.skip(f"English not supported for cloning on tier={config}")
    with pytest.raises((TypeError, AttributeError)):
        tts_model.synthesize(
            "Hello world",
            lang="en",
            reference_audio="/tmp/nonexistent.wav",  # wrong type: str instead of bytes
            reference_text="Hello world",
        )
