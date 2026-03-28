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
