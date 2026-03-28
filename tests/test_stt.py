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
