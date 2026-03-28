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
