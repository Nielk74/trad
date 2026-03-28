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
