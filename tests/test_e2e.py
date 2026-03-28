# tests/test_e2e.py
"""
End-to-end tests: audio clip -> STT -> translate -> TTS.
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
    """French audio -> transcript -> Chinese -> synthesized Chinese audio."""
    wav = load_wav_bytes("fr")
    text, _ = stt_model.transcribe(wav, lang="fr")
    assert len(text) > 0

    translation = translation_model.translate(text, source="fr", target="zh")
    assert len(translation) > 0

    audio_bytes, _ = tts_model.synthesize(translation, lang="zh")
    arr, sr = sf.read(io.BytesIO(audio_bytes))
    assert len(arr) / sr > 0.3


def test_full_pipeline_en_to_vi(stt_model, translation_model, tts_model, config):
    """English audio -> transcript -> Vietnamese -> synthesized Vietnamese audio.
    In high tier, TTS for VI must trigger fallback=True."""
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
    try:
        resp = httpx.get("http://localhost:8080/status", timeout=2.0)
    except Exception:
        pytest.skip("App not running — start with: python app.py --config high")

    data = resp.json()
    assert data["config"] == config
    if config == "high":
        assert "vi_stt" in data["fallback_active"]
        assert "vi_tts" in data["fallback_active"]
