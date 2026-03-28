# tests/test_translation.py
import pytest


def test_translate_returns_string(translation_model):
    result = translation_model.translate("Hello world", source="en", target="fr")
    assert isinstance(result, str)
    assert len(result) > 0


def test_translate_en_to_fr(translation_model):
    result = translation_model.translate("Hello", source="en", target="fr")
    assert result.strip() != ""
    assert result.strip().lower() != "hello"


def test_translate_same_lang_returns_input(translation_model):
    result = translation_model.translate("Bonjour", source="fr", target="fr")
    assert result == "Bonjour"


def test_translate_pivot_non_en_pair(translation_model, config):
    """zh->fr must work even if no direct model exists (pivot through EN)."""
    if config != "small":
        pytest.skip("Pivot only needed for small tier opus-mt")
    result = translation_model.translate("你好", source="zh", target="fr")
    assert isinstance(result, str)
    assert len(result) > 0
