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
    "zh": None,  # kokoro-onnx cannot phonemize Chinese; falls back to Piper/MeloTTS
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
        "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-fr_FR-siwis-medium.tar.bz2",
        "dir": "vits-piper-fr_FR-siwis-medium",
        "model": "fr_FR-siwis-medium.onnx",
        "tokens": "tokens.txt",
        "espeak_lang": "fr",
    },
    "zh": {
        "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-melo-tts-zh_en.tar.bz2",
        "dir": "vits-melo-tts-zh_en",
        "model": "model.onnx",
        "tokens": "tokens.txt",
        "lexicon": "lexicon.txt",
        "dict_dir": "dict",
        "espeak_lang": "zh",
    },
    "de": {
        "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-de_DE-thorsten-medium.tar.bz2",
        "dir": "vits-piper-de_DE-thorsten-medium",
        "model": "de_DE-thorsten-medium.onnx",
        "tokens": "tokens.txt",
        "espeak_lang": "de",
    },
    "es": {
        "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-es_ES-davefx-medium.tar.bz2",
        "dir": "vits-piper-es_ES-davefx-medium",
        "model": "es_ES-davefx-medium.onnx",
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
        "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-nl_NL-pim-medium.tar.bz2",
        "dir": "vits-piper-nl_NL-pim-medium",
        "model": "nl_NL-pim-medium.onnx",
        "tokens": "tokens.txt",
        "espeak_lang": "nl",
    },
    "hi": {
        "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-hi_IN-pratham-medium.tar.bz2",
        "dir": "vits-piper-hi_IN-pratham-medium",
        "model": "hi_IN-pratham-medium.onnx",
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
        "stt_fallback_model": "large-v3-turbo",
        "translation_model_repo": "tencent/HY-MT1.5-7B-GGUF",
        "translation_model_file": "HY-MT1.5-7B-Q4_K_M.gguf",
        "tts_vllm_url": "http://localhost:8000",
        "tts_vllm_model": "mistralai/Voxtral-4B-TTS-2603",
    },
    "medium": {
        "stt_model": "large-v3-turbo",
        "translation_model_repo": "tencent/HY-MT1.5-1.8B-GGUF",
        "translation_model_file": "HY-MT1.5-1.8B-Q4_K_M.gguf",
        "tts_kokoro_model": "NeuML/kokoro-int8-onnx",
    },
    "small": {
        "stt_model": "small",
        "translation_model": "opus-mt",
        "tts_model": "piper",
    },
}

# Directory where downloaded models/voices are cached
MODELS_CACHE_DIR = "models_cache"

# Voice cloning model configs per tier
# Small: not supported — OuteTTS requires torchcodec/CUDA NPP libs not available on CPU-only setups
# Medium: Qwen3-TTS 1.7B (GPU, 10 languages, Apache 2.0)
# High: Qwen3-TTS 1.7B (GPU, same model — vLLM Voxtral does not support ref_audio)

VOICE_CLONE_CONFIGS: dict[str, dict] = {
    "small": {
        "model": None,
        "supported_langs": set(),  # not supported — falls back to standard Piper TTS
    },
    "medium": {
        "model": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        "supported_langs": {"en", "zh", "fr", "de", "es", "it", "pt", "ru", "ja", "ko"},
    },
    "high": {
        "model": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        "supported_langs": {"en", "zh", "fr", "de", "es", "it", "pt", "ru", "ja", "ko"},
    },
}

# Language name mapping for Qwen3-TTS (requires full language name)
QWEN3_TTS_LANG_NAMES: dict[str, str] = {
    "en": "English",
    "zh": "Chinese",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "ja": "Japanese",
    "ko": "Korean",
}
