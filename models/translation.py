# models/translation.py
from __future__ import annotations
import logging
import os
from config import OPUS_MT_MODELS, TIER_CONFIGS, MODELS_CACHE_DIR

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

    def _get_opusmt_model(self, src: str, tgt: str):
        key = (src, tgt)
        if key not in self._models:
            model_id = OPUS_MT_MODELS.get(key)
            if model_id is None:
                raise ValueError(f"No opus-mt model for {src}→{tgt}")
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            logger.info("Loading opus-mt %s→%s (%s)", src, tgt, model_id)
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
            self._models[key] = (tokenizer, model)
        return self._models[key]

    def _run_opusmt(self, text: str, src: str, tgt: str) -> str:
        tokenizer, model = self._get_opusmt_model(src, tgt)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        outputs = model.generate(**inputs, max_length=512)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    def _translate_opusmt(self, text: str, source: str, target: str) -> str:
        direct_key = (source, target)
        if direct_key in OPUS_MT_MODELS:
            return self._run_opusmt(text, source, target)

        # Pivot through English
        if source != "en":
            text = self._run_opusmt(text, source, "en")
        if target != "en":
            text = self._run_opusmt(text, "en", target)
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
        n_gpu = -1 if self.config == "high" else 0
        self._models["gguf"] = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_gpu_layers=n_gpu,
            verbose=False,
        )

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
        text = output["choices"][0]["text"].strip()
        # Model occasionally emits a malformed stop token variant — strip it
        for suffix in ("<|im_end|>", "<|im_end", "<|im_end|"):
            if text.endswith(suffix):
                text = text[: -len(suffix)].strip()
                break
        return text
