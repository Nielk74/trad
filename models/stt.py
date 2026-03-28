# models/stt.py
from __future__ import annotations
import logging
import numpy as np
from audio_utils import to_pcm16k, AudioTooLongError
from config import VOXTRAL_LANGS, TIER_CONFIGS

logger = logging.getLogger(__name__)


class STTModel:
    def __init__(self, config: str):
        self.config = config
        self._primary: object = None
        self._fallback: object = None   # faster-whisper, used for VI in high tier
        self._loaded = False

    def load(self):
        if self._loaded:
            return
        cfg = TIER_CONFIGS[self.config]
        if self.config == "small":
            self._primary = self._load_faster_whisper(cfg["stt_model"])
        elif self.config == "medium":
            self._primary = self._load_faster_whisper(cfg["stt_model"])
        elif self.config == "high":
            self._primary = self._load_voxtral(cfg["stt_model"])
            self._fallback = self._load_faster_whisper(cfg["stt_fallback_model"])
        self._loaded = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transcribe(self, audio_bytes: bytes, lang: str) -> tuple[str, bool]:
        """
        Returns (transcript_text, is_fallback).
        is_fallback=True when high-tier falls back to faster-whisper for VI.
        """
        self.load()
        arr, _sr = to_pcm16k(audio_bytes)  # raises AudioTooLongError if > 30s

        needs_fallback = self.config == "high" and lang not in VOXTRAL_LANGS
        if needs_fallback:
            text = self._transcribe_faster_whisper(self._fallback, arr, lang)
            return text, True

        if self.config in ("small", "medium"):
            text = self._transcribe_faster_whisper(self._primary, arr, lang)
            return text, False

        # high tier, supported language
        text = self._transcribe_voxtral(self._primary, arr, lang)
        return text, False

    def is_loaded(self) -> bool:
        return self._loaded

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    def _load_faster_whisper(self, model_size: str):
        from faster_whisper import WhisperModel
        device = "cpu"
        compute = "int8"
        if self.config == "high":
            device = "auto"
            compute = "float16"
        logger.info("Loading faster-whisper %s (device=%s)", model_size, device)
        return WhisperModel(model_size, device=device, compute_type=compute)

    def _load_voxtral(self, model_id: str):
        # Voxtral-Mini-4B-Realtime-2602 uses the transformers ASR pipeline.
        # Verify exact pipeline class against model card if issues arise.
        from transformers import pipeline
        import torch
        device = 0 if torch.cuda.is_available() else -1
        logger.info("Loading Voxtral STT %s", model_id)
        return pipeline(
            "automatic-speech-recognition",
            model=model_id,
            device=device,
            torch_dtype=torch.float16,
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _transcribe_faster_whisper(self, model, arr: np.ndarray, lang: str) -> str:
        segments, _info = model.transcribe(arr, language=lang, beam_size=5)
        return " ".join(s.text.strip() for s in segments).strip()

    def _transcribe_voxtral(self, pipe, arr: np.ndarray, lang: str) -> str:
        result = pipe({"array": arr, "sampling_rate": 16000}, generate_kwargs={"language": lang})
        return result["text"].strip()
