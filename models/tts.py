# models/tts.py
from __future__ import annotations
import base64
import io
import json
import logging
import os
import struct
import subprocess
import tarfile
import tempfile
import threading
import urllib.request
import numpy as np
import soundfile as sf
from config import (
    VOXTRAL_LANGS, VOXTRAL_TTS_VOICES, KOKORO_VOICES,
    PIPER_MODELS, TIER_CONFIGS, MODELS_CACHE_DIR,
    VOICE_CLONE_CONFIGS, QWEN3_TTS_LANG_NAMES,
)

logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class CloneWorkerProxy:
    """Persistent subprocess running workers/clone_worker.py in an isolated venv.

    Communication protocol (stdin/stdout, binary):
      Request:  4-byte big-endian uint32 length + UTF-8 JSON
      Response: 4-byte big-endian uint32 length + bytes
        - b'RIFF...' → WAV audio (success)
        - b'\\x00...' → UTF-8 JSON error envelope
        - b''        → ready sentinel (emitted once after model loads)
    """

    MAX_RESTARTS = 3

    def __init__(self, tier: str) -> None:
        self.tier = tier
        self._venv_python = os.path.join(_PROJECT_ROOT, "venvs", f"clone_{tier}", "bin", "python")
        self._worker_script = os.path.join(_PROJECT_ROOT, "workers", "clone_worker.py")
        self._proc: subprocess.Popen | None = None
        self._restart_count = 0
        self._lock = threading.Lock()

    def _ensure_running(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return
        if not os.path.exists(self._venv_python):
            raise RuntimeError(
                f"Clone worker venv not found: {self._venv_python}\n"
                f"Run: python workers/setup_clone_venvs.py --tier {self.tier}"
            )
        logger.info("Starting clone worker for tier=%s", self.tier)
        self._proc = subprocess.Popen(
            [self._venv_python, self._worker_script, "--model", self.tier],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        threading.Thread(target=self._forward_stderr, daemon=True).start()
        # Block until the worker sends its 0-length ready sentinel (or an error)
        sentinel = self._recv_payload()
        if sentinel:
            # Non-empty response during startup = error payload
            err = json.loads(sentinel[1:]) if sentinel[:1] == b"\x00" else {"error": sentinel.decode(errors="replace")}
            raise RuntimeError(f"Clone worker failed to start: {err.get('error')}\n{err.get('traceback','')}")
        logger.info("Clone worker ready (tier=%s)", self.tier)

    def _forward_stderr(self) -> None:
        for line in self._proc.stderr:
            logger.info("[clone_worker/%s] %s", self.tier, line.decode(errors="replace").rstrip())

    def _send_request(self, payload: bytes) -> None:
        length = struct.pack(">I", len(payload))
        self._proc.stdin.write(length + payload)
        self._proc.stdin.flush()

    def _recv_payload(self) -> bytes:
        raw = self._proc.stdout.read(4)
        if len(raw) < 4:
            raise BrokenPipeError("Clone worker stdout closed unexpectedly")
        (length,) = struct.unpack(">I", raw)
        if length == 0:
            return b""
        return self._proc.stdout.read(length)

    def synthesize(self, text: str, lang: str, ref_audio: bytes, ref_text: str | None = None) -> bytes:
        req_payload = json.dumps({
            "text": text,
            "lang": lang,
            "ref_audio_b64": base64.b64encode(ref_audio).decode(),
            "ref_text": ref_text,
        }).encode("utf-8")

        with self._lock:
            for attempt in range(2):
                try:
                    self._ensure_running()
                    self._send_request(req_payload)
                    response = self._recv_payload()
                    if response[:4] == b"RIFF":
                        return response
                    if response[:1] == b"\x00":
                        err = json.loads(response[1:])
                        raise RuntimeError(f"Clone worker error: {err['error']}\n{err.get('traceback', '')}")
                    raise RuntimeError(f"Unexpected worker response: {response[:16]!r}")
                except (BrokenPipeError, OSError) as e:
                    logger.warning("Clone worker (%s) crashed (attempt %d): %s", self.tier, attempt, e)
                    self._restart_count += 1
                    self._proc = None
                    if self._restart_count > self.MAX_RESTARTS or attempt > 0:
                        raise RuntimeError(f"Clone worker {self.tier} failed after {self._restart_count} restarts") from e
        raise RuntimeError("Clone worker failed")

    def shutdown(self) -> None:
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.stdin.close()
                self._proc.wait(timeout=5)
            except Exception:
                self._proc.kill()


class TTSModel:
    def __init__(self, config: str):
        self.config = config
        self._kokoro = None
        self._piper: dict[str, object] = {}   # lang -> sherpa_onnx.OfflineTts
        self._loaded = False
        # Voice cloning: subprocess proxy (isolated venv) for small tier;
        # inline Qwen3-TTS for medium/high (already in main venv)
        self._clone_proxy: CloneWorkerProxy | None = None
        self._qwen3_tts = None

    def load(self):
        if self._loaded:
            return
        if self.config == "medium":
            self._load_kokoro()
        # Piper loaded lazily per language on first use
        self._loaded = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def synthesize(self, text: str, lang: str, reference_audio: bytes | None = None, reference_text: str | None = None) -> tuple[bytes, bool]:
        """Returns (wav_bytes, is_fallback).

        If reference_audio is provided (raw WAV bytes), attempts voice cloning
        using the appropriate model for the current tier. Falls back to the
        standard synthesis if the language is unsupported for cloning.
        """
        self.load()

        if reference_audio is not None:
            result = self._synthesize_clone(text, lang, reference_audio, reference_text)
            if result is not None:
                return result, False
            # Language not supported by clone model — fall through to standard

        if self.config == "high":
            if lang not in VOXTRAL_LANGS:
                wav = self._synthesize_piper(text, lang)
                return wav, True
            return self._synthesize_vllm(text, lang), False

        if self.config == "medium":
            voice = KOKORO_VOICES.get(lang)
            if voice:
                return self._synthesize_kokoro(text, lang, voice), False
            return self._synthesize_piper(text, lang), False

        # small — always piper
        return self._synthesize_piper(text, lang), False

    def is_loaded(self) -> bool:
        return self._loaded

    # ------------------------------------------------------------------
    # Voice cloning dispatcher
    # ------------------------------------------------------------------

    def _synthesize_clone(self, text: str, lang: str, reference_audio: bytes, reference_text: str | None = None) -> bytes | None:
        """Try to synthesize with voice cloning. Returns None if unsupported for this lang."""
        clone_cfg = VOICE_CLONE_CONFIGS.get(self.config, {})
        supported = clone_cfg.get("supported_langs", set())
        if lang not in supported:
            logger.info("Voice cloning not supported for lang=%s on tier=%s, falling back", lang, self.config)
            return None

        if self.config == "small":
            # OuteTTS runs in an isolated venv subprocess to avoid dep conflicts
            if self._clone_proxy is None:
                self._clone_proxy = CloneWorkerProxy("small")
            return self._clone_proxy.synthesize(text, lang, reference_audio, reference_text)

        # medium and high: try isolated venv worker first (avoids transformers pin conflict),
        # fall back to inline Qwen3-TTS if the venv hasn't been set up
        medium_venv = os.path.join(_PROJECT_ROOT, "venvs", "clone_medium", "bin", "python")
        if os.path.exists(medium_venv):
            if self._clone_proxy is None:
                self._clone_proxy = CloneWorkerProxy("medium")
            return self._clone_proxy.synthesize(text, lang, reference_audio, reference_text)
        return self._synthesize_qwen3(text, lang, reference_audio, ref_text=reference_text)

    # ------------------------------------------------------------------
    # Qwen3-TTS 1.7B (Medium + High tier voice cloning, GPU)
    # ------------------------------------------------------------------

    def _load_qwen3_tts(self):
        if self._qwen3_tts is not None:
            return
        import torch
        from qwen_tts import Qwen3TTSModel
        cfg = VOICE_CLONE_CONFIGS[self.config]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._qwen3_tts = Qwen3TTSModel.from_pretrained(
            cfg["model"],
            device_map=device,
            dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        )
        logger.info("Loaded Qwen3-TTS from %s on %s", cfg["model"], device)

    def _synthesize_qwen3(self, text: str, lang: str, reference_audio: bytes, ref_text: str | None = None) -> bytes:
        self._load_qwen3_tts()
        lang_name = QWEN3_TTS_LANG_NAMES.get(lang, "English")
        # Qwen3-TTS requires a file path — write reference audio to a temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(reference_audio)
            ref_path = f.name
        try:
            if ref_text:
                wavs, sr = self._qwen3_tts.generate_voice_clone(
                    text=text,
                    language=lang_name,
                    ref_audio=ref_path,
                    ref_text=ref_text,
                )
            else:
                # x_vector_only_mode skips ICL — no transcript needed, lower quality
                wavs, sr = self._qwen3_tts.generate_voice_clone(
                    text=text,
                    language=lang_name,
                    ref_audio=ref_path,
                    x_vector_only_mode=True,
                )
        finally:
            os.unlink(ref_path)
        return self._to_wav_bytes(np.array(wavs[0], dtype=np.float32), sr)

    # ------------------------------------------------------------------
    # Kokoro-82M (Medium tier)
    # ------------------------------------------------------------------

    def _load_kokoro(self):
        from kokoro_onnx import Kokoro
        from huggingface_hub import snapshot_download
        model_dir = snapshot_download(
            repo_id="NeuML/kokoro-int8-onnx",
            cache_dir=MODELS_CACHE_DIR,
        )
        onnx_path = next(
            (os.path.join(model_dir, f) for f in ("kokoro-int8.onnx", "model.onnx")
             if os.path.exists(os.path.join(model_dir, f))),
            os.path.join(model_dir, "kokoro-int8.onnx"),
        )
        voices_path = os.path.join(model_dir, "voices.bin")
        logger.info("Loading Kokoro-82M from %s", model_dir)
        self._kokoro = Kokoro(onnx_path, voices_path)

    def _synthesize_kokoro(self, text: str, lang: str, voice: str) -> bytes:
        lang_map = {
            "en": "en-us", "zh": "zh", "fr": "fr-fr",
            "es": "es", "it": "it", "pt": "pt-br",
            "hi": "hi", "ar": "ar",
        }
        kokoro_lang = lang_map.get(lang, "en-us")
        samples, sr = self._kokoro.create(text, voice=voice, speed=1.0, lang=kokoro_lang)
        return self._to_wav_bytes(np.array(samples), sr)

    # ------------------------------------------------------------------
    # Piper via sherpa-onnx (Small tier + VI fallback)
    # ------------------------------------------------------------------

    def _get_piper(self, lang: str):
        if lang not in self._piper:
            self._piper[lang] = self._load_piper(lang)
        return self._piper[lang]

    def _load_piper(self, lang: str):
        import sherpa_onnx
        info = PIPER_MODELS.get(lang)
        if info is None:
            raise ValueError(f"No Piper model configured for language '{lang}'")

        model_dir = self._download_piper_model(lang, info)
        model_path = os.path.join(model_dir, info["model"])
        tokens_path = os.path.join(model_dir, info["tokens"])

        # Lexicon and dict_dir for models that use them (e.g. MeloTTS zh)
        lexicon = ""
        if info.get("lexicon"):
            lexicon = os.path.join(model_dir, info["lexicon"])
        dict_dir = ""
        if info.get("dict_dir"):
            dict_dir = os.path.join(model_dir, info["dict_dir"])

        # Locate espeak-ng data directory (needed for phoneme-based VITS models)
        espeak_data_candidates = [
            "/usr/lib/x86_64-linux-gnu/espeak-ng-data",
            "/usr/lib/aarch64-linux-gnu/espeak-ng-data",
            "/usr/lib/arm-linux-gnueabihf/espeak-ng-data",
            "/usr/share/espeak-ng-data",
        ]
        data_dir = "" if lexicon else next((d for d in espeak_data_candidates if os.path.isdir(d)), "")

        config = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                    model=model_path,
                    lexicon=lexicon,
                    tokens=tokens_path,
                    dict_dir=dict_dir,
                    data_dir=data_dir,
                ),
                provider="cpu",
                num_threads=2,
            ),
        )
        logger.info("Loading Piper TTS for lang=%s", lang)
        return sherpa_onnx.OfflineTts(config)

    def _download_piper_model(self, lang: str, info: dict) -> str:
        cache_dir = os.path.join(MODELS_CACHE_DIR, "piper")
        os.makedirs(cache_dir, exist_ok=True)
        model_dir = os.path.join(cache_dir, info["dir"])
        if os.path.isdir(model_dir):
            return model_dir
        archive_path = os.path.join(cache_dir, info["dir"] + ".tar.bz2")
        if not os.path.exists(archive_path):
            logger.info("Downloading Piper model for %s from %s", lang, info["url"])
            urllib.request.urlretrieve(info["url"], archive_path)
        with tarfile.open(archive_path, "r:bz2") as tar:
            tar.extractall(cache_dir)
        return model_dir

    def _synthesize_piper(self, text: str, lang: str) -> bytes:
        tts = self._get_piper(lang)
        audio = tts.generate(text, sid=0, speed=1.0)
        return self._to_wav_bytes(np.array(audio.samples), audio.sample_rate)

    # ------------------------------------------------------------------
    # Voxtral-TTS via vLLM HTTP (High tier)
    # ------------------------------------------------------------------

    def _synthesize_vllm(self, text: str, lang: str) -> bytes:
        import httpx
        cfg = TIER_CONFIGS["high"]
        voice = VOXTRAL_TTS_VOICES.get(lang, "casual_male")
        payload = {
            "input": text,
            "model": cfg["tts_vllm_model"],
            "response_format": "wav",
            "voice": voice,
        }
        resp = httpx.post(
            f"{cfg['tts_vllm_url']}/v1/audio/speech",
            json=payload,
            timeout=120.0,
        )
        resp.raise_for_status()
        return resp.content

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _to_wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
        buf = io.BytesIO()
        sf.write(buf, samples, sample_rate, format="WAV", subtype="PCM_16")
        return buf.getvalue()
