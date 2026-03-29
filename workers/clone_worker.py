#!/usr/bin/env python3
"""
Clone worker process — runs inside an isolated venv (venvs/clone_small or venvs/clone_medium).

Protocol (stdin/stdout, binary):
  Request  (parent → worker): 4-byte big-endian uint32 length, then UTF-8 JSON payload.
  Response (worker → parent): 4-byte big-endian uint32 length, then bytes payload.
    - payload starts with b'RIFF' → success, raw WAV audio
    - payload starts with 0x00   → error, followed by UTF-8 JSON {"error": "...", "traceback": "..."}
    - payload is empty (length=0) → ready sentinel (sent once after model loads)

All stderr is forwarded to the parent's logger by a daemon thread in CloneWorkerProxy.
"""
import argparse
import base64
import io
import json
import os
import struct
import sys
import tempfile
import traceback


# ── I/O protocol ────────────────────────────────────────────────────────────

def read_request() -> dict:
    raw = sys.stdin.buffer.read(4)
    if len(raw) < 4:
        raise EOFError("stdin closed")
    (length,) = struct.unpack(">I", raw)
    payload = sys.stdin.buffer.read(length)
    return json.loads(payload.decode("utf-8"))


def write_payload(buf: "io.RawIOBase", payload: bytes) -> None:
    buf.write(struct.pack(">I", len(payload)))
    buf.write(payload)
    buf.flush()


def write_error(buf: "io.RawIOBase", message: str, tb: str = "") -> None:
    body = json.dumps({"error": message, "traceback": tb}).encode("utf-8")
    write_payload(buf, b"\x00" + body)


# ── Model: Qwen3-TTS (small=0.6B, medium/high=1.7B) ─────────────────────────

QWEN3_MODELS = {
    "small":  "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    "medium": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
}

QWEN3_LANG_NAMES = {
    "en": "English", "zh": "Chinese", "fr": "French", "de": "German",
    "es": "Spanish", "it": "Italian", "pt": "Portuguese", "ru": "Russian",
    "ja": "Japanese", "ko": "Korean",
}


def load_model(tier: str):
    import torch
    from qwen_tts import Qwen3TTSModel
    torch.set_num_threads(os.cpu_count() or 4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return Qwen3TTSModel.from_pretrained(
        QWEN3_MODELS[tier],
        device_map=device,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    )


def _max_tokens(text: str) -> int:
    """Estimate max_new_tokens from text length: ~12 tokens/word at 12 Hz, +50% headroom."""
    words = max(len(text.split()), 1)
    return min(int(words * 12 * 1.5) + 50, 2048)


def synthesize(model, text: str, lang: str, ref_audio_bytes: bytes, ref_text: str | None) -> bytes:
    import numpy as np
    import soundfile as sf

    lang_name = QWEN3_LANG_NAMES.get(lang, "English")
    max_tokens = _max_tokens(text)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(ref_audio_bytes)
        ref_path = f.name
    try:
        if ref_text:
            wavs, sr = model.generate_voice_clone(
                text=text, language=lang_name, ref_audio=ref_path, ref_text=ref_text,
                max_new_tokens=max_tokens,
            )
        else:
            wavs, sr = model.generate_voice_clone(
                text=text, language=lang_name, ref_audio=ref_path, x_vector_only_mode=True,
                max_new_tokens=max_tokens,
            )
    finally:
        os.unlink(ref_path)

    buf = io.BytesIO()
    sf.write(buf, np.array(wavs[0], dtype="float32"), sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["small", "medium"], required=True)
    args = parser.parse_args()

    # Capture real stdout before anything else can pollute it
    real_stdout = sys.stdout.buffer
    # Redirect print() noise from model loading to stderr
    sys.stdout = sys.stderr

    try:
        model = load_model(args.model)
    except Exception as e:
        write_error(real_stdout, str(e), traceback.format_exc())
        sys.exit(1)

    # Send ready sentinel (0-length payload) to unblock the parent
    write_payload(real_stdout, b"")

    # Request/response loop
    while True:
        try:
            req = read_request()
        except EOFError:
            break
        try:
            ref_audio = base64.b64decode(req["ref_audio_b64"])
            wav_bytes = synthesize(model, req["text"], req["lang"],
                                   ref_audio, req.get("ref_text"))
            write_payload(real_stdout, wav_bytes)
        except Exception as e:
            write_error(real_stdout, str(e), traceback.format_exc())


if __name__ == "__main__":
    main()
