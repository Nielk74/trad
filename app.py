# app.py
import argparse
import base64
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from audio_utils import AudioTooLongError
from config import LANGUAGES, VOXTRAL_LANGS
from models.stt import STTModel
from models.translation import TranslationModel
from models.tts import TTSModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--config", choices=["high", "medium", "small"], default="small")
parser.add_argument("--host", default="0.0.0.0")
parser.add_argument("--port", type=int, default=3003)
args, _ = parser.parse_known_args()

CURRENT_CONFIG = {"value": args.config}

# -----------------------------------------------------------------------
# Model singletons
# -----------------------------------------------------------------------
stt = STTModel(args.config)
translation = TranslationModel(args.config)
tts = TTSModel(args.config)


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    # nothing to clean up — models are cleaned up by GC


app = FastAPI(lifespan=lifespan)

# -----------------------------------------------------------------------
# Request / Response schemas
# -----------------------------------------------------------------------

class TranscribeRequest(BaseModel):
    audio: str    # base64-encoded WAV or webm
    lang: str


class TranslateRequest(BaseModel):
    text: str
    source: str
    target: str


class SynthesizeRequest(BaseModel):
    text: str
    lang: str


class ConfigRequest(BaseModel):
    config: str


# -----------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------

@app.get("/languages")
def get_languages():
    result = []
    for lang in LANGUAGES:
        tiers = ["small", "medium"]
        if lang["voxtral_stt"] and lang["voxtral_tts"]:
            tiers.append("high")
        result.append({
            "code": lang["code"],
            "name": lang["name"],
            "flag": lang["flag"],
            "tiers": tiers,
            "fallback_in_high": not (lang["voxtral_stt"] and lang["voxtral_tts"]),
        })
    return {"languages": result}


@app.get("/status")
def get_status():
    fallback_active = []
    if CURRENT_CONFIG["value"] == "high":
        fallback_active = [
            f"{l['code']}_stt" for l in LANGUAGES if not l["voxtral_stt"]
        ] + [
            f"{l['code']}_tts" for l in LANGUAGES if not l["voxtral_tts"]
        ]
    return {
        "config": CURRENT_CONFIG["value"],
        "models_loaded": {
            "stt": stt.is_loaded(),
            "translation": translation.is_loaded(),
            "tts": tts.is_loaded(),
        },
        "fallback_active": fallback_active,
    }


@app.get("/health")
def health():
    return {
        "stt": stt.is_loaded(),
        "translation": translation.is_loaded(),
        "tts": tts.is_loaded(),
    }


@app.post("/transcribe")
def transcribe(req: TranscribeRequest):
    try:
        audio_bytes = base64.b64decode(req.audio)
        text, is_fallback = stt.transcribe(audio_bytes, req.lang)
        return {"text": text, "fallback": is_fallback}
    except AudioTooLongError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Transcription error")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/translate")
def translate(req: TranslateRequest):
    try:
        result = translation.translate(req.text, source=req.source, target=req.target)
        return {"translation": result}
    except Exception as e:
        logger.exception("Translation error")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/synthesize")
def synthesize(req: SynthesizeRequest):
    try:
        wav_bytes, is_fallback = tts.synthesize(req.text, req.lang)
        audio_b64 = base64.b64encode(wav_bytes).decode()
        return {"audio": audio_b64, "fallback": is_fallback}
    except Exception as e:
        logger.exception("TTS error")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/config")
def set_config(req: ConfigRequest):
    global stt, translation, tts
    if req.config not in ("high", "medium", "small"):
        raise HTTPException(status_code=400, detail="Invalid config")
    CURRENT_CONFIG["value"] = req.config
    stt = STTModel(req.config)
    translation = TranslationModel(req.config)
    tts = TTSModel(req.config)
    return {"config": req.config, "status": "reloading"}


@app.get("/")
def index():
    return FileResponse("frontend/index.html")


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    import os
    ssl_keyfile = "key.pem" if os.path.exists("key.pem") else None
    ssl_certfile = "cert.pem" if os.path.exists("cert.pem") else None
    uvicorn.run(app, host=args.host, port=args.port, ssl_keyfile=ssl_keyfile, ssl_certfile=ssl_certfile)
