import asyncio
import json
import logging
import os
import subprocess
import sys
import tempfile
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import resampy
from fastapi import FastAPI, WebSocket, UploadFile, File, Form
from fastapi.responses import JSONResponse, PlainTextResponse

from app.config import load_config, Config, MODEL_MAP
from app.factory import build_engine
from app.streaming_session import StreamingSession
from app.asr_engines.base import ASREngine
from app.asr_number_normalizer import normalize_asr_numbers


cfg = load_config()

logging.basicConfig(
    level=cfg.log_level,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("asr_server")

app = FastAPI(title="Nemotron 3.5 ASR Server", version="1.0.0")

ENGINE_CACHE: dict[str, ASREngine] = {}


# ---------------------------------------------------------------------
# Additional logic: server-side audio/session logging
# ---------------------------------------------------------------------
AUDIO_LOG_DIR = Path(os.getenv("AUDIO_LOG_DIR", "/srv/audio_logs"))
AUDIO_LOG_DIR.mkdir(parents=True, exist_ok=True)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_session_id(client) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")

    client_host = "unknown"
    client_port = "unknown"

    try:
        client_host = str(client.host)
        client_port = str(client.port)
    except Exception:
        pass

    safe_host = client_host.replace(".", "_").replace(":", "_")
    return f"{ts}_{safe_host}_{client_port}"


def save_pcm_as_wav(pcm_bytes: bytes, wav_path: Path, sample_rate: int):
    """
    Save raw PCM16 mono audio bytes as WAV.
    """
    if not pcm_bytes:
        return

    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)


def write_json(path: Path, data: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def append_jsonl(path: Path, data: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


def normalize_language(language: Optional[str]) -> str:
    if not language:
        return "auto"

    lang = language.strip()

    if not lang:
        return "auto"

    low = lang.lower()

    if low == "auto":
        return "auto"

    if low == "en":
        return "en-US"

    if low == "es":
        return "es-US"

    return lang


def clean_model_leaked_tags(text: str) -> str:
    """
    Removes leaked prompt tags like <en-US>, <es-US>.
    """
    if not text:
        return ""

    import re

    return re.sub(r"<[a-z]{2}-[A-Z]{2}>\s*", "", text).strip()


# ---------------------------------------------------------------------
# Existing logic: preload engines
# ---------------------------------------------------------------------
async def preload_engines():
    log.info("Preloading ASR engines...")

    for backend, model_name in MODEL_MAP.items():
        try:
            log.info(f"Initializing engine: {backend} ({model_name})")

            tmp_cfg = Config()
            object.__setattr__(tmp_cfg, "asr_backend", backend)
            object.__setattr__(tmp_cfg, "model_name", model_name)
            object.__setattr__(tmp_cfg, "device", cfg.device)
            object.__setattr__(tmp_cfg, "sample_rate", cfg.sample_rate)

            engine = build_engine(tmp_cfg)
            load_sec = engine.load()

            ENGINE_CACHE[backend] = engine
            log.info(f"✅ Preloaded '{backend}' in {load_sec:.2f}s")

        except Exception:
            log.exception(f"Failed to preload '{backend}'")

    log.info(f"All engines preloaded. Available: {list(ENGINE_CACHE.keys())}")


@app.on_event("startup")
async def startup_event():
    log.info("Server startup initiated")
    await preload_engines()


def get_engine(backend: str) -> ASREngine:
    if backend not in ENGINE_CACHE:
        raise ValueError(
            f"Engine '{backend}' not loaded. Available: {list(ENGINE_CACHE.keys())}"
        )
    return ENGINE_CACHE[backend]


# ---------------------------------------------------------------------
# Existing + additional health route
# ---------------------------------------------------------------------
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "engines": list(ENGINE_CACHE.keys()),
        "device": cfg.device,
        "sample_rate": cfg.sample_rate,
        "audio_log_dir": str(AUDIO_LOG_DIR),
    }


@app.get("/")
async def root():
    return {
        "status": "ok",
        "service": "Nemotron 3.5 ASR Server",
        "websocket_endpoint": "/asr/realtime-custom-vad",
        "openai_transcription_endpoint": "/v1/audio/transcriptions",
        "audio_log_dir": str(AUDIO_LOG_DIR),
    }


# ---------------------------------------------------------------------
# Additional logic: OpenAI-compatible endpoint helpers
# ---------------------------------------------------------------------
def convert_upload_to_pcm16_16k_mono(
    input_bytes: bytes,
    suffix: str = ".wav",
) -> bytes:
    """
    Convert uploaded audio into raw PCM16 mono audio at cfg.sample_rate.
    Supports WAV, MP3, M4A, FLAC, OGG, WEBM, etc. through ffmpeg.
    """

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as in_file:
        in_file.write(input_bytes)
        input_path = in_file.name

    output_path = input_path + ".pcm"

    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            input_path,
            "-ac",
            "1",
            "-ar",
            str(cfg.sample_rate),
            "-f",
            "s16le",
            "-acodec",
            "pcm_s16le",
            output_path,
        ]

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {result.stderr}")

        with open(output_path, "rb") as f:
            return f.read()

    finally:
        try:
            os.remove(input_path)
        except FileNotFoundError:
            pass

        try:
            os.remove(output_path)
        except FileNotFoundError:
            pass


async def run_pcm_through_session(
    pcm_bytes: bytes,
    engine: ASREngine,
    language: str,
    source: str = "openai-http",
    log_audio: bool = True,
) -> str:
    """
    Used by /v1/audio/transcriptions.
    Runs raw PCM16 audio through the same StreamingSession as WebSocket mode.
    """

    session_id = f"openai_http_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
    session_dir = AUDIO_LOG_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    raw_pcm_path = session_dir / "audio.raw.pcm"
    wav_path = session_dir / "audio.wav"
    metadata_path = session_dir / "metadata.json"
    events_path = session_dir / "events.jsonl"

    started_at = utc_now_iso()
    start_perf = asyncio.get_running_loop().time()

    session = StreamingSession(engine, cfg)

    final_texts: list[str] = []
    transcript_events: list[dict] = []

    chunk_ms = 100
    chunk_bytes = int(cfg.sample_rate * chunk_ms / 1000) * 2

    if log_audio:
        with open(raw_pcm_path, "wb") as f:
            f.write(pcm_bytes)

        try:
            save_pcm_as_wav(
                pcm_bytes=pcm_bytes,
                wav_path=wav_path,
                sample_rate=cfg.sample_rate,
            )
        except Exception:
            log.exception(f"Failed to save OpenAI HTTP WAV | session_id={session_id}")

    write_json(
        metadata_path,
        {
            "session_id": session_id,
            "source": source,
            "started_at_utc": started_at,
            "language": language,
            "server_sample_rate": cfg.sample_rate,
            "raw_pcm_bytes": len(pcm_bytes),
            "audio_duration_sec": round(len(pcm_bytes) / 2 / cfg.sample_rate, 3)
            if pcm_bytes
            else 0.0,
            "audio_raw_pcm_path": str(raw_pcm_path),
            "audio_wav_path": str(wav_path),
            "events_path": str(events_path),
            "status": "started",
        },
    )

    loop = asyncio.get_running_loop()

    for i in range(0, len(pcm_bytes), chunk_bytes):
        chunk = pcm_bytes[i : i + chunk_bytes]

        if not chunk:
            continue

        events = await loop.run_in_executor(
            None,
            session.process_chunk,
            chunk,
        )

        for ev_type, text, ttfb in events:
            text = clean_model_leaked_tags(text)
            text = normalize_asr_numbers(text, use_itn=True)
            _log_transcript(ev_type, text, ttfb, language, source)

            event_payload = {
                "timestamp_utc": utc_now_iso(),
                "session_id": session_id,
                "source": source,
                "type": ev_type,
                "text": text,
                "language": language,
                "ttfb_ms": ttfb,
            }

            transcript_events.append(event_payload)
            append_jsonl(events_path, event_payload)

            if ev_type == "final" and text:
                final_texts.append(text.strip())

    flush_events = await loop.run_in_executor(None, session.flush)

    for ev_type, text, ttfb in flush_events:
        text = clean_model_leaked_tags(text)
        text = normalize_asr_numbers(text, use_itn=True)
        _log_transcript(ev_type, text, ttfb, language, source)

        event_payload = {
            "timestamp_utc": utc_now_iso(),
            "session_id": session_id,
            "source": source,
            "type": ev_type,
            "text": text,
            "language": language,
            "ttfb_ms": ttfb,
        }

        transcript_events.append(event_payload)
        append_jsonl(events_path, event_payload)

        if ev_type == "final" and text:
            final_texts.append(text.strip())

    transcript = " ".join(final_texts).strip()

    ended_at = utc_now_iso()
    end_perf = asyncio.get_running_loop().time()
    wall_duration_sec = end_perf - start_perf

    write_json(
        metadata_path,
        {
            "session_id": session_id,
            "source": source,
            "started_at_utc": started_at,
            "ended_at_utc": ended_at,
            "wall_duration_sec": round(wall_duration_sec, 3),
            "audio_duration_sec": round(len(pcm_bytes) / 2 / cfg.sample_rate, 3)
            if pcm_bytes
            else 0.0,
            "language": language,
            "server_sample_rate": cfg.sample_rate,
            "raw_pcm_bytes": len(pcm_bytes),
            "audio_raw_pcm_path": str(raw_pcm_path),
            "audio_wav_path": str(wav_path),
            "events_path": str(events_path),
            "final_transcript": transcript,
            "final_transcripts": [
                ev["text"]
                for ev in transcript_events
                if ev.get("type") == "final" and ev.get("text")
            ],
            "status": "completed",
        },
    )

    log.info(
        f"OPENAI_HTTP_AUDIO_LOG_END | session_id={session_id} "
        f"wall_duration_sec={wall_duration_sec:.3f} "
        f"audio_duration_sec={len(pcm_bytes) / 2 / cfg.sample_rate:.3f} "
        f"wav={wav_path}"
    )

    return transcript


# ---------------------------------------------------------------------
# Additional logic: OpenAI-compatible endpoints for LiveKit
# ---------------------------------------------------------------------
@app.get("/v1/models")
async def list_openai_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "nemotron-3.5-asr-streaming-0.6b",
                "object": "model",
                "owned_by": "local",
            }
        ],
    }


@app.post("/v1/audio/transcriptions")
async def openai_audio_transcriptions(
    file: UploadFile = File(...),
    model: str = Form("nemotron-3.5-asr-streaming-0.6b"),
    language: Optional[str] = Form("auto"),
    response_format: Optional[str] = Form("json"),
):
    """
    OpenAI-compatible transcription endpoint.

    LiveKit should use:
        base_url="http://HOST:8003/v1"
    """

    lang = normalize_language(language)

    log.info(
        f"OpenAI-compatible STT request | model={model} language={lang} "
        f"response_format={response_format} filename={file.filename}"
    )

    if model != "nemotron-3.5-asr-streaming-0.6b":
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": f"Unsupported model: {model}",
                    "type": "invalid_request_error",
                }
            },
        )

    try:
        engine = get_engine("nemotron")
    except ValueError as e:
        log.exception("Nemotron engine not available")
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": str(e),
                    "type": "server_error",
                }
            },
        )

    try:
        engine.set_language(lang)

        audio_bytes = await file.read()

        suffix = ".wav"
        if file.filename and "." in file.filename:
            suffix = "." + file.filename.rsplit(".", 1)[-1]

        pcm_bytes = convert_upload_to_pcm16_16k_mono(
            input_bytes=audio_bytes,
            suffix=suffix,
        )

        transcript = await run_pcm_through_session(
            pcm_bytes=pcm_bytes,
            engine=engine,
            language=lang,
            source="openai-http",
            log_audio=True,
        )

        if response_format == "text":
            return PlainTextResponse(transcript)

        if response_format == "verbose_json":
            return {
                "task": "transcribe",
                "language": lang,
                "duration": round(len(pcm_bytes) / 2 / cfg.sample_rate, 3)
                if pcm_bytes
                else 0.0,
                "text": transcript,
                "segments": [],
            }

        return {
            "text": transcript,
        }

    except Exception as e:
        log.exception("OpenAI-compatible transcription failed")
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": str(e),
                    "type": "server_error",
                }
            },
        )


# ---------------------------------------------------------------------
# Existing WebSocket endpoint with added logging logic
# ---------------------------------------------------------------------
@app.websocket("/asr/realtime-custom-vad")
async def ws_asr(ws: WebSocket):
    log.info(f"WS connection request from {ws.client}")
    await ws.accept()

    # Additional logic: create per-session logging folder
    session_id = make_session_id(ws.client)
    session_dir = AUDIO_LOG_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    raw_pcm_path = session_dir / "audio.raw.pcm"
    wav_path = session_dir / "audio.wav"
    metadata_path = session_dir / "metadata.json"
    events_path = session_dir / "events.jsonl"

    session_started_at = utc_now_iso()
    session_start_perf = asyncio.get_running_loop().time()

    captured_pcm = bytearray()
    transcript_events: list[dict] = []

    log.info(
        f"AUDIO_LOG_START | session_id={session_id} client={ws.client} "
        f"dir={session_dir} started_at={session_started_at}"
    )

    backend = None
    client_sample_rate = None
    language = None

    try:
        raw_init = await ws.receive_text()
        init = json.loads(raw_init)
    except Exception as e:
        log.warning(f"Bad init message: {e}")

        write_json(
            metadata_path,
            {
                "session_id": session_id,
                "client": str(ws.client),
                "started_at_utc": session_started_at,
                "status": "bad_init_message",
                "error": str(e),
            },
        )

        await ws.close(code=4001)
        return

    backend = init.get("backend", "nemotron")
    client_sample_rate = int(init.get("sample_rate", cfg.sample_rate))

    language = init.get("language")
    if not language:
        log.warning(f"Client {ws.client} did not send 'language' — defaulting to 'auto'")
        language = "auto"

    language = normalize_language(language)

    write_json(
        metadata_path,
        {
            "session_id": session_id,
            "client": str(ws.client),
            "started_at_utc": session_started_at,
            "backend": backend,
            "language": language,
            "client_sample_rate": client_sample_rate,
            "server_sample_rate": cfg.sample_rate,
            "audio_raw_pcm_path": str(raw_pcm_path),
            "audio_wav_path": str(wav_path),
            "events_path": str(events_path),
            "status": "started",
        },
    )

    if backend not in MODEL_MAP:
        log.warning(f"Invalid backend requested: '{backend}'")

        error_payload = {
            "timestamp_utc": utc_now_iso(),
            "session_id": session_id,
            "type": "error",
            "text": f"Unknown backend '{backend}'",
            "language": language,
        }

        append_jsonl(events_path, error_payload)

        await ws.send_text(json.dumps({"error": f"Unknown backend '{backend}'"}))
        await ws.close(code=4000)
        return

    try:
        engine = get_engine(backend)
    except ValueError as e:
        log.error(str(e))

        error_payload = {
            "timestamp_utc": utc_now_iso(),
            "session_id": session_id,
            "type": "error",
            "text": str(e),
            "language": language,
        }

        append_jsonl(events_path, error_payload)

        await ws.send_text(json.dumps({"error": str(e)}))
        await ws.close(code=4000)
        return

    engine.set_language(language)

    log.info(
        f"WS connected | backend={backend} language={language} "
        f"client_sr={client_sample_rate} server_sr={cfg.sample_rate} "
        f"client={ws.client} session_id={session_id}"
    )

    def upsample_if_needed(pcm: bytes) -> bytes:
        if not pcm or client_sample_rate == cfg.sample_rate:
            return pcm
        x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        y = resampy.resample(x, client_sample_rate, cfg.sample_rate)
        y = np.clip(y, -1.0, 1.0)
        return (y * 32767.0).astype(np.int16).tobytes()

    session = StreamingSession(engine, cfg)

    try:
        while True:
            msg = await ws.receive()

            if msg["type"] == "websocket.disconnect":
                log.info(f"Client disconnected: {ws.client}")
                break

            # Text frame: EOF signal {"type": "eof"}
            if msg.get("text"):
                try:
                    ctrl = json.loads(msg["text"])
                    if ctrl.get("type") == "eof":
                        log.info(
                            f"EOF from {ws.client} — flushing last utterance | "
                            f"session_id={session_id}"
                        )

                        loop = asyncio.get_running_loop()
                        events = await loop.run_in_executor(None, session.flush)

                        for ev_type, text, ttfb in events:
                            text = clean_model_leaked_tags(text)
                            text = normalize_asr_numbers(text, use_itn=True)

                            _log_transcript(ev_type, text, ttfb, language, ws.client)

                            event_payload = {
                                "timestamp_utc": utc_now_iso(),
                                "session_id": session_id,
                                "type": ev_type,
                                "text": text,
                                "language": language,
                                "ttfb_ms": ttfb,
                            }

                            transcript_events.append(event_payload)
                            append_jsonl(events_path, event_payload)

                            await ws.send_text(
                                json.dumps(
                                    {
                                        "type": ev_type,
                                        "text": text,
                                        "t_start": ttfb,
                                    }
                                )
                            )

                        done_payload = {
                            "timestamp_utc": utc_now_iso(),
                            "session_id": session_id,
                            "type": "done",
                            "text": "",
                            "language": language,
                        }

                        append_jsonl(events_path, done_payload)

                        await ws.send_text(json.dumps({"type": "done"}))

                except (json.JSONDecodeError, AttributeError):
                    pass
                continue

            data = msg.get("bytes")
            if data is None:
                continue

            data = upsample_if_needed(data)

            # Additional logic: save audio server-side
            captured_pcm.extend(data)

            with open(raw_pcm_path, "ab") as f:
                f.write(data)

            loop = asyncio.get_running_loop()
            events = await loop.run_in_executor(None, session.process_chunk, data)

            for ev_type, text, ttfb in events:
                text = clean_model_leaked_tags(text)
                text = normalize_asr_numbers(text, use_itn=True)

                _log_transcript(ev_type, text, ttfb, language, ws.client)

                event_payload = {
                    "timestamp_utc": utc_now_iso(),
                    "session_id": session_id,
                    "type": ev_type,
                    "text": text,
                    "language": language,
                    "ttfb_ms": ttfb,
                }

                transcript_events.append(event_payload)
                append_jsonl(events_path, event_payload)

                await ws.send_text(
                    json.dumps(
                        {
                            "type": ev_type,
                            "text": text,
                            "t_start": ttfb,
                        }
                    )
                )

    except Exception:
        log.exception(f"Error during WebSocket session for {ws.client}")

    finally:
        # Additional logic: finalize session metadata and WAV
        session_ended_at = utc_now_iso()
        session_end_perf = asyncio.get_running_loop().time()
        wall_duration_sec = session_end_perf - session_start_perf

        audio_duration_sec = 0.0
        if captured_pcm:
            audio_duration_sec = len(captured_pcm) / 2 / cfg.sample_rate

        try:
            save_pcm_as_wav(
                pcm_bytes=bytes(captured_pcm),
                wav_path=wav_path,
                sample_rate=cfg.sample_rate,
            )
        except Exception:
            log.exception(f"Failed to save WAV for session_id={session_id}")

        final_metadata = {
            "session_id": session_id,
            "client": str(ws.client),
            "started_at_utc": session_started_at,
            "ended_at_utc": session_ended_at,
            "wall_duration_sec": round(wall_duration_sec, 3),
            "audio_duration_sec": round(audio_duration_sec, 3),
            "backend": backend,
            "language": language,
            "client_sample_rate": client_sample_rate,
            "server_sample_rate": cfg.sample_rate,
            "raw_pcm_bytes": len(captured_pcm),
            "audio_raw_pcm_path": str(raw_pcm_path),
            "audio_wav_path": str(wav_path),
            "events_path": str(events_path),
            "final_transcripts": [
                ev["text"]
                for ev in transcript_events
                if ev.get("type") == "final" and ev.get("text")
            ],
            "status": "completed",
        }

        write_json(metadata_path, final_metadata)

        log.info(
            f"AUDIO_LOG_END | session_id={session_id} client={ws.client} "
            f"wall_duration_sec={wall_duration_sec:.3f} "
            f"audio_duration_sec={audio_duration_sec:.3f} "
            f"wav={wav_path}"
        )

        log.info(f"WS session closed for {ws.client}")


def _log_transcript(ev_type: str, text: str, ttfb_ms, language: str, client):
    if ev_type == "partial":
        log.debug(f"PARTIAL | lang={language} client={client} | {text}")
    elif ev_type == "final":
        ttfb_str = f" ttfb={ttfb_ms}ms" if ttfb_ms is not None else ""
        log.info(f"FINAL   | lang={language} client={client}{ttfb_str} | {text}")
