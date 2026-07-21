docker run --gpus all -d --name nemotron_server -p 8003:8003 -e MODEL_NAME=/srv/nemotron-3.5-asr-streaming-0.6b.nemo nemotron_3.5
docker run --gpus all -d --name nemotron_en --restart unless-stopped -p 8003:8003 -v /home/CORP/re_nikitav/nemotron_finetuned/ft_models:/srv/models -e MODEL_NAME=/srv/models/finetuned_nemotron_final.nemo nemotron_3.5
i am getting this 
(base) root@EC03-E01-AICOE1:/home/CORP/re_nikitav/nemotron_finetuned_updated# docker logs 80f2bf70e7f2

==========
== CUDA ==
==========

CUDA Version 12.4.1

Container image Copyright (c) 2016-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

This container image and its contents are governed by the NVIDIA Deep Learning Container License.
By pulling and using the container, you accept the terms and conditions of this license:
https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license

A copy of this license is made available in this container at /NGC-DL-CONTAINER-LICENSE for your convenience.

DEBUG: Startup cfg.model_name='/srv/models/finetuned_nemotron_final.nemo' cfg.asr_backend='nemotron'
INFO:     Started server process [1]
INFO:     Waiting for application startup.
2026-07-21 13:55:05,123 | INFO | asr_server | Server startup initiated
2026-07-21 13:55:05,123 | INFO | asr_server | Preloading ASR engines...
2026-07-21 13:55:05,123 | INFO | asr_server | Initializing engine: nemotron (/srv/models/finetuned_nemotron_final.nemo)
2026-07-21 13:55:18,525 | WARNING | nv_one_logger.api.config | OneLogger: Setting error_handling_strategy to DISABLE_QUIETLY_AND_REPORT_METRIC_ERROR for rank (rank=0) with OneLogger disabled. To override: explicitly set error_handling_strategy parameter.
2026-07-21 13:55:18,536 | INFO | nv_one_logger.exporter.export_config_manager | Final configuration contains 0 exporter(s)
2026-07-21 13:55:18,536 | WARNING | nv_one_logger.training_telemetry.api.training_telemetry_provider | No exporters were provided. This means that no telemetry data will be collected.
2026-07-21 13:55:22,055 | ERROR | asr_server | Failed to preload 'nemotron'
Traceback (most recent call last):
  File "/srv/app/main.py", line 142, in preload_engines
    load_sec = engine.load()
               ^^^^^^^^^^^^^
  File "/srv/app/asr_engines/nemotron_asr.py", line 107, in load
    self.model = nemo_asr.models.EncDecRNNTBPEModelWithPrompt.restore_from(
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/nemo/collections/asr/models/rnnt_bpe_models_prompt.py", line 132, in restore_from
INFO:     Application startup complete.
    return EncDecRNNTBPEModel.restore_from(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/nemo/core/classes/modelPT.py", line 483, in restore_from
    raise FileNotFoundError(f"Can't find {restore_path}")
FileNotFoundError: Can't find /srv/models/finetuned_nemotron_final.nemo
2026-07-21 13:55:22,058 | INFO | asr_server | All engines preloaded. Available: []
INFO:     Uvicorn running on http://0.0.0.0:8002 (Press CTRL+C to quit)


#app/main.py-
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



#app/config.py-
from dataclasses import dataclass, replace
import os


@dataclass(frozen=True)
class Config:
    # ── Core ASR selection ──────────────────────────────────────────────────
    asr_backend: str = os.getenv("ASR_BACKEND", "nemotron")
    model_name: str = os.getenv("MODEL_NAME", "")
    device: str = os.getenv("DEVICE", "cuda")

    # ── Common audio ────────────────────────────────────────────────────────
    sample_rate: int = int(os.getenv("SAMPLE_RATE", "16000"))

    # ── Common VAD ──────────────────────────────────────────────────────────
    vad_frame_ms: int = int(os.getenv("VAD_FRAME_MS", "20"))
    vad_start_margin: float = float(os.getenv("VAD_START_MARGIN", "1.5"))
    vad_min_noise_rms: float = float(os.getenv("VAD_MIN_NOISE_RMS", "0.0015"))
    pre_speech_ms: int = int(os.getenv("PRE_SPEECH_MS", "700"))

    # ── Global safety constraint ────────────────────────────────────────────
    max_utt_ms: int = int(os.getenv("MAX_UTT_MS", "30000"))

    log_level: str = os.getenv("LOG_LEVEL", "INFO")



MODEL_MAP = {
    # Points to the .nemo file saved during Docker build.
    # Falls back to HuggingFace download if running outside Docker.
    "nemotron": os.getenv(
        "MODEL_NAME",
        "/srv/nemotron-3.5-asr-streaming-0.6b.nemo"
    ),
}


def load_config() -> Config:
    cfg = Config()

    if not cfg.model_name:
        cfg = replace(cfg, model_name=MODEL_MAP.get(cfg.asr_backend, ""))

    print(
        f"DEBUG: Startup cfg.model_name='{cfg.model_name}' "
        f"cfg.asr_backend='{cfg.asr_backend}'"
    )

    return cfg

#app/asr_engines/nemotron.py-
import time
from dataclasses import dataclass
from typing import Optional, Any, Tuple
import os
import numpy as np
import torch
from omegaconf import OmegaConf

from app.asr_engines.base import ASREngine, EngineCaps


def safe_text(h: Any) -> str:
    if h is None:
        return ""
    if isinstance(h, str):
        return h
    if isinstance(h, (list, tuple)) and len(h) > 0:
        return safe_text(h[0])
    if hasattr(h, "text"):
        try:
            return h.text or ""
        except Exception:
            return ""
    try:
        return str(h)
    except Exception:
        return ""


@dataclass
class StreamTimings:
    preproc_sec: float = 0.0
    infer_sec: float = 0.0
    flush_sec: float = 0.0


class NemotronStreamingASR(ASREngine):

    caps = EngineCaps(
        streaming=True,
        partials=True,
        ttft_meaningful=True,
    )

    def __init__(
        self,
        model_name: str,
        device: str,
        sample_rate: int,
    ):
        self.model_name = model_name
        self.device = device
        self.sr = sample_rate

        self.context_right = int(os.getenv("CONTEXT_RIGHT", "1"))
        self.end_silence_ms = int(os.getenv("NEMO_END_SILENCE_MS", "700"))
        self.min_utt_ms = int(os.getenv("NEMO_MIN_UTT_MS", "200"))
        self.finalize_pad_ms = int(os.getenv("FINALIZE_PAD_MS", "500"))

        self.model = None

        self.shift_frames: int = 0
        self.pre_cache_frames: int = 0
        self.hop_samples: int = 0
        self.drop_extra: int = 0
        self._frame_stride_sec: float = 0.01

    @property
    def chunk_samples(self) -> int:
        if self.shift_frames <= 0 or self.hop_samples <= 0:
            return int(0.08 * self.sr)
        return int(self.shift_frames * self.hop_samples)

    def _to_device(self, x: torch.Tensor) -> torch.Tensor:
        if self.device == "cuda":
            return x.cuda(non_blocking=True)
        return x.cpu()

    def _move_cache_to_device(
        self, cache: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        c0, c1, c2 = cache
        return (self._to_device(c0), self._to_device(c1), self._to_device(c2))

    def set_language(self, language: str):
        """
        Set language prompt for the next session.
        Called per WebSocket connection before session starts.
        NOT thread-safe across concurrent sessions with different languages.
        """
        if self.model is None:
            return
        try:
            self.model.set_inference_prompt(language)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(
                f"set_inference_prompt('{language}') failed: {e}"
            )

    def load(self) -> float:
        import nemo.collections.asr as nemo_asr

        t0 = time.time()

        if self.model_name.endswith(".nemo"):
            self.model = nemo_asr.models.EncDecRNNTBPEModelWithPrompt.restore_from(
                self.model_name, map_location="cpu"
            )
        else:
            self.model = nemo_asr.models.EncDecRNNTBPEModelWithPrompt.from_pretrained(
                self.model_name, map_location="cpu"
            )

        if self.device == "cuda":
            self.model = self.model.cuda()
        else:
            self.model = self.model.cpu()

        try:
            self.model.encoder.set_default_att_context_size(
                [70, int(self.context_right)]
            )
        except Exception:
            self.model.encoder.set_default_att_context_size(
                (70, int(self.context_right))
            )

        self.model.change_decoding_strategy(
            decoding_cfg=OmegaConf.create({
                "strategy": "greedy",
                "greedy": {
                    "max_symbols": 10,
                    "loop_labels": False,
                    "use_cuda_graph_decoder": False,
                }
            })
        )

        self.model.eval()

        try:
            self.model.preprocessor.featurizer.dither = 0.0
        except Exception:
            pass

        scfg = self.model.encoder.streaming_cfg
        self.shift_frames = (
            scfg.shift_size[1]
            if isinstance(scfg.shift_size, (list, tuple))
            else scfg.shift_size
        )
        pre_cache = scfg.pre_encode_cache_size
        self.pre_cache_frames = (
            pre_cache[1]
            if isinstance(pre_cache, (list, tuple))
            else pre_cache
        )
        self.drop_extra = int(getattr(scfg, "drop_extra_pre_encoded", 0))

        self._frame_stride_sec = float(
            self.model.cfg.preprocessor.get("window_stride", 0.01)
        )
        self.hop_samples = int(self._frame_stride_sec * self.sr)

        self._warmup()

        return time.time() - t0

    @torch.inference_mode()
    def _warmup(self):
        try:
            sess = self.new_session(max_buffer_ms=3000)
            silence = np.zeros(int(self.sr * 1.0), dtype=np.float32)
            pcm16 = (np.clip(silence, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
            sess.accept_pcm16(pcm16)
            _ = sess.finalize(pad_ms=400)
        except Exception:
            pass

    def new_session(self, max_buffer_ms: int):
        return StreamingSession(self, max_buffer_ms=max_buffer_ms)

    @torch.inference_mode()
    def stream_transcribe(
        self,
        audio_f32: np.ndarray,
        cache: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        prev_hyp: Any,
        prev_pred_out: Any,
        emitted_frames: int,
        force_flush: bool = False,
    ):
        assert self.model is not None
        timings = StreamTimings()

        t0 = time.perf_counter()
        audio_tensor = torch.from_numpy(audio_f32).unsqueeze(0)
        audio_tensor = self._to_device(audio_tensor)
        audio_len = torch.tensor([len(audio_f32)], device=audio_tensor.device)
        mel, mel_len = self.model.preprocessor(
            input_signal=audio_tensor, length=audio_len
        )
        timings.preproc_sec += time.perf_counter() - t0

        available = int(mel.shape[-1]) - 1
        if available <= 0:
            return None, cache, prev_hyp, prev_pred_out, emitted_frames, timings

        enough = (available - emitted_frames) >= self.shift_frames

        if not enough and not force_flush:
            return None, cache, prev_hyp, prev_pred_out, emitted_frames, timings

        if emitted_frames == 0:
            chunk_start = 0
            chunk_end   = min(self.shift_frames, available)
            drop_extra  = 0
        else:
            chunk_start = max(0, emitted_frames - self.pre_cache_frames)
            chunk_end   = min(emitted_frames + self.shift_frames, available)
            drop_extra  = self.drop_extra

        chunk_mel = mel[:, :, chunk_start:chunk_end]
        chunk_len = torch.tensor(
            [chunk_mel.shape[-1]], device=chunk_mel.device
        )

        cache = self._move_cache_to_device(cache)

        t1 = time.perf_counter()
        (prev_pred_out, texts, cache0, cache1, cache2, prev_hyp) = (
            self.model.conformer_stream_step(
                processed_signal=chunk_mel,
                processed_signal_length=chunk_len,
                cache_last_channel=cache[0],
                cache_last_time=cache[1],
                cache_last_channel_len=cache[2],
                keep_all_outputs=False,
                previous_hypotheses=prev_hyp,
                previous_pred_out=prev_pred_out,
                drop_extra_pre_encoded=drop_extra,
                return_transcription=True,
            )
        )
        timings.infer_sec += time.perf_counter() - t1

        new_cache = (cache0, cache1, cache2)

        if emitted_frames < available:
            emitted_frames = min(emitted_frames + self.shift_frames, available)

        text = safe_text(texts).strip() if texts is not None else ""

        return text, new_cache, prev_hyp, prev_pred_out, emitted_frames, timings


class StreamingSession:

    def __init__(self, engine: NemotronStreamingASR, max_buffer_ms: int):
        self.engine = engine
        self.max_buffer_samples = int(engine.sr * (max_buffer_ms / 1000.0))

        self.audio = np.array([], dtype=np.float32)
        self.cache = None
        self.prev_hyp = None
        self.prev_pred = None
        self.emitted_frames = 0

        self.current_text = ""
        self.last_final_text = ""

        self.utt_preproc = 0.0
        self.utt_infer = 0.0
        self.utt_flush = 0.0
        self.chunks = 0

        self._trimmed_since_last_step = False

        self.reset_stream_state()

    def reset_stream_state(self):
        cache = self.engine.model.encoder.get_initial_cache_state(batch_size=1)
        self.cache = self.engine._move_cache_to_device(
            (cache[0], cache[1], cache[2])
        )
        self.prev_hyp = None
        self.prev_pred = None
        self.emitted_frames = 0
        self.current_text = ""
        self.audio = np.array([], dtype=np.float32)
        self.utt_preproc = 0.0
        self.utt_infer = 0.0
        self.utt_flush = 0.0
        self.chunks = 0
        self._trimmed_since_last_step = False

    def accept_pcm16(self, pcm16: bytes):
        x = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
        self.audio = np.concatenate([self.audio, x])
        if len(self.audio) > self.max_buffer_samples:
            self.audio = self.audio[-self.max_buffer_samples:]
            self._trimmed_since_last_step = True

    def backlog_ms(self) -> int:
        return int(1000 * (len(self.audio) / self.engine.sr))

    def _is_new_text(self, new_text: Optional[str]) -> bool:
        if new_text is None:
            return False
        new = new_text.strip()
        old = self.current_text.strip()
        if new == "":
            return False
        if new == old:
            return False
        if new.startswith(old):
            return True
        if old.startswith(new):
            return False
        return True

    def step_if_ready(self) -> Optional[str]:
        if self._trimmed_since_last_step and self.emitted_frames > 0:
            cache = self.engine.model.encoder.get_initial_cache_state(batch_size=1)
            self.cache = self.engine._move_cache_to_device(
                (cache[0], cache[1], cache[2])
            )
            self.prev_hyp = None
            self.prev_pred = None
            self.emitted_frames = 0
            self._trimmed_since_last_step = False

        text, self.cache, self.prev_hyp, self.prev_pred, self.emitted_frames, t = (
            self.engine.stream_transcribe(
                audio_f32=self.audio,
                cache=self.cache,
                prev_hyp=self.prev_hyp,
                prev_pred_out=self.prev_pred,
                emitted_frames=self.emitted_frames,
                force_flush=False,
            )
        )

        self.utt_preproc += t.preproc_sec
        self.utt_infer += t.infer_sec

        if not self._is_new_text(text):
            return None

        self.current_text = text.strip()
        self.chunks += 1
        return self.current_text

    def finalize(self, pad_ms: int) -> str:
        pad = np.zeros(
            int(self.engine.sr * (pad_ms / 1000.0)), dtype=np.float32
        )
        self.audio = np.concatenate([self.audio, pad])

        t0 = time.perf_counter()
        text, self.cache, self.prev_hyp, self.prev_pred, self.emitted_frames, t = (
            self.engine.stream_transcribe(
                audio_f32=self.audio,
                cache=self.cache,
                prev_hyp=self.prev_hyp,
                prev_pred_out=self.prev_pred,
                emitted_frames=self.emitted_frames,
                force_flush=True,
            )
        )
        self.utt_preproc += t.preproc_sec
        self.utt_infer += t.infer_sec
        self.utt_flush += time.perf_counter() - t0

        if text:
            self.current_text = text.strip()

        final = self.current_text.strip()
        self.last_final_text = (
            (self.last_final_text + " " + final).strip()
            if final else self.last_final_text
        )

        self.reset_stream_state()
        return final

#app/streaming_session.py-
import time
from app.vad import AdaptiveEnergyVAD


class StreamingSession:
    def __init__(self, engine, cfg):
        self.engine = engine
        self.cfg = cfg
        self.vad = AdaptiveEnergyVAD(cfg.sample_rate, cfg.vad_frame_ms, cfg.vad_start_margin, cfg.vad_min_noise_rms, cfg.pre_speech_ms)
        self.session = engine.new_session(max_buffer_ms=cfg.max_utt_ms)
        self.frame_bytes = int(cfg.sample_rate * cfg.vad_frame_ms / 1000) * 2
        self.raw_buf = bytearray()
        self.utt_started = False
        self.utt_audio_ms = 0
        self.t_utt_start = None
        self.t_first_partial = None
        self.silence_ms = 0
        self.last_partial = ""

    def process_chunk(self, pcm: bytes) -> list:
        events = []
        self.raw_buf.extend(pcm)
        while len(self.raw_buf) >= self.frame_bytes:
            frame = bytes(self.raw_buf[: self.frame_bytes])
            del self.raw_buf[: self.frame_bytes]
            is_speech, pre = self.vad.push_frame(frame)
            self.silence_ms = 0 if is_speech else self.silence_ms + self.cfg.vad_frame_ms

            if pre and not self.utt_started:
                self.utt_started = True
                self.utt_audio_ms = 0
                self.t_utt_start = time.time()
                self.t_first_partial = None
                self.last_partial = ""
                self.session.accept_pcm16(pre)

            if not self.utt_started:
                continue

            self.session.accept_pcm16(frame)
            self.utt_audio_ms += self.cfg.vad_frame_ms

            if self.engine.caps.partials:
                text = self.session.step_if_ready()
                if text:
                    if self.t_first_partial is None:
                        self.t_first_partial = time.time()
                    self.last_partial = text
                    ttfb_ms = int((self.t_first_partial - self.t_utt_start) * 1000)
                    events.append(("partial", text, ttfb_ms))

            if not is_speech and self.utt_audio_ms >= self.engine.min_utt_ms and self.silence_ms >= self.engine.end_silence_ms:
                final = self.session.finalize(self.engine.finalize_pad_ms)
                if not final and self.last_partial:
                    final = self.last_partial
                if final:
                    ttfb_ms = int((self.t_first_partial - self.t_utt_start) * 1000) if self.t_first_partial else None
                    events.append(("final", final, ttfb_ms))
                self.reset()
        return events

    def flush(self) -> list:
        events = []
        if not self.utt_started:
            return events
        final = self.session.finalize(self.engine.finalize_pad_ms)
        if not final and self.last_partial:
            final = self.last_partial
        if final:
            ttfb_ms = int((self.t_first_partial - self.t_utt_start) * 1000) if self.t_first_partial else None
            events.append(("final", final, ttfb_ms))
        self.reset()
        return events

    def reset(self):
        self.vad.reset()
        self.utt_started = False
        self.utt_audio_ms = 0
        self.silence_ms = 0
        self.last_partial = ""


#app/factory.py-
from app.config import Config
from app.asr_engines.nemotron_asr import NemotronStreamingASR


def build_engine(cfg: Config):
    """
    Instantiate and return the correct ASR engine from config.
    Does NOT call engine.load() — that happens at startup.
    """
    if cfg.asr_backend == "nemotron":
        return NemotronStreamingASR(
            model_name=cfg.model_name,
            device=cfg.device,
            sample_rate=cfg.sample_rate,
        )

    raise ValueError(f"Unsupported ASR_BACKEND='{cfg.asr_backend}'")
