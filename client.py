#!/usr/bin/env python3
"""
Python 3.11 Nemotron ASR client.

Supported modes:
1. Live microphone streaming over WebSocket
2. WAV file streaming over WebSocket
3. OpenAI-compatible REST transcription
4. Health check

Long-call handling:
- Cloud/service connection limit: 240 seconds
- Soft WebSocket rotation: 180 seconds
- Hard WebSocket rotation: 210 seconds
- Microphone remains active while reconnecting
- Automatically reconnects after unexpected disconnects
"""

from __future__ import annotations

import argparse
import asyncio
import json
import mimetypes
import re
import sys
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import websockets
from websockets.exceptions import ConnectionClosed


# Local:
# SERVER_URL = "ws://localhost:8002/asr/realtime-custom-vad"

SERVER_URL = (
    "wss://nemotron-3-5-150916788856.us-central1.run.app/"
    "asr/realtime-custom-vad"
)

SAMPLE_RATE = 16000
CHUNK_MS = 100
CHUNK_BYTES = int(SAMPLE_RATE * CHUNK_MS / 1000) * 2

# Rotate before the 240-second platform limit.
DEFAULT_ROTATE_AFTER = 180.0
DEFAULT_HARD_ROTATE_AFTER = 210.0
DEFAULT_RECONNECT_DELAY = 1.0
DEFAULT_EOF_WAIT_SECONDS = 20.0

GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"

_LANG_TAG_RE = re.compile(r"<[a-z]{2}-[A-Z]{2}>\s*")


@dataclass
class SessionState:
    started_at: float = field(default_factory=time.monotonic)

    ended_event: asyncio.Event = field(
        default_factory=asyncio.Event
    )
    done_event: asyncio.Event = field(
        default_factory=asyncio.Event
    )
    final_event: asyncio.Event = field(
        default_factory=asyncio.Event
    )

    last_partial: str = ""
    active_partial: bool = False
    eof_requested: bool = False

    close_code: Optional[int] = None
    close_reason: str = ""
    error: Optional[str] = None


def clean_text(text: str) -> str:
    return _LANG_TAG_RE.sub("", text or "").strip()


def print_partial(text: str) -> None:
    sys.stdout.write(
        f"\r{YELLOW}[partial]{RESET} {text}    "
    )
    sys.stdout.flush()


def print_final(text: str, ttfb_ms=None) -> None:
    ttfb_text = ""

    if ttfb_ms is not None:
        ttfb_text = (
            f"  {DIM}(TTFB {ttfb_ms}ms){RESET}"
        )

    sys.stdout.write(
        f"\r{GREEN}{BOLD}[final]  {RESET}"
        f"{GREEN}{text}{RESET}{ttfb_text}\n"
    )
    sys.stdout.flush()


def print_info(message: str) -> None:
    print(f"{CYAN}[info]{RESET} {message}")


def print_warning(message: str) -> None:
    print(f"{YELLOW}[warn]{RESET} {message}")


def print_error(message: str) -> None:
    print(f"{RED}[error]{RESET} {message}")


def websocket_options() -> dict:
    """
    WebSocket heartbeat and connection settings.

    The original client used ping_interval=None, which disabled
    WebSocket keepalive.
    """
    return {
        "ping_interval": 20,
        "ping_timeout": 60,
        "close_timeout": 10,
        "open_timeout": 30,
        "max_size": None,
    }


def websocket_to_http_base(url: str) -> str:
    parsed = urlparse(url)

    if not parsed.netloc:
        raise ValueError(f"Invalid URL: {url}")

    if parsed.scheme == "wss":
        scheme = "https"
    elif parsed.scheme == "ws":
        scheme = "http"
    elif parsed.scheme in {"http", "https"}:
        scheme = parsed.scheme
    else:
        raise ValueError(
            f"Unsupported URL scheme: {parsed.scheme}"
        )

    return f"{scheme}://{parsed.netloc}"


def openai_transcription_url(url: str) -> str:
    return (
        f"{websocket_to_http_base(url)}"
        "/v1/audio/transcriptions"
    )


def validate_rotation_settings(
    rotate_after: float,
    hard_rotate_after: float,
) -> None:
    if rotate_after <= 0:
        raise ValueError(
            "--rotate-after must be greater than zero"
        )

    if hard_rotate_after <= rotate_after:
        raise ValueError(
            "--hard-rotate-after must be greater than "
            "--rotate-after"
        )

    if hard_rotate_after >= 230:
        raise ValueError(
            "--hard-rotate-after must be below 230 seconds "
            "because the connection limit is 240 seconds"
        )


async def receive_loop(
    ws,
    state: SessionState,
) -> None:
    """
    Receive ASR events from one WebSocket session.

    Important:
    An interrupted partial is not printed as a final transcript.
    """
    try:
        async for raw_message in ws:
            if isinstance(raw_message, bytes):
                continue

            try:
                message = json.loads(raw_message)
            except json.JSONDecodeError:
                print_warning(
                    f"Ignoring non-JSON message: "
                    f"{raw_message!r}"
                )
                continue

            event_type = str(
                message.get("type", "")
            )

            text = clean_text(
                str(message.get("text", ""))
            )

            ttfb_ms = message.get("t_start")

            if event_type == "partial":
                if text:
                    state.last_partial = text
                    state.active_partial = True
                    print_partial(text)

            elif event_type == "final":
                if text:
                    print_final(text, ttfb_ms)

                state.last_partial = ""
                state.active_partial = False

                # Used to perform a safe session rotation after
                # the current utterance has completed.
                state.final_event.set()

            elif event_type == "done":
                if state.last_partial:
                    print_warning(
                        "Server returned done while a partial "
                        "remained unconfirmed: "
                        f"{state.last_partial}"
                    )

                state.done_event.set()
                break

            elif event_type == "error":
                state.error = (
                    text
                    or json.dumps(
                        message,
                        ensure_ascii=False,
                    )
                )

                print_error(
                    f"Server error: {state.error}"
                )

            else:
                # Optional server events such as ready/config_ack.
                if text:
                    print_info(
                        f"Server event {event_type}: {text}"
                    )

    except ConnectionClosed as exc:
        state.close_code = exc.code
        state.close_reason = exc.reason or ""

        elapsed = (
            time.monotonic() - state.started_at
        )

        if not state.eof_requested:
            print_warning(
                "WebSocket receive connection closed "
                f"after {elapsed:.1f}s; "
                f"code={exc.code}, "
                f"reason={exc.reason or 'none'}"
            )

        if state.last_partial:
            print_warning(
                "Unconfirmed partial: "
                f"{state.last_partial}"
            )

    except asyncio.CancelledError:
        raise

    except Exception as exc:
        state.error = (
            f"{type(exc).__name__}: {exc}"
        )

        elapsed = (
            time.monotonic() - state.started_at
        )

        print_error(
            f"Receive loop failed after "
            f"{elapsed:.1f}s: {state.error}"
        )

        if state.last_partial:
            print_warning(
                "Unconfirmed partial: "
                f"{state.last_partial}"
            )

    finally:
        state.ended_event.set()


async def open_streaming_session(
    url: str,
    language: str,
    session_number: int,
):
    print_info(
        f"Opening WebSocket session #{session_number}"
    )

    ws = await websockets.connect(
        url,
        **websocket_options(),
    )

    state = SessionState()

    await ws.send(
        json.dumps(
            {
                "backend": "nemotron",
                "sample_rate": SAMPLE_RATE,
                "language": language,
            }
        )
    )

    receive_task = asyncio.create_task(
        receive_loop(ws, state)
    )

    return ws, state, receive_task


async def stop_receive_task(
    receive_task: asyncio.Task,
) -> None:
    if not receive_task.done():
        receive_task.cancel()

    try:
        await receive_task
    except asyncio.CancelledError:
        pass
    except Exception as exc:
        print_warning(
            f"Receive-task cleanup error: {exc}"
        )


async def graceful_session_end(
    ws,
    state: SessionState,
    receive_task: asyncio.Task,
    wait_seconds: float = DEFAULT_EOF_WAIT_SECONDS,
) -> None:
    """
    Flush the current VAD/ASR utterance before reconnecting.
    """
    state.eof_requested = True

    try:
        await ws.send(
            json.dumps({"type": "eof"})
        )

    except ConnectionClosed as exc:
        print_warning(
            "Could not send EOF because the WebSocket "
            "was already closed: "
            f"code={exc.code}, "
            f"reason={exc.reason or 'none'}"
        )

    except Exception as exc:
        print_warning(
            f"Could not send EOF: "
            f"{type(exc).__name__}: {exc}"
        )

    try:
        await asyncio.wait_for(
            state.done_event.wait(),
            timeout=wait_seconds,
        )

    except asyncio.TimeoutError:
        print_warning(
            "No done event received within "
            f"{wait_seconds:.1f}s; closing session"
        )

    try:
        await ws.close(
            code=1000,
            reason="Client session rotation",
        )
    except Exception:
        pass

    await stop_receive_task(receive_task)


def load_wav_as_16khz_mono(path: str):
    import numpy as np

    wav_path = Path(path)

    if not wav_path.exists():
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    with wave.open(str(wav_path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        original_sample_rate = (
            wav_file.getframerate()
        )
        frame_count = wav_file.getnframes()
        raw_audio = wav_file.readframes(
            frame_count
        )

    if sample_width != 2:
        raise ValueError(
            "WebSocket file mode supports only "
            "16-bit PCM WAV files. "
            f"Received {sample_width * 8}-bit audio."
        )

    audio_i16 = np.frombuffer(
        raw_audio,
        dtype=np.int16,
    )

    if channels > 1:
        usable_samples = (
            len(audio_i16)
            - len(audio_i16) % channels
        )

        audio_i16 = (
            audio_i16[:usable_samples]
            .reshape(-1, channels)
            .astype(np.float32)
            .mean(axis=1)
            .clip(-32768, 32767)
            .astype(np.int16)
        )

    if original_sample_rate != SAMPLE_RATE:
        print_info(
            f"Resampling {original_sample_rate}Hz "
            f"to {SAMPLE_RATE}Hz"
        )

        try:
            import resampy
        except ImportError as exc:
            raise RuntimeError(
                "Install resampy with: "
                "py -3.11 -m pip install resampy"
            ) from exc

        audio_f32 = (
            audio_i16.astype(np.float32)
            / 32768.0
        )

        audio_f32 = resampy.resample(
            audio_f32,
            original_sample_rate,
            SAMPLE_RATE,
        )

        audio_i16 = (
            np.clip(
                audio_f32,
                -1.0,
                1.0,
            )
            * 32767.0
        ).astype(np.int16)

    return {
        "path": wav_path,
        "audio_i16": audio_i16,
        "channels": channels,
        "sample_width": sample_width,
        "original_sample_rate": original_sample_rate,
        "frame_count": frame_count,
    }


async def run_file(
    path: str,
    language: str,
    realtime: bool,
    url: str,
    rotate_after: float,
    hard_rotate_after: float,
    reconnect_delay: float,
    reconnect_overlap_ms: int,
) -> None:
    validate_rotation_settings(
        rotate_after,
        hard_rotate_after,
    )

    wav_data = load_wav_as_16khz_mono(path)

    wav_path = wav_data["path"]
    audio_i16 = wav_data["audio_i16"]

    original_sample_rate = wav_data[
        "original_sample_rate"
    ]
    channels = wav_data["channels"]
    sample_width = wav_data["sample_width"]
    frame_count = wav_data["frame_count"]

    audio_seconds = (
        len(audio_i16) / SAMPLE_RATE
    )

    print_info(f"File: {wav_path.name}")

    print_info(
        "Original audio: "
        f"{original_sample_rate}Hz "
        f"{channels}ch "
        f"{sample_width * 8}bit "
        f"{frame_count / original_sample_rate:.1f}s"
    )

    print_info(
        "Streaming audio: "
        f"{SAMPLE_RATE}Hz mono 16bit "
        f"{audio_seconds:.1f}s"
    )

    print_info(f"Language: {language}")
    print_info(
        f"Realtime simulation: {realtime}"
    )

    print_info(
        "Session rotation: "
        f"soft={rotate_after:.0f}s, "
        f"hard={hard_rotate_after:.0f}s"
    )

    print_info(f"Connecting to {url}\n")

    raw_bytes = audio_i16.tobytes()

    chunks = [
        raw_bytes[index:index + CHUNK_BYTES]
        for index in range(
            0,
            len(raw_bytes),
            CHUNK_BYTES,
        )
    ]

    overlap_chunks = max(
        0,
        reconnect_overlap_ms // CHUNK_MS,
    )

    file_started_at = time.monotonic()
    chunk_index = 0
    session_number = 0

    while chunk_index < len(chunks):
        session_number += 1

        try:
            ws, state, receive_task = (
                await open_streaming_session(
                    url,
                    language,
                    session_number,
                )
            )

        except Exception as exc:
            print_error(
                "Connection failed: "
                f"{type(exc).__name__}: {exc}"
            )

            await asyncio.sleep(
                reconnect_delay
            )
            continue

        rotation_requested = False
        rotation_reason: Optional[str] = None
        unexpected_disconnect = False

        try:
            while chunk_index < len(chunks):
                if (
                    state.ended_event.is_set()
                    and not state.done_event.is_set()
                ):
                    unexpected_disconnect = True
                    break

                session_age = (
                    time.monotonic()
                    - state.started_at
                )

                if session_age >= hard_rotate_after:
                    rotation_reason = (
                        "hard safety rotation"
                    )
                    break

                if (
                    session_age >= rotate_after
                    and not rotation_requested
                ):
                    if not state.active_partial:
                        rotation_reason = (
                            "soft rotation while idle"
                        )
                        break

                    rotation_requested = True
                    state.final_event.clear()

                    print_info(
                        "Rotation requested; waiting for "
                        "the active utterance to produce "
                        "a final result"
                    )

                if (
                    rotation_requested
                    and state.final_event.is_set()
                ):
                    rotation_reason = (
                        "soft rotation after final"
                    )
                    break

                chunk = chunks[chunk_index]

                try:
                    await ws.send(chunk)

                except ConnectionClosed as exc:
                    print_warning(
                        "File send disconnected: "
                        f"code={exc.code}, "
                        f"reason={exc.reason or 'none'}"
                    )

                    unexpected_disconnect = True
                    break

                chunk_index += 1

                if realtime:
                    expected_elapsed = (
                        chunk_index
                        * CHUNK_MS
                        / 1000.0
                    )

                    actual_elapsed = (
                        time.monotonic()
                        - file_started_at
                    )

                    sleep_seconds = (
                        expected_elapsed
                        - actual_elapsed
                    )

                    if sleep_seconds > 0:
                        await asyncio.sleep(
                            sleep_seconds
                        )
                else:
                    await asyncio.sleep(0.001)

            if chunk_index >= len(chunks):
                print_info(
                    "File sent; sending EOF and "
                    "waiting for final results"
                )

                await graceful_session_end(
                    ws,
                    state,
                    receive_task,
                )
                break

            if rotation_reason:
                print_info(
                    f"{rotation_reason}; rotating "
                    f"WebSocket after "
                    f"{time.monotonic() - state.started_at:.1f}s"
                )

                await graceful_session_end(
                    ws,
                    state,
                    receive_task,
                )
                continue

            if unexpected_disconnect:
                await stop_receive_task(
                    receive_task
                )

                try:
                    await ws.close()
                except Exception:
                    pass

                if overlap_chunks > 0:
                    previous_index = chunk_index

                    chunk_index = max(
                        0,
                        chunk_index - overlap_chunks,
                    )

                    replay_ms = (
                        previous_index - chunk_index
                    ) * CHUNK_MS

                    print_warning(
                        "Reconnecting and replaying "
                        f"{replay_ms}ms of audio"
                    )

                await asyncio.sleep(
                    reconnect_delay
                )

        finally:
            if not receive_task.done():
                await stop_receive_task(
                    receive_task
                )

            try:
                await ws.close()
            except Exception:
                pass

    elapsed = (
        time.monotonic() - file_started_at
    )

    rtf = (
        elapsed / audio_seconds
        if audio_seconds > 0
        else 0.0
    )

    print_info(
        f"Done. Audio={audio_seconds:.1f}s "
        f"Wall={elapsed:.2f}s "
        f"RTF={rtf:.2f}x"
    )


async def run_mic(
    language: str,
    url: str,
    rotate_after: float,
    hard_rotate_after: float,
    reconnect_delay: float,
    max_call_seconds: float,
) -> None:
    validate_rotation_settings(
        rotate_after,
        hard_rotate_after,
    )

    try:
        import numpy as np
        import sounddevice as sd

    except ImportError as exc:
        raise RuntimeError(
            "Microphone mode requires numpy and "
            "sounddevice. Install with:\n"
            "py -3.11 -m pip install numpy sounddevice"
        ) from exc

    print_info(f"Connecting to {url}")
    print_info(f"Language: {language}")

    print_info(
        "Session rotation: "
        f"soft={rotate_after:.0f}s, "
        f"hard={hard_rotate_after:.0f}s"
    )

    print_info(
        "Speak into your microphone. "
        "Press Ctrl+C to stop.\n"
    )

    event_loop = asyncio.get_running_loop()

    # 300 chunks × 100ms = approximately 30 seconds
    # of buffering while reconnecting.
    audio_queue: asyncio.Queue[bytes] = (
        asyncio.Queue(maxsize=300)
    )

    dropped_chunks = 0

    def enqueue_audio(pcm: bytes) -> None:
        nonlocal dropped_chunks

        if audio_queue.full():
            try:
                audio_queue.get_nowait()
                dropped_chunks += 1
            except asyncio.QueueEmpty:
                pass

        try:
            audio_queue.put_nowait(pcm)
        except asyncio.QueueFull:
            dropped_chunks += 1

    def audio_callback(
        indata,
        frames,
        time_info,
        status,
    ) -> None:
        if status:
            event_loop.call_soon_threadsafe(
                print_warning,
                f"Audio input status: {status}",
            )

        mono_audio = np.clip(
            indata[:, 0],
            -1.0,
            1.0,
        )

        pcm = (
            mono_audio
            * 32767.0
        ).astype(np.int16).tobytes()

        event_loop.call_soon_threadsafe(
            enqueue_audio,
            pcm,
        )

    call_started_at = time.monotonic()
    session_number = 0

    # Keep a chunk here if send() fails before the
    # WebSocket accepts it.
    pending_pcm: Optional[bytes] = None

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=int(
            SAMPLE_RATE
            * CHUNK_MS
            / 1000
        ),
        callback=audio_callback,
    ):
        while True:
            total_call_age = (
                time.monotonic()
                - call_started_at
            )

            if (
                max_call_seconds > 0
                and total_call_age >= max_call_seconds
            ):
                print_info(
                    "Reached maximum call duration: "
                    f"{max_call_seconds:.0f}s"
                )
                break

            session_number += 1

            try:
                ws, state, receive_task = (
                    await open_streaming_session(
                        url,
                        language,
                        session_number,
                    )
                )

            except Exception as exc:
                print_error(
                    "Connection failed: "
                    f"{type(exc).__name__}: {exc}"
                )

                await asyncio.sleep(
                    reconnect_delay
                )
                continue

            rotation_requested = False
            rotation_reason: Optional[str] = None
            unexpected_disconnect = False

            try:
                while True:
                    total_call_age = (
                        time.monotonic()
                        - call_started_at
                    )

                    if (
                        max_call_seconds > 0
                        and total_call_age
                        >= max_call_seconds
                    ):
                        rotation_reason = (
                            "call complete"
                        )
                        break

                    if (
                        state.ended_event.is_set()
                        and not state.done_event.is_set()
                    ):
                        unexpected_disconnect = True
                        break

                    session_age = (
                        time.monotonic()
                        - state.started_at
                    )

                    if (
                        session_age
                        >= hard_rotate_after
                    ):
                        rotation_reason = (
                            "hard safety rotation"
                        )
                        break

                    if (
                        session_age >= rotate_after
                        and not rotation_requested
                    ):
                        if not state.active_partial:
                            rotation_reason = (
                                "soft rotation while idle"
                            )
                            break

                        rotation_requested = True
                        state.final_event.clear()

                        print_info(
                            "Rotation requested; waiting "
                            "for the active utterance to "
                            "produce a final result"
                        )

                    if (
                        rotation_requested
                        and state.final_event.is_set()
                    ):
                        rotation_reason = (
                            "soft rotation after final"
                        )
                        break

                    if pending_pcm is None:
                        try:
                            pending_pcm = (
                                await asyncio.wait_for(
                                    audio_queue.get(),
                                    timeout=0.25,
                                )
                            )

                        except asyncio.TimeoutError:
                            continue

                    try:
                        await ws.send(pending_pcm)
                        pending_pcm = None

                    except ConnectionClosed as exc:
                        print_warning(
                            "Microphone send disconnected: "
                            f"code={exc.code}, "
                            f"reason="
                            f"{exc.reason or 'none'}"
                        )

                        # Keep pending_pcm. It will be sent
                        # again after reconnection.
                        unexpected_disconnect = True
                        break

                if rotation_reason:
                    print_info(
                        f"{rotation_reason}; ending "
                        f"WebSocket session after "
                        f"{time.monotonic() - state.started_at:.1f}s"
                    )

                    await graceful_session_end(
                        ws,
                        state,
                        receive_task,
                    )

                    if rotation_reason == "call complete":
                        break

                    continue

                if unexpected_disconnect:
                    await stop_receive_task(
                        receive_task
                    )

                    try:
                        await ws.close()
                    except Exception:
                        pass

                    print_info(
                        "Reconnecting in "
                        f"{reconnect_delay:.1f}s; "
                        "microphone capture remains active"
                    )

                    await asyncio.sleep(
                        reconnect_delay
                    )
                    continue

            finally:
                if not receive_task.done():
                    await stop_receive_task(
                        receive_task
                    )

                try:
                    await ws.close()
                except Exception:
                    pass

    if dropped_chunks > 0:
        print_warning(
            f"Dropped {dropped_chunks} microphone "
            "chunks because the reconnect buffer "
            "became full"
        )

    print_info(
        "Microphone call finished after "
        f"{time.monotonic() - call_started_at:.1f}s"
    )


def run_openai_compatible_test(
    path: str,
    endpoint: str,
    model: str,
    language: str,
    response_format: str,
    prompt: Optional[str],
    temperature: Optional[float],
    api_key: Optional[str],
    request_timeout: float,
) -> None:
    """
    Send an OpenAI-compatible multipart transcription request.

    POST /v1/audio/transcriptions

    Multipart fields:
    - file
    - model
    - language
    - response_format
    - prompt, optional
    - temperature, optional
    """
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError(
            "OpenAI-compatible REST mode requires "
            "requests. Install with:\n"
            "py -3.11 -m pip install requests"
        ) from exc

    audio_path = Path(path)

    if not audio_path.exists():
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    mime_type = (
        mimetypes.guess_type(
            audio_path.name
        )[0]
        or "application/octet-stream"
    )

    form_data = {
        "model": model,
        "language": language,
        "response_format": response_format,
    }

    if prompt:
        form_data["prompt"] = prompt

    if temperature is not None:
        form_data["temperature"] = str(
            temperature
        )

    headers = {}

    if api_key:
        headers["Authorization"] = (
            f"Bearer {api_key}"
        )

    print_info(
        f"OpenAI-compatible endpoint: {endpoint}"
    )

    print_info(
        "Multipart fields: "
        f"file={audio_path.name}, "
        f"model={model}, "
        f"language={language}, "
        f"response_format={response_format}"
    )

    request_started_at = time.monotonic()

    with audio_path.open("rb") as audio_file:
        response = requests.post(
            endpoint,
            headers=headers,
            data=form_data,
            files={
                "file": (
                    audio_path.name,
                    audio_file,
                    mime_type,
                )
            },
            timeout=request_timeout,
        )

    latency_ms = (
        time.monotonic()
        - request_started_at
    ) * 1000.0

    content_type = response.headers.get(
        "content-type",
        "",
    )

    print_info(
        f"REST latency: {latency_ms:.0f}ms"
    )

    print_info(
        f"HTTP {response.status_code}; "
        f"content-type="
        f"{content_type or 'unknown'}"
    )

    if not response.ok:
        print_error(response.text)
        response.raise_for_status()

    if "application/json" in content_type.lower():
        try:
            print(
                json.dumps(
                    response.json(),
                    indent=2,
                    ensure_ascii=False,
                )
            )
            return

        except ValueError:
            pass

    print(response.text)


async def check_health(url: str) -> bool:
    import urllib.request

    try:
        http_base = websocket_to_http_base(
            url
        )

        health_url = f"{http_base}/health"

        with urllib.request.urlopen(
            health_url,
            timeout=10,
        ) as response:
            response_data = json.loads(
                response.read()
            )

        print_info(
            f"Server health: {response_data}"
        )

        return True

    except Exception as exc:
        print_warning(
            "Health check failed: "
            f"{type(exc).__name__}: {exc} "
            "(server may still be starting)"
        )

        return False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Python 3.11 Nemotron streaming and "
            "OpenAI-compatible ASR client"
        )
    )

    mode = parser.add_mutually_exclusive_group(
        required=True
    )

    mode.add_argument(
        "--mic",
        action="store_true",
        help="Stream microphone audio",
    )

    mode.add_argument(
        "--file",
        metavar="PATH",
        help="Stream a 16-bit PCM WAV file",
    )

    mode.add_argument(
        "--openai-file",
        metavar="PATH",
        help=(
            "Upload an audio file to the "
            "OpenAI-compatible endpoint"
        ),
    )

    mode.add_argument(
        "--health",
        action="store_true",
        help="Check the health endpoint",
    )

    parser.add_argument(
        "--language",
        default="en-US",
    )

    parser.add_argument(
        "--url",
        default=SERVER_URL,
        help="WebSocket endpoint",
    )

    parser.add_argument(
        "--realtime",
        action="store_true",
        help=(
            "Stream a WAV file at its actual "
            "real-time speed"
        ),
    )

    # Long-call options.
    parser.add_argument(
        "--rotate-after",
        type=float,
        default=DEFAULT_ROTATE_AFTER,
        help=(
            "Request a graceful rotation after "
            "this many seconds"
        ),
    )

    parser.add_argument(
        "--hard-rotate-after",
        type=float,
        default=DEFAULT_HARD_ROTATE_AFTER,
        help=(
            "Force session rotation after this "
            "many seconds"
        ),
    )

    parser.add_argument(
        "--reconnect-delay",
        type=float,
        default=DEFAULT_RECONNECT_DELAY,
    )

    parser.add_argument(
        "--reconnect-overlap-ms",
        type=int,
        default=500,
        help=(
            "Replay this much WAV audio after an "
            "unexpected disconnect"
        ),
    )

    parser.add_argument(
        "--max-call-seconds",
        type=float,
        default=0,
        help=(
            "Automatically stop microphone mode "
            "after N seconds; zero means unlimited"
        ),
    )

    # OpenAI-compatible options.
    parser.add_argument(
        "--openai-url",
        default=None,
        help=(
            "Full /v1/audio/transcriptions URL. "
            "By default it is derived from --url."
        ),
    )

    parser.add_argument(
        "--model",
        default=(
            "nemotron-3.5-asr-streaming-0.6b"
        ),
    )

    parser.add_argument(
        "--response-format",
        default="json",
        choices=[
            "json",
            "text",
            "verbose_json",
            "srt",
            "vtt",
        ],
    )

    parser.add_argument(
        "--prompt",
        default=None,
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--api-key",
        default=None,
    )

    parser.add_argument(
        "--request-timeout",
        type=float,
        default=300,
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.health:
            asyncio.run(
                check_health(args.url)
            )
            return

        if args.openai_file:
            endpoint = (
                args.openai_url
                or openai_transcription_url(
                    args.url
                )
            )

            run_openai_compatible_test(
                path=args.openai_file,
                endpoint=endpoint,
                model=args.model,
                language=args.language,
                response_format=(
                    args.response_format
                ),
                prompt=args.prompt,
                temperature=args.temperature,
                api_key=args.api_key,
                request_timeout=(
                    args.request_timeout
                ),
            )

            return

        asyncio.run(
            check_health(args.url)
        )

        if args.mic:
            asyncio.run(
                run_mic(
                    language=args.language,
                    url=args.url,
                    rotate_after=(
                        args.rotate_after
                    ),
                    hard_rotate_after=(
                        args.hard_rotate_after
                    ),
                    reconnect_delay=(
                        args.reconnect_delay
                    ),
                    max_call_seconds=(
                        args.max_call_seconds
                    ),
                )
            )

        else:
            asyncio.run(
                run_file(
                    path=args.file,
                    language=args.language,
                    realtime=args.realtime,
                    url=args.url,
                    rotate_after=(
                        args.rotate_after
                    ),
                    hard_rotate_after=(
                        args.hard_rotate_after
                    ),
                    reconnect_delay=(
                        args.reconnect_delay
                    ),
                    reconnect_overlap_ms=(
                        args.reconnect_overlap_ms
                    ),
                )
            )

    except KeyboardInterrupt:
        print_info("Stopped by user")

    except Exception as exc:
        print_error(
            f"{type(exc).__name__}: {exc}"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
