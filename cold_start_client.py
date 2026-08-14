#!/usr/bin/env python3

import argparse
import asyncio
import http.client
import json
import mimetypes
import os
import re
import ssl
import tempfile
import time
import uuid
import wave
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import websockets
from websockets.exceptions import ConnectionClosed


# =============================================================================
# CONFIG
# =============================================================================

SERVER_URL = (
    "wss://nemotron-3-5-150916788856."
    "us-central1.run.app/asr/realtime-custom-vad"
)

SAMPLE_RATE = 16000
CHUNK_MS = 100
CHUNK_BYTES = int(
    SAMPLE_RATE * CHUNK_MS / 1000
) * 2

_LANG_TAG_RE = re.compile(
    r"<[a-z]{2}-[A-Z]{2}>\s*"
)


# =============================================================================
# TIMING
# =============================================================================

def now_ns():
    return time.perf_counter_ns()


def elapsed_ms(start_ns, end_ns=None):
    if end_ns is None:
        end_ns = now_ns()

    return (
        end_ns - start_ns
    ) / 1_000_000.0


def clean_text(text):
    return _LANG_TAG_RE.sub(
        "",
        text or "",
    ).strip()


def ws_to_openai_url(ws_url):
    parsed = urlparse(ws_url)

    if parsed.scheme == "wss":
        scheme = "https"

    elif parsed.scheme == "ws":
        scheme = "http"

    else:
        scheme = parsed.scheme

    return (
        f"{scheme}://{parsed.netloc}"
        "/v1/audio/transcriptions"
    )


# =============================================================================
# RESULTS
# =============================================================================

@dataclass
class LatencyResult:
    connection_startup_ms: float
    connection_response_ms: float
    connection_transcription_ms: float

    e2e_ttfb_ms: float
    e2e_ttft_ms: float
    e2e_total_ms: float

    first_event_type: str = ""
    first_partial: str = ""
    first_final: str = ""


# =============================================================================
# PRINT RESULTS
# =============================================================================

def print_latency(title, result):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)

    print(
        f"Connection / startup       : "
        f"{result.connection_startup_ms:.2f} ms"
    )

    print(
        f"Connection -> response     : "
        f"{result.connection_response_ms:.2f} ms"
    )

    print(
        f"Connection -> transcription: "
        f"{result.connection_transcription_ms:.2f} ms"
    )

    print(
        f"E2E TTFB                   : "
        f"{result.e2e_ttfb_ms:.2f} ms"
    )

    print(
        f"E2E TTFT/TTFA              : "
        f"{result.e2e_ttft_ms:.2f} ms"
    )

    print(
        f"E2E TOTAL                  : "
        f"{result.e2e_total_ms:.2f} ms"
    )


def print_comparison(cold, warm, title):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)

    metrics = [
        (
            "Connection / startup",
            "connection_startup_ms",
        ),
        (
            "Connection -> response",
            "connection_response_ms",
        ),
        (
            "Connection -> transcription",
            "connection_transcription_ms",
        ),
        (
            "E2E TTFB",
            "e2e_ttfb_ms",
        ),
        (
            "E2E TTFT/TTFA",
            "e2e_ttft_ms",
        ),
        (
            "E2E TOTAL",
            "e2e_total_ms",
        ),
    ]

    for name, attribute in metrics:
        cold_value = getattr(
            cold,
            attribute,
        )

        warm_value = getattr(
            warm,
            attribute,
        )

        delta = (
            cold_value
            - warm_value
        )

        ratio = (
            cold_value / warm_value
            if warm_value > 0
            else 0
        )

        print(
            f"{name:29}: "
            f"cold={cold_value:.2f} ms | "
            f"warm={warm_value:.2f} ms | "
            f"delta={delta:.2f} ms | "
            f"{ratio:.2f}x"
        )


# =============================================================================
# MIC RECORDING FOR OPENAI REST
# =============================================================================

def record_microphone_to_wav(duration):
    """
    Record microphone audio BEFORE starting REST timing.

    Recording time is intentionally excluded from:
      Connection / startup
      Connection -> response
      Connection -> transcription
      E2E TTFB
      E2E TTFT/TTFA
      E2E TOTAL
    """

    try:
        import numpy as np
        import sounddevice as sd

    except ImportError:
        raise RuntimeError(
            "Microphone mode requires:\n"
            "py -3.11 -m pip install numpy sounddevice"
        )

    print()
    print("=" * 80)
    print("MICROPHONE RECORDING")
    print("=" * 80)

    print(
        f"[info] Recording for "
        f"{duration:.1f} seconds"
    )

    print(
        "[info] START SPEAKING NOW"
    )

    recording_start = time.perf_counter()

    audio = sd.rec(
        int(
            duration
            * SAMPLE_RATE
        ),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
    )

    sd.wait()

    recording_elapsed = (
        time.perf_counter()
        - recording_start
    )

    print(
        f"[info] Recording complete: "
        f"{recording_elapsed:.2f}s"
    )

    temp_file = tempfile.NamedTemporaryFile(
        suffix=".wav",
        delete=False,
    )

    temp_path = temp_file.name
    temp_file.close()

    with wave.open(
        temp_path,
        "wb",
    ) as wf:

        wf.setnchannels(1)

        wf.setsampwidth(2)

        wf.setframerate(
            SAMPLE_RATE
        )

        wf.writeframes(
            audio.tobytes()
        )

    print(
        f"[info] Temporary WAV: "
        f"{temp_path}"
    )

    print(
        "[info] Recording time is EXCLUDED "
        "from REST latency metrics"
    )

    return temp_path


# =============================================================================
# WAV LOADER FOR WEBSOCKET
# =============================================================================

def load_wav_16khz_mono(file_path):
    import numpy as np

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    with wave.open(
        str(path),
        "rb",
    ) as wf:

        channels = (
            wf.getnchannels()
        )

        sample_width = (
            wf.getsampwidth()
        )

        sample_rate = (
            wf.getframerate()
        )

        nframes = (
            wf.getnframes()
        )

        raw_audio = wf.readframes(
            nframes
        )

    if sample_width != 2:
        raise ValueError(
            "WebSocket file mode requires "
            "16-bit PCM WAV input."
        )

    audio = np.frombuffer(
        raw_audio,
        dtype=np.int16,
    )

    # -------------------------------------------------------------------------
    # MULTICHANNEL -> MONO
    # -------------------------------------------------------------------------

    if channels > 1:

        usable = (
            len(audio)
            - (
                len(audio)
                % channels
            )
        )

        audio = (
            audio[:usable]
            .reshape(
                -1,
                channels,
            )
            .astype(
                np.float32
            )
            .mean(
                axis=1
            )
            .clip(
                -32768,
                32767,
            )
            .astype(
                np.int16
            )
        )

    # -------------------------------------------------------------------------
    # RESAMPLE -> 16 KHz
    # -------------------------------------------------------------------------

    if sample_rate != SAMPLE_RATE:

        print(
            f"[info] Resampling "
            f"{sample_rate}Hz -> "
            f"{SAMPLE_RATE}Hz"
        )

        try:
            import resampy

        except ImportError:
            raise RuntimeError(
                "Install resampy:\n"
                "py -3.11 -m pip install resampy"
            )

        audio_f32 = (
            audio.astype(
                np.float32
            )
            / 32768.0
        )

        audio_f32 = resampy.resample(
            audio_f32,
            sample_rate,
            SAMPLE_RATE,
        )

        audio = (
            audio_f32.clip(
                -1.0,
                1.0,
            )
            * 32767.0
        ).astype(
            np.int16
        )

    return audio


# =============================================================================
# WEBSOCKET RECEIVE STATE
# =============================================================================

@dataclass
class WSState:
    first_event_ns: int | None = None
    first_transcription_ns: int | None = None
    first_final_ns: int | None = None

    first_event_type: str = ""

    first_partial: str = ""
    first_final: str = ""

    done_event: asyncio.Event | None = None


# =============================================================================
# WEBSOCKET RECEIVER
# =============================================================================

async def websocket_receiver(
    ws,
    state,
):
    try:

        async for raw_message in ws:

            event_ns = now_ns()

            if isinstance(
                raw_message,
                bytes,
            ):
                continue

            try:
                message = json.loads(
                    raw_message
                )

            except json.JSONDecodeError:
                continue

            event_type = str(
                message.get(
                    "type",
                    "",
                )
            )

            text = clean_text(
                str(
                    message.get(
                        "text",
                        "",
                    )
                )
            )

            # -----------------------------------------------------------------
            # FIRST SERVER EVENT
            # -----------------------------------------------------------------

            if state.first_event_ns is None:

                state.first_event_ns = (
                    event_ns
                )

                state.first_event_type = (
                    event_type
                )

            # -----------------------------------------------------------------
            # FIRST TRANSCRIPTION
            # -----------------------------------------------------------------

            if event_type == "partial":

                if (
                    state.first_transcription_ns
                    is None
                ):

                    state.first_transcription_ns = (
                        event_ns
                    )

                    state.first_partial = (
                        text
                    )

            elif event_type == "final":

                if (
                    state.first_transcription_ns
                    is None
                ):

                    state.first_transcription_ns = (
                        event_ns
                    )

                if state.first_final_ns is None:

                    state.first_final_ns = (
                        event_ns
                    )

                    state.first_final = (
                        text
                    )

            elif event_type == "error":

                print(
                    f"[server error] "
                    f"{text or message}"
                )

            elif event_type == "done":

                state.done_event.set()

                break

    except ConnectionClosed as exc:

        print(
            "\n[warn] WebSocket closed: "
            f"code={exc.code}, "
            f"reason="
            f"{exc.reason or 'none'}"
        )

    finally:

        state.done_event.set()


# =============================================================================
# WEBSOCKET FILE
# =============================================================================

async def websocket_file_once(
    url,
    file_path,
    language,
    realtime,
):
    audio = load_wav_16khz_mono(
        file_path
    )

    raw_bytes = (
        audio.tobytes()
    )

    chunks = [
        raw_bytes[
            index:
            index + CHUNK_BYTES
        ]
        for index in range(
            0,
            len(raw_bytes),
            CHUNK_BYTES,
        )
    ]

    audio_duration = (
        len(audio)
        / SAMPLE_RATE
    )

    print(
        f"[info] Audio duration: "
        f"{audio_duration:.2f}s"
    )

    # =========================================================================
    # E2E START
    # =========================================================================

    e2e_start_ns = now_ns()

    # =========================================================================
    # WEBSOCKET CONNECT
    # =========================================================================

    connection_start_ns = (
        now_ns()
    )

    ws = await websockets.connect(
        url,
        ping_interval=20,
        ping_timeout=60,
        open_timeout=240,
        close_timeout=10,
        max_size=None,
    )

    connected_ns = now_ns()

    connection_startup_ms = (
        elapsed_ms(
            connection_start_ns,
            connected_ns,
        )
    )

    # =========================================================================
    # CONFIG
    # =========================================================================

    await ws.send(
        json.dumps(
            {
                "backend": "nemotron",
                "sample_rate": SAMPLE_RATE,
                "language": language,
            }
        )
    )

    # =========================================================================
    # RECEIVE TASK
    # =========================================================================

    state = WSState(
        done_event=asyncio.Event()
    )

    receiver_task = (
        asyncio.create_task(
            websocket_receiver(
                ws,
                state,
            )
        )
    )

    # =========================================================================
    # AUDIO STREAM
    # =========================================================================

    audio_start_ns = now_ns()

    try:

        for index, chunk in enumerate(
            chunks
        ):

            await ws.send(
                chunk
            )

            if realtime:

                expected_seconds = (
                    (index + 1)
                    * CHUNK_MS
                    / 1000.0
                )

                actual_seconds = (
                    now_ns()
                    - audio_start_ns
                ) / 1_000_000_000.0

                sleep_seconds = (
                    expected_seconds
                    - actual_seconds
                )

                if sleep_seconds > 0:

                    await asyncio.sleep(
                        sleep_seconds
                    )

            else:

                await asyncio.sleep(
                    0.001
                )

        # ---------------------------------------------------------------------
        # EOF
        # ---------------------------------------------------------------------

        await ws.send(
            json.dumps(
                {
                    "type": "eof"
                }
            )
        )

        try:

            await asyncio.wait_for(
                state.done_event.wait(),
                timeout=60,
            )

        except asyncio.TimeoutError:

            print(
                "[warn] Timeout waiting "
                "for server done"
            )

    finally:

        total_end_ns = now_ns()

        try:

            await ws.close()

        except Exception:

            pass

        if not receiver_task.done():

            receiver_task.cancel()

        try:

            await receiver_task

        except asyncio.CancelledError:

            pass

    response_ns = (
        state.first_event_ns
        or
        state.first_transcription_ns
        or
        total_end_ns
    )

    transcription_ns = (
        state.first_transcription_ns
        or
        state.first_final_ns
        or
        total_end_ns
    )

    return LatencyResult(
        connection_startup_ms=(
            connection_startup_ms
        ),

        connection_response_ms=elapsed_ms(
            connected_ns,
            response_ns,
        ),

        connection_transcription_ms=elapsed_ms(
            connected_ns,
            transcription_ns,
        ),

        e2e_ttfb_ms=elapsed_ms(
            e2e_start_ns,
            response_ns,
        ),

        e2e_ttft_ms=elapsed_ms(
            e2e_start_ns,
            transcription_ns,
        ),

        e2e_total_ms=elapsed_ms(
            e2e_start_ns,
            total_end_ns,
        ),

        first_event_type=(
            state.first_event_type
        ),

        first_partial=(
            state.first_partial
        ),

        first_final=(
            state.first_final
        ),
    )


# =============================================================================
# WEBSOCKET MIC
# =============================================================================

async def websocket_mic_once(
    url,
    language,
    duration,
):
    try:
        import numpy as np
        import sounddevice as sd

    except ImportError:

        raise RuntimeError(
            "Mic mode requires:\n"
            "py -3.11 -m pip install "
            "numpy sounddevice"
        )

    # =========================================================================
    # E2E START
    # =========================================================================

    e2e_start_ns = now_ns()

    # =========================================================================
    # CONNECT
    # =========================================================================

    connection_start_ns = (
        now_ns()
    )

    ws = await websockets.connect(
        url,
        ping_interval=20,
        ping_timeout=60,
        open_timeout=240,
        close_timeout=10,
        max_size=None,
    )

    connected_ns = now_ns()

    connection_startup_ms = (
        elapsed_ms(
            connection_start_ns,
            connected_ns,
        )
    )

    print(
        f"[info] WebSocket connected "
        f"in {connection_startup_ms:.2f} ms"
    )

    # =========================================================================
    # CONFIG
    # =========================================================================

    await ws.send(
        json.dumps(
            {
                "backend": "nemotron",
                "sample_rate": SAMPLE_RATE,
                "language": language,
            }
        )
    )

    # =========================================================================
    # RECEIVER
    # =========================================================================

    state = WSState(
        done_event=asyncio.Event()
    )

    receiver_task = (
        asyncio.create_task(
            websocket_receiver(
                ws,
                state,
            )
        )
    )

    # =========================================================================
    # MIC
    # =========================================================================

    print(
        "[info] START SPEAKING NOW"
    )

    event_loop = (
        asyncio.get_running_loop()
    )

    audio_queue = (
        asyncio.Queue(
            maxsize=200
        )
    )

    dropped_chunks = 0

    def enqueue_audio(pcm):

        nonlocal dropped_chunks

        if audio_queue.full():

            try:

                audio_queue.get_nowait()

                dropped_chunks += 1

            except asyncio.QueueEmpty:

                pass

        try:

            audio_queue.put_nowait(
                pcm
            )

        except asyncio.QueueFull:

            dropped_chunks += 1

    def audio_callback(
        indata,
        frames,
        time_info,
        status,
    ):

        if status:

            print(
                f"\n[audio warning] "
                f"{status}"
            )

        mono = np.clip(
            indata[:, 0],
            -1.0,
            1.0,
        )

        pcm = (
            mono
            * 32767.0
        ).astype(
            np.int16
        ).tobytes()

        event_loop.call_soon_threadsafe(
            enqueue_audio,
            pcm,
        )

    mic_start_ns = now_ns()

    try:

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

                elapsed_seconds = (
                    now_ns()
                    - mic_start_ns
                ) / 1_000_000_000.0

                if elapsed_seconds >= duration:
                    break

                if state.done_event.is_set():

                    print(
                        "[warn] Server ended "
                        "connection early"
                    )

                    break

                try:

                    pcm = (
                        await asyncio.wait_for(
                            audio_queue.get(),
                            timeout=0.5,
                        )
                    )

                except asyncio.TimeoutError:

                    continue

                try:

                    await ws.send(
                        pcm
                    )

                except ConnectionClosed as exc:

                    print(
                        "[warn] Send failed: "
                        f"code={exc.code}, "
                        f"reason="
                        f"{exc.reason or 'none'}"
                    )

                    break

        try:

            await ws.send(
                json.dumps(
                    {
                        "type": "eof"
                    }
                )
            )

        except ConnectionClosed:

            pass

        try:

            await asyncio.wait_for(
                state.done_event.wait(),
                timeout=30,
            )

        except asyncio.TimeoutError:

            print(
                "[warn] Timeout waiting "
                "for final response"
            )

    finally:

        total_end_ns = (
            now_ns()
        )

        try:

            await ws.close()

        except Exception:

            pass

        if not receiver_task.done():

            receiver_task.cancel()

        try:

            await receiver_task

        except asyncio.CancelledError:

            pass

    if dropped_chunks:

        print(
            f"[warn] Dropped chunks: "
            f"{dropped_chunks}"
        )

    response_ns = (
        state.first_event_ns
        or
        state.first_transcription_ns
        or
        total_end_ns
    )

    transcription_ns = (
        state.first_transcription_ns
        or
        state.first_final_ns
        or
        total_end_ns
    )

    return LatencyResult(
        connection_startup_ms=(
            connection_startup_ms
        ),

        connection_response_ms=elapsed_ms(
            connected_ns,
            response_ns,
        ),

        connection_transcription_ms=elapsed_ms(
            connected_ns,
            transcription_ns,
        ),

        e2e_ttfb_ms=elapsed_ms(
            e2e_start_ns,
            response_ns,
        ),

        e2e_ttft_ms=elapsed_ms(
            e2e_start_ns,
            transcription_ns,
        ),

        e2e_total_ms=elapsed_ms(
            e2e_start_ns,
            total_end_ns,
        ),

        first_event_type=(
            state.first_event_type
        ),

        first_partial=(
            state.first_partial
        ),

        first_final=(
            state.first_final
        ),
    )


# =============================================================================
# MULTIPART BUILDER FOR OPENAI
# =============================================================================

def build_multipart_body(
    file_path,
    model,
    language,
    response_format,
):
    path = Path(
        file_path
    )

    boundary = (
        "----NemotronBoundary"
        + uuid.uuid4().hex
    )

    mime_type = (
        mimetypes.guess_type(
            path.name
        )[0]
        or
        "application/octet-stream"
    )

    with path.open(
        "rb"
    ) as audio_file:

        file_bytes = (
            audio_file.read()
        )

    body = bytearray()

    # -------------------------------------------------------------------------
    # FIELD
    # -------------------------------------------------------------------------

    def add_field(
        name,
        value,
    ):

        body.extend(
            f"--{boundary}\r\n".encode()
        )

        body.extend(
            (
                "Content-Disposition: "
                "form-data; "
                f'name="{name}"'
                "\r\n\r\n"
            ).encode()
        )

        body.extend(
            str(value).encode()
        )

        body.extend(
            b"\r\n"
        )

    add_field(
        "model",
        model,
    )

    add_field(
        "language",
        language,
    )

    add_field(
        "response_format",
        response_format,
    )

    # -------------------------------------------------------------------------
    # FILE
    # -------------------------------------------------------------------------

    body.extend(
        f"--{boundary}\r\n".encode()
    )

    body.extend(
        (
            "Content-Disposition: "
            'form-data; name="file"; '
            f'filename="{path.name}"'
            "\r\n"
        ).encode()
    )

    body.extend(
        (
            f"Content-Type: "
            f"{mime_type}"
            "\r\n\r\n"
        ).encode()
    )

    body.extend(
        file_bytes
    )

    body.extend(
        b"\r\n"
    )

    body.extend(
        f"--{boundary}--\r\n".encode()
    )

    return (
        bytes(body),
        boundary,
    )


# =============================================================================
# OPENAI REST TEST
# =============================================================================

def openai_once(
    endpoint,
    file_path,
    model,
    language,
    response_format,
    timeout,
):
    """
    IMPORTANT:

    Timing starts AFTER microphone recording.

    Therefore for OpenAI + mic:

        microphone recording
             |
             | NOT measured
             v
        HTTP test starts here
             |
             v
        POST /v1/audio/transcriptions
    """

    path = Path(
        file_path
    )

    if not path.exists():

        raise FileNotFoundError(
            f"File not found: "
            f"{path}"
        )

    parsed = urlparse(
        endpoint
    )

    if not parsed.hostname:

        raise ValueError(
            f"Invalid endpoint: "
            f"{endpoint}"
        )

    request_path = (
        parsed.path
        or "/"
    )

    if parsed.query:

        request_path += (
            "?"
            + parsed.query
        )

    # Build body BEFORE latency starts.
    #
    # We don't want local file reading / multipart
    # construction counted as Cloud Run latency.

    body, boundary = (
        build_multipart_body(
            file_path=path,
            model=model,
            language=language,
            response_format=(
                response_format
            ),
        )
    )

    # =========================================================================
    # E2E START
    # =========================================================================

    e2e_start_ns = (
        now_ns()
    )

    # =========================================================================
    # TCP + TLS
    # =========================================================================

    connection_start_ns = (
        now_ns()
    )

    if parsed.scheme == "https":

        conn = (
            http.client.HTTPSConnection(
                parsed.hostname,
                port=(
                    parsed.port
                    or 443
                ),
                timeout=timeout,
                context=(
                    ssl.create_default_context()
                ),
            )
        )

    else:

        conn = (
            http.client.HTTPConnection(
                parsed.hostname,
                port=(
                    parsed.port
                    or 80
                ),
                timeout=timeout,
            )
        )

    conn.connect()

    connected_ns = (
        now_ns()
    )

    connection_startup_ms = (
        elapsed_ms(
            connection_start_ns,
            connected_ns,
        )
    )

    # =========================================================================
    # HTTP POST
    # =========================================================================

    conn.putrequest(
        "POST",
        request_path,
    )

    conn.putheader(
        "Content-Type",
        (
            "multipart/form-data; "
            f"boundary={boundary}"
        ),
    )

    conn.putheader(
        "Content-Length",
        str(
            len(body)
        ),
    )

    # Disable connection reuse so each run
    # performs its own TCP/TLS connection.

    conn.putheader(
        "Connection",
        "close",
    )

    conn.endheaders()

    conn.send(
        body
    )

    # =========================================================================
    # RESPONSE HEADERS
    # =========================================================================

    response = (
        conn.getresponse()
    )

    response_headers_ns = (
        now_ns()
    )

    # =========================================================================
    # RESPONSE BODY / TRANSCRIPTION
    # =========================================================================

    response_body = (
        response.read()
    )

    transcription_ns = (
        now_ns()
    )

    content_type = (
        response.getheader(
            "Content-Type",
            "",
        )
        or ""
    )

    text_response = (
        response_body.decode(
            "utf-8",
            errors="replace",
        )
    )

    try:

        if (
            "application/json"
            in content_type.lower()
        ):

            response_json = (
                json.loads(
                    text_response
                )
            )

            formatted_response = (
                json.dumps(
                    response_json,
                    indent=2,
                    ensure_ascii=False,
                )
            )

        else:

            formatted_response = (
                text_response
            )

    except Exception:

        formatted_response = (
            text_response
        )

    total_end_ns = (
        now_ns()
    )

    status = (
        response.status
    )

    result = LatencyResult(
        connection_startup_ms=(
            connection_startup_ms
        ),

        connection_response_ms=elapsed_ms(
            connected_ns,
            response_headers_ns,
        ),

        connection_transcription_ms=elapsed_ms(
            connected_ns,
            transcription_ns,
        ),

        e2e_ttfb_ms=elapsed_ms(
            e2e_start_ns,
            response_headers_ns,
        ),

        e2e_ttft_ms=elapsed_ms(
            e2e_start_ns,
            transcription_ns,
        ),

        e2e_total_ms=elapsed_ms(
            e2e_start_ns,
            total_end_ns,
        ),
    )

    conn.close()

    return (
        result,
        status,
        content_type,
        formatted_response,
    )


# =============================================================================
# WEBSOCKET TEST RUNNER
# =============================================================================

async def run_websocket_tests(
    args,
):
    results = []

    for run_number in range(
        1,
        args.runs + 1,
    ):

        label = (
            "COLD-CANDIDATE"
            if run_number == 1
            else
            "WARM"
        )

        print()
        print("=" * 80)

        if args.mic:

            print(
                f"WEBSOCKET MIC RUN "
                f"{run_number} [{label}]"
            )

        else:

            print(
                f"WEBSOCKET FILE RUN "
                f"{run_number} [{label}]"
            )

        print("=" * 80)

        if args.mic:

            result = (
                await websocket_mic_once(
                    url=args.url,
                    language=args.language,
                    duration=args.duration,
                )
            )

            title = (
                "WEBSOCKET MIC LATENCY - "
                f"RUN {run_number} "
                f"[{label}]"
            )

        else:

            result = (
                await websocket_file_once(
                    url=args.url,
                    file_path=args.file,
                    language=args.language,
                    realtime=args.realtime,
                )
            )

            title = (
                "WEBSOCKET FILE LATENCY - "
                f"RUN {run_number} "
                f"[{label}]"
            )

        results.append(
            result
        )

        print_latency(
            title,
            result,
        )

        print()

        print(
            f"First server event          : "
            f"{result.first_event_type or 'N/A'}"
        )

        if result.first_partial:

            print(
                f"First partial               : "
                f"{result.first_partial}"
            )

        if result.first_final:

            print(
                f"First final                 : "
                f"{result.first_final}"
            )

        if run_number < args.runs:

            print()

            print(
                f"[info] Waiting "
                f"{args.delay:.1f}s before "
                f"warm run..."
            )

            await asyncio.sleep(
                args.delay
            )

    if len(results) >= 2:

        title = (
            "WEBSOCKET MIC COLD VS WARM"
            if args.mic
            else
            "WEBSOCKET FILE COLD VS WARM"
        )

        print_comparison(
            results[0],
            results[1],
            title,
        )


# =============================================================================
# OPENAI TEST RUNNER
# =============================================================================

def run_openai_tests(
    args,
):
    endpoint = (
        args.openai_url
        or
        ws_to_openai_url(
            args.url
        )
    )

    print(
        f"[info] OpenAI endpoint: "
        f"{endpoint}"
    )

    results = []

    for run_number in range(
        1,
        args.runs + 1,
    ):

        label = (
            "COLD-CANDIDATE"
            if run_number == 1
            else
            "WARM"
        )

        temp_mic_file = None

        # =====================================================================
        # MIC INPUT
        # =====================================================================

        if args.mic:

            print()
            print("=" * 80)
            print(
                f"OPENAI MIC RUN "
                f"{run_number} [{label}]"
            )
            print("=" * 80)

            temp_mic_file = (
                record_microphone_to_wav(
                    args.duration
                )
            )

            audio_file = (
                temp_mic_file
            )

        else:

            print()
            print("=" * 80)
            print(
                f"OPENAI FILE RUN "
                f"{run_number} [{label}]"
            )
            print("=" * 80)

            audio_file = (
                args.file
            )

        try:

            # ================================================================
            # HTTP TIMING STARTS INSIDE openai_once()
            # ================================================================

            (
                result,
                status,
                content_type,
                transcription,
            ) = openai_once(
                endpoint=endpoint,
                file_path=audio_file,
                model=args.model,
                language=args.language,
                response_format=(
                    args.response_format
                ),
                timeout=args.timeout,
            )

            results.append(
                result
            )

            if args.mic:

                title = (
                    "OPENAI COMPATIBLE MIC LATENCY - "
                    f"RUN {run_number} "
                    f"[{label}]"
                )

            else:

                title = (
                    "OPENAI COMPATIBLE FILE LATENCY - "
                    f"RUN {run_number} "
                    f"[{label}]"
                )

            print_latency(
                title,
                result,
            )

            print()

            print(
                f"HTTP Status                 : "
                f"{status}"
            )

            print(
                f"Content-Type                : "
                f"{content_type}"
            )

            print()

            print("=" * 80)
            print("TRANSCRIPTION")
            print("=" * 80)

            print(
                transcription
            )

        finally:

            # Delete temporary microphone WAV.
            if (
                temp_mic_file
                and
                os.path.exists(
                    temp_mic_file
                )
            ):

                try:

                    os.remove(
                        temp_mic_file
                    )

                except Exception as exc:

                    print(
                        f"[warn] Could not delete "
                        f"temporary WAV: {exc}"
                    )

        if run_number < args.runs:

            print()

            print(
                f"[info] Waiting "
                f"{args.delay:.1f}s before "
                f"next run..."
            )

            time.sleep(
                args.delay
            )

    if len(results) >= 2:

        title = (
            "OPENAI COMPATIBLE MIC COLD VS WARM"
            if args.mic
            else
            "OPENAI COMPATIBLE FILE COLD VS WARM"
        )

        print_comparison(
            results[0],
            results[1],
            title,
        )


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Python 3.11 Cloud Run "
            "cold-start and ASR latency tester"
        )
    )

    # =========================================================================
    # API MODE
    # =========================================================================

    parser.add_argument(
        "--mode",
        required=True,
        choices=[
            "websocket",
            "openai",
        ],
        help=(
            "websocket = realtime-custom-vad\n"
            "openai = /v1/audio/transcriptions"
        ),
    )

    # =========================================================================
    # AUDIO INPUT
    # =========================================================================

    input_group = (
        parser.add_mutually_exclusive_group(
            required=True
        )
    )

    input_group.add_argument(
        "--file",
        help="Audio file path",
    )

    input_group.add_argument(
        "--mic",
        action="store_true",
        help="Use microphone",
    )

    # =========================================================================
    # ENDPOINTS
    # =========================================================================

    parser.add_argument(
        "--url",
        default=SERVER_URL,
        help="WebSocket endpoint",
    )

    parser.add_argument(
        "--openai-url",
        default=None,
        help=(
            "Explicit OpenAI-compatible "
            "/v1/audio/transcriptions URL"
        ),
    )

    # =========================================================================
    # ASR
    # =========================================================================

    parser.add_argument(
        "--language",
        default="en-US",
    )

    parser.add_argument(
        "--model",
        default=(
            "nemotron-3.5-"
            "asr-streaming-0.6b"
        ),
    )

    parser.add_argument(
        "--response-format",
        default="json",
    )

    # =========================================================================
    # TEST CONFIG
    # =========================================================================

    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help=(
            "1 = single test, "
            "2 = cold candidate + warm comparison"
        ),
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help=(
            "Seconds between runs"
        ),
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=240,
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help=(
            "Mic recording/streaming duration "
            "in seconds"
        ),
    )

    parser.add_argument(
        "--realtime",
        action="store_true",
        help=(
            "For WebSocket file mode, "
            "stream WAV at realtime speed"
        ),
    )

    args = parser.parse_args()

    # =========================================================================
    # VALIDATION
    # =========================================================================

    if args.runs < 1:

        parser.error(
            "--runs must be >= 1"
        )

    if args.duration <= 0:

        parser.error(
            "--duration must be > 0"
        )

    print()
    print("=" * 80)
    print("CLOUD RUN ASR LATENCY TEST")
    print("=" * 80)

    print(
        "[info] No /health request "
        "will be sent."
    )

    if args.runs >= 2:

        print(
            "[info] Run 1 = "
            "COLD-CANDIDATE"
        )

        print(
            "[info] Run 2+ = "
            "WARM"
        )

    # =========================================================================
    # RUN
    # =========================================================================

    if args.mode == "websocket":

        asyncio.run(
            run_websocket_tests(
                args
            )
        )

    else:

        run_openai_tests(
            args
        )


if __name__ == "__main__":
    main()
