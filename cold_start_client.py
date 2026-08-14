#!/usr/bin/env python3

import argparse
import asyncio
import http.client
import json
import mimetypes
import re
import ssl
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
# TIMER UTILITIES
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
# OUTPUT
# =============================================================================

def print_latency(
    title,
    connection_startup,
    connection_response,
    connection_transcription,
    e2e_ttfb,
    e2e_ttft,
    e2e_total,
):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)

    print(
        f"Connection / startup       : "
        f"{connection_startup:.2f} ms"
    )

    print(
        f"Connection -> response     : "
        f"{connection_response:.2f} ms"
    )

    print(
        f"Connection -> transcription: "
        f"{connection_transcription:.2f} ms"
    )

    print(
        f"E2E TTFB                   : "
        f"{e2e_ttfb:.2f} ms"
    )

    print(
        f"E2E TTFT/TTFA              : "
        f"{e2e_ttft:.2f} ms"
    )

    print(
        f"E2E TOTAL                  : "
        f"{e2e_total:.2f} ms"
    )


# =============================================================================
# WAV LOADING
# =============================================================================

def load_wav_16khz_mono(file_path):
    import numpy as np

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    with wave.open(
        str(path),
        "rb",
    ) as wf:

        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        frames = wf.getnframes()

        raw = wf.readframes(frames)

    if sample_width != 2:
        raise ValueError(
            "WebSocket mode currently requires "
            "16-bit PCM WAV input."
        )

    audio = np.frombuffer(
        raw,
        dtype=np.int16,
    )

    # -------------------------------------------------------------------------
    # Stereo -> mono
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
    # Resample -> 16 kHz
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
            np.clip(
                audio_f32,
                -1.0,
                1.0,
            )
            * 32767
        ).astype(
            np.int16
        )

    return audio


# =============================================================================
# WEBSOCKET METRICS
# =============================================================================

@dataclass
class WebSocketResult:
    connection_startup_ms: float
    connection_response_ms: float
    connection_transcription_ms: float
    e2e_ttfb_ms: float
    e2e_ttft_ms: float
    e2e_total_ms: float

    first_event_type: str = ""
    first_partial: str = ""
    first_final: str = ""


async def run_websocket_once(
    url,
    file_path,
    language,
    realtime,
):
    """
    Metrics:

    Connection / startup
        test start -> WebSocket handshake complete

    Connection -> response
        WebSocket connected -> first server event

    Connection -> transcription
        WebSocket connected -> first partial transcript

    E2E TTFB
        test start -> first server event

    E2E TTFT/TTFA
        test start -> first partial transcript

    E2E TOTAL
        test start -> server done / connection completion
    """

    audio = load_wav_16khz_mono(
        file_path
    )

    raw_bytes = audio.tobytes()

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

    # =========================================================================
    # E2E START
    # =========================================================================

    e2e_start_ns = now_ns()

    # =========================================================================
    # CONNECTION
    # =========================================================================

    connect_start_ns = now_ns()

    ws = await websockets.connect(
        url,
        ping_interval=20,
        ping_timeout=60,

        # Cold Cloud Run startup may take time.
        open_timeout=240,

        close_timeout=10,
        max_size=None,
    )

    connected_ns = now_ns()

    connection_startup_ms = elapsed_ms(
        connect_start_ns,
        connected_ns,
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
    # RECEIVE STATE
    # =========================================================================

    first_event_ns = None
    first_partial_ns = None
    first_final_ns = None

    first_event_type = ""
    first_partial_text = ""
    first_final_text = ""

    done_event = asyncio.Event()

    # =========================================================================
    # RECEIVER
    # =========================================================================

    async def receiver():

        nonlocal first_event_ns
        nonlocal first_partial_ns
        nonlocal first_final_ns

        nonlocal first_event_type
        nonlocal first_partial_text
        nonlocal first_final_text

        try:

            async for raw_message in ws:

                event_ns = now_ns()

                if isinstance(
                    raw_message,
                    bytes,
                ):
                    continue

                try:
                    msg = json.loads(
                        raw_message
                    )

                except json.JSONDecodeError:
                    continue

                event_type = str(
                    msg.get(
                        "type",
                        "",
                    )
                )

                text = clean_text(
                    str(
                        msg.get(
                            "text",
                            "",
                        )
                    )
                )

                # -------------------------------------------------------------
                # FIRST SERVER EVENT
                # -------------------------------------------------------------

                if first_event_ns is None:

                    first_event_ns = (
                        event_ns
                    )

                    first_event_type = (
                        event_type
                    )

                # -------------------------------------------------------------
                # FIRST PARTIAL
                # -------------------------------------------------------------

                if (
                    event_type
                    == "partial"
                    and
                    first_partial_ns
                    is None
                ):

                    first_partial_ns = (
                        event_ns
                    )

                    first_partial_text = (
                        text
                    )

                # -------------------------------------------------------------
                # FIRST FINAL
                # -------------------------------------------------------------

                if (
                    event_type
                    == "final"
                    and
                    first_final_ns
                    is None
                ):

                    first_final_ns = (
                        event_ns
                    )

                    first_final_text = (
                        text
                    )

                # -------------------------------------------------------------
                # DONE
                # -------------------------------------------------------------

                if event_type == "done":

                    done_event.set()
                    break

                if event_type == "error":

                    print(
                        f"[server error] "
                        f"{text or msg}"
                    )

        except ConnectionClosed as exc:

            print(
                "[warn] WebSocket closed: "
                f"code={exc.code}, "
                f"reason="
                f"{exc.reason or 'none'}"
            )

        finally:
            done_event.set()

    receive_task = asyncio.create_task(
        receiver()
    )

    # =========================================================================
    # STREAM AUDIO
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

                expected_elapsed = (
                    (index + 1)
                    * CHUNK_MS
                    / 1000.0
                )

                actual_elapsed = (
                    (
                        now_ns()
                        - audio_start_ns
                    )
                    / 1_000_000_000
                )

                sleep_for = (
                    expected_elapsed
                    - actual_elapsed
                )

                if sleep_for > 0:

                    await asyncio.sleep(
                        sleep_for
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
                done_event.wait(),
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

        if not receive_task.done():
            receive_task.cancel()

        try:
            await receive_task

        except asyncio.CancelledError:
            pass

    # =========================================================================
    # FALLBACK
    # =========================================================================

    # Some servers may not send partials.
    # In that case use first final as transcription timing.

    transcription_ns = (
        first_partial_ns
        or first_final_ns
        or total_end_ns
    )

    response_ns = (
        first_event_ns
        or transcription_ns
    )

    # =========================================================================
    # METRIC CALCULATION
    # =========================================================================

    connection_response_ms = elapsed_ms(
        connected_ns,
        response_ns,
    )

    connection_transcription_ms = elapsed_ms(
        connected_ns,
        transcription_ns,
    )

    e2e_ttfb_ms = elapsed_ms(
        e2e_start_ns,
        response_ns,
    )

    e2e_ttft_ms = elapsed_ms(
        e2e_start_ns,
        transcription_ns,
    )

    e2e_total_ms = elapsed_ms(
        e2e_start_ns,
        total_end_ns,
    )

    return WebSocketResult(
        connection_startup_ms=(
            connection_startup_ms
        ),

        connection_response_ms=(
            connection_response_ms
        ),

        connection_transcription_ms=(
            connection_transcription_ms
        ),

        e2e_ttfb_ms=(
            e2e_ttfb_ms
        ),

        e2e_ttft_ms=(
            e2e_ttft_ms
        ),

        e2e_total_ms=(
            e2e_total_ms
        ),

        first_event_type=(
            first_event_type
        ),

        first_partial=(
            first_partial_text
        ),

        first_final=(
            first_final_text
        ),
    )


# =============================================================================
# MULTIPART BUILDER
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
    # FORM FIELD
    # -------------------------------------------------------------------------

    def add_field(
        field_name,
        field_value,
    ):

        body.extend(
            (
                f"--{boundary}\r\n"
            ).encode()
        )

        body.extend(
            (
                "Content-Disposition: "
                "form-data; "
                f'name="{field_name}"'
                "\r\n\r\n"
            ).encode()
        )

        body.extend(
            str(
                field_value
            ).encode()
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
        (
            f"--{boundary}\r\n"
        ).encode()
    )

    body.extend(
        (
            "Content-Disposition: "
            "form-data; "
            'name="file"; '
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
        (
            f"--{boundary}--\r\n"
        ).encode()
    )

    return (
        bytes(body),
        boundary,
    )


# =============================================================================
# OPENAI-COMPATIBLE TEST
# =============================================================================

def run_openai_once(
    endpoint,
    file_path,
    model,
    language,
    response_format,
    timeout,
):
    """
    Metrics:

    Connection / startup
        TCP + TLS connection

    Connection -> response
        established connection -> HTTP response headers

    Connection -> transcription
        established connection -> complete transcription body

    E2E TTFB
        test start -> HTTP response headers

    E2E TTFT/TTFA
        test start -> complete transcription response

    E2E TOTAL
        test start -> response parsing complete
    """

    path = Path(
        file_path
    )

    if not path.exists():

        raise FileNotFoundError(
            f"File not found: "
            f"{file_path}"
        )

    parsed = urlparse(
        endpoint
    )

    host = parsed.hostname

    if not host:

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

    e2e_start_ns = now_ns()

    # =========================================================================
    # TCP / TLS CONNECTION
    # =========================================================================

    connection_start_ns = now_ns()

    if parsed.scheme == "https":

        context = (
            ssl.create_default_context()
        )

        conn = (
            http.client.HTTPSConnection(
                host,
                port=(
                    parsed.port
                    or 443
                ),
                timeout=timeout,
                context=context,
            )
        )

    else:

        conn = (
            http.client.HTTPConnection(
                host,
                port=(
                    parsed.port
                    or 80
                ),
                timeout=timeout,
            )
        )

    # Explicit connection so we can
    # measure TCP/TLS separately.
    conn.connect()

    connected_ns = now_ns()

    connection_startup_ms = (
        elapsed_ms(
            connection_start_ns,
            connected_ns,
        )
    )

    # =========================================================================
    # SEND REQUEST
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

    # Avoid connection reuse.
    conn.putheader(
        "Connection",
        "close",
    )

    conn.endheaders()

    conn.send(
        body
    )

    # =========================================================================
    # RESPONSE HEADERS = TTFB
    # =========================================================================

    response = (
        conn.getresponse()
    )

    response_headers_ns = (
        now_ns()
    )

    # =========================================================================
    # TRANSCRIPTION BODY
    # =========================================================================

    response_body = (
        response.read()
    )

    transcription_ns = (
        now_ns()
    )

    # =========================================================================
    # PARSE RESPONSE
    # =========================================================================

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
            in
            content_type.lower()
        ):

            parsed_response = (
                json.loads(
                    text_response
                )
            )

            formatted_response = (
                json.dumps(
                    parsed_response,
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

    total_end_ns = now_ns()

    # =========================================================================
    # METRICS
    # =========================================================================

    connection_response_ms = (
        elapsed_ms(
            connected_ns,
            response_headers_ns,
        )
    )

    connection_transcription_ms = (
        elapsed_ms(
            connected_ns,
            transcription_ns,
        )
    )

    e2e_ttfb_ms = (
        elapsed_ms(
            e2e_start_ns,
            response_headers_ns,
        )
    )

    e2e_ttft_ms = (
        elapsed_ms(
            e2e_start_ns,
            transcription_ns,
        )
    )

    e2e_total_ms = (
        elapsed_ms(
            e2e_start_ns,
            total_end_ns,
        )
    )

    status = (
        response.status
    )

    conn.close()

    return {
        "connection_startup_ms": (
            connection_startup_ms
        ),

        "connection_response_ms": (
            connection_response_ms
        ),

        "connection_transcription_ms": (
            connection_transcription_ms
        ),

        "e2e_ttfb_ms": (
            e2e_ttfb_ms
        ),

        "e2e_ttft_ms": (
            e2e_ttft_ms
        ),

        "e2e_total_ms": (
            e2e_total_ms
        ),

        "status": status,

        "content_type": (
            content_type
        ),

        "response": (
            formatted_response
        ),
    }


# =============================================================================
# COLD VS WARM
# =============================================================================

def print_comparison(
    cold,
    warm,
    mode,
):
    print()
    print("=" * 80)
    print(
        f"{mode.upper()} "
        "COLD VS WARM"
    )
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

    for label, key in metrics:

        if mode == "websocket":

            cold_value = getattr(
                cold,
                key,
            )

            warm_value = getattr(
                warm,
                key,
            )

        else:

            cold_value = cold[key]
            warm_value = warm[key]

        delta = (
            cold_value
            - warm_value
        )

        ratio = (
            cold_value
            / warm_value
            if warm_value > 0
            else 0
        )

        print(
            f"{label:29}: "
            f"cold={cold_value:.2f} ms | "
            f"warm={warm_value:.2f} ms | "
            f"delta={delta:.2f} ms | "
            f"{ratio:.2f}x"
        )


# =============================================================================
# WEBSOCKET MULTI-RUN
# =============================================================================

async def websocket_runs(args):

    results = []

    for run_number in range(
        1,
        args.runs + 1,
    ):

        label = (
            "COLD-CANDIDATE"
            if run_number == 1
            else "WARM"
        )

        print()
        print(
            f"[info] Starting "
            f"WebSocket run "
            f"{run_number}/{args.runs} "
            f"[{label}]"
        )

        result = (
            await run_websocket_once(
                url=args.url,
                file_path=args.file,
                language=args.language,
                realtime=args.realtime,
            )
        )

        results.append(
            result
        )

        print_latency(
            title=(
                f"WEBSOCKET LATENCY - "
                f"RUN {run_number} "
                f"[{label}]"
            ),

            connection_startup=(
                result.connection_startup_ms
            ),

            connection_response=(
                result.connection_response_ms
            ),

            connection_transcription=(
                result.connection_transcription_ms
            ),

            e2e_ttfb=(
                result.e2e_ttfb_ms
            ),

            e2e_ttft=(
                result.e2e_ttft_ms
            ),

            e2e_total=(
                result.e2e_total_ms
            ),
        )

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

            print(
                f"\n[info] Waiting "
                f"{args.delay:.1f}s..."
            )

            await asyncio.sleep(
                args.delay
            )

    if len(results) >= 2:

        print_comparison(
            results[0],
            results[1],
            "websocket",
        )


# =============================================================================
# OPENAI MULTI-RUN
# =============================================================================

def openai_runs(args):

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
            else "WARM"
        )

        print()
        print(
            f"[info] Starting "
            f"OpenAI-compatible run "
            f"{run_number}/{args.runs} "
            f"[{label}]"
        )

        result = run_openai_once(
            endpoint=endpoint,
            file_path=args.file,
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

        print_latency(
            title=(
                "OPENAI COMPATIBLE LATENCY - "
                f"RUN {run_number} "
                f"[{label}]"
            ),

            connection_startup=(
                result[
                    "connection_startup_ms"
                ]
            ),

            connection_response=(
                result[
                    "connection_response_ms"
                ]
            ),

            connection_transcription=(
                result[
                    "connection_transcription_ms"
                ]
            ),

            e2e_ttfb=(
                result[
                    "e2e_ttfb_ms"
                ]
            ),

            e2e_ttft=(
                result[
                    "e2e_ttft_ms"
                ]
            ),

            e2e_total=(
                result[
                    "e2e_total_ms"
                ]
            ),
        )

        print(
            f"HTTP Status                 : "
            f"{result['status']}"
        )

        print(
            f"Content-Type                : "
            f"{result['content_type']}"
        )

        print()
        print("TRANSCRIPTION")
        print("-" * 80)

        print(
            result["response"]
        )

        if run_number < args.runs:

            print(
                f"\n[info] Waiting "
                f"{args.delay:.1f}s..."
            )

            time.sleep(
                args.delay
            )

    if len(results) >= 2:

        print_comparison(
            results[0],
            results[1],
            "openai",
        )


# =============================================================================
# MAIN
# =============================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Python 3.11 Cloud Run cold-start "
            "latency tester for Nemotron ASR"
        )
    )

    parser.add_argument(
        "--mode",
        required=True,
        choices=[
            "websocket",
            "openai",
        ],
    )

    parser.add_argument(
        "--file",
        required=True,
        help="Audio WAV file",
    )

    parser.add_argument(
        "--url",
        default=SERVER_URL,
        help=(
            "WebSocket endpoint"
        ),
    )

    parser.add_argument(
        "--openai-url",
        default=None,
        help=(
            "Optional explicit "
            "/v1/audio/transcriptions URL"
        ),
    )

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

    parser.add_argument(
        "--runs",
        type=int,
        default=2,
        help=(
            "Number of tests. "
            "Run 1 = cold candidate; "
            "Run 2+ = warm."
        ),
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help=(
            "Delay between test runs"
        ),
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=240,
    )

    parser.add_argument(
        "--realtime",
        action="store_true",
        help=(
            "Send WebSocket audio "
            "at real-time speed"
        ),
    )

    args = parser.parse_args()

    print()
    print("=" * 80)
    print("CLOUD RUN COLD START TEST")
    print("=" * 80)

    print(
        "[info] No /health request "
        "will be sent."
    )

    print(
        "[info] Run 1 is treated as "
        "COLD-CANDIDATE."
    )

    print(
        "[info] Run 2+ are immediate "
        "WARM comparisons."
    )

    print(
        "[info] Make sure nothing has "
        "called the service before Run 1."
    )

    if args.mode == "websocket":

        asyncio.run(
            websocket_runs(
                args
            )
        )

    else:

        openai_runs(
            args
        )


if __name__ == "__main__":
    main()
