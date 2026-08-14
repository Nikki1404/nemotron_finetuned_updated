#!/usr/bin/env python3

import argparse
import asyncio
import json
import mimetypes
import re
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import websockets
from websockets.exceptions import ConnectionClosed


# ============================================================
# CONFIG
# ============================================================

SERVER_URL = (
    "wss://nemotron-3-5-150916788856.us-central1.run.app/"
    "asr/realtime-custom-vad"
)

SAMPLE_RATE = 16000
CHUNK_MS = 100
CHUNK_BYTES = int(SAMPLE_RATE * CHUNK_MS / 1000) * 2

_LANG_TAG_RE = re.compile(r"<[a-z]{2}-[A-Z]{2}>\s*")


# ============================================================
# COMMON UTILITIES
# ============================================================

def now():
    return time.perf_counter()


def elapsed_ms(start, end=None):
    if end is None:
        end = now()

    return (end - start) * 1000.0


def fmt(value):
    if value is None:
        return "N/A"

    return f"{value:.2f} ms"


def clean_text(text):
    return _LANG_TAG_RE.sub("", text or "").strip()


def get_http_base(ws_url):
    parsed = urlparse(ws_url)

    if parsed.scheme == "wss":
        scheme = "https"
    elif parsed.scheme == "ws":
        scheme = "http"
    else:
        scheme = parsed.scheme

    return f"{scheme}://{parsed.netloc}"


def get_openai_url(ws_url):
    return (
        f"{get_http_base(ws_url)}"
        "/v1/audio/transcriptions"
    )


# ============================================================
# WAV LOADING FOR WEBSOCKET
# ============================================================

def load_wav_16khz_mono(path):
    import numpy as np

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        nframes = wf.getnframes()

        raw = wf.readframes(nframes)

    if sample_width != 2:
        raise ValueError(
            "WebSocket test requires "
            "16-bit PCM WAV."
        )

    audio = np.frombuffer(
        raw,
        dtype=np.int16,
    )

    # Convert stereo/multichannel to mono
    if channels > 1:
        usable = (
            len(audio)
            - (len(audio) % channels)
        )

        audio = (
            audio[:usable]
            .reshape(-1, channels)
            .astype(np.float32)
            .mean(axis=1)
            .clip(-32768, 32767)
            .astype(np.int16)
        )

    # Resample if required
    if sample_rate != SAMPLE_RATE:
        print(
            f"[info] Resampling "
            f"{sample_rate}Hz -> {SAMPLE_RATE}Hz"
        )

        try:
            import resampy

        except ImportError:
            raise RuntimeError(
                "Install resampy:\n"
                "py -3.11 -m pip install resampy"
            )

        audio_f32 = (
            audio.astype(np.float32)
            / 32768.0
        )

        audio_f32 = resampy.resample(
            audio_f32,
            sample_rate,
            SAMPLE_RATE,
        )

        audio = (
            audio_f32.clip(-1.0, 1.0)
            * 32767.0
        ).astype(np.int16)

    return audio


# ============================================================
# WEBSOCKET COLD START METRICS
# ============================================================

@dataclass
class WebSocketMetrics:
    connect_ms: float | None = None

    config_send_ms: float | None = None

    first_audio_from_start_ms: float | None = None

    first_event_from_start_ms: float | None = None
    first_event_from_audio_ms: float | None = None

    first_partial_from_start_ms: float | None = None
    first_partial_from_audio_ms: float | None = None

    first_final_from_start_ms: float | None = None
    first_final_from_audio_ms: float | None = None

    total_ms: float | None = None

    first_partial_text: str = ""
    first_final_text: str = ""


async def websocket_test_once(
    url,
    file_path,
    language,
    realtime=True,
):
    audio = load_wav_16khz_mono(
        file_path
    )

    raw_bytes = audio.tobytes()

    chunks = [
        raw_bytes[
            i:i + CHUNK_BYTES
        ]
        for i in range(
            0,
            len(raw_bytes),
            CHUNK_BYTES,
        )
    ]

    metrics = WebSocketMetrics()

    # --------------------------------------------------------
    # VERY IMPORTANT:
    #
    # This is BEFORE websockets.connect().
    #
    # Therefore a true cold Cloud Run instance startup
    # should show up mainly in connect_ms.
    # --------------------------------------------------------

    request_start = now()

    connect_start = now()

    ws = await websockets.connect(
        url,

        # Keepalive
        ping_interval=20,
        ping_timeout=60,

        # Cold GPU/model startup may be slow
        open_timeout=240,

        close_timeout=10,
        max_size=None,
    )

    connected_at = now()

    metrics.connect_ms = elapsed_ms(
        connect_start,
        connected_at,
    )

    # --------------------------------------------------------
    # SEND INITIAL CONFIG
    # --------------------------------------------------------

    config_start = now()

    await ws.send(
        json.dumps(
            {
                "backend": "nemotron",
                "sample_rate": SAMPLE_RATE,
                "language": language,
            }
        )
    )

    metrics.config_send_ms = elapsed_ms(
        config_start
    )

    first_audio_time = None

    done_event = asyncio.Event()

    # --------------------------------------------------------
    # RECEIVE LOOP
    # --------------------------------------------------------

    async def receiver():

        try:
            async for raw in ws:

                if isinstance(raw, bytes):
                    continue

                event_time = now()

                try:
                    msg = json.loads(raw)

                except json.JSONDecodeError:
                    continue

                event_type = msg.get(
                    "type",
                    "",
                )

                text = clean_text(
                    str(
                        msg.get(
                            "text",
                            "",
                        )
                    )
                )

                # --------------------------------------------
                # FIRST SERVER EVENT
                # --------------------------------------------

                if (
                    metrics.first_event_from_start_ms
                    is None
                ):
                    metrics.first_event_from_start_ms = (
                        elapsed_ms(
                            request_start,
                            event_time,
                        )
                    )

                    if first_audio_time is not None:
                        metrics.first_event_from_audio_ms = (
                            elapsed_ms(
                                first_audio_time,
                                event_time,
                            )
                        )

                # --------------------------------------------
                # FIRST PARTIAL
                # --------------------------------------------

                if (
                    event_type == "partial"
                    and
                    metrics.first_partial_from_start_ms
                    is None
                ):

                    metrics.first_partial_from_start_ms = (
                        elapsed_ms(
                            request_start,
                            event_time,
                        )
                    )

                    if first_audio_time is not None:
                        metrics.first_partial_from_audio_ms = (
                            elapsed_ms(
                                first_audio_time,
                                event_time,
                            )
                        )

                    metrics.first_partial_text = text

                # --------------------------------------------
                # FIRST FINAL
                # --------------------------------------------

                if (
                    event_type == "final"
                    and
                    metrics.first_final_from_start_ms
                    is None
                ):

                    metrics.first_final_from_start_ms = (
                        elapsed_ms(
                            request_start,
                            event_time,
                        )
                    )

                    if first_audio_time is not None:
                        metrics.first_final_from_audio_ms = (
                            elapsed_ms(
                                first_audio_time,
                                event_time,
                            )
                        )

                    metrics.first_final_text = text

                if event_type == "done":
                    done_event.set()
                    break

        except ConnectionClosed:
            done_event.set()

        finally:
            done_event.set()

    receive_task = asyncio.create_task(
        receiver()
    )

    # --------------------------------------------------------
    # SEND AUDIO
    # --------------------------------------------------------

    audio_stream_start = now()

    try:

        for index, chunk in enumerate(
            chunks
        ):

            send_time = now()

            await ws.send(chunk)

            if first_audio_time is None:

                first_audio_time = send_time

                metrics.first_audio_from_start_ms = (
                    elapsed_ms(
                        request_start,
                        send_time,
                    )
                )

            if realtime:

                expected_elapsed = (
                    (index + 1)
                    * CHUNK_MS
                    / 1000.0
                )

                actual_elapsed = (
                    now()
                    - audio_stream_start
                )

                sleep_time = (
                    expected_elapsed
                    - actual_elapsed
                )

                if sleep_time > 0:
                    await asyncio.sleep(
                        sleep_time
                    )

            else:
                await asyncio.sleep(
                    0.001
                )

        # Flush ASR
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
                "for done"
            )

    finally:

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

    metrics.total_ms = elapsed_ms(
        request_start
    )

    return metrics


def print_websocket_metrics(
    run_number,
    label,
    m,
):
    print()
    print("=" * 80)
    print(
        f"WEBSOCKET RUN {run_number} "
        f"- {label}"
    )
    print("=" * 80)

    print(
        f"WS CONNECT / HANDSHAKE     : "
        f"{fmt(m.connect_ms)}"
    )

    print(
        f"Config send                : "
        f"{fmt(m.config_send_ms)}"
    )

    print(
        f"First audio sent @         : "
        f"{fmt(m.first_audio_from_start_ms)}"
    )

    print(
        f"First server event         : "
        f"{fmt(m.first_event_from_start_ms)} "
        "from connection start"
    )

    print(
        f"First server event         : "
        f"{fmt(m.first_event_from_audio_ms)} "
        "from first audio"
    )

    print(
        f"FIRST PARTIAL              : "
        f"{fmt(m.first_partial_from_start_ms)} "
        "from connection start"
    )

    print(
        f"FIRST PARTIAL / TTFT       : "
        f"{fmt(m.first_partial_from_audio_ms)} "
        "from first audio"
    )

    print(
        f"FIRST FINAL                : "
        f"{fmt(m.first_final_from_start_ms)} "
        "from connection start"
    )

    print(
        f"FIRST FINAL FROM AUDIO     : "
        f"{fmt(m.first_final_from_audio_ms)}"
    )

    print(
        f"TOTAL                      : "
        f"{fmt(m.total_ms)}"
    )

    if m.first_partial_text:
        print(
            "First partial text         : "
            f"{m.first_partial_text}"
        )

    if m.first_final_text:
        print(
            "First final text           : "
            f"{m.first_final_text}"
        )


# ============================================================
# OPENAI-COMPATIBLE COLD START
# ============================================================

def openai_test_once(
    endpoint,
    file_path,
    model,
    language,
    response_format,
    timeout,
):
    try:
        import requests

    except ImportError:
        raise RuntimeError(
            "Install requests:\n"
            "py -3.11 -m pip install requests"
        )

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(path)

    mime_type = (
        mimetypes.guess_type(
            path.name
        )[0]
        or "application/octet-stream"
    )

    # --------------------------------------------------------
    # START BEFORE TCP/TLS/HTTP/CLOUD RUN
    # --------------------------------------------------------

    request_start = now()

    with path.open("rb") as audio_file:

        response = requests.post(
            endpoint,

            data={
                "model": model,
                "language": language,
                "response_format": (
                    response_format
                ),
            },

            files={
                "file": (
                    path.name,
                    audio_file,
                    mime_type,
                )
            },

            # Do not reuse HTTP connection between
            # cold/warm tests.
            headers={
                "Connection": "close"
            },

            timeout=timeout,

            # Return once headers arrive.
            stream=True,
        )

        headers_received = now()

        # Force body download
        body = response.content

        body_received = now()

    return {
        "status": response.status_code,

        # Includes:
        #
        # DNS/TCP/TLS +
        # Cloud Run routing +
        # container cold start +
        # app/model initialization +
        # ASR processing until response
        "headers_ms": elapsed_ms(
            request_start,
            headers_received,
        ),

        "body_ms": elapsed_ms(
            headers_received,
            body_received,
        ),

        "total_ms": elapsed_ms(
            request_start,
            body_received,
        ),

        "content_type": (
            response.headers.get(
                "content-type",
                "",
            )
        ),

        "body": body.decode(
            response.encoding or "utf-8",
            errors="replace",
        ),
    }


def print_openai_metrics(
    run_number,
    label,
    result,
):
    print()
    print("=" * 80)

    print(
        f"OPENAI-COMPATIBLE RUN "
        f"{run_number} - {label}"
    )

    print("=" * 80)

    print(
        f"HTTP status                : "
        f"{result['status']}"
    )

    print(
        f"TIME TO RESPONSE HEADERS   : "
        f"{result['headers_ms']:.2f} ms"
    )

    print(
        f"Response body read         : "
        f"{result['body_ms']:.2f} ms"
    )

    print(
        f"TOTAL REST LATENCY         : "
        f"{result['total_ms']:.2f} ms"
    )

    print(
        f"Content-Type               : "
        f"{result['content_type']}"
    )

    print()
    print("Response:")
    print(result["body"])


# ============================================================
# COLD VS WARM COMPARISON
# ============================================================

def print_delta(
    name,
    cold,
    warm,
):
    if (
        cold is None
        or warm is None
    ):
        print(
            f"{name:30}: N/A"
        )
        return

    delta = cold - warm

    ratio = (
        cold / warm
        if warm > 0
        else 0
    )

    print(
        f"{name:30}: "
        f"cold={cold:.2f} ms  "
        f"warm={warm:.2f} ms  "
        f"delta={delta:.2f} ms  "
        f"ratio={ratio:.2f}x"
    )


async def run_websocket_tests(args):

    results = []

    for index in range(
        args.runs
    ):

        if index == 0:
            label = "COLD-CANDIDATE"
        else:
            label = "WARM"

        metrics = (
            await websocket_test_once(
                args.url,
                args.file,
                args.language,
                args.realtime,
            )
        )

        results.append(metrics)

        print_websocket_metrics(
            index + 1,
            label,
            metrics,
        )

        if index + 1 < args.runs:
            print(
                f"\n[info] Waiting "
                f"{args.delay}s before "
                "next warm test..."
            )

            await asyncio.sleep(
                args.delay
            )

    if len(results) >= 2:

        cold = results[0]
        warm = results[1]

        print()
        print("=" * 80)
        print(
            "WEBSOCKET COLD vs WARM"
        )
        print("=" * 80)

        print_delta(
            "WS handshake",
            cold.connect_ms,
            warm.connect_ms,
        )

        print_delta(
            "First server event",
            cold.first_event_from_start_ms,
            warm.first_event_from_start_ms,
        )

        print_delta(
            "First partial from start",
            cold.first_partial_from_start_ms,
            warm.first_partial_from_start_ms,
        )

        print_delta(
            "First partial from audio",
            cold.first_partial_from_audio_ms,
            warm.first_partial_from_audio_ms,
        )

        print_delta(
            "First final from start",
            cold.first_final_from_start_ms,
            warm.first_final_from_start_ms,
        )


def run_openai_tests(args):

    endpoint = (
        args.openai_url
        or get_openai_url(
            args.url
        )
    )

    print(
        "[info] OpenAI endpoint:",
        endpoint,
    )

    results = []

    for index in range(
        args.runs
    ):

        if index == 0:
            label = "COLD-CANDIDATE"
        else:
            label = "WARM"

        result = openai_test_once(
            endpoint=endpoint,
            file_path=args.file,
            model=args.model,
            language=args.language,
            response_format=(
                args.response_format
            ),
            timeout=args.timeout,
        )

        results.append(result)

        print_openai_metrics(
            index + 1,
            label,
            result,
        )

        if index + 1 < args.runs:

            print(
                f"\n[info] Waiting "
                f"{args.delay}s before "
                "warm request..."
            )

            time.sleep(
                args.delay
            )

    if len(results) >= 2:

        cold = results[0]
        warm = results[1]

        print()
        print("=" * 80)
        print(
            "OPENAI-COMPATIBLE "
            "COLD vs WARM"
        )
        print("=" * 80)

        print_delta(
            "Response headers",
            cold["headers_ms"],
            warm["headers_ms"],
        )

        print_delta(
            "Total REST latency",
            cold["total_ms"],
            warm["total_ms"],
        )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Cloud Run cold-start tester "
            "for Nemotron ASR"
        )
    )

    parser.add_argument(
        "--mode",
        choices=[
            "websocket",
            "openai",
        ],
        required=True,
    )

    parser.add_argument(
        "--file",
        required=True,
    )

    parser.add_argument(
        "--language",
        default="en-US",
    )

    parser.add_argument(
        "--url",
        default=SERVER_URL,
    )

    parser.add_argument(
        "--openai-url",
        default=None,
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
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=240.0,
    )

    parser.add_argument(
        "--realtime",
        action="store_true",
    )

    args = parser.parse_args()

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "- This script does NOT call /health."
    )

    print(
        "- Run #1 is a COLD-START CANDIDATE."
    )

    print(
        "- Run #2+ are immediate WARM comparisons."
    )

    print(
        "- Client-only testing cannot prove "
        "that Cloud Run had scaled to zero."
    )

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
