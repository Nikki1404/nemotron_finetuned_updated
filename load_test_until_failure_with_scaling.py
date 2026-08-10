#!/usr/bin/env python3
"""
load_test_local.py

Local load tester for your existing deployed Nemotron Cloud Run service.

Supports:
  --mode http  -> OpenAI-compatible POST /v1/audio/transcriptions
  --mode ws    -> /asr/realtime-custom-vad
  --mode both

Reads manifest.jsonl created by prepare_chunks_local.py.
Each concurrent worker gets a DIFFERENT chunk where possible.

For every request it prints:
  reference
  prediction
  WER
  latency / TTFB
  HTTP or WebSocket error

Also writes JSON + CSV reports.
"""

import argparse
import asyncio
import csv
import json
import random
import re
import statistics
import time
import wave
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse
from datetime import datetime, timedelta, timezone

import httpx
import websockets
from websockets.exceptions import ConnectionClosed
from google.cloud import monitoring_v3
from google.protobuf.timestamp_pb2 import Timestamp


MODEL = "nemotron-3.5-asr-streaming-0.6b"
SAMPLE_RATE = 16000
CHUNK_MS = 100
CHUNK_BYTES = int(SAMPLE_RATE * CHUNK_MS / 1000) * 2


@dataclass
class Result:
    mode: str
    stage_concurrency: int
    round_no: int
    request_id: int
    chunk_id: str
    use_case: str
    success: bool = False
    status_code: int = 0
    latency_sec: float = 0.0
    ttfb_ms: Optional[float] = None
    wer_pct: Optional[float] = None
    reference: str = ""
    transcript: str = ""
    error: str = ""


def normalize_for_wer(text: str) -> list[str]:
    """
    Conservative WER normalization:
    - lower case
    - remove language tags/punctuation
    - keep digits as digits
    It intentionally does NOT convert 'one two three four' <-> '1234'.
    """
    text = text.lower()
    text = re.sub(r"<[a-z]{2}-[A-Z]{2}>", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split()


def wer(reference: str, hypothesis: str) -> float:
    ref = normalize_for_wer(reference)
    hyp = normalize_for_wer(hypothesis)

    if not ref:
        return 0.0 if not hyp else 100.0

    prev = list(range(len(hyp) + 1))
    for i, rw in enumerate(ref, start=1):
        cur = [i]
        for j, hw in enumerate(hyp, start=1):
            cost = 0 if rw == hw else 1
            cur.append(
                min(
                    cur[-1] + 1,      # insertion
                    prev[j] + 1,      # deletion
                    prev[j - 1] + cost,
                )
            )
        prev = cur

    return 100.0 * prev[-1] / len(ref)


def percentile(values, p):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    idx = (len(values) - 1) * p / 100.0
    lo = int(idx)
    hi = min(lo + 1, len(values) - 1)
    frac = idx - lo
    return values[lo] * (1 - frac) + values[hi] * frac


def load_manifest(path: Path, include_review: bool):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if not include_review and row.get("status") != "OK":
                continue

            audio = Path(row["audio_path"])
            ref = Path(row["reference_path"])
            if not audio.exists() or not ref.exists():
                continue

            row["reference"] = ref.read_text(encoding="utf-8").strip()
            rows.append(row)

    if not rows:
        raise RuntimeError("No usable chunks found in manifest")
    return rows


def derive_urls(base_url: str):
    base = base_url.rstrip("/")
    parsed = urlparse(base)

    if parsed.scheme not in ("http", "https"):
        raise ValueError("--base-url must start with http:// or https://")

    if base.endswith("/v1"):
        http_url = base + "/audio/transcriptions"
        root = base[:-3]
    else:
        http_url = base + "/v1/audio/transcriptions"
        root = base

    p = urlparse(root)
    ws_scheme = "wss" if p.scheme == "https" else "ws"
    ws_url = f"{ws_scheme}://{p.netloc}/asr/realtime-custom-vad"

    return http_url, ws_url


def _to_timestamp(dt: datetime) -> Timestamp:
    return Timestamp(seconds=int(dt.timestamp()))


def get_instance_count_samples(
    project_id: str,
    service_name: str,
    region: str,
    start_time: datetime,
    end_time: datetime,
):
    """
    Read Cloud Run actual instance-count samples for the completed load test.
    The metric is delayed, so this is intentionally called after the test.
    """
    client = monitoring_v3.MetricServiceClient()

    metric_filter = (
        'metric.type="run.googleapis.com/container/instance_count" '
        'AND resource.type="cloud_run_revision" '
        f'AND resource.labels.service_name="{service_name}" '
        f'AND resource.labels.location="{region}"'
    )

    series = client.list_time_series(
        request={
            "name": f"projects/{project_id}",
            "filter": metric_filter,
            "interval": monitoring_v3.TimeInterval(
                {
                    "start_time": _to_timestamp(
                        start_time - timedelta(seconds=30)
                    ),
                    "end_time": _to_timestamp(
                        end_time + timedelta(seconds=30)
                    ),
                }
            ),
            "view": monitoring_v3.ListTimeSeriesRequest.TimeSeriesView.FULL,
        }
    )

    totals_by_time = {}

    for ts in series:
        state = ts.metric.labels.get("state", "unknown")
        revision = ts.resource.labels.get("revision_name", "unknown")

        for point in ts.points:
            epoch = int(point.interval.end_time.timestamp())
            value = int(point.value.int64_value)

            if epoch not in totals_by_time:
                totals_by_time[epoch] = {
                    "count": 0,
                    "parts": [],
                }

            totals_by_time[epoch]["count"] += value
            totals_by_time[epoch]["parts"].append(
                {
                    "state": state,
                    "revision": revision,
                    "count": value,
                }
            )

    samples = []

    for epoch in sorted(totals_by_time):
        samples.append(
            {
                "timestamp": datetime.fromtimestamp(
                    epoch,
                    timezone.utc,
                ),
                "count": totals_by_time[epoch]["count"],
                "parts": totals_by_time[epoch]["parts"],
            }
        )

    return samples


def concurrency_at_time(sample_time, stage_windows):
    """
    Return the concurrency stage active at the metric sample time.
    If the sample falls in the short rest gap, map it to the most
    recently completed concurrency stage.
    """
    for stage in stage_windows:
        if stage["start"] <= sample_time <= stage["end"]:
            return stage["concurrency"], "during"

    previous = [
        stage
        for stage in stage_windows
        if stage["end"] < sample_time
    ]

    if previous:
        latest = max(previous, key=lambda x: x["end"])
        return latest["concurrency"], "after"

    return None, "before"


async def show_scaling_result(args, stage_windows):
    if not args.monitor_scaling or not stage_windows:
        return

    print()
    print("=" * 100)
    print("CLOUD RUN AUTOSCALING CHECK")
    print("=" * 100)
    print(
        f"Waiting {args.metrics_wait:.0f}s once for Cloud Monitoring "
        "instance-count data..."
    )

    await asyncio.sleep(args.metrics_wait)

    try:
        samples = get_instance_count_samples(
            project_id=args.project_id,
            service_name=args.service_name,
            region=args.region,
            start_time=stage_windows[0]["start"],
            end_time=stage_windows[-1]["end"],
        )
    except Exception as exc:
        print(
            f"[monitoring error] {type(exc).__name__}: {exc}"
        )
        return

    if not samples:
        print("No Cloud Run instance-count samples found for this test window.")
        return

    first_two = None

    print()
    print("Instance-count samples:")

    for sample in samples:
        concurrency, relation = concurrency_at_time(
            sample["timestamp"],
            stage_windows,
        )

        print(
            f"  {sample['timestamp'].isoformat()} | "
            f"instances={sample['count']} | "
            f"mapped_concurrency={concurrency} ({relation})"
        )

        if first_two is None and sample["count"] >= 2:
            first_two = {
                "timestamp": sample["timestamp"],
                "count": sample["count"],
                "concurrency": concurrency,
                "relation": relation,
            }

    print()
    print("-" * 100)

    if first_two is None:
        print(
            "RESULT: No second Cloud Run instance was observed "
            "during the tested concurrency range."
        )
    else:
        print(
            f"FIRST OBSERVED SCALE TO 2+ INSTANCES: "
            f"concurrency={first_two['concurrency']}"
        )
        print(
            f"Metric timestamp: {first_two['timestamp'].isoformat()}"
        )
        print(
            f"Observed instances: {first_two['count']}"
        )

    print("-" * 100)
    print(
        "Note: Cloud Monitoring instance_count is sampled periodically, "
        "so this is the first observed concurrency associated with pod 2, "
        "not a millisecond-exact autoscaler trigger."
    )
    print("=" * 100)


async def http_one(
    client,
    start_event,
    item,
    concurrency,
    round_no,
    request_id,
    endpoint,
    language,
):
    r = Result(
        mode="http",
        stage_concurrency=concurrency,
        round_no=round_no,
        request_id=request_id,
        chunk_id=item["chunk_id"],
        use_case=item["use_case"],
        reference=item["reference"],
    )

    await start_event.wait()
    started = time.perf_counter()

    try:
        audio_path = Path(item["audio_path"])
        audio_bytes = audio_path.read_bytes()

        response = await client.post(
            endpoint,
            files={
                "file": (
                    audio_path.name,
                    audio_bytes,
                    "audio/wav",
                )
            },
            data={
                "model": MODEL,
                "language": language,
                "response_format": "json",
            },
        )

        r.latency_sec = time.perf_counter() - started
        r.status_code = response.status_code

        if response.status_code != 200:
            r.error = f"HTTP {response.status_code}: {response.text[:500]}"
            return r

        try:
            body = response.json()
            text = body.get("text", "") if isinstance(body, dict) else ""
        except Exception:
            text = response.text

        r.transcript = str(text).strip()
        r.success = bool(r.transcript)
        if not r.success:
            r.error = "HTTP 200 but empty transcript"
        else:
            r.wer_pct = wer(r.reference, r.transcript)

    except Exception as exc:
        r.latency_sec = time.perf_counter() - started
        r.error = f"{type(exc).__name__}: {exc}"

    return r


async def ws_one(
    start_event,
    item,
    concurrency,
    round_no,
    request_id,
    ws_url,
    language,
):
    r = Result(
        mode="ws",
        stage_concurrency=concurrency,
        round_no=round_no,
        request_id=request_id,
        chunk_id=item["chunk_id"],
        use_case=item["use_case"],
        reference=item["reference"],
    )

    audio_path = Path(item["audio_path"])

    with wave.open(str(audio_path), "rb") as wf:
        if wf.getframerate() != SAMPLE_RATE or wf.getnchannels() != 1 or wf.getsampwidth() != 2:
            r.error = "Chunk must be 16 kHz mono PCM16 WAV"
            return r
        raw_audio = wf.readframes(wf.getnframes())

    audio_chunks = [
        raw_audio[i:i + CHUNK_BYTES]
        for i in range(0, len(raw_audio), CHUNK_BYTES)
    ]

    await start_event.wait()
    started = time.perf_counter()
    first_text_at = None
    finals = []

    try:
        async with websockets.connect(
            ws_url,
            ping_interval=20,
            ping_timeout=30,
            close_timeout=10,
            open_timeout=30,
            max_size=None,
        ) as ws:
            await ws.send(json.dumps({
                "backend": "nemotron",
                "sample_rate": SAMPLE_RATE,
                "language": language,
            }))

            done = asyncio.Event()

            async def receiver():
                nonlocal first_text_at
                try:
                    async for raw in ws:
                        if isinstance(raw, bytes):
                            continue
                        try:
                            msg = json.loads(raw)
                        except json.JSONDecodeError:
                            continue

                        typ = msg.get("type", "")
                        text = str(msg.get("text", "")).strip()

                        if typ in ("partial", "final") and text and first_text_at is None:
                            first_text_at = time.perf_counter()

                        if typ == "final" and text:
                            finals.append(text)

                        elif typ == "error":
                            r.error = f"server error: {msg}"

                        elif typ == "done":
                            done.set()
                            return

                except ConnectionClosed as exc:
                    if not r.error:
                        r.error = f"connection closed code={exc.code} reason={exc.reason}"
                    done.set()

            recv_task = asyncio.create_task(receiver())

            stream_start = time.perf_counter()

            for i, chunk in enumerate(audio_chunks):
                if recv_task.done():
                    break

                await ws.send(chunk)

                # Real-time 100 ms pacing.
                expected = (i + 1) * CHUNK_MS / 1000.0
                actual = time.perf_counter() - stream_start
                if expected > actual:
                    await asyncio.sleep(expected - actual)

            if not r.error:
                try:
                    await ws.send(json.dumps({"type": "eof"}))
                    await asyncio.wait_for(done.wait(), timeout=20)
                except asyncio.TimeoutError:
                    r.error = "timeout waiting for done"
                except ConnectionClosed as exc:
                    r.error = f"EOF connection closed code={exc.code}"

            if not recv_task.done():
                recv_task.cancel()
                try:
                    await recv_task
                except asyncio.CancelledError:
                    pass

        r.latency_sec = time.perf_counter() - started
        if first_text_at is not None:
            r.ttfb_ms = (first_text_at - started) * 1000.0

        r.transcript = " ".join(finals).strip()
        if r.transcript and not r.error:
            r.success = True
            r.wer_pct = wer(r.reference, r.transcript)
        elif not r.error:
            r.error = "no final transcript"

    except Exception as exc:
        r.latency_sec = time.perf_counter() - started
        r.error = f"{type(exc).__name__}: {exc}"

    return r


def print_stage(results):
    mode = results[0].mode
    concurrency = results[0].stage_concurrency
    ok = [r for r in results if r.success]
    bad = [r for r in results if not r.success]

    wers = [r.wer_pct for r in ok if r.wer_pct is not None]
    lats = [r.latency_sec for r in ok]
    ttfbs = [r.ttfb_ms for r in ok if r.ttfb_ms is not None]

    print("\n" + "=" * 100)
    print(
        f"{mode.upper()} | CONCURRENCY={concurrency} | "
        f"success={len(ok)}/{len(results)} | failures={len(bad)}"
    )
    if lats:
        print(
            f"Latency: p50={percentile(lats,50):.2f}s "
            f"p95={percentile(lats,95):.2f}s "
            f"max={max(lats):.2f}s"
        )
    if ttfbs:
        print(
            f"TTFB:    p50={percentile(ttfbs,50):.0f}ms "
            f"p95={percentile(ttfbs,95):.0f}ms"
        )
    if wers:
        print(
            f"WER:     avg={statistics.mean(wers):.2f}% "
            f"p95={percentile(wers,95):.2f}% "
            f"worst={max(wers):.2f}%"
        )
    print("=" * 100)

    for r in results:
        print(
            f"\n[{'PASS' if r.success else 'FAIL'}] "
            f"req={r.request_id} chunk={r.chunk_id} use_case={r.use_case}"
        )
        print(
            f"latency={r.latency_sec:.2f}s "
            f"TTFB={('-' if r.ttfb_ms is None else f'{r.ttfb_ms:.0f}ms')} "
            f"WER={('-' if r.wer_pct is None else f'{r.wer_pct:.2f}%')}"
        )
        print("REFERENCE :")
        print(r.reference)
        print("\nTRANSCRIPT:")
        print(r.transcript or "<EMPTY>")
        if r.error:
            print(f"\nERROR: {r.error}")
        print("-" * 100)


async def run_stage(
    mode,
    concurrency,
    round_no,
    selected,
    args,
    http_url,
    ws_url,
    http_client,
):
    start_event = asyncio.Event()
    tasks = []

    for i, item in enumerate(selected, start=1):
        if mode == "http":
            tasks.append(asyncio.create_task(
                http_one(
                    http_client,
                    start_event,
                    item,
                    concurrency,
                    round_no,
                    i,
                    http_url,
                    args.language,
                )
            ))
        else:
            tasks.append(asyncio.create_task(
                ws_one(
                    start_event,
                    item,
                    concurrency,
                    round_no,
                    i,
                    ws_url,
                    args.language,
                )
            ))

    # Barrier: workers are created before releasing them together.
    await asyncio.sleep(0.25)
    start_event.set()
    results = await asyncio.gather(*tasks)
    print_stage(results)
    return results


async def run_mode(mode, items, args, http_url, ws_url, stage_windows):
    rng = random.Random(args.seed)
    all_results = []

    if args.until_failure:
        max_requested_concurrency = args.max_concurrency
    else:
        max_requested_concurrency = max(
            [int(x) for x in args.ramp.split(",") if x.strip()] + [1]
        )

    max_connections = max_requested_concurrency + 10

    timeout = httpx.Timeout(
        connect=30,
        read=args.timeout,
        write=args.timeout,
        pool=args.timeout,
    )
    limits = httpx.Limits(
        max_connections=max_connections,
        max_keepalive_connections=max_connections,
    )

    async with httpx.AsyncClient(
        timeout=timeout,
        limits=limits,
        follow_redirects=True,
        http2=False,
    ) as client:
        baseline_avg_wer = None

        if args.until_failure:
            levels = list(
                range(
                    args.start_concurrency,
                    args.max_concurrency + 1,
                    args.step,
                )
            )
        else:
            levels = [
                int(x.strip())
                for x in args.ramp.split(",")
                if x.strip()
            ]

        last_full_success = None
        first_failure = None

        for concurrency in levels:
            stage_results = []
            stage_started = datetime.now(timezone.utc)

            for round_no in range(1, args.rounds + 1):
                if concurrency <= len(items):
                    selected = rng.sample(items, concurrency)
                else:
                    # If requested concurrency exceeds available unique chunks,
                    # use all unique chunks first and then sample additional ones.
                    selected = list(items)
                    while len(selected) < concurrency:
                        selected.append(rng.choice(items))
                    rng.shuffle(selected)

                results = await run_stage(
                    mode,
                    concurrency,
                    round_no,
                    selected,
                    args,
                    http_url,
                    ws_url,
                    client,
                )
                stage_results.extend(results)
                all_results.extend(results)

                if round_no < args.rounds:
                    await asyncio.sleep(args.round_gap)

            stage_ended = datetime.now(timezone.utc)
            stage_windows.append(
                {
                    "mode": mode,
                    "concurrency": concurrency,
                    "start": stage_started,
                    "end": stage_ended,
                }
            )

            ok = [r for r in stage_results if r.success]
            wers = [r.wer_pct for r in ok if r.wer_pct is not None]

            if concurrency == 1 and wers:
                baseline_avg_wer = statistics.mean(wers)

            failed = len(ok) != len(stage_results)
            degraded = False
            if baseline_avg_wer is not None and wers:
                avg_wer = statistics.mean(wers)
                degraded = avg_wer > baseline_avg_wer + args.max_wer_regression

            if failed:
                first_failure = concurrency
                print(
                    f"\n[FAILURE BOUNDARY] Failures detected at concurrency={concurrency}"
                )
                print(
                    f"[LAST FULL SUCCESS] {last_full_success if last_full_success is not None else 'none'}"
                )

                if args.until_failure and args.confirm_failure:
                    print(
                        f"\n[CONFIRM] Re-running concurrency={concurrency} once..."
                    )
                    await asyncio.sleep(args.rest)

                    if concurrency <= len(items):
                        selected = rng.sample(items, concurrency)
                    else:
                        selected = list(items)
                        while len(selected) < concurrency:
                            selected.append(rng.choice(items))
                        rng.shuffle(selected)

                    confirm_results = await run_stage(
                        mode,
                        concurrency,
                        args.rounds + 1,
                        selected,
                        args,
                        http_url,
                        ws_url,
                        client,
                    )
                    all_results.extend(confirm_results)

                    confirm_ok = sum(r.success for r in confirm_results)
                    print(
                        f"[CONFIRM RESULT] {confirm_ok}/{len(confirm_results)} successful"
                    )

                if args.until_failure or args.stop_on_failure:
                    break

            else:
                last_full_success = concurrency

                if degraded:
                    print(
                        f"\n[ACCURACY WARNING] Average WER regressed by more than "
                        f"{args.max_wer_regression:.1f} points at concurrency={concurrency}, "
                        f"but requests are still completing."
                    )

            await asyncio.sleep(args.rest)

        if args.until_failure:
            print("\n" + "#" * 90)
            print(f"AUTO-RAMP RESULT ({mode.upper()})")
            print(f"Last fully successful concurrency : {last_full_success}")
            print(f"First failing concurrency         : {first_failure}")
            if first_failure is None:
                print(
                    f"No failure was reached before the safety ceiling "
                    f"of {args.max_concurrency}."
                )
            print("#" * 90)

    return all_results


def save_reports(results, prefix):
    ts = time.strftime("%Y%m%d_%H%M%S")
    json_path = Path(f"{prefix}_{ts}.json")
    csv_path = Path(f"{prefix}_{ts}.csv")

    json_path.write_text(
        json.dumps([asdict(r) for r in results], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if results:
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
            writer.writeheader()
            for r in results:
                writer.writerow(asdict(r))

    print(f"\nSaved JSON: {json_path}")
    print(f"Saved CSV : {csv_path}")


async def main_async(args):
    items = load_manifest(Path(args.manifest), args.include_review)
    http_url, ws_url = derive_urls(args.base_url)

    print(f"Loaded chunks : {len(items)}")
    print(f"HTTP endpoint : {http_url}")
    print(f"WS endpoint   : {ws_url}")
    if args.until_failure:
        print(
            f"Auto ramp     : {args.start_concurrency} -> "
            f"{args.max_concurrency} step {args.step}, until first failure"
        )
    else:
        print(f"Ramp          : {args.ramp}")
    print(f"Rounds        : {args.rounds}")

    all_results = []
    stage_windows = []

    if args.mode in ("http", "both"):
        print("\n\n######## HTTP LOAD TEST ########")
        all_results.extend(
            await run_mode("http", items, args, http_url, ws_url, stage_windows)
        )

    if args.mode in ("ws", "both"):
        print("\n\n######## WEBSOCKET LOAD TEST ########")
        all_results.extend(
            await run_mode("ws", items, args, http_url, ws_url, stage_windows)
        )

    save_reports(all_results, "nemotron_loadtest")
    await show_scaling_result(args, stage_windows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["http", "ws", "both"], required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--language", default="en-US")
    ap.add_argument(
        "--ramp",
        default="1,2,3,4,5,6,8,10",
        help="Manual comma-separated concurrency levels. Ignored with --until-failure.",
    )
    ap.add_argument(
        "--until-failure",
        action="store_true",
        help="Automatically increase concurrency until the first failed stage.",
    )
    ap.add_argument("--start-concurrency", type=int, default=1)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument(
        "--max-concurrency",
        type=int,
        default=50,
        help="Safety ceiling for auto-ramp so Cloud Run autoscaling cannot run forever.",
    )
    ap.add_argument(
        "--confirm-failure",
        action="store_true",
        help="Repeat the first failing concurrency once to confirm the boundary.",
    )
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--round-gap", type=float, default=2.0)
    ap.add_argument("--rest", type=float, default=5.0)
    ap.add_argument("--timeout", type=float, default=230.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--include-review", action="store_true")
    ap.add_argument("--stop-on-failure", action="store_true")
    ap.add_argument(
        "--max-wer-regression",
        type=float,
        default=3.0,
        help="Warn if avg WER rises this many absolute points over concurrency=1",
    )
    ap.add_argument(
        "--monitor-scaling",
        action="store_true",
        help="After the normal load test, read Cloud Run instance-count metrics.",
    )
    ap.add_argument(
        "--project-id",
        default="",
        help="GCP project ID. Required with --monitor-scaling.",
    )
    ap.add_argument(
        "--service-name",
        default="nemotron-3-5",
    )
    ap.add_argument(
        "--region",
        default="us-central1",
    )
    ap.add_argument(
        "--metrics-wait",
        type=float,
        default=130.0,
        help="Single wait after the entire test for Cloud Monitoring metric visibility.",
    )

    args = ap.parse_args()

    if args.monitor_scaling and not args.project_id:
        ap.error("--project-id is required when --monitor-scaling is used")

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
