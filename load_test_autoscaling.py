#!/usr/bin/env python3
import argparse
import asyncio
import json
import random
import re
import statistics
import time
import wave
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import httpx
import websockets
from google.cloud import monitoring_v3
from google.protobuf.timestamp_pb2 import Timestamp
from websockets.exceptions import ConnectionClosed

MODEL = "nemotron-3.5-asr-streaming-0.6b"
SAMPLE_RATE = 16000
CHUNK_MS = 100
CHUNK_BYTES = int(SAMPLE_RATE * CHUNK_MS / 1000) * 2

@dataclass
class Result:
    mode: str
    concurrency: int
    worker_id: int
    chunk_id: str
    success: bool
    latency_sec: float
    wer_pct: Optional[float]
    status_code: int = 0
    ttfb_ms: Optional[float] = None
    reference: str = ""
    transcript: str = ""
    error: str = ""

def norm(text):
    text = text.lower()
    text = re.sub(r"<[a-z]{2}-[A-Z]{2}>", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip().split()

def wer(ref_text, hyp_text):
    ref, hyp = norm(ref_text), norm(hyp_text)
    if not ref:
        return 0.0 if not hyp else 100.0
    prev = list(range(len(hyp) + 1))
    for i, rw in enumerate(ref, 1):
        cur = [i]
        for j, hw in enumerate(hyp, 1):
            cost = 0 if rw == hw else 1
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j-1] + cost))
        prev = cur
    return 100.0 * prev[-1] / len(ref)

def percentile(values, p):
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    idx = (len(vals)-1) * p / 100.0
    lo, hi = int(idx), min(int(idx)+1, len(vals)-1)
    f = idx - lo
    return vals[lo]*(1-f) + vals[hi]*f

def load_manifest(path, include_review=False):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            x = json.loads(line)
            if not include_review and x.get("status") != "OK":
                continue
            ap = Path(x["audio_path"])
            rp = Path(x["reference_path"])
            if not ap.exists() or not rp.exists():
                continue
            x["reference"] = rp.read_text(encoding="utf-8").strip()
            out.append(x)
    if not out:
        raise RuntimeError("No usable chunks in manifest.")
    return out

def derive_urls(base):
    base = base.rstrip("/")
    p = urlparse(base)
    if p.scheme not in ("http", "https"):
        raise ValueError("--base-url must start with http:// or https://")
    if base.endswith("/v1"):
        http_url = base + "/audio/transcriptions"
        root = base[:-3]
    else:
        http_url = base + "/v1/audio/transcriptions"
        root = base
    p = urlparse(root)
    ws_url = f"{'wss' if p.scheme == 'https' else 'ws'}://{p.netloc}/asr/realtime-custom-vad"
    return http_url, ws_url

def proto_ts(dt):
    return Timestamp(seconds=int(dt.timestamp()))

def metric_series(project_id, metric_type, service, region, start, end):
    client = monitoring_v3.MetricServiceClient()
    filt = (
        f'metric.type="{metric_type}" '
        'AND resource.type="cloud_run_revision" '
        f'AND resource.labels.service_name="{service}" '
        f'AND resource.labels.location="{region}"'
    )
    return list(client.list_time_series(request={
        "name": f"projects/{project_id}",
        "filter": filt,
        "interval": monitoring_v3.TimeInterval({
            "start_time": proto_ts(start),
            "end_time": proto_ts(end),
        }),
        "view": monitoring_v3.ListTimeSeriesRequest.TimeSeriesView.FULL,
    }))

def read_scaling(project_id, service, region, stage_start, stage_end):
    start = stage_start - timedelta(seconds=30)
    end = stage_end + timedelta(seconds=30)

    actual = metric_series(
        project_id, "run.googleapis.com/container/instance_count",
        service, region, start, end
    )
    recommended = metric_series(
        project_id, "run.googleapis.com/scaling/recommended_instances",
        service, region, start, end
    )

    actual_by_t = defaultdict(lambda: defaultdict(int))
    for ts in actual:
        state = ts.metric.labels.get("state", "unknown")
        for pt in ts.points:
            t = pt.interval.end_time.timestamp()
            actual_by_t[t][state] += pt.value.int64_value

    rec_by_t = defaultdict(lambda: defaultdict(int))
    for ts in recommended:
        driver = ts.metric.labels.get("scaling_driver", "unknown")
        for pt in ts.points:
            t = pt.interval.end_time.timestamp()
            rec_by_t[t][driver] += pt.value.int64_value

    actual_samples = []
    for t, states in sorted(actual_by_t.items()):
        actual_samples.append({
            "timestamp": datetime.fromtimestamp(t, timezone.utc).isoformat(),
            "total": sum(states.values()),
            "states": dict(states),
        })

    rec_samples = []
    for t, drivers in sorted(rec_by_t.items()):
        rec_samples.append({
            "timestamp": datetime.fromtimestamp(t, timezone.utc).isoformat(),
            "max_recommended": max(drivers.values()) if drivers else 0,
            "drivers": dict(drivers),
        })

    return {
        "max_actual_instances": max((x["total"] for x in actual_samples), default=0),
        "max_recommended_instances": max((x["max_recommended"] for x in rec_samples), default=0),
        "actual_samples": actual_samples,
        "recommended_samples": rec_samples,
    }

async def http_worker(worker_id, concurrency, item, client, endpoint, language, deadline):
    results = []
    audio_path = Path(item["audio_path"])
    audio_bytes = audio_path.read_bytes()

    while time.perf_counter() < deadline:
        started = time.perf_counter()
        try:
            resp = await client.post(
                endpoint,
                files={"file": (audio_path.name, audio_bytes, "audio/wav")},
                data={
                    "model": MODEL,
                    "language": language,
                    "response_format": "json",
                },
            )
            latency = time.perf_counter() - started
            transcript = ""
            error = ""
            if resp.status_code == 200:
                try:
                    body = resp.json()
                    transcript = str(body.get("text", "")).strip() if isinstance(body, dict) else resp.text.strip()
                except Exception:
                    transcript = resp.text.strip()
            else:
                error = f"HTTP {resp.status_code}: {resp.text[:300]}"

            success = resp.status_code == 200 and bool(transcript)
            results.append(Result(
                "http", concurrency, worker_id, item["chunk_id"], success,
                latency, wer(item["reference"], transcript) if success else None,
                status_code=resp.status_code,
                reference=item["reference"], transcript=transcript,
                error=error if error else ("" if success else "empty transcript")
            ))
        except Exception as exc:
            results.append(Result(
                "http", concurrency, worker_id, item["chunk_id"], False,
                time.perf_counter() - started, None,
                reference=item["reference"],
                error=f"{type(exc).__name__}: {exc}"
            ))
    return results

async def ws_worker(worker_id, concurrency, item, ws_url, language, stage_seconds):
    r = Result("ws", concurrency, worker_id, item["chunk_id"], False, 0.0, None)
    p = Path(item["audio_path"])

    with wave.open(str(p), "rb") as wf:
        if wf.getframerate() != SAMPLE_RATE or wf.getnchannels() != 1 or wf.getsampwidth() != 2:
            r.error = "Chunk must be 16 kHz mono PCM16 WAV"
            return r
        raw = wf.readframes(wf.getnframes())

    chunks = [raw[i:i+CHUNK_BYTES] for i in range(0, len(raw), CHUNK_BYTES)]
    started = time.perf_counter()
    deadline = started + stage_seconds
    finals, refs = [], []
    first_text = None

    try:
        async with websockets.connect(
            ws_url, ping_interval=20, ping_timeout=30,
            close_timeout=10, open_timeout=30, max_size=None
        ) as ws:
            await ws.send(json.dumps({
                "backend": "nemotron",
                "sample_rate": SAMPLE_RATE,
                "language": language,
            }))
            done = asyncio.Event()

            async def recv():
                nonlocal first_text
                try:
                    async for raw_msg in ws:
                        if isinstance(raw_msg, bytes):
                            continue
                        try:
                            msg = json.loads(raw_msg)
                        except json.JSONDecodeError:
                            continue
                        typ = msg.get("type", "")
                        text = str(msg.get("text", "")).strip()
                        if typ in ("partial", "final") and text and first_text is None:
                            first_text = time.perf_counter()
                        if typ == "final" and text:
                            finals.append(text)
                        elif typ == "error":
                            r.error = f"server error: {msg}"
                        elif typ == "done":
                            done.set()
                            return
                except ConnectionClosed as exc:
                    r.error = f"connection closed code={exc.code} reason={exc.reason}"
                    done.set()

            recv_task = asyncio.create_task(recv())

            while time.perf_counter() < deadline and not recv_task.done():
                refs.append(item["reference"])
                loop_start = time.perf_counter()

                for i, chunk in enumerate(chunks):
                    if time.perf_counter() >= deadline or recv_task.done():
                        break
                    await ws.send(chunk)
                    expected = (i + 1) * CHUNK_MS / 1000.0
                    actual = time.perf_counter() - loop_start
                    if expected > actual:
                        await asyncio.sleep(expected - actual)

                if time.perf_counter() < deadline and not recv_task.done():
                    await ws.send(bytes(int(SAMPLE_RATE * 0.30) * 2))
                    await asyncio.sleep(0.30)

            if not r.error:
                try:
                    await ws.send(json.dumps({"type": "eof"}))
                    await asyncio.wait_for(done.wait(), timeout=20)
                except Exception as exc:
                    r.error = f"{type(exc).__name__}: {exc}"

            if not recv_task.done():
                recv_task.cancel()
                try:
                    await recv_task
                except asyncio.CancelledError:
                    pass

    except Exception as exc:
        r.error = f"{type(exc).__name__}: {exc}"

    r.latency_sec = time.perf_counter() - started
    r.ttfb_ms = ((first_text - started) * 1000.0) if first_text else None
    r.reference = " ".join(refs).strip()
    r.transcript = " ".join(finals).strip()
    r.success = bool(r.transcript) and not r.error
    r.wer_pct = wer(r.reference, r.transcript) if r.success else None
    return r

def summarize(results, mode, concurrency):
    ok = [r for r in results if r.success]
    bad = [r for r in results if not r.success]
    lats = [r.latency_sec for r in ok]
    wers = [r.wer_pct for r in ok if r.wer_pct is not None]
    ttfbs = [r.ttfb_ms for r in ok if r.ttfb_ms is not None]

    print("\n" + "="*90)
    print(f"{mode.upper()} CONCURRENCY={concurrency}")
    print(f"Success: {len(ok)}/{len(results)}  Failures: {len(bad)}")
    if lats:
        print(f"Latency p50/p95: {percentile(lats,50):.2f}s / {percentile(lats,95):.2f}s")
    if ttfbs:
        print(f"TTFB p50/p95: {percentile(ttfbs,50):.0f}ms / {percentile(ttfbs,95):.0f}ms")
    if wers:
        print(f"WER avg/p95/worst: {statistics.mean(wers):.2f}% / {percentile(wers,95):.2f}% / {max(wers):.2f}%")
    for r in bad[:10]:
        print(f"FAIL worker={r.worker_id} chunk={r.chunk_id} error={r.error}")
    print("="*90)

    return {
        "successes": len(ok),
        "failures": len(bad),
        "observations": len(results),
        "latency_p50_sec": percentile(lats, 50),
        "latency_p95_sec": percentile(lats, 95),
        "ttfb_p50_ms": percentile(ttfbs, 50),
        "ttfb_p95_ms": percentile(ttfbs, 95),
        "wer_avg_pct": statistics.mean(wers) if wers else None,
        "wer_p95_pct": percentile(wers, 95),
        "wer_worst_pct": max(wers) if wers else None,
    }

async def run(args):
    items = load_manifest(Path(args.manifest), args.include_review)
    http_url, ws_url = derive_urls(args.base_url)
    rng = random.Random(args.seed)
    levels = list(range(args.start_concurrency, args.max_concurrency + 1, args.step))

    print(f"HTTP: {http_url}")
    print(f"WS:   {ws_url}")
    print(f"Stages: {levels}")
    print(f"Stage seconds: {args.stage_seconds}")
    print(f"Metrics delay: {args.metrics_delay}")

    report = []
    last_one = None
    first_rec_two = None
    first_actual_two = None

    for concurrency in levels:
        print(f"\n### START CONCURRENCY {concurrency} ###")
        stage_start = datetime.now(timezone.utc)

        selected = [rng.choice(items) for _ in range(concurrency)]

        if args.mode == "http":
            max_conn = args.max_concurrency + 10
            limits = httpx.Limits(max_connections=max_conn, max_keepalive_connections=max_conn)
            timeout = httpx.Timeout(
                connect=30, read=args.request_timeout,
                write=args.request_timeout, pool=args.request_timeout
            )
            deadline = time.perf_counter() + args.stage_seconds
            async with httpx.AsyncClient(
                timeout=timeout, limits=limits,
                follow_redirects=True, http2=False
            ) as client:
                nested = await asyncio.gather(*[
                    http_worker(i+1, concurrency, selected[i], client, http_url, args.language, deadline)
                    for i in range(concurrency)
                ])
            results = [r for group in nested for r in group]
        else:
            results = await asyncio.gather(*[
                ws_worker(i+1, concurrency, selected[i], ws_url, args.language, args.stage_seconds)
                for i in range(concurrency)
            ])

        stage_end = datetime.now(timezone.utc)
        load_summary = summarize(results, args.mode, concurrency)

        print(f"\nWaiting {args.metrics_delay}s for Cloud Monitoring...")
        await asyncio.sleep(args.metrics_delay)

        try:
            scaling = read_scaling(
                args.project_id, args.service_name, args.region,
                stage_start, stage_end
            )
        except Exception as exc:
            scaling = {
                "max_actual_instances": 0,
                "max_recommended_instances": 0,
                "actual_samples": [],
                "recommended_samples": [],
                "error": f"{type(exc).__name__}: {exc}",
            }

        actual = scaling["max_actual_instances"]
        recommended = scaling["max_recommended_instances"]

        print("\n--- CLOUD RUN SCALING ---")
        print(f"Concurrency:              {concurrency}")
        print(f"Max actual instances:     {actual}")
        print(f"Max recommended instances:{recommended}")

        for s in scaling.get("recommended_samples", []):
            print(f"REC {s['timestamp']} -> {s['max_recommended']} {s['drivers']}")
        for s in scaling.get("actual_samples", []):
            print(f"ACT {s['timestamp']} -> {s['total']} {s['states']}")

        if 0 < actual <= 1:
            last_one = concurrency

        if recommended >= 2 and first_rec_two is None:
            first_rec_two = concurrency
            print(f"\n>>> AUTOSCALER RECOMMENDED 2+ AT CONCURRENCY {concurrency} <<<")

        if actual >= 2 and first_actual_two is None:
            first_actual_two = concurrency
            print(f"\n>>> SECOND INSTANCE OBSERVED AT CONCURRENCY {concurrency} <<<")

        report.append({
            "concurrency": concurrency,
            "stage_start_utc": stage_start.isoformat(),
            "stage_end_utc": stage_end.isoformat(),
            "load_summary": load_summary,
            "scaling": scaling,
            "results": [asdict(r) for r in results],
        })

        if args.stop_when_scaled and actual >= 2:
            break

        await asyncio.sleep(args.rest_seconds)

    print("\n" + "#"*90)
    print("FINAL RESULT")
    print(f"Last concurrency observed with <=1 actual instance: {last_one}")
    print(f"First concurrency with recommended instances >=2:  {first_rec_two}")
    print(f"First concurrency with actual instances >=2:       {first_actual_two}")

    if last_one is not None and first_actual_two is not None and args.step > 1:
        print(
            f"Rerun exact range: --start-concurrency {last_one+1} "
            f"--max-concurrency {first_actual_two} --step 1"
        )
    print("#"*90)

    out = Path(f"nemotron_autoscaling_{args.mode}_{time.strftime('%Y%m%d_%H%M%S')}.json")
    out.write_text(json.dumps({
        "mode": args.mode,
        "project_id": args.project_id,
        "service_name": args.service_name,
        "region": args.region,
        "last_actual_one": last_one,
        "first_recommended_two": first_rec_two,
        "first_actual_two": first_actual_two,
        "stages": report,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved report: {out}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["http", "ws"], required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--project-id", required=True)
    ap.add_argument("--service-name", default="nemotron-3-5")
    ap.add_argument("--region", default="us-central1")
    ap.add_argument("--language", default="en-US")
    ap.add_argument("--start-concurrency", type=int, default=1)
    ap.add_argument("--step", type=int, default=5)
    ap.add_argument("--max-concurrency", type=int, default=60)
    ap.add_argument("--stage-seconds", type=float, default=90)
    ap.add_argument("--metrics-delay", type=float, default=135)
    ap.add_argument("--rest-seconds", type=float, default=10)
    ap.add_argument("--request-timeout", type=float, default=230)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--include-review", action="store_true")
    ap.add_argument("--stop-when-scaled", action="store_true")
    args = ap.parse_args()
    asyncio.run(run(args))

if __name__ == "__main__":
    main()
