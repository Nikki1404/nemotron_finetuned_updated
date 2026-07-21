
#!/usr/bin/env python3

import argparse
import asyncio
import json
import re
import sys
import time
import wave
from pathlib import Path
from urllib.parse import urlparse

import websockets

SERVER_URL = "ws://10.90.126.61:8003/asr/realtime-custom-vad"
#SERVER_URL = "wss://nemotron-finetuned-150916788856.us-central1.run.app/asr/realtime-custom-vad"

SAMPLE_RATE = 16000
CHUNK_MS = 100
CHUNK_BYTES = int(SAMPLE_RATE * CHUNK_MS / 1000) * 2

GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"

_LANG_TAG_RE = re.compile(r"<[a-z]{2}-[A-Z]{2}>\s*")


def clean_text(text: str) -> str:
    return _LANG_TAG_RE.sub("", text or "").strip()


def print_partial(text: str):
    sys.stdout.write(f"\r{YELLOW}[partial]{RESET} {text}    ")
    sys.stdout.flush()


def print_final(text: str, ttfb_ms=None):
    ttfb_str = f"  {DIM}(TTFB {ttfb_ms}ms){RESET}" if ttfb_ms else ""
    sys.stdout.write(f"\r{GREEN}{BOLD}[final]  {RESET}{GREEN}{text}{RESET}{ttfb_str}\n")
    sys.stdout.flush()


def print_info(msg: str):
    print(f"{CYAN}[info]{RESET} {msg}")


async def receive_loop(ws, stop_event: asyncio.Event):
    last_partial = ""

    try:
        async for raw in ws:
            if isinstance(raw, bytes):
                continue

            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            ev_type = msg.get("type", "")
            text = clean_text(msg.get("text", ""))
            ttfb = msg.get("t_start")

            if ev_type == "partial":
                if text:
                    last_partial = text
                    print_partial(text)

            elif ev_type == "final":
                if text:
                    print_final(text, ttfb)
                    last_partial = ""

            elif ev_type == "done":
                if last_partial:
                    print_final(last_partial, None)
                    last_partial = ""
                stop_event.set()
                break

            elif ev_type == "error":
                print(f"\n[server error] {text}")

    except Exception as e:
        print(f"\n[receive error] {e}")
        if last_partial:
            print_final(last_partial, None)

    finally:
        stop_event.set()


async def run_file(path: str, language: str, realtime: bool, url: str):
    wav_path = Path(path)

    if not wav_path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    with wave.open(str(wav_path), "rb") as wf:
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        file_sr = wf.getframerate()
        n_frames = wf.getnframes()
        raw_audio = wf.readframes(n_frames)

    print_info(f"File: {wav_path.name}")
    print_info(
        f"Audio: {file_sr}Hz {n_channels}ch {sample_width * 8}bit "
        f"{n_frames / file_sr:.1f}s"
    )
    print_info(f"Language: {language}")
    print_info(f"Realtime simulation: {realtime}")
    print_info(f"Connecting to {url}\n")

    import numpy as np

    audio_i16 = np.frombuffer(raw_audio, dtype=np.int16)

    if n_channels == 2:
        audio_i16 = audio_i16.reshape(-1, 2).mean(axis=1).astype(np.int16)

    if file_sr != SAMPLE_RATE:
        print_info(f"Resampling {file_sr}Hz → {SAMPLE_RATE}Hz")

        try:
            import resampy
        except ImportError:
            print("resampy not installed. Run: pip install resampy")
            sys.exit(1)

        audio_f32 = audio_i16.astype(np.float32) / 32768.0
        audio_f32 = resampy.resample(audio_f32, file_sr, SAMPLE_RATE)
        audio_i16 = (np.clip(audio_f32, -1.0, 1.0) * 32767).astype(np.int16)

    raw_bytes = audio_i16.tobytes()
    chunk_samples = int(SAMPLE_RATE * CHUNK_MS / 1000)
    chunk_bytes = chunk_samples * 2

    chunks = [
        raw_bytes[i:i + chunk_bytes]
        for i in range(0, len(raw_bytes), chunk_bytes)
    ]

    t_start = time.time()

    async with websockets.connect(url, ping_interval=None) as ws:
        await ws.send(json.dumps({
            "backend": "nemotron",
            "sample_rate": SAMPLE_RATE,
            "language": language,
        }))

        stop_event = asyncio.Event()
        recv_task = asyncio.create_task(receive_loop(ws, stop_event))

        try:
            for i, chunk in enumerate(chunks):
                await ws.send(chunk)

                if realtime:
                    expected_elapsed = (i + 1) * CHUNK_MS / 1000.0
                    actual_elapsed = time.time() - t_start
                    sleep_for = expected_elapsed - actual_elapsed

                    if sleep_for > 0:
                        await asyncio.sleep(sleep_for)
                else:
                    await asyncio.sleep(0.001)

        except KeyboardInterrupt:
            print_info("Interrupted while sending audio")

        print_info("\nFile sent — sending EOF and waiting for final results...")
        await ws.send(json.dumps({"type": "eof"}))

        try:
            await asyncio.wait_for(stop_event.wait(), timeout=60.0)
        except asyncio.TimeoutError:
            print_info("Timeout waiting for final response")

        recv_task.cancel()

        try:
            await recv_task
        except asyncio.CancelledError:
            pass

    elapsed = time.time() - t_start
    audio_sec = len(audio_i16) / SAMPLE_RATE
    rtf = elapsed / audio_sec if audio_sec > 0 else 0

    print_info(f"\nDone. Audio={audio_sec:.1f}s Wall={elapsed:.2f}s RTF={rtf:.2f}x")


async def run_mic(language: str, url: str):
    try:
        import sounddevice as sd
    except ImportError:
        print("sounddevice not installed. Run: pip install sounddevice")
        sys.exit(1)

    print_info(f"Connecting to {url}")
    print_info(f"Language: {language}")
    print_info("Speak into your microphone. Press Ctrl+C to stop.\n")

    async with websockets.connect(url, ping_interval=None) as ws:
        await ws.send(json.dumps({
            "backend": "nemotron",
            "sample_rate": SAMPLE_RATE,
            "language": language,
        }))

        stop_event = asyncio.Event()
        recv_task = asyncio.create_task(receive_loop(ws, stop_event))

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def audio_callback(indata, frames, time_info, status):
            pcm = (indata[:, 0] * 32767).astype("int16").tobytes()
            loop.call_soon_threadsafe(queue.put_nowait, pcm)

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=int(SAMPLE_RATE * CHUNK_MS / 1000),
            callback=audio_callback,
        ):
            try:
                while not stop_event.is_set():
                    try:
                        pcm = await asyncio.wait_for(queue.get(), timeout=0.5)
                        await ws.send(pcm)
                    except asyncio.TimeoutError:
                        continue
            except KeyboardInterrupt:
                print_info("Stopping...")

        await ws.send(json.dumps({"type": "eof"}))

        try:
            await asyncio.wait_for(stop_event.wait(), timeout=60.0)
        except asyncio.TimeoutError:
            print_info("Timeout waiting for final response")

        recv_task.cancel()

        try:
            await recv_task
        except asyncio.CancelledError:
            pass


async def check_health(url: str):
    try:
        import urllib.request

        parsed = urlparse(url)
        http_host = f"{'https' if parsed.scheme == 'wss' else 'http'}://{parsed.netloc}"

        with urllib.request.urlopen(f"{http_host}/health", timeout=5) as r:
            data = json.loads(r.read())

        print_info(f"Server health: {data}")
        return True

    except Exception as e:
        print(f"[warn] Health check failed: {e} (server may still be starting)")
        return False


def main():
    parser = argparse.ArgumentParser(description="Nemotron ASR WebSocket client")

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--mic", action="store_true")
    mode.add_argument("--file", metavar="PATH")

    parser.add_argument("--language", default="en-US")
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--url", default=SERVER_URL)
    parser.add_argument("--health", action="store_true")

    args = parser.parse_args()

    if args.health:
        asyncio.run(check_health(args.url))
        return

    asyncio.run(check_health(args.url))

    if args.mic:
        asyncio.run(run_mic(args.language, args.url))
    else:
        asyncio.run(run_file(args.file, args.language, args.realtime, args.url))


if __name__ == "__main__":
    main()

Hi Nikita,
 
Here's the background of our hackathon task:
 
Ghost Denials uses a counterfactual equivalence audit to make “silent” wrongful automated denials measurable and actionable.
Starting with health-plan/insurance claim and prior-authorization auto-denials (excluding manual denials, fraud-flagged cases, 
and incomplete-submission rejections), it learns the decision-relevant structured feature set from historical adjudication logs 
(e.g., diagnosis/procedure codes, plan/benefit terms, clinical criteria, dollar bands, network status) and forms equivalence classes 
over those features. For each auto-denied case, it retrieves approved “near-twins” via propensity-score matching on the policy’s own 
decision function. If approved twins are statistically indistinguishable across all decision-relevant variables, the denial is flagged 
as a candidate wrongful denial and shipped with a minimal counterexample dossier (“this near-identical case was paid”). Risk of unobserved 
confounding is mitigated by requiring agreement on documentation-completeness and manual-review flags, excluding any case touched by manual 
clinical review, and validating the method against known ground truth. Ground truth comes from denials later overturned on appeal; the matcher 
is validated on these known outcomes before being applied to un-appealed denials. A conformal prediction layer provides calibrated confidence 
with a bounded false-positive rate (tunable to reviewer capacity). Disparate-impact tests run across protected attributes to surface bias, and 
matched data is also used to infer the decision boundary actually enforced and diff it against written benefit/medical policy to surface 
systemic rule divergences. The system is read-only/offline and produces a prioritized human-review queue; clinical/appeals staff approve any 
correction. An LLM is used only to draft plain-language review/appeal dossiers from structured evidence, not to render the wrongful-denial 
judgment.

Business Outcomes-
ROI is proven immediately via a “back-book” scan of the last 12–24 months of auto-denials to surface high-confidence candidates, 
route them to clinical/appeals review, and correct them—then run continuously to detect new issues within days instead of only on 
appeal or never. The business case is supported by observed high overturn rates on appealed denials (e.g., Medicare Advantage prior-auth 
appeal overturns reported >80% in some CMS reporting); if even a fraction holds for the ~99% who never appeal, a mid-size plan could have 
tens of thousands of wrongful denials unmeasured today. Targets/KPIs: detect ≥80% of wrongful auto-denials at a bounded ≤10% false-positive 
rate (conformal-calibrated; tunable to reviewer capacity); reduce time-to-detection from “never/on-appeal” to days; quantify dollars and 
members recovered; and quantify/flag disparate-impact exposure. Measurement plan: offline precision/recall against appeal-overturn ground 
truth plus reviewer-confirmed precision on flagged queues; then track live wrongful-denial rate and overturn rate pre/post. Each “leak pattern 
class” (e.g., criteria mismatch, dollar-band edge cases, network-status errors) ships with acceptance tests and per-class error budgets for 
governed precision and regression testing. Commercial model: per-recovered-decision fee + continuous-assurance subscription + pattern-pack 
licensing.
 
I did a basic research and here are few things we can start with, unless you have any other idea:
 
dataset: Synthetic AR Medical Dataset with Realistic Denial
Algorithms:
Exact Matching
logistic regression or gradient boosting.
k-Nearest Neighbors
FAISS/embeddings
 
Synthetic AR Medical Billing Dataset with Realistic Denial Workflow

https://www.kaggle.com/datasets/abuthahir1998/synthetic-ar-medical-dataset-with-realistic-denial
