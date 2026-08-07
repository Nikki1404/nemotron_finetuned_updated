#!/usr/bin/env python3
"""
prepare_chunks_local.py

Local-only dataset preparation for Nemotron load testing.

What it does:
1) Reads inspira_transcripts.csv (columns: use_case, transcript)
2) Fuzzy-matches each CSV use_case to the corresponding WAV filename
3) Resamples WAVs locally to 16 kHz mono PCM16
4) Splits audio into ~10 s chunks, preferring low-energy boundaries
5) Sends EACH chunk sequentially (not load testing) to your existing
   OpenAI-compatible /v1/audio/transcriptions endpoint to obtain a draft ASR
6) Aligns that draft to the known full CSV transcript
7) Writes:
      chunk.wav
      chunk.txt       <- reference transcript for that chunk
      manifest.jsonl
      alignment_audit.csv
      REVIEW_LOW_SCORE.txt

No Docker, gcloud, NeMo, or server changes are required.
"""

import argparse
import csv
import json
import math
import re
import time
import wave
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf
from rapidfuzz import fuzz, process
from scipy.signal import resample_poly


TARGET_SR = 16000
MODEL = "nemotron-3.5-asr-streaming-0.6b"


def norm_key(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\(\d+\)", "", s)
    s = re.sub(r"[^a-z0-9]+", "", s)
    for suffix in ("faq", "issue", "status"):
        # don't remove universally; filename fuzzy matching handles most cases
        pass
    return s


def norm_words(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"<[a-z]{2}-[A-Z]{2}>", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip().split()


def match_use_case_to_wav(use_case: str, wavs: list[Path]) -> Path:
    """
    Fuzzy name matching. Examples:
      Withdraw Money -> withdraw_money(2).wav
      Card Lost -> Card_lost(2).wav
      COBRA Coverage FAQ -> COBRA_coverage(2).wav
      Account Not Found / Bank Issue -> bank_issue(2).wav
    """
    use_norm = norm_key(use_case)
    choices = {norm_key(w.stem): w for w in wavs}

    # Fast exact/containment matches first.
    if use_norm in choices:
        return choices[use_norm]

    for k, p in choices.items():
        if k in use_norm or use_norm in k:
            return p

    # Extra aliases for your current seven use cases.
    aliases = {
        "withdrawmoney": ["withdrawmoney"],
        "cardlost": ["cardlost"],
        "carddeliverystatus": ["carddeliverystatus"],
        "cobracoveragefaq": ["cobracoverage", "cobra"],
        "profileupdate": ["profileupdate"],
        "accountnotfoundbankissue": ["bankissue", "accountnotfoundbankissue"],
        "verificationcodeissue": ["verificationcodeissue"],
    }
    for alias in aliases.get(use_norm, []):
        for k, p in choices.items():
            if alias in k or k in alias:
                return p

    best = process.extractOne(use_norm, list(choices.keys()), scorer=fuzz.ratio)
    if not best or best[1] < 45:
        raise RuntimeError(f"Could not confidently map use_case={use_case!r} to a WAV")
    return choices[best[0]]


def read_audio_16k_mono(path: Path) -> np.ndarray:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)

    if sr != TARGET_SR:
        g = math.gcd(sr, TARGET_SR)
        up = TARGET_SR // g
        down = sr // g
        audio = resample_poly(audio, up, down).astype(np.float32)

    audio = np.clip(audio, -1.0, 1.0)
    return audio


def low_energy_boundary(
    audio: np.ndarray,
    desired_sample: int,
    search_radius_sec: float = 1.5,
    rms_window_ms: int = 120,
) -> int:
    radius = int(search_radius_sec * TARGET_SR)
    win = max(1, int(rms_window_ms / 1000 * TARGET_SR))

    lo = max(win, desired_sample - radius)
    hi = min(len(audio) - win, desired_sample + radius)
    if hi <= lo:
        return max(0, min(desired_sample, len(audio)))

    step = max(1, int(0.02 * TARGET_SR))  # 20 ms
    best_pos = desired_sample
    best_rms = float("inf")

    for pos in range(lo, hi + 1, step):
        seg = audio[pos - win // 2 : pos + win // 2]
        if len(seg) == 0:
            continue
        rms = float(np.sqrt(np.mean(seg * seg) + 1e-12))
        if rms < best_rms:
            best_rms = rms
            best_pos = pos

    return best_pos


def make_boundaries(
    audio: np.ndarray,
    target_sec: float,
    min_sec: float,
    max_sec: float,
) -> list[tuple[int, int]]:
    n = len(audio)
    target = int(target_sec * TARGET_SR)
    min_len = int(min_sec * TARGET_SR)
    max_len = int(max_sec * TARGET_SR)

    spans = []
    start = 0

    while start < n:
        remaining = n - start
        if remaining <= max_len:
            spans.append((start, n))
            break

        desired = start + target
        end = low_energy_boundary(audio, desired)

        end = max(start + min_len, end)
        end = min(start + max_len, end)

        # Avoid leaving a tiny tail.
        if n - end < min_len:
            end = n

        spans.append((start, end))
        start = end

    return spans


def write_pcm16_wav(path: Path, audio: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(audio, -1, 1) * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(TARGET_SR)
        wf.writeframes(pcm.tobytes())


def transcribe_http(
    client: httpx.Client,
    endpoint: str,
    wav_path: Path,
    language: str,
) -> str:
    with open(wav_path, "rb") as f:
        files = {"file": (wav_path.name, f, "audio/wav")}
        data = {
            "model": MODEL,
            "language": language,
            "response_format": "json",
        }
        r = client.post(endpoint, files=files, data=data)
    r.raise_for_status()

    try:
        body = r.json()
        if isinstance(body, dict):
            return str(body.get("text", "")).strip()
    except Exception:
        pass
    return r.text.strip()


def span_score(draft_words: list[str], ref_words: list[str]) -> float:
    if not draft_words or not ref_words:
        return 0.0
    return float(
        fuzz.ratio(
            " ".join(draft_words),
            " ".join(ref_words),
        )
    )


def align_chunk(
    draft: str,
    full_words: list[str],
    cursor: int,
) -> tuple[int, int, float]:
    """
    Sequential fuzzy alignment of a chunk draft to the known full transcript.
    Searches near the previous chunk's endpoint to preserve chronology.
    """
    d = norm_words(draft)
    if not d:
        return cursor, cursor, 0.0

    n = len(d)

    # Allow a little overlap because ASR/chunk boundaries can straddle words.
    start_lo = max(0, cursor - 8)
    start_hi = min(len(full_words), cursor + max(30, n))

    min_len = max(2, int(n * 0.60))
    max_len = min(len(full_words), int(n * 1.50) + 12)

    best = (cursor, min(len(full_words), cursor + n), -1.0)

    for s in range(start_lo, start_hi + 1):
        max_here = min(max_len, len(full_words) - s)
        for length in range(min_len, max_here + 1):
            e = s + length
            score = span_score(d, full_words[s:e])

            # Mild penalty for moving backwards too far.
            if s < cursor:
                score -= min(6.0, (cursor - s) * 0.35)

            if score > best[2]:
                best = (s, e, score)

    return best


def load_csv(csv_path: Path):
    rows = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"use_case", "transcript"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(
                f"CSV must contain columns {sorted(required)}. "
                f"Found: {reader.fieldnames}"
            )
        for row in reader:
            rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--wav-dir", required=True)
    ap.add_argument("--out-dir", default="loadtest_chunks")
    ap.add_argument(
        "--base-url",
        required=True,
        help="Example: https://nemotron-xxx.us-central1.run.app",
    )
    ap.add_argument("--language", default="en-US")
    ap.add_argument("--chunk-sec", type=float, default=10.0)
    ap.add_argument("--min-sec", type=float, default=7.0)
    ap.add_argument("--max-sec", type=float, default=13.0)
    ap.add_argument("--review-threshold", type=float, default=72.0)
    ap.add_argument("--request-timeout", type=float, default=120.0)
    ap.add_argument(
        "--skip-existing-drafts",
        action="store_true",
        help="Reuse saved .draft.txt files when rerunning alignment",
    )
    args = ap.parse_args()

    csv_path = Path(args.csv)
    wav_dir = Path(args.wav_dir)
    out_dir = Path(args.out_dir)
    chunks_root = out_dir / "chunks"
    chunks_root.mkdir(parents=True, exist_ok=True)

    endpoint = args.base_url.rstrip("/")
    if endpoint.endswith("/v1"):
        endpoint += "/audio/transcriptions"
    else:
        endpoint += "/v1/audio/transcriptions"

    rows = load_csv(csv_path)
    wavs = sorted(wav_dir.glob("*.wav"))
    if not wavs:
        raise FileNotFoundError(f"No WAV files found in {wav_dir}")

    print("\n=== FILE MAPPING ===")
    mapping = {}
    for row in rows:
        wav = match_use_case_to_wav(row["use_case"], wavs)
        mapping[row["use_case"]] = wav
        print(f"{row['use_case']:<35} -> {wav.name}")

    manifest_path = out_dir / "manifest.jsonl"
    audit_path = out_dir / "alignment_audit.csv"
    review_path = out_dir / "REVIEW_LOW_SCORE.txt"

    manifest_rows = []
    audit_rows = []
    review_lines = []

    timeout = httpx.Timeout(args.request_timeout)
    limits = httpx.Limits(max_connections=1, max_keepalive_connections=1)

    with httpx.Client(timeout=timeout, limits=limits, follow_redirects=True) as client:
        for row in rows:
            use_case = row["use_case"].strip()
            full_transcript = row["transcript"].strip()
            full_words = norm_words(full_transcript)
            source_wav = mapping[use_case]

            slug = re.sub(r"[^a-z0-9]+", "_", use_case.lower()).strip("_")
            case_dir = chunks_root / slug
            case_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n=== {use_case} ===")
            audio = read_audio_16k_mono(source_wav)
            spans = make_boundaries(
                audio,
                target_sec=args.chunk_sec,
                min_sec=args.min_sec,
                max_sec=args.max_sec,
            )
            print(
                f"{source_wav.name}: {len(audio)/TARGET_SR:.1f}s -> "
                f"{len(spans)} chunks"
            )

            cursor = 0

            for idx, (samp_start, samp_end) in enumerate(spans):
                chunk_id = f"{slug}_{idx:03d}"
                wav_path = case_dir / f"{chunk_id}.wav"
                txt_path = case_dir / f"{chunk_id}.txt"
                draft_path = case_dir / f"{chunk_id}.draft.txt"

                write_pcm16_wav(wav_path, audio[samp_start:samp_end])

                if args.skip_existing_drafts and draft_path.exists():
                    draft = draft_path.read_text(encoding="utf-8").strip()
                else:
                    print(f"  [{idx+1:02d}/{len(spans):02d}] transcribing {chunk_id} ...")
                    draft = transcribe_http(
                        client,
                        endpoint,
                        wav_path,
                        args.language,
                    )
                    draft_path.write_text(draft + "\n", encoding="utf-8")
                    # Keep preparation sequential and gentle.
                    time.sleep(0.05)

                ref_start, ref_end, score = align_chunk(
                    draft=draft,
                    full_words=full_words,
                    cursor=cursor,
                )

                reference = " ".join(full_words[ref_start:ref_end]).strip()
                txt_path.write_text(reference + "\n", encoding="utf-8")

                duration = (samp_end - samp_start) / TARGET_SR
                status = "OK" if score >= args.review_threshold else "REVIEW"

                manifest_row = {
                    "chunk_id": chunk_id,
                    "use_case": use_case,
                    "audio_path": str(wav_path.resolve()),
                    "reference_path": str(txt_path.resolve()),
                    "draft_path": str(draft_path.resolve()),
                    "source_wav": str(source_wav.resolve()),
                    "start_sec": round(samp_start / TARGET_SR, 3),
                    "end_sec": round(samp_end / TARGET_SR, 3),
                    "duration_sec": round(duration, 3),
                    "alignment_score": round(score, 2),
                    "status": status,
                    "language": args.language,
                }
                manifest_rows.append(manifest_row)

                audit_rows.append({
                    **manifest_row,
                    "draft_transcript": draft,
                    "reference_transcript": reference,
                    "reference_word_start": ref_start,
                    "reference_word_end": ref_end,
                })

                print(
                    f"      score={score:5.1f} {status:<6} "
                    f"{samp_start/TARGET_SR:6.1f}-{samp_end/TARGET_SR:6.1f}s"
                )

                if status == "REVIEW":
                    review_lines.append(
                        f"{chunk_id} | score={score:.1f}\n"
                        f"WAV: {wav_path}\n"
                        f"DRAFT: {draft}\n"
                        f"REF:   {reference}\n"
                    )

                # Advance monotonically; retain a tiny overlap allowance.
                cursor = max(cursor, max(ref_start, ref_end - 3))

    with open(manifest_path, "w", encoding="utf-8") as f:
        for row in manifest_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    if audit_rows:
        with open(audit_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(audit_rows[0].keys()))
            writer.writeheader()
            writer.writerows(audit_rows)

    review_path.write_text(
        "\n\n".join(review_lines) if review_lines else "No low-score chunks.\n",
        encoding="utf-8",
    )

    ok = sum(r["status"] == "OK" for r in manifest_rows)
    review = len(manifest_rows) - ok

    print("\n=== DONE ===")
    print(f"Chunks       : {len(manifest_rows)}")
    print(f"OK           : {ok}")
    print(f"Needs review : {review}")
    print(f"Manifest     : {manifest_path}")
    print(f"Audit        : {audit_path}")
    print(f"Review list  : {review_path}")
    print("\nBefore using absolute WER, manually review any REVIEW chunks.")


if __name__ == "__main__":
    main()
