#!/usr/bin/env python3
"""
Prepare Inspira ASR fine-tuning data for NVIDIA NeMo/Nemotron.

Input:
  - CSV with two columns: use_case, transcript
  - WAV files in raw_wavs/

Output:
  - 16k mono WAVs in data/audio_16k/
  - train/val/test NeMo JSONL manifests in data/manifests/

Manifest row format:
  {"audio_filepath":"/abs/path/file.wav","duration":12.3,"text":"normalized transcript"}
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import subprocess
import wave
from pathlib import Path
from typing import Dict, List, Tuple


DEFAULT_FILE_MAP = {
    "withdraw money": "withdraw_money.wav",
    "card lost": "Card_lost.wav",
    "card delivery status": "Card_Delivery_Status.wav",
    "cobra coverage faq": "COBRA_coverage.wav",
    "cobra coverage": "COBRA_coverage.wav",
    "profile update": "Profile_Update.wav",
    "account not found bank issue": "bank_issue.wav",
    "account not found / bank issue": "bank_issue.wav",
    "bank issue": "bank_issue.wav",
    "verification code issue": "Verification_Code_Issue.wav",
}


def norm_key(s: str) -> str:
    s = s.strip().lower()
    s = s.replace("/", " ")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def normalize_transcript(text: str, keep_digits: bool = True) -> str:
    text = text.strip()
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    text = text.replace("Social SecurityNumber", "Social Security Number")
    text = text.replace("onForm", "on Form")
    text = text.replace("Member ID?sure", "Member ID sure")
    text = text.lower()

    # remove punctuation that ASR usually should not be forced to predict
    allowed = r"a-z0-9\s" if keep_digits else r"a-z\s"
    text = re.sub(fr"[^{allowed}]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as wf:
        return wf.getnframes() / float(wf.getframerate())


def ffmpeg_to_16k(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(src),
        "-ac", "1",
        "-ar", "16000",
        "-sample_fmt", "s16",
        str(dst),
    ]
    subprocess.run(cmd, check=True)


def read_csv(csv_path: Path) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("CSV has no header. Expected: use_case,transcript")
        lower_map = {c.lower().strip(): c for c in reader.fieldnames}
        use_col = lower_map.get("use_case") or lower_map.get("use case") or lower_map.get("usecase")
        txt_col = lower_map.get("transcript") or lower_map.get("text")
        if not use_col or not txt_col:
            raise ValueError(f"CSV columns must include use_case and transcript. Found: {reader.fieldnames}")
        for row in reader:
            use_case = (row.get(use_col) or "").strip()
            transcript = (row.get(txt_col) or "").strip()
            if use_case and transcript:
                rows.append((use_case, transcript))
    if not rows:
        raise ValueError("No usable rows found in CSV")
    return rows


def find_wav(use_case: str, wav_dir: Path) -> Path:
    key = norm_key(use_case)
    file_name = DEFAULT_FILE_MAP.get(key)
    if file_name and (wav_dir / file_name).exists():
        return wav_dir / file_name

    # fallback: compare normalized stem names
    candidates = list(wav_dir.glob("*.wav")) + list(wav_dir.glob("*.WAV"))
    for p in candidates:
        if norm_key(p.stem) == key:
            return p
    for p in candidates:
        if key in norm_key(p.stem) or norm_key(p.stem) in key:
            return p
    raise FileNotFoundError(f"Could not find WAV for use case '{use_case}' in {wav_dir}")


def write_jsonl(path: Path, records: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV path with use_case,transcript")
    ap.add_argument("--wav-dir", required=True, help="Directory containing original WAV files")
    ap.add_argument("--out-dir", required=True, help="Output data directory")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--keep-digits", action="store_true", default=True)
    args = ap.parse_args()

    csv_path = Path(args.csv).resolve()
    wav_dir = Path(args.wav_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    audio_out = out_dir / "audio_16k"
    manifest_dir = out_dir / "manifests"

    rows = read_csv(csv_path)
    records: List[Dict] = []

    for use_case, transcript in rows:
        src = find_wav(use_case, wav_dir)
        dst_name = norm_key(use_case).replace(" ", "_") + ".wav"
        dst = audio_out / dst_name
        print(f"[prepare] {use_case}: {src.name} -> {dst}")
        ffmpeg_to_16k(src, dst)
        duration = wav_duration(dst)
        text = normalize_transcript(transcript, keep_digits=args.keep_digits)
        records.append({
            "audio_filepath": str(dst.resolve()),
            "duration": round(duration, 3),
            "text": text,
            "use_case": use_case,
        })

    random.seed(args.seed)
    random.shuffle(records)

    # For 7 files, use 5 train / 1 val / 1 test.
    n = len(records)
    if n < 3:
        raise ValueError("Need at least 3 files for train/val/test split")
    test = [records[-1]]
    val = [records[-2]]
    train = records[:-2]

    write_jsonl(manifest_dir / "train_manifest.json", train)
    write_jsonl(manifest_dir / "val_manifest.json", val)
    write_jsonl(manifest_dir / "test_manifest.json", test)
    write_jsonl(manifest_dir / "all_manifest.json", records)

    print("\nCreated:")
    print(f"  train: {manifest_dir / 'train_manifest.json'} ({len(train)} files)")
    print(f"  val:   {manifest_dir / 'val_manifest.json'} ({len(val)} files)")
    print(f"  test:  {manifest_dir / 'test_manifest.json'} ({len(test)} files)")
    print(f"  all:   {manifest_dir / 'all_manifest.json'} ({len(records)} files)")


if __name__ == "__main__":
    main()
