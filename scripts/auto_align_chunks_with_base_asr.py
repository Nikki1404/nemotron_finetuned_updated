#!/usr/bin/env python3

import argparse
import csv
import json
import re
import subprocess
import wave
from difflib import SequenceMatcher
from pathlib import Path

import torch
import nemo.collections.asr as nemo_asr


def norm_name(s):
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def clean_text(s):
    s = s.lower()
    s = s.replace("’", "'")
    s = re.sub(r"[^a-z0-9ñáéíóúü\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def duration_sec(wav_path):
    with wave.open(str(wav_path), "rb") as w:
        return w.getnframes() / w.getframerate()


def patch_nemotron_prompt(default_lang):
    import inspect
    import nemo.collections.asr.data.audio_to_text_lhotse_prompt_index as mod

    for _, cls in vars(mod).items():
        if inspect.isclass(cls) and hasattr(cls, "_get_prompt_index"):
            old_fn = cls._get_prompt_index

            def new_get_prompt_index(self, prompt_key, _old_fn=old_fn):
                if prompt_key is None or str(prompt_key).lower() == "none":
                    prompt_key = default_lang
                return _old_fn(self, prompt_key)

            cls._get_prompt_index = new_get_prompt_index


def extract_text(x):
    if isinstance(x, str):
        return x
    if isinstance(x, list) and x:
        return extract_text(x[0])
    if hasattr(x, "text"):
        return x.text
    if isinstance(x, dict):
        return x.get("text") or str(x)
    return str(x)


def split_audio(wav_path, out_dir, chunk_sec):
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(out_dir.glob("*.wav"))
    if existing:
        return existing

    pattern = str(out_dir / f"{wav_path.stem}_%03d.wav")

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(wav_path),
        "-f",
        "segment",
        "-segment_time",
        str(chunk_sec),
        "-reset_timestamps",
        "1",
        pattern,
    ]

    subprocess.run(cmd, check=True)
    return sorted(out_dir.glob("*.wav"))


def transcribe_chunk(model, wav_path, language):
    try:
        model.set_inference_prompt(language)
    except Exception:
        pass

    with torch.no_grad():
        out = model.transcribe([str(wav_path)], batch_size=1, verbose=False)

    return clean_text(extract_text(out))


def best_match_window(full_words, draft_words, cursor):
    if not draft_words:
        return cursor, min(cursor + 5, len(full_words)), 0.0

    expected_len = len(draft_words)
    min_len = max(3, int(expected_len * 0.6))
    max_len = min(len(full_words), int(expected_len * 1.6) + 5)

    search_start = max(0, cursor - 10)
    search_end = min(len(full_words), cursor + expected_len + 80)

    best_score = -1
    best_start = cursor
    best_end = min(cursor + expected_len, len(full_words))

    draft_text = " ".join(draft_words)

    for start in range(search_start, search_end):
        for length in range(min_len, max_len + 1):
            end = start + length
            if end > len(full_words):
                break

            cand_text = " ".join(full_words[start:end])
            score = SequenceMatcher(None, draft_text, cand_text).ratio()

            if score > best_score:
                best_score = score
                best_start = start
                best_end = end

    return best_start, best_end, best_score


def read_csv(csv_path):
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            use_case = r.get("use_case") or r.get("Use Case") or r.get("UseCase")
            transcript = r.get("transcript") or r.get("Transcript")
            if use_case and transcript:
                rows.append({
                    "use_case": use_case.strip(),
                    "transcript": transcript.strip(),
                })
    return rows


def find_wav_for_use_case(use_case, wav_dir):
    target = norm_name(use_case)

    wavs = list(Path(wav_dir).glob("*.wav"))

    for w in wavs:
        if norm_name(w.stem) == target:
            return w

    for w in wavs:
        if target in norm_name(w.stem) or norm_name(w.stem) in target:
            return w

    raise FileNotFoundError(f"No WAV found for use case: {use_case}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--wav-dir", default="data/audio_16k")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--out-dir", default="data/aligned_chunks")
    parser.add_argument("--manifest", default="data/manifests/aligned_chunk_manifest.json")
    parser.add_argument("--audit", default="data/manifests/alignment_audit.csv")
    parser.add_argument("--language", default="en-US")
    parser.add_argument("--chunk-sec", type=int, default=10)
    parser.add_argument("--min-score", type=float, default=0.20)
    args = parser.parse_args()

    patch_nemotron_prompt(args.language)

    print("[load]", args.base_model)
    model = nemo_asr.models.ASRModel.restore_from(args.base_model, map_location="cuda")
    model = model.cuda()
    model.eval()

    csv_rows = read_csv(args.csv)

    manifest_rows = []
    audit_rows = []

    for item in csv_rows:
        use_case = item["use_case"]
        full_text = clean_text(item["transcript"])
        full_words = full_text.split()

        wav = find_wav_for_use_case(use_case, args.wav_dir)
        chunk_dir = Path(args.out_dir) / norm_name(use_case)

        chunks = split_audio(wav, chunk_dir, args.chunk_sec)

        print(f"\n[use_case] {use_case}")
        print(f"[wav] {wav}")
        print(f"[chunks] {len(chunks)}")

        cursor = 0

        for idx, chunk in enumerate(chunks):
            dur = duration_sec(chunk)
            if dur < 1.0:
                continue

            draft = transcribe_chunk(model, chunk, args.language)
            draft_words = draft.split()

            start, end, score = best_match_window(full_words, draft_words, cursor)

            aligned_words = full_words[start:end]
            aligned_text = " ".join(aligned_words)

            if score < args.min_score:
                print(f"[warn] low match score {score:.2f}: {chunk}")

            cursor = max(end, cursor)

            row = {
                "audio_filepath": str(chunk.resolve()),
                "duration": round(dur, 3),
                "text": aligned_text,
                "target_lang": args.language,
                "language": args.language,
                "use_case": use_case,
                "match_score": round(score, 4),
                "draft_asr": draft,
            }

            manifest_rows.append(row)

            audit_rows.append({
                "use_case": use_case,
                "chunk": str(chunk),
                "duration": round(dur, 3),
                "match_score": round(score, 4),
                "draft_asr": draft,
                "aligned_text": aligned_text,
            })

            print(f"  {idx+1}/{len(chunks)} score={score:.2f} text={aligned_text[:80]}")

    Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)

    with open(args.manifest, "w", encoding="utf-8") as f:
        for r in manifest_rows:
            clean_row = {
                "audio_filepath": r["audio_filepath"],
                "duration": r["duration"],
                "text": r["text"],
                "target_lang": r["target_lang"],
                "language": r["language"],
                "use_case": r["use_case"],
            }
            f.write(json.dumps(clean_row, ensure_ascii=False) + "\n")

    with open(args.audit, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "use_case",
                "chunk",
                "duration",
                "match_score",
                "draft_asr",
                "aligned_text",
            ],
        )
        writer.writeheader()
        writer.writerows(audit_rows)

    print("\n[done]")
    print("manifest:", args.manifest)
    print("audit:", args.audit)
    print("rows:", len(manifest_rows))


if __name__ == "__main__":
    main()
