#!/usr/bin/env python3
import argparse, inspect, json, re
from pathlib import Path
import torch
import nemo.collections.asr as nemo_asr


def patch_prompt(default_lang: str):
    import nemo.collections.asr.data.audio_to_text_lhotse_prompt_index as mod
    for _, cls in vars(mod).items():
        if inspect.isclass(cls) and hasattr(cls, "_get_prompt_index"):
            old = cls._get_prompt_index
            def patched(self, prompt_key, _old=old):
                if prompt_key is None or str(prompt_key).lower() == "none":
                    prompt_key = default_lang
                return _old(self, prompt_key)
            cls._get_prompt_index = patched
    print(f"[patch] using default prompt language: {default_lang}")


def norm(s):
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9ñáéíóúü\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def edit_distance(a, b):
    dp = [[0]*(len(b)+1) for _ in range(len(a)+1)]
    for i in range(len(a)+1): dp[i][0] = i
    for j in range(len(b)+1): dp[0][j] = j
    for i in range(1, len(a)+1):
        for j in range(1, len(b)+1):
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+cost)
    return dp[-1][-1]


def wer(ref, hyp):
    r, h = norm(ref).split(), norm(hyp).split()
    return 0.0 if not r and not h else (100.0 if not r else 100.0*edit_distance(r, h)/len(r))


def cer(ref, hyp):
    r, h = norm(ref).replace(" ", ""), norm(hyp).replace(" ", "")
    return 0.0 if not r and not h else (100.0 if not r else 100.0*edit_distance(list(r), list(h))/len(r))


def get_text(x):
    if isinstance(x, str): return x
    if isinstance(x, list) and x: return get_text(x[0])
    if hasattr(x, "text"): return x.text
    if isinstance(x, dict): return x.get("text") or x.get("pred_text") or str(x)
    return str(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--language", default="en-US")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output-jsonl", default="eval_predictions.jsonl")
    args = ap.parse_args()

    patch_prompt(args.language)
    rows = [json.loads(x) for x in Path(args.manifest).read_text().splitlines() if x.strip()]
    print(f"[eval] Loading model: {args.model}")
    model = nemo_asr.models.ASRModel.restore_from(args.model, map_location=args.device).to(args.device)
    model.eval()

    outputs, total_wer, total_cer = [], 0.0, 0.0
    for i, row in enumerate(rows, 1):
        audio = row["audio_filepath"]
        lang = row.get("target_lang") or row.get("language") or args.language
        try: model.set_default_prompt(lang)
        except Exception: pass
        try: model.set_inference_prompt(lang)
        except Exception: pass
        print(f"[eval] {i}/{len(rows)} {audio} lang={lang}")
        with torch.no_grad():
            hyp = get_text(model.transcribe([audio], batch_size=1, verbose=False))
        ref = row.get("text", "")
        w, c = wer(ref, hyp), cer(ref, hyp)
        total_wer += w; total_cer += c
        print(f"WER: {w:.2f}% | CER: {c:.2f}%")
        print(f"PRED: {hyp[:300]}")
        outputs.append({"audio_filepath": audio, "language": lang, "reference": ref, "prediction": hyp, "wer": w, "cer": c})

    Path(args.output_jsonl).write_text("\n".join(json.dumps(o, ensure_ascii=False) for o in outputs)+"\n")
    print("\n========== SUMMARY ==========")
    print(f"Files: {len(rows)}")
    print(f"Average WER: {total_wer/max(1,len(rows)):.2f}%")
    print(f"Average CER: {total_cer/max(1,len(rows)):.2f}%")
    print(f"Saved predictions: {args.output_jsonl}")

if __name__ == "__main__":
    main()
