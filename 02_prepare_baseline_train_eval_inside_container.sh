#!/usr/bin/env bash
set -euo pipefail

cd /workspace
mkdir -p /srv/models logs

python3.11 scripts/prepare_dataset.py \
  --csv data/inspira_transcripts.csv \
  --wav-dir raw_wavs \
  --out-dir data

python3.11 scripts/evaluate_manifest.py \
  --model /srv/nemotron-3.5-asr-streaming-0.6b.nemo \
  --manifest data/manifests/test_manifest.json \
  --language en-US \
  --output-jsonl logs/base_predictions.jsonl

python3.11 scripts/finetune_nemotron.py \
  --train-manifest data/manifests/train_manifest.json \
  --val-manifest data/manifests/val_manifest.json \
  --base-model /srv/nemotron-3.5-asr-streaming-0.6b.nemo \
  --output-nemo /srv/models/nemotron_inspira_decoder_ft.nemo \
  --freeze-mode decoder_only \
  --max-epochs 5 \
  --batch-size 1 \
  --lr 5e-6 \
  --language en-US

python3.11 scripts/evaluate_manifest.py \
  --model /srv/models/nemotron_inspira_decoder_ft.nemo \
  --manifest data/manifests/test_manifest.json \
  --language en-US \
  --output-jsonl logs/finetuned_predictions.jsonl
