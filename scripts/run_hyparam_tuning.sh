#!/bin/bash
set -e
cd /workspace
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_HOME=/usr/local/cuda-12.4
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/nvvm/lib64:$LD_LIBRARY_PATH
unset NUMBA_CUDA_USE_NVIDIA_BINDING
export TOKENIZERS_PARALLELISM=false
BASE_MODEL="/srv/nemotron-3.5-asr-streaming-0.6b.nemo"
TRAIN_MANIFEST="data/manifests/train_aligned_aug_manifest.json"
VAL_MANIFEST="data/manifests/val_aligned_manifest.json"
TEST_MANIFEST="data/manifests/test_aligned_manifest.json"
mkdir -p /srv/models results/hparam_tuning
python3.11 scripts/split_by_usecase_manifest.py --input data/manifests/aligned_chunk_manifest.json --out-dir data/manifests
rm -rf data/audio_aug data/manifests/train_aligned_aug_manifest.json
python3.11 scripts/augment_train_manifest.py --train-manifest data/manifests/train_aligned_manifest.json --out-manifest data/manifests/train_aligned_aug_manifest.json --out-audio-dir data/audio_aug --keep-original
python3.11 scripts/evaluate_manifest.py --model $BASE_MODEL --manifest $TEST_MANIFEST --language en-US --output-jsonl results/hparam_tuning/base_eval.jsonl
python3.11 scripts/finetune_nemotron.py --train-manifest $TRAIN_MANIFEST --val-manifest $VAL_MANIFEST --base-model $BASE_MODEL --output-nemo /srv/models/finetuned_nemotron_v1_lr3e6_ep2.nemo --freeze-mode decoder_only --max-epochs 2 --batch-size 1 --accumulate-grad-batches 8 --lr 3e-6 --language en-US --precision bf16-mixed --num-workers 0
python3.11 scripts/evaluate_manifest.py --model /srv/models/finetuned_nemotron_v1_lr3e6_ep2.nemo --manifest $TEST_MANIFEST --language en-US --output-jsonl results/hparam_tuning/v1_eval.jsonl
python3.11 scripts/finetune_nemotron.py --train-manifest $TRAIN_MANIFEST --val-manifest $VAL_MANIFEST --base-model $BASE_MODEL --output-nemo /srv/models/finetuned_nemotron_v2_lr2e6_ep3.nemo --freeze-mode decoder_only --max-epochs 3 --batch-size 1 --accumulate-grad-batches 8 --lr 2e-6 --language en-US --precision bf16-mixed --num-workers 0
python3.11 scripts/evaluate_manifest.py --model /srv/models/finetuned_nemotron_v2_lr2e6_ep3.nemo --manifest $TEST_MANIFEST --language en-US --output-jsonl results/hparam_tuning/v2_eval.jsonl
python3.11 scripts/finetune_nemotron.py --train-manifest $TRAIN_MANIFEST --val-manifest $VAL_MANIFEST --base-model $BASE_MODEL --output-nemo /srv/models/finetuned_nemotron_v3_last_encoder_lr5e7_ep1.nemo --freeze-mode last_encoder --max-epochs 1 --batch-size 1 --accumulate-grad-batches 8 --lr 5e-7 --language en-US --precision bf16-mixed --num-workers 0
python3.11 scripts/evaluate_manifest.py --model /srv/models/finetuned_nemotron_v3_last_encoder_lr5e7_ep1.nemo --manifest $TEST_MANIFEST --language en-US --output-jsonl results/hparam_tuning/v3_eval.jsonl
python3.11 scripts/compare_models_report.py --base results/hparam_tuning/base_eval.jsonl --v1 results/hparam_tuning/v1_eval.jsonl --v2 results/hparam_tuning/v2_eval.jsonl --v3 results/hparam_tuning/v3_eval.jsonl --out results/hparam_tuning/model_comparison_report.md
cat results/hparam_tuning/model_comparison_report.md
