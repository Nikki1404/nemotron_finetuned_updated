docker run --gpus all -it --rm -v $PWD:/workspace -v $PWD/ft_models:/srv/models nemotron_finetuned bash

cd /workspace

pwd

ls

apt update

apt install -y cuda-nvvm-12-4

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export CUDA_HOME=/usr/local/cuda-12.4

unset NUMBA_CUDA_USE_NVIDIA_BINDING

mkdir -p /srv/models

mkdir -p results/hparam_tuning

python3.11 scripts/prepare_dataset.py --csv data/inspira_transcripts.csv --wav-dir raw_wavs --out-dir data

ls -lh data/audio_16k

ls -lh data/manifests

python3.11 scripts/auto_align_chunks_with_base_asr.py --csv data/inspira_transcripts.csv --wav-dir data/audio_16k --base-model /srv/nemotron-3.5-asr-streaming-0.6b.nemo --out-dir data/audio_chunks --manifest data/manifests/aligned_chunk_manifest.json --audit data/manifests/alignment_audit.csv --language en-US --chunk-sec 10

ls -lh data/audio_chunks

ls -lh data/manifests/aligned_chunk_manifest.json

head -n 3 data/manifests/aligned_chunk_manifest.json

python3.11 scripts/split_by_usecase_manifest.py --input data/manifests/aligned_chunk_manifest.json --out-dir data/manifests

ls -lh data/manifests/train_aligned_manifest.json

ls -lh data/manifests/val_aligned_manifest.json

ls -lh data/manifests/test_aligned_manifest.json

rm -rf data/audio_aug

rm -f data/manifests/train_aligned_aug_manifest.json

python3.11 scripts/augment_train_manifest.py --train-manifest data/manifests/train_aligned_manifest.json --out-manifest data/manifests/train_aligned_aug_manifest.json --out-audio-dir data/audio_aug --keep-original

ls -lh data/audio_aug

ls -lh data/manifests/train_aligned_aug_manifest.json

python3.11 scripts/evaluate_manifest.py --model /srv/nemotron-3.5-asr-streaming-0.6b.nemo --manifest data/manifests/test_aligned_manifest.json --language en-US --output-jsonl results/hparam_tuning/base_eval.jsonl

python3.11 scripts/finetune_nemotron.py --train-manifest data/manifests/train_aligned_aug_manifest.json --val-manifest data/manifests/val_aligned_manifest.json --base-model /srv/nemotron-3.5-asr-streaming-0.6b.nemo --output-nemo /srv/models/finetuned_nemotron_v1_lr3e6_ep2.nemo --freeze-mode decoder_only --max-epochs 2 --batch-size 1 --accumulate-grad-batches 8 --lr 3e-6 --language en-US --precision bf16-mixed --num-workers 0

ls -lh /srv/models/finetuned_nemotron_v1_lr3e6_ep2.nemo

python3.11 scripts/evaluate_manifest.py --model /srv/models/finetuned_nemotron_v1_lr3e6_ep2.nemo --manifest data/manifests/test_aligned_manifest.json --language en-US --output-jsonl results/hparam_tuning/v1_eval.jsonl

python3.11 scripts/finetune_nemotron.py --train-manifest data/manifests/train_aligned_aug_manifest.json --val-manifest data/manifests/val_aligned_manifest.json --base-model /srv/nemotron-3.5-asr-streaming-0.6b.nemo --output-nemo /srv/models/finetuned_nemotron_v2_lr2e6_ep3.nemo --freeze-mode decoder_only --max-epochs 3 --batch-size 1 --accumulate-grad-batches 8 --lr 2e-6 --language en-US --precision bf16-mixed --num-workers 0

ls -lh /srv/models/finetuned_nemotron_v2_lr2e6_ep3.nemo

python3.11 scripts/evaluate_manifest.py --model /srv/models/finetuned_nemotron_v2_lr2e6_ep3.nemo --manifest data/manifests/test_aligned_manifest.json --language en-US --output-jsonl results/hparam_tuning/v2_eval.jsonl

python3.11 scripts/finetune_nemotron.py --train-manifest data/manifests/train_aligned_aug_manifest.json --val-manifest data/manifests/val_aligned_manifest.json --base-model /srv/nemotron-3.5-asr-streaming-0.6b.nemo --output-nemo /srv/models/finetuned_nemotron_v3_last_encoder_lr5e7_ep1.nemo --freeze-mode last_encoder --max-epochs 1 --batch-size 1 --accumulate-grad-batches 8 --lr 5e-7 --language en-US --precision bf16-mixed --num-workers 0

ls -lh /srv/models/finetuned_nemotron_v3_last_encoder_lr5e7_ep1.nemo

python3.11 scripts/evaluate_manifest.py --model /srv/models/finetuned_nemotron_v3_last_encoder_lr5e7_ep1.nemo --manifest data/manifests/test_aligned_manifest.json --language en-US --output-jsonl results/hparam_tuning/v3_eval.jsonl

python3.11 scripts/compare_models_report.py --base results/hparam_tuning/base_eval.jsonl --v1 results/hparam_tuning/v1_eval.jsonl --v2 results/hparam_tuning/v2_eval.jsonl --v3 results/hparam_tuning/v3_eval.jsonl --out results/hparam_tuning/model_comparison_report.md

cat results/hparam_tuning/model_comparison_report.md

cp /srv/models/finetuned_nemotron_v1_lr3e6_ep2.nemo /srv/models/finetuned_nemotron_final.nemo

cp /srv/models/finetuned_nemotron_v2_lr2e6_ep3.nemo /srv/models/finetuned_nemotron_final.nemo

cp /srv/models/finetuned_nemotron_v3_last_encoder_lr5e7_ep1.nemo /srv/models/finetuned_nemotron_final.nemo

ls -lh /srv/models/finetuned_nemotron_final.nemo

exit

cd /home/CORP/re_nikitav/nemotron_finetuned

ls -lh ft_models/finetuned_nemotron_final.nemo

docker build -t nemotron_finetuned .

docker run --gpus all -it --rm -p 8003:8003 -v $PWD:/workspace -v $PWD/ft_models:/srv/models -e MODEL_NAME=/srv/models/finetuned_nemotron_final.nemo -e VAD_START_MARGIN=1.8 -e VAD_MIN_NOISE_RMS=0.002 -e PRE_SPEECH_MS=500 -e NEMO_END_SILENCE_MS=900 -e FINALIZE_PAD_MS=800 -e CONTEXT_RIGHT=2 -e NEMO_MAX_SYMBOLS=15 nemotron_finetuned uvicorn app.main:app --host 0.0.0.0 --port 8003

