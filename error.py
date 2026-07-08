python3.11 -m pip uninstall -y numba llvmlite cuda-python cuda-bindings cuda-core cuda-pathfinder

python3.11 -m pip install --no-cache-dir "numba==0.60.0" "llvmlite==0.43.0"

apt update && apt install -y cuda-nvvm-12-4

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export CUDA_HOME=/usr/local/cuda-12.4

export LD_LIBRARY_PATH=/usr/local/cuda-12.4/nvvm/lib64:$LD_LIBRARY_PATH

unset NUMBA_CUDA_USE_NVIDIA_BINDING

python3.11 scripts/finetune_nemotron.py --train-manifest data/manifests/train_aligned_aug_manifest.json --val-manifest data/manifests/val_aligned_manifest.json --base-model /srv/nemotron-3.5-asr-streaming-0.6b.nemo --output-nemo /srv/models/finetuned_nemotron_v1_lr3e6_ep2.nemo --freeze-mode decoder_only --max-epochs 2 --batch-size 1 --accumulate-grad-batches 8 --lr 3e-6 --language en-US --precision bf16-mixed --num-workers 0
