#!/usr/bin/env bash
set -euo pipefail
docker run --gpus all -it --rm -p 8003:8003 -v "$PWD":/workspace -v "$PWD/ft_models":/srv/models -e MODEL_NAME=/srv/models/finetuned_nemotron_final.nemo -e VAD_START_MARGIN=1.8 -e VAD_MIN_NOISE_RMS=0.002 -e PRE_SPEECH_MS=500 -e NEMO_END_SILENCE_MS=900 -e FINALIZE_PAD_MS=800 -e CONTEXT_RIGHT=2 -e NEMO_MAX_SYMBOLS=15 nemotron_finetuned uvicorn app.main:app --host 0.0.0.0 --port 8003
