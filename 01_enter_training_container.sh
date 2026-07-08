#!/usr/bin/env bash
set -euo pipefail
docker build -t nemotron_finetuned .
mkdir -p ft_models results/hparam_tuning
docker run --gpus all -it --rm -v "$PWD":/workspace -v "$PWD/ft_models":/srv/models nemotron_finetuned bash
