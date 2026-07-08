#!/usr/bin/env bash
set -euo pipefail

# Run existing server image with base model saved inside Docker image.
docker run --gpus all -it --rm \
  -p 8003:8003 \
  nemotron-asr
