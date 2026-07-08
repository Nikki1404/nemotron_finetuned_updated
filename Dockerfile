FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# ---- Proxy support ----
ENV http_proxy="http://163.116.128.80:8080"
ENV https_proxy="http://163.116.128.80:8080"

ENV PYTHONUNBUFFERED=1
ENV PIP_TRUSTED_HOST="pypi.org files.pythonhosted.org pypi.python.org"

ENV HF_HOME=/srv/hf_cache
ENV TRANSFORMERS_CACHE=/srv/hf_cache
ENV NEMO_CACHE_DIR=/srv/nemo_cache

WORKDIR /srv

# 1. System dependencies
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get dist-upgrade -y && \
    apt-get install -y --no-install-recommends \
        wget \
        git \
        ca-certificates \
        build-essential \
        libssl-dev \
        zlib1g-dev \
        libbz2-dev \
        libreadline-dev \
        libsqlite3-dev \
        libffi-dev \
        liblzma-dev \
        curl \
        ffmpeg \
        libsndfile1 \
        sox \
    && rm -rf /var/lib/apt/lists/*

# 2. Install Python 3.11
RUN wget --no-check-certificate https://www.python.org/ftp/python/3.11.9/Python-3.11.9.tgz && \
    tar -xzf Python-3.11.9.tgz && \
    cd Python-3.11.9 && \
    ./configure --enable-optimizations && \
    make -j$(nproc) && \
    make install && \
    cd / && rm -rf Python-3.11.9*

# 3. Setup pip
RUN python3.11 -m ensurepip && \
    python3.11 -m pip install --upgrade pip setuptools wheel

# 4. Install minimal project dependencies
COPY requirements.txt .
RUN python3.11 -m pip install --no-cache-dir -r requirements.txt

# 5. Install NeMo from GitHub main
RUN python3.11 -m pip install --no-cache-dir \
    "nemo_toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git@main"

# 6. Force reinstall pinned Torch AFTER NeMo
# Torch 2.6 has torch.nn.Buffer, required by current NeMo main.
# cu124 is compatible with host CUDA 12.6 driver.
RUN python3.11 -m pip install --no-cache-dir --force-reinstall \
    torch==2.6.0 \
    torchvision==0.21.0 \
    torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124

# 7. Validate Torch version and required API
RUN python3.11 - <<'EOF'
import torch

print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available during build:", torch.cuda.is_available())

assert torch.__version__.startswith("2.6.0"), torch.__version__
assert torch.version.cuda == "12.4", torch.version.cuda
assert hasattr(torch.nn, "Buffer"), "torch.nn.Buffer missing"

print("Torch validation passed.")
EOF

# 8. Validate required Nemotron class exists
RUN python3.11 - <<'EOF'
import importlib

module_name = "nemo.collections.asr.models.rnnt_bpe_models_prompt"
class_name = "EncDecRNNTBPEModelWithPrompt"

mod = importlib.import_module(module_name)
cls = getattr(mod, class_name)

print("NeMo Nemotron prompt RNNT class found:", cls)
EOF

# 9. Download Nemotron model during build
RUN python3.11 - <<'EOF'
import nemo.collections.asr as nemo_asr

print("Downloading Nemotron 3.5 ASR Streaming from Hugging Face...")

model = nemo_asr.models.ASRModel.from_pretrained(
    model_name="nvidia/nemotron-3.5-asr-streaming-0.6b"
)

model.save_to("/srv/nemotron-3.5-asr-streaming-0.6b.nemo")

print("Nemotron saved successfully at /srv/nemotron-3.5-asr-streaming-0.6b.nemo")
EOF

# 10. Copy app
COPY app ./app

EXPOSE 8003

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8003", "--ws-ping-interval", "20", "--ws-ping-timeout", "120"]
