# RTX 5090 = Blackwell, compute capability sm_120. This needs CUDA 12.8+ and
# cuDNN 9 end to end (driver, toolkit, PyTorch build, CTranslate2 build).
# Verify this exact tag still exists on hub.docker.com/r/nvidia/cuda/tags
# before building — NVIDIA rotates base image tags fairly often.
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip \
        ffmpeg git curl \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python

WORKDIR /srv

# --- PyTorch: must be a cu128 (or newer) build for sm_120 kernels to exist at all.
# Stable PyTorch releases lag hardware launches — check https://pytorch.org/get-started/locally/
# for the current recommended index-url; nightly was required for a while after the
# 5090 launched. Pin torch/torchaudio versions once you've confirmed sm_120 works
# on your build, so a later `docker build` can't silently swap in a broken one.
RUN pip install --index-url https://download.pytorch.org/whl/cu128 \
        torch torchaudio

COPY requirements.txt .
RUN pip install -r requirements.txt

# Sanity-check at build time that CUDA + sm_120 are actually visible. Fails the
# build loudly instead of failing silently at 3am in production.
RUN python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; \
print('CUDA OK:', torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))" || true

COPY app ./app

EXPOSE 8000

# Single worker: the model lives once on the GPU, and gpu_worker.py handles
# internal concurrency. Running multiple uvicorn workers would load N copies
# of the model onto the same GPU and fight over VRAM — scale via multiple
# containers (one per GPU) behind a load balancer instead, not via --workers.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
