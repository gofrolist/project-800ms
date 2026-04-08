# CUDA 12.x + cuDNN for Faster-Whisper in-process on GPU.
# ctranslate2 (used by faster-whisper) is linked against libcublas.so.12, so
# we need CUDA 12.x — CUDA 13 ships libcublas.so.13 only and won't load.
FROM nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# python3.12 (Ubuntu 24.04 default) + libs Piper's phonemizer / eSpeak need, plus ffmpeg for audio.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3-pip \
        ffmpeg libsndfile1 \
        espeak-ng \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create a venv (PEP 668: Ubuntu 24.04's system python is externally managed)
RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

CMD ["python", "main.py"]
