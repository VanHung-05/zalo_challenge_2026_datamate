# =============================================================================
# DataMate — Zalo AI Challenge 2026
# Production image: NVIDIA CUDA + Python 3.11 + vLLM + DuckDB
# =============================================================================
# Build:  docker build -t datamate:latest .
# Run:    docker run --gpus all \
#           -v /path/to/data:/data:ro \
#           -v /path/to/result:/result \
#           -v /path/to/models:/models:ro \
#           datamate:latest
# =============================================================================

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# ── System ────────────────────────────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-venv \
        python3-pip \
        python3.11-dev \
        build-essential \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python  python  /usr/bin/python3.11 1

# ── Python deps ───────────────────────────────────────────────────────────────
WORKDIR /workspace

# Copy requirements first (layer cache: only re-run pip when requirements change)
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Install vLLM (GPU inference) — kept separate so it can be skipped for CPU-only builds
RUN pip install --no-cache-dir "vllm>=0.4.0"

# ── Source code ───────────────────────────────────────────────────────────────
COPY src/       src/
COPY configs/   configs/
COPY scripts/   scripts/
COPY predict.sh .

RUN chmod +x predict.sh

# ── Runtime ───────────────────────────────────────────────────────────────────
# /data    → mount read-only: split dir (data/, questions.jsonl, rulebook.md …)
# /result  → mount read-write: submission.jsonl + logs/
# /models  → mount read-only: LLM weight dir (e.g. /models/Qwen2.5-14B-Instruct)
VOLUME ["/data", "/result", "/models"]

ENTRYPOINT ["./predict.sh", "/data", "/result/submission.jsonl"]

