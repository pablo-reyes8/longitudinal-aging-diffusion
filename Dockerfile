# CUDA runtime for training/inference. Model weights and datasets are mounted at runtime.
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/workspace/.cache/huggingface

WORKDIR /workspace/project

COPY pyproject.toml README.md LICENSE ./
COPY data ./data
COPY src ./src
COPY scripts ./scripts
COPY config ./config

RUN python -m pip install --upgrade pip && \
    python -m pip install -e ".[notebooks,auxiliary]"

VOLUME ["/workspace/data", "/workspace/checkpoints", "/workspace/outputs", "/workspace/.cache"]

CMD ["python", "-m", "scripts.train", "--help"]
