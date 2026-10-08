# API image: FastAPI + embedding model.
#
#   docker build -t llm-studio-api .
#
# Cloud Run sets $PORT; it defaults to 8080 here.

FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/huggingface \
    PORT=8080

WORKDIR /srv

# Install the CPU-only build of PyTorch first. The default Linux build
# bundles CUDA libraries (several GB) that are useless without a GPU.
RUN pip install --index-url https://download.pytorch.org/whl/cpu \
    torch==2.13.0+cpu

# Dependencies are copied and installed before the code, so Docker can
# reuse this (slow) layer when only the code changes.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Download the embedding model at build time, not at startup, so a new
# container can serve requests without waiting on the network
# (important for Cloud Run cold starts).
ARG EMBEDDING_MODEL=all-MiniLM-L6-v2
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${EMBEDDING_MODEL}')"

COPY app ./app
COPY data ./data

# Don't run as root inside the container
RUN useradd --create-home --uid 1000 appuser
USER appuser

EXPOSE 8080

# Shell form so $PORT is expanded; exec so uvicorn receives stop signals
CMD exec uvicorn api.main:app --app-dir app --host 0.0.0.0 --port ${PORT}
