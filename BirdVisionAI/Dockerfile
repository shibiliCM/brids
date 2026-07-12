# Stage 1: Builder — install dependencies
FROM python:3.11-slim AS builder

WORKDIR /install

COPY requirements.txt .

RUN pip install --no-cache-dir --prefix=/deps -r requirements.txt

# Stage 2: Runtime — lean image
FROM python:3.11-slim AS runtime

LABEL maintainer="BirdVision AI Team"
LABEL description="BirdVision AI v2 — Hybrid Bird Species Recognition System"
LABEL version="2.0.0"

# System libs required by opencv-python-headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Copy pre-installed packages from builder
COPY --from=builder /deps /usr/local

WORKDIR /app

# Copy project files
COPY app/        ./app/
COPY checkpoints/ ./checkpoints/
COPY data/        ./data/
COPY .streamlit/  ./.streamlit/

# Expose Streamlit default port
EXPOSE 8501

# Streamlit config via env vars (avoids needing config.toml mounted)
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLE_CORS=false
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s \
  CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Entry point
CMD ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
