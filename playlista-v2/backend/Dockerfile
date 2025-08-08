# Multi-stage build for production optimization
FROM python:3.11-slim as essentia-builder

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

# Install build dependencies for Essentia
RUN apt-get update && apt-get install -y \
    build-essential \
    libeigen3-dev \
    libyaml-dev \
    libfftw3-dev \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswresample-dev \
    python3-dev \
    python3-numpy-dev \
    python3-yaml \
    libsamplerate0-dev \
    libtag1-dev \
    libchromaprint-dev \
    git \
    cmake \
    wget \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies for building
RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    tensorflow==2.15.0 \
    pyyaml

# Clone and build Essentia with TensorFlow support
WORKDIR /tmp
RUN git clone https://github.com/MTG/essentia.git && \
    cd essentia && \
    git checkout v2.1_beta6 && \
    ./waf configure --build-static --with-python --with-tensorflow && \
    ./waf && \
    ./waf install

# Production stage
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libfftw3-3 \
    libavcodec59 \
    libavformat59 \
    libavutil57 \
    libswresample4 \
    libsamplerate0 \
    libtag1v5 \
    libchromaprint1 \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Essentia from builder stage
COPY --from=essentia-builder /usr/local/lib/python3.11/site-packages/essentia /usr/local/lib/python3.11/site-packages/essentia
COPY --from=essentia-builder /usr/local/lib/libessentia.so /usr/local/lib/
COPY --from=essentia-builder /usr/local/include/essentia /usr/local/include/essentia

# Update library path
RUN ldconfig

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/

# Create non-root user
RUN useradd --create-home --shell /bin/bash playlista && \
    chown -R playlista:playlista /app

USER playlista

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Default command
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
