# --- Build stage ---
FROM python:3.7-slim-bullseye AS builder

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libffi-dev \
    libssl-dev \
    libasound2-dev \
    portaudio19-dev \
    python3-dev \
    libfftw3-dev \
    libtag1-dev && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies in a virtualenv (for clean copy)
WORKDIR /install
COPY requirements_docker.txt requirements.txt
RUN pip install --prefix=/install/deps --no-cache-dir -r requirements.txt

# --- Final stage ---
FROM python:3.7-slim-bullseye

# Install runtime dependencies only
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    libavcodec-extra \
    libfftw3-3 \
    libtag1v5 \
    sqlite3 \
    libmagic1 \
    gosu && \
    rm -rf /var/lib/apt/lists/*

# Copy only the installed Python packages from builder
COPY --from=builder /install/deps /usr/local

# Set working directory
WORKDIR /app

# Copy application code (this layer will be rebuilt when code changes)
COPY src /app/src

# Create directories with proper permissions
RUN mkdir -p /app/playlists /app/cache/checkpoints /app/cache/metrics /app/logs

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CACHE_DIR=/app/cache \
    MODEL_DIR=/app/feature_extraction/models \
    MUSIC_PATH=/music \
    LOG_DIR=/app/logs \
    OUTPUT_DIR=/app/playlists

# Add src to Python path
ENV PYTHONPATH="${PYTHONPATH}:/app/src"

# Default command
CMD ["python", "-m", "src.presentation.cli.main"] 