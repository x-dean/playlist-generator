# --- Build stage ---
FROM python:3.7-slim AS builder

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
COPY requirements.txt requirements.txt
RUN pip install --prefix=/install/deps --no-cache-dir -r requirements.txt

# --- Final stage ---
FROM python:3.7-slim

# Install runtime dependencies only
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    libavcodec-extra \
    libfftw3-double3 \
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

# Copy CLI entry point and requirements
COPY playlista /usr/local/bin/playlista
COPY README.md /app/README.md
COPY requirements.txt /app/requirements.txt

# Make playlista executable
RUN chmod +x /usr/local/bin/playlista

# Create directories with proper permissions
RUN mkdir -p /app/playlists /app/cache/checkpoints /app/cache/metrics /app/logs

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CACHE_DIR=/app/cache \
    MODEL_DIR=/app/feature_extraction/models \
    MUSICNN_MODEL_PATH=/app/feature_extraction/models/musicnn/msd-musicnn-1.pb \
    MUSIC_PATH=/music \
    LOG_DIR=/app/logs \
    OUTPUT_DIR=/app/playlists

# Add src to Python path
ENV PYTHONPATH="${PYTHONPATH}:/app/src"

# Default command
CMD ["playlista"] 