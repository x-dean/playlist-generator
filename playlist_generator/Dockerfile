# --- Build stage ---
FROM python:3.11-slim-bullseye AS builder

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libfftw3-dev \
    libtag1-dev

# Install Python dependencies in a virtualenv (for clean copy)
WORKDIR /install
COPY requirements.txt .
RUN pip install --prefix=/install/deps --no-cache-dir -r requirements.txt

# --- Final stage ---
FROM python:3.11-slim-bullseye

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libffi-dev \
    libssl-dev \
    libasound2-dev \
    portaudio19-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

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

# Copy requirements first for better caching
COPY requirements.txt .

# Clean install of essentia-tensorflow and compatible TensorFlow
RUN pip install --upgrade pip && \
    pip uninstall -y essentia essentia-tensorflow tensorflow tensorflow-cpu tensorflow-gpu || true && \
    pip install -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/cache /app/logs /app/feature_extraction/models/musicnn

# Set environment variables
ENV PYTHONPATH=/app
ENV MODEL_DIR=/app/feature_extraction/models

# Create __init__.py files to make directories importable
RUN touch /app/database/__init__.py && \
    touch /app/music_analyzer/__init__.py && \
    touch /app/playlist_generator/__init__.py

# Create directories with proper permissions
RUN mkdir -p /app/playlists /app/cache/checkpoints /app/cache/metrics && \
    chmod -R 777 /app/playlists /app/cache

# Make playlista executable and add to PATH
RUN chmod +x /app/playlista
ENV PATH="${PATH}:/app"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    CACHE_DIR=/app/cache 

# Default command
CMD ["python", "playlista"]