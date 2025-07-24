# --- Build stage ---
FROM python:3.9-slim-bullseye AS builder

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
FROM python:3.9-slim-bullseye

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

# Copy only the necessary application files
COPY app /app

# Create __init__.py files to make directories importable
RUN touch /app/database/__init__.py && \
    touch /app/music_analyzer/__init__.py && \
    touch /app/playlist_generator/__init__.py

# Create directories with proper permissions
RUN mkdir -p /app/playlists /app/cache/checkpoints /app/cache/metrics && \
    chmod -R 777 /app/playlists /app/cache

# Make audiolizer executable and add to PATH
RUN chmod +x /app/audiolizer
ENV PATH="/app:${PATH}"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    CACHE_DIR=/app/cache 

CMD ["audiolizer"]