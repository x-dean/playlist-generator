# Use a multi-stage build for a smaller final image
FROM python:3.9-slim-bullseye as builder

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libfftw3-dev \
    libtag1-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Final stage
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
    gosu \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages

# Set working directory
WORKDIR /app

# Copy application files
COPY . /app

# Create __init__.py files to make directories importable
RUN touch /app/database/__init__.py && \
    touch /app/music_analyzer/__init__.py && \
    touch /app/playlist_generator/__init__.py

# Create directories with proper permissions
RUN mkdir -p /app/playlists /app/cache/checkpoints /app/cache/metrics && \
    chmod -R 777 /app/playlists /app/cache

# Add entrypoint script
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    CACHE_DIR=/app/cache \
    ESSENTIA_LOGGING_LEVEL=warning \
    ESSENTIA_STREAM_LOGGING=none

# Use shell form for entrypoint to allow variable expansion
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh" | grep -v 'AudioLoader:' | grep -v 'MusicExtractorSVM:' | grep -v 'invalid frame']
CMD ["python", "main.py"]