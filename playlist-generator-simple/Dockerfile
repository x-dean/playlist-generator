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
COPY requirements.txt .
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
    libmagic1 \
    gosu \
    curl && \
    rm -rf /var/lib/apt/lists/*

# Copy only the installed Python packages from builder
COPY --from=builder /install/deps /usr/local

# Set working directory
WORKDIR /app

# Copy application code and files
COPY src /app/src
COPY playlista /app/playlista
COPY entrypoint.sh /app/entrypoint.sh
COPY database/postgresql_schema.sql /app/database/postgresql_schema.sql
COPY scripts /app/scripts
COPY playlista.conf /app/playlista.conf

# Create directories with proper permissions
RUN mkdir -p /app/playlists /app/cache/checkpoints /app/cache/metrics /app/logs /app/music /app/database

# Make scripts executable
RUN chmod +x /app/playlista /app/entrypoint.sh

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    CACHE_DIR=/app/cache \
    MODEL_DIR=/app/feature_extraction/models \
    LOG_LEVEL=INFO \
    PROMETHEUS_MULTIPROC_DIR=/tmp

# Expose port
EXPOSE 8500

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8500/api/v1/health || exit 1

# Default command (API mode)
CMD ["/app/entrypoint.sh"] 