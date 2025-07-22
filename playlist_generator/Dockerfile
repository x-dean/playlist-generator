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
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -s /bin/bash appuser

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages

# Set working directory and permissions
WORKDIR /app
RUN chown appuser:appuser /app

# Copy application files
COPY --chown=appuser:appuser . /app

# Create __init__.py files to make directories importable
RUN touch /app/database/__init__.py && \
    touch /app/music_analyzer/__init__.py && \
    touch /app/playlist_generator/__init__.py

# Switch to non-root user
USER appuser

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

# Set entrypoint
ENTRYPOINT ["python", "main.py"]