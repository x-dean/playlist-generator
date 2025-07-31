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

# Install runtime dependencies including comprehensive audio codec support
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    libavcodec-extra \
    libavformat-extra \
    libavutil-extra \
    libswresample-extra \
    libavfilter-extra \
    libavdevice-extra \
    libpostproc-extra \
    libswscale-extra \
    # Additional audio codec libraries
    libmp3lame0 \
    libvorbis0a \
    libvorbisenc2 \
    libflac8 \
    libopus0 \
    libfaad2 \
    libfaac0 \
    libmad0 \
    libmpg123-0 \
    # Additional system libraries
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

# Copy application code
COPY src /app/src
COPY playlista.conf /app/playlista.conf
COPY playlista /app/playlista

# Create directories with proper permissions
RUN mkdir -p /app/playlists /app/cache/checkpoints /app/cache/metrics /app/logs /app/feature_extraction/models

# Make playlista executable
RUN chmod +x /app/playlista

# Add to PATH
ENV PATH="${PATH}:/app"

# Set Python environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TF_CPP_MIN_LOG_LEVEL=3 \
    TF_ENABLE_ONEDNN_OPTS=0 \
    CUDA_VISIBLE_DEVICES=-1

# Default command
CMD ["playlista"] 