FROM python:3.7-slim-bullseye

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libffi-dev \
    libssl-dev \
    libasound2-dev \
    portaudio19-dev \
    python3-dev \
    libfftw3-dev \
    libtag1-dev \
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
# COPY --from=builder /install/deps /usr/local # This line is removed as per the new_code

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Clean install of essentia-tensorflow and compatible TensorFlow
RUN pip install --upgrade pip && \
    pip uninstall -y essentia essentia-tensorflow tensorflow tensorflow-cpu tensorflow-gpu || true && \
    pip install -r requirements.txt

# Copy application code
COPY app /app

# Create directories with proper permissions
RUN mkdir -p /app/playlists /app/cache/checkpoints /app/cache/metrics

# Make playlista executable and add to PATH
RUN chmod +x /app/playlista
ENV PATH="${PATH}:/app"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CACHE_DIR=/app/cache \
    MODEL_DIR=/app/feature_extraction/models

# Default command
CMD ["playlista"]