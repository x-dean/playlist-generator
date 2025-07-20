FROM python:3.9-slim-bullseye

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    libavcodec-extra \
    libfftw3-dev \
    libtag1-dev \
    libyaml-dev \
    sqlite3 \
    ffmpeg-python \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    pandas \
    scikit-learn \
    tqdm \
    essentia \
    numpy==1.26.4 \
    matplotlib \
    psutil \
    colorlog \
    scipy \
    pydub

# Copy application files
COPY playlist_generator.py /app/
COPY analyze_music.py /app/

# Set working directory
WORKDIR /app

# Set entrypoint
ENTRYPOINT ["python", "playlist_generator.py"]