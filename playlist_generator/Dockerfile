FROM python:3.9-slim-bullseye

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    libavcodec-extra \
    libfftw3-dev \
    libtag1-dev \
    sqlite3 \
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

# Set working directory
WORKDIR /app

# Copy all application files
COPY . /app

# Create __init__.py files to make directories importable
RUN touch /app/database/__init__.py && \
    touch /app/music_analyzer/__init__.py && \
    touch /app/playlist_generator/__init__.py

# Set entrypoint
ENTRYPOINT ["python", "main.py"]