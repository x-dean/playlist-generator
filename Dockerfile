FROM python:3.9-slim

# Install essentia with minimal dependencies
RUN apt-get update && apt-get install -y \
    libfftw3-dev \
    libavcodec-dev \
    libavformat-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

# Set optimal environment variables
ENV ESSENTIA_THREADS=1
ENV OMP_NUM_THREADS=1
ENV NUMBA_NUM_THREADS=1

CMD ["python", "-O", "playlist_generator.py"]