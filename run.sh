#!/bin/bash

# Run with proper resource allocation
docker run --rm \
  --cpus=8 \
  --memory=8g \
  --memory-swap=12g \
  -v /root/music/library:/music:ro \
  -v /root/music/library/by_bpm:/output \
  -v /root/music/cache:/cache \
  -e ESSENTIA_THREADS=8 \
  playlist-generator \
  --music_dir /music \
  --output_dir /output \
  --workers 8 \
  --db_path /cache/audio.db