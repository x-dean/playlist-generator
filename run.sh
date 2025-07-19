#!/bin/bash

# Run with proper resource allocation
docker run --rm \
  -v /root/music/library:/music:ro \
  -v /root/music/library/by_bpm:/output \
  -v /root/music/cache:/cache \
  -e ESSENTIA_THREADS=8 \
  playlist-generator-playlist-generator \
  --output_dir /output \
  --workers 8 \
  --db_path /cache/audio.db