#!/bin/bash
set -euo pipefail

# Default parameters
HOST_MUSIC_DIR="${PWD}/music"
MUSIC_DIR="/music"
OUTPUT_DIR="${PWD}/playlists"
CACHE_DIR="${PWD}/cache"
WORKERS=$(($(nproc) - 1))

while [[ $# -gt 0 ]]; do
    case "$1" in
        --host_music_dir=*) HOST_MUSIC_DIR="${1#*=}" ;;
        --music_dir=*) MUSIC_DIR="${1#*=}" ;;
        --output_dir=*) OUTPUT_DIR="${1#*=}" ;;
        --cache_dir=*) CACHE_DIR="${1#*=}" ;;
        --workers=*) WORKERS="${1#*=}" ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
    shift
done

mkdir -p "$OUTPUT_DIR" "$CACHE_DIR"

export HOST_MUSIC_DIR MUSIC_DIR OUTPUT_DIR CACHE_DIR WORKERS

docker compose run --rm \
  playlist-generator \
  --music_dir "$MUSIC_DIR" \
  --workers "$WORKERS"

echo "Processing complete. Results in: $OUTPUT_DIR"