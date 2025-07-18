#!/bin/bash
set -euo pipefail

# Default parameters
REBUILD=false
HOST_MUSIC_DIR="${PWD}/music"
MUSIC_DIR="/music"
OUTPUT_DIR="${PWD}/playlists"
CACHE_DIR="${PWD}/cache"
WORKERS=$(($(nproc) - 1))
TIMEOUT=30

while [[ $# -gt 0 ]]; do
    case "$1" in
        --rebuild) REBUILD=true ;;
        --host_music_dir=*) HOST_MUSIC_DIR="${1#*=}" ;;
        --music_dir=*) MUSIC_DIR="${1#*=}" ;;
        --output_dir=*) OUTPUT_DIR="${1#*=}" ;;
        --cache_dir=*) CACHE_DIR="${1#*=}" ;;
        --workers=*) WORKERS="${1#*=}" ;;
        --timeout=*) TIMEOUT="${1#*=}" ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
    shift
done

# Create necessary directories
mkdir -p "$OUTPUT_DIR" "$CACHE_DIR"

echo "=== Music Playlist Generator ==="
echo "Music Directory: $HOST_MUSIC_DIR (container: $MUSIC_DIR)"
echo "Output Directory: $OUTPUT_DIR"
echo "Workers: $WORKERS"
echo "Timeout: $TIMEOUT seconds"
echo "Rebuild Image: $REBUILD"

# Rebuild Docker image if requested
if "$REBUILD"; then
    echo "Rebuilding Docker image..."
    docker compose build --no-cache
fi

# Export environment variables
export HOST_MUSIC_DIR MUSIC_DIR OUTPUT_DIR CACHE_DIR WORKERS TIMEOUT

# Run the generator
docker compose run --rm \
  -e "WORKERS=$WORKERS" \
  -e "TIMEOUT=$TIMEOUT" \
  playlist-generator \
  --music_dir "$MUSIC_DIR" \
  --output_dir /app/playlists \
  --workers "$WORKERS" \
  --timeout "$TIMEOUT"

echo "Processing complete. Playlists available in: $OUTPUT_DIR"