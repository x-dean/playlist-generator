#!/bin/bash
set -eo pipefail

# Default configuration
HOST_MUSIC_DIR="/root/music/library"
OUTPUT_DIR="/root/music/library/by_bpm"
CACHE_DIR="/root/music/cache"
WORKERS=8
TIMEOUT=30
REBUILD=false

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --host_music_dir=*) HOST_MUSIC_DIR="${1#*=}" ;;
        --output_dir=*) OUTPUT_DIR="${1#*=}" ;;
        --workers=*) WORKERS="${1#*=}" ;;
        --timeout=*) TIMEOUT="${1#*=}" ;;
        --rebuild) REBUILD=true ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# Validate workers count
if [[ ! "$WORKERS" =~ ^[0-9]+$ ]] || [[ "$WORKERS" -lt 1 ]]; then
    echo "Workers must be a positive integer"
    exit 1
fi

# Create necessary directories
mkdir -p "$OUTPUT_DIR" "$CACHE_DIR"

# Performance optimization (run if root)
if [[ $EUID -eq 0 ]]; then
    echo "Optimizing system performance..."
    echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
    sync
    echo 3 > /proc/sys/vm/drop_caches
fi

# Docker build if requested
if [[ "$REBUILD" = true ]]; then
    echo "Rebuilding Docker image..."
    docker compose build --no-cache --pull
fi

echo "=== Music Playlist Generator ==="
echo "Music Directory: $HOST_MUSIC_DIR"
echo "Output Directory: $OUTPUT_DIR"
echo "Cache Directory: $CACHE_DIR"
echo "Workers: $WORKERS"
echo "Timeout: $TIMEOUT seconds"
echo "Rebuild: $REBUILD"

# Run the generator
docker compose run --rm \
  -e "WORKERS=$WORKERS" \
  -e "TIMEOUT=$TIMEOUT" \
  playlist-generator \
  python playlist_generator.py \
    --music_dir /music \
    --output_dir /playlists \
    --workers "$WORKERS" \
    --timeout "$TIMEOUT" \
    --db_path /cache/audio.db

# Fix permissions on output
chown -R $(id -u):$(id -g) "$OUTPUT_DIR"

echo "Processing complete. Playlists available in: $OUTPUT_DIR"