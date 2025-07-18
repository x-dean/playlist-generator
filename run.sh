#!/bin/bash
set -euo pipefail

# Default parameters
REBUILD=false
HOST_MUSIC_DIR="${PWD}/music"
MUSIC_DIR="/music"
OUTPUT_DIR="${PWD}/playlists"
CACHE_DIR="${PWD}/cache"
WORKERS=$(($(nproc) - 1))
NUM_PLAYLISTS=10
CHUNK_SIZE=1000
TIMEOUT=45
BATCH_SIZE=30
USE_DB=false
FORCE_SEQUENTIAL=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --rebuild) REBUILD=true ;;
        --host_music_dir=*) HOST_MUSIC_DIR="${1#*=}" ;;
        --music_dir=*) MUSIC_DIR="${1#*=}" ;;
        --output_dir=*) OUTPUT_DIR="${1#*=}" ;;
        --cache_dir=*) CACHE_DIR="${1#*=}" ;;
        --workers=*) WORKERS="${1#*=}" ;;
        --num_playlists=*) NUM_PLAYLISTS="${1#*=}" ;;
        --chunk_size=*) CHUNK_SIZE="${1#*=}" ;;
        --timeout=*) TIMEOUT="${1#*=}" ;;
        --batch_size=*) BATCH_SIZE="${1#*=}" ;;
        --use_db) USE_DB=true ;;
        --force_sequential) FORCE_SEQUENTIAL=true ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
    shift
done

mkdir -p "$OUTPUT_DIR" "$CACHE_DIR"

echo "=== Optimized Playlist Generator ==="
echo "Batch Size: $BATCH_SIZE"
echo "Workers: $WORKERS"
echo "Timeout: $TIMEOUT seconds"

if "$REBUILD"; then
    docker compose build --no-cache
fi

export HOST_MUSIC_DIR OUTPUT_DIR CACHE_DIR TIMEOUT

docker compose run --rm \
  -e "MEMORY_MANAGEMENT=aggressive" \
  playlist-generator \
  --music_dir "$MUSIC_DIR" \
  --host_music_dir "$HOST_MUSIC_DIR" \
  --output_dir /app/playlists \
  --workers "$WORKERS" \
  --num_playlists "$NUM_PLAYLISTS" \
  --chunk_size "$CHUNK_SIZE" \
  --timeout "$TIMEOUT" \
  --batch_size "$BATCH_SIZE" \
  $( [ "$USE_DB" = true ] && echo "--use_db" ) \
  $( [ "$FORCE_SEQUENTIAL" = true ] && echo "--force_sequential" )

echo "Processing complete. Results in: $OUTPUT_DIR"