#!/bin/bash
set -euo pipefail

# Default parameters
REBUILD=false
HOST_MUSIC_DIR="${PWD}/music"
MUSIC_DIR="/music"
OUTPUT_DIR="${PWD}/playlists"
CACHE_DIR="${PWD}/cache"
WORKERS=$(nproc)
NUM_PLAYLISTS=10
CHUNK_SIZE=1000
TIMEOUT=60
USE_DB=false
FORCE_SEQUENTIAL=false

# Parse command-line arguments
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
        --use_db) USE_DB=true ;;
        --force_sequential) FORCE_SEQUENTIAL=true ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
    shift
done

# Create directories
mkdir -p "$OUTPUT_DIR" "$CACHE_DIR"

echo "=== Playlist Generator Configuration ==="
echo "Host Music Directory: $HOST_MUSIC_DIR"
echo "Container Music Directory: $MUSIC_DIR"
echo "Output Directory: $OUTPUT_DIR"
echo "Cache Directory: $CACHE_DIR"
echo "Workers: $WORKERS"
echo "Playlists: $NUM_PLAYLISTS"
echo "Chunk Size: $CHUNK_SIZE"
echo "Timeout: $TIMEOUT seconds"
echo "Use DB: $USE_DB"
echo "Force Sequential: $FORCE_SEQUENTIAL"

if "$REBUILD"; then
    echo "=== Rebuilding Docker image ==="
    docker compose build --no-cache
fi

export HOST_MUSIC_DIR OUTPUT_DIR CACHE_DIR TIMEOUT

docker compose run --rm \
  playlist-generator \
  --music_dir "$MUSIC_DIR" \
  --host_music_dir "$HOST_MUSIC_DIR" \
  --output_dir /app/playlists \
  --workers "$WORKERS" \
  --num_playlists "$NUM_PLAYLISTS" \
  --chunk_size "$CHUNK_SIZE" \
  --timeout "$TIMEOUT" \
  $( [ "$USE_DB" = true ] && echo "--use_db" ) \
  $( [ "$FORCE_SEQUENTIAL" = true ] && echo "--force_sequential" )

echo "Playlists generated successfully!"
echo "Output available in: $OUTPUT_DIR"