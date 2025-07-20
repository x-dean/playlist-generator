#!/bin/bash
set -euo pipefail

# Default parameters
REBUILD=false
MUSIC_DIR="${HOME}/music/library"
OUTPUT_DIR="${HOME}/music/playlists/by_bpm"
CACHE_DIR="${HOME}/music/playlists/by_bpm/cache"
WORKERS=$(nproc)
NUM_PLAYLISTS=10
CHUNK_SIZE=1000
USE_DB=false
FORCE_SEQUENTIAL=false

# Fixed container paths
CONTAINER_MUSIC="/music"
CONTAINER_OUTPUT="/app/playlists"
CONTAINER_CACHE="/app/cache"

# Enhanced argument parsing
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --rebuild)
            REBUILD=true
            shift
            ;;
        --host_music_dir|--host_music_dir=*)
            if [[ $key == *=* ]]; then
                MUSIC_DIR="${key#*=}"
            else
                MUSIC_DIR="$2"
                shift
            fi
            shift
            ;;
        --host_output_dir|--host_output_dir=*)
            if [[ $key == *=* ]]; then
                OUTPUT_DIR="${key#*=}"
            else
                OUTPUT_DIR="$2"
                shift
            fi
            shift
            ;;
        --host_cache_dir|--host_cache_dir=*)
            if [[ $key == *=* ]]; then
                CACHE_DIR="${key#*=}"
            else
                CACHE_DIR="$2"
                shift
            fi
            shift
            ;;
        --workers|--workers=*)
            if [[ $key == *=* ]]; then
                WORKERS="${key#*=}"
            else
                WORKERS="$2"
                shift
            fi
            shift
            ;;
        --num_playlists|--num_playlists=*)
            if [[ $key == *=* ]]; then
                NUM_PLAYLISTS="${key#*=}"
            else
                NUM_PLAYLISTS="$2"
                shift
            fi
            shift
            ;;
        --chunk_size|--chunk_size=*)
            if [[ $key == *=* ]]; then
                CHUNK_SIZE="${key#*=}"
            else
                CHUNK_SIZE="$2"
                shift
            fi
            shift
            ;;
        --use_db)
            USE_DB=true
            shift
            ;;
        --force_sequential)
            FORCE_SEQUENTIAL=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --rebuild                Rebuild the Docker image"
            echo "  --host_music_dir PATH    Host path to music directory"
            echo "  --host_output_dir PATH   Host path for output playlists"
            echo "  --host_cache_dir PATH    Host path for cache"
            echo "  --workers NUM            Number of worker threads"
            echo "  --num_playlists NUM      Number of playlists to generate"
            echo "  --chunk_size SIZE        Size of each chunk for processing"
            echo "  --use_db                 Use database for storing features"
            echo "  --force_sequential       Force sequential processing"
            echo ""
            echo "Container Paths (fixed):"
            echo "  Music: $CONTAINER_MUSIC"
            echo "  Output: $CONTAINER_OUTPUT"
            echo "  Cache: $CONTAINER_CACHE"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Resolve paths
resolve_path() {
    local path="$1"
    if [[ "$path" != /* ]]; then
        path="${PWD%/}/${path}"
    fi
    path="${path//\/\//\/}"
    echo "${path%/}"
}

MUSIC_DIR=$(resolve_path "$MUSIC_DIR")
OUTPUT_DIR=$(resolve_path "$OUTPUT_DIR")
CACHE_DIR=$(resolve_path "$CACHE_DIR")

# Create directories
mkdir -p "$OUTPUT_DIR" "$CACHE_DIR"
chmod -R a+rwX "$OUTPUT_DIR" "$CACHE_DIR"

# Print configuration
echo "=== Playlist Generator Configuration ==="
echo "Host Music Directory: $MUSIC_DIR"
echo "Host Output Directory: $OUTPUT_DIR"
echo "Host Cache Directory: $CACHE_DIR"
echo "Container Music Mount: $CONTAINER_MUSIC"
echo "Container Output Mount: $CONTAINER_OUTPUT"
echo "Container Cache Mount: $CONTAINER_CACHE"
echo "Workers: $WORKERS"
echo "Playlists: $NUM_PLAYLISTS"
echo "Chunk Size: $CHUNK_SIZE"
echo "Use DB: $USE_DB"
echo "Force Sequential: $FORCE_SEQUENTIAL"
echo "========================================"

# Validate music directory exists
if [ ! -d "$MUSIC_DIR" ]; then
    echo "ERROR: Music directory not found: $MUSIC_DIR"
    exit 1
fi

# Build only if requested
if [ "$REBUILD" = true ]; then
    echo "=== Rebuilding Docker image ==="
    docker compose build --no-cache
fi

# Run the generator
docker compose run --rm \
  -v "$MUSIC_DIR:$CONTAINER_MUSIC:ro" \
  -v "$OUTPUT_DIR:$CONTAINER_OUTPUT" \
  -v "$CACHE_DIR:$CONTAINER_CACHE" \
  playlist-generator \
  --host_music_dir "$MUSIC_DIR" \
  --host_output_dir "$OUTPUT_DIR" \
  $( [ -n "$CACHE_DIR" ] && echo "--host_cache_dir $CACHE_DIR" ) \
  --num_playlists "$NUM_PLAYLISTS" \
  --workers "$WORKERS" \
  --chunk_size "$CHUNK_SIZE" \
  $( [ "$USE_DB" = true ] && echo "--use_db" ) \
  $( [ "$FORCE_SEQUENTIAL" = true ] && echo "--force_sequential" )

echo "Playlists generated!"
echo "Output available in: $OUTPUT_DIR"