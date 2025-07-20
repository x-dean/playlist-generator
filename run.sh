#!/bin/bash
set -euo pipefail

# Default parameters
REBUILD=false
MUSIC_DIR="/root/music/library"
OUTPUT_DIR="/root/music/library/playlists/by_bpm"
CACHE_DIR="/root/music/library/playlists/by_bpm/cache"
WORKERS=$(nproc)
NUM_PLAYLISTS=10
CHUNK_SIZE=1000
USE_DB=false
FORCE_SEQUENTIAL=false

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --rebuild)
            REBUILD=true
            shift
            ;;
        --music_dir=*)
            MUSIC_DIR="${1#*=}"
            shift
            ;;
        --output_dir=*)
            OUTPUT_DIR="${1#*=}"
            shift
            ;;
        --cache_dir=*)
            CACHE_DIR="${1#*=}"
            shift
            ;;
        --workers=*)
            WORKERS="${1#*=}"
            shift
            ;;
        --num_playlists=*)
            NUM_PLAYLISTS="${1#*=}"
            shift
            ;;
        --chunk_size=*)
            CHUNK_SIZE="${1#*=}"
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
            echo "  --music_dir=<path>       Path to the music directory (default: $MUSIC_DIR)"
            echo "  --output_dir=<path>      Path to the output directory (default: $OUTPUT_DIR)"
            echo "  --cache_dir=<path>       Path to the cache directory (default: $CACHE_DIR)"
            echo "  --workers=<num>          Number of worker threads (default: $(nproc))"
            echo "  --num_playlists=<num>     Number of playlists to generate (default: $NUM_PLAYLISTS)"
            echo "  --chunk_size=<size>      Size of each chunk for processing (default: $CHUNK_SIZE)"
            echo "  --use_db                 Use database for storing features (default: false)"
            echo "  --force_sequential       Force sequential processing (default: false)"
            echo "  --help, -h              Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create directories
mkdir -p "$OUTPUT_DIR" "$CACHE_DIR"

# Export environment variables for compose
export MUSIC_DIR
export OUTPUT_DIR
export CACHE_DIR
export CONFIG_DIR
export WORKERS
export NUM_PLAYLISTS
export CHUNK_SIZE
export USE_DB
export FORCE_SEQUENTIAL

# Print configuration
echo "=== Playlist Generator Configuration ==="
echo "Music Directory: $MUSIC_DIR"
echo "Output Directory: $OUTPUT_DIR"
echo "Cache Directory: $CACHE_DIR"
echo "Workers: $WORKERS"
echo "Playlists: $NUM_PLAYLISTS"
echo "Chunk Size: $CHUNK_SIZE"
echo "Use DB: $USE_DB"
echo "Force Sequential: $FORCE_SEQUENTIAL"
echo "========================================"

# Build only if requested
if [ "$REBUILD" = true ]; then
    echo "=== Rebuilding Docker image ==="
    docker compose build --no-cache
fi

# Run the generator with all parameters
docker compose run --rm \
  -v "$MUSIC_DIR:/music:ro" \
  -v "$OUTPUT_DIR:/app/playlists" \
  -v "$CACHE_DIR:/app/cache" \
  playlist-generator \
  --music_dir /music \
  --host_music_dir "$MUSIC_DIR" \
  --output_dir /app/playlists \
  --workers "$WORKERS" \
  --num_playlists "$NUM_PLAYLISTS" \
  --chunk_size "$CHUNK_SIZE" \
   $( [ "$USE_DB" = true ] && echo "--use_db" ) \
   $( [ "$FORCE_SEQUENTIAL" = true ] && echo "--force_sequential" )

echo "âœ… Playlists generated successfully!"
echo "Output available in: $OUTPUT_DIR"