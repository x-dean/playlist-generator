#!/bin/bash
set -euo pipefail

# Default parameters
MUSIC_DIR="/music"
OUTPUT_DIR="/app/playlists"
CACHE_DIR="/app/cache"
WORKERS=$(nproc)
NUM_PLAYLISTS=10

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
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
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create directories
mkdir -p "$OUTPUT_DIR" "$CACHE_DIR"

# Export environment variables
export MUSIC_DIR
export OUTPUT_DIR
export CACHE_DIR
export WORKERS
export NUM_PLAYLISTS

# Run the generator
docker compose run --rm \
  -v "${MUSIC_DIR}:/music:ro" \
  -v "${OUTPUT_DIR}:/app/playlists" \
  -v "${CACHE_DIR}:/app/cache" \
  playlist-generator \
  --music_dir /music \
  --output_dir /app/playlists \
  --workers "$WORKERS" \
  --num_playlists "$NUM_PLAYLISTS"

echo "Playlists generated successfully!"