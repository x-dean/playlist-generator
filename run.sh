#!/bin/bash
set -euo pipefail

# Default parameters (with more flexible defaults)
REBUILD=false
MUSIC_DIR="${HOME}/music/library"  # Changed from /root to $HOME
OUTPUT_DIR="${HOME}/music/playlists/by_bpm"
CACHE_DIR="${HOME}/music/playlists/by_bpm/cache"
WORKERS=$(nproc)
NUM_PLAYLISTS=10
CHUNK_SIZE=1000
USE_DB=false
FORCE_SEQUENTIAL=false

# Fixed container paths (shouldn't change)
CONTAINER_MUSIC="/music"
CONTAINER_OUTPUT="/app/playlists"
CONTAINER_CACHE="/app/cache"

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
            echo "  --music_dir=<path>       Host path to music directory (default: $MUSIC_DIR)"
            echo "  --output_dir=<path>      Host path for output playlists (default: $OUTPUT_DIR)"
            echo "  --cache_dir=<path>       Host path for cache (default: $CACHE_DIR)"
            echo "  --workers=<num>          Number of worker threads (default: $(nproc))"
            echo "  --num_playlists=<num>    Number of playlists to generate (default: $NUM_PLAYLISTS)"
            echo "  --chunk_size=<size>      Size of each chunk for processing (default: $CHUNK_SIZE)"
            echo "  --use_db                 Use database for storing features (default: false)"
            echo "  --force_sequential       Force sequential processing (default: false)"
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

# Portable path resolution (replaces realpath -m)
resolve_path() {
    local path="$1"
    # If path is relative, prepend current directory
    if [[ "$path" != /* ]]; then
        path="${PWD%/}/${path}"
    fi
    # Normalize path (remove .., ., etc)
    while [[ "$path" =~ (.*)/[^/]+/\.\.(.*) ]]; do
        path="${BASH_REMATCH[1]}${BASH_REMATCH[2]}"
    done
    path="${path//\/.\//\/}"
    echo "${path%/}"
}

# Resolve all paths to absolute form
MUSIC_DIR=$(resolve_path "$MUSIC_DIR")
OUTPUT_DIR=$(resolve_path "$OUTPUT_DIR")
CACHE_DIR=$(resolve_path "$CACHE_DIR")

# Create directories with proper permissions
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

# Run the generator with all parameters
docker compose run --rm \
  -v "$MUSIC_DIR:$CONTAINER_MUSIC:ro" \
  -v "$OUTPUT_DIR:$CONTAINER_OUTPUT" \
  -v "$CACHE_DIR:$CONTAINER_CACHE" \
  playlist-generator \
  --music_dir "$CONTAINER_MUSIC" \
  --host_music_dir "$MUSIC_DIR" \
  --output_dir "$CONTAINER_OUTPUT" \
  --cache_dir "$CONTAINER_CACHE" \
  --workers "$WORKERS" \
  --num_playlists "$NUM_PLAYLISTS" \
  --chunk_size "$CHUNK_SIZE" \
  $( [ "$USE_DB" = true ] && echo "--use_db" ) \
  $( [ "$FORCE_SEQUENTIAL" = true ] && echo "--force_sequential" )

echo "✅ Playlists generated successfully!"
echo "Output available in: $OUTPUT_DIR"