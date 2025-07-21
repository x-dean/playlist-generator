#!/bin/bash
set -euo pipefail

# Default parameters
REBUILD=false
MUSIC_DIR="/root/music/library"
OUTPUT_DIR="/root/music/library/playlists/by_bpm"
CACHE_DIR="/root/music/library/playlists/by_bpm/cache"
WORKERS=$(( $(command -v nproc >/dev/null 2>&1 && nproc || echo 2) / 2 ))
ANALYZE_ONLY=false
GENERATE_ONLY=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --rebuild) REBUILD=true; shift ;;
        --music_dir=*) MUSIC_DIR="${1#*=}"; shift ;;
        --output_dir=*) OUTPUT_DIR="${1#*=}"; shift ;;
        --cache_dir=*) CACHE_DIR="${1#*=}"; shift ;;
        --workers=*) WORKERS="${1#*=}"; shift ;;
        --analyze_only) ANALYZE_ONLY=true; shift ;;
        --generate_only) GENERATE_ONLY=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Create directories
mkdir -p "$OUTPUT_DIR" "$CACHE_DIR"

# Build Docker image if requested
if [ "$REBUILD" = true ]; then
    docker compose build --no-cache
fi

# Prepare command arguments
CMD_ARGS=(
    "--music_dir" "/music"
    "--host_music_dir" "$MUSIC_DIR"
    "--output_dir" "/app/playlists"
    "--workers" "$WORKERS"
)

if [ "$ANALYZE_ONLY" = true ]; then
    CMD_ARGS+=("--analyze_only")
elif [ "$GENERATE_ONLY" = true ]; then
    CMD_ARGS+=("--generate_only")
fi

# Run Docker container
docker compose run --rm \
  -v "${MUSIC_DIR}:/music:ro" \
  -v "${OUTPUT_DIR}:/app/playlists" \
  -v "${CACHE_DIR}:/app/cache" \
  playlist-generator \
  "${CMD_ARGS[@]}"

echo "Operation completed successfully!"