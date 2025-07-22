#!/bin/bash
set -euo pipefail

# Default parameters
REBUILD=false
MUSIC_DIR="/root/music/library"
HOST_MUSIC_DIR="$MUSIC_DIR"
OUTPUT_DIR="/root/music/library/playlists"
CACHE_DIR="/root/music/library/playlists/cache"
WORKERS=$(($(nproc) / 2))
if [ "$WORKERS" -lt 1 ]; then
    WORKERS=1
fi
NUM_PLAYLISTS=10
FORCE_SEQUENTIAL=false
GENERATE_ONLY=false
ANALYZE_ONLY=false
UPDATE=false
PLAYLIST_METHOD="all"  # Default to all methods

# Get current user's UID and GID
CURRENT_UID=$(id -u)
CURRENT_GID=$(id -g)

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --rebuild)
            REBUILD=true
            shift
            ;;
        --music_dir=*)
            MUSIC_DIR="${1#*=}"
            HOST_MUSIC_DIR="$MUSIC_DIR"
            shift
            ;;
        --host_music_dir=*)
            HOST_MUSIC_DIR="${1#*=}"
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
        --generate_only)
            GENERATE_ONLY=true
            shift
            ;;
        --analyze_only)
            ANALYZE_ONLY=true
            shift
            ;;
        --update)
            UPDATE=true
            shift
            ;;
        --num_playlists=*)
            NUM_PLAYLISTS="${1#*=}"
            shift
            ;;
        --force_sequential)
            FORCE_SEQUENTIAL=true
            shift
            ;;
        --playlist_method=*)
            PLAYLIST_METHOD="${1#*=}"
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --rebuild                Rebuild the Docker image"
            echo "  --music_dir=<path>       Path to the music directory (default: $MUSIC_DIR)"
            echo "  --host_music_dir=<path>  Host path to music directory (default: $HOST_MUSIC_DIR)"
            echo "  --output_dir=<path>      Path to the output directory (default: $OUTPUT_DIR)"
            echo "  --cache_dir=<path>       Path to the cache directory (default: $CACHE_DIR)"
            echo "  --workers=<num>          Number of worker threads (default: $(nproc))"
            echo "  --num_playlists=<num>    Number of playlists to generate (default: $NUM_PLAYLISTS)"
            echo "  --force_sequential       Force sequential processing (default: false)"
            echo "  --generate_only          Only generate playlists from database without analysis"
            echo "  --analyze_only           Only run audio analysis without generating playlists"
            echo "  --update                 Update playlists from existing database"
            echo "  --playlist_method=<method> Playlist generation method (default: all)"
            echo "                           Options: all, time, kmeans, cache"
            echo "  --help, -h               Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate playlist method
case "$PLAYLIST_METHOD" in
    all|time|kmeans|cache)
        ;;
    *)
        echo "Invalid playlist method: $PLAYLIST_METHOD"
        echo "Valid options are: all, time, kmeans, cache"
        exit 1
        ;;
esac

# Create directories with proper permissions
mkdir -p "$OUTPUT_DIR" "$CACHE_DIR"
chmod 755 "$OUTPUT_DIR" "$CACHE_DIR"

# Export environment variables for compose
export MUSIC_DIR
export HOST_MUSIC_DIR
export OUTPUT_DIR
export CACHE_DIR
export WORKERS
export NUM_PLAYLISTS
export CURRENT_UID
export CURRENT_GID
export PLAYLIST_METHOD

# Export boolean flags
export FORCE_SEQUENTIAL=${FORCE_SEQUENTIAL}
export GENERATE_ONLY=${GENERATE_ONLY}
export ANALYZE_ONLY=${ANALYZE_ONLY}
export UPDATE=${UPDATE}

# Create .env file for docker-compose
cat > .env << EOF
MUSIC_DIR=${MUSIC_DIR}
HOST_MUSIC_DIR=${HOST_MUSIC_DIR}
OUTPUT_DIR=${OUTPUT_DIR}
CACHE_DIR=${CACHE_DIR}
WORKERS=${WORKERS}
NUM_PLAYLISTS=${NUM_PLAYLISTS}
CURRENT_UID=${CURRENT_UID}
CURRENT_GID=${CURRENT_GID}
FORCE_SEQUENTIAL=${FORCE_SEQUENTIAL}
GENERATE_ONLY=${GENERATE_ONLY}
ANALYZE_ONLY=${ANALYZE_ONLY}
UPDATE=${UPDATE}
PLAYLIST_METHOD=${PLAYLIST_METHOD}
EOF

# Print configuration
echo "=== Playlist Generator Configuration ==="
echo "Music Directory (Container): $MUSIC_DIR"
echo "Music Directory (Host): $HOST_MUSIC_DIR"
echo "Output Directory: $OUTPUT_DIR"
echo "Cache Directory: $CACHE_DIR"
echo "Workers: $WORKERS"
echo "Playlists: $NUM_PLAYLISTS"
echo "Force Sequential: ${FORCE_SEQUENTIAL}"
echo "Generate Only: ${GENERATE_ONLY}"
echo "Analyze Only: ${ANALYZE_ONLY}"
echo "Update Mode: ${UPDATE}"
echo "Playlist Method: ${PLAYLIST_METHOD}"
echo "Running as UID:GID = $CURRENT_UID:$CURRENT_GID"
echo "========================================"

# Build only if requested
if [ "$REBUILD" = true ]; then
    echo "=== Rebuilding Docker image ==="
    docker compose build --no-cache
fi

# Determine which mutually exclusive flag to pass
MUTEX_FLAG=""
if [ "$ANALYZE_ONLY" = true ]; then
    MUTEX_FLAG="--analyze_only"
elif [ "$GENERATE_ONLY" = true ]; then
    MUTEX_FLAG="--generate_only"
elif [ "$UPDATE" = true ]; then
    MUTEX_FLAG="--update"
fi

# Run the generator
echo "=== Starting Playlist Generation ==="
docker compose up --force-recreate --remove-orphans --build --detach

docker compose exec playlist-generator python main.py \
  --music_dir /music \
  --host_music_dir ${HOST_MUSIC_DIR} \
  --output_dir /app/playlists \
  --workers ${WORKERS} \
  --num_playlists ${NUM_PLAYLISTS} \
  --playlist_method ${PLAYLIST_METHOD} \
  $MUTEX_FLAG \
  ${FORCE_SEQUENTIAL:+--force_sequential}

echo "Playlists generated successfully!"
echo "Output available in: $OUTPUT_DIR"