#!/bin/bash
# Adaptive, memory-aware parallel analysis: No need to set --workers or --worker_max_mem_mb by default.
# The system will automatically manage parallelism and memory per worker.
# Advanced: Set MAX_MEMORY_MB for total memory cap, or WORKER_MAX_MEM_MB_FORCE to force per-worker memory.
set -euo pipefail

# Suppress Essentia logs globally
export ESSENTIA_LOGGING_LEVEL=error
export ESSENTIA_STREAM_LOGGING=none

# Default parameters
REBUILD=false
MUSIC_DIR="/root/music/library"
HOST_MUSIC_DIR="$MUSIC_DIR"
OUTPUT_DIR="/root/music/library/playlists/by_bpm"
CACHE_DIR="/root/music/library/playlists/cache"
NUM_PLAYLISTS=10
FORCE_SEQUENTIAL=false
GENERATE_ONLY=false
ANALYZE_ONLY=false
UPDATE=false
PLAYLIST_METHOD="all"  # Default to all methods
ENRICH_TAGS=false
FORCE_ENRICH_TAGS=false
ENRICH_ONLY=false
FORCE=false
STATUS=false

# Advanced/override: Only set if provided by user
# WORKERS and WORKER_MAX_MEM_MB are not set by default (adaptive pool will manage)
# MAX_MEMORY_MB and WORKER_MAX_MEM_MB_FORCE can be set by user for advanced control

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
        --music_dir=*|-music_dir=*)
            MUSIC_DIR="${1#*=}"
            HOST_MUSIC_DIR="$MUSIC_DIR"
            shift
            ;;
        --host_music_dir=*|-host_music_dir=*)
            HOST_MUSIC_DIR="${1#*=}"
            shift
            ;;
        --output_dir=*|-output_dir=*)
            OUTPUT_DIR="${1#*=}"
            shift
            ;;
        --cache_dir=*|-cache_dir=*)
            CACHE_DIR="${1#*=}"
            shift
            ;;
        --workers=*|-workers=*)
            WORKERS="${1#*=}"
            shift
            ;;
        --worker_max_mem_mb=*)
            WORKER_MAX_MEM_MB="${1#*=}"
            shift
            ;;
        --max_memory_mb=*)
            MAX_MEMORY_MB="${1#*=}"
            shift
            ;;
        --worker_max_mem_mb_force=*)
            WORKER_MAX_MEM_MB_FORCE="${1#*=}"
            shift
            ;;
        --generate_only|-g)
            GENERATE_ONLY=true
            shift
            ;;
        --analyze_only|-a)
            ANALYZE_ONLY=true
            shift
            ;;
        --update|-u)
            UPDATE=true
            shift
            ;;
        --num_playlists=*|-num_playlists=*)
            NUM_PLAYLISTS="${1#*=}"
            shift
            ;;
        --force_sequential)
            FORCE_SEQUENTIAL=true
            shift
            ;;
        --playlist_method=*|-m)
            if [[ "$1" == --playlist_method=* ]]; then
                PLAYLIST_METHOD="${1#*=}"
                shift
            else
                shift
                PLAYLIST_METHOD="$1"
                shift
            fi
            ;;
        --enrich_tags)
            ENRICH_TAGS=true
            shift
            ;;
        --force_enrich_tags)
            FORCE_ENRICH_TAGS=true
            shift
            ;;
        --enrich_only)
            ENRICH_ONLY=true
            shift
            ;;
        --force)
            FORCE=true
            shift
            ;;
        --status)
            STATUS=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --rebuild                Rebuild the Docker image"
            echo "  --music_dir, -music_dir <path>       Path to the music directory (default: $MUSIC_DIR)"
            echo "  --host_music_dir, -host_music_dir <path>  Host path to music directory (default: $HOST_MUSIC_DIR)"
            echo "  --output_dir, -output_dir <path>      Path to the output directory (default: $OUTPUT_DIR)"
            echo "  --cache_dir, -cache_dir <path>        Path to the cache directory (default: $CACHE_DIR)"
            echo "  --workers, -workers <num>             (Advanced) Number of worker threads (default: adaptive)"
            echo "  --worker_max_mem_mb=<MB>   (Advanced) Max memory (MB) per worker process (default: adaptive)"
            echo "  --max_memory_mb=<MB>       (Advanced) Total memory (MB) allowed for all workers (default: 8192)"
            echo "  --worker_max_mem_mb_force=<MB> (Advanced) Force fixed per-worker memory limit (MB), overrides adaptive"
            echo "  --num_playlists, -num_playlists <num> Number of playlists to generate (default: $NUM_PLAYLISTS)"
            echo "  --force_sequential       Force sequential processing (default: false)"
            echo "  --generate_only, -g      Only generate playlists from database without analysis"
            echo "  --analyze_only, -a       Only run audio analysis without generating playlists"
            echo "  --update, -u             Update playlists from existing database"
            echo "  --playlist_method, -m <method> Playlist generation method (default: all)"
            echo "                           Options: all, time, kmeans, cache, tags"
            echo "  --enrich_tags            Enrich tags using MusicBrainz/Last.fm APIs (default: false)"
            echo "  --force_enrich_tags      Force re-enrichment of tags and overwrite metadata in the database (default: false)"
            echo "  --enrich_only           Enrich tags for all tracks in the database using MusicBrainz/Last.fm APIs (no analysis or playlist generation)"
            echo "  --force                 Force re-enrichment for all tracks in the database (use with --enrich_only)"
            echo "  --status                 Show library/database statistics and exit"
            echo "  --help, -h               Show this help message"
            echo ""
            echo "[Adaptive Parallel Analysis]"
            echo "  By default, you do NOT need to set --workers or --worker_max_mem_mb."
            echo "  The system will automatically manage parallelism and memory per worker."
            echo "  Advanced:"
            echo "    --max_memory_mb=<MB>       Set total memory cap for all workers (default: 8192)"
            echo "    --worker_max_mem_mb_force=<MB>  Force fixed per-worker memory limit (MB)"
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
    all|time|kmeans|cache|tags)
        ;;
    *)
        echo "Invalid playlist method: $PLAYLIST_METHOD"
        echo "Valid options are: all, time, kmeans, cache, tags"
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
export NUM_PLAYLISTS
export CURRENT_UID
export CURRENT_GID
export PLAYLIST_METHOD
# Only export WORKERS, WORKER_MAX_MEM_MB, MAX_MEMORY_MB, WORKER_MAX_MEM_MB_FORCE if set by user
if [[ -n "${WORKERS:-}" ]]; then export WORKERS; fi
if [[ -n "${WORKER_MAX_MEM_MB:-}" ]]; then export WORKER_MAX_MEM_MB; fi
if [[ -n "${MAX_MEMORY_MB:-}" ]]; then export MAX_MEMORY_MB; fi
if [[ -n "${WORKER_MAX_MEM_MB_FORCE:-}" ]]; then export WORKER_MAX_MEM_MB_FORCE; fi

# Set FORCE_SEQUENTIAL_FLAG only if true
FORCE_SEQUENTIAL_FLAG=""
if [ "$FORCE_SEQUENTIAL" = true ]; then
    FORCE_SEQUENTIAL_FLAG="--force_sequential"
fi

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
# Only print Workers if set
if [[ -n "${WORKERS:-}" ]]; then
    echo "Workers: $WORKERS"
fi
echo "Playlists: $NUM_PLAYLISTS"
echo "Force Sequential: ${FORCE_SEQUENTIAL}"
echo "Generate Only: ${GENERATE_ONLY}"
echo "Analyze Only: ${ANALYZE_ONLY}"
echo "Update Mode: ${UPDATE}"
echo "Playlist Method: ${PLAYLIST_METHOD}"
echo "Enrich Tags: ${ENRICH_TAGS}"
echo "Force Enrich Tags: ${FORCE_ENRICH_TAGS}"
echo "Enrich Only: ${ENRICH_ONLY}"
echo "Force Enrich Only: ${FORCE}"
echo "Status Mode: ${STATUS}"
echo "Running as UID:GID = $CURRENT_UID:$CURRENT_GID"
echo "Worker Max Mem (MB): ${WORKER_MAX_MEM_MB:-}"
echo "========================================"

# Build only if requested
if [ "$REBUILD" = true ]; then
    echo "=== Rebuilding Docker image ==="
    docker compose build --no-cache
fi

# Determine which mutually exclusive flag to pass
MUTEX_FLAG=""
if [ "$ANALYZE_ONLY" = true ]; then
    MUTEX_FLAG="-a"
elif [ "$GENERATE_ONLY" = true ]; then
    MUTEX_FLAG="-g"
elif [ "$UPDATE" = true ]; then
    MUTEX_FLAG="-u"
fi

# Add playlist method flag if not default
PLAYLIST_METHOD_FLAG=""
if [ "$PLAYLIST_METHOD" != "all" ]; then
    PLAYLIST_METHOD_FLAG="-m $PLAYLIST_METHOD"
fi

# Add enrich tags flag if enabled
ENRICH_TAGS_FLAG=""
if [ "$ENRICH_TAGS" = true ]; then
    ENRICH_TAGS_FLAG="--enrich_tags"
fi
# Add force enrich tags flag if enabled
FORCE_ENRICH_TAGS_FLAG=""
if [ "$FORCE_ENRICH_TAGS" = true ]; then
    FORCE_ENRICH_TAGS_FLAG="--force_enrich_tags"
fi

# Add enrich only flags if enabled
ENRICH_ONLY_FLAG=""
if [ "$ENRICH_ONLY" = true ]; then
    ENRICH_ONLY_FLAG="--enrich_only"
fi
FORCE_FLAG=""
if [ "$FORCE" = true ]; then
    FORCE_FLAG="--force"
fi

# Build docker-compose or docker run arguments safely
DOCKER_WORKERS_ARG=""
if [[ -n "${WORKERS:-}" ]]; then
    DOCKER_WORKERS_ARG="--workers ${WORKERS}"
fi
# Use $DOCKER_WORKERS_ARG in compose/run commands

# Run the generator
if [ "$STATUS" = true ]; then
    # Only run status, ignore other flags
    echo "=== Showing Library/Database Status ==="
    docker compose up --force-recreate --remove-orphans --build --detach
    docker compose exec playlist-generator python main.py \
      --music_dir /music \
      --host_music_dir ${HOST_MUSIC_DIR} \
      --output_dir /app/playlists \
      --workers ${WORKERS:-} \
      --num_playlists ${NUM_PLAYLISTS} \
      --status
    exit $?
fi

echo "=== Starting Playlist Generation ==="
docker compose up --force-recreate --remove-orphans --build --detach

docker compose exec playlist-generator python main.py \
  --music_dir /music \
  --host_music_dir ${HOST_MUSIC_DIR} \
  --output_dir /app/playlists \
  --workers ${WORKERS:-} \
  --num_playlists ${NUM_PLAYLISTS} \
  $MUTEX_FLAG \
  $PLAYLIST_METHOD_FLAG \
  $FORCE_SEQUENTIAL_FLAG \
  $ENRICH_TAGS_FLAG \
  $FORCE_ENRICH_TAGS_FLAG \
  $ENRICH_ONLY_FLAG \
  $FORCE_FLAG