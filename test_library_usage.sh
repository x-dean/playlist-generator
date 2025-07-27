#!/bin/bash

# Script to test the playlist generator analysis pipeline with custom paths
# Now supports multiple pipeline variants and other modules
#
# IMPORTANT: This script uses docker-compose.test.yaml for test volume mappings:
#   - /root/music/test_files:/music
#   - /root/music/test/cache:/app/cache
#   - /root/music/test/logs:/app/logs
#   - /root/music/playlista/models/musicnn:/app/feature_extraction/models/musicnn
#   - /root/music/playlist_output:/app/playlists
#
# Host paths:         Container paths:
#   /root/music/test_files      -> /music
#   /root/music/test/cache      -> /app/cache
#   /root/music/test/logs       -> /app/logs
#   /root/music/playlista/models/musicnn -> /app/feature_extraction/models/musicnn
#   /root/music/playlist_output -> /app/playlists
#
# All docker compose commands use: -f docker-compose.test.yaml

HOST_LIBRARY_DIR="/root/music/test_files"            # Host test music library
HOST_CACHE_DIR="/root/music/test/cache"              # Host cache directory
HOST_LOGS_DIR="/root/music/test/logs"                # Host logs directory
HOST_MODELS_DIR="/root/music/playlista/models/musicnn" # Host models directory (musicnn)
HOST_OUTPUT_DIR="/root/music/playlist_output"        # Host output directory

CONTAINER_LIBRARY_DIR="/music"             # Container music library
CONTAINER_CACHE_DIR="/app/cache"           # Container cache directory
CONTAINER_LOGS_DIR="/app/logs"             # Container logs directory
CONTAINER_MODELS_DIR="/app/feature_extraction/models/musicnn"  # Container models directory (musicnn)
CONTAINER_OUTPUT_DIR="/app/playlists"      # Container output directory

COMPOSE_FILE="-f docker-compose.test.yaml"

# Create directories if they don't exist
mkdir -p "$HOST_LIBRARY_DIR"
mkdir -p "$HOST_CACHE_DIR"
mkdir -p "$HOST_LOGS_DIR"
mkdir -p "$HOST_MODELS_DIR"
mkdir -p "$HOST_OUTPUT_DIR"

# Print mapping info
cat <<EOF
=== Playlist Generator: Analysis Pipeline Variants (TEST) ===

Host paths:
  Library: $HOST_LIBRARY_DIR
  Cache:   $HOST_CACHE_DIR
  Logs:    $HOST_LOGS_DIR
  Models:  $HOST_MODELS_DIR
  Output:  $HOST_OUTPUT_DIR

Container paths (used in CLI arguments):
  Library: $CONTAINER_LIBRARY_DIR
  Cache:   $CONTAINER_CACHE_DIR
  Logs:    $CONTAINER_LOGS_DIR
  Models:  $CONTAINER_MODELS_DIR
  Output:  $CONTAINER_OUTPUT_DIR

Docker Compose file: docker-compose.test.yaml
  volumes:
    - $HOST_LIBRARY_DIR:$CONTAINER_LIBRARY_DIR
    - $HOST_CACHE_DIR:$CONTAINER_CACHE_DIR
    - $HOST_LOGS_DIR:$CONTAINER_LOGS_DIR
    - $HOST_MODELS_DIR:$CONTAINER_MODELS_DIR
    - $HOST_OUTPUT_DIR:$CONTAINER_OUTPUT_DIR
EOF

echo ""
echo "Select an operation:"
echo "1) Standard analysis pipeline (automatic)"
echo "2) Low memory analysis pipeline"
echo "3) Playlist generation (future)"
echo "4) Show statistics (future)"
echo "5) Custom command"
echo "6) Exit"
echo ""
read -p "Enter your choice (1-6): " choice

case $choice in
    1)
        echo "Running standard analysis pipeline..."
        docker compose $COMPOSE_FILE run --rm --remove-orphans playlista playlista --analyze --library "$CONTAINER_LIBRARY_DIR" --cache_dir "$CONTAINER_CACHE_DIR" --output_dir "$CONTAINER_OUTPUT_DIR" --workers 2
        ;;
    2)
        echo "Running low memory analysis pipeline..."
        docker compose $COMPOSE_FILE run --rm --remove-orphans playlista playlista --analyze --low_memory --workers 1 --library "$CONTAINER_LIBRARY_DIR" --cache_dir "$CONTAINER_CACHE_DIR" --output_dir "$CONTAINER_OUTPUT_DIR"
        ;;
    3)
        echo "[Playlist generation is for future use. Uncomment below to enable.]"
        # docker compose $COMPOSE_FILE run --rm --remove-orphans playlista playlista --generate_only --library "$CONTAINER_LIBRARY_DIR" --cache_dir "$CONTAINER_CACHE_DIR" --output_dir "$CONTAINER_OUTPUT_DIR"
        ;;
    4)
        echo "[Show statistics is for future use. Uncomment below to enable.]"
        # docker compose $COMPOSE_FILE run --rm --remove-orphans playlista playlista --status --library "$CONTAINER_LIBRARY_DIR" --cache_dir "$CONTAINER_CACHE_DIR"
        ;;
    5)
        echo "Enter your custom command:"
        read -p "docker compose $COMPOSE_FILE run --rm --remove-orphans playlista playlista " custom_cmd
        eval "docker compose $COMPOSE_FILE run --rm --remove-orphans playlista playlista $custom_cmd --library $CONTAINER_LIBRARY_DIR --cache_dir $CONTAINER_CACHE_DIR --output_dir $CONTAINER_OUTPUT_DIR"
        ;;
    6)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo "Invalid choice. Exiting..."
        exit 1
        ;;
esac

echo ""
echo "=== Operation Complete ==="
echo "Check the cache directory for analysis data (on host):"
echo "   ls -la $HOST_CACHE_DIR"
echo "Check the logs directory for log files (on host):"
echo "   ls -la $HOST_LOGS_DIR"
echo "Check the models directory for model files (on host):"
echo "   ls -la $HOST_MODELS_DIR"
echo "Check the output directory for generated playlists (on host):"
echo "   ls -la $HOST_OUTPUT_DIR"
echo ""
# Playlist generation and stats can be enabled in the future as needed 