#!/bin/bash

# Script to test the playlist generator analysis pipeline with custom paths
# Now supports multiple pipeline variants and other modules

echo "=== Playlist Generator: Analysis Pipeline Variants ==="
echo ""

# Set your custom paths here
LIBRARY_DIR="/tmp"                    # Your test music library
CACHE_DIR="/tmp/playlist_cache"       # Cache directory for analysis data
OUTPUT_DIR="/tmp/playlist_output"     # Output directory for generated playlists

# Create directories if they don't exist
mkdir -p "$CACHE_DIR"
mkdir -p "$OUTPUT_DIR"

echo "Using paths:"
echo "  Library: $LIBRARY_DIR"
echo "  Cache:   $CACHE_DIR"
echo "  Output:  $OUTPUT_DIR"
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
        docker compose run --rm --remove-orphans playlista --analyze --library "$LIBRARY_DIR" --cache_dir "$CACHE_DIR" --output_dir "$OUTPUT_DIR" --workers 2
        ;;
    2)
        echo "Running low memory analysis pipeline..."
        docker compose run --rm --remove-orphans playlista --analyze --low_memory --workers 1 --library "$LIBRARY_DIR" --cache_dir "$CACHE_DIR" --output_dir "$OUTPUT_DIR"
        ;;
    3)
        echo "[Playlist generation is for future use. Uncomment below to enable.]"
        # docker compose run --rm --remove-orphans playlista --generate_only --library "$LIBRARY_DIR" --cache_dir "$CACHE_DIR" --output_dir "$OUTPUT_DIR"
        ;;
    4)
        echo "[Show statistics is for future use. Uncomment below to enable.]"
        # docker compose run --rm --remove-orphans playlista --status --library "$LIBRARY_DIR" --cache_dir "$CACHE_DIR"
        ;;
    5)
        echo "Enter your custom command:"
        read -p "docker compose run --rm --remove-orphans playlista " custom_cmd
        eval "docker compose run --rm --remove-orphans playlista $custom_cmd --library $LIBRARY_DIR --cache_dir $CACHE_DIR --output_dir $OUTPUT_DIR"
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
echo "Check the cache directory for analysis data:"
echo "   ls -la $CACHE_DIR"
echo "Check the output directory for generated playlists:"
echo "   ls -la $OUTPUT_DIR"
echo ""
# Playlist generation and stats can be enabled in the future as needed 