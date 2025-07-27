#!/bin/bash

# Script to test the playlist generator with custom paths
# This allows you to use any library, cache, and output directories

echo "=== Playlist Generator Test with Custom Paths ==="
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

# Example commands for different operations:
echo "=== Available Commands ==="
echo ""
echo "1. Analyze your test library:"
echo "   docker compose run --rm --remove-orphans playlista --analyze --library $LIBRARY_DIR --cache_dir $CACHE_DIR --output_dir $OUTPUT_DIR --workers 2"
echo ""
echo "2. Generate playlists from existing analysis:"
echo "   docker compose run --rm --remove-orphans playlista --generate_only --library $LIBRARY_DIR --cache_dir $CACHE_DIR --output_dir $OUTPUT_DIR"
echo ""
echo "3. Show library statistics:"
echo "   docker compose run --rm --remove-orphans playlista --status --library $LIBRARY_DIR --cache_dir $CACHE_DIR"
echo ""
echo "4. Analyze with specific playlist method:"
echo "   docker compose run --rm --remove-orphans playlista --analyze --playlist_method tags --library $LIBRARY_DIR --cache_dir $CACHE_DIR --output_dir $OUTPUT_DIR"
echo ""
echo "5. Force re-analyze all files:"
echo "   docker compose run --rm --remove-orphans playlista --analyze --force --library $LIBRARY_DIR --cache_dir $CACHE_DIR --output_dir $OUTPUT_DIR"
echo ""
echo "6. Re-analyze only failed files:"
echo "   docker compose run --rm --remove-orphans playlista --analyze --failed --library $LIBRARY_DIR --cache_dir $CACHE_DIR --output_dir $OUTPUT_DIR"
echo ""
echo "7. Low memory mode (for large files):"
echo "   docker compose run --rm --remove-orphans playlista --analyze --low_memory --workers 1 --library $LIBRARY_DIR --cache_dir $CACHE_DIR --output_dir $OUTPUT_DIR"
echo ""
# Ask user which command to run
echo "Which operation would you like to perform?"
echo "1) Analyze library"
echo "2) Generate playlists only"
echo "3) Show statistics"
echo "4) Custom command"
echo "5) Exit"
echo ""
read -p "Enter your choice (1-5): " choice

case $choice in
    1)
        echo "Running analysis..."
        docker compose run --rm --remove-orphans playlista --analyze --library "$LIBRARY_DIR" --cache_dir "$CACHE_DIR" --output_dir "$OUTPUT_DIR" --workers 2
        ;;
    2)
        echo "Generating playlists..."
        docker compose run --rm --remove-orphans playlista --generate_only --library "$LIBRARY_DIR" --cache_dir "$CACHE_DIR" --output_dir "$OUTPUT_DIR"
        ;;
    3)
        echo "Showing statistics..."
        docker compose run --rm --remove-orphans playlista --status --library "$LIBRARY_DIR" --cache_dir "$CACHE_DIR"
        ;;
    4)
        echo "Enter your custom command:"
        read -p "docker compose run --rm --remove-orphans playlista " custom_cmd
        eval "docker compose run --rm --remove-orphans playlista $custom_cmd --library $LIBRARY_DIR --cache_dir $CACHE_DIR --output_dir $OUTPUT_DIR"
        ;;
    5)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo "Invalid choice. Exiting..."
        exit 1
        ;;
esac

echo ""
echo "=== Test Complete ==="
echo "Check the output directory for generated playlists:"
echo "   ls -la $OUTPUT_DIR"
echo ""
echo "Check the cache directory for analysis data:"
echo "   ls -la $CACHE_DIR" 