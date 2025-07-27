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

# Check if we're running in Docker or locally
if [ -f "/.dockerenv" ] || [ -f "/app/playlista" ]; then
    # Running in Docker container
    echo "Running in Docker container..."
    
    # Example commands for different operations:
    echo ""
    echo "=== Available Commands ==="
    echo ""
    echo "1. Analyze your test library:"
    echo "   playlista --analyze --library $LIBRARY_DIR --cache_dir $CACHE_DIR --output_dir $OUTPUT_DIR --workers 2"
    echo ""
    echo "2. Generate playlists from existing analysis:"
    echo "   playlista --generate_only --library $LIBRARY_DIR --cache_dir $CACHE_DIR --output_dir $OUTPUT_DIR"
    echo ""
    echo "3. Show library statistics:"
    echo "   playlista --status --library $LIBRARY_DIR --cache_dir $CACHE_DIR"
    echo ""
    echo "4. Analyze with specific playlist method:"
    echo "   playlista --analyze --playlist_method tags --library $LIBRARY_DIR --cache_dir $CACHE_DIR --output_dir $OUTPUT_DIR"
    echo ""
    echo "5. Force re-analyze all files:"
    echo "   playlista --analyze --force --library $LIBRARY_DIR --cache_dir $CACHE_DIR --output_dir $OUTPUT_DIR"
    echo ""
    echo "6. Re-analyze only failed files:"
    echo "   playlista --analyze --failed --library $LIBRARY_DIR --cache_dir $CACHE_DIR --output_dir $OUTPUT_DIR"
    echo ""
    echo "7. Low memory mode (for large files):"
    echo "   playlista --analyze --low_memory --workers 1 --library $LIBRARY_DIR --cache_dir $CACHE_DIR --output_dir $OUTPUT_DIR"
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
            playlista --analyze --library "$LIBRARY_DIR" --cache_dir "$CACHE_DIR" --output_dir "$OUTPUT_DIR" --workers 2
            ;;
        2)
            echo "Generating playlists..."
            playlista --generate_only --library "$LIBRARY_DIR" --cache_dir "$CACHE_DIR" --output_dir "$OUTPUT_DIR"
            ;;
        3)
            echo "Showing statistics..."
            playlista --status --library "$LIBRARY_DIR" --cache_dir "$CACHE_DIR"
            ;;
        4)
            echo "Enter your custom command:"
            read -p "playlista " custom_cmd
            eval "playlista $custom_cmd --library $LIBRARY_DIR --cache_dir $CACHE_DIR --output_dir $OUTPUT_DIR"
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
    
else
    # Running locally (outside Docker)
    echo "Running locally (outside Docker)..."
    echo ""
    echo "To use custom paths, you need to run this inside the Docker container."
    echo "Start the container first:"
    echo "   ./run.sh"
    echo ""
    echo "Then run this script again from inside the container."
fi

echo ""
echo "=== Test Complete ==="
echo "Check the output directory for generated playlists:"
echo "   ls -la $OUTPUT_DIR"
echo ""
echo "Check the cache directory for analysis data:"
echo "   ls -la $CACHE_DIR" 