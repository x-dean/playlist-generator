#!/bin/bash

# Entrypoint script for Playlist Generator Simple
# Supports both API server and CLI modes

set -e

# Default to keeping container alive if no arguments provided
if [ $# -eq 0 ]; then
    echo "Playlist Generator Simple container is running"
    echo "Available commands: analyze, playlist, stats, status, db"
    echo "Use: docker exec playlist-generator python -m src.main --help for more information"
    echo "Container will stay running for interactive use"
    echo "Press Ctrl+C to stop the container"
    
    # Keep container running
    tail -f /dev/null
fi

# Check if first argument is 'api' or 'server'
if [ "$1" = "api" ] || [ "$1" = "server" ]; then
    echo "Starting Playlist Generator API server..."
    shift  # Remove the 'api' argument
    exec python -m uvicorn src.main:app --host 0.0.0.0 --port 8500 "$@"
fi

# Check if first argument is 'cli' or 'playlista'
if [ "$1" = "cli" ] || [ "$1" = "playlista" ]; then
    echo "Starting Playlist Generator CLI..."
    shift  # Remove the 'cli' argument
    exec python -m src.main "$@"
fi

# If no recognized mode, assume CLI mode and pass all arguments
echo "Starting Playlist Generator CLI..."
exec python -m src.main "$@" 