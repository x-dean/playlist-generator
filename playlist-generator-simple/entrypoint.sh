#!/bin/bash

# Entrypoint script for Playlist Generator
# Supports both API server and CLI modes

set -e

# Default to API mode if no arguments provided
if [ $# -eq 0 ]; then
    echo "Starting Playlist Generator API server..."
    exec python -m uvicorn src.main:app --host 0.0.0.0 --port 8500
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
    exec python /app/playlista "$@"
fi

# If no recognized mode, assume CLI mode and pass all arguments
echo "Starting Playlist Generator CLI..."
exec python /app/playlista "$@" 