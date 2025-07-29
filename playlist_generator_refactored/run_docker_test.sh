#!/bin/bash
# Docker test runner for the refactored playlista application

echo "🐳 Building and running Docker test for playlista-refactored..."

# Build the Docker image
echo "🔨 Building Docker image..."
docker build -t playlista-refactored .

# Run the test
echo "🧪 Running Docker test..."
docker run --rm \
  -v "$(pwd)/logs:/app/logs" \
  -v "$(pwd)/cache:/app/cache" \
  playlista-refactored \
  python test_docker_setup.py

echo "✅ Docker test completed!" 