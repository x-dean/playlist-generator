#!/bin/bash

# Docker test script for playlist generator analysis system
# Runs comprehensive integration tests in a container with all requirements

set -e

echo "🚀 Starting Docker Integration Tests for Analysis System"
echo "========================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed or not in PATH"
    exit 1
fi

print_status "Docker is available"

# Create test Dockerfile
cat > Dockerfile.test << 'EOF'
FROM python:3.7-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libffi-dev \
    libssl-dev \
    libasound2-dev \
    portaudio19-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install additional test dependencies
RUN pip install --no-cache-dir \
    psutil \
    "numpy<2.0" \
    pytest \
    pytest-cov \
    mock

# Copy source code
COPY src/ ./src/
COPY test_analysis_integration.py .

# Create test directories
RUN mkdir -p /music /app/cache /root/music/library

# Set environment variables
ENV PYTHONPATH=/app/src
ENV MUSIC_PATH=/music
ENV CACHE_FILE=/app/cache/audio_analysis.db

# Run tests
CMD ["python", "test_analysis_integration.py"]
EOF

print_status "Created test Dockerfile"

# Build test image
print_status "Building test Docker image..."
docker build -f Dockerfile.test -t playlist-generator-test .

if [ $? -eq 0 ]; then
    print_success "Docker image built successfully"
else
    print_error "Failed to build Docker image"
    exit 1
fi

# Run tests in container
print_status "Running integration tests in Docker container..."
echo "========================================================"

docker run --rm \
    --memory=4g \
    --cpus=2 \
    -v $(pwd)/logs:/app/logs \
    playlist-generator-test

TEST_EXIT_CODE=$?

echo "========================================================"

# Check test results
if [ $TEST_EXIT_CODE -eq 0 ]; then
    print_success "All tests passed!"
    echo ""
    echo "🎉 Analysis System Integration Test Results:"
    echo "✅ Analysis Manager: Deterministic decisions based on file size"
    echo "✅ Resource Manager: Forced guidance based on resource constraints"
    echo "✅ Audio Analyzer: On/off feature extraction with configuration"
    echo "✅ Parallel Analyzer: Simplified worker behavior"
    echo "✅ Sequential Analyzer: Large file processing"
    echo "✅ Docker Compatibility: Paths and dependencies"
    echo ""
    echo "📊 Key Test Scenarios Verified:"
    echo "  • File size-based analysis decisions (deterministic)"
    echo "  • Resource constraint handling (forced basic analysis)"
    echo "  • Feature extraction with on/off control"
    echo "  • Worker simplification (just do the job)"
    echo "  • Docker environment compatibility"
    echo ""
else
    print_error "Some tests failed (exit code: $TEST_EXIT_CODE)"
    echo ""
    echo "🔍 Test Failure Analysis:"
    echo "  • Check the test output above for specific failures"
    echo "  • Verify that all dependencies are properly installed"
    echo "  • Ensure Docker has sufficient resources (memory/CPU)"
    echo "  • Check that the source code structure is correct"
    echo ""
fi

# Cleanup
print_status "Cleaning up test artifacts..."
rm -f Dockerfile.test

print_status "Test run completed"

exit $TEST_EXIT_CODE 