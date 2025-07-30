# Docker test script for playlist generator analysis system (PowerShell)
# Runs comprehensive integration tests in a container with all requirements

param(
    [switch]$Verbose
)

# Set error action preference
$ErrorActionPreference = "Stop"

# Colors for output
$Red = "Red"
$Green = "Green"
$Yellow = "Yellow"
$Blue = "Blue"

# Function to print colored output
function Write-Status {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor $Blue
}

function Write-Success {
    param([string]$Message)
    Write-Host "[SUCCESS] $Message" -ForegroundColor $Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor $Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor $Red
}

Write-Host "🚀 Starting Docker Integration Tests for Analysis System" -ForegroundColor $Blue
Write-Host "========================================================" -ForegroundColor $Blue

# Check if Docker is available
try {
    $dockerVersion = docker --version
    Write-Success "Docker is available: $dockerVersion"
} catch {
    Write-Error "Docker is not installed or not in PATH"
    exit 1
}

Write-Status "Creating test Dockerfile..."

# Create test Dockerfile
$dockerfileContent = @"
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
"@

$dockerfileContent | Out-File -FilePath "Dockerfile.test" -Encoding UTF8

# Build test image
Write-Status "Building test Docker image..."
try {
    docker build -f Dockerfile.test -t playlist-generator-test .
    Write-Success "Docker image built successfully"
} catch {
    Write-Error "Failed to build Docker image"
    exit 1
}

# Run tests in container
Write-Status "Running integration tests in Docker container..."
Write-Host "========================================================" -ForegroundColor $Blue

try {
    docker run --rm --memory=4g --cpus=2 -v "${PWD}/logs:/app/logs" playlist-generator-test
    $testExitCode = $LASTEXITCODE
} catch {
    Write-Error "Failed to run tests in Docker container"
    $testExitCode = 1
}

Write-Host "========================================================" -ForegroundColor $Blue

# Check test results
if ($testExitCode -eq 0) {
    Write-Success "All tests passed!"
    Write-Host ""
    Write-Host "🎉 Analysis System Integration Test Results:" -ForegroundColor $Green
    Write-Host "✅ Analysis Manager: Deterministic decisions based on file size" -ForegroundColor $Green
    Write-Host "✅ Resource Manager: Forced guidance based on resource constraints" -ForegroundColor $Green
    Write-Host "✅ Audio Analyzer: On/off feature extraction with configuration" -ForegroundColor $Green
    Write-Host "✅ Parallel Analyzer: Simplified worker behavior" -ForegroundColor $Green
    Write-Host "✅ Sequential Analyzer: Large file processing" -ForegroundColor $Green
    Write-Host "✅ Docker Compatibility: Paths and dependencies" -ForegroundColor $Green
    Write-Host ""
    Write-Host "📊 Key Test Scenarios Verified:" -ForegroundColor $Blue
    Write-Host "  • File size-based analysis decisions (deterministic)" -ForegroundColor $Blue
    Write-Host "  • Resource constraint handling (forced basic analysis)" -ForegroundColor $Blue
    Write-Host "  • Feature extraction with on/off control" -ForegroundColor $Blue
    Write-Host "  • Worker simplification (just do the job)" -ForegroundColor $Blue
    Write-Host "  • Docker environment compatibility" -ForegroundColor $Blue
    Write-Host ""
} else {
    Write-Error "Some tests failed (exit code: $testExitCode)"
    Write-Host ""
    Write-Host "🔍 Test Failure Analysis:" -ForegroundColor $Yellow
    Write-Host "  • Check the test output above for specific failures" -ForegroundColor $Yellow
    Write-Host "  • Verify that all dependencies are properly installed" -ForegroundColor $Yellow
    Write-Host "  • Ensure Docker has sufficient resources (memory/CPU)" -ForegroundColor $Yellow
    Write-Host "  • Check that the source code structure is correct" -ForegroundColor $Yellow
    Write-Host ""
}

# Cleanup
Write-Status "Cleaning up test artifacts..."
if (Test-Path "Dockerfile.test") {
    Remove-Item "Dockerfile.test" -Force
}

Write-Status "Test run completed"

exit $testExitCode 