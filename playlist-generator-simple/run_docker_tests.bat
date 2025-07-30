@echo off
REM Docker test script for playlist generator analysis system (Windows)
REM Runs comprehensive integration tests in a container with all requirements

echo 🚀 Starting Docker Integration Tests for Analysis System
echo ========================================================

REM Check if Docker is available
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ [ERROR] Docker is not installed or not in PATH
    exit /b 1
)

echo ✅ [INFO] Docker is available

REM Create test Dockerfile
echo ✅ [INFO] Creating test Dockerfile...
(
echo FROM python:3.7-slim
echo.
echo REM Install system dependencies
echo RUN apt-get update ^&^& apt-get install -y \
echo     build-essential \
echo     libffi-dev \
echo     libssl-dev \
echo     libasound2-dev \
echo     portaudio19-dev \
echo     python3-dev \
echo     ^&^& rm -rf /var/lib/apt/lists/*
echo.
echo REM Set working directory
echo WORKDIR /app
echo.
echo REM Copy requirements first for better caching
echo COPY requirements.txt .
echo.
echo REM Install Python dependencies
echo RUN pip install --no-cache-dir -r requirements.txt
echo.
echo REM Install additional test dependencies
echo RUN pip install --no-cache-dir \
echo     psutil \
echo     "numpy<2.0" \
echo     pytest \
echo     pytest-cov \
echo     mock
echo.
echo REM Copy source code
echo COPY src/ ./src/
echo COPY test_analysis_integration.py .
echo.
echo REM Create test directories
echo RUN mkdir -p /music /app/cache /root/music/library
echo.
echo REM Set environment variables
echo ENV PYTHONPATH=/app/src
echo ENV MUSIC_PATH=/music
echo ENV CACHE_FILE=/app/cache/audio_analysis.db
echo.
echo REM Run tests
echo CMD ["python", "test_analysis_integration.py"]
) > Dockerfile.test

REM Build test image
echo ✅ [INFO] Building test Docker image...
docker build -f Dockerfile.test -t playlist-generator-test .

if %errorlevel% neq 0 (
    echo ❌ [ERROR] Failed to build Docker image
    exit /b 1
)

echo ✅ [SUCCESS] Docker image built successfully

REM Run tests in container
echo ✅ [INFO] Running integration tests in Docker container...
echo ========================================================

docker run --rm --memory=4g --cpus=2 -v %cd%/logs:/app/logs playlist-generator-test

set TEST_EXIT_CODE=%errorlevel%

echo ========================================================

REM Check test results
if %TEST_EXIT_CODE% equ 0 (
    echo ✅ [SUCCESS] All tests passed!
    echo.
    echo 🎉 Analysis System Integration Test Results:
    echo ✅ Analysis Manager: Deterministic decisions based on file size
    echo ✅ Resource Manager: Forced guidance based on resource constraints
    echo ✅ Audio Analyzer: On/off feature extraction with configuration
    echo ✅ Parallel Analyzer: Simplified worker behavior
    echo ✅ Sequential Analyzer: Large file processing
    echo ✅ Docker Compatibility: Paths and dependencies
    echo.
    echo 📊 Key Test Scenarios Verified:
    echo   • File size-based analysis decisions (deterministic)
    echo   • Resource constraint handling (forced basic analysis)
    echo   • Feature extraction with on/off control
    echo   • Worker simplification (just do the job)
    echo   • Docker environment compatibility
    echo.
) else (
    echo ❌ [ERROR] Some tests failed (exit code: %TEST_EXIT_CODE%)
    echo.
    echo 🔍 Test Failure Analysis:
    echo   • Check the test output above for specific failures
    echo   • Verify that all dependencies are properly installed
    echo   • Ensure Docker has sufficient resources (memory/CPU)
    echo   • Check that the source code structure is correct
    echo.
)

REM Cleanup
echo ✅ [INFO] Cleaning up test artifacts...
del Dockerfile.test

echo ✅ [INFO] Test run completed

exit /b %TEST_EXIT_CODE% 