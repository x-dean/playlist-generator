@echo off
echo 🚀 Starting Real-Life Playlista Tests
echo ======================================

echo.
echo 📋 Available test commands:
echo 1. Comprehensive test (recommended)
echo 2. Audio analysis only
echo 3. Playlist generation only
echo 4. CLI commands only
echo.

set /p choice="Enter your choice (1-4): "

if "%choice%"=="1" (
    echo.
    echo 🎵 Running comprehensive real-life test...
    docker-compose -f docker-compose.real-test.yaml run --rm playlista-real-test
) else if "%choice%"=="2" (
    echo.
    echo 🎼 Running audio analysis test...
    docker-compose -f docker-compose.real-test.yaml run --rm playlista-analyze-real
) else if "%choice%"=="3" (
    echo.
    echo 🎵 Running playlist generation test...
    docker-compose -f docker-compose.real-test.yaml run --rm playlista-playlist-real
) else if "%choice%"=="4" (
    echo.
    echo 🖥️ Running CLI commands test...
    docker-compose -f docker-compose.real-test.yaml run --rm playlista-real-test python -c "import real_test; real_test.test_cli_commands()"
) else (
    echo ❌ Invalid choice. Please run the script again.
    exit /b 1
)

echo.
echo ✅ Test completed!
pause 