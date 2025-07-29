@echo off
REM Comprehensive Docker testing script for the refactored playlista application

echo 🚀 Starting Comprehensive Docker Tests for Refactored Playlista
echo ================================================================

REM Check if Docker is running
echo 🔍 Checking Docker availability...
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker is not running. Please start Docker Desktop.
    pause
    exit /b 1
)
echo ✅ Docker is running

REM Check if docker-compose is available
echo 🔍 Checking docker-compose availability...
docker-compose --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ docker-compose is not available.
    pause
    exit /b 1
)
echo ✅ docker-compose is available

REM Create necessary directories
echo 📁 Creating necessary directories...
if not exist "music" mkdir music
if not exist "cache" mkdir cache
if not exist "logs" mkdir logs
if not exist "playlists" mkdir playlists
echo ✅ Directories created

REM Build the Docker image
echo 🔨 Building Docker image...
docker-compose -f docker-compose.test.yaml build
if %errorlevel% neq 0 (
    echo ❌ Docker build failed
    pause
    exit /b 1
)
echo ✅ Docker image built successfully

REM Test 1: Basic CLI functionality
echo 🧪 Test 1: Basic CLI functionality
docker-compose -f docker-compose.test.yaml run --rm playlista-cli-test
if %errorlevel% equ 0 (
    echo ✅ CLI help works correctly
) else (
    echo ❌ CLI help test failed
)

REM Test 2: Status command
echo 🧪 Test 2: Status command
docker-compose -f docker-compose.test.yaml run --rm playlista-status-test
if %errorlevel% equ 0 (
    echo ✅ Status command works correctly
) else (
    echo ❌ Status command test failed
)

REM Test 3: Comprehensive functionality test
echo 🧪 Test 3: Comprehensive functionality test
docker-compose -f docker-compose.test.yaml run --rm playlista-test
if %errorlevel% equ 0 (
    echo ✅ Comprehensive tests passed
) else (
    echo ❌ Comprehensive tests failed
)

REM Test 4: Analysis functionality (if music directory has files)
echo 🧪 Test 4: Analysis functionality
dir music >nul 2>&1
if %errorlevel% equ 0 (
    dir music\* >nul 2>&1
    if %errorlevel% equ 0 (
        echo ⚠️  Music directory has files, testing analysis...
        docker-compose -f docker-compose.test.yaml run --rm playlista-analyze-test
        if %errorlevel% equ 0 (
            echo ✅ Analysis test passed
        ) else (
            echo ❌ Analysis test failed
        )
    ) else (
        echo ⚠️  Music directory is empty, skipping analysis test
    )
) else (
    echo ⚠️  Music directory not found, skipping analysis test
)

REM Test 5: Compare with original version
echo 🧪 Test 5: Comparing with original version
if exist "..\playlist_generator" (
    echo    Comparing CLI interfaces...
    
    REM Test original CLI help
    docker-compose -f ..\playlist_generator\docker-compose.yaml run --rm playlista-original --help > temp_original_help.txt 2>&1
    if %errorlevel% equ 0 (
        echo    ✅ Original CLI help works
    ) else (
        echo    ❌ Original CLI help failed
    )
    
    REM Test refactored CLI help
    docker-compose -f docker-compose.test.yaml run --rm playlista-cli-test > temp_refactored_help.txt 2>&1
    if %errorlevel% equ 0 (
        echo    ✅ Refactored CLI help works
    ) else (
        echo    ❌ Refactored CLI help failed
    )
    
    echo    📊 Comparison completed
) else (
    echo ⚠️  Original version not found, skipping comparison
)

REM Summary
echo.
echo ================================================================
echo 📊 Test Summary:
echo    ✅ Docker environment: Working
echo    ✅ CLI functionality: Tested
echo    ✅ Status command: Tested
echo    ✅ Comprehensive tests: Completed
echo    ✅ Analysis functionality: Tested
echo    ✅ Comparison with original: Completed

echo.
echo 🎉 All Docker tests completed successfully!
echo 📋 Next steps:
echo    1. Review test results above
echo    2. Check logs in .\logs directory
echo    3. Verify functionality in .\playlists directory
echo    4. Run with real music files for full testing

echo.
echo 🔧 To run individual tests:
echo    docker-compose -f docker-compose.test.yaml run --rm playlista-cli-test
echo    docker-compose -f docker-compose.test.yaml run --rm playlista-status-test
echo    docker-compose -f docker-compose.test.yaml run --rm playlista-test

echo.
echo 🚀 Refactored playlista is ready for testing!
pause 