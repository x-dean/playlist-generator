@echo off
REM Docker test runner for the refactored playlista application

echo 🐳 Building and running Docker test for playlista-refactored...

REM Build the Docker image
echo 🔨 Building Docker image...
docker build -t playlista-refactored .

REM Run the test
echo 🧪 Running Docker test...
docker run --rm ^
  -v "%cd%/logs:/app/logs" ^
  -v "%cd%/cache:/app/cache" ^
  playlista-refactored ^
  python test_docker_setup.py

echo ✅ Docker test completed!
pause 