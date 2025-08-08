# Playlista v2 Development Startup Script
# Run this script to start the development environment

Write-Host "Starting Playlista v2 Development Environment..." -ForegroundColor Green

# Check if Docker is running
try {
    docker info | Out-Null
    Write-Host "✓ Docker is running" -ForegroundColor Green
} catch {
    Write-Host "✗ Docker is not running. Please start Docker Desktop." -ForegroundColor Red
    exit 1
}

# Copy environment file if it doesn't exist
if (-not (Test-Path ".env")) {
    Write-Host "Creating .env file from template..." -ForegroundColor Yellow
    Copy-Item "env.example" ".env"
    Write-Host "✓ .env file created. Please update paths if needed." -ForegroundColor Green
}

# Check if music and model directories exist
$musicPath = "C:\Users\Dean\Documents\test_music"
$modelPath = "C:\Users\Dean\Desktop\musicnn"

if (-not (Test-Path $musicPath)) {
    Write-Host "⚠ Music directory not found: $musicPath" -ForegroundColor Yellow
    Write-Host "  Please update PLAYLISTA_MUSIC_LIBRARY_PATH in .env" -ForegroundColor Yellow
}

if (-not (Test-Path $modelPath)) {
    Write-Host "⚠ Model directory not found: $modelPath" -ForegroundColor Yellow  
    Write-Host "  Please update PLAYLISTA_ML_MODEL_PATH in .env" -ForegroundColor Yellow
}

# Build and start services
Write-Host "Building and starting services..." -ForegroundColor Blue
docker-compose up --build

Write-Host "Playlista v2 services started!" -ForegroundColor Green
Write-Host ""
Write-Host "Access points:" -ForegroundColor Cyan
Write-Host "  • Frontend: http://localhost:3000" -ForegroundColor White
Write-Host "  • Backend API: http://localhost:8000" -ForegroundColor White
Write-Host "  • API Docs: http://localhost:8000/api/docs" -ForegroundColor White
Write-Host "  • Health Check: http://localhost:8000/api/health" -ForegroundColor White
Write-Host ""
Write-Host "To stop services: docker-compose down" -ForegroundColor Yellow
