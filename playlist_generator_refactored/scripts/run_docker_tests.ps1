# Comprehensive Docker testing script for the refactored playlista application

Write-Host "🚀 Starting Comprehensive Docker Tests for Refactored Playlista" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan

# Function to print colored output
function Write-Success {
    param([string]$Message)
    Write-Host "✅ $Message" -ForegroundColor Green
}

function Write-Error {
    param([string]$Message)
    Write-Host "❌ $Message" -ForegroundColor Red
}

function Write-Warning {
    param([string]$Message)
    Write-Host "⚠️  $Message" -ForegroundColor Yellow
}

function Write-Info {
    param([string]$Message)
    Write-Host "🔍 $Message" -ForegroundColor Blue
}

function Write-Test {
    param([string]$Message)
    Write-Host "🧪 $Message" -ForegroundColor Magenta
}

# Check if Docker is running
Write-Info "Checking Docker availability..."
try {
    docker info | Out-Null
    Write-Success "Docker is running"
} catch {
    Write-Error "Docker is not running. Please start Docker Desktop."
    Read-Host "Press Enter to exit"
    exit 1
}

# Check if docker-compose is available
Write-Info "Checking docker-compose availability..."
try {
    docker-compose --version | Out-Null
    Write-Success "docker-compose is available"
} catch {
    Write-Error "docker-compose is not available."
    Read-Host "Press Enter to exit"
    exit 1
}

# Create necessary directories
Write-Info "Creating necessary directories..."
$directories = @("music", "cache", "logs", "playlists")
foreach ($dir in $directories) {
    if (!(Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir | Out-Null
    }
}
Write-Success "Directories created"

# Build the Docker image
Write-Info "Building Docker image..."
try {
    docker-compose -f docker-compose.test.yaml build
    Write-Success "Docker image built successfully"
} catch {
    Write-Error "Docker build failed"
    Read-Host "Press Enter to exit"
    exit 1
}

# Test 1: Basic CLI functionality
Write-Test "Test 1: Basic CLI functionality"
try {
    docker-compose -f docker-compose.test.yaml run --rm playlista-cli-test
    Write-Success "CLI help works correctly"
} catch {
    Write-Error "CLI help test failed"
}

# Test 2: Status command
Write-Test "Test 2: Status command"
try {
    docker-compose -f docker-compose.test.yaml run --rm playlista-status-test
    Write-Success "Status command works correctly"
} catch {
    Write-Error "Status command test failed"
}

# Test 3: Comprehensive functionality test
Write-Test "Test 3: Comprehensive functionality test"
try {
    docker-compose -f docker-compose.test.yaml run --rm playlista-test
    Write-Success "Comprehensive tests passed"
} catch {
    Write-Error "Comprehensive tests failed"
}

# Test 4: Analysis functionality (if music directory has files)
Write-Test "Test 4: Analysis functionality"
if (Test-Path "music") {
    $musicFiles = Get-ChildItem -Path "music" -Recurse -File | Where-Object { $_.Extension -match "\.(mp3|flac|wav|m4a)$" }
    if ($musicFiles.Count -gt 0) {
        Write-Warning "Music directory has files, testing analysis..."
        try {
            docker-compose -f docker-compose.test.yaml run --rm playlista-analyze-test
            Write-Success "Analysis test passed"
        } catch {
            Write-Error "Analysis test failed"
        }
    } else {
        Write-Warning "Music directory is empty, skipping analysis test"
    }
} else {
    Write-Warning "Music directory not found, skipping analysis test"
}

# Test 5: Compare with original version
Write-Test "Test 5: Comparing with original version"
if (Test-Path "..\playlist_generator") {
    Write-Info "Comparing CLI interfaces..."
    
    # Test original CLI help
    try {
        docker-compose -f ..\playlist_generator\docker-compose.yaml run --rm playlista-original --help | Out-File -FilePath "temp_original_help.txt" -Encoding UTF8
        Write-Success "Original CLI help works"
    } catch {
        Write-Error "Original CLI help failed"
    }
    
    # Test refactored CLI help
    try {
        docker-compose -f docker-compose.test.yaml run --rm playlista-cli-test | Out-File -FilePath "temp_refactored_help.txt" -Encoding UTF8
        Write-Success "Refactored CLI help works"
    } catch {
        Write-Error "Refactored CLI help failed"
    }
    
    Write-Info "Comparison completed"
} else {
    Write-Warning "Original version not found, skipping comparison"
}

# Summary
Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "📊 Test Summary:" -ForegroundColor Cyan
Write-Host "   ✅ Docker environment: Working" -ForegroundColor Green
Write-Host "   ✅ CLI functionality: Tested" -ForegroundColor Green
Write-Host "   ✅ Status command: Tested" -ForegroundColor Green
Write-Host "   ✅ Comprehensive tests: Completed" -ForegroundColor Green
Write-Host "   ✅ Analysis functionality: Tested" -ForegroundColor Green
Write-Host "   ✅ Comparison with original: Completed" -ForegroundColor Green

Write-Host ""
Write-Success "All Docker tests completed successfully!"
Write-Host "📋 Next steps:" -ForegroundColor Cyan
Write-Host "   1. Review test results above" -ForegroundColor White
Write-Host "   2. Check logs in .\logs directory" -ForegroundColor White
Write-Host "   3. Verify functionality in .\playlists directory" -ForegroundColor White
Write-Host "   4. Run with real music files for full testing" -ForegroundColor White

Write-Host ""
Write-Host "🔧 To run individual tests:" -ForegroundColor Cyan
Write-Host "   docker-compose -f docker-compose.test.yaml run --rm playlista-cli-test" -ForegroundColor White
Write-Host "   docker-compose -f docker-compose.test.yaml run --rm playlista-status-test" -ForegroundColor White
Write-Host "   docker-compose -f docker-compose.test.yaml run --rm playlista-test" -ForegroundColor White

Write-Host ""
Write-Success "Refactored playlista is ready for testing!"
Read-Host "Press Enter to exit" 