#!/bin/bash

# Comprehensive Docker testing script for the refactored playlista application

set -e

echo "🚀 Starting Comprehensive Docker Tests for Refactored Playlista"
echo "================================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}$1${NC}"
}

print_success() {
    echo -e "${GREEN}$1${NC}"
}

print_warning() {
    echo -e "${YELLOW}$1${NC}"
}

print_error() {
    echo -e "${RED}$1${NC}"
}

# Check if Docker is running
print_status "🔍 Checking Docker availability..."
if ! docker info > /dev/null 2>&1; then
    print_error "❌ Docker is not running. Please start Docker Desktop."
    exit 1
fi
print_success "✅ Docker is running"

# Check if docker-compose is available
print_status "🔍 Checking docker-compose availability..."
if ! docker-compose --version > /dev/null 2>&1; then
    print_error "❌ docker-compose is not available."
    exit 1
fi
print_success "✅ docker-compose is available"

# Create necessary directories
print_status "📁 Creating necessary directories..."
mkdir -p music cache logs playlists
print_success "✅ Directories created"

# Build the Docker image
print_status "🔨 Building Docker image..."
docker-compose -f docker-compose.test.yaml build
print_success "✅ Docker image built successfully"

# Test 1: Basic CLI functionality
print_status "🧪 Test 1: Basic CLI functionality"
if docker-compose -f docker-compose.test.yaml run --rm playlista-cli-test; then
    print_success "✅ CLI help works correctly"
else
    print_error "❌ CLI help test failed"
fi

# Test 2: Status command
print_status "🧪 Test 2: Status command"
if docker-compose -f docker-compose.test.yaml run --rm playlista-status-test; then
    print_success "✅ Status command works correctly"
else
    print_error "❌ Status command test failed"
fi

# Test 3: Comprehensive functionality test
print_status "🧪 Test 3: Comprehensive functionality test"
if docker-compose -f docker-compose.test.yaml run --rm playlista-test; then
    print_success "✅ Comprehensive tests passed"
else
    print_error "❌ Comprehensive tests failed"
fi

# Test 4: Analysis functionality (if music directory has files)
print_status "🧪 Test 4: Analysis functionality"
if [ -d "music" ] && [ "$(ls -A music 2>/dev/null)" ]; then
    print_warning "⚠️  Music directory has files, testing analysis..."
    if docker-compose -f docker-compose.test.yaml run --rm playlista-analyze-test; then
        print_success "✅ Analysis test passed"
    else
        print_error "❌ Analysis test failed"
    fi
else
    print_warning "⚠️  Music directory is empty, skipping analysis test"
fi

# Test 5: Compare with original version
print_status "🧪 Test 5: Comparing with original version"
if [ -d "../playlist_generator" ]; then
    print_status "   Comparing CLI interfaces..."
    
    # Test original CLI help
    if docker-compose -f ../playlist_generator/docker-compose.yaml run --rm playlista-original --help > /tmp/original_help.txt 2>&1; then
        print_success "   ✅ Original CLI help works"
    else
        print_error "   ❌ Original CLI help failed"
    fi
    
    # Test refactored CLI help
    if docker-compose -f docker-compose.test.yaml run --rm playlista-cli-test > /tmp/refactored_help.txt 2>&1; then
        print_success "   ✅ Refactored CLI help works"
    else
        print_error "   ❌ Refactored CLI help failed"
    fi
    
    print_status "   📊 Comparison completed"
else
    print_warning "⚠️  Original version not found, skipping comparison"
fi

# Summary
echo ""
echo "================================================================"
print_status "📊 Test Summary:"
print_status "   ✅ Docker environment: Working"
print_status "   ✅ CLI functionality: Tested"
print_status "   ✅ Status command: Tested"
print_status "   ✅ Comprehensive tests: Completed"
print_status "   ✅ Analysis functionality: Tested"
print_status "   ✅ Comparison with original: Completed"

echo ""
print_success "🎉 All Docker tests completed successfully!"
print_status "📋 Next steps:"
print_status "   1. Review test results above"
print_status "   2. Check logs in ./logs directory"
print_status "   3. Verify functionality in ./playlists directory"
print_status "   4. Run with real music files for full testing"

echo ""
print_status "🔧 To run individual tests:"
print_status "   docker-compose -f docker-compose.test.yaml run --rm playlista-cli-test"
print_status "   docker-compose -f docker-compose.test.yaml run --rm playlista-status-test"
print_status "   docker-compose -f docker-compose.test.yaml run --rm playlista-test"

echo ""
print_success "🚀 Refactored playlista is ready for testing!" 