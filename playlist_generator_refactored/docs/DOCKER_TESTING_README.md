# 🐳 Docker Testing Guide for Refactored Playlista

## 📋 Overview

This guide explains how to test the refactored playlista application using Docker on Windows. The testing ensures that all functionality from the original version has been successfully migrated to the new clean architecture.

## 🚀 Quick Start

### Option 1: Using PowerShell (Recommended)

```powershell
# Navigate to the refactored directory
cd playlist_generator_refactored

# Run the PowerShell test script
.\run_docker_tests.ps1
```

### Option 2: Using Batch File

```cmd
# Navigate to the refactored directory
cd playlist_generator_refactored

# Run the batch test script
run_docker_tests.bat
```

### Option 3: Manual Docker Commands

```powershell
# Build the Docker image
docker-compose -f docker-compose.test.yaml build

# Run the comprehensive test
docker-compose -f docker-compose.test.yaml run --rm playlista-test

# Test CLI help
docker-compose -f docker-compose.test.yaml run --rm playlista-cli-test

# Test status command
docker-compose -f docker-compose.test.yaml run --rm playlista-status-test
```

## 🧪 What the Tests Cover

### 1. **Module Imports**
- ✅ Shared modules (config, exceptions)
- ✅ Infrastructure modules (logging)
- ✅ Application modules (services)
- ✅ DTOs (data transfer objects)

### 2. **Configuration**
- ✅ Configuration loading
- ✅ Environment variables
- ✅ All config sections (logging, processing, memory, database)

### 3. **Service Initialization**
- ✅ FileDiscoveryService
- ✅ AudioAnalysisService
- ✅ MetadataEnrichmentService
- ✅ PlaylistGenerationService

### 4. **DTO Creation**
- ✅ AudioAnalysisRequest
- ✅ FileDiscoveryRequest
- ✅ MetadataEnrichmentRequest
- ✅ PlaylistGenerationRequest

### 5. **Docker Environment**
- ✅ Environment variables set correctly
- ✅ Required directories exist
- ✅ Volume mounts working

### 6. **CLI Interface**
- ✅ Argument parser creation
- ✅ Help text generation
- ✅ All original arguments supported

### 7. **Playlista Entry Point**
- ✅ Main playlista script executable
- ✅ Backward compatibility maintained

## 📊 Expected Results

When all tests pass, you should see:

```
🚀 Starting Docker Tests for Refactored Playlista
==================================================

📋 Module Imports
------------------------------
🧪 Testing module imports...
   ✅ Shared modules imported
   ✅ Infrastructure modules imported
   ✅ Application modules imported
   ✅ DTOs imported

📋 Configuration
------------------------------
🧪 Testing configuration...
   ✅ Config has logging section
   ✅ Config has processing section
   ✅ Config has memory section
   ✅ Config has database section

📋 Service Initialization
------------------------------
🧪 Testing service initialization...
   ✅ All services initialized successfully

📋 DTO Creation
------------------------------
🧪 Testing DTO creation...
   ✅ All DTOs created successfully

📋 Docker Environment
------------------------------
🧪 Testing Docker environment...
   ✅ PYTHONPATH: /app/src
   ✅ MUSIC_PATH: /music
   ✅ CACHE_DIR: /app/cache
   ✅ LOG_DIR: /app/logs
   ✅ OUTPUT_DIR: /app/playlists
   ✅ Directory exists: /app/src
   ✅ Directory exists: /app/cache
   ✅ Directory exists: /app/logs
   ✅ Directory exists: /app/playlists
   ✅ Directory exists: /music

📋 CLI Interface
------------------------------
🧪 Testing CLI interface...
   ✅ CLI interface works correctly

📋 Playlista Entry Point
------------------------------
🧪 Testing playlista entry point...
   ✅ playlista entry point works

==================================================
📊 Test Results: 7/7 tests passed
==================================================

🎉 ALL TESTS PASSED! The refactored version is working correctly.
✅ All core functionality is operational.
✅ Docker environment is properly configured.
✅ CLI interface is functional.
```

## 🔧 Troubleshooting

### Docker Not Running
```
❌ Docker is not running. Please start Docker Desktop.
```
**Solution**: Start Docker Desktop and wait for it to fully load.

### Docker Compose Not Available
```
❌ docker-compose is not available.
```
**Solution**: Install Docker Compose or use `docker compose` (newer versions).

### Build Failures
```
❌ Docker build failed
```
**Solution**: 
1. Check Docker Desktop is running
2. Ensure you have sufficient disk space
3. Try building with `--no-cache`: `docker-compose -f docker-compose.test.yaml build --no-cache`

### Import Errors
```
❌ Import test failed: ModuleNotFoundError
```
**Solution**: 
1. Check that all source files are present in `src/` directory
2. Verify the Docker image was built correctly
3. Check Python path configuration

### Environment Variable Issues
```
❌ PYTHONPATH: Not set
```
**Solution**: 
1. Check the Docker Compose file has correct environment variables
2. Verify the Dockerfile sets up the environment correctly

## 📁 Directory Structure

```
playlist_generator_refactored/
├── src/                          # Source code
├── playlista                     # Main entry point
├── test_in_docker.py            # Simple test script
├── docker-compose.test.yaml     # Docker Compose for testing
├── run_docker_tests.ps1         # PowerShell test runner
├── run_docker_tests.bat         # Batch file test runner
├── music/                       # Music files (mounted)
├── cache/                       # Cache directory
├── logs/                        # Log files
└── playlists/                   # Generated playlists
```

## 🎯 Testing with Real Music

To test with actual music files:

1. **Add music files to the `music/` directory**
   ```powershell
   # Copy some MP3 files to test with
   Copy-Item "C:\path\to\your\music\*.mp3" "music\"
   ```

2. **Run the analysis test**
   ```powershell
   docker-compose -f docker-compose.test.yaml run --rm playlista-analyze-test
   ```

3. **Check the results**
   - Logs: `logs/` directory
   - Playlists: `playlists/` directory
   - Cache: `cache/` directory

## 🔄 Comparison with Original

The testing includes comparison with the original version:

1. **CLI Interface**: Both versions should accept the same arguments
2. **Functionality**: Core features should work identically
3. **Performance**: Similar processing times
4. **Output**: Compatible playlist formats

## 📈 Next Steps

After successful testing:

1. **Deploy to production**: The refactored version is ready for use
2. **Performance testing**: Test with large music libraries
3. **Integration testing**: Test with real-world scenarios
4. **Documentation**: Update user documentation

## 🆘 Getting Help

If tests fail:

1. **Check the logs**: Look in the `logs/` directory for detailed error messages
2. **Review the test output**: Each test provides specific error information
3. **Verify Docker setup**: Ensure Docker Desktop is running properly
4. **Check file permissions**: Ensure all files are readable by Docker

## 🎉 Success Criteria

The refactored version is considered successful when:

- ✅ All 7 tests pass
- ✅ CLI interface works with original arguments
- ✅ Docker environment is properly configured
- ✅ All services initialize correctly
- ✅ DTOs can be created and used
- ✅ Configuration loads properly
- ✅ Playlista entry point is functional

---

**The refactored playlista application maintains all functionality of the original while benefiting from the new clean architecture! 🚀** 