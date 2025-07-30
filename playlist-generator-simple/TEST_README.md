# Analysis System Integration Tests

This directory contains comprehensive integration tests for the reorganized analysis system. The tests verify the behavior of different components in various scenarios using Docker containers.

## 🎯 Test Objectives

The tests verify the following key aspects of the reorganized system:

### 1. **Analysis Manager - Deterministic Decisions**
- ✅ File size-based analysis type determination
- ✅ Predictable behavior regardless of system resources
- ✅ Clear reasoning for analysis decisions

### 2. **Resource Manager - Forced Guidance**
- ✅ Resource monitoring and constraint detection
- ✅ Forced basic analysis when resources are limited
- ✅ Automatic recovery when resources improve

### 3. **Audio Analyzer - On/Off Feature Extraction**
- ✅ Configurable feature extraction
- ✅ Resource-aware feature selection
- ✅ Graceful degradation under constraints

### 4. **Worker Simplification**
- ✅ Workers focus on processing, not decision-making
- ✅ Clear separation of concerns
- ✅ Simplified worker logic

### 5. **Docker Compatibility**
- ✅ Path handling in containerized environments
- ✅ Dependency availability
- ✅ Resource constraints in containers

## 🚀 Running the Tests

### Prerequisites

1. **Docker**: Must be installed and running
2. **Disk Space**: At least 2GB available for Docker images
3. **Memory**: At least 4GB RAM available for Docker containers

### Quick Start

#### Linux/macOS
```bash
# Make the script executable
chmod +x run_docker_tests.sh

# Run the tests
./run_docker_tests.sh
```

#### Windows (Command Prompt)
```cmd
run_docker_tests.bat
```

#### Windows (PowerShell)
```powershell
.\run_docker_tests.ps1
```

### Manual Docker Commands

If you prefer to run Docker commands manually:

```bash
# Build the test image
docker build -f Dockerfile.test -t playlist-generator-test .

# Run tests with resource limits
docker run --rm \
  --memory=4g \
  --cpus=2 \
  -v $(pwd)/logs:/app/logs \
  playlist-generator-test
```

## 📊 Test Scenarios

### 1. File Size Scenarios
Tests analysis decisions for different file sizes:

| File Size | Expected Analysis | MusiCNN | Reason |
|-----------|------------------|---------|---------|
| 5MB | Full | ✅ | Small files get full analysis |
| 50MB | Full | ✅ | Medium files get full analysis |
| 100MB | Full | ✅ | Threshold files get full analysis |
| 150MB | Basic | ❌ | Large files get basic analysis |
| 500MB | Basic | ❌ | Very large files get basic analysis |

### 2. Resource Constraint Scenarios
Tests behavior under different resource conditions:

| Scenario | Memory | CPU | Disk | Expected Behavior |
|----------|--------|-----|------|-------------------|
| Normal | 60% | 30% | 50% | Full analysis allowed |
| High Memory | 90% | 40% | 60% | Forced basic analysis |
| High CPU | 70% | 85% | 50% | Forced basic analysis |
| High Disk | 65% | 35% | 95% | Forced basic analysis |

### 3. Feature Extraction Scenarios
Tests on/off feature extraction:

| Feature | Basic Analysis | Full Analysis | Resource Forced |
|---------|----------------|---------------|-----------------|
| Rhythm | ✅ | ✅ | ✅ |
| Spectral | ✅ | ✅ | ✅ |
| Loudness | ✅ | ✅ | ✅ |
| Key | ✅ | ✅ | ✅ |
| MFCC | ✅ | ✅ | ✅ |
| MusiCNN | ❌ | ✅ | ❌ |
| Metadata | ✅ | ✅ | ✅ |

## 🔍 Test Details

### Analysis Manager Tests
- **Deterministic Decisions**: Verifies that analysis type is determined by file size only
- **Configuration Handling**: Tests proper configuration loading and application
- **Error Handling**: Tests graceful handling of missing files and errors

### Resource Manager Tests
- **Resource Monitoring**: Tests real-time resource monitoring
- **Forced Guidance**: Tests forced basic analysis under resource constraints
- **Recovery**: Tests automatic recovery when resources improve
- **Threshold Handling**: Tests different resource threshold scenarios

### Audio Analyzer Tests
- **Feature Configuration**: Tests on/off feature extraction
- **Resource Guidance**: Tests response to resource manager guidance
- **Error Handling**: Tests graceful handling of missing libraries
- **Performance**: Tests extraction performance under different configurations

### Worker Tests
- **Simplification**: Tests that workers focus on processing
- **Configuration**: Tests proper configuration passing to workers
- **Error Handling**: Tests worker error handling and recovery
- **Resource Limits**: Tests worker behavior under resource constraints

### Docker Compatibility Tests
- **Path Handling**: Tests Docker-style path handling
- **Dependencies**: Tests availability of required dependencies
- **Resource Limits**: Tests behavior under container resource limits
- **Environment**: Tests proper environment variable handling

## 📈 Expected Test Results

When all tests pass, you should see output like:

```
🎉 Analysis System Integration Test Results:
✅ Analysis Manager: Deterministic decisions based on file size
✅ Resource Manager: Forced guidance based on resource constraints
✅ Audio Analyzer: On/off feature extraction with configuration
✅ Parallel Analyzer: Simplified worker behavior
✅ Sequential Analyzer: Large file processing
✅ Docker Compatibility: Paths and dependencies

📊 Key Test Scenarios Verified:
  • File size-based analysis decisions (deterministic)
  • Resource constraint handling (forced basic analysis)
  • Feature extraction with on/off control
  • Worker simplification (just do the job)
  • Docker environment compatibility
```

## 🛠️ Troubleshooting

### Common Issues

1. **Docker not available**
   ```
   ❌ [ERROR] Docker is not installed or not in PATH
   ```
   **Solution**: Install Docker Desktop or Docker Engine

2. **Insufficient resources**
   ```
   ❌ [ERROR] Failed to build Docker image
   ```
   **Solution**: Ensure at least 4GB RAM and 2GB disk space available

3. **Permission denied**
   ```
   ❌ [ERROR] Permission denied
   ```
   **Solution**: Run with appropriate permissions or use `sudo` (Linux/macOS)

4. **Test failures**
   ```
   ❌ [ERROR] Some tests failed
   ```
   **Solution**: Check the test output for specific failure details

### Debug Mode

To run tests with verbose output:

```bash
# Linux/macOS
./run_docker_tests.sh

# Windows PowerShell
.\run_docker_tests.ps1 -Verbose
```

### Manual Debugging

1. **Check Docker status**:
   ```bash
   docker --version
   docker ps
   ```

2. **Check available resources**:
   ```bash
   docker system df
   docker stats
   ```

3. **Run individual test**:
   ```bash
   docker run --rm -it playlist-generator-test python -m unittest test_analysis_integration.TestAnalysisSystemIntegration.test_analysis_manager_deterministic_decisions
   ```

## 📝 Test Customization

### Adding New Test Scenarios

1. **Add test method** to `TestAnalysisSystemIntegration` class
2. **Follow naming convention**: `test_<component>_<scenario>`
3. **Use descriptive names** that explain what is being tested
4. **Include assertions** to verify expected behavior

### Modifying Test Configuration

Edit the `setUp` method in `TestAnalysisSystemIntegration` to modify test configuration:

```python
self.config = {
    'MUSIC_PATH': self.music_dir,
    'BIG_FILE_SIZE_MB': 50,  # Modify this
    'MAX_FULL_ANALYSIS_SIZE_MB': 100,  # Modify this
    # ... other settings
}
```

### Adding New Test Files

1. **Create test file**: `test_<component>_<scenario>.py`
2. **Import required modules**: Follow the pattern in existing tests
3. **Extend unittest.TestCase**: Create test class
4. **Add to test suite**: Update `run_tests()` function

## 🎯 Test Coverage

The tests cover:

- ✅ **Unit Tests**: Individual component behavior
- ✅ **Integration Tests**: Component interaction
- ✅ **Resource Tests**: Memory, CPU, disk constraints
- ✅ **Configuration Tests**: Different settings and scenarios
- ✅ **Error Tests**: Error handling and recovery
- ✅ **Performance Tests**: Resource usage and timing
- ✅ **Docker Tests**: Containerized environment compatibility

## 📚 Related Documentation

- [Analysis Manager Documentation](../docs/ANALYSIS_MANAGER.md)
- [Resource Manager Documentation](../docs/RESOURCE_MANAGER.md)
- [Audio Analyzer Documentation](../docs/AUDIO_ANALYZER.md)
- [Docker Setup Guide](../docs/DOCKER_SETUP.md)

## 🤝 Contributing

When adding new tests:

1. **Follow existing patterns**: Use the same structure and naming
2. **Add comprehensive assertions**: Test both positive and negative cases
3. **Include documentation**: Add comments explaining test purpose
4. **Update this README**: Document new test scenarios
5. **Test in Docker**: Ensure tests work in containerized environment 