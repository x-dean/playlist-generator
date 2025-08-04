# MusiCNN Docker Setup Guide

## Overview
This guide explains how to set up MusiCNN model files for the Docker environment where the files are hosted on the Linux host and mapped into the container.

## Host File Structure
Based on your setup, the MusiCNN model files are located on the Linux host at:
```
~/music/playlista/models/musicnn/
├── msd-musicnn-1.pb     # MusiCNN model file
└── msd-musicnn-1.json   # Tag configuration
```

## Docker Compose Mapping
The Docker compose file maps the host directory to the container:
```yaml
volumes:
  # Models directory (for feature extraction models)
  - /root/music/playlista/models/musicnn:/app/models:ro
```

This means:
- **Host path**: `/root/music/playlista/models/musicnn/`
- **Container path**: `/app/models/`
- **Files in container**: `/app/models/msd-musicnn-1.pb` and `/app/models/msd-musicnn-1.json`

## Configuration
The application is configured to look for the files at the correct container paths:

```ini
# In playlista.conf
EXTRACT_MUSICNN=true
MUSICNN_MODEL_PATH=/app/models/msd-musicnn-1.pb
MUSICNN_JSON_PATH=/app/models/msd-musicnn-1.json
MUSICNN_TIMEOUT_SECONDS=60
```

## Verification Steps

### 1. Check Host Files
On the Linux host, verify the files exist:
```bash
ls -la ~/music/playlista/models/musicnn/
```

Expected output:
```
msd-musicnn-1.json  msd-musicnn-1.pb
```

### 2. Test Inside Container
Run the test script inside the Docker container:
```bash
# Start the container
docker-compose up -d

# Execute the test script
docker exec playlist-api python /app/test_musicnn_paths.py
```

### 3. Check Logs
Monitor the application logs for MusiCNN loading:
```bash
docker logs playlist-api | grep -i musicnn
```

## Troubleshooting

### Issue 1: Files Not Found in Container
**Problem**: The error `Failed to load MusiCNN model: SavedModel file does not exist at: /app/models/msd-musicnn-1.pb`

**Solutions**:
1. **Verify host files exist**:
   ```bash
   ls -la ~/music/playlista/models/musicnn/
   ```

2. **Check Docker volume mapping**:
   ```bash
   docker inspect playlist-api | grep -A 10 "Mounts"
   ```

3. **Test file access in container**:
   ```bash
   docker exec playlist-api ls -la /app/models/
   ```

### Issue 2: Permission Problems
**Problem**: Files exist but cannot be read

**Solutions**:
1. **Check file permissions on host**:
   ```bash
   ls -la ~/music/playlista/models/musicnn/
   ```

2. **Fix permissions if needed**:
   ```bash
   chmod 644 ~/music/playlista/models/musicnn/*
   ```

### Issue 3: TensorFlow Loading Issues
**Problem**: Files found but TensorFlow cannot load them

**Solutions**:
1. **Check file format**: Ensure the `.pb` file is a valid SavedModel
2. **Check TensorFlow version**: Ensure compatibility
3. **Test model loading manually**:
   ```bash
   docker exec -it playlist-api python
   ```
   ```python
   import tensorflow as tf
   model = tf.saved_model.load('/app/models/msd-musicnn-1.pb')
   ```

## Testing

### Run Path Verification
```bash
# Test from host
python test_musicnn_paths.py

# Test inside container
docker exec playlist-api python /app/test_musicnn_paths.py
```

### Run Analysis Test
```bash
# Test with a small audio file
docker exec playlist-api python -c "
from src.core.audio_analyzer import AudioAnalyzer
analyzer = AudioAnalyzer()
print('AudioAnalyzer initialized successfully')
"
```

## Expected Behavior

### When Files Are Correctly Mapped
- ✅ Model file found at `/app/models/msd-musicnn-1.pb`
- ✅ JSON config found at `/app/models/msd-musicnn-1.json`
- ✅ TensorFlow loads the model successfully
- ✅ MusiCNN features are extracted during analysis

### When Files Are Missing
- ⚠️ Warning: `MusiCNN model not found: /app/models/msd-musicnn-1.pb`
- ⚠️ MusiCNN features are disabled
- ✅ Other analysis features continue to work normally

## Debugging Commands

### Check Container File System
```bash
# List files in container
docker exec playlist-api ls -la /app/models/

# Check file sizes
docker exec playlist-api du -h /app/models/*

# Test file readability
docker exec playlist-api cat /app/models/msd-musicnn-1.json
```

### Check Application Logs
```bash
# Follow logs in real-time
docker logs -f playlist-api

# Filter for MusiCNN messages
docker logs playlist-api | grep -i musicnn

# Check for errors
docker logs playlist-api | grep -i error
```

### Test Model Loading
```bash
# Interactive Python session in container
docker exec -it playlist-api python

# Then run:
import tensorflow as tf
import os
model_path = '/app/models/msd-musicnn-1.pb'
print(f"File exists: {os.path.exists(model_path)}")
if os.path.exists(model_path):
    model = tf.saved_model.load(model_path)
    print("Model loaded successfully")
```

## Configuration Options

### Enable MusiCNN (Default)
```ini
EXTRACT_MUSICNN=true
MUSICNN_MODEL_PATH=/app/models/msd-musicnn-1.pb
MUSICNN_JSON_PATH=/app/models/msd-musicnn-1.json
```

### Disable MusiCNN (Fallback)
```ini
EXTRACT_MUSICNN=false
```

## Performance Notes

- **Model Loading**: ~2-3 seconds on first load
- **Memory Usage**: ~200MB per model instance
- **Analysis Time**: +5-10 seconds per file with MusiCNN
- **Threading**: Model is shared across threads for efficiency

## Conclusion

The setup should work correctly with the Docker volume mapping. The key points are:

1. **Host files exist** at `~/music/playlista/models/musicnn/`
2. **Docker mapping** is correct in `docker-compose.yml`
3. **Application paths** are configured correctly in `playlista.conf`
4. **File permissions** allow reading in the container

If issues persist, use the debugging commands above to identify the specific problem. 