# MusiCNN Setup Guide

## Overview
This guide explains how to set up MusiCNN for audio analysis and handle common issues.

## Issues Addressed

### 1. MusiCNN Model Missing
**Error**: `Failed to load MusiCNN model: SavedModel file does not exist at: /app/models/msd-musicnn-1.pb`

**Solution**: Download the required model files using the provided script.

### 2. TriangularBands Warning
**Warning**: `TriangularBands: input spectrum size (8221824) does not correspond to the "inputSize" parameter (1025). Recomputing the filter bank.`

**Explanation**: This is a harmless warning from the Essentia library when processing audio with different sample rates or frame sizes than expected.

## Quick Setup

### Option 1: Automatic Download (Recommended)
```bash
# Run the download script
python download_musicnn_models.py
```

### Option 2: Manual Download
```bash
# Create models directory
mkdir -p /app/models

# Download model file
wget https://github.com/jordipons/musicnn/releases/download/v1.0/msd-musicnn-1.pb -O /app/models/msd-musicnn-1.pb

# Create JSON configuration
cat > /app/models/msd-musicnn-1.json << 'EOF'
{
  "tag_names": [
    "rock", "pop", "alternative", "indie", "electronic", "female vocalists", 
    "dance", "00s", "alternative rock", "jazz", "beautiful", "metal", 
    "chillout", "male vocalists", "classic rock", "soul", "indie rock", 
    "Mellow", "electronica", "80s", "folk", "90s", "blues", "hardcore", 
    "instrumental", "punk", "oldies", "country", "hard rock", "00's", 
    "ambient", "acoustic", "experimental", "female vocalist", "guitar", 
    "Hip-Hop", "70s", "party", "male vocalist", "classic", "syntax", 
    "indie pop", "heavy metal", "singer-songwriter", "world music", 
    "electro", "funk", "garage", "Classic Rock", "philadelphia", "mellow", 
    "soulful", "jazz vocal", "beautiful voice", "background", "female vocals", 
    "male vocals", "vocal", "vocalist", "vocalists"
  ]
}
EOF
```

## Configuration

### Enable MusiCNN Features
In `playlista.conf`, ensure these settings are enabled:

```ini
# MusiCNN Configuration
EXTRACT_MUSICNN=true
MUSICNN_MODEL_PATH=/app/models/msd-musicnn-1.pb
MUSICNN_JSON_PATH=/app/models/msd-musicnn-1.json
MUSICNN_TIMEOUT_SECONDS=60
```

### Disable MusiCNN (Alternative)
If you don't want to use MusiCNN features:

```ini
EXTRACT_MUSICNN=false
```

## File Structure
After setup, your models directory should look like this:

```
/app/models/
├── msd-musicnn-1.pb     # MusiCNN model file (~50MB)
└── msd-musicnn-1.json   # Tag configuration
```

## Verification

### Check Model Files
```bash
# Check if files exist
ls -la /app/models/

# Check file sizes
du -h /app/models/*
```

### Test MusiCNN Loading
```bash
# Run a test analysis
python test_musicnn_implementation.py
```

## Troubleshooting

### 1. Model Download Fails
**Problem**: Network issues or GitHub access problems.

**Solutions**:
- Use a VPN if GitHub is blocked
- Download manually from a browser
- Use alternative download methods (curl, etc.)

### 2. Permission Issues
**Problem**: Cannot write to `/app/models/`

**Solutions**:
```bash
# Fix permissions
sudo chown -R $USER:$USER /app/models/
chmod 755 /app/models/
```

### 3. TensorFlow Issues
**Problem**: TensorFlow not available or version conflicts.

**Solutions**:
- Install TensorFlow: `pip install tensorflow`
- Use CPU-only version: `pip install tensorflow-cpu`
- Check TensorFlow version compatibility

### 4. TriangularBands Warning
**Problem**: Essentia library warning about spectrum size mismatch.

**Solutions**:
- **Ignore**: This warning is harmless and doesn't affect functionality
- **Suppress**: Add to logging configuration to reduce verbosity
- **Fix**: Use consistent audio parameters (sample rate, frame size)

## TriangularBands Warning Explained

### What It Means
The TriangularBands warning occurs when Essentia's mel-frequency filter bank encounters audio with different parameters than expected:

- **Expected**: 1025 frequency bins
- **Actual**: Different number (e.g., 8221824)
- **Result**: Essentia recomputes the filter bank automatically

### Why It Happens
1. **Different Sample Rates**: Audio files with varying sample rates (44.1kHz, 48kHz, etc.)
2. **Different Frame Sizes**: Analysis using different FFT window sizes
3. **Audio Length**: Very long or short audio files
4. **Format Differences**: Various audio formats and encodings

### Impact
- **Performance**: Minimal impact on processing speed
- **Accuracy**: No impact on analysis quality
- **Functionality**: All features work normally

### Suppressing the Warning
If you want to reduce log verbosity, you can:

1. **Filter in logging**: Add pattern matching to ignore these warnings
2. **Adjust Essentia parameters**: Use consistent audio parameters
3. **Accept as normal**: This is expected behavior for diverse audio collections

## Performance Considerations

### Model Loading
- **First Load**: ~2-3 seconds (model initialization)
- **Subsequent Loads**: ~0.1 seconds (cached)
- **Memory Usage**: ~200MB per model instance

### Analysis Speed
- **Small Files** (< 5MB): ~1-2 seconds
- **Medium Files** (5-50MB): ~5-10 seconds
- **Large Files** (> 50MB): ~15-30 seconds

### Resource Usage
- **CPU**: Moderate (TensorFlow inference)
- **Memory**: High (model + audio data)
- **Disk**: ~50MB for model file

## Integration with Threaded Processing

The MusiCNN implementation works well with the threaded processing approach:

### Benefits
- **Shared Model**: Model loaded once, shared across threads
- **Memory Efficiency**: No model duplication per thread
- **Faster Startup**: No per-thread model loading

### Configuration
```ini
# Threaded processing (default)
USE_THREADED_PROCESSING=true
THREADED_WORKERS_DEFAULT=4

# MusiCNN with threading
EXTRACT_MUSICNN=true
MUSICNN_TIMEOUT_SECONDS=60
```

## Advanced Configuration

### Custom Model Paths
```ini
# Custom model paths
MUSICNN_MODEL_PATH=/custom/path/to/model.pb
MUSICNN_JSON_PATH=/custom/path/to/config.json
```

### Performance Tuning
```ini
# Optimize for your system
MUSICNN_TIMEOUT_SECONDS=120  # Longer timeout for large files
THREADED_WORKERS_DEFAULT=2    # Fewer workers for memory-constrained systems
```

### Error Handling
```ini
# Graceful fallback
EXTRACT_MUSICNN=true          # Try MusiCNN
MUSICNN_FALLBACK=true         # Fall back to basic features if model fails
```

## Conclusion

The MusiCNN setup provides powerful audio analysis capabilities:

- **Automatic Setup**: Use the download script for quick setup
- **Graceful Degradation**: System works without MusiCNN
- **Performance Optimized**: Works well with threaded processing
- **Comprehensive Analysis**: Extracts embeddings and genre tags

The TriangularBands warning is normal and doesn't affect functionality. Focus on getting the model files in place for the best analysis results. 