# MusiCNN Model Setup Guide

This guide explains how to set up MusiCNN model files for advanced audio analysis features.

## What is MusiCNN?

MusiCNN is a deep learning model that provides **200-dimensional embeddings** for music similarity matching, genre classification, and advanced audio analysis. This is what makes the playlist generator special and different from basic audio analyzers.

## Required Model Files

You need to provide two model files:

1. **`msd-musicnn-1.pb`** - TensorFlow model file (~50MB)
2. **`msd-musicnn-1.json`** - Model configuration file (~1KB)

## Setup Instructions

### 1. Download Model Files

Download the MusiCNN model files from the official repository or use pre-trained models:

```bash
# Create model directory
mkdir -p /path/to/your/models/musicnn

# Download model files (example URLs - replace with actual sources)
wget https://example.com/msd-musicnn-1.pb -O /path/to/your/models/musicnn/msd-musicnn-1.pb
wget https://example.com/msd-musicnn-1.json -O /path/to/your/models/musicnn/msd-musicnn-1.json
```

### 2. Mount Model Files in Docker

When running the Docker container, mount your model files:

```bash
docker run -v /path/to/your/models:/app/feature_extraction/models playlista
```

Or in docker-compose:

```yaml
version: '3.8'
services:
  playlista:
    image: playlista
    volumes:
      - /path/to/your/models:/app/feature_extraction/models:ro
      - /path/to/your/music:/music:ro
      - ./cache:/app/cache
```

### 3. Expected Directory Structure

The container expects this structure:

```
/app/feature_extraction/models/
├── msd-musicnn-1.pb
└── musicnn/
    └── msd-musicnn-1.json
```

## Model File Sources

### Option 1: Official MusiCNN Repository
- **Repository**: https://github.com/jordipons/musicnn
- **Model**: Pre-trained on Million Song Dataset
- **License**: MIT License

### Option 2: Pre-trained Models
- **Essentia Models**: https://essentia.upf.edu/models/
- **TensorFlow Hub**: https://tfhub.dev/
- **Hugging Face**: https://huggingface.co/models

### Option 3: Build from Source
```bash
git clone https://github.com/jordipons/musicnn.git
cd musicnn
python setup.py install
# Follow instructions to export model
```

## Verification

To verify the model files are correctly mounted:

```bash
# Check if files exist in container
docker exec playlista ls -la /app/feature_extraction/models/

# Expected output:
# -rw-r--r-- 1 root root 50M msd-musicnn-1.pb
# drwxr-xr-x 2 root root 4.0K musicnn/
# -rw-r--r-- 1 root root 1.2K musicnn/msd-musicnn-1.json
```

## Features Enabled

With MusiCNN models, you get:

### 🎵 Advanced Audio Features
- **200-dimensional embeddings** for similarity matching
- **Deep learning-based genre classification**
- **Advanced mood and emotion detection**
- **State-of-the-art music similarity**

### 📊 Enhanced Analysis
```python
# Example MusiCNN features
{
    'musicnn_features': [0.1, 0.2, 0.3, ...],  # 200 dimensions
    'bpm': 120.5,
    'loudness': -12.5,
    'key': 'C',
    'spectral_centroid': 2500.5,
    # ... other features
}
```

### 🎯 Better Playlist Generation
- **Similarity-based playlists** using deep learning embeddings
- **Genre-aware clustering** with advanced classification
- **Mood-based playlists** with emotional feature extraction
- **Research-grade accuracy** for music analysis

## Troubleshooting

### Model Not Found
```
⚠️ MusiCNN model files not found:
   Model: /app/feature_extraction/models/msd-musicnn-1.pb
   Config: /app/feature_extraction/models/musicnn/msd-musicnn-1.json
```

**Solution**: Check your mount paths and file permissions.

### TensorFlow Not Available
```
⚠️ TensorFlow not available - MusiCNN features will be limited
```

**Solution**: Ensure TensorFlow is installed in the container.

### Memory Issues
```
⚠️ Could not initialize MusiCNN: Memory allocation failed
```

**Solution**: Increase Docker memory limits or use GPU acceleration.

## Performance Notes

- **Model Loading**: ~2-3 seconds on first use
- **Memory Usage**: ~200MB additional RAM
- **Processing Speed**: ~2-3x slower than basic features
- **Accuracy**: Significantly better than basic features

## Fallback Behavior

If MusiCNN models are not available, the system gracefully falls back to basic features:

- ✅ BPM detection
- ✅ Spectral features  
- ✅ Loudness analysis
- ✅ Key detection
- ✅ MFCC coefficients
- ❌ Advanced similarity matching
- ❌ Deep learning features

## License

MusiCNN models are typically released under MIT License. Please check the specific model's license terms. 