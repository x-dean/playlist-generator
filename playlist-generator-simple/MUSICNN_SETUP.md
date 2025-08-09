# MusiCNN Model Setup Guide

## Quick Setup

1. **Create the models directory on your host:**
   ```bash
   mkdir -p /root/music/playlista/models
   ```

2. **Your MusiCNN models are already in place:**
   ```bash
   /root/music/playlista/models/musicnn/msd-musicnn-1.pb
   /root/music/playlista/models/musicnn/msd-musicnn-1.json
   ```

3. **Verify model paths in config:**
   The system will automatically look for models at:
   - `/app/models/musicnn/msd-musicnn-1.pb` (mounted from `/root/music/playlista/models/musicnn/`)
   - `/app/models/musicnn/msd-musicnn-1.json`

## Supported Models

### MusiCNN Models
- **MSD (Million Song Dataset) trained models** - Best for general music analysis
- **Format**: TensorFlow `.pb` files with `.json` metadata
- **Input**: 3-second audio segments at 16kHz
- **Your models**: `msd-musicnn-1.pb` and `msd-musicnn-1.json`

## Model Configuration

Edit `playlista.conf` to customize model paths:

```ini
# MusiCNN model paths (configured for your models)
MUSICNN_MODEL_PATH=/app/models/musicnn/msd-musicnn-1.pb
MUSICNN_JSON_PATH=/app/models/musicnn/msd-musicnn-1.json

# Enable/disable MusiCNN analysis
EXTRACT_MUSICNN=true
```

## Fallback Behavior

If no TensorFlow models are found, the system will:
1. Use Essentia descriptive analysis (tempo, spectral features, etc.)
2. Map acoustic features to music tags
3. Generate embeddings from MFCC and spectral features
4. Still provide comprehensive audio analysis

## Testing Model Integration

After setup, check the logs for:
```
DEBUG - MusiCNN: Found model at /app/models/musicnn_msd.pb
DEBUG - MusiCNN: Extracted 30 tags using MusiCNN model
```

If models aren't found:
```
INFO - MusiCNN: No TensorFlow models available, using descriptive analysis
```

## Directory Structure

```
/root/music/playlista/
├── models/                    # Your MusiCNN models
│   ├── musicnn_msd.pb
│   ├── musicnn_msd.json
│   └── discogs-effnet-bs64-1.pb
├── database/                  # PostgreSQL data
├── cache/                     # Analysis cache
├── logs/                      # Application logs
└── playlists/                 # Generated playlists
```

## What You'll Get

With MusiCNN models properly configured:

### Music Tags (30+ genres/moods)
- rock, pop, electronic, classical, jazz, metal, etc.
- happy, sad, energetic, peaceful, aggressive, etc.
- instrumental, vocal, acoustic, dance, etc.

### Derived Features
- **Energy**: 0-1 scale based on spectral content
- **Danceability**: Rhythm + tempo + energy
- **Valence**: Happiness/sadness from tag analysis
- **Acousticness**: Acoustic vs electronic content
- **Instrumentalness**: Vocal vs instrumental

### High-Dimensional Embeddings
- 50-dimensional feature vectors for similarity matching
- Used for playlist generation and music recommendations

The analysis will be **significantly more accurate** with real MusiCNN models compared to the descriptive fallback!
