# 🐳 Docker Production Setup

## 🚀 Quick Start

The production Docker setup is now at the root level for easy access and building.

### **Build the Docker Image**
```bash
cd playlist_generator_refactored
docker-compose build
```

### **Run the Container**
```bash
# Run with help
docker-compose run --rm playlista

# Run with specific arguments
docker-compose run --rm playlista --analyze --workers 4

# Run with music directory mounted
docker-compose run --rm -v /path/to/music:/music playlista --analyze
```

## 📁 Directory Structure

```
playlist_generator_refactored/
├── 📄 Dockerfile              # Production Docker configuration
├── 📄 docker-compose.yaml     # Production Docker Compose
├── 📄 requirements.txt        # All dependencies
├── 📁 music/                  # Mount point for music files
├── 📁 cache/                  # Cache directory (persistent)
├── 📁 playlists/              # Output directory (persistent)
├── 📁 logs/                   # Log files directory
└── 📁 src/                    # Application source code
```

## 🔧 Configuration

### **Environment Variables**
- `PYTHONPATH=/app/src` - Python path for imports
- `MUSIC_PATH=/music` - Music files directory
- `CACHE_DIR=/app/cache` - Cache directory
- `LOG_DIR=/app/logs` - Log files directory
- `OUTPUT_DIR=/app/playlists` - Playlist output directory
- `MEMORY_LIMIT_GB=6.0` - Memory limit for processing
- `MEMORY_AWARE=true` - Enable memory-aware processing

### **Volumes**
- `./src:/app/src:ro` - Application source code (read-only)
- `./music:/music:ro` - Music files (read-only)
- `./cache:/app/cache` - Cache directory (persistent)
- `./logs:/app/logs` - Log files (persistent)
- `./playlists:/app/playlists` - Output directory (persistent)

## 🎯 Usage Examples

### **Basic Analysis**
```bash
docker-compose run --rm playlista --analyze
```

### **Analysis with Specific Workers**
```bash
docker-compose run --rm playlista --analyze --workers 4
```

### **Generate Playlists**
```bash
docker-compose run --rm playlista --generate_only --playlist_method kmeans
```

### **Full Pipeline**
```bash
docker-compose run --rm playlista --pipeline
```

### **With Custom Music Directory**
```bash
docker-compose run --rm -v /custom/music/path:/music playlista --analyze
```

## 🛠️ Development

### **Interactive Mode**
```bash
docker-compose run --rm -it playlista bash
```

### **View Logs**
```bash
docker-compose logs playlista
```

### **Rebuild After Changes**
```bash
docker-compose build --no-cache
```

## ✅ Verification

The Docker setup has been tested and verified to work:
- ✅ Docker image builds successfully
- ✅ Container starts and runs the CLI
- ✅ All required directories are created
- ✅ Environment variables are properly set
- ✅ Volume mounts work correctly

The production Docker setup is now ready for use! 