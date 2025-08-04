# Docker Compose Usage Guide

## Quick Start

### Start the API Server
```bash
# Start the API server (always running)
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f
```

### Access the API
- **API Documentation**: http://localhost:8500/docs
- **Health Check**: http://localhost:8500/api/v1/health
- **Root Endpoint**: http://localhost:8500/

### Use the CLI

#### Show CLI Help
```bash
docker-compose run --rm playlist-api /app/entrypoint.sh cli --help
```

#### Run CLI Commands
```bash
# Show statistics
docker-compose run --rm playlist-api /app/entrypoint.sh cli stats

# Analyze music files
docker-compose run --rm playlist-api /app/entrypoint.sh cli analyze --music-path /music

# Generate playlists
docker-compose run --rm playlist-api /app/entrypoint.sh cli playlist --method kmeans --num-playlists 3

# Show database status
docker-compose run --rm playlist-api /app/entrypoint.sh cli status

# List playlist methods
docker-compose run --rm playlist-api /app/entrypoint.sh cli playlist-methods
```

### Stop Services
```bash
# Stop all services
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

## Directory Structure

The Docker Compose setup expects these host directories:

```
/root/music/
├── library/                    # Your music files
└── playlista/
    ├── playlists/             # Generated playlists output
    ├── cache/                 # Analysis cache and database
    ├── logs/                  # Application logs
    ├── models/
    │   └── musicnn/          # AI models (optional)
    └── config/
        └── playlista.conf     # Configuration file
```

## Setup for New Host

### 1. Create Directory Structure
```bash
# Create the required directories
sudo mkdir -p /root/music/library
sudo mkdir -p /root/music/playlista/playlists
sudo mkdir -p /root/music/playlista/cache
sudo mkdir -p /root/music/playlista/logs
sudo mkdir -p /root/music/playlista/models/musicnn
sudo mkdir -p /root/music/playlista/config

# Set proper permissions
sudo chown -R $USER:$USER /root/music
```

### 2. Add Your Music Files
```bash
# Copy your music files to the library directory
cp -r /path/to/your/music/* /root/music/library/
```

### 3. Copy Configuration File
```bash
# Copy the configuration file
cp playlista.conf /root/music/playlista/config/
```

### 4. Start the Services
```bash
# Start the API server
docker-compose up -d
```

## Configuration

### Music Directory
Place your music files in `/root/music/library/`. The container will mount this as `/music` inside the container.

### Configuration File
The `/root/music/playlista/config/playlista.conf` file is mounted into the container. Edit this file to customize settings.

## Examples

### Full Analysis Pipeline
```bash
# 1. Start the API server
docker-compose up -d

# 2. Analyze music files
docker-compose run --rm playlist-api /app/entrypoint.sh cli analyze --music-path /music --fast-mode

# 3. Generate playlists
docker-compose run --rm playlist-api /app/entrypoint.sh cli playlist --method kmeans --num-playlists 5

# 4. Show results
docker-compose run --rm playlist-api /app/entrypoint.sh cli stats
```

### Development Workflow
```bash
# Start API and access it
docker-compose up -d
curl http://localhost:8500/api/v1/health

# Run CLI commands as needed
docker-compose run --rm playlist-api /app/entrypoint.sh cli stats
docker-compose run --rm playlist-api /app/entrypoint.sh cli status

# Stop when done
docker-compose down
```

## Troubleshooting

### Check Container Status
```bash
docker-compose ps
```

### View Logs
```bash
# All services
docker-compose logs

# Follow logs
docker-compose logs -f

# Specific service
docker-compose logs playlist-api
```

### Restart Services
```bash
docker-compose restart
```

### Clean Up
```bash
# Stop and remove containers
docker-compose down

# Remove everything including volumes
docker-compose down -v

# Remove images too
docker-compose down -v --rmi all
```

### Check Directory Permissions
```bash
# Ensure directories exist and have proper permissions
ls -la /root/music/
ls -la /root/music/playlista/
```

### Verify Music Files
```bash
# Check if music files are accessible
ls -la /root/music/library/
``` 