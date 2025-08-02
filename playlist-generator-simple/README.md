# 🎵 Playlist Generator

A modern, scalable playlist generation application built with Clean Architecture principles, featuring audio analysis, metadata enrichment, and comprehensive monitoring.

## 🏗️ Architecture

This application follows **Clean Architecture** principles with clear separation of concerns:

- **Domain Layer**: Core business entities and interfaces
- **Application Layer**: Use cases, commands, and queries
- **Infrastructure Layer**: Repositories, services, and external integrations
- **API Layer**: FastAPI REST endpoints with OpenAPI documentation

## 🚀 Quick Start

### Prerequisites

- Python 3.7+
- Docker & Docker Compose
- FFmpeg (for audio processing)

### Local Development

```bash
# Clone the repository
git clone <repository-url>
cd playlist-generator-simple

# Install dependencies
pip install -r requirements.txt

# Start the application
python -m uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

### Docker Development

```bash
# Build and run with Docker Compose
docker-compose up --build

# Or build and run manually
docker build -t playlist-generator .
docker run -p 8000:8000 playlist-generator
```

## 📊 Monitoring & Observability

### Metrics Collection

- **Prometheus**: Application and system metrics
- **Custom Metrics**: Track analysis, repository operations, and use case executions

### Logging

- **Structured Logging**: JSON-formatted logs with context
- **Colored Console**: Easy-to-read development logs
- **File Rotation**: Automatic log rotation and compression

### Health Checks

- **Application Health**: `/health` endpoint with system stats
- **Docker Health**: Container health checks with curl
- **Metrics Endpoint**: `/api/v1/metrics` for Prometheus scraping

## 🐳 Docker Setup

### Services

- **playlist-generator**: Main application (port 8000)

### Commands

```bash
# Build all services
docker-compose build

# Run application
docker-compose up

# View logs
docker-compose logs -f
```

## 📚 API Documentation

### Endpoints

- **API Docs**: http://localhost:8000/docs (Swagger UI)
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health
- **Metrics**: http://localhost:8000/api/v1/metrics

### Key Endpoints

```
POST /api/v1/tracks/analyze     - Analyze audio track
POST /api/v1/tracks/import      - Import tracks from directory
POST /api/v1/playlists/generate - Generate playlist
GET  /api/v1/tracks            - List all tracks
GET  /api/v1/playlists         - List all playlists
GET  /api/v1/stats/analysis    - Get analysis statistics
```

## 🔧 Development

### Project Structure

```
playlist-generator-simple/
├── src/
│   ├── domain/           # Business entities and interfaces
│   ├── application/      # Use cases, commands, queries
│   ├── infrastructure/   # Repositories, services, config
│   └── api/             # FastAPI routes and models
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

## 🎯 Features

### Core Functionality

- ✅ **Audio Analysis**: Essentia and Librosa integration
- ✅ **Metadata Enrichment**: MusicBrainz API integration
- ✅ **Playlist Generation**: Multiple algorithms (random, k-means, similarity)
- ✅ **Data Persistence**: SQLite with repository pattern
- ✅ **REST API**: FastAPI with automatic documentation

### Quality & Monitoring

- ✅ **Structured Logging**: Colored console and file logging
- ✅ **Metrics Collection**: Prometheus integration
- ✅ **Health Monitoring**: Application and system health checks

### Architecture

- ✅ **Clean Architecture**: Clear separation of concerns
- ✅ **Dependency Injection**: Container-based DI
- ✅ **SOLID Principles**: Well-structured, maintainable code
- ✅ **Error Handling**: Domain-specific exceptions
- ✅ **Configuration Management**: Environment-based config

## 📈 Performance

### Metrics Tracked

- **Track Analysis**: Duration, confidence, format
- **Repository Operations**: CRUD operations timing
- **Use Case Execution**: Business logic performance
- **System Resources**: CPU, memory, disk usage
- **Database Metrics**: Track, playlist, and analysis counts

### Optimization

- **Caching**: Redis integration for performance
- **Async Operations**: Non-blocking API endpoints
- **Resource Monitoring**: Real-time system metrics
- **Performance Profiling**: Detailed timing analysis

## 🔒 Security

### Security Features

- **Input Validation**: Pydantic model validation
- **Error Handling**: Secure error responses
- **CORS Configuration**: Cross-origin request handling

## 🚀 Deployment

### Production Ready

- **Docker Containers**: Isolated, reproducible environments
- **Health Checks**: Application and container health monitoring
- **Logging**: Structured logs for production debugging
- **Metrics**: Prometheus integration for monitoring
- **Documentation**: Comprehensive API documentation

### Environment Variables

```bash
PYTHONPATH=/app
LOG_LEVEL=INFO
PROMETHEUS_MULTIPROC_DIR=/tmp
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Ensure code quality
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

- **Issues**: Report bugs and feature requests
- **Documentation**: API docs and architecture guides
- **Logs**: Structured logging for debugging

---

**Built with ❤️ using Clean Architecture and modern Python practices** 