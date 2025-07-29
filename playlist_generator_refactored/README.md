# 🎵 Playlista Refactored

A clean, modular, and well-architected version of the Playlista music playlist generator.

## 🏗️ Architecture Overview

This project follows **Domain-Driven Design (DDD)** principles with a clean architecture structure:

```
playlist_generator_refactored/
├── src/                          # Source code
│   ├── domain/                   # Core business logic
│   │   ├── entities/            # Core business entities
│   │   ├── value_objects/       # Immutable value objects
│   │   ├── services/            # Domain services
│   │   └── repositories/        # Data access interfaces
│   ├── application/             # Application layer
│   │   ├── services/           # Application services
│   │   ├── commands/           # Command handlers
│   │   ├── queries/            # Query handlers
│   │   └── dtos/               # Data transfer objects
│   ├── infrastructure/          # Infrastructure layer
│   │   ├── persistence/        # Database implementations
│   │   ├── external_apis/      # External service integrations
│   │   ├── file_system/        # File system operations
│   │   └── logging/            # Logging infrastructure
│   ├── presentation/           # Presentation layer
│   │   ├── cli/               # CLI interface
│   │   └── api/               # REST API (future)
│   └── shared/                # Shared utilities
│       ├── config/            # Configuration management
│       ├── exceptions/         # Custom exceptions
│       ├── utils/             # Utility functions
│       └── constants/         # Application constants
├── tests/                     # Comprehensive test suite
│   ├── unit/                 # Unit tests
│   ├── integration/          # Integration tests
│   └── e2e/                 # End-to-end tests
└── docs/                     # Documentation
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Docker (for containerized deployment)
- Audio analysis libraries (Essentia, TensorFlow)

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd playlist_generator_refactored
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Test the structure:**
   ```bash
   python test_new_structure.py
   ```

## 🧪 Testing

Run the test suite to verify everything works:

```bash
# Run all tests
python test_new_structure.py

# Run specific test categories
python -m pytest tests/unit/
python -m pytest tests/integration/
python -m pytest tests/e2e/
```

## 📋 Development Status

### ✅ Completed

- **Phase 1.1: Foundation & Infrastructure**
  - ✅ Configuration management system
  - ✅ Exception handling hierarchy
  - ✅ Clean directory structure
  - ✅ Basic testing framework

### 🚧 In Progress

- **Phase 1.2: Logging Infrastructure**
  - 🔄 Structured logging implementation
  - 🔄 Log level management
  - 🔄 Log aggregation

### 📅 Planned

- **Phase 2: Domain Layer**
  - Core entities (AudioFile, Playlist, AnalysisResult)
  - Value objects (FeatureSet, Metadata)
  - Domain services
  - Repository interfaces

- **Phase 3: Application Layer**
  - Application services
  - Command/Query handlers
  - DTOs and mapping

- **Phase 4: Infrastructure Layer**
  - Database implementations
  - External API integrations
  - File system operations

- **Phase 5: Presentation Layer**
  - CLI refactoring
  - API design
  - Web interface (future)

## 🎯 Key Benefits

### **Maintainability**
- Clear separation of concerns
- Reduced coupling between components
- Easy to understand and modify
- Better code organization

### **Testability**
- Isolated components for unit testing
- Mockable dependencies
- Comprehensive test coverage
- Faster test execution

### **Extensibility**
- Easy to add new playlist generation methods
- Simple to integrate new audio analysis features
- Flexible configuration management
- Plugin architecture support

### **Performance**
- Better memory management
- Optimized database queries
- Efficient caching strategies
- Parallel processing improvements

## 🔧 Configuration

The application uses a centralized configuration system:

```python
from src.shared.config import get_config

config = get_config()
print(f"Log level: {config.logging.level}")
print(f"Cache directory: {config.database.cache_dir}")
```

### Environment Variables

- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `CACHE_DIR`: Cache directory path
- `LOG_DIR`: Log directory path
- `OUTPUT_DIR`: Output directory for playlists
- `HOST_LIBRARY_PATH`: Host music library path
- `LARGE_FILE_THRESHOLD`: File size threshold in MB
- `MEMORY_AWARE`: Enable memory-aware processing
- `MIN_TRACKS_PER_GENRE`: Minimum tracks per genre for tag-based playlists
- `LASTFM_API_KEY`: Last.fm API key for metadata enrichment

## 🐛 Error Handling

The application uses a comprehensive exception hierarchy:

```python
from src.shared.exceptions import (
    PlaylistaException,
    AudioAnalysisError,
    PlaylistGenerationError,
    ConfigurationError
)

try:
    # Your code here
    pass
except AudioAnalysisError as e:
    print(f"Audio analysis failed: {e}")
    print(f"File: {e.file_path}")
    print(f"Step: {e.analysis_step}")
```

## 📊 Monitoring & Logging

- Structured logging with correlation IDs
- Performance monitoring
- Memory usage tracking
- Error aggregation and reporting

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Original Playlista project contributors
- Domain-Driven Design community
- Clean Architecture principles
- Python testing best practices 