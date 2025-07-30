# Playlist Generator Simple

A simplified, robust audio analysis and playlist generation tool.

## 🎯 Overview

This is a simplified version of the playlist generator that focuses on:
- **Simplicity**: Clear, linear code flow
- **Reliability**: Fewer abstraction layers
- **Maintainability**: Easy to understand and modify
- **Performance**: Direct operations, minimal overhead

## 📁 Project Structure

```
playlist-generator-simple/
├── src/
│   ├── core/
│   │   ├── file_discovery.py      # ✅ File discovery (COMPLETED)
│   │   ├── database.py           # ✅ Database operations (COMPLETED)
│   │   ├── audio_analyzer.py     # 🔄 Audio analysis
│   │   └── playlist_generator.py # 🔄 Playlist generation
│   ├── utils/
│   │   ├── cli.py               # 🔄 Command line interface
│   │   ├── logging.py           # 🔄 Logging setup
│   │   └── config.py            # 🔄 Configuration
│   └── main.py                  # 🔄 Entry point
├── tests/
├── requirements.txt             # ✅ Dependencies
└── README.md                   # ✅ Documentation
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Test File Discovery
```bash
python test_file_discovery.py
```

### 3. Test Database Manager
```bash
python test_database.py
```

### 4. Run Analysis (Coming Soon)
```bash
python src/main.py analyze /path/to/music
```

## 🔧 Core Components

### ✅ File Discovery (`src/core/file_discovery.py`)
- **Purpose**: Find and validate audio files
- **Features**:
  - Directory scanning with exclusion logic
  - File validation (size, extension)
  - Failed files handling
  - Change tracking
  - Statistics generation

### ✅ Logging System (`src/core/logging_setup.py`)
- **Purpose**: Production-grade logging with configurable output
- **Features**:
  - Colored console output
  - JSON/Text file logging
  - Runtime log level changes
  - Environment variable monitoring
  - Performance logging
  - Function call decorators
  - Signal-based control

### ✅ Database Manager (`src/core/database.py`)
- **Purpose**: Comprehensive database operations for playlists, analysis, and caching
- **Features**:
  - **Playlist Operations**: Save, retrieve, update, delete playlists with metadata
  - **Analysis Results**: Store and retrieve audio analysis results with change tracking
  - **Caching System**: Configurable cache with expiration and cleanup
  - **Tags & Metadata**: Store file tags and enrichment data from external APIs
  - **Failed Analysis Tracking**: Track and retry failed analysis with error messages
  - **Statistics & Reporting**: Comprehensive database statistics and activity tracking
  - **Data Management**: Cleanup old data, export databases, backup operations

### 🔄 Audio Analyzer (`src/core/audio_analyzer.py`) - Coming Soon
- **Purpose**: Extract audio features and metadata
- **Features**:
  - Essentia integration for core features
  - BPM, key, energy extraction
  - Metadata extraction
  - Quality scoring

### 🔄 Playlist Generator (`src/core/playlist_generator.py`) - Coming Soon
- **Purpose**: Generate playlists using various algorithms
- **Features**:
  - K-means clustering
  - Tag-based generation
  - Time-based generation
  - Cache-based generation

## 🎵 Supported Audio Formats

- MP3 (.mp3)
- WAV (.wav)
- FLAC (.flac)
- OGG (.ogg)
- M4A (.m4a)
- AAC (.aac)
- OPUS (.opus)

## 📊 Current Status

- ✅ **File Discovery**: Complete and tested
- ✅ **Logging System**: Complete and production-grade
- ✅ **Database Manager**: Complete and tested
- 🔄 **Audio Analysis**: Coming soon
- 🔄 **Playlist Generation**: Coming soon
- 🔄 **CLI Interface**: Coming soon

## 🧪 Testing

Run the file discovery test:
```bash
python test_file_discovery.py
```

Run the logging test:
```bash
python test_logging.py
```

Run the database test:
```bash
python test_database.py
```

## 🔄 Migration Progress

This simplified version is being created by migrating from the original working implementation:

1. ✅ **File Discovery**: Ported from `old_working_setup/app/music_analyzer/file_discovery.py`
2. ✅ **Logging System**: Production-grade implementation with configurable settings
3. ✅ **Database Manager**: Comprehensive database operations with playlist, analysis, and caching support
4. 🔄 **Audio Analysis**: Port from `old_working_setup/app/music_analyzer/feature_extractor.py`
5. 🔄 **Playlist Generation**: Port from `old_working_setup/app/playlist_generator/`

## 🎯 Benefits of This Approach

1. **Simpler**: Fewer layers, direct operations
2. **More Reliable**: Less abstraction, fewer failure points
3. **Easier to Debug**: Linear flow, clear functions
4. **Better Performance**: Less overhead, direct database access
5. **Easier to Maintain**: Clear structure, minimal dependencies

## 📝 License

This project is part of the playlist generator refactoring effort. 