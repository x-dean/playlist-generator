# Playlista v2 - High-Performance Music Analysis and Playlist Generation

A modern, high-performance music analysis and playlist generation system with a web interface.

## Features

- **10x Performance**: Advanced async processing and GPU acceleration
- **Enhanced Analysis**: ML-powered feature extraction with 50+ genres
- **Intelligent Playlists**: Neural similarity metrics and harmonic mixing
- **Modern Web UI**: Real-time interface with professional visualizations
- **Large Libraries**: Support for 100,000+ tracks with smooth performance

## Architecture

### Backend
- **FastAPI**: High-performance async API
- **PostgreSQL**: Optimized database with advanced indexing
- **Redis**: In-memory caching for features and results
- **PyTorch**: ML models for audio classification
- **Librosa + Essentia**: Advanced audio analysis

### Frontend
- **React 18**: Modern UI with concurrent features
- **TypeScript**: Type-safe development
- **Vite**: Lightning-fast build tool
- **Mantine**: Performance-focused UI components
- **D3.js**: High-performance data visualizations

## Project Structure

```
playlista-v2/
├── backend/                 # FastAPI backend
│   ├── app/
│   │   ├── api/            # API routes
│   │   ├── core/           # Core business logic
│   │   ├── analysis/       # Audio analysis engine
│   │   ├── playlist/       # Playlist generation engine
│   │   ├── database/       # Database models and operations
│   │   └── utils/          # Utilities and helpers
│   ├── requirements.txt    # Python dependencies
│   └── Dockerfile         # Backend container
├── frontend/               # React frontend
│   ├── src/
│   │   ├── components/     # React components
│   │   ├── pages/          # Page components
│   │   ├── stores/         # State management
│   │   ├── api/            # API client
│   │   └── utils/          # Frontend utilities
│   ├── package.json        # Node dependencies
│   └── Dockerfile         # Frontend container
├── database/               # Database schemas and migrations
├── docker-compose.yml      # Development environment
└── docs/                   # Documentation
```

## Development

```bash
# Start development environment
docker-compose up --build

# Backend development
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload

# Frontend development
cd frontend
npm install
npm run dev
```

## Performance Targets

- **Analysis**: 50+ tracks per minute
- **Memory**: <4GB for 10,000 track library
- **Startup**: <10 seconds for full system
- **UI Response**: <100ms for all interactions
- **Playlist Generation**: <5 seconds for 50-track playlist

## License

MIT License