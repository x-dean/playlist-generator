# 🎵 Playlist Generator - Database Setup Guide

## 🚀 PostgreSQL Setup (Required - Clean Architecture)

### **Prerequisites**
- PostgreSQL 12+ installed
- Python dependencies: `pip install psycopg2-binary`
- Optional: pgvector extension for music similarity

### **Quick Setup with Docker**
```bash
# Start PostgreSQL with pgvector support
docker-compose -f docker-compose.postgresql.yml up -d postgres

# Run database setup script
python database/setup_postgresql.py

# Test connection
python database/setup_postgresql.py test
```

### **Manual PostgreSQL Setup**

#### **1. Install PostgreSQL**
```bash
# Ubuntu/Debian
sudo apt-get install postgresql postgresql-contrib

# macOS with Homebrew
brew install postgresql

# Windows - Download from postgresql.org
```

#### **2. Install pgvector (Optional - for music similarity)**
```bash
# Ubuntu/Debian
sudo apt-get install postgresql-14-pgvector

# From source
git clone https://github.com/pgvector/pgvector.git
cd pgvector
make
sudo make install
```

#### **3. Configure Database**
```sql
-- Connect as postgres user
sudo -u postgres psql

-- Create user and database
CREATE USER playlista WITH PASSWORD 'playlista_password';
CREATE DATABASE playlista OWNER playlista ENCODING 'UTF8';
GRANT ALL PRIVILEGES ON DATABASE playlista TO playlista;

-- Connect to playlista database
\c playlista

-- Install extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "vector";  -- Optional
```

#### **4. Initialize Schema**
```bash
# Run setup script
python database/setup_postgresql.py

# Or manually
psql -U playlista -d playlista -f database/postgresql_schema.sql
```

#### **5. Update Configuration**
Edit `playlista.conf`:
```ini
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=playlista
POSTGRES_USER=playlista
POSTGRES_PASSWORD=playlista_password
```

---

## 🔧 Configuration Options

### **Environment Variables**
```bash
# Database connection
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5432
export POSTGRES_DB=playlista
export POSTGRES_USER=playlista
export POSTGRES_PASSWORD=your_secure_password

# Connection pooling
export POSTGRES_MIN_CONNECTIONS=2
export POSTGRES_MAX_CONNECTIONS=10

# Force PostgreSQL usage
export ENVIRONMENT=production
```

### **Config File (`playlista.conf`)**
```ini
# PostgreSQL settings
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=playlista
POSTGRES_USER=playlista
POSTGRES_PASSWORD=playlista_password

# Connection pooling
POSTGRES_MIN_CONNECTIONS=2
POSTGRES_MAX_CONNECTIONS=10
```

---

## 📊 Database Schema Overview

### **Core Tables**
- **`tracks`** - Main track information (fast queries)
- **`track_analysis`** - Complete analysis data (Essentia + MusiCNN)
- **`music_tags`** - Normalized genre/mood tags
- **`track_tags`** - Track-to-tag relationships with confidence

### **Playlist System**
- **`users`** - User accounts
- **`playlists`** - Playlist metadata
- **`playlist_tracks`** - Track ordering in playlists

### **Performance Features**
- **Indexes** - Optimized for tempo, key, energy queries
- **Vector Search** - pgvector for music similarity
- **Full-text Search** - pg_trgm for fuzzy text search
- **JSON Storage** - Complete analysis data preserved

---

## 🎯 Usage Examples

### **Basic Analysis Storage**
```python
from src.core.db_factory import get_database_manager

db = get_database_manager()

# Save analysis results
success = db.save_track_analysis(
    file_path="/music/song.mp3",
    filename="song.mp3", 
    file_size_bytes=5242880,
    file_hash="abc123def456",
    metadata={"title": "Amazing Song", "artist": "Great Artist"},
    analysis_data={
        "tempo": 128.5,
        "key": "C",
        "scale": "major",
        "musicnn_tags": {"electronic": 0.89, "dance": 0.67},
        "musicnn_embeddings": [0.1, -0.3, 0.8, ...]
    }
)
```

### **Playlist Generation**
```python
# Find similar tracks
similar = db.find_similar_tracks(track_id=123, limit=20)

# Generate by features
playlist = db.generate_playlist_by_features(
    tempo_range=(120, 140),
    energy_range=(0.6, 1.0),
    limit=25
)

# Search tracks
results = db.search_tracks("electronic dance", limit=50)
```

### **Web API Integration**
```python
from fastapi import FastAPI
from src.core.db_factory import get_database_manager

app = FastAPI()
db = get_database_manager()

@app.get("/api/tracks/similar/{track_id}")
async def get_similar_tracks(track_id: int):
    return db.find_similar_tracks(track_id, limit=20)

@app.get("/api/playlists/generate")
async def generate_playlist(tempo_min: float = 120, tempo_max: float = 140):
    return db.generate_playlist_by_features(
        tempo_range=(tempo_min, tempo_max),
        limit=25
    )
```

---

## 🎯 Clean Architecture Benefits

### **Single Database System**
No more complexity - PostgreSQL only:

```python
# Simple, clean imports
from src.core.database import get_db_manager

db = get_db_manager()  # Always PostgreSQL - no fallbacks, no confusion
```

### **Modern Web Architecture**
- **Concurrent users** - Multiple people can use the web UI simultaneously
- **Real-time features** - Live playlist updates and recommendations
- **Scalable design** - Ready for production deployment

---

## 🛠️ Troubleshooting

### **Common Issues**

#### **Connection Failed**
```bash
# Check PostgreSQL is running
sudo systemctl status postgresql

# Check configuration
psql -U playlista -d playlista -c "SELECT version();"
```

#### **pgvector Not Available**
```bash
# Install pgvector extension
sudo apt-get install postgresql-14-pgvector

# Or disable vector features in config
# (Music similarity will use basic features instead)
```

#### **Permission Denied**
```sql
-- Connect as postgres superuser
sudo -u postgres psql

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE playlista TO playlista;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO playlista;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO playlista;
```

### **Performance Tuning**

#### **PostgreSQL Configuration**
```sql
-- Increase shared buffers for better performance
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET random_page_cost = 1.1;

-- Reload configuration
SELECT pg_reload_conf();
```

#### **Connection Pooling**
```ini
# Adjust based on your system
POSTGRES_MIN_CONNECTIONS=5
POSTGRES_MAX_CONNECTIONS=20
```

---

## 📈 Monitoring

### **Database Health Check**
```python
from src.core.db_factory import get_database_manager

db = get_database_manager()
count = db.get_track_count()
print(f"Total analyzed tracks: {count}")
```

### **Using pgAdmin**
```bash
# Start pgAdmin with Docker
docker-compose -f docker-compose.postgresql.yml --profile admin up pgadmin

# Access at http://localhost:8080
# Email: admin@playlista.local
# Password: admin
```

### **Performance Queries**
```sql
-- Check table sizes
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
FROM pg_tables 
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;

-- Check index usage
SELECT 
    indexrelname,
    idx_tup_read,
    idx_tup_fetch,
    idx_scan
FROM pg_stat_user_indexes
ORDER BY idx_scan DESC;
```

---

## ✅ Verification

After setup, verify everything works:

```bash
# Test database connection
python database/setup_postgresql.py test

# Run analysis on a few tracks
python -m src.cli.main analyse /path/to/music --max-files 5

# Check data was saved
python -c "
from src.core.db_factory import get_database_manager
db = get_database_manager()
print(f'Tracks in database: {db.get_track_count()}')
"
```

🎉 **You're ready for high-performance playlist generation with PostgreSQL!**