-- ============================================================================= 
-- PLAYLISTA POSTGRESQL SCHEMA - WEB UI OPTIMIZED
-- =============================================================================
-- Modern schema designed for:
-- - Fast playlist generation
-- - Music similarity/recommendations  
-- - Web application performance
-- - Concurrent user access
-- - Rich music analysis data
-- =============================================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";    -- For fuzzy text search
CREATE EXTENSION IF NOT EXISTS "vector";     -- For embeddings similarity

-- =============================================================================
-- CORE MUSIC TABLES
-- =============================================================================

-- Main tracks table (optimized for fast queries)
CREATE TABLE tracks (
    id SERIAL PRIMARY KEY,
    uuid UUID DEFAULT uuid_generate_v4() UNIQUE NOT NULL,
    
    -- File identification
    file_path TEXT UNIQUE NOT NULL,
    file_hash VARCHAR(64) NOT NULL,
    filename VARCHAR(500) NOT NULL,
    file_size_bytes BIGINT NOT NULL,
    
    -- Basic metadata (from mutagen)
    title VARCHAR(500),
    artist VARCHAR(500),
    album VARCHAR(500),
    genre JSONB,                    -- Array of genres
    year INTEGER,
    track_number INTEGER,
    disc_number INTEGER,
    duration_seconds REAL,
    bitrate INTEGER,
    sample_rate INTEGER,
    channels INTEGER,
    
    -- Key analysis features (indexed for fast playlist queries)
    tempo REAL,
    key VARCHAR(10),
    mode VARCHAR(20),
    key_confidence REAL,
    
    -- Derived features for playlist generation
    energy REAL,                    -- 0-1 scale
    danceability REAL,             -- 0-1 scale  
    valence REAL,                  -- 0-1 (sad to happy)
    acousticness REAL,             -- 0-1 scale
    instrumentalness REAL,         -- 0-1 scale
    liveness REAL,                 -- 0-1 scale
    speechiness REAL,              -- 0-1 scale
    loudness REAL,                 -- dB
    
    -- Analysis completion tracking
    analysis_completed BOOLEAN DEFAULT FALSE,
    analysis_date TIMESTAMPTZ,
    analysis_method VARCHAR(50),   -- 'optimized', 'standard'
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Full analysis data storage
CREATE TABLE track_analysis (
    id SERIAL PRIMARY KEY,
    track_id INTEGER UNIQUE REFERENCES tracks(id) ON DELETE CASCADE,
    
    -- Essentia features (complete data)
    essentia_rhythm JSONB,         -- BPM, beats, tempo confidence
    essentia_spectral JSONB,       -- Spectral centroid, rolloff, etc.
    essentia_harmonic JSONB,       -- Key detection, chroma features
    essentia_mfcc JSONB,          -- MFCC coefficients
    
    -- MusiCNN results
    musicnn_tags JSONB,           -- Genre/mood predictions with confidence
    musicnn_embeddings vector(50), -- High-dimensional feature vector
    musicnn_confidence REAL,
    
    -- Processing metadata
    segments_analyzed INTEGER,
    segment_times JSONB,          -- Array of [start, end] times
    processing_time_seconds REAL,
    cache_key VARCHAR(64),        -- For linking to JSON cache files
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- =============================================================================
-- USER MANAGEMENT
-- =============================================================================

CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    uuid UUID DEFAULT uuid_generate_v4() UNIQUE NOT NULL,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE,
    password_hash VARCHAR(255),   -- For future authentication
    preferences JSONB,            -- User music preferences
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_login TIMESTAMPTZ
);

-- =============================================================================  
-- PLAYLIST SYSTEM
-- =============================================================================

CREATE TABLE playlists (
    id SERIAL PRIMARY KEY,
    uuid UUID DEFAULT uuid_generate_v4() UNIQUE NOT NULL,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    
    name VARCHAR(255) NOT NULL,
    description TEXT,
    
    -- Generation method tracking
    generation_method VARCHAR(50), -- 'manual', 'similarity', 'mood', 'genre', 'tempo'
    generation_params JSONB,       -- Parameters used for generation
    seed_track_id INTEGER REFERENCES tracks(id), -- Track used as similarity seed
    
    -- Playlist metadata
    track_count INTEGER DEFAULT 0,
    total_duration_seconds REAL DEFAULT 0,
    is_public BOOLEAN DEFAULT FALSE,
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE playlist_tracks (
    playlist_id INTEGER REFERENCES playlists(id) ON DELETE CASCADE,
    track_id INTEGER REFERENCES tracks(id) ON DELETE CASCADE,
    position INTEGER NOT NULL,
    added_at TIMESTAMPTZ DEFAULT NOW(),
    added_by_user_id INTEGER REFERENCES users(id),
    
    PRIMARY KEY (playlist_id, track_id),
    UNIQUE (playlist_id, position)
);

-- =============================================================================
-- MUSIC DISCOVERY & RECOMMENDATIONS  
-- =============================================================================

-- Track similarity cache (pre-computed for performance)
CREATE TABLE track_similarities (
    track_id_1 INTEGER REFERENCES tracks(id) ON DELETE CASCADE,
    track_id_2 INTEGER REFERENCES tracks(id) ON DELETE CASCADE,
    similarity_score REAL NOT NULL, -- 0-1 scale
    similarity_method VARCHAR(50),   -- 'embedding', 'features', 'tags'
    computed_at TIMESTAMPTZ DEFAULT NOW(),
    
    PRIMARY KEY (track_id_1, track_id_2),
    CHECK (track_id_1 < track_id_2)  -- Prevent duplicates
);

-- Music genre/mood tags (normalized)
CREATE TABLE music_tags (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL,
    category VARCHAR(50),           -- 'genre', 'mood', 'instrument', 'style'
    description TEXT
);

CREATE TABLE track_tags (
    track_id INTEGER REFERENCES tracks(id) ON DELETE CASCADE,
    tag_id INTEGER REFERENCES music_tags(id) ON DELETE CASCADE,
    confidence REAL NOT NULL,       -- 0-1 from MusiCNN
    source VARCHAR(50),             -- 'musicnn', 'manual', 'lastfm'
    
    PRIMARY KEY (track_id, tag_id)
);

-- =============================================================================
-- PERFORMANCE INDEXES
-- =============================================================================

-- Core search indexes
CREATE INDEX idx_tracks_artist ON tracks USING gin(artist gin_trgm_ops);
CREATE INDEX idx_tracks_title ON tracks USING gin(title gin_trgm_ops);
CREATE INDEX idx_tracks_album ON tracks USING gin(album gin_trgm_ops);
CREATE INDEX idx_tracks_genre ON tracks USING gin(genre);

-- Analysis feature indexes (for playlist generation)
CREATE INDEX idx_tracks_tempo ON tracks(tempo) WHERE tempo IS NOT NULL;
CREATE INDEX idx_tracks_key ON tracks(key) WHERE key IS NOT NULL;
CREATE INDEX idx_tracks_energy ON tracks(energy) WHERE energy IS NOT NULL;
CREATE INDEX idx_tracks_valence ON tracks(valence) WHERE valence IS NOT NULL;
CREATE INDEX idx_tracks_danceability ON tracks(danceability) WHERE danceability IS NOT NULL;

-- Compound indexes for common playlist queries
CREATE INDEX idx_tracks_tempo_energy ON tracks(tempo, energy) WHERE tempo IS NOT NULL AND energy IS NOT NULL;
CREATE INDEX idx_tracks_key_mode ON tracks(key, mode) WHERE key IS NOT NULL AND mode IS NOT NULL;

-- Vector similarity index (for embeddings)
CREATE INDEX idx_analysis_embeddings ON track_analysis USING ivfflat (musicnn_embeddings vector_cosine_ops);

-- Playlist performance indexes
CREATE INDEX idx_playlist_tracks_playlist ON playlist_tracks(playlist_id, position);
CREATE INDEX idx_playlist_tracks_track ON playlist_tracks(track_id);
CREATE INDEX idx_playlists_user ON playlists(user_id);
CREATE INDEX idx_playlists_method ON playlists(generation_method);

-- Similarity cache indexes
CREATE INDEX idx_similarities_track1 ON track_similarities(track_id_1, similarity_score DESC);
CREATE INDEX idx_similarities_track2 ON track_similarities(track_id_2, similarity_score DESC);

-- =============================================================================
-- HELPFUL VIEWS FOR WEB API
-- =============================================================================

-- Complete track info with analysis
CREATE VIEW tracks_with_analysis AS
SELECT 
    t.*,
    ta.musicnn_tags,
    ta.musicnn_confidence,
    ta.segments_analyzed,
    ta.processing_time_seconds
FROM tracks t
LEFT JOIN track_analysis ta ON t.id = ta.track_id;

-- Playlist summary view
CREATE VIEW playlist_summary AS
SELECT 
    p.*,
    COUNT(pt.track_id) as actual_track_count,
    SUM(t.duration_seconds) as actual_duration,
    AVG(t.tempo) as avg_tempo,
    AVG(t.energy) as avg_energy,
    AVG(t.valence) as avg_valence
FROM playlists p
LEFT JOIN playlist_tracks pt ON p.id = pt.playlist_id
LEFT JOIN tracks t ON pt.track_id = t.id
GROUP BY p.id;

-- =============================================================================
-- FUNCTIONS FOR COMMON OPERATIONS
-- =============================================================================

-- Update playlist metadata when tracks change
CREATE OR REPLACE FUNCTION update_playlist_metadata()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE playlists SET
        track_count = (
            SELECT COUNT(*) FROM playlist_tracks WHERE playlist_id = NEW.playlist_id
        ),
        total_duration_seconds = (
            SELECT COALESCE(SUM(t.duration_seconds), 0)
            FROM playlist_tracks pt
            JOIN tracks t ON pt.track_id = t.id
            WHERE pt.playlist_id = NEW.playlist_id
        ),
        updated_at = NOW()
    WHERE id = NEW.playlist_id;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER playlist_metadata_trigger
    AFTER INSERT OR DELETE ON playlist_tracks
    FOR EACH ROW EXECUTE FUNCTION update_playlist_metadata();

-- =============================================================================
-- INITIAL DATA
-- =============================================================================

-- Insert common music tags from MusiCNN
INSERT INTO music_tags (name, category) VALUES
-- Genres
('rock', 'genre'), ('pop', 'genre'), ('electronic', 'genre'), ('jazz', 'genre'),
('classical', 'genre'), ('hip-hop', 'genre'), ('country', 'genre'), ('folk', 'genre'),
('metal', 'genre'), ('blues', 'genre'), ('reggae', 'genre'), ('punk', 'genre'),
('indie', 'genre'), ('alternative', 'genre'), ('dance', 'genre'),

-- Moods
('happy', 'mood'), ('sad', 'mood'), ('energetic', 'mood'), ('calm', 'mood'),
('aggressive', 'mood'), ('peaceful', 'mood'), ('dark', 'mood'), ('uplifting', 'mood'),
('emotional', 'mood'), ('romantic', 'mood'), ('melancholic', 'mood'),

-- Styles
('acoustic', 'style'), ('instrumental', 'style'), ('vocal', 'style'),
('ambient', 'style'), ('atmospheric', 'style'), ('experimental', 'style'),
('melodic', 'style'), ('rhythmic', 'style'),

-- Instruments/Vocals
('guitar', 'instrument'), ('piano', 'instrument'), ('drums', 'instrument'),
('bass', 'instrument'), ('violin', 'instrument'), ('saxophone', 'instrument'),
('female-vocals', 'vocal'), ('male-vocals', 'vocal'), ('choir', 'vocal');

-- Create default admin user
INSERT INTO users (username, email) VALUES ('admin', 'admin@playlista.local');

COMMIT;
