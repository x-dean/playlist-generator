-- =============================================================================
-- COMPONENT-BASED PLAYLIST GENERATOR SCHEMA
-- =============================================================================
-- This schema separates raw extraction data into component tables
-- and optimizes main tables for fast playlist queries

-- =============================================================================
-- COMPONENT TABLES (Raw Extraction Data)
-- =============================================================================

-- File discovery table (initial file detection)
CREATE TABLE file_discovery (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path TEXT UNIQUE NOT NULL,
    file_hash TEXT NOT NULL,
    file_size_bytes INTEGER NOT NULL,
    filename TEXT NOT NULL,
    discovery_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    discovery_source TEXT DEFAULT 'file_system', -- 'file_system', 'manual', 'api'
    status TEXT DEFAULT 'discovered', -- 'discovered', 'analyzed', 'failed', 'moved'
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Mutagen metadata extraction (ID3 tags, etc.)
CREATE TABLE mutagen_metadata (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    track_id INTEGER NOT NULL,
    title TEXT,
    artist TEXT,
    album TEXT,
    track_number INTEGER,
    disc_number INTEGER,
    year INTEGER,
    genre TEXT, -- JSON array
    duration REAL,
    bitrate INTEGER,
    sample_rate INTEGER,
    channels INTEGER,
    encoded_by TEXT,
    language TEXT,
    copyright TEXT,
    publisher TEXT,
    original_artist TEXT,
    original_album TEXT,
    original_year INTEGER,
    original_filename TEXT,
    content_group TEXT,
    encoder TEXT,
    file_type TEXT,
    playlist_delay INTEGER,
    recording_time TEXT,
    tempo REAL,
    length REAL,
    replaygain_track_gain REAL,
    replaygain_album_gain REAL,
    replaygain_track_peak REAL,
    replaygain_album_peak REAL,
    lyricist TEXT,
    band TEXT,
    conductor TEXT,
    remixer TEXT,
    custom_tags TEXT, -- JSON object of custom TXXX and iTunes tags
    extraction_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (track_id) REFERENCES tracks(id) ON DELETE CASCADE
);

-- MusicNN features (embeddings, tags, confidence)
CREATE TABLE musicnn_features (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    track_id INTEGER NOT NULL,
    embedding TEXT, -- JSON array of embedding values
    tags TEXT, -- JSON object of tags with confidence
    confidence REAL,
    model_version TEXT DEFAULT 'v1',
    processing_time REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (track_id) REFERENCES tracks(id) ON DELETE CASCADE
);

-- Essentia features (rhythm, spectral, harmonic analysis)
CREATE TABLE essentia_features (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    track_id INTEGER NOT NULL,
    rhythm_features TEXT, -- JSON object with bpm, confidence, etc.
    spectral_features TEXT, -- JSON object with centroid, flatness, etc.
    mfcc_features TEXT, -- JSON object with MFCC coefficients
    harmonic_features TEXT, -- JSON object with key, scale, etc.
    bpm REAL,
    key TEXT,
    scale TEXT,
    rhythm_confidence REAL,
    key_confidence REAL,
    processing_time REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (track_id) REFERENCES tracks(id) ON DELETE CASCADE
);

-- External API metadata (MusicBrainz, Spotify, Discogs)
CREATE TABLE external_metadata (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    track_id INTEGER NOT NULL,
    musicbrainz_id TEXT,
    musicbrainz_artist_id TEXT,
    musicbrainz_album_id TEXT,
    spotify_id TEXT,
    discogs_id TEXT,
    metadata TEXT, -- JSON object with all external data
    enrichment_sources TEXT, -- JSON array of sources used
    last_updated TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (track_id) REFERENCES tracks(id) ON DELETE CASCADE
);

-- Audio analysis features (librosa, custom analysis)
CREATE TABLE audio_analysis (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    track_id INTEGER NOT NULL,
    analysis_type TEXT NOT NULL, -- 'librosa', 'custom', 'advanced'
    features TEXT, -- JSON object with all analysis features
    confidence REAL,
    processing_time REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (track_id) REFERENCES tracks(id) ON DELETE CASCADE
);

-- =============================================================================
-- MAIN TABLES (Optimized for Queries)
-- =============================================================================

-- Core tracks table (minimal, fast queries)
CREATE TABLE tracks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    
    -- File identification (references file_discovery)
    file_path TEXT UNIQUE NOT NULL,
    file_hash TEXT NOT NULL,
    filename TEXT NOT NULL,
    file_size_bytes INTEGER NOT NULL,
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Status fields
    status TEXT DEFAULT 'discovered', -- 'discovered', 'analyzed', 'failed'
    analysis_status TEXT DEFAULT 'pending', -- 'pending', 'in_progress', 'completed', 'failed'
    retry_count INTEGER DEFAULT 0,
    error_message TEXT,
    
    -- Essential metadata for fast queries (from mutagen, with external fallback)
    title TEXT,
    artist TEXT,
    album TEXT,
    genre TEXT, -- JSON array
    year INTEGER,
    duration REAL,
    
    -- Essential playlist features (indexed for fast queries)
    bpm REAL,
    key TEXT,
    mode TEXT,
    energy REAL,
    danceability REAL,
    valence REAL,
    acousticness REAL,
    instrumentalness REAL,
    speechiness REAL,
    liveness REAL,
    loudness REAL,
    
    -- Analysis completion flags
    mutagen_completed BOOLEAN DEFAULT FALSE,
    musicnn_completed BOOLEAN DEFAULT FALSE,
    essentia_completed BOOLEAN DEFAULT FALSE,
    external_completed BOOLEAN DEFAULT FALSE,
    audio_analysis_completed BOOLEAN DEFAULT FALSE
);

-- Playlists table
CREATE TABLE playlists (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    description TEXT,
    generation_method TEXT DEFAULT 'manual',
    generation_params TEXT, -- JSON object
    track_count INTEGER DEFAULT 0,
    total_duration REAL DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Playlist tracks junction table
CREATE TABLE playlist_tracks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    playlist_id INTEGER NOT NULL,
    track_id INTEGER NOT NULL,
    position INTEGER NOT NULL,
    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (playlist_id) REFERENCES playlists(id) ON DELETE CASCADE,
    FOREIGN KEY (track_id) REFERENCES tracks(id) ON DELETE CASCADE,
    UNIQUE(playlist_id, position)
);

-- =============================================================================
-- SUPPORTING TABLES
-- =============================================================================

-- Tags table (for MusicNN and external tags)
CREATE TABLE tags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    track_id INTEGER NOT NULL,
    source TEXT NOT NULL, -- 'musicnn', 'external', 'manual'
    tag_name TEXT NOT NULL,
    tag_value TEXT,
    confidence REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (track_id) REFERENCES tracks(id) ON DELETE CASCADE
);

-- Cache table (for temporary data)
CREATE TABLE cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cache_key TEXT UNIQUE NOT NULL,
    cache_value TEXT NOT NULL,
    cache_type TEXT DEFAULT 'general',
    expires_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Discovery cache table
CREATE TABLE discovery_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    directory_path TEXT NOT NULL,
    file_count INTEGER NOT NULL,
    scan_duration REAL,
    status TEXT DEFAULT 'completed',
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Analysis queue table (track processing status)
CREATE TABLE analysis_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path TEXT UNIQUE NOT NULL,
    priority INTEGER DEFAULT 0,
    status TEXT DEFAULT 'pending', -- 'pending', 'processing', 'completed', 'failed'
    assigned_worker TEXT,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Batch processing table (group related operations)
CREATE TABLE batch_processing (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id TEXT UNIQUE NOT NULL,
    batch_type TEXT NOT NULL, -- 'discovery', 'analysis', 'enrichment'
    total_files INTEGER DEFAULT 0,
    processed_files INTEGER DEFAULT 0,
    failed_files INTEGER DEFAULT 0,
    status TEXT DEFAULT 'pending', -- 'pending', 'processing', 'completed', 'failed'
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Error logging table (detailed error tracking)
CREATE TABLE error_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path TEXT,
    error_type TEXT NOT NULL, -- 'discovery', 'analysis', 'enrichment', 'system'
    error_message TEXT NOT NULL,
    error_details TEXT, -- JSON object with stack trace, etc.
    severity TEXT DEFAULT 'error', -- 'debug', 'info', 'warning', 'error', 'critical'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Performance metrics table (processing times, memory usage)
CREATE TABLE performance_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    operation_type TEXT NOT NULL, -- 'discovery', 'analysis', 'enrichment'
    operation_name TEXT NOT NULL,
    duration_ms REAL,
    memory_usage_mb REAL,
    cpu_usage_percent REAL,
    file_path TEXT,
    batch_id TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Statistics table
CREATE TABLE statistics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    category TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    metric_value REAL NOT NULL,
    metric_data TEXT, -- JSON object
    date_recorded TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- =============================================================================
-- INDEXES (Optimized for Playlist Queries)
-- =============================================================================

-- Main table indexes (fast playlist queries)
CREATE INDEX idx_tracks_file_path ON tracks(file_path);
CREATE INDEX idx_tracks_file_hash ON tracks(file_hash);
CREATE INDEX idx_tracks_artist ON tracks(artist);
CREATE INDEX idx_tracks_title ON tracks(title);
CREATE INDEX idx_tracks_album ON tracks(album);
CREATE INDEX idx_tracks_genre ON tracks(genre);
CREATE INDEX idx_tracks_year ON tracks(year);
CREATE INDEX idx_tracks_bpm ON tracks(bpm);
CREATE INDEX idx_tracks_key ON tracks(key);
CREATE INDEX idx_tracks_energy ON tracks(energy);
CREATE INDEX idx_tracks_danceability ON tracks(danceability);
CREATE INDEX idx_tracks_valence ON tracks(valence);
CREATE INDEX idx_tracks_acousticness ON tracks(acousticness);
CREATE INDEX idx_tracks_instrumentalness ON tracks(instrumentalness);
CREATE INDEX idx_tracks_status ON tracks(status);
CREATE INDEX idx_tracks_analysis_status ON tracks(analysis_status);

-- Component table indexes
CREATE INDEX idx_file_discovery_path ON file_discovery(file_path);
CREATE INDEX idx_file_discovery_hash ON file_discovery(file_hash);
CREATE INDEX idx_file_discovery_status ON file_discovery(status);
CREATE INDEX idx_mutagen_track_id ON mutagen_metadata(track_id);
CREATE INDEX idx_mutagen_artist ON mutagen_metadata(artist);
CREATE INDEX idx_mutagen_title ON mutagen_metadata(title);
CREATE INDEX idx_mutagen_album ON mutagen_metadata(album);
CREATE INDEX idx_mutagen_genre ON mutagen_metadata(genre);
CREATE INDEX idx_mutagen_year ON mutagen_metadata(year);
CREATE INDEX idx_musicnn_track_id ON musicnn_features(track_id);
CREATE INDEX idx_essentia_track_id ON essentia_features(track_id);
CREATE INDEX idx_external_track_id ON external_metadata(track_id);
CREATE INDEX idx_audio_analysis_track_id ON audio_analysis(track_id);
CREATE INDEX idx_audio_analysis_type ON audio_analysis(analysis_type);

-- Supporting table indexes
CREATE INDEX idx_tags_track_id ON tags(track_id);
CREATE INDEX idx_tags_source ON tags(source);
CREATE INDEX idx_tags_tag_name ON tags(tag_name);
CREATE INDEX idx_playlist_tracks_playlist_id ON playlist_tracks(playlist_id);
CREATE INDEX idx_playlist_tracks_track_id ON playlist_tracks(track_id);
CREATE INDEX idx_cache_key ON cache(cache_key);
CREATE INDEX idx_cache_type ON cache(cache_type);
CREATE INDEX idx_cache_expires ON cache(expires_at);
CREATE INDEX idx_statistics_category ON statistics(category);
CREATE INDEX idx_statistics_metric_name ON statistics(metric_name);

-- New table indexes
CREATE INDEX idx_analysis_queue_path ON analysis_queue(file_path);
CREATE INDEX idx_analysis_queue_status ON analysis_queue(status);
CREATE INDEX idx_analysis_queue_priority ON analysis_queue(priority);
CREATE INDEX idx_batch_processing_id ON batch_processing(batch_id);
CREATE INDEX idx_batch_processing_type ON batch_processing(batch_type);
CREATE INDEX idx_batch_processing_status ON batch_processing(status);
CREATE INDEX idx_error_logs_path ON error_logs(file_path);
CREATE INDEX idx_error_logs_type ON error_logs(error_type);
CREATE INDEX idx_error_logs_severity ON error_logs(severity);
CREATE INDEX idx_error_logs_created ON error_logs(created_at);
CREATE INDEX idx_performance_metrics_type ON performance_metrics(operation_type);
CREATE INDEX idx_performance_metrics_name ON performance_metrics(operation_name);
CREATE INDEX idx_performance_metrics_created ON performance_metrics(created_at);

-- =============================================================================
-- VIEWS (Convenient Data Access)
-- =============================================================================

-- Complete track view with all component data
CREATE VIEW track_complete AS
SELECT 
    t.*,
    fd.discovery_timestamp,
    fd.discovery_source,
    mm.title,
    mm.artist,
    mm.album,
    mm.track_number,
    mm.genre,
    mm.year,
    mm.duration,
    mm.bitrate,
    mm.sample_rate,
    mm.channels,
    mf.embedding as musicnn_embedding,
    mf.tags as musicnn_tags,
    mf.confidence as musicnn_confidence,
    ef.rhythm_features as essentia_rhythm,
    ef.spectral_features as essentia_spectral,
    ef.mfcc_features as essentia_mfcc,
    ef.harmonic_features as essentia_harmonic,
    em.metadata as external_metadata,
    em.enrichment_sources as external_sources
FROM tracks t
LEFT JOIN file_discovery fd ON t.file_path = fd.file_path
LEFT JOIN mutagen_metadata mm ON t.id = mm.track_id
LEFT JOIN musicnn_features mf ON t.id = mf.track_id
LEFT JOIN essentia_features ef ON t.id = ef.track_id
LEFT JOIN external_metadata em ON t.id = em.track_id;

-- Track summary view for web UI
CREATE VIEW track_summary AS
SELECT 
    t.id, t.file_path, t.filename, 
    t.title, t.artist, t.album, t.genre, t.year, t.duration,
    t.bpm, t.key, t.energy, t.danceability, t.valence, t.acousticness, t.instrumentalness,
    t.status, t.analysis_status, t.created_at, t.updated_at
FROM tracks t
WHERE t.status = 'analyzed';

-- Playlist features view
CREATE VIEW playlist_features AS
SELECT 
    p.id as playlist_id,
    p.name as playlist_name,
    p.track_count,
    p.total_duration,
    AVG(t.bpm) as avg_bpm,
    AVG(t.energy) as avg_energy,
    AVG(t.danceability) as avg_danceability,
    AVG(t.valence) as avg_valence,
    COUNT(DISTINCT t.key) as unique_keys,
    COUNT(DISTINCT t.artist) as unique_artists
FROM playlists p
JOIN playlist_tracks pt ON p.id = pt.playlist_id
JOIN tracks t ON pt.track_id = t.id
GROUP BY p.id, p.name, p.track_count, p.total_duration;

-- =============================================================================
-- TRIGGERS (Data Synchronization)
-- =============================================================================

-- Update track completion flags when component data is added
CREATE TRIGGER update_mutagen_completed
AFTER INSERT ON mutagen_metadata
BEGIN
    UPDATE tracks SET 
        mutagen_completed = TRUE, 
        title = COALESCE(NEW.title, title),
        artist = COALESCE(NEW.artist, artist),
        album = COALESCE(NEW.album, album),
        genre = COALESCE(NEW.genre, genre),
        year = COALESCE(NEW.year, year),
        duration = COALESCE(NEW.duration, duration),
        updated_at = CURRENT_TIMESTAMP
    WHERE id = NEW.track_id;
END;

CREATE TRIGGER update_musicnn_completed
AFTER INSERT ON musicnn_features
BEGIN
    UPDATE tracks SET musicnn_completed = TRUE, updated_at = CURRENT_TIMESTAMP
    WHERE id = NEW.track_id;
END;

CREATE TRIGGER update_essentia_completed
AFTER INSERT ON essentia_features
BEGIN
    UPDATE tracks SET essentia_completed = TRUE, updated_at = CURRENT_TIMESTAMP
    WHERE id = NEW.track_id;
END;

CREATE TRIGGER update_external_completed
AFTER INSERT ON external_metadata
BEGIN
    UPDATE tracks SET external_completed = TRUE, updated_at = CURRENT_TIMESTAMP
    WHERE id = NEW.track_id;
END;

CREATE TRIGGER update_audio_analysis_completed
AFTER INSERT ON audio_analysis
BEGIN
    UPDATE tracks SET audio_analysis_completed = TRUE, updated_at = CURRENT_TIMESTAMP
    WHERE id = NEW.track_id;
END;

-- Update playlist statistics when tracks are added/removed
CREATE TRIGGER update_playlist_stats_insert
AFTER INSERT ON playlist_tracks
BEGIN
    UPDATE playlists 
    SET track_count = (
        SELECT COUNT(*) FROM playlist_tracks WHERE playlist_id = NEW.playlist_id
    ),
    total_duration = (
        SELECT COALESCE(SUM(t.duration), 0) 
        FROM playlist_tracks pt 
        JOIN tracks t ON pt.track_id = t.id 
        WHERE pt.playlist_id = NEW.playlist_id
    ),
    updated_at = CURRENT_TIMESTAMP
    WHERE id = NEW.playlist_id;
END;

CREATE TRIGGER update_playlist_stats_delete
AFTER DELETE ON playlist_tracks
BEGIN
    UPDATE playlists 
    SET track_count = (
        SELECT COUNT(*) FROM playlist_tracks WHERE playlist_id = OLD.playlist_id
    ),
    total_duration = (
        SELECT COALESCE(SUM(t.duration), 0) 
        FROM playlist_tracks pt 
        JOIN tracks t ON pt.track_id = t.id 
        WHERE pt.playlist_id = OLD.playlist_id
    ),
    updated_at = CURRENT_TIMESTAMP
    WHERE id = OLD.playlist_id;
END; 