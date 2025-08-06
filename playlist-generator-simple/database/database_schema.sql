-- =============================================================================
-- OPTIMIZED DATABASE SCHEMA FOR WEB UI PERFORMANCE
-- =============================================================================
-- Streamlined schema with only essential tables and fields for fast web UI queries

-- Main tracks table with essential fields only
CREATE TABLE tracks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    
    -- File identification
    file_path TEXT UNIQUE NOT NULL,
    file_hash TEXT NOT NULL,
    filename TEXT NOT NULL,
    file_size_bytes INTEGER NOT NULL,
    
    -- Timestamps
    analysis_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Status fields
    status TEXT DEFAULT 'discovered', -- 'discovered', 'analyzed', 'failed'
    analysis_status TEXT DEFAULT 'pending', -- 'pending', 'in_progress', 'completed', 'failed'
    retry_count INTEGER DEFAULT 0,
    error_message TEXT,
    
    -- Core metadata (essential for web UI) - made nullable for discovery
    title TEXT DEFAULT 'Unknown',
    artist TEXT DEFAULT 'Unknown',
    album TEXT,
    track_number INTEGER,
    genre TEXT,
    year INTEGER,
    duration REAL,
    
    -- Audio properties
    bitrate INTEGER,
    sample_rate INTEGER,
    channels INTEGER,
    
    -- Essential audio features for playlist generation
    bpm REAL,
    key TEXT,
    mode TEXT,
    loudness REAL,
    energy REAL,
    danceability REAL,
    valence REAL,
    acousticness REAL,
    instrumentalness REAL,
    speechiness REAL,
    liveness REAL,
    
    -- Rhythm features (Essentia)
    rhythm_confidence REAL,
    bpm_estimates TEXT, -- JSON array
    bpm_intervals TEXT, -- JSON array
    external_bpm REAL, -- BPM from metadata
    
    -- Spectral features (Essentia)
    spectral_centroid REAL,
    spectral_flatness REAL,
    spectral_rolloff REAL,
    spectral_bandwidth REAL,
    spectral_contrast_mean REAL,
    spectral_contrast_std REAL,
    
    -- Loudness features (Essentia)
    dynamic_complexity REAL,
    loudness_range REAL,
    dynamic_range REAL,
    
    -- Key features (Essentia)
    scale TEXT, -- 'major', 'minor'
    key_strength REAL,
    key_confidence REAL,
    
    -- Additional metadata (optional)
    composer TEXT,
    mood TEXT,
    style TEXT,
    
    -- External API data
    musicbrainz_id TEXT,
    musicbrainz_artist_id TEXT,
    musicbrainz_album_id TEXT,
    discogs_id TEXT,
    spotify_id TEXT,
    release_date TEXT,
    disc_number INTEGER,
    duration_ms INTEGER,
    play_count INTEGER,
    listeners INTEGER,
    rating REAL,
    popularity REAL,
    url TEXT,
    image_url TEXT,
    enrichment_sources TEXT, -- JSON array of sources
    
    -- Mutagen-specific metadata
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
    playlist_delay TEXT,
    recording_time TEXT,
    tempo REAL,
    length REAL,
    replaygain_track_gain REAL,
    replaygain_album_gain REAL,
    replaygain_track_peak REAL,
    replaygain_album_peak REAL,
    
    -- Component-based analysis data (JSON)
    essentia_features TEXT, -- JSON
    musicnn_features TEXT, -- JSON
    librosa_features TEXT, -- JSON
    spotify_features TEXT, -- JSON
    external_metadata TEXT, -- JSON
    advanced_categorization TEXT, -- JSON
    
    -- Discovery information
    discovery_source TEXT DEFAULT 'file_system',
    discovery_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- File discovery table for tracking discovery process
CREATE TABLE file_discovery (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path TEXT UNIQUE NOT NULL,
    file_hash TEXT NOT NULL,
    file_size_bytes INTEGER NOT NULL,
    filename TEXT NOT NULL,
    discovery_source TEXT DEFAULT 'file_system',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tags table for external API data
CREATE TABLE tags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    track_id INTEGER NOT NULL,
    tag_name TEXT NOT NULL,
    tag_value TEXT,
    source TEXT NOT NULL, -- 'musicbrainz', 'lastfm', 'spotify', 'musicnn'
    confidence REAL DEFAULT 1.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (track_id) REFERENCES tracks(id) ON DELETE CASCADE,
    UNIQUE(track_id, tag_name, source)
);

-- Playlists table
CREATE TABLE playlists (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    description TEXT,
    generation_method TEXT DEFAULT 'manual', -- 'manual', 'kmeans', 'tag_based', 'time_based'
    generation_params TEXT, -- JSON object of generation parameters
    track_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Playlist tracks junction table
CREATE TABLE playlist_tracks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    playlist_id INTEGER NOT NULL,
    track_id INTEGER NOT NULL,
    position INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (playlist_id) REFERENCES playlists(id) ON DELETE CASCADE,
    FOREIGN KEY (track_id) REFERENCES tracks(id) ON DELETE CASCADE,
    UNIQUE(playlist_id, track_id, position)
);

-- Unified cache table (replaces multiple cache tables)
CREATE TABLE cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cache_key TEXT UNIQUE NOT NULL,
    cache_value TEXT NOT NULL, -- JSON
    cache_type TEXT NOT NULL, -- 'general', 'failed_analysis', 'discovery', 'api_response'
    expires_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Statistics table for metrics and analytics
CREATE TABLE statistics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    category TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    metric_value REAL NOT NULL,
    metric_data TEXT, -- JSON for additional data
    date_recorded TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Discovery cache table for tracking file discovery operations
CREATE TABLE discovery_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    directory_path TEXT NOT NULL,
    file_count INTEGER NOT NULL,
    scan_duration REAL NOT NULL, -- Duration in seconds
    status TEXT NOT NULL, -- 'completed', 'failed', 'in_progress'
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance

-- Essential indexes for fast queries
CREATE INDEX idx_tracks_file_path ON tracks(file_path);
CREATE INDEX idx_tracks_status ON tracks(status);
CREATE INDEX idx_tracks_analysis_status ON tracks(analysis_status);
CREATE INDEX idx_tracks_artist ON tracks(artist);
CREATE INDEX idx_tracks_title ON tracks(title);
CREATE INDEX idx_tracks_artist_title ON tracks(artist, title);
CREATE INDEX idx_tracks_genre ON tracks(genre);
CREATE INDEX idx_tracks_year ON tracks(year);
CREATE INDEX idx_tracks_bpm ON tracks(bpm);
CREATE INDEX idx_tracks_key ON tracks(key);
CREATE INDEX idx_tracks_energy ON tracks(energy);
CREATE INDEX idx_tracks_danceability ON tracks(danceability);
CREATE INDEX idx_tracks_analysis_date ON tracks(analysis_date);
CREATE INDEX idx_tracks_speechiness ON tracks(speechiness);
CREATE INDEX idx_tracks_liveness ON tracks(liveness);

-- Composite indexes for common queries
CREATE INDEX idx_tracks_genre_year ON tracks(genre, year);
CREATE INDEX idx_tracks_bpm_energy ON tracks(bpm, energy);
CREATE INDEX idx_tracks_key_mode ON tracks(key, mode);
CREATE INDEX idx_tracks_artist_album ON tracks(artist, album);

-- Tags indexes
CREATE INDEX idx_tags_track_id ON tags(track_id);
CREATE INDEX idx_tags_source ON tags(source);
CREATE INDEX idx_tags_name ON tags(tag_name);
CREATE INDEX idx_tags_source_name ON tags(source, tag_name);

-- Playlist indexes
CREATE INDEX idx_playlists_name ON playlists(name);
CREATE INDEX idx_playlist_tracks_playlist_id ON playlist_tracks(playlist_id);
CREATE INDEX idx_playlist_tracks_track_id ON playlist_tracks(track_id);
CREATE INDEX idx_playlist_tracks_position ON playlist_tracks(position);

-- Cache indexes
CREATE INDEX idx_cache_key ON cache(cache_key);
CREATE INDEX idx_cache_type ON cache(cache_type);
CREATE INDEX idx_cache_expires_at ON cache(expires_at);

-- Statistics indexes
CREATE INDEX idx_statistics_category ON statistics(category);
CREATE INDEX idx_statistics_date ON statistics(date_recorded);

-- Discovery cache indexes
CREATE INDEX idx_discovery_cache_directory ON discovery_cache(directory_path);
CREATE INDEX idx_discovery_cache_status ON discovery_cache(status);
CREATE INDEX idx_discovery_cache_created_at ON discovery_cache(created_at);

-- File hash index for tracking
CREATE INDEX idx_tracks_file_hash ON tracks(file_hash);

-- =============================================================================
-- VIEWS FOR WEB UI OPTIMIZATION
-- =============================================================================

-- Complete track data with tags
CREATE VIEW track_complete AS
SELECT 
    t.*,
    GROUP_CONCAT(DISTINCT tg.tag_name || ':' || tg.tag_value) as all_tags
FROM tracks t
LEFT JOIN tags tg ON t.id = tg.track_id
GROUP BY t.id;

-- Track summary for lists
CREATE VIEW track_summary AS
SELECT 
    id, file_path, filename, title, artist, album, genre, year, duration,
    bpm, key, mode, energy, danceability, status, analysis_date
FROM tracks
WHERE status = 'analyzed';

-- Audio analysis features
CREATE VIEW audio_analysis_complete AS
SELECT 
    id, file_path, title, artist,
    bpm, key, mode, loudness, energy, danceability, valence, acousticness, instrumentalness,
    rhythm_features, spectral_features, mfcc_features, musicnn_features, spotify_features
FROM tracks
WHERE status = 'analyzed';

-- Failed analysis summary
CREATE VIEW failed_analysis_summary AS
SELECT 
    cache_key, cache_value, created_at
FROM cache
WHERE cache_type = 'failed_analysis';

-- Playlist features for generation
CREATE VIEW playlist_features AS
SELECT 
    t.id, t.title, t.artist, t.album, t.genre, t.year,
    t.bpm, t.key, t.mode, t.energy, t.danceability, t.valence, t.acousticness,
    pt.position
FROM tracks t
JOIN playlist_tracks pt ON t.id = pt.track_id
WHERE t.status = 'analyzed';

-- Playlist summary
CREATE VIEW playlist_summary AS
SELECT 
    p.id, p.name, p.description, p.generation_method, p.track_count,
    p.created_at, p.updated_at,
    COUNT(pt.track_id) as actual_track_count
FROM playlists p
LEFT JOIN playlist_tracks pt ON p.id = pt.playlist_id
GROUP BY p.id;

-- Genre analysis
CREATE VIEW genre_analysis AS
SELECT 
    genre,
    COUNT(*) as track_count,
    AVG(bpm) as avg_bpm,
    AVG(energy) as avg_energy,
    AVG(danceability) as avg_danceability,
    AVG(duration) as avg_duration
FROM tracks
WHERE status = 'analyzed' AND genre IS NOT NULL
GROUP BY genre;

-- Statistics summary
CREATE VIEW statistics_summary AS
SELECT 
    category,
    metric_name,
    AVG(metric_value) as avg_value,
    MAX(metric_value) as max_value,
    MIN(metric_value) as min_value,
    COUNT(*) as record_count
FROM statistics
GROUP BY category, metric_name;

-- =============================================================================
-- TRIGGERS FOR DATA INTEGRITY
-- =============================================================================

-- Update playlist track count
CREATE TRIGGER update_playlist_track_count_insert
AFTER INSERT ON playlist_tracks
BEGIN
    UPDATE playlists 
    SET track_count = (
        SELECT COUNT(*) FROM playlist_tracks WHERE playlist_id = NEW.playlist_id
    ),
    updated_at = CURRENT_TIMESTAMP
    WHERE id = NEW.playlist_id;
END;

CREATE TRIGGER update_playlist_track_count_delete
AFTER DELETE ON playlist_tracks
BEGIN
    UPDATE playlists 
    SET track_count = (
        SELECT COUNT(*) FROM playlist_tracks WHERE playlist_id = OLD.playlist_id
    ),
    updated_at = CURRENT_TIMESTAMP
    WHERE id = OLD.playlist_id;
END;

-- Update track updated_at timestamp
CREATE TRIGGER update_track_timestamp
AFTER UPDATE ON tracks
BEGIN
    UPDATE tracks SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

-- Update playlist updated_at timestamp
CREATE TRIGGER update_playlist_timestamp
AFTER UPDATE ON playlists
BEGIN
    UPDATE playlists SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END; 