-- =============================================================================
-- OPTIMIZED DATABASE SCHEMA
-- =============================================================================
-- Removed redundant tables and merged functionality for better performance

-- Main tracks table with all audio features and metadata
CREATE TABLE tracks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    
    -- File identification
    file_path TEXT UNIQUE NOT NULL,
    file_hash TEXT NOT NULL,
    filename TEXT NOT NULL,
    file_size_bytes INTEGER NOT NULL,
    
    -- Timestamps
    analysis_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    discovery_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Status and control fields
    status TEXT DEFAULT 'discovered', -- 'discovered', 'analyzed', 'failed'
    analysis_status TEXT DEFAULT 'pending', -- 'pending', 'in_progress', 'completed', 'failed'
    modified_time REAL, -- File modification time
    retry_count INTEGER DEFAULT 0,
    last_retry_date TIMESTAMP,
    error_message TEXT,
    
    -- Core metadata (from mutagen + external APIs + filename fallback)
    title TEXT NOT NULL,
    artist TEXT NOT NULL,
    album TEXT,
    track_number INTEGER,
    genre TEXT,
    year INTEGER,
    duration REAL,
    
    -- Audio properties (from mutagen)
    bitrate INTEGER,
    sample_rate INTEGER,
    channels INTEGER,
    
    -- Additional metadata (from mutagen)
    composer TEXT,
    lyricist TEXT,
    band TEXT,
    conductor TEXT,
    remixer TEXT,
    subtitle TEXT,
    grouping TEXT,
    publisher TEXT,
    copyright TEXT,
    encoded_by TEXT,
    language TEXT,
    mood TEXT,
    style TEXT,
    quality TEXT,
    original_artist TEXT,
    original_album TEXT,
    original_year INTEGER,
    original_filename TEXT,
    content_group TEXT,
    encoder TEXT,
    file_type TEXT,
    playlist_delay TEXT,
    recording_time TEXT,
    tempo TEXT,
    length TEXT,
    replaygain_track_gain TEXT,
    replaygain_album_gain TEXT,
    replaygain_track_peak TEXT,
    replaygain_album_peak TEXT,
    
    -- =============================================================================
    -- EXTRACTED AUDIO FEATURES
    -- =============================================================================
    
    -- Rhythm features
    bpm REAL,
    rhythm_confidence REAL,
    bpm_estimates TEXT, -- JSON array
    bpm_intervals TEXT, -- JSON array
    external_bpm REAL, -- BPM from metadata
    
    -- Spectral features
    spectral_centroid REAL,
    spectral_flatness REAL,
    spectral_rolloff REAL,
    spectral_bandwidth REAL,
    spectral_contrast_mean REAL,
    spectral_contrast_std REAL,
    
    -- Loudness features
    loudness REAL,
    dynamic_complexity REAL,
    loudness_range REAL,
    dynamic_range REAL,
    
    -- Key features
    key TEXT,
    scale TEXT, -- 'major', 'minor'
    key_strength REAL,
    key_confidence REAL,
    
    -- MFCC features
    mfcc_coefficients TEXT, -- JSON array
    mfcc_bands TEXT, -- JSON array
    mfcc_std TEXT, -- JSON array
    mfcc_delta TEXT, -- JSON array
    mfcc_delta2 TEXT, -- JSON array
    
    -- MusiCNN features
    embedding TEXT, -- JSON array of 200-dimensional embedding
    embedding_std TEXT, -- JSON array
    embedding_min TEXT, -- JSON array
    embedding_max TEXT, -- JSON array
    tags TEXT, -- JSON object of MusiCNN tag names to confidence scores
    musicnn_skipped INTEGER DEFAULT 0,
    
    -- Chroma features
    chroma_mean TEXT, -- JSON array of 12-dimensional chroma means
    chroma_std TEXT, -- JSON array of 12-dimensional chroma standard deviations
    
    -- =============================================================================
    -- ANALYSIS METADATA
    -- =============================================================================
    analysis_type TEXT DEFAULT 'full', -- 'full', 'simplified', 'categorization_optimized'
    analyzed BOOLEAN DEFAULT FALSE,
    audio_type TEXT DEFAULT 'normal', -- 'normal', 'long', 'large_file'
    long_audio_category TEXT, -- 'radio', 'podcast', 'mix', etc.
    discovery_source TEXT DEFAULT 'file_system',
    
    -- =============================================================================
    -- SPOTIFY-STYLE FEATURES FOR PLAYLIST GENERATION
    -- =============================================================================
    danceability REAL,
    energy REAL,
    mode REAL, -- 0.0 = minor, 1.0 = major
    acousticness REAL,
    instrumentalness REAL,
    speechiness REAL,
    valence REAL,
    liveness REAL,
    popularity REAL
);

-- Tags table for external API tags (MusicBrainz, LastFM, etc.)
CREATE TABLE tags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    track_id INTEGER NOT NULL,
    tag_name TEXT NOT NULL,
    tag_value TEXT,
    source TEXT NOT NULL, -- 'musicbrainz', 'lastfm', 'mutagen', 'musicnn'
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
    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (playlist_id) REFERENCES playlists(id) ON DELETE CASCADE,
    FOREIGN KEY (track_id) REFERENCES tracks(id) ON DELETE CASCADE,
    UNIQUE(playlist_id, track_id)
);

-- General cache table (merged analysis_cache, discovery_cache, cache)
CREATE TABLE cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cache_key TEXT UNIQUE NOT NULL,
    cache_value TEXT NOT NULL,
    cache_type TEXT DEFAULT 'general', -- 'general', 'analysis', 'discovery', 'api', 'failed_analysis'
    expires_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Statistics table for web UI dashboards (merged analysis_statistics, statistics)
CREATE TABLE statistics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    category TEXT NOT NULL, -- 'analysis', 'discovery', 'playlists', 'performance'
    metric_name TEXT NOT NULL,
    metric_value REAL NOT NULL,
    metric_data TEXT, -- JSON object of additional data
    file_path TEXT, -- For analysis-specific statistics
    analysis_status TEXT, -- For analysis-specific statistics
    date_recorded TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- =============================================================================
-- INDEXES FOR PERFORMANCE
-- =============================================================================

-- Core lookup indexes
CREATE INDEX idx_tracks_file_path ON tracks(file_path);
CREATE INDEX idx_tracks_file_hash ON tracks(file_hash);
CREATE INDEX idx_tracks_filename ON tracks(filename);

-- Status and control indexes
CREATE INDEX idx_tracks_status ON tracks(status);
CREATE INDEX idx_tracks_analysis_status ON tracks(analysis_status);
CREATE INDEX idx_tracks_retry_count ON tracks(retry_count);
CREATE INDEX idx_tracks_modified_time ON tracks(modified_time);

-- Metadata lookup indexes
CREATE INDEX idx_tracks_artist ON tracks(artist);
CREATE INDEX idx_tracks_title ON tracks(title);
CREATE INDEX idx_tracks_artist_title ON tracks(artist, title);
CREATE INDEX idx_tracks_album ON tracks(album);
CREATE INDEX idx_tracks_genre ON tracks(genre);
CREATE INDEX idx_tracks_year ON tracks(year);

-- Analysis date indexes
CREATE INDEX idx_tracks_analysis_date ON tracks(analysis_date);
CREATE INDEX idx_tracks_discovery_date ON tracks(discovery_date);

-- Audio feature indexes (for playlist generation)
CREATE INDEX idx_tracks_bpm ON tracks(bpm);
CREATE INDEX idx_tracks_key ON tracks(key);
CREATE INDEX idx_tracks_loudness ON tracks(loudness);
CREATE INDEX idx_tracks_duration ON tracks(duration);
CREATE INDEX idx_tracks_rhythm_confidence ON tracks(rhythm_confidence);
CREATE INDEX idx_tracks_spectral_centroid ON tracks(spectral_centroid);
CREATE INDEX idx_tracks_long_audio_category ON tracks(long_audio_category);
CREATE INDEX idx_tracks_audio_type ON tracks(audio_type);

-- Composite indexes for common queries
CREATE INDEX idx_tracks_artist_album ON tracks(artist, album);
CREATE INDEX idx_tracks_genre_year ON tracks(genre, year);
CREATE INDEX idx_tracks_bpm_energy ON tracks(bpm, energy);
CREATE INDEX idx_tracks_key_mode ON tracks(key, mode);

-- Tags indexes
CREATE INDEX idx_tags_track_id ON tags(track_id);
CREATE INDEX idx_tags_source ON tags(source);
CREATE INDEX idx_tags_name ON tags(tag_name);
CREATE INDEX idx_tags_source_name ON tags(source, tag_name);

-- Playlist indexes
CREATE INDEX idx_playlists_name ON playlists(name);
CREATE INDEX idx_playlists_created_at ON playlists(created_at);
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
CREATE INDEX idx_statistics_category_date ON statistics(category, date_recorded);

-- =============================================================================
-- VIEWS FOR COMMON QUERIES
-- =============================================================================

-- Complete track information with tags
CREATE VIEW track_complete AS
SELECT 
    t.*,
    GROUP_CONCAT(tag.tag_name || ':' || tag.tag_value) as all_tags
FROM tracks t
LEFT JOIN tags tag ON t.id = tag.track_id
GROUP BY t.id;

-- Track summary for web UI
CREATE VIEW track_summary AS
SELECT 
    id, file_path, filename, title, artist, album, genre, year, duration,
    bpm, key, scale, loudness, rhythm_confidence, spectral_centroid,
    long_audio_category, analysis_type, analyzed, analysis_date,
    status, analysis_status, retry_count
FROM tracks;

-- Audio analysis features for playlist generation
CREATE VIEW audio_analysis_complete AS
SELECT 
    id, file_path, title, artist, album, genre, year, duration,
    -- Rhythm features
    bpm, rhythm_confidence, bpm_estimates, bpm_intervals, external_bpm,
    -- Spectral features
    spectral_centroid, spectral_flatness, spectral_rolloff, spectral_bandwidth,
    spectral_contrast_mean, spectral_contrast_std,
    -- Loudness features
    loudness, dynamic_complexity, loudness_range, dynamic_range,
    -- Key features
    key, scale, key_strength, key_confidence,
    -- MFCC features
    mfcc_coefficients, mfcc_bands, mfcc_std, mfcc_delta, mfcc_delta2,
    -- MusiCNN features
    embedding, tags,
    -- Chroma features
    chroma_mean, chroma_std,
    -- Analysis metadata
    analysis_type, long_audio_category, analyzed, analysis_date,
    -- Status fields
    status, analysis_status, retry_count
FROM tracks
WHERE analyzed = TRUE;

-- Failed analysis summary
CREATE VIEW failed_analysis_summary AS
SELECT 
    file_path, filename, error_message, retry_count, last_retry_date,
    status, analysis_status
FROM tracks 
WHERE status = 'failed' OR analysis_status = 'failed'
UNION
SELECT 
    file_path, filename, error_message, retry_count, last_retry_date,
    status, 'failed' as analysis_status
FROM failed_analysis;

-- Discovery summary
CREATE VIEW discovery_summary AS
SELECT 
    directory_path,
    file_count,
    scan_duration,
    status,
    created_at,
    COUNT(*) as scan_count
FROM discovery_cache
GROUP BY directory_path, file_count, scan_duration, status, created_at;

-- Playlist features for generation
CREATE VIEW playlist_features AS
SELECT 
    p.id as playlist_id, p.name as playlist_name, p.generation_method,
    p.generation_params, p.track_count, p.created_at,
    pt.position, pt.track_id,
    t.title, t.artist, t.album, t.genre, t.year, t.duration,
    t.bpm, t.key, t.scale, t.loudness, t.rhythm_confidence,
    t.spectral_centroid, t.long_audio_category
FROM playlists p
JOIN playlist_tracks pt ON p.id = pt.playlist_id
JOIN tracks t ON pt.track_id = t.id
ORDER BY p.id, pt.position;

-- Playlist summary
CREATE VIEW playlist_summary AS
SELECT 
    p.id, p.name, p.description, p.generation_method,
    p.track_count, p.created_at, p.updated_at,
    COUNT(pt.track_id) as actual_track_count,
    AVG(t.duration) as avg_duration,
    AVG(t.bpm) as avg_bpm,
    GROUP_CONCAT(DISTINCT t.genre) as genres
FROM playlists p
LEFT JOIN playlist_tracks pt ON p.id = pt.playlist_id
LEFT JOIN tracks t ON pt.track_id = t.id
GROUP BY p.id;

-- Genre analysis
CREATE VIEW genre_analysis AS
SELECT 
    genre,
    COUNT(*) as track_count,
    AVG(duration) as avg_duration,
    AVG(bpm) as avg_bpm,
    AVG(loudness) as avg_loudness,
    AVG(rhythm_confidence) as avg_rhythm_confidence,
    AVG(spectral_centroid) as avg_spectral_centroid,
    GROUP_CONCAT(DISTINCT long_audio_category) as categories
FROM tracks
WHERE genre IS NOT NULL AND analyzed = TRUE
GROUP BY genre
ORDER BY track_count DESC;

-- Statistics summary
CREATE VIEW statistics_summary AS
SELECT 
    category,
    metric_name,
    AVG(metric_value) as avg_value,
    MAX(metric_value) as max_value,
    MIN(metric_value) as min_value,
    COUNT(*) as record_count,
    MAX(date_recorded) as last_recorded
FROM statistics
GROUP BY category, metric_name
ORDER BY category, metric_name;

-- Analysis performance summary
CREATE VIEW analysis_performance AS
SELECT 
    analysis_status,
    COUNT(*) as file_count,
    AVG(analysis_duration) as avg_duration,
    AVG(memory_usage_mb) as avg_memory_mb,
    AVG(cpu_usage_percent) as avg_cpu_percent,
    MAX(created_at) as last_analysis
FROM analysis_statistics
GROUP BY analysis_status;

-- =============================================================================
-- COMMENTS AND DOCUMENTATION
-- =============================================================================

/*
DATABASE SCHEMA DOCUMENTATION

This schema is based on ACTUAL data extraction and usage from:
1. audio_analyzer.py - Audio feature extraction
2. database.py - Database storage methods
3. file_discovery.py - File discovery and tracking
4. mutagen - Metadata extraction
5. External APIs - MusicBrainz, LastFM

EXTRACTED FEATURES:
- Rhythm: bpm, rhythm_confidence, bpm_estimates, bpm_intervals, external_bpm
- Spectral: spectral_centroid, spectral_flatness (others default to 0.0)
- Loudness: loudness, dynamic_complexity (others default to 0.0)
- Key: key, scale, key_strength (key_confidence defaults to 0.0)
- MFCC: mfcc_coefficients, mfcc_bands, mfcc_std (delta fields default to [])
- MusiCNN: embedding (200-dim), tags (dict of tag names to scores)
- Chroma: chroma_mean (12-dim), chroma_std (12-dim)

CONTROL FIELDS:
- status: 'discovered', 'analyzed', 'failed'
- analysis_status: 'pending', 'in_progress', 'completed', 'failed'
- retry_count, last_retry_date, error_message
- modified_time, discovered_date

ADDITIONAL TABLES:
- file_metadata: File discovery tracking
- analysis_statistics: Performance metrics
- failed_analysis: Failed analysis tracking
- analysis_cache: Cache for failed/skipped analysis
- discovery_cache: Discovery process tracking

PLACEHOLDER FIELDS:
- danceability, energy, mode (for future playlist generation features)
- acousticness, instrumentalness, speechiness, valence, liveness, popularity (Spotify-style features)

USAGE:
1. Run this schema to create a new database
2. All extracted features will be properly stored
3. All control fields and tracking will work correctly
4. Placeholder fields can be populated when feature extraction is implemented
*/ 