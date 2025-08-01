-- Ultimate Playlist Generator Database Schema
-- Captures ALL analysis data in normalized, indexed structure

-- Main tracks table
CREATE TABLE tracks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path TEXT UNIQUE NOT NULL,  -- File storage reference
    file_hash TEXT NOT NULL,         -- Change detection key (triggers re-analysis)
    filename TEXT NOT NULL,
    file_size_bytes INTEGER NOT NULL,
    analysis_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    analysis_version TEXT,
    analysis_type TEXT DEFAULT 'full', -- 'basic', 'full', 'simplified'
    long_audio_category TEXT, -- 'podcast', 'audiobook', 'music', 'other'
    
    -- Music identifiers (main lookup keys)
    title TEXT NOT NULL,
    artist TEXT NOT NULL,
    album TEXT,
    track_number INTEGER,
    
    -- Music metadata (indexed for fast queries)
    genre TEXT,
    year INTEGER,
    duration REAL, -- seconds
    bitrate INTEGER,
    sample_rate INTEGER,
    channels INTEGER,
    
    -- Extended metadata from file tags (Mutagen)
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
    playlist_delay INTEGER,
    recording_time TEXT,
    
    -- Music-specific metadata from file tags
    tempo TEXT,  -- Alternative BPM field
    length TEXT, -- Track length in milliseconds
    
    -- ReplayGain metadata
    replaygain_track_gain REAL,
    replaygain_album_gain REAL,
    replaygain_track_peak REAL,
    replaygain_album_peak REAL,
    
    -- Audio features (indexed for playlist generation)
    bpm REAL,
    key TEXT,
    mode TEXT, -- 'major', 'minor'
    scale TEXT, -- 'major', 'minor', etc.
    key_strength REAL,
    loudness REAL,
    danceability REAL,
    energy REAL,
    
    -- Analysis confidence
    rhythm_confidence REAL,
    key_confidence REAL,
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- External API data
CREATE TABLE external_metadata (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    track_id INTEGER NOT NULL,
    source TEXT NOT NULL, -- 'musicbrainz', 'lastfm', 'spotify'
    
    -- MusicBrainz data
    musicbrainz_id TEXT,
    musicbrainz_artist_id TEXT,
    musicbrainz_album_id TEXT,
    release_date DATE,
    disc_number INTEGER,
    duration_ms INTEGER,
    
    -- Last.fm data
    lastfm_url TEXT,
    play_count INTEGER,
    listeners INTEGER,
    rating REAL,
    
    -- Confidence and timestamps
    confidence REAL DEFAULT 1.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (track_id) REFERENCES tracks(id) ON DELETE CASCADE,
    UNIQUE(track_id, source)
);

-- Tags from external APIs
CREATE TABLE tags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    track_id INTEGER NOT NULL,
    source TEXT NOT NULL, -- 'musicbrainz', 'lastfm', 'musicnn', 'user'
    tag_name TEXT NOT NULL,
    tag_value TEXT, -- For key-value tags
    confidence REAL DEFAULT 1.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (track_id) REFERENCES tracks(id) ON DELETE CASCADE,
    UNIQUE(track_id, source, tag_name, tag_value)
);

-- Spectral features
CREATE TABLE spectral_features (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    track_id INTEGER NOT NULL,
    spectral_centroid REAL,
    spectral_rolloff REAL,
    spectral_flatness REAL,
    spectral_bandwidth REAL,
    spectral_contrast_mean REAL,
    spectral_contrast_std REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (track_id) REFERENCES tracks(id) ON DELETE CASCADE,
    UNIQUE(track_id)
);

-- Loudness features
CREATE TABLE loudness_features (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    track_id INTEGER NOT NULL,
    integrated_loudness REAL,
    loudness_range REAL,
    momentary_loudness_mean REAL,
    momentary_loudness_std REAL,
    short_term_loudness_mean REAL,
    short_term_loudness_std REAL,
    dynamic_complexity REAL,
    dynamic_range REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (track_id) REFERENCES tracks(id) ON DELETE CASCADE,
    UNIQUE(track_id)
);

-- MFCC features (stored as JSON for efficiency)
CREATE TABLE mfcc_features (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    track_id INTEGER NOT NULL,
    mfcc_coefficients TEXT, -- JSON array of 13+ values
    mfcc_bands TEXT, -- JSON array
    mfcc_std TEXT, -- JSON array
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (track_id) REFERENCES tracks(id) ON DELETE CASCADE,
    UNIQUE(track_id)
);

-- MusiCNN features (neural embeddings)
CREATE TABLE musicnn_features (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    track_id INTEGER NOT NULL,
    embedding TEXT, -- JSON array of 200+ dimensions
    tags TEXT, -- JSON object of predicted tags
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (track_id) REFERENCES tracks(id) ON DELETE CASCADE,
    UNIQUE(track_id)
);

-- Chroma features
CREATE TABLE chroma_features (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    track_id INTEGER NOT NULL,
    chroma_mean TEXT, -- JSON array of 12 values
    chroma_std TEXT, -- JSON array of 12 values
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (track_id) REFERENCES tracks(id) ON DELETE CASCADE,
    UNIQUE(track_id)
);

-- Rhythm analysis details
CREATE TABLE rhythm_features (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    track_id INTEGER NOT NULL,
    bpm_estimates TEXT, -- JSON array
    bpm_intervals TEXT, -- JSON array
    external_bpm REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (track_id) REFERENCES tracks(id) ON DELETE CASCADE,
    UNIQUE(track_id)
);

-- Advanced features
CREATE TABLE advanced_features (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    track_id INTEGER NOT NULL,
    onset_rate REAL,
    zero_crossing_rate REAL,
    harmonic_complexity REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (track_id) REFERENCES tracks(id) ON DELETE CASCADE,
    UNIQUE(track_id)
);

-- Playlists
CREATE TABLE playlists (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Playlist tracks
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

-- Analysis cache (for failed/partial analyses)
CREATE TABLE analysis_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path TEXT UNIQUE NOT NULL,
    analysis_data TEXT, -- JSON blob of partial analysis
    status TEXT DEFAULT 'failed', -- 'failed', 'partial', 'success'
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- =============================================================================
-- INDEXES FOR LIGHTNING FAST QUERIES
-- =============================================================================

-- Music identifier indexes (highest priority)
CREATE INDEX idx_tracks_artist ON tracks(artist);        -- Main music lookup key
CREATE INDEX idx_tracks_title ON tracks(title);          -- Track identification
CREATE INDEX idx_tracks_artist_title ON tracks(artist, title);  -- Combined lookup
CREATE INDEX idx_tracks_album ON tracks(album);          -- Album grouping

-- File system indexes (for change detection)
CREATE INDEX idx_tracks_file_path ON tracks(file_path);  -- File storage reference
CREATE INDEX idx_tracks_file_hash ON tracks(file_hash);  -- Change detection key

-- Music metadata indexes (for filtering and search)
CREATE INDEX idx_tracks_genre ON tracks(genre);
CREATE INDEX idx_tracks_year ON tracks(year);
CREATE INDEX idx_tracks_composer ON tracks(composer);
CREATE INDEX idx_tracks_band ON tracks(band);
CREATE INDEX idx_tracks_mood ON tracks(mood);
CREATE INDEX idx_tracks_style ON tracks(style);
CREATE INDEX idx_tracks_lyricist ON tracks(lyricist);
CREATE INDEX idx_tracks_conductor ON tracks(conductor);
CREATE INDEX idx_tracks_remixer ON tracks(remixer);
CREATE INDEX idx_tracks_publisher ON tracks(publisher);
CREATE INDEX idx_tracks_language ON tracks(language);
CREATE INDEX idx_tracks_quality ON tracks(quality);

-- Audio feature indexes (for playlist generation)
CREATE INDEX idx_tracks_bpm ON tracks(bpm);
CREATE INDEX idx_tracks_key ON tracks(key);
CREATE INDEX idx_tracks_scale ON tracks(scale);
CREATE INDEX idx_tracks_loudness ON tracks(loudness);
CREATE INDEX idx_tracks_danceability ON tracks(danceability);
CREATE INDEX idx_tracks_energy ON tracks(energy);
CREATE INDEX idx_tracks_duration ON tracks(duration);
CREATE INDEX idx_tracks_analysis_date ON tracks(analysis_date);

-- Tag queries
CREATE INDEX idx_tags_track_id ON tags(track_id);
CREATE INDEX idx_tags_source ON tags(source);
CREATE INDEX idx_tags_name ON tags(tag_name);
CREATE INDEX idx_tags_value ON tags(tag_value);

-- External metadata queries
CREATE INDEX idx_external_metadata_track_id ON external_metadata(track_id);
CREATE INDEX idx_external_metadata_source ON external_metadata(source);
CREATE INDEX idx_external_metadata_musicbrainz_id ON external_metadata(musicbrainz_id);

-- Playlist queries
CREATE INDEX idx_playlist_tracks_playlist_id ON playlist_tracks(playlist_id);
CREATE INDEX idx_playlist_tracks_track_id ON playlist_tracks(track_id);
CREATE INDEX idx_playlist_tracks_position ON playlist_tracks(position);

-- Analysis cache
CREATE INDEX idx_analysis_cache_status ON analysis_cache(status);
CREATE INDEX idx_analysis_cache_retry_count ON analysis_cache(retry_count);

-- =============================================================================
-- VIEWS FOR COMMON QUERIES
-- =============================================================================

-- Complete track view with all data
CREATE VIEW track_complete AS
SELECT 
    t.*,
    sf.spectral_centroid,
    sf.spectral_rolloff,
    sf.spectral_flatness,
    lf.integrated_loudness,
    lf.loudness_range,
    af.onset_rate,
    af.zero_crossing_rate,
    rf.bpm_estimates,
    rf.external_bpm,
    GROUP_CONCAT(DISTINCT tag.tag_name) as all_tags
FROM tracks t
LEFT JOIN spectral_features sf ON t.id = sf.track_id
LEFT JOIN loudness_features lf ON t.id = lf.track_id
LEFT JOIN advanced_features af ON t.id = af.track_id
LEFT JOIN rhythm_features rf ON t.id = rf.track_id
LEFT JOIN tags tag ON t.id = tag.track_id
GROUP BY t.id;

-- Track summary for web UI
CREATE VIEW track_summary AS
SELECT 
    id,
    file_path,
    filename,
    title,
    artist,
    album,
    genre,
    year,
    composer,
    lyricist,
    band,
    conductor,
    remixer,
    mood,
    style,
    language,
    quality,
    duration,
    bpm,
    key,
    mode,
    scale,
    key_strength,
    loudness,
    danceability,
    energy,
    analysis_date,
    long_audio_category
FROM tracks;

-- =============================================================================
-- TRIGGERS FOR DATA INTEGRITY
-- =============================================================================

-- Update timestamp trigger
CREATE TRIGGER update_tracks_timestamp 
    AFTER UPDATE ON tracks
    FOR EACH ROW
BEGIN
    UPDATE tracks SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

-- Update playlist timestamp
CREATE TRIGGER update_playlists_timestamp 
    AFTER UPDATE ON playlists
    FOR EACH ROW
BEGIN
    UPDATE playlists SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

-- =============================================================================
-- SAMPLE QUERIES FOR WEB UI
-- =============================================================================

-- =============================================================================
-- MUSIC LOOKUP QUERIES (Primary Operations)
-- =============================================================================

-- Find track by artist and title (main music lookup)
-- SELECT * FROM track_summary WHERE artist = 'Queen' AND title = 'Bohemian Rhapsody';

-- Find all tracks by artist
-- SELECT * FROM track_summary WHERE artist = 'Queen' ORDER BY album, track_number;

-- Find all tracks from an album
-- SELECT * FROM track_summary WHERE artist = 'Queen' AND album = 'A Night at the Opera' ORDER BY track_number;

-- Find track by title (across all artists)
-- SELECT * FROM track_summary WHERE title LIKE '%Bohemian%';

-- Get complete track data with all features
-- SELECT * FROM track_complete WHERE artist = 'Queen' AND title = 'Bohemian Rhapsody';

-- =============================================================================
-- FILTERING AND SEARCH QUERIES
-- =============================================================================

-- Find tracks by BPM range and key
-- SELECT * FROM track_summary WHERE bpm BETWEEN 120 AND 140 AND key = 'C';

-- Find tracks by genre and year
-- SELECT * FROM track_summary WHERE genre = 'Rock' AND year >= 1990;

-- Find tracks by composer
-- SELECT * FROM track_summary WHERE composer = 'Mozart';

-- Find tracks by mood/style
-- SELECT * FROM track_summary WHERE mood = 'energetic' OR style = 'electronic';

-- =============================================================================
-- PLAYLIST GENERATION QUERIES
-- =============================================================================

-- Find similar tracks (by BPM and key)
-- SELECT * FROM track_summary WHERE bpm BETWEEN 125 AND 135 AND key = 'G' LIMIT 10;

-- Find high-energy dance tracks
-- SELECT * FROM track_summary WHERE energy > 0.8 AND danceability > 0.7;

-- Find tracks by audio features
-- SELECT * FROM track_summary WHERE loudness BETWEEN -20 AND -10 AND duration > 180;

-- =============================================================================
-- RELATED DATA QUERIES
-- =============================================================================

-- Get all tags for a track
-- SELECT tag_name, tag_value, source FROM tags WHERE track_id = (
--     SELECT id FROM tracks WHERE artist = 'Queen' AND title = 'Bohemian Rhapsody'
-- );

-- Get external API data for a track
-- SELECT * FROM external_metadata WHERE track_id = (
--     SELECT id FROM tracks WHERE artist = 'Queen' AND title = 'Bohemian Rhapsody'
-- );

-- Get spectral features for a track
-- SELECT * FROM spectral_features WHERE track_id = (
--     SELECT id FROM tracks WHERE artist = 'Queen' AND title = 'Bohemian Rhapsody'
-- );

-- =============================================================================
-- PLAYLIST MANAGEMENT QUERIES
-- =============================================================================

-- Get playlist with track details
-- SELECT pt.position, ts.* FROM playlist_tracks pt 
-- JOIN track_summary ts ON pt.track_id = ts.id 
-- WHERE pt.playlist_id = ? ORDER BY pt.position;

-- =============================================================================
-- CHANGE DETECTION QUERIES
-- =============================================================================

-- Find tracks that need re-analysis (hash changed)
-- SELECT file_path, file_hash FROM tracks WHERE file_hash != (
--     SELECT file_hash FROM analysis_cache WHERE file_path = tracks.file_path
-- );

-- =============================================================================
-- STATISTICS QUERIES
-- =============================================================================

-- Count tracks by genre
-- SELECT genre, COUNT(*) as count FROM track_summary GROUP BY genre ORDER BY count DESC;

-- Average BPM by year
-- SELECT year, AVG(bpm) as avg_bpm FROM track_summary 
-- WHERE bpm IS NOT NULL GROUP BY year ORDER BY year;

-- Energy distribution
-- SELECT 
--     CASE 
--         WHEN energy < 0.3 THEN 'Low'
--         WHEN energy < 0.7 THEN 'Medium'
--         ELSE 'High'
--     END as energy_level,
--     COUNT(*) as count
-- FROM track_summary WHERE energy IS NOT NULL GROUP BY energy_level; 