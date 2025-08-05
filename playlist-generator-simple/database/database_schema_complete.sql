-- =============================================================================
-- COMPLETE DATABASE SCHEMA WITH ALL AUDIO ANALYSIS FIELDS
-- =============================================================================
-- This schema includes all original fields plus all missing audio analysis fields

-- Main tracks table with complete audio analysis fields
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
    
    -- Core metadata (essential for web UI)
    title TEXT NOT NULL,
    artist TEXT NOT NULL,
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
    
    -- Advanced Rhythm Analysis (NEW)
    tempo_confidence REAL,
    tempo_strength REAL,
    rhythm_pattern TEXT,
    beat_positions TEXT, -- JSON array of beat timestamps
    onset_times TEXT, -- JSON array of onset detection times
    rhythm_complexity REAL,
    
    -- Spectral features (Essentia)
    spectral_centroid REAL,
    spectral_flatness REAL,
    spectral_rolloff REAL,
    spectral_bandwidth REAL,
    spectral_contrast_mean REAL,
    spectral_contrast_std REAL,
    
    -- Extended Spectral Analysis (NEW)
    spectral_flux REAL,
    spectral_entropy REAL,
    spectral_crest REAL,
    spectral_decrease REAL,
    spectral_kurtosis REAL,
    spectral_skewness REAL,
    
    -- Loudness features (Essentia)
    dynamic_complexity REAL,
    loudness_range REAL,
    dynamic_range REAL,
    
    -- Key features (Essentia)
    scale TEXT, -- 'major', 'minor'
    key_strength REAL,
    key_confidence REAL,
    
    -- Advanced Key Analysis (NEW)
    key_scale_notes TEXT, -- JSON array
    key_chord_progression TEXT, -- JSON array
    modulation_points TEXT, -- JSON array
    tonal_centroid REAL,
    
    -- Harmonic Analysis (NEW)
    harmonic_complexity REAL,
    chord_progression TEXT, -- JSON array of detected chords
    harmonic_centroid REAL,
    harmonic_contrast REAL,
    chord_changes INTEGER, -- Number of chord changes per minute
    
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
    lyricist TEXT,
    band TEXT,
    conductor TEXT,
    remixer TEXT,
    subtitle TEXT,
    grouping TEXT,
    quality TEXT,
    
    -- Analysis metadata
    analysis_type TEXT DEFAULT 'full', -- 'full', 'basic', 'discovery_only'
    long_audio_category TEXT, -- For long audio files
    
    -- MFCC features (Essentia)
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
    
    -- Chroma features (Essentia)
    chroma_mean TEXT, -- JSON array of 12-dimensional chroma means
    chroma_std TEXT, -- JSON array of 12-dimensional chroma standard deviations
    
    -- Advanced Audio Features (NEW)
    zero_crossing_rate REAL,
    root_mean_square REAL,
    peak_amplitude REAL,
    crest_factor REAL,
    signal_to_noise_ratio REAL,
    
    -- Timbre Analysis (NEW)
    timbre_brightness REAL,
    timbre_warmth REAL,
    timbre_hardness REAL,
    timbre_depth REAL,
    
    -- Musical Structure Analysis (NEW)
    intro_duration REAL,
    verse_duration REAL,
    chorus_duration REAL,
    bridge_duration REAL,
    outro_duration REAL,
    section_boundaries TEXT, -- JSON array of section timestamps
    repetition_rate REAL,
    
    -- Audio Quality Metrics (NEW)
    bitrate_quality REAL,
    sample_rate_quality REAL,
    encoding_quality REAL,
    compression_artifacts REAL,
    clipping_detection REAL,
    
    -- Genre-Specific Features (NEW)
    electronic_elements REAL,
    classical_period TEXT,
    jazz_style TEXT,
    rock_subgenre TEXT,
    folk_style TEXT,
    
    -- Extended features (JSON for flexibility)
    rhythm_features TEXT, -- JSON
    spectral_features TEXT, -- JSON
    mfcc_features TEXT, -- JSON
    musicnn_features TEXT, -- JSON
    spotify_features TEXT, -- JSON
    harmonic_features TEXT, -- JSON (NEW)
    timbre_features TEXT, -- JSON (NEW)
    structure_features TEXT -- JSON (NEW)
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
    generation_method TEXT DEFAULT 'manual',
    generation_params TEXT, -- JSON parameters used for generation
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

-- Analysis cache table for failed analysis tracking
CREATE TABLE analysis_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path TEXT UNIQUE NOT NULL,
    file_hash TEXT NOT NULL,
    filename TEXT NOT NULL,
    file_size_bytes INTEGER NOT NULL,
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    last_attempt TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Discovery cache table for file discovery results
CREATE TABLE discovery_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    directory_path TEXT NOT NULL,
    file_count INTEGER NOT NULL,
    scan_duration REAL NOT NULL,
    status TEXT DEFAULT 'completed',
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- General cache table for API responses and computed data
CREATE TABLE cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cache_key TEXT UNIQUE NOT NULL,
    cache_value TEXT NOT NULL,
    cache_type TEXT DEFAULT 'general',
    expires_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Statistics table for dashboard metrics
CREATE TABLE statistics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    category TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    metric_value REAL NOT NULL,
    metric_data TEXT, -- JSON additional data
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX idx_tracks_file_path ON tracks(file_path);
CREATE INDEX idx_tracks_file_hash ON tracks(file_hash);
CREATE INDEX idx_tracks_artist ON tracks(artist);
CREATE INDEX idx_tracks_title ON tracks(title);
CREATE INDEX idx_tracks_genre ON tracks(genre);
CREATE INDEX idx_tracks_year ON tracks(year);
CREATE INDEX idx_tracks_bpm ON tracks(bpm);
CREATE INDEX idx_tracks_key ON tracks(key);
CREATE INDEX idx_tracks_energy ON tracks(energy);
CREATE INDEX idx_tracks_danceability ON tracks(danceability);
CREATE INDEX idx_tracks_valence ON tracks(valence);
CREATE INDEX idx_tracks_acousticness ON tracks(acousticness);
CREATE INDEX idx_tracks_instrumentalness ON tracks(instrumentalness);
CREATE INDEX idx_tracks_analysis_date ON tracks(analysis_date);
CREATE INDEX idx_tracks_status ON tracks(status);

-- Indexes for new fields
CREATE INDEX idx_tracks_tempo_confidence ON tracks(tempo_confidence);
CREATE INDEX idx_tracks_rhythm_complexity ON tracks(rhythm_complexity);
CREATE INDEX idx_tracks_harmonic_complexity ON tracks(harmonic_complexity);
CREATE INDEX idx_tracks_timbre_brightness ON tracks(timbre_brightness);
CREATE INDEX idx_tracks_spectral_flux ON tracks(spectral_flux);
CREATE INDEX idx_tracks_root_mean_square ON tracks(root_mean_square);

-- Indexes for tags
CREATE INDEX idx_tags_track_id ON tags(track_id);
CREATE INDEX idx_tags_source ON tags(source);
CREATE INDEX idx_tags_tag_name ON tags(tag_name);

-- Indexes for playlists
CREATE INDEX idx_playlist_tracks_playlist_id ON playlist_tracks(playlist_id);
CREATE INDEX idx_playlist_tracks_track_id ON playlist_tracks(track_id);

-- Indexes for cache
CREATE INDEX idx_cache_key ON cache(cache_key);
CREATE INDEX idx_cache_type ON cache(cache_type);
CREATE INDEX idx_cache_expires ON cache(expires_at);

-- Indexes for statistics
CREATE INDEX idx_statistics_category ON statistics(category);
CREATE INDEX idx_statistics_metric_name ON statistics(metric_name);
CREATE INDEX idx_statistics_created_at ON statistics(created_at);

-- Create views for web UI optimization
CREATE VIEW track_complete AS
SELECT t.*, 
       GROUP_CONCAT(DISTINCT tag.tag_name || ':' || tag.tag_value) as all_tags
FROM tracks t
LEFT JOIN tags tag ON t.id = tag.track_id
GROUP BY t.id;

CREATE VIEW track_summary AS
SELECT id, file_path, filename, title, artist, album, genre, year, 
       duration, bpm, key, energy, danceability, valence, 
       analysis_date, status
FROM tracks;

CREATE VIEW audio_analysis_complete AS
SELECT id, file_path, title, artist,
       bpm, key, mode, loudness, energy, danceability, valence,
       acousticness, instrumentalness, speechiness, liveness,
       rhythm_confidence, tempo_confidence, rhythm_complexity,
       harmonic_complexity, chord_changes,
       spectral_centroid, spectral_flatness, spectral_flux,
       timbre_brightness, timbre_warmth,
       root_mean_square, peak_amplitude, crest_factor
FROM tracks;

CREATE VIEW playlist_features AS
SELECT t.id, t.file_path, t.title, t.artist,
       t.bpm, t.key, t.energy, t.danceability, t.valence,
       t.acousticness, t.instrumentalness,
       t.rhythm_complexity, t.harmonic_complexity,
       t.timbre_brightness, t.spectral_flux
FROM tracks t
WHERE t.status = 'analyzed';

CREATE VIEW playlist_summary AS
SELECT p.id, p.name, p.description, p.generation_method,
       p.track_count, p.created_at,
       COUNT(pt.track_id) as actual_track_count
FROM playlists p
LEFT JOIN playlist_tracks pt ON p.id = pt.playlist_id
GROUP BY p.id;

CREATE VIEW genre_analysis AS
SELECT genre,
       COUNT(*) as track_count,
       AVG(bpm) as avg_bpm,
       AVG(energy) as avg_energy,
       AVG(danceability) as avg_danceability,
       AVG(valence) as avg_valence
FROM tracks
WHERE genre IS NOT NULL AND status = 'analyzed'
GROUP BY genre;

-- Create triggers for data integrity
CREATE TRIGGER update_tracks_updated_at
    AFTER UPDATE ON tracks
    FOR EACH ROW
BEGIN
    UPDATE tracks SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

CREATE TRIGGER update_playlists_updated_at
    AFTER UPDATE ON playlists
    FOR EACH ROW
BEGIN
    UPDATE playlists SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

CREATE TRIGGER update_playlist_track_count
    AFTER INSERT ON playlist_tracks
    FOR EACH ROW
BEGIN
    UPDATE playlists 
    SET track_count = (SELECT COUNT(*) FROM playlist_tracks WHERE playlist_id = NEW.playlist_id)
    WHERE id = NEW.playlist_id;
END;

CREATE TRIGGER update_playlist_track_count_delete
    AFTER DELETE ON playlist_tracks
    FOR EACH ROW
BEGIN
    UPDATE playlists 
    SET track_count = (SELECT COUNT(*) FROM playlist_tracks WHERE playlist_id = OLD.playlist_id)
    WHERE id = OLD.playlist_id;
END; 