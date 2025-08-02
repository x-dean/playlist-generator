-- Complete Database Schema for Playlist Generator Simple
-- Includes all original fields plus comprehensive audio analysis fields
-- This is a complete, unified schema that can be used for new installations

-- Main tracks table with comprehensive audio analysis data
CREATE TABLE tracks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path TEXT UNIQUE NOT NULL,
    file_hash TEXT NOT NULL,
    filename TEXT NOT NULL,
    file_size_bytes INTEGER NOT NULL,
    analysis_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Core music metadata (indexed for fast lookups)
    title TEXT NOT NULL,
    artist TEXT NOT NULL,
    album TEXT,
    track_number INTEGER,
    genre TEXT,
    year INTEGER,
    duration REAL,
    
    -- Basic audio features for playlist generation (indexed)
    bpm REAL,
    key TEXT,
    mode TEXT,
    loudness REAL,
    danceability REAL,
    energy REAL,
    
    -- Analysis metadata
    analysis_type TEXT DEFAULT 'full',
    analyzed BOOLEAN DEFAULT FALSE,
    long_audio_category TEXT,
    
    -- Discovery metadata
    discovery_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    discovery_source TEXT, -- 'file_system', 'user_input', 'api'
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- =============================================================================
    -- RHYTHM & TEMPO ANALYSIS
    -- =============================================================================
    tempo_confidence REAL, -- Confidence level of tempo detection
    tempo_strength REAL, -- Strength of tempo signal
    rhythm_pattern TEXT, -- Rhythmic pattern classification
    beat_positions TEXT, -- JSON array of beat timestamps
    onset_times TEXT, -- JSON array of onset detection times
    rhythm_complexity REAL, -- Complexity score of rhythm
    
    -- =============================================================================
    -- HARMONIC ANALYSIS
    -- =============================================================================
    harmonic_complexity REAL, -- Complexity of harmonic progression
    chord_progression TEXT, -- JSON array of detected chords
    harmonic_centroid REAL, -- Centroid of harmonic spectrum
    harmonic_contrast REAL, -- Contrast in harmonic content
    chord_changes INTEGER, -- Number of chord changes per minute
    
    -- =============================================================================
    -- EXTENDED AUDIO FEATURES (Original + Extended)
    -- =============================================================================
    
    -- Rhythm features (original)
    rhythm_confidence REAL,
    bpm_estimates TEXT, -- JSON array of BPM estimates
    bpm_intervals TEXT, -- JSON array of BPM intervals
    external_bpm REAL, -- BPM from metadata
    
    -- Spectral features (original + extended)
    spectral_centroid REAL,
    spectral_flatness REAL,
    spectral_rolloff REAL,
    spectral_bandwidth REAL,
    spectral_contrast_mean REAL,
    spectral_contrast_std REAL,
    spectral_flux REAL, -- Rate of spectral change over time
    spectral_crest REAL, -- Crest factor of spectrum
    spectral_decrease REAL, -- Spectral decrease rate
    spectral_entropy REAL, -- Entropy of spectral distribution
    spectral_kurtosis REAL, -- Kurtosis of spectral distribution
    spectral_skewness REAL, -- Skewness of spectral distribution
    spectral_slope REAL, -- Spectral slope
    spectral_rolloff_85 REAL, -- 85th percentile rolloff
    spectral_rolloff_95 REAL, -- 95th percentile rolloff
    
    -- Loudness features (original)
    dynamic_complexity REAL,
    loudness_range REAL,
    dynamic_range REAL,
    
    -- Key features (original + extended)
    scale TEXT, -- 'major', 'minor', etc.
    key_strength REAL,
    key_confidence REAL,
    key_scale_notes TEXT, -- JSON array of scale notes
    key_chord_progression TEXT, -- JSON array of key chords
    modulation_points TEXT, -- JSON array of key modulation points
    tonal_centroid REAL, -- Tonal centroid
    harmonic_centroid REAL, -- Harmonic centroid
    
    -- MFCC features (original + extended)
    mfcc_coefficients TEXT, -- JSON array of MFCC coefficients
    mfcc_bands TEXT, -- JSON array of MFCC bands
    mfcc_std TEXT, -- JSON array of MFCC standard deviations
    mfcc_delta TEXT, -- JSON array of MFCC delta coefficients
    mfcc_delta2 TEXT, -- JSON array of MFCC delta-delta coefficients
    
    -- MusiCNN features (original)
    embedding TEXT, -- JSON array of MusiCNN embedding
    tags TEXT, -- JSON object of MusiCNN tags
    
    -- Chroma features (original)
    chroma_mean TEXT, -- JSON array of chroma means
    chroma_std TEXT, -- JSON array of chroma standard deviations
    
    -- =============================================================================
    -- TIMBRE ANALYSIS
    -- =============================================================================
    timbre_brightness REAL, -- Brightness of timbre
    timbre_warmth REAL, -- Warmth of timbre
    timbre_hardness REAL, -- Hardness of timbre
    timbre_depth REAL, -- Depth of timbre
    
    -- =============================================================================
    -- PERCEPTUAL FEATURES (Spotify-style)
    -- =============================================================================
    acousticness REAL, -- Acoustic vs electronic nature
    instrumentalness REAL, -- Instrumental vs vocal content
    speechiness REAL, -- Presence of speech
    valence REAL, -- Positivity/negativity
    liveness REAL, -- Presence of live audience
    popularity REAL, -- Popularity score
    
    -- =============================================================================
    -- ADVANCED AUDIO FEATURES
    -- =============================================================================
    zero_crossing_rate REAL, -- Zero crossing rate
    root_mean_square REAL, -- RMS energy
    peak_amplitude REAL, -- Peak amplitude
    crest_factor REAL, -- Crest factor
    signal_to_noise_ratio REAL, -- SNR
    
    -- =============================================================================
    -- MUSICAL STRUCTURE ANALYSIS
    -- =============================================================================
    intro_duration REAL, -- Duration of intro section
    verse_duration REAL, -- Duration of verse sections
    chorus_duration REAL, -- Duration of chorus sections
    bridge_duration REAL, -- Duration of bridge sections
    outro_duration REAL, -- Duration of outro section
    section_boundaries TEXT, -- JSON array of section timestamps
    repetition_rate REAL, -- Rate of musical repetition
    
    -- =============================================================================
    -- AUDIO QUALITY METRICS
    -- =============================================================================
    bitrate_quality REAL, -- Quality based on bitrate
    sample_rate_quality REAL, -- Quality based on sample rate
    encoding_quality REAL, -- Encoding quality score
    compression_artifacts REAL, -- Presence of compression artifacts
    clipping_detection REAL, -- Detection of audio clipping
    
    -- =============================================================================
    -- GENRE-SPECIFIC FEATURES
    -- =============================================================================
    electronic_elements REAL, -- Electronic music elements
    classical_period TEXT, -- Classical period classification
    jazz_style TEXT, -- Jazz style classification
    rock_subgenre TEXT, -- Rock subgenre classification
    folk_style TEXT, -- Folk style classification
    
    -- =============================================================================
    -- ADDITIONAL METADATA FIELDS (Original)
    -- =============================================================================
    bitrate INTEGER,
    sample_rate INTEGER,
    channels INTEGER,
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
    tempo TEXT,
    length TEXT,
    replaygain_track_gain REAL,
    replaygain_album_gain REAL,
    replaygain_track_peak REAL,
    replaygain_album_peak REAL
);

-- Tags table for external API data and enrichment
CREATE TABLE tags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    track_id INTEGER NOT NULL,
    source TEXT NOT NULL, -- 'musicbrainz', 'lastfm', 'spotify', 'user'
    tag_name TEXT NOT NULL,
    tag_value TEXT,
    confidence REAL DEFAULT 1.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (track_id) REFERENCES tracks(id) ON DELETE CASCADE,
    UNIQUE(track_id, source, tag_name, tag_value)
);

-- Playlists table
CREATE TABLE playlists (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    generation_method TEXT, -- 'manual', 'auto', 'hybrid'
    generation_params TEXT, -- JSON parameters used for generation
    track_count INTEGER DEFAULT 0,
    total_duration REAL DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Playlist tracks junction table with ordering
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

-- Analysis cache for failed/partial analyses
CREATE TABLE analysis_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path TEXT UNIQUE NOT NULL,
    filename TEXT NOT NULL,
    analysis_data TEXT, -- JSON blob of partial analysis
    status TEXT DEFAULT 'failed', -- 'failed', 'partial', 'success'
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    last_retry_date TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Discovery cache for file system scanning
CREATE TABLE discovery_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    directory_path TEXT NOT NULL,
    file_count INTEGER DEFAULT 0,
    last_scan_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    scan_duration REAL DEFAULT 0,
    status TEXT DEFAULT 'completed', -- 'completed', 'failed', 'in_progress'
    error_message TEXT,
    
    UNIQUE(directory_path)
);

-- General cache table for API responses and computed data
CREATE TABLE cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cache_key TEXT UNIQUE NOT NULL,
    cache_value TEXT, -- JSON serialized data
    cache_type TEXT DEFAULT 'general', -- 'api_response', 'computed', 'statistics'
    expires_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Statistics table for web UI dashboards
CREATE TABLE statistics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    category TEXT NOT NULL, -- 'tracks', 'playlists', 'analysis', 'discovery'
    metric_name TEXT NOT NULL,
    metric_value REAL,
    metric_data TEXT, -- JSON for complex metrics
    date_recorded TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(category, metric_name, date_recorded)
);

-- =============================================================================
-- PERFORMANCE INDEXES FOR WEB UI
-- =============================================================================

-- Music lookup indexes (highest priority for web UI)
CREATE INDEX idx_tracks_artist ON tracks(artist);
CREATE INDEX idx_tracks_title ON tracks(title);
CREATE INDEX idx_tracks_artist_title ON tracks(artist, title);
CREATE INDEX idx_tracks_album ON tracks(album);
CREATE INDEX idx_tracks_genre ON tracks(genre);
CREATE INDEX idx_tracks_year ON tracks(year);
CREATE INDEX idx_tracks_analysis_date ON tracks(analysis_date);
CREATE INDEX idx_tracks_discovery_date ON tracks(discovery_date);

-- File system indexes
CREATE INDEX idx_tracks_file_path ON tracks(file_path);
CREATE INDEX idx_tracks_file_hash ON tracks(file_hash);

-- Basic audio feature indexes for playlist generation
CREATE INDEX idx_tracks_bpm ON tracks(bpm);
CREATE INDEX idx_tracks_key ON tracks(key);
CREATE INDEX idx_tracks_loudness ON tracks(loudness);
CREATE INDEX idx_tracks_danceability ON tracks(danceability);
CREATE INDEX idx_tracks_energy ON tracks(energy);
CREATE INDEX idx_tracks_duration ON tracks(duration);

-- Advanced audio feature indexes
CREATE INDEX idx_tracks_tempo_confidence ON tracks(tempo_confidence);
CREATE INDEX idx_tracks_rhythm_complexity ON tracks(rhythm_complexity);
CREATE INDEX idx_tracks_harmonic_complexity ON tracks(harmonic_complexity);
CREATE INDEX idx_tracks_acousticness ON tracks(acousticness);
CREATE INDEX idx_tracks_instrumentalness ON tracks(instrumentalness);
CREATE INDEX idx_tracks_valence ON tracks(valence);
CREATE INDEX idx_tracks_liveness ON tracks(liveness);
CREATE INDEX idx_tracks_zero_crossing_rate ON tracks(zero_crossing_rate);
CREATE INDEX idx_tracks_spectral_flux ON tracks(spectral_flux);
CREATE INDEX idx_tracks_timbre_brightness ON tracks(timbre_brightness);
CREATE INDEX idx_tracks_timbre_warmth ON tracks(timbre_warmth);

-- Composite indexes for common queries
CREATE INDEX idx_tracks_artist_album ON tracks(artist, album);
CREATE INDEX idx_tracks_genre_year ON tracks(genre, year);
CREATE INDEX idx_tracks_bpm_energy ON tracks(bpm, energy);
CREATE INDEX idx_tracks_key_mode ON tracks(key, mode);
CREATE INDEX idx_tracks_acousticness_instrumentalness ON tracks(acousticness, instrumentalness);
CREATE INDEX idx_tracks_valence_energy ON tracks(valence, energy);
CREATE INDEX idx_tracks_tempo_rhythm_complexity ON tracks(bpm, rhythm_complexity);
CREATE INDEX idx_tracks_key_harmonic_complexity ON tracks(key, harmonic_complexity);

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
-- VIEWS FOR WEB UI PERFORMANCE
-- =============================================================================

-- Complete track view with all audio analysis data
CREATE VIEW track_complete AS
SELECT 
    t.*,
    GROUP_CONCAT(DISTINCT tag.tag_name) as all_tags,
    GROUP_CONCAT(DISTINCT tag.source) as tag_sources
FROM tracks t
LEFT JOIN tags tag ON t.id = tag.track_id
GROUP BY t.id;

-- Track summary for web UI lists (basic features)
CREATE VIEW track_summary AS
SELECT 
    id, file_path, filename, title, artist, album, genre, year,
    duration, bpm, key, mode, loudness, danceability, energy,
    analysis_date, discovery_date, long_audio_category
FROM tracks;

-- Advanced track view with comprehensive audio features
CREATE VIEW audio_analysis_complete AS
SELECT 
    id, file_path, filename, title, artist, album, genre, year, duration,
    -- Basic audio features
    bpm, key, mode, loudness, danceability, energy,
    -- Rhythm analysis
    tempo_confidence, tempo_strength, rhythm_pattern, rhythm_complexity,
    beat_positions, onset_times,
    -- Harmonic analysis
    harmonic_complexity, chord_progression, harmonic_centroid, harmonic_contrast,
    chord_changes, key_scale_notes, key_chord_progression, modulation_points,
    tonal_centroid,
    -- Spectral analysis
    spectral_centroid, spectral_flatness, spectral_rolloff, spectral_bandwidth,
    spectral_contrast_mean, spectral_contrast_std, spectral_flux, spectral_crest,
    spectral_decrease, spectral_entropy, spectral_kurtosis, spectral_skewness,
    spectral_slope, spectral_rolloff_85, spectral_rolloff_95,
    -- Timbre analysis
    timbre_brightness, timbre_warmth, timbre_hardness, timbre_depth,
    mfcc_coefficients, mfcc_delta, mfcc_delta2,
    -- Perceptual features
    acousticness, instrumentalness, speechiness, valence, liveness, popularity,
    -- Advanced audio features
    zero_crossing_rate, root_mean_square, peak_amplitude, crest_factor,
    signal_to_noise_ratio, dynamic_complexity, loudness_range, dynamic_range,
    -- Musical structure
    intro_duration, verse_duration, chorus_duration, bridge_duration, outro_duration,
    section_boundaries, repetition_rate,
    -- Audio quality
    bitrate_quality, sample_rate_quality, encoding_quality, compression_artifacts,
    clipping_detection,
    -- Genre-specific
    electronic_elements, classical_period, jazz_style, rock_subgenre, folk_style,
    -- Analysis metadata
    analysis_type, analyzed, long_audio_category, analysis_date
FROM tracks;

-- Playlist generation features view
CREATE VIEW playlist_features AS
SELECT 
    id, file_path, title, artist, album, genre, year, duration,
    -- Core features for playlist generation
    bpm, key, mode, loudness, danceability, energy,
    -- Advanced features for sophisticated playlists
    valence, acousticness, instrumentalness, liveness,
    tempo_confidence, rhythm_complexity, harmonic_complexity,
    timbre_brightness, timbre_warmth,
    spectral_flux, spectral_entropy,
    -- Quality indicators
    encoding_quality, compression_artifacts
FROM tracks
WHERE analyzed = TRUE;

-- Playlist summary with track count
CREATE VIEW playlist_summary AS
SELECT 
    p.*,
    COUNT(pt.track_id) as actual_track_count,
    SUM(t.duration) as actual_total_duration
FROM playlists p
LEFT JOIN playlist_tracks pt ON p.id = pt.playlist_id
LEFT JOIN tracks t ON pt.track_id = t.id
GROUP BY p.id;

-- Genre analysis view
CREATE VIEW genre_analysis AS
SELECT 
    genre,
    COUNT(*) as track_count,
    AVG(bpm) as avg_bpm,
    AVG(energy) as avg_energy,
    AVG(danceability) as avg_danceability,
    AVG(loudness) as avg_loudness,
    AVG(valence) as avg_valence,
    AVG(acousticness) as avg_acousticness,
    AVG(instrumentalness) as avg_instrumentalness,
    AVG(harmonic_complexity) as avg_harmonic_complexity,
    AVG(rhythm_complexity) as avg_rhythm_complexity,
    AVG(timbre_brightness) as avg_timbre_brightness,
    AVG(spectral_flux) as avg_spectral_flux
FROM tracks
WHERE genre IS NOT NULL AND analyzed = TRUE
GROUP BY genre;

-- Statistics summary for web UI dashboard
CREATE VIEW statistics_summary AS
SELECT 
    category,
    metric_name,
    AVG(metric_value) as avg_value,
    MAX(metric_value) as max_value,
    MIN(metric_value) as min_value,
    COUNT(*) as data_points,
    MAX(date_recorded) as last_updated
FROM statistics
GROUP BY category, metric_name;

-- =============================================================================
-- TRIGGERS FOR DATA INTEGRITY AND CACHING
-- =============================================================================

-- Update timestamp triggers
CREATE TRIGGER update_tracks_timestamp 
    AFTER UPDATE ON tracks
    FOR EACH ROW
BEGIN
    UPDATE tracks SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

CREATE TRIGGER update_playlists_timestamp 
    AFTER UPDATE ON playlists
    FOR EACH ROW
BEGIN
    UPDATE playlists SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

-- Auto-update playlist track count
CREATE TRIGGER update_playlist_track_count_insert
    AFTER INSERT ON playlist_tracks
    FOR EACH ROW
BEGIN
    UPDATE playlists SET track_count = (
        SELECT COUNT(*) FROM playlist_tracks WHERE playlist_id = NEW.playlist_id
    ) WHERE id = NEW.playlist_id;
END;

CREATE TRIGGER update_playlist_track_count_delete
    AFTER DELETE ON playlist_tracks
    FOR EACH ROW
BEGIN
    UPDATE playlists SET track_count = (
        SELECT COUNT(*) FROM playlist_tracks WHERE playlist_id = OLD.playlist_id
    ) WHERE id = OLD.playlist_id;
END;

-- Auto-cleanup expired cache entries
CREATE TRIGGER cleanup_expired_cache
    AFTER INSERT ON cache
    FOR EACH ROW
BEGIN
    DELETE FROM cache WHERE expires_at IS NOT NULL AND expires_at < CURRENT_TIMESTAMP;
END; 