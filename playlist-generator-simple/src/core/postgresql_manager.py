"""
PostgreSQL Database Manager for Playlist Generator.

Modern database manager optimized for web UI and playlist generation.
Replaces SQLite with PostgreSQL for better performance and concurrency.
"""

import json
import os
import psycopg2
import psycopg2.extras
from psycopg2.pool import ThreadedConnectionPool
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from contextlib import contextmanager

from .logging_setup import get_logger, log_universal
from .config_loader import config_loader

logger = get_logger('playlista.postgresql')


class PostgreSQLManager:
    """
    PostgreSQL database manager for playlist generator.
    
    Features:
    - Connection pooling for web applications
    - Optimized for music analysis data storage
    - Fast playlist generation queries
    - Vector similarity for recommendations
    - Proper transaction handling
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize PostgreSQL manager."""
        self.config = config or config_loader.get_config()
        self.pool = None
        self._initialize_connection_pool()
        
        log_universal('INFO', 'Database', 'PostgreSQL manager initialized')
    
    def _initialize_connection_pool(self):
        """Initialize PostgreSQL connection pool."""
        try:
            # Database connection parameters
            db_config = {
                'host': self.config.get('POSTGRES_HOST', 'localhost'),
                'port': self.config.get('POSTGRES_PORT', 5432),
                'database': self.config.get('POSTGRES_DB', 'playlista'),
                'user': self.config.get('POSTGRES_USER', 'playlista'),
                'password': self.config.get('POSTGRES_PASSWORD', ''),
            }
            
            # Connection pool settings
            min_conn = self.config.get('POSTGRES_MIN_CONNECTIONS', 2)
            max_conn = self.config.get('POSTGRES_MAX_CONNECTIONS', 10)
            
            self.pool = ThreadedConnectionPool(
                min_conn, max_conn,
                **db_config
            )
            
            # Test connection and initialize schema if needed
            self._initialize_database()
            
            log_universal('INFO', 'Database', f'Connection pool created: {min_conn}-{max_conn} connections')
            
        except Exception as e:
            log_universal('ERROR', 'Database', f'Failed to initialize PostgreSQL: {str(e)}')
            raise
    
    @contextmanager
    def get_connection(self):
        """Get connection from pool with automatic cleanup."""
        conn = None
        try:
            conn = self.pool.getconn()
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            log_universal('ERROR', 'Database', f'Database operation failed: {str(e)}')
            raise
        finally:
            if conn:
                self.pool.putconn(conn)
    
    def _initialize_database(self):
        """Initialize database schema if needed."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Check if tables exist
                cursor.execute("""
                    SELECT table_name FROM information_schema.tables 
                    WHERE table_schema = 'public' AND table_name = 'tracks';
                """)
                
                if not cursor.fetchone():
                    log_universal('INFO', 'Database', 'Initializing database schema...')
                    self._create_schema(conn)
                else:
                    log_universal('DEBUG', 'Database', 'Schema already exists')
                    
        except Exception as e:
            log_universal('ERROR', 'Database', f'Schema initialization failed: {str(e)}')
            raise
    
    def _create_schema(self, conn):
        """Create database schema from SQL file."""
        import os
        schema_file = os.path.join(
            os.path.dirname(__file__), '..', '..', 'database', 'postgresql_schema.sql'
        )
        
        try:
            with open(schema_file, 'r') as f:
                schema_sql = f.read()
            
            cursor = conn.cursor()
            cursor.execute(schema_sql)
            conn.commit()
            
            log_universal('INFO', 'Database', 'Schema created successfully')
            
        except Exception as e:
            log_universal('ERROR', 'Database', f'Schema creation failed: {str(e)}')
            raise
    
    # =========================================================================
    # TRACK MANAGEMENT
    # =========================================================================
    
    def save_track_analysis(self, file_path: str, filename: str, file_size_bytes: int, 
                           file_hash: str, metadata: Dict[str, Any], 
                           analysis_data: Dict[str, Any], **kwargs) -> bool:
        """
        Save complete track analysis to database.
        
        Args:
            file_path: Full path to audio file
            filename: Just the filename
            file_size_bytes: File size in bytes
            file_hash: Hash of file content
            metadata: Basic metadata (mutagen)
            analysis_data: Complete analysis results
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
                
                # Extract features for main tracks table
                tempo = self._extract_tempo(analysis_data)
                key = analysis_data.get('key')
                mode = analysis_data.get('scale', analysis_data.get('mode'))
                key_confidence = analysis_data.get('key_confidence', analysis_data.get('key_strength'))
                
                # Always define musicnn_tags first
                musicnn_tags = analysis_data.get('musicnn_tags', {})
                
                # Debug: Log what we received in analysis_data
                log_universal('DEBUG', 'Database', f'Analysis data keys: {list(analysis_data.keys())}')
                log_universal('DEBUG', 'Database', f'Energy value: {analysis_data.get("energy")}, Valence: {analysis_data.get("valence")}')
                log_universal('DEBUG', 'Database', f'MusiCNN tags count: {len(musicnn_tags)}')
                
                # Derive playlist features - first try direct features, then MusiCNN derivation
                # All our new methods include both Essentia and MusiCNN, so check for direct features first
                if (analysis_data.get('energy') is not None or 
                    analysis_data.get('danceability') is not None or
                    analysis_data.get('valence') is not None):
                    # Use direct features from analysis (Essentia + MusiCNN hybrid)
                    features = {
                        'energy': analysis_data.get('energy'),
                        'danceability': analysis_data.get('danceability'),
                        'valence': analysis_data.get('valence'),
                        'acousticness': analysis_data.get('acousticness'),
                        'instrumentalness': analysis_data.get('instrumentalness'),
                        'liveness': analysis_data.get('liveness'),
                        'speechiness': analysis_data.get('speechiness'),
                        'loudness': analysis_data.get('loudness')
                    }
                    log_universal('DEBUG', 'Database', f'Using direct features from analysis: {features}')
                else:
                    # Fallback: derive from MusiCNN tags if direct features not available
                    features = self._derive_playlist_features(musicnn_tags)
                    log_universal('DEBUG', 'Database', f'Derived features from MusiCNN tags: {features}')
                
                # Insert/update track
                cursor.execute("""
                    INSERT INTO tracks (
                        file_path, file_hash, filename, file_size_bytes,
                        title, artist, album, genre, year, duration_seconds,
                        bitrate, sample_rate, channels,
                        tempo, key, mode, key_confidence,
                        energy, danceability, valence, acousticness,
                        instrumentalness, liveness, speechiness, loudness,
                        analysis_completed, analysis_date, analysis_method,
                        content_type, content_subtype, content_confidence,
                        content_features, estimated_track_count, content_description
                    ) VALUES (
                        %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s,
                        %s, %s, %s,
                        %s, %s, %s, %s,
                        %s, %s, %s, %s,
                        %s, %s, %s, %s,
                        %s, %s, %s,
                        %s, %s, %s,
                        %s, %s, %s
                    )
                    ON CONFLICT (file_path) DO UPDATE SET
                        file_hash = EXCLUDED.file_hash,
                        filename = EXCLUDED.filename,
                        title = EXCLUDED.title,
                        artist = EXCLUDED.artist,
                        album = EXCLUDED.album,
                        tempo = EXCLUDED.tempo,
                        key = EXCLUDED.key,
                        mode = EXCLUDED.mode,
                        energy = EXCLUDED.energy,
                        danceability = EXCLUDED.danceability,
                        valence = EXCLUDED.valence,
                        analysis_completed = EXCLUDED.analysis_completed,
                        analysis_date = EXCLUDED.analysis_date,
                        updated_at = NOW()
                    RETURNING id;
                """, (
                    # Basic info
                    file_path, file_hash, filename, file_size_bytes,
                    # Metadata
                    self._safe_string(metadata.get('title')), 
                    self._safe_string(metadata.get('artist')), 
                    self._safe_string(metadata.get('album')),
                    json.dumps(metadata.get('genre', [])) if metadata.get('genre') else None,
                    self._extract_year(metadata.get('date')), metadata.get('duration_seconds'),
                    metadata.get('bitrate'), metadata.get('sample_rate'), metadata.get('channels'),
                    # Analysis features
                    tempo, key, mode, key_confidence,
                    # Derived features
                    features.get('energy'), features.get('danceability'), features.get('valence'),
                    features.get('acousticness'), features.get('instrumentalness'),
                    features.get('liveness'), features.get('speechiness'), features.get('loudness'),
                    # Analysis metadata
                    True, datetime.now(), analysis_data.get('analysis_method', 'unknown'),
                    # Content classification (for large files)
                    analysis_data.get('content_type'), analysis_data.get('content_subtype'),
                    analysis_data.get('content_confidence'),
                    json.dumps(analysis_data.get('content_features', [])) if analysis_data.get('content_features') else None,
                    analysis_data.get('estimated_track_count'), analysis_data.get('content_description')
                ))
                
                track_id = cursor.fetchone()[0]
                
                # Save complete analysis data
                self._save_analysis_details(cursor, track_id, analysis_data, file_hash)
                
                # Extract and save music tags
                self._save_music_tags(cursor, track_id, musicnn_tags)
                
                conn.commit()
                log_universal('DEBUG', 'Database', f'Saved analysis for: {filename}')
                return True
                
        except Exception as e:
            log_universal('ERROR', 'Database', f'Failed to save analysis for {filename}: {str(e)}')
            return False
    
    def _extract_tempo(self, analysis_data: Dict[str, Any]) -> Optional[float]:
        """Extract tempo from analysis data (handles both formats)."""
        # Multi-segment analysis
        if 'tempo_mean' in analysis_data:
            return analysis_data['tempo_mean']
        # Single analysis
        elif 'tempo' in analysis_data:
            return analysis_data['tempo']
        return None
    
    def _extract_year(self, date_value: Any) -> Optional[int]:
        """Extract year as integer from various date formats."""
        if not date_value:
            return None
            
        try:
            # Convert to string if not already
            date_str = str(date_value).strip()
            
            # Handle various date formats
            if '-' in date_str:
                # Format: "2025-05-09" or "1996-02-13"
                year_str = date_str.split('-')[0]
            elif '/' in date_str:
                # Format: "05/09/2025" or "2025/05/09"
                parts = date_str.split('/')
                # Assume 4-digit part is year
                year_str = max(parts, key=len) if any(len(p) == 4 for p in parts) else parts[-1]
            elif len(date_str) == 4 and date_str.isdigit():
                # Just a year: "2025"
                year_str = date_str
            else:
                # Try to extract 4-digit year from string
                import re
                year_match = re.search(r'\b(19|20)\d{2}\b', date_str)
                if year_match:
                    year_str = year_match.group()
                else:
                    return None
            
            # Convert to integer and validate range
            year = int(year_str)
            if 1900 <= year <= 2030:  # Reasonable range for music
                return year
            else:
                return None
                
        except (ValueError, AttributeError, IndexError):
            # If parsing fails, return None
            return None
    
    def _derive_playlist_features(self, musicnn_tags: Dict[str, float]) -> Dict[str, float]:
        """Derive playlist features from MusiCNN tags."""
        if not musicnn_tags:
            return {}
        
        # Map MusiCNN tags to playlist features (0-1 scale)
        features = {}
        
        # Energy: energetic, aggressive, dance vs peaceful, calm
        energy_positive = ['energetic', 'aggressive', 'dance', 'metal', 'rock']
        energy_negative = ['peaceful', 'calm', 'ambient', 'chillout']
        features['energy'] = self._calculate_feature_score(musicnn_tags, energy_positive, energy_negative)
        
        # Danceability: dance, electronic, pop vs classical, folk
        dance_positive = ['dance', 'electronic', 'pop', 'hip-hop']
        dance_negative = ['classical', 'folk', 'acoustic', 'jazz']
        features['danceability'] = self._calculate_feature_score(musicnn_tags, dance_positive, dance_negative)
        
        # Valence (happiness): happy, uplifting vs sad, dark
        valence_positive = ['happy', 'uplifting', 'cheerful']
        valence_negative = ['sad', 'dark', 'melancholic', 'emotional']
        features['valence'] = self._calculate_feature_score(musicnn_tags, valence_positive, valence_negative)
        
        # Acousticness: acoustic, folk, country vs electronic, dance
        acoustic_positive = ['acoustic', 'folk', 'country', 'singer-songwriter']
        acoustic_negative = ['electronic', 'dance', 'techno', 'house']
        features['acousticness'] = self._calculate_feature_score(musicnn_tags, acoustic_positive, acoustic_negative)
        
        # Instrumentalness: instrumental vs vocal
        features['instrumentalness'] = musicnn_tags.get('instrumental', 0.5)
        
        # Liveness: live, concert vs studio
        features['liveness'] = musicnn_tags.get('live', 0.1)  # Default low
        
        # Speechiness: spoken word, rap vs musical
        features['speechiness'] = musicnn_tags.get('spoken', 0.1)  # Default low
        
        # Loudness: metal, rock vs ambient, classical
        loud_positive = ['metal', 'rock', 'punk', 'aggressive']
        loud_negative = ['ambient', 'classical', 'peaceful', 'quiet']
        features['loudness'] = self._calculate_feature_score(musicnn_tags, loud_positive, loud_negative)
        
        return features
    
    def _calculate_feature_score(self, tags: Dict[str, float], positive_tags: List[str], 
                                negative_tags: List[str]) -> float:
        """Calculate feature score from positive and negative tag influences."""
        positive_score = sum(tags.get(tag, 0) for tag in positive_tags) / len(positive_tags)
        negative_score = sum(tags.get(tag, 0) for tag in negative_tags) / len(negative_tags)
        
        # Combine scores (positive - negative, normalized to 0-1)
        score = (positive_score - negative_score + 1) / 2
        return max(0, min(1, score))  # Clamp to 0-1
    
    def _save_analysis_details(self, cursor, track_id: int, analysis_data: Dict[str, Any], 
                              cache_key: str):
        """Save complete analysis data to track_analysis table."""
        try:
            # Prepare analysis components
            essentia_data = self._extract_essentia_data(analysis_data)
            
            # Only include musicnn_embeddings if they actually exist
            musicnn_embeddings = analysis_data.get('musicnn_embeddings')
            if not (musicnn_embeddings and isinstance(musicnn_embeddings, list) and len(musicnn_embeddings) > 0):
                # No valid embeddings - skip MusiCNN data entirely
                musicnn_embeddings = None
            
            cursor.execute("""
                INSERT INTO track_analysis (
                    track_id, essentia_rhythm, essentia_spectral, essentia_harmonic,
                    essentia_mfcc, essentia_other, musicnn_tags, musicnn_embeddings, 
                    musicnn_genre, musicnn_mood, musicnn_top_tags, musicnn_confidence,
                    segments_analyzed, segment_times, processing_time_seconds, cache_key
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (track_id) DO UPDATE SET
                    essentia_rhythm = EXCLUDED.essentia_rhythm,
                    essentia_spectral = EXCLUDED.essentia_spectral,
                    essentia_harmonic = EXCLUDED.essentia_harmonic,
                    essentia_mfcc = EXCLUDED.essentia_mfcc,
                    essentia_other = EXCLUDED.essentia_other,
                    musicnn_tags = EXCLUDED.musicnn_tags,
                    musicnn_embeddings = EXCLUDED.musicnn_embeddings,
                    musicnn_genre = EXCLUDED.musicnn_genre,
                    musicnn_mood = EXCLUDED.musicnn_mood,
                    musicnn_top_tags = EXCLUDED.musicnn_top_tags,
                    cache_key = EXCLUDED.cache_key;
            """, (
                track_id,
                json.dumps(essentia_data.get('rhythm')) if essentia_data.get('rhythm') else None,
                json.dumps(essentia_data.get('spectral')) if essentia_data.get('spectral') else None,
                json.dumps(essentia_data.get('harmonic')) if essentia_data.get('harmonic') else None,
                json.dumps(essentia_data.get('mfcc')) if essentia_data.get('mfcc') else None,
                json.dumps(essentia_data.get('other')) if essentia_data.get('other') else None,
                json.dumps(analysis_data.get('musicnn_tags')) if analysis_data.get('musicnn_tags') else None,
                musicnn_embeddings,
                analysis_data.get('musicnn_genre'),
                analysis_data.get('musicnn_mood'),
                json.dumps(analysis_data.get('musicnn_top_tags')) if analysis_data.get('musicnn_top_tags') else None,
                analysis_data.get('musicnn_confidence'),
                analysis_data.get('segments_analyzed'),
                json.dumps(analysis_data.get('segment_times')) if analysis_data.get('segment_times') else None,
                analysis_data.get('processing_time'),
                cache_key
            ))
            
        except Exception as e:
            log_universal('WARNING', 'Database', f'Failed to save analysis details: {str(e)}')
    
    def _extract_essentia_data(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and organize ALL Essentia features."""
        essentia = {
            'rhythm': {},
            'spectral': {},
            'harmonic': {},
            'mfcc': {},
            'other': {}
        }
        
        # Comprehensive rhythm features
        rhythm_keys = [
            'tempo', 'tempo_confidence', 'tempo_mean', 'tempo_std', 'tempo_median',
            'beats_confidence', 'rhythm_confidence', 'tempo_strength', 'rhythm_pattern',
            'beat_positions', 'onset_times', 'rhythm_complexity', 'bpm_estimates', 
            'bpm_intervals', 'external_bpm'
        ]
        for key in rhythm_keys:
            if key in analysis_data:
                essentia['rhythm'][key] = analysis_data[key]
        
        # Comprehensive spectral features  
        spectral_keys = [
            'spectral_centroid_mean', 'spectral_centroid_std', 'spectral_centroid',
            'spectral_flatness', 'spectral_rolloff', 'spectral_bandwidth',
            'spectral_contrast_mean', 'spectral_contrast_std', 'spectral_flux',
            'spectral_entropy', 'spectral_crest', 'spectral_decrease',
            'spectral_kurtosis', 'spectral_skewness', 'loudness', 'loudness_range',
            'dynamic_complexity', 'dynamic_range', 'zero_crossing_rate',
            'root_mean_square', 'peak_amplitude', 'crest_factor',
            'signal_to_noise_ratio'
        ]
        for key in spectral_keys:
            if key in analysis_data:
                essentia['spectral'][key] = analysis_data[key]
        
        # Comprehensive harmonic features
        harmonic_keys = [
            'key', 'scale', 'mode', 'key_strength', 'key_confidence', 'scale_confidence',
            'key_scale_notes', 'key_chord_progression', 'modulation_points',
            'tonal_centroid', 'harmonic_complexity', 'chord_progression',
            'harmonic_centroid', 'harmonic_contrast', 'chord_changes'
        ]
        for key in harmonic_keys:
            if key in analysis_data:
                essentia['harmonic'][key] = analysis_data[key]
        
        # MFCC and chroma features
        mfcc_keys = [
            'mfcc_coefficients', 'mfcc_bands', 'mfcc_std', 'mfcc_delta', 'mfcc_delta2',
            'chroma_mean', 'chroma_std'
        ]
        for key in mfcc_keys:
            if key in analysis_data:
                essentia['mfcc'][key] = analysis_data[key]
        
        # Other audio features
        other_keys = [
            'energy', 'danceability', 'valence', 'acousticness', 'instrumentalness',
            'speechiness', 'liveness', 'duration', 'sample_rate', 'segments_analyzed',
            'analysis_strategy', 'method'
        ]
        for key in other_keys:
            if key in analysis_data:
                essentia['other'][key] = analysis_data[key]
        
        return essentia
    
    def _save_music_tags(self, cursor, track_id: int, musicnn_tags: Dict[str, float]):
        """Save music tags to normalized tag tables."""
        try:
            # Clear existing tags for this track
            cursor.execute("DELETE FROM track_tags WHERE track_id = %s", (track_id,))
            
            # Insert new tags
            for tag_name, confidence in musicnn_tags.items():
                if confidence > 0.1:  # Only save tags with meaningful confidence
                    # Get or create tag
                    cursor.execute("""
                        INSERT INTO music_tags (name, category) VALUES (%s, 'musicnn')
                        ON CONFLICT (name) DO NOTHING
                        RETURNING id;
                    """, (tag_name,))
                    
                    result = cursor.fetchone()
                    if result:
                        tag_id = result[0]
                    else:
                        cursor.execute("SELECT id FROM music_tags WHERE name = %s", (tag_name,))
                        tag_id = cursor.fetchone()[0]
                    
                    # Link tag to track
                    cursor.execute("""
                        INSERT INTO track_tags (track_id, tag_id, confidence, source)
                        VALUES (%s, %s, %s, 'musicnn')
                    """, (track_id, tag_id, confidence))
                    
        except Exception as e:
            log_universal('WARNING', 'Database', f'Failed to save music tags: {str(e)}')
    
    # =========================================================================
    # PLAYLIST GENERATION QUERIES
    # =========================================================================
    
    def find_similar_tracks(self, track_id: int, limit: int = 20) -> List[Dict[str, Any]]:
        """Find tracks similar to given track using vector similarity."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
                
                cursor.execute("""
                    SELECT 
                        t.id, t.title, t.artist, t.album, t.tempo, t.key,
                        t.energy, t.valence, t.danceability,
                        ta.musicnn_embeddings <-> ref_ta.musicnn_embeddings as similarity
                    FROM tracks t
                    JOIN track_analysis ta ON t.id = ta.track_id
                    CROSS JOIN track_analysis ref_ta
                    WHERE ref_ta.track_id = %s 
                    AND t.id != %s
                    AND ta.musicnn_embeddings IS NOT NULL
                    ORDER BY similarity
                    LIMIT %s;
                """, (track_id, track_id, limit))
                
                return [dict(row) for row in cursor.fetchall()]
                
        except Exception as e:
            log_universal('ERROR', 'Database', f'Similarity search failed: {str(e)}')
            return []
    
    def generate_playlist_by_features(self, tempo_range: Tuple[float, float] = None,
                                    key: str = None, energy_range: Tuple[float, float] = None,
                                    valence_range: Tuple[float, float] = None,
                                    limit: int = 20) -> List[Dict[str, Any]]:
        """Generate playlist based on musical features."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
                
                conditions = ["analysis_completed = true"]
                params = []
                
                if tempo_range:
                    conditions.append("tempo BETWEEN %s AND %s")
                    params.extend(tempo_range)
                
                if key:
                    conditions.append("key = %s")
                    params.append(key)
                
                if energy_range:
                    conditions.append("energy BETWEEN %s AND %s")
                    params.extend(energy_range)
                
                if valence_range:
                    conditions.append("valence BETWEEN %s AND %s")
                    params.extend(valence_range)
                
                params.append(limit)
                
                query = f"""
                    SELECT id, title, artist, album, tempo, key, mode,
                           energy, valence, danceability, duration_seconds
                    FROM tracks
                    WHERE {' AND '.join(conditions)}
                    ORDER BY random()
                    LIMIT %s;
                """
                
                cursor.execute(query, params)
                return [dict(row) for row in cursor.fetchall()]
                
        except Exception as e:
            log_universal('ERROR', 'Database', f'Feature-based playlist generation failed: {str(e)}')
            return []
    
    def search_tracks(self, query: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Search tracks by title, artist, or album."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
                
                cursor.execute("""
                    SELECT id, title, artist, album, duration_seconds, tempo, key
                    FROM tracks
                    WHERE title ILIKE %s OR artist ILIKE %s OR album ILIKE %s
                    ORDER BY 
                        CASE 
                            WHEN title ILIKE %s THEN 1
                            WHEN artist ILIKE %s THEN 2
                            ELSE 3
                        END,
                        similarity(title || ' ' || artist, %s) DESC
                    LIMIT %s;
                """, (f'%{query}%', f'%{query}%', f'%{query}%', 
                     f'%{query}%', f'%{query}%', query, limit))
                
                return [dict(row) for row in cursor.fetchall()]
                
        except Exception as e:
            log_universal('ERROR', 'Database', f'Track search failed: {str(e)}')
            return []
    
    # =========================================================================
    # CACHE/ANALYSIS RETRIEVAL METHODS
    # =========================================================================
    
    def get_analysis_result(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Get existing analysis result for a file path.
        
        Args:
            file_path: Full path to the audio file
            
        Returns:
            Analysis result dict if found, None otherwise
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
                
                # Query for track with analysis data
                cursor.execute("""
                    SELECT 
                        t.id, t.file_path, t.filename, t.file_size_bytes, t.file_hash,
                        t.title, t.artist, t.album, t.genre, t.year,
                        t.tempo, t.key, t.mode, t.time_signature,
                        t.energy, t.valence, t.danceability, t.acousticness,
                        t.instrumentalness, t.liveness, t.speechiness, t.loudness,
                        t.content_type, t.content_subtype, t.content_confidence,
                        t.content_features, t.estimated_track_count, t.content_description,
                        t.analysis_completed, t.created_at, t.updated_at,
                        t.analysis_method, ta.essentia_rhythm, ta.essentia_spectral,
                        ta.essentia_harmonic, ta.essentia_mfcc, ta.musicnn_tags,
                        ta.musicnn_embeddings, ta.segments_analyzed, ta.segment_times
                    FROM tracks t
                    LEFT JOIN track_analysis ta ON t.id = ta.track_id
                    WHERE t.file_path = %s AND t.analysis_completed = true;
                """, (file_path,))
                
                row = cursor.fetchone()
                if not row:
                    log_universal('DEBUG', 'Database', f'No analysis found for {os.path.basename(file_path)}')
                    return None
                
                # Convert to analysis result format expected by SingleAnalyzer
                result = {
                    'success': True,
                    'file_path': row['file_path'],
                    'filename': row['filename'],
                    'file_size_mb': row['file_size_bytes'] / (1024 * 1024) if row['file_size_bytes'] else 0,
                    'analysis_method': row['analysis_method'] or 'unknown',
                    'analysis_time': 0.0,  # Not stored separately
                    
                    # Metadata
                    'metadata': {
                        'title': row['title'],
                        'artist': row['artist'],
                        'album': row['album'],
                        'genre': self._safe_json_loads(row['genre'], []),
                        'year': row['year'],
                        'file_size_mb': row['file_size_bytes'] / (1024 * 1024) if row['file_size_bytes'] else 0
                    },
                    
                    # Audio features
                    'audio_features': {
                        'duration': 0.0,  # Could be calculated from other data
                        'tempo': row['tempo'],
                        'key': row['key'],
                        'mode': row['mode'],
                        'time_signature': row['time_signature'],
                        'energy': row['energy'],
                        'valence': row['valence'],
                        'danceability': row['danceability'],
                        'acousticness': row['acousticness'],
                        'instrumentalness': row['instrumentalness'],
                        'liveness': row['liveness'],
                        'speechiness': row['speechiness'],
                        'loudness': row['loudness']
                    },
                    
                    # Content classification (for large files)
                    'content_type': row['content_type'],
                    'content_subtype': row['content_subtype'],
                    'content_confidence': row['content_confidence'],
                    'content_features': self._safe_json_loads(row['content_features'], {}),
                    'estimated_track_count': row['estimated_track_count'],
                    'content_description': row['content_description'],
                    
                    # Advanced features (if available)
                    'essentia_features': {
                        'rhythm': self._safe_json_loads(row['essentia_rhythm'], {}),
                        'spectral': self._safe_json_loads(row['essentia_spectral'], {}),
                        'harmonic': self._safe_json_loads(row['essentia_harmonic'], {}),
                        'mfcc': self._safe_json_loads(row['essentia_mfcc'], {})
                    } if row['essentia_rhythm'] else {},
                    
                    'musicnn_features': {
                        'tags': self._safe_json_loads(row['musicnn_tags'], {}),
                        'embeddings': row['musicnn_embeddings'] if row['musicnn_embeddings'] else []
                    } if row['musicnn_tags'] else {}
                }
                
                log_universal('DEBUG', 'Database', f'Retrieved cached analysis for {row["filename"]}')
                return result
                
        except Exception as e:
            log_universal('WARNING', 'Database', f'Failed to get analysis result for {file_path}: {str(e)}')
            return None
    
    def _safe_json_loads(self, json_str: Any, default_value: Any) -> Any:
        """Safely parse JSON string with fallback to default value."""
        if not json_str:
            return default_value
            
        # Handle if json_str is a list (should not happen but being safe)
        if isinstance(json_str, list):
            if len(json_str) > 0:
                json_str = str(json_str[0])
            else:
                return default_value
        
        # Convert to string and check if empty
        json_str = str(json_str)
        if json_str.strip() == '':
            return default_value
            
        try:
            return json.loads(json_str)
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            log_universal('DEBUG', 'Database', f'JSON parse error: {str(e)}, using default')
            return default_value
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def get_track_count(self) -> int:
        """Get total number of analyzed tracks."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM tracks WHERE analysis_completed = true;")
                return cursor.fetchone()[0]
        except Exception as e:
            log_universal('ERROR', 'Database', f'Failed to get track count: {str(e)}')
            return 0
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                stats = {}
                
                # Track counts
                cursor.execute("SELECT COUNT(*) FROM tracks;")
                stats['total_tracks'] = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM tracks WHERE analysis_completed = true;")
                stats['analyzed_tracks'] = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM tracks WHERE analysis_completed = false;")
                stats['pending_tracks'] = cursor.fetchone()[0]
                
                # Analysis methods
                cursor.execute("""
                    SELECT analysis_method, COUNT(*) 
                    FROM tracks 
                    WHERE analysis_completed = true 
                    GROUP BY analysis_method;
                """)
                analysis_methods = dict(cursor.fetchall())
                stats['analysis_methods'] = analysis_methods
                
                # Database size info
                cursor.execute("""
                    SELECT 
                        pg_size_pretty(pg_database_size(current_database())) as db_size,
                        pg_size_pretty(pg_total_relation_size('tracks')) as tracks_size,
                        pg_size_pretty(pg_total_relation_size('track_analysis')) as analysis_size;
                """)
                size_info = cursor.fetchone()
                stats['database_size'] = size_info[0]
                stats['tracks_table_size'] = size_info[1] 
                stats['analysis_table_size'] = size_info[2]
                
                # Recent activity
                cursor.execute("""
                    SELECT COUNT(*) FROM tracks 
                    WHERE created_at >= NOW() - INTERVAL '24 hours';
                """)
                stats['tracks_added_today'] = cursor.fetchone()[0]
                
                return stats
                
        except Exception as e:
            log_universal('ERROR', 'Database', f'Failed to get statistics: {str(e)}')
            return {
                'total_tracks': 0,
                'analyzed_tracks': 0,
                'pending_tracks': 0,
                'analysis_methods': {},
                'database_size': 'Unknown',
                'tracks_table_size': 'Unknown',
                'analysis_table_size': 'Unknown',
                'tracks_added_today': 0
            }
    
    def _extract_year(self, date_value):
        """Extract year from date string or return None."""
        if not date_value:
            return None
        
        # Handle if date_value is a list (from mutagen tags)
        if isinstance(date_value, list):
            if len(date_value) > 0:
                date_value = str(date_value[0])
            else:
                return None
        
        # Convert to string and strip whitespace
        date_str = str(date_value).strip()
        
        # Try to extract year from various date formats
        try:
            # Handle YYYY format
            if len(date_str) == 4 and date_str.isdigit():
                year = int(date_str)
                if 1900 <= year <= 2100:
                    return year
            
            # Handle YYYY-MM-DD format
            if '-' in date_str:
                year_part = date_str.split('-')[0]
                if year_part.isdigit():
                    year = int(year_part)
                    if 1900 <= year <= 2100:
                        return year
                        
        except (ValueError, AttributeError):
            pass
            
        return None
    
    def _safe_string(self, value):
        """Safely convert value to string, handling lists and None."""
        if value is None:
            return None
        
        # Handle lists (from mutagen tags)
        if isinstance(value, list):
            if len(value) > 0:
                return str(value[0]).strip()
            else:
                return None
        
        # Convert to string and strip
        return str(value).strip() if value else None
    
    def close(self):
        """Close connection pool."""
        if self.pool:
            self.pool.closeall()
            log_universal('INFO', 'Database', 'Connection pool closed')


# Global instance
_postgresql_manager = None

def get_postgresql_manager() -> PostgreSQLManager:
    """Get global PostgreSQL manager instance."""
    global _postgresql_manager
    if _postgresql_manager is None:
        _postgresql_manager = PostgreSQLManager()
    return _postgresql_manager
