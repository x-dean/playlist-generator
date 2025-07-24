# music_analyzer/audio_analyzer.py
import numpy as np
import essentia.standard as es
import os
import logging
import hashlib
import signal
import sqlite3
import time
import traceback
import musicbrainzngs
from mutagen import File as MutagenFile
import json
from typing import Optional
from functools import wraps
from audiolizer import convert_to_host_path

logger = logging.getLogger(__name__)

class TimeoutException(Exception):
    pass

def timeout(seconds=60, error_message="Processing timed out"):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            def _handle_timeout(signum, frame):
                raise TimeoutException(error_message)
            
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result
        return wrapper
    return decorator

class AudioAnalyzer:
    """Analyze audio files and extract features for playlist generation."""
    def __init__(self, cache_file: str = None, host_music_dir: str = None, container_music_dir: str = None) -> None:
        """Initialize the AudioAnalyzer.

        Args:
            cache_file (str, optional): Path to the cache database file. Defaults to None.
            host_music_dir (str, optional): Host music directory for path normalization.
            container_music_dir (str, optional): Container music directory for path normalization.
        """
        self.timeout_seconds = 120
        cache_dir = os.getenv('CACHE_DIR', '/app/cache')
        self.cache_file = cache_file or os.path.join(cache_dir, 'audio_analysis.db')
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        self._init_db()
        self.cleanup_database()  # Clean up immediately on init
        # Set MusicBrainz user agent
        musicbrainzngs.set_useragent("PlaylistGenerator", "1.0", "noreply@example.com")
        self.host_music_dir = host_music_dir
        self.container_music_dir = container_music_dir

    def _init_db(self):
        self.conn = sqlite3.connect(self.cache_file, timeout=600)
        with self.conn:
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA busy_timeout=30000")
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS audio_features (
                file_hash TEXT PRIMARY KEY,
                file_path TEXT NOT NULL,
                duration REAL,
                bpm REAL,
                beat_confidence REAL,
                centroid REAL,
                loudness REAL,
                danceability REAL,
                key INTEGER,
                scale INTEGER,
                onset_rate REAL,
                zcr REAL,
                last_modified REAL,
                last_analyzed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata JSON,
                failed INTEGER DEFAULT 0
            )
            """)
            # Migration: add 'failed' column if missing
            columns = [row[1] for row in self.conn.execute("PRAGMA table_info(audio_features)")]
            if 'failed' not in columns:
                self.conn.execute("ALTER TABLE audio_features ADD COLUMN failed INTEGER DEFAULT 0")
            self._verify_db_schema()
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_file_path ON audio_features(file_path)")

    def _verify_db_schema(self):
        """Ensure all required columns exist with correct types"""
        cursor = self.conn.cursor()
        cursor.execute("PRAGMA table_info(audio_features)")
        existing_columns = {row[1]: row[2] for row in cursor.fetchall()}

        required_columns = {
            'loudness': 'REAL DEFAULT 0',
            'danceability': 'REAL DEFAULT 0',
            'key': 'INTEGER DEFAULT -1',
            'scale': 'INTEGER DEFAULT 0',
            'onset_rate': 'REAL DEFAULT 0',
            'zcr': 'REAL DEFAULT 0',
            'metadata': 'JSON'
        }

        for col, col_type in required_columns.items():
            if col not in existing_columns:
                logger.info(f"Adding missing column {col} to database")
                self.conn.execute(f"ALTER TABLE audio_features ADD COLUMN {col} {col_type}")

    def _ensure_float(self, value):
        """Convert value to float safely"""
        try:
            return float(value) if value is not None else 0.0
        except (TypeError, ValueError):
            return 0.0

    def _ensure_int(self, value):
        """Convert value to int safely"""
        try:
            return int(value) if value is not None else 0
        except (TypeError, ValueError):
            return 0

    def _get_file_info(self, filepath):
        # Always normalize to host path
        host_path = self._normalize_to_host_path(filepath)
        try:
            stat = os.stat(host_path)
            return {
                'file_hash': f"{os.path.basename(host_path)}_{stat.st_size}_{stat.st_mtime}",
                'last_modified': stat.st_mtime,
                'file_path': host_path
            }
        except Exception as e:
            logger.warning(f"Couldn't get file stats for {host_path}: {str(e)}")
            return {
                'file_hash': hashlib.md5(host_path.encode()).hexdigest(),
                'last_modified': 0,
                'file_path': host_path
            }

    def _normalize_to_host_path(self, path):
        if self.host_music_dir and self.container_music_dir:
            return convert_to_host_path(path, self.host_music_dir, self.container_music_dir)
        return os.path.normpath(path)

    def _safe_audio_load(self, audio_path):
        try:
            loader = es.MonoLoader(filename=audio_path, sampleRate=44100)
            audio = loader()
            del loader  # Explicit cleanup
            return audio
        except Exception as e:
            logger.error(f"AudioLoader error for {audio_path}: {str(e)}")
            return None


    def _extract_rhythm_features(self, audio):
        try:
            rhythm_extractor = es.RhythmExtractor()
            bpm, _, confidence, _ = rhythm_extractor(audio)
            beat_conf = float(np.nanmean(confidence)) if isinstance(confidence, (list, np.ndarray)) else float(confidence)
            return float(bpm), max(0.0, min(1.0, beat_conf))
        except Exception as e:
            logger.warning(f"Rhythm extraction failed: {str(e)}")
            return 0.0, 0.0

    def _extract_spectral_features(self, audio):
        try:
            spectral = es.SpectralCentroidTime(sampleRate=44100)
            centroid_values = spectral(audio)
            return float(np.nanmean(centroid_values)) if isinstance(centroid_values, (list, np.ndarray)) else float(centroid_values)
        except Exception as e:
            logger.warning(f"Spectral extraction failed: {str(e)}")
            return 0.0

    def _extract_loudness(self, audio):
        try:
            return float(es.RMS()(audio))
        except Exception as e:
            logger.warning(f"Loudness extraction failed: {str(e)}")
            return 0.0

    def _extract_danceability(self, audio):
        try:
            danceability, _ = es.Danceability()(audio)
            return float(danceability)
        except Exception as e:
            logger.warning(f"Danceability extraction failed: {str(e)}")
            return 0.0

    def _extract_key(self, audio):
        try:
            if len(audio) < 44100 * 3:  # Need at least 3 seconds
                return -1, 0

            key, scale, _ = es.KeyExtractor(frameSize=4096, hopSize=2048)(audio)

            # Convert key to numerical index
            keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            try:
                key_idx = keys.index(key) if key in keys else -1
            except ValueError:
                key_idx = -1

            scale_idx = 1 if scale == 'major' else 0
            return key_idx, scale_idx

        except Exception as e:
            logger.warning(f"Key extraction failed: {str(e)}")
            return -1, 0

    def _extract_onset_rate(self, audio):
        try:
            # Skip if audio is too short (less than 1 second)
            if len(audio) < 44100:
                return 0.0

            # Get the onset rate - returns (onset_rate, onset_times)
            result = es.OnsetRate()(audio)
            
            # Extract the onset rate value
            if isinstance(result, tuple) and len(result) > 0:
                onset_rate = result[0]
                # If it's an array, take the mean
                if isinstance(onset_rate, (list, np.ndarray)):
                    return float(np.nanmean(onset_rate))
                return float(onset_rate)
            return 0.0
        except Exception as e:
            logger.warning(f"Onset rate extraction failed: {str(e)}")
            return 0.0

    def _extract_zcr(self, audio):
        try:
            return float(np.mean(es.ZeroCrossingRate()(audio)))
        except Exception as e:
            logger.warning(f"Zero crossing rate extraction failed: {str(e)}")
            return 0.0

    def _musicbrainz_lookup(self, artist, title):
        try:
            result = musicbrainzngs.search_recordings(artist=artist, recording=title, limit=1)
            if result['recording-list']:
                rec = result['recording-list'][0]
                tags = {
                    'artist': rec['artist-credit'][0]['artist']['name'] if 'artist-credit' in rec and rec['artist-credit'] else None,
                    'title': rec['title'],
                    'album': rec['release-list'][0]['title'] if 'release-list' in rec and rec['release-list'] else None,
                    'date': rec.get('first-release-date'),
                    'genre': [tag['name'] for tag in rec['tag-list']] if 'tag-list' in rec and rec['tag-list'] else [],
                    'musicbrainz_id': rec['id']
                }
                return tags
        except Exception as e:
            logger.warning(f"MusicBrainz lookup failed: {e}")
        return {}

    def extract_features(self, audio_path: str) -> Optional[tuple]:
        """Extract features from an audio file.

        Args:
            audio_path (str): Path to the audio file.

        Returns:
            Optional[tuple]: (features dict, db_write_success bool, file_hash str) or None on failure.
        """
        try:
            file_info = self._get_file_info(audio_path)
            self.timeout_seconds = 180
            cached_features = self._get_cached_features(file_info)
            if cached_features:
                logger.info(f"Using cached features for {file_info['file_path']}")
                return cached_features, True, file_info['file_hash']
            audio = self._safe_audio_load(audio_path)
            if audio is None:
                logger.warning(f"Audio loading failed for {audio_path}")
                self._mark_failed(file_info)
                return None, False, None
            features = self._extract_all_features(audio_path, audio)
            db_write_success = self._save_features_to_db(file_info, features, failed=0)
            if db_write_success:
                logger.info(f"DB WRITE: {file_info['file_path']}")
            else:
                logger.error(f"DB WRITE FAILED: {file_info['file_path']}")
            return features, db_write_success, file_info['file_hash']
        except Exception as e:
            logger.error(f"Error processing {audio_path}: {str(e)}")
            logger.warning(traceback.format_exc())
            self._mark_failed(self._get_file_info(audio_path))
            return None, False, None
        except TimeoutException:
            logger.warning(f"Timeout on {audio_path}")
            self._mark_failed(self._get_file_info(audio_path))
            return None, False, None

    def _get_cached_features(self, file_info):
        cursor = self.conn.cursor()
        cursor.execute("""
        SELECT duration, bpm, beat_confidence, centroid,
               loudness, danceability, key, scale, onset_rate, zcr, metadata
        FROM audio_features
        WHERE file_hash = ? AND last_modified >= ?
        """, (file_info['file_hash'], file_info['last_modified']))

        if row := cursor.fetchone():
            logger.debug(f"Using cached features for {file_info['file_path']}")
            return {
                'duration': row[0],
                'bpm': row[1],
                'beat_confidence': row[2],
                'centroid': row[3],
                'loudness': row[4],
                'danceability': row[5],
                'key': row[6],
                'scale': row[7],
                'onset_rate': row[8],
                'zcr': row[9],
                'metadata': json.loads(row[10]) if row[10] else {},
                'filepath': file_info['file_path'],
                'filename': os.path.basename(file_info['file_path'])
            }
        return None

    def _extract_all_features(self, audio_path, audio):
        # Initialize with default values
        features = {
            'duration': float(len(audio) / 44100.0),
            'filepath': str(audio_path),
            'filename': str(os.path.basename(audio_path)),
            'bpm': 0.0,
            'beat_confidence': 0.0,
            'centroid': 0.0,
            'loudness': 0.0,
            'danceability': 0.0,
            'key': -1,
            'scale': 0,
            'onset_rate': 0.0,
            'zcr': 0.0
        }
        # --- Metadata extraction ---
        meta = {}
        try:
            audiofile = MutagenFile(audio_path, easy=True)
            if audiofile:
                meta = {k: (v[0] if isinstance(v, list) and v else v) for k, v in audiofile.items()}
        except Exception as e:
            logger.warning(f"Mutagen tag extraction failed: {e}")
        # If artist/title found, try MusicBrainz
        artist = meta.get('artist')
        title = meta.get('title')
        mb_tags = {}
        if artist and title:
            mb_tags = self._musicbrainz_lookup(artist, title)
        # Merge tags
        meta.update({k: v for k, v in mb_tags.items() if v})
        features['metadata'] = meta

        # Print/log MusicBrainz info if present
        # (Removed per user request)
        if mb_tags:
            logger.debug(f"MusicBrainz info for '{artist} - {title}': {mb_tags}")

        if len(audio) >= 44100:  # At least 1 second
            try:
                # Extract features
                bpm, confidence = self._extract_rhythm_features(audio)
                centroid = self._extract_spectral_features(audio)
                loudness = self._extract_loudness(audio)
                danceability = self._extract_danceability(audio)
                key, scale = self._extract_key(audio)
                onset_rate = self._extract_onset_rate(audio)
                zcr = self._extract_zcr(audio)

                # Update features
                features.update({
                    'bpm': self._ensure_float(bpm),
                    'beat_confidence': self._ensure_float(confidence),
                    'centroid': self._ensure_float(centroid),
                    'loudness': self._ensure_float(loudness),
                    'danceability': self._ensure_float(danceability),
                    'key': self._ensure_int(key),
                    'scale': self._ensure_int(scale),
                    'onset_rate': self._ensure_float(onset_rate),
                    'zcr': self._ensure_float(zcr)
                })
            except Exception as e:
                logger.error(f"Feature extraction error for {audio_path}: {str(e)}")
        return features

    def _mark_failed(self, file_info):
        # Ensure file_path is host path
        file_info = dict(file_info)
        file_info['file_path'] = self._normalize_to_host_path(file_info['file_path'])
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO audio_features (file_hash, file_path, last_modified, metadata, failed) VALUES (?, ?, ?, ?, 1)",
                (file_info['file_hash'], file_info['file_path'], file_info['last_modified'], '{}')
            )

    def _save_features_to_db(self, file_info, features, failed=0):
        # Ensure file_path is host path
        file_info = dict(file_info)
        file_info['file_path'] = self._normalize_to_host_path(file_info['file_path'])
        try:
            with self.conn:
                self.conn.execute(
                    """
                    INSERT OR REPLACE INTO audio_features (
                        file_hash, file_path, duration, bpm, beat_confidence, centroid, loudness, danceability, key, scale, onset_rate, zcr, last_modified, metadata, failed
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        file_info['file_hash'],
                        file_info['file_path'],
                        features.get('duration'),
                        features.get('bpm'),
                        features.get('beat_confidence'),
                        features.get('centroid'),
                        features.get('loudness'),
                        features.get('danceability'),
                        features.get('key'),
                        features.get('scale'),
                        features.get('onset_rate'),
                        features.get('zcr'),
                        file_info['last_modified'],
                        json.dumps(features.get('metadata', {})),
                        failed
                    )
                )
            return True
        except Exception as e:
            logger.error(f"Error saving features to DB: {str(e)}")
            return False

    def get_all_features(self, include_failed=False):
        try:
            cursor = self.conn.cursor()
            if include_failed:
                cursor.execute("""
                SELECT file_path, duration, bpm, beat_confidence, centroid,
                       loudness, danceability, key, scale, onset_rate, zcr, metadata, failed
                FROM audio_features
                """)
            else:
                cursor.execute("""
                SELECT file_path, duration, bpm, beat_confidence, centroid,
                       loudness, danceability, key, scale, onset_rate, zcr, metadata, failed
                FROM audio_features WHERE failed=0
                """)
            return [{
                'filepath': row[0],
                'duration': row[1],
                'bpm': row[2],
                'beat_confidence': row[3],
                'centroid': row[4],
                'loudness': row[5],
                'danceability': row[6],
                'key': row[7],
                'scale': row[8],
                'onset_rate': row[9],
                'zcr': row[10],
                'metadata': json.loads(row[11]) if row[11] else {},
                'failed': row[12]
            } for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error fetching features: {str(e)}")
            return []

    def cleanup_database(self) -> list[str]:
        """Remove entries for files that no longer exist.

        Returns:
            list[str]: List of file paths that were removed from the database.
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT file_path FROM audio_features")
            db_files = [row[0] for row in cursor.fetchall()]

            missing_files = [f for f in db_files if not os.path.exists(f)]

            if missing_files:
                logger.info(f"Cleaning up {len(missing_files)} missing files from database")
                placeholders = ','.join(['?'] * len(missing_files))
                cursor.execute(
                    f"DELETE FROM audio_features WHERE file_path IN ({placeholders})",
                    missing_files
                )
                self.conn.commit()
            return missing_files
        except Exception as e:
            logger.error(f"Database cleanup failed: {str(e)}")
            return []

audio_analyzer = AudioAnalyzer()