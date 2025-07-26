import numpy as np
import essentia.standard as es
import essentia
essentia.log.infoActive = False
essentia.log.warningActive = False
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
from utils.path_utils import convert_to_host_path
from utils.path_converter import PathConverter
import requests
import tensorflow as tf
import librosa
import vggish_keras as vgk

logger = logging.getLogger()

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
    def __init__(self, cache_file: str = None, library: str = None, music: str = None) -> None:
        """Initialize the AudioAnalyzer.

        Args:
            cache_file (str, optional): Path to the cache database file. Defaults to None.
            library (str, optional): Music library directory for path normalization.
            music (str, optional): Container music directory for path normalization.
        """
        self.timeout_seconds = 120
        cache_dir = os.getenv('CACHE_DIR', '/app/cache')
        self.cache_file = cache_file or os.path.join(cache_dir, 'audio_analysis.db')
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        self._init_db()
        self.cleanup_database()  # Clean up immediately on init
        # Set MusicBrainz user agent
        musicbrainzngs.set_useragent("PlaylistGenerator", "1.0", "noreply@example.com")
        self.library = library
        self.music = music
        # Load TensorFlow models
        self.vggish_model = self._load_vggish_model()

    def _load_vggish_model(self):
        """Load VGGish model for audio embeddings using vggish-keras (auto-downloads weights)."""
        try:
            # Get the embedding function which handles model loading and preprocessing
            self.vggish_compute = vgk.get_embedding_function(hop_duration=0.25)
            logger.info("Loaded VGGish model using vggish-keras.")
            return self.vggish_compute
        except Exception as e:
            logger.error(f"Failed to load VGGish model with vggish-keras: {e}")
            return None

    def _audio_to_mel_spectrogram(self, audio, sr=44100):
        """Convert audio to mel-spectrogram for VGGish input."""
        try:
            # VGGish expects 96 mel bands, 64 time frames
            mel_spec = librosa.feature.melspectrogram(
                y=audio, 
                sr=sr, 
                n_mels=96, 
                hop_length=512,
                n_fft=2048
            )
            # Convert to log scale
            mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
            # Pad or truncate to 64 time frames
            if mel_spec.shape[1] < 64:
                mel_spec = np.pad(mel_spec, ((0, 0), (0, 64 - mel_spec.shape[1])))
            else:
                mel_spec = mel_spec[:, :64]
            return mel_spec
        except Exception as e:
            logger.warning(f"Mel-spectrogram conversion failed: {str(e)}")
            return None

    def _extract_vggish_embedding(self, audio):
        """Extract VGGish embeddings from audio using vggish-keras."""
        if not hasattr(self, 'vggish_compute') or self.vggish_compute is None:
            return None
        try:
            # vggish-keras expects audio as numpy array with sample rate
            # We need to convert the audio to the right format
            sr = 44100  # Standard sample rate
            # Get embeddings using the compute function
            embeddings, timestamps = self.vggish_compute(y=audio, sr=sr)
            # Return the mean embedding across time (global representation)
            return np.mean(embeddings, axis=0).tolist()
        except Exception as e:
            logger.warning(f"VGGish embedding extraction failed: {str(e)}")
            return None

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
                mfcc JSON,
                chroma JSON,
                spectral_contrast JSON,
                spectral_flatness REAL,
                spectral_rolloff REAL,
                last_modified REAL,
                last_analyzed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata JSON,
                failed INTEGER DEFAULT 0,
                vggish_embedding JSON
            )
            """)
            # Migration: add 'failed' column if missing
            columns = [row[1] for row in self.conn.execute("PRAGMA table_info(audio_features)")]
            if 'failed' not in columns:
                self.conn.execute("ALTER TABLE audio_features ADD COLUMN failed INTEGER DEFAULT 0")
            if 'vggish_embedding' not in columns:
                self.conn.execute("ALTER TABLE audio_features ADD COLUMN vggish_embedding JSON")
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
            'metadata': 'JSON',
            'mfcc': 'JSON',
            'chroma': 'JSON',
            'spectral_contrast': 'JSON',
            'spectral_flatness': 'REAL DEFAULT 0',
            'spectral_rolloff': 'REAL DEFAULT 0'
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
        # Always normalize to library path
        library_path = self._normalize_to_library_path(filepath)
        try:
            stat = os.stat(library_path)
            return {
                'file_hash': f"{os.path.basename(library_path)}_{stat.st_size}_{stat.st_mtime}",
                'last_modified': stat.st_mtime,
                'file_path': library_path
            }
        except Exception as e:
            logger.warning(f"Couldn't get file stats for {library_path}: {str(e)}")
            return {
                'file_hash': hashlib.md5(library_path.encode()).hexdigest(),
                'last_modified': 0,
                'file_path': library_path
            }

    def _normalize_to_library_path(self, path):
        if self.library and self.music:
            path_converter = PathConverter(self.library, self.music)
            return path_converter.container_to_host(path)
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

    def _extract_mfcc(self, audio, num_coeffs=13):
        try:
            mfcc = es.MFCC(numberCoefficients=num_coeffs)
            _, mfcc_coeffs = mfcc(audio)
            return np.mean(mfcc_coeffs, axis=0).tolist()  # global average
        except Exception as e:
            logger.warning(f"MFCC extraction failed: {str(e)}")
            return [0.0] * num_coeffs

    def _extract_chroma(self, audio):
        try:
            chromagram = es.Chromagram()
            chroma = chromagram(audio)
            return np.mean(chroma, axis=1).tolist()  # average per pitch class
        except Exception as e:
            logger.warning(f"Chroma extraction failed: {str(e)}")
            return [0.0] * 12

    def _extract_spectral_contrast(self, audio):
        try:
            spectral_contrast = es.SpectralContrast()
            contrast = spectral_contrast(audio)
            return np.mean(contrast, axis=1).tolist() if isinstance(contrast, np.ndarray) else [float(contrast)]
        except Exception as e:
            logger.warning(f"Spectral contrast extraction failed: {str(e)}")
            return [0.0] * 6  # Essentia default: 6 bands

    def _extract_spectral_flatness(self, audio):
        try:
            flatness = es.SpectralFlatness()(audio)
            return float(np.mean(flatness)) if isinstance(flatness, (list, np.ndarray)) else float(flatness)
        except Exception as e:
            logger.warning(f"Spectral flatness extraction failed: {str(e)}")
            return 0.0

    def _extract_spectral_rolloff(self, audio):
        try:
            rolloff = es.SpectralRollOff()(audio)
            return float(np.mean(rolloff)) if isinstance(rolloff, (list, np.ndarray)) else float(rolloff)
        except Exception as e:
            logger.warning(f"Spectral rolloff extraction failed: {str(e)}")
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

    def _lastfm_lookup(self, artist, title):
        """Query Last.fm for track info if MusicBrainz is missing fields."""
        api_key = os.getenv('LASTFM_API_KEY')
        if not api_key:
            logger.warning("LASTFM_API_KEY not set; skipping Last.fm enrichment.")
            return {}
        url = "http://ws.audioscrobbler.com/2.0/"
        params = {
            'method': 'track.getInfo',
            'api_key': api_key,
            'artist': artist,
            'track': title,
            'format': 'json'
        }
        try:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            track = data.get('track', {})
            tags = track.get('toptags', {}).get('tag', [])
            genre = tags[0]['name'] if tags else None
            album = track.get('album', {}).get('title')
            year = None
            # Last.fm does not provide year directly; try to parse from album or wiki
            if 'wiki' in track and 'published' in track['wiki']:
                import re
                match = re.search(r'(\d{4})', track['wiki']['published'])
                if match:
                    year = match.group(1)
            return {
                'genre': genre,
                'album': album,
                'year': year
            }
        except Exception as e:
            logger.warning(f"Last.fm lookup failed for {artist} - {title}: {e}")
            return {}

    def extract_features(self, audio_path: str, force_reextract: bool = False) -> Optional[tuple]:
        """Extract features from an audio file.

        Args:
            audio_path (str): Path to the audio file.
            force_reextract (bool): If True, bypass the cache and re-extract features.

        Returns:
            Optional[tuple]: (features dict, db_write_success bool, file_hash str) or None on failure.
        """
        try:
            file_info = self._get_file_info(audio_path)
            self.timeout_seconds = 180
            if not force_reextract:
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
               loudness, danceability, key, scale, onset_rate, zcr,
               mfcc, chroma, spectral_contrast, spectral_flatness, spectral_rolloff, metadata, vggish_embedding
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
                'mfcc': json.loads(row[10]) if row[10] else [0.0] * 13,
                'chroma': json.loads(row[11]) if row[11] else [0.0] * 12,
                'spectral_contrast': json.loads(row[12]) if row[12] else [0.0] * 6,
                'spectral_flatness': row[13],
                'spectral_rolloff': row[14],
                'metadata': json.loads(row[15]) if row[15] else {},
                'vggish_embedding': json.loads(row[16]) if row[16] else None,
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
            'zcr': 0.0,
            'mfcc': [0.0] * 13,
            'chroma': [0.0] * 12,
            'spectral_contrast': [0.0] * 6,
            'spectral_flatness': 0.0,
            'spectral_rolloff': 0.0,
            'vggish_embedding': None
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
            updated_fields_mb = [k for k, v in mb_tags.items() if v and (k not in meta or meta[k] != v)]
            meta.update({k: v for k, v in mb_tags.items() if v})
            logger.info(f"MusicBrainz enrichment: {artist} - {title} (fields updated: {updated_fields_mb})")

        # Define a comprehensive set of non-real genres
        NON_REAL_GENRES = {None, '', 'Other', 'UnknownGenre', 'Unknown', 'Misc', 'Various', 'VA', 'General', 'Soundtrack', 'OST', 'N/A', 'Not Available', 'No Genre', 'Unclassified', 'Unsorted', 'Undefined', 'Genre', 'Genres', 'Music', 'Song', 'Songs', 'Audio', 'MP3', 'Instrumental', 'Vocal', 'Various Artists', 'VA', 'Compilation', 'Compilations', 'Album', 'Albums', 'CD', 'CDs', 'Record', 'Records', 'Single', 'Singles', 'EP', 'EPs', 'LP', 'LPs', 'Demo', 'Demos', 'Test', 'Tests', 'Sample', 'Samples', 'Example', 'Examples', 'Untitled', 'Unknown Artist', 'Unknown Album', 'Unknown Title', 'No Title', 'No Album', 'No Artist'}
        genre = meta.get('genre')
        missing_fields = [field for field in ['genre', 'year', 'album'] if not meta.get(field)]
        if genre is None or (isinstance(genre, str) and genre.strip() in NON_REAL_GENRES):
            if 'genre' not in missing_fields:
                missing_fields.append('genre')
        if missing_fields and artist and title:
            logger.debug(f"Last.fm enrichment triggered for {artist} - {title}, missing fields: {missing_fields}")
            lastfm_tags = self._lastfm_lookup(artist, title)
            logger.debug(f"Last.fm response for {artist} - {title}: {lastfm_tags}")
            updated_fields_lastfm = []
            for field in missing_fields:
                if lastfm_tags.get(field):
                    meta[field] = lastfm_tags[field]
                    updated_fields_lastfm.append(field)
            logger.info(f"Last.fm enrichment: {artist} - {title} (fields updated: {updated_fields_lastfm})")

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
                mfcc = self._extract_mfcc(audio)
                chroma = self._extract_chroma(audio)
                spectral_contrast = self._extract_spectral_contrast(audio)
                spectral_flatness = self._extract_spectral_flatness(audio)
                spectral_rolloff = self._extract_spectral_rolloff(audio)
                vggish_embedding = self._extract_vggish_embedding(audio)

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
                    'zcr': self._ensure_float(zcr),
                    'mfcc': mfcc,
                    'chroma': chroma,
                    'spectral_contrast': spectral_contrast,
                    'spectral_flatness': self._ensure_float(spectral_flatness),
                    'spectral_rolloff': self._ensure_float(spectral_rolloff),
                    'vggish_embedding': vggish_embedding
                })
            except Exception as e:
                logger.error(f"Feature extraction error for {audio_path}: {str(e)}")
        return features

    def _mark_failed(self, file_info):
        # Ensure file_path is host path
        file_info = dict(file_info)
        file_info['file_path'] = self._normalize_to_library_path(file_info['file_path'])
        logging.getLogger().warning(f"Marking file as failed: {file_info['file_path']} (hash: {file_info['file_hash']})")
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO audio_features (file_hash, file_path, last_modified, metadata, failed) VALUES (?, ?, ?, ?, 1)",
                (file_info['file_hash'], file_info['file_path'], file_info['last_modified'], '{}')
            )

    def _save_features_to_db(self, file_info, features, failed=0):
        # Ensure file_path is host path
        file_info = dict(file_info)
        file_info['file_path'] = self._normalize_to_library_path(file_info['file_path'])
        try:
            with self.conn:
                self.conn.execute(
                    """
                    INSERT OR REPLACE INTO audio_features (
                        file_hash, file_path, duration, bpm, beat_confidence, centroid, loudness, danceability, key, scale, onset_rate, zcr,
                        mfcc, chroma, spectral_contrast, spectral_flatness, spectral_rolloff,
                        last_modified, metadata, failed, vggish_embedding
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                        json.dumps(features.get('mfcc', [])),
                        json.dumps(features.get('chroma', [])),
                        json.dumps(features.get('spectral_contrast', [])),
                        features.get('spectral_flatness'),
                        features.get('spectral_rolloff'),
                        file_info['last_modified'],
                        json.dumps(features.get('metadata', {})),
                        failed,
                        json.dumps(features.get('vggish_embedding', None))
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

    def enrich_metadata_for_failed_file(self, file_info):
        """Enrich metadata for a failed file using MusicBrainz and Last.fm, and update DB."""
        # Try to extract artist/title from tags
        meta = {}
        try:
            audiofile = MutagenFile(file_info['file_path'], easy=True)
            if audiofile:
                meta = {k: (v[0] if isinstance(v, list) and v else v) for k, v in audiofile.items()}
        except Exception as e:
            logger.warning(f"Mutagen tag extraction failed for enrichment: {e}")
        artist = meta.get('artist')
        title = meta.get('title')
        mb_tags = {}
        lastfm_tags = {}
        updated_fields = []
        if artist and title:
            mb_tags = self._musicbrainz_lookup(artist, title)
            for k, v in mb_tags.items():
                if v and (k not in meta or meta[k] != v):
                    meta[k] = v
                    updated_fields.append(f"mb:{k}")
            # Fallback to Last.fm for missing fields
            missing_fields = [field for field in ['genre', 'year', 'album'] if not meta.get(field)]
            if missing_fields:
                lastfm_tags = self._lastfm_lookup(artist, title)
                for field in missing_fields:
                    v = lastfm_tags.get(field)
                    if v and not meta.get(field):
                        meta[field] = v
                        updated_fields.append(f"lastfm:{field}")
        # Update only the metadata column, keep failed=1
        try:
            with self.conn:
                self.conn.execute(
                    "UPDATE audio_features SET metadata = ?, failed = 1 WHERE file_hash = ?",
                    (json.dumps(meta), file_info['file_hash'])
                )
            logger.info(f"Enriched metadata for failed file {file_info['file_path']} (fields updated: {updated_fields})")
        except Exception as e:
            logger.error(f"Error updating metadata for failed file {file_info['file_path']}: {e}")

audio_analyzer = AudioAnalyzer()