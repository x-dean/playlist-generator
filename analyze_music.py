# analyze_music.py
import numpy as np
import essentia.standard as es
import os
import logging
import sqlite3
import hashlib
import signal
from functools import wraps
import traceback

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TimeoutException(Exception):
    pass

def timeout(seconds=120, error_message="Processing timed out"):
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
    def __init__(self, cache_file=None):
        cache_dir = os.getenv('CACHE_DIR', '/app/cache')
        self.cache_file = cache_file or os.path.join(cache_dir, 'audio_analysis.db')
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        self.timeout_seconds = 30
        self._init_db()

    def _init_db(self):
        self.conn = sqlite3.connect(self.cache_file, timeout=30)
        with self.conn:
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA busy_timeout=30000")
            
            # Check if we need to migrate schema
            cursor = self.conn.cursor()
            cursor.execute("PRAGMA table_info(audio_features)")
            columns = [row[1] for row in cursor.fetchall()]
            
            if 'loudness' not in columns:
                logger.info("Migrating database schema to add new features")
                self.conn.execute("ALTER TABLE audio_features ADD COLUMN loudness REAL DEFAULT 0")
                self.conn.execute("ALTER TABLE audio_features ADD COLUMN danceability REAL DEFAULT 0")
                self.conn.execute("ALTER TABLE audio_features ADD COLUMN key INTEGER DEFAULT -1")
                self.conn.execute("ALTER TABLE audio_features ADD COLUMN scale INTEGER DEFAULT 0")
                self.conn.execute("ALTER TABLE audio_features ADD COLUMN onset_rate REAL DEFAULT 0")
                self.conn.execute("ALTER TABLE audio_features ADD COLUMN zcr REAL DEFAULT 0")

    def _get_file_info(self, filepath):
        try:
            stat = os.stat(filepath)
            return {
                'file_hash': f"{os.path.basename(filepath)}_{stat.st_size}_{stat.st_mtime}",
                'last_modified': stat.st_mtime,
                'file_path': filepath
            }
        except Exception as e:
            logger.warning(f"Couldn't get file stats for {filepath}: {str(e)}")
            return {
                'file_hash': hashlib.md5(filepath.encode()).hexdigest(),
                'last_modified': 0,
                'file_path': filepath
            }

    @timeout()
    def _safe_audio_load(self, audio_path):
        try:
            loader = es.MonoLoader(filename=audio_path, sampleRate=44100)
            audio = loader()
            return audio if audio.size > 0 else None
        except Exception as e:
            logger.warning(f"AudioLoader error for {audio_path}: {str(e)}")
            return None

    @timeout()
    def _extract_rhythm_features(self, audio):
        try:
            rhythm_extractor = es.RhythmExtractor()
            bpm, _, confidence, _ = rhythm_extractor(audio)

            if isinstance(confidence, (list, np.ndarray)):
                beat_conf = float(np.nanmean(confidence))
            else:
                beat_conf = float(confidence)

            beat_conf = max(0.0, min(1.0, beat_conf))
            return float(bpm), beat_conf
        except Exception as e:
            logger.warning(f"Rhythm extraction failed: {str(e)}")
            return 0.0, 0.0

    @timeout()
    def _extract_spectral_features(self, audio):
        try:
            spectral = es.SpectralCentroidTime(sampleRate=44100)
            centroid_values = spectral(audio)

            if isinstance(centroid_values, (list, np.ndarray)):
                centroid = float(np.nanmean(centroid_values))
            else:
                centroid = float(centroid_values)

            return centroid
        except Exception as e:
            logger.warning(f"Spectral extraction failed: {str(e)}")
            return 0.0

    @timeout()
    def _extract_loudness(self, audio):
        try:
            rms = es.RMS()
            return rms(audio)
        except Exception as e:
            logger.warning(f"Loudness extraction failed: {str(e)}")
            return 0.0

    @timeout()
    def _extract_danceability(self, audio):
        try:
            danceability, _ = es.Danceability()(audio)
            return danceability
        except Exception as e:
            logger.warning(f"Danceability extraction failed: {str(e)}")
            return 0.0

    @timeout()
    def _extract_key(self, audio):
        try:
            # Ensure audio is long enough for key detection
            if len(audio) < 44100:  # At least 1 second of audio
                return -1, 0
                
            key, scale, _ = es.Key()(audio)
            keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            try:
                key_index = keys.index(key) if key in keys else -1
            except ValueError:
                key_index = -1
            scale_index = 1 if scale == 'major' else 0
            return key_index, scale_index
        except Exception as e:
            logger.warning(f"Key extraction failed: {str(e)}")
            return -1, 0

    @timeout()
    def _extract_onset_rate(self, audio):
        try:
            onset_rate = es.OnsetRate()(audio)
            return onset_rate
        except Exception as e:
            logger.warning(f"Onset rate extraction failed: {str(e)}")
            return 0.0

    @timeout()
    def _extract_zcr(self, audio):
        try:
            zcr = es.ZeroCrossingRate()
            return np.mean(zcr(audio))
        except Exception as e:
            logger.warning(f"Zero crossing rate extraction failed: {str(e)}")
            return 0.0        

    def extract_features(self, audio_path):
        try:
            file_info = self._get_file_info(audio_path)
            cursor = self.conn.cursor()
            cursor.execute("""
            SELECT duration, bpm, beat_confidence, centroid, 
                loudness, danceability, key, scale, onset_rate, zcr
            FROM audio_features
            WHERE file_hash = ? AND last_modified >= ?
            """, (file_info['file_hash'], file_info['last_modified']))

            if row := cursor.fetchone():
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
                    'filepath': audio_path,
                    'filename': os.path.basename(audio_path)
                }, True, file_info['file_hash']

            logger.info(f"Processing: {os.path.basename(audio_path)}")
            audio = self._safe_audio_load(audio_path)
            if audio is None:
                return None, False, None

            features = {
                'duration': len(audio) / 44100.0,
                'filepath': audio_path,
                'filename': os.path.basename(audio_path)
            }

            # Only extract features if we have enough audio
            if len(audio) >= 44100:  # At least 1 second
                try:
                    features['bpm'], features['beat_confidence'] = self._extract_rhythm_features(audio)
                except Exception as e:
                    logger.error(f"Rhythm extraction error for {audio_path}: {str(e)}")
                    features['bpm'] = 0.0
                    features['beat_confidence'] = 0.0

                try:
                    features['centroid'] = self._extract_spectral_features(audio)
                except Exception as e:
                    logger.error(f"Spectral extraction error for {audio_path}: {str(e)}")
                    features['centroid'] = 0.0
                    
                try:
                    features['loudness'] = self._extract_loudness(audio)
                except Exception as e:
                    logger.error(f"Loudness extraction error for {audio_path}: {str(e)}")
                    features['loudness'] = 0.0
                    
                try:
                    features['danceability'] = self._extract_danceability(audio)
                except Exception as e:
                    logger.error(f"Danceability extraction error for {audio_path}: {str(e)}")
                    features['danceability'] = 0.0
                    
                try:
                    features['key'], features['scale'] = self._extract_key(audio)
                except Exception as e:
                    logger.error(f"Key extraction error for {audio_path}: {str(e)}")
                    features['key'] = -1
                    features['scale'] = 0
                    
                try:
                    features['onset_rate'] = self._extract_onset_rate(audio)
                except Exception as e:
                    logger.error(f"Onset rate extraction error for {audio_path}: {str(e)}")
                    features['onset_rate'] = 0.0
                    
                try:
                    features['zcr'] = self._extract_zcr(audio)
                except Exception as e:
                    logger.error(f"Zero crossing rate extraction error for {audio_path}: {str(e)}")
                    features['zcr'] = 0.0
            else:
                logger.warning(f"Audio file too short for feature extraction: {audio_path}")
                # Set default values for short files
                features.update({
                    'bpm': 0.0,
                    'beat_confidence': 0.0,
                    'centroid': 0.0,
                    'loudness': 0.0,
                    'danceability': 0.0,
                    'key': -1,
                    'scale': 0,
                    'onset_rate': 0.0,
                    'zcr': 0.0
                })

            with self.conn:
                self.conn.execute("""
                INSERT OR REPLACE INTO audio_features 
                (file_hash, file_path, duration, bpm, beat_confidence, centroid, 
                loudness, danceability, key, scale, onset_rate, zcr, last_modified)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    file_info['file_hash'],
                    file_info['file_path'],
                    features['duration'],
                    features['bpm'],
                    features['beat_confidence'],
                    features['centroid'],
                    features['loudness'],
                    features['danceability'],
                    features['key'],
                    features['scale'],
                    features['onset_rate'],
                    features['zcr'],
                    file_info['last_modified']
                ))

            return features, False, file_info['file_hash']
        except Exception as e:
            logger.error(f"Error processing {audio_path}: {str(e)}")
            logger.error(traceback.format_exc())
            return None, False, None

    def get_all_features(self):
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
            SELECT file_path, duration, bpm, beat_confidence, centroid, 
                   loudness, danceability, key, scale, onset_rate, zcr
            FROM audio_features
            """)
            return [
                {
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
                    'zcr': row[10]
                }
                for row in cursor.fetchall()
            ]
        except Exception as e:
            logger.error(f"Error fetching features: {str(e)}")
            return []

audio_analyzer = AudioAnalyzer()

def extract_features(audio_path):
    return audio_analyzer.extract_features(audio_path)

def get_all_features():
    return audio_analyzer.get_all_features()