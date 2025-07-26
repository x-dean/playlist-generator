import os
import json
import sqlite3
import logging
from typing import Optional, Dict, Any, List, Tuple
from functools import wraps
import time
import numpy as np
import essentia.standard as es
import essentia
import librosa
import requests
import musicbrainzngs
from mutagen import File as MutagenFile

# Set up logger first
logger = logging.getLogger(__name__)

# Check if Essentia was built with TensorFlow support (only once)
_essentia_tf_support_checked = False
def check_essentia_tf_support():
    global _essentia_tf_support_checked
    if not _essentia_tf_support_checked:
        try:
            from essentia.standard import TensorflowPredictMusiCNN
            logger.info("Essentia TensorFlow support: AVAILABLE")
        except ImportError as e:
            logger.warning(f"Essentia TensorFlow support: NOT AVAILABLE - {e}")
            logger.warning("MusiCNN embeddings will not work. Install Essentia with TensorFlow support.")
        _essentia_tf_support_checked = True

# Run the check once at module import
check_essentia_tf_support()

# Configure Numba to use fallback mode to avoid compilation errors
import numba
# Only disable CUDA JIT, keep CPU JIT for librosa
numba.config.CUDA_DISABLE_JIT = True

# Configure TensorFlow to suppress warnings (only if TensorFlow is used)
try:
    import tensorflow as tf
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.get_logger().setLevel('ERROR')
except ImportError:
    pass

logger = logging.getLogger(__name__)

# Lean metadata fields for playlisting
LEAN_FIELDS = [
    'artist', 'title', 'album', 'year', 'genre', 'tracknumber', 'discnumber',
    'composer', 'lyricist', 'arranger', 'publisher', 'label', 'isrc',
    'musicbrainz_id', 'isrc', 'mb_artist_id', 'mb_album_id',
    'release_date', 'country', 'work', 'composer',
    'genre_lastfm', 'album_lastfm', 'listeners', 'playcount', 'wiki', 'album_mbid', 'artist_mbid'
]

def filter_metadata(meta):
    """Filter metadata to keep only lean fields relevant for playlisting."""
    return {k: v for k, v in meta.items() if k in LEAN_FIELDS and v is not None and v != ''}

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
        # self.vggish_model = self._load_vggish_model() # Removed VGGish model loading

    # Removed _load_vggish_model

    # Removed _audio_to_mel_spectrogram

    def _extract_musicnn_embedding(self, audio_path):
        """Extract MusiCNN embedding and auto-tags using Essentia's TensorflowPredictMusiCNN and MonoLoader, matching the official tutorial."""
        try:
            import essentia.standard as es
            import numpy as np
            import os
            import json
            
            model_path = os.getenv(
                'MUSICNN_MODEL_PATH',
                '/app/feature_extraction/models/musicnn/msd-musicnn-1.pb'
            )
            json_path = os.getenv(
                'MUSICNN_JSON_PATH',
                '/app/feature_extraction/models/musicnn/msd-musicnn-1.json'
            )
            
            if not os.path.exists(model_path):
                logger.warning(f"MusiCNN model not found at {model_path}")
                return None
            if not os.path.exists(json_path):
                logger.warning(f"MusiCNN JSON metadata not found at {json_path}")
                return None
            
            # Load tag names from JSON
            with open(json_path, 'r') as json_file:
                metadata = json.load(json_file)
            tag_names = metadata.get('classes', [])
            # Get output layer for embeddings
            output_layer = 'model/dense_1/BiasAdd'
            if 'schema' in metadata and 'outputs' in metadata['schema']:
                for output in metadata['schema']['outputs']:
                    if 'description' in output and output['description'] == 'embeddings':
                        output_layer = output['name']
                        break
            
            # Load audio using MonoLoader at 16kHz (tutorial pattern)
            audio = es.MonoLoader(filename=audio_path, sampleRate=16000)()
            
            # Run MusiCNN for activations (auto-tagging)
            musicnn = es.TensorflowPredictMusiCNN(graphFilename=model_path)
            activations = musicnn(audio)  # shape: [time, tags]
            tag_probs = activations.mean(axis=0)
            tags = dict(zip(tag_names, tag_probs))
            
            # Run MusiCNN for embeddings (using correct output layer)
            musicnn_emb = es.TensorflowPredictMusiCNN(graphFilename=model_path, output=output_layer)
            embeddings = musicnn_emb(audio)
            embedding = np.mean(embeddings, axis=0)
            
            logger.info("Successfully extracted MusiCNN embedding and tags")
            return {
                'embedding': embedding.tolist(),
                'tags': tags
            }
        except Exception as e:
            logger.warning(f"MusiCNN embedding/tag extraction failed: {str(e)}")
            logger.warning(f"Exception type: {type(e).__name__}")
            import traceback
            logger.warning(f"Full traceback: {traceback.format_exc()}")
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
                spectral_contrast REAL,
                spectral_flatness REAL,
                spectral_rolloff REAL,
                musicnn_embedding JSON,
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
            if 'musicnn_embedding' not in columns:
                self.conn.execute("ALTER TABLE audio_features ADD COLUMN musicnn_embedding JSON")
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
            'spectral_contrast': 'REAL',
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
        """Extract chroma features from audio using HPCP."""
        try:
            # Set up parameters
            frame_size = 2048
            hop_size = 1024
            
            # Initialize algorithms
            window = es.Windowing(type='hann')
            spectrum = es.Spectrum()
            spectral_peaks = es.SpectralPeaks()
            hpcp = es.HPCP()
            
            hpcp_list = []
            
            # Process audio frame by frame
            for frame in es.FrameGenerator(audio, frameSize=frame_size, hopSize=hop_size, startFromZero=True):
                spec = spectrum(window(frame))
                freqs, mags = spectral_peaks(spec)
                
                if len(freqs) > 0:
                    hpcp_frame = hpcp(freqs, mags)
                    hpcp_list.append(hpcp_frame)
            
            # Aggregate results over time (mean)
            if hpcp_list:
                hpcp_mean = np.mean(hpcp_list, axis=0).tolist()
                return hpcp_mean  # Return full 12-dimensional HPCP vector
            else:
                return [0.0] * 12  # Return 12 zeros if no frames processed
                
        except Exception as e:
            logger.warning(f"Chroma extraction failed: {str(e)}")
            return [0.0] * 12

    def _extract_spectral_contrast(self, audio):
        """Extract spectral contrast features from audio."""
        try:
            # Use frame-by-frame processing for spectral contrast
            frame_size = 2048
            hop_size = 1024
            
            window = es.Windowing(type='hann')
            spectrum = es.Spectrum()
            spectral_peaks = es.SpectralPeaks()
            
            contrast_list = []
            
            # Process audio frame by frame
            for frame in es.FrameGenerator(audio, frameSize=frame_size, hopSize=hop_size, startFromZero=True):
                spec = spectrum(window(frame))
                freqs, mags = spectral_peaks(spec)
                
                if len(freqs) > 0:
                    # Calculate spectral contrast manually
                    # Sort magnitudes and find valleys
                    sorted_mags = np.sort(mags)
                    valleys = sorted_mags[:len(sorted_mags)//3]  # Bottom third
                    peaks = sorted_mags[-len(sorted_mags)//3:]   # Top third
                    contrast = np.mean(peaks) - np.mean(valleys) if len(peaks) > 0 and len(valleys) > 0 else 0.0
                    contrast_list.append(contrast)
            
            # Return mean contrast across all frames
            if contrast_list:
                return float(np.mean(contrast_list))
            else:
                return 0.0
                
        except Exception as e:
            logger.warning(f"Spectral contrast extraction failed: {str(e)}")
            return 0.0

    def _extract_spectral_flatness(self, audio):
        """Extract spectral flatness from audio."""
        try:
            # Use frame-by-frame processing for spectral flatness
            frame_size = 2048
            hop_size = 1024
            
            window = es.Windowing(type='hann')
            spectrum = es.Spectrum()
            
            flatness_list = []
            
            # Process audio frame by frame
            for frame in es.FrameGenerator(audio, frameSize=frame_size, hopSize=hop_size, startFromZero=True):
                spec = spectrum(window(frame))
                
                # Calculate spectral flatness manually
                # Flatness = geometric_mean / arithmetic_mean
                spec_positive = spec[spec > 0]  # Avoid log(0)
                if len(spec_positive) > 0:
                    geometric_mean = np.exp(np.mean(np.log(spec_positive)))
                    arithmetic_mean = np.mean(spec_positive)
                    flatness = geometric_mean / arithmetic_mean if arithmetic_mean > 0 else 0.0
                    flatness_list.append(flatness)
            
            # Return mean flatness across all frames
            if flatness_list:
                return float(np.mean(flatness_list))
            else:
                return 0.0
                
        except Exception as e:
            logger.warning(f"Spectral flatness extraction failed: {str(e)}")
            return 0.0

    def _extract_spectral_rolloff(self, audio):
        """Extract spectral rolloff from audio."""
        try:
            # Use frame-by-frame processing for spectral rolloff
            frame_size = 2048
            hop_size = 1024
            
            window = es.Windowing(type='hann')
            spectrum = es.Spectrum()
            
            rolloff_list = []
            
            # Process audio frame by frame
            for frame in es.FrameGenerator(audio, frameSize=frame_size, hopSize=hop_size, startFromZero=True):
                spec = spectrum(window(frame))
                
                # Calculate spectral rolloff manually
                # Rolloff is the frequency below which 85% of the energy is contained
                total_energy = np.sum(spec)
                if total_energy > 0:
                    cumulative_energy = np.cumsum(spec)
                    threshold = 0.85 * total_energy
                    rolloff_idx = np.where(cumulative_energy >= threshold)[0]
                    if len(rolloff_idx) > 0:
                        # Convert bin index to frequency (assuming 44.1kHz sample rate)
                        rolloff_freq = (rolloff_idx[0] / len(spec)) * 22050  # Nyquist frequency
                        rolloff_list.append(rolloff_freq)
            
            # Return mean rolloff across all frames
            if rolloff_list:
                return float(np.mean(rolloff_list))
            else:
                return 0.0
                
        except Exception as e:
            logger.warning(f"Spectral rolloff extraction failed: {str(e)}")
            return 0.0

    def _musicbrainz_lookup(self, artist, title):
        try:
            # Step 1: Search for the recording
            result = musicbrainzngs.search_recordings(artist=artist, recording=title, limit=1)
            if result['recording-list']:
                rec = result['recording-list'][0]
                mbid = rec['id']
                # Step 2: Lookup by MBID for full info - get everything available
                rec_full = musicbrainzngs.get_recording_by_id(
                    mbid,
                    includes=[
                        'artists', 'releases', 'tags', 'isrcs', 'work-rels', 'artist-credits',
                        'artist-rels', 'aliases', 'recording-rels'
                    ]
                )['recording']
                
                # Extract all available data, then filter to lean fields
                all_mb_data = {
                    'artist': rec_full['artist-credit'][0]['artist']['name'] if 'artist-credit' in rec_full and rec_full['artist-credit'] else None,
                    'title': rec_full['title'],
                    'album': rec_full['releases'][0]['title'] if 'releases' in rec_full and rec_full['releases'] else None,
                    'release_date': rec_full['releases'][0].get('date') if 'releases' in rec_full and rec_full['releases'] and 'date' in rec_full['releases'][0] else None,
                    'country': rec_full['releases'][0].get('country') if 'releases' in rec_full and rec_full['releases'] and 'country' in rec_full['releases'][0] else None,
                    'label': rec_full['releases'][0]['label-info-list'][0]['label']['name'] if 'releases' in rec_full and rec_full['releases'] and 'label-info-list' in rec_full['releases'][0] and rec_full['releases'][0]['label-info-list'] else None,
                    'genre': [tag['name'] for tag in rec_full.get('tag-list', [])],
                    'musicbrainz_id': rec_full['id'],
                    'isrc': rec_full.get('isrc-list', [None])[0],
                    'mb_artist_id': rec_full['artist-credit'][0]['artist']['id'] if 'artist-credit' in rec_full and rec_full['artist-credit'] else None,
                    'mb_album_id': rec_full['releases'][0]['id'] if 'releases' in rec_full and rec_full['releases'] else None,
                    'work': rec_full['work-relation-list'][0]['work']['title'] if 'work-relation-list' in rec_full and rec_full['work-relation-list'] else None,
                    'composer': rec_full['work-relation-list'][0]['work']['artist-relation-list'][0]['artist']['name'] if 'work-relation-list' in rec_full and rec_full['work-relation-list'] and 'artist-relation-list' in rec_full['work-relation-list'][0]['work'] and rec_full['work-relation-list'][0]['work']['artist-relation-list'] else None,
                }
                
                # Filter to only lean fields for database
                tags = {k: v for k, v in all_mb_data.items() if k in LEAN_FIELDS and v is not None and v != ''}
                return tags
        except Exception as e:
            logger.warning(f"MusicBrainz lookup failed: {e}")
        return {}

    def _lastfm_lookup(self, artist, title):
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
            
            # Extract all available Last.fm data
            all_lastfm_data = {
                'genre_lastfm': [t['name'] for t in track.get('toptags', {}).get('tag', [])] if track.get('toptags', {}).get('tag') else None,
                'album_lastfm': track.get('album', {}).get('title') if 'album' in track else None,
                'listeners': track.get('listeners'),
                'playcount': track.get('playcount'),
                'wiki': track.get('wiki', {}).get('summary') if 'wiki' in track else None,
                'album_mbid': track.get('album', {}).get('mbid') if 'album' in track else None,
                'artist_mbid': track.get('artist', {}).get('mbid') if isinstance(track.get('artist'), dict) else None,
                # Additional Last.fm fields that might be useful
                'duration': track.get('duration'),
                'url': track.get('url'),
                'mbid': track.get('mbid'),
                'streamable': track.get('streamable'),
                'rank': track.get('@attr', {}).get('rank') if '@attr' in track else None,
                'userplaycount': track.get('userplaycount'),
                'userloved': track.get('userloved'),
                'album_artist': track.get('album', {}).get('artist') if 'album' in track else None,
                'album_url': track.get('album', {}).get('url') if 'album' in track else None,
                'album_image': track.get('album', {}).get('image') if 'album' in track else None,
            }
            
            # Filter to only lean fields for database
            return {k: v for k, v in all_lastfm_data.items() if k in LEAN_FIELDS and v is not None and v != ''}
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
               mfcc, chroma, spectral_contrast, spectral_flatness, spectral_rolloff, musicnn_embedding, metadata
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
                'spectral_contrast': row[12],
                'spectral_flatness': row[13],
                'spectral_rolloff': row[14],
                'musicnn_embedding': json.loads(row[15]) if row[15] else None,
                'metadata': json.loads(row[16]) if row[16] else {},
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
            'chroma': [0.0] * 12,  # 12-dimensional HPCP vector
            'spectral_contrast': 0.0,  # Single float value
            'spectral_flatness': 0.0,
            'spectral_rolloff': 0.0,
            'musicnn_embedding': None
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
            logger.info(f"Attempting MusicBrainz lookup for: {artist} - {title}")
            mb_tags = self._musicbrainz_lookup(artist, title)
            updated_fields_mb = [k for k, v in mb_tags.items() if v and (k not in meta or meta[k] != v)]
            meta.update({k: v for k, v in mb_tags.items() if v})
            logger.info(f"MusicBrainz enrichment: {artist} - {title} (fields updated: {updated_fields_mb})")
        # Last.fm enrichment
        genre = meta.get('genre')
        missing_fields = [field for field in ['genre', 'year', 'album'] if not meta.get(field)]
        if genre is None or (isinstance(genre, str) and genre.strip() in {None, '', 'Other', 'UnknownGenre', 'Unknown', 'Misc', 'Various', 'VA', 'General', 'Soundtrack', 'OST', 'N/A', 'Not Available', 'No Genre', 'Unclassified', 'Unsorted', 'Undefined', 'Genre', 'Genres', 'Music', 'Song', 'Songs', 'Audio', 'MP3', 'Instrumental', 'Vocal', 'Various Artists', 'VA', 'Compilation', 'Compilations', 'Album', 'Albums', 'CD', 'CDs', 'Record', 'Records', 'Single', 'Singles', 'EP', 'EPs', 'LP', 'LPs', 'Demo', 'Demos', 'Test', 'Tests', 'Sample', 'Samples', 'Example', 'Examples', 'Untitled', 'Unknown Artist', 'Unknown Album', 'Unknown Title', 'No Title', 'No Album', 'No Artist'}):
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
        # Filter metadata to keep only lean fields
        features['metadata'] = filter_metadata(meta)

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
                musicnn_embedding = self._extract_musicnn_embedding(audio_path)

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
                    'musicnn_embedding': musicnn_embedding
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
            # Debug: check database schema first
            cursor = self.conn.execute("PRAGMA table_info(audio_features)")
            columns = cursor.fetchall()
            logger.debug(f"Database columns: {[col[1] for col in columns]}")
            logger.debug(f"Number of columns: {len(columns)}")
            
            # Debug: count the values
            values_tuple = (
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
                features.get('spectral_contrast'),
                features.get('spectral_flatness'),
                features.get('spectral_rolloff'),
                json.dumps(features.get('musicnn_embedding', None)),
                file_info['last_modified'],
                json.dumps(features.get('metadata', {})),
                failed
            )
            logger.debug(f"Values count: {len(values_tuple)}")
            logger.debug(f"Values: {values_tuple}")
            
            with self.conn:
                self.conn.execute(
                    """
                    INSERT OR REPLACE INTO audio_features (
                        file_hash, file_path, duration, bpm, beat_confidence, centroid, loudness, danceability, key, scale, onset_rate, zcr,
                        mfcc, chroma, spectral_contrast, spectral_flatness, spectral_rolloff, musicnn_embedding,
                        last_modified, metadata, failed
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    values_tuple
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