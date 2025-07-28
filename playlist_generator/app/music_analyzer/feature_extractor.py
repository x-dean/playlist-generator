import numba
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
from utils.path_converter import PathConverter
import threading
from contextlib import contextmanager

# Set up logger first (before any imports that might use it)
logger = logging.getLogger(__name__)

# Import new robustness utility modules
try:
    from utils.circuit_breaker import CircuitBreaker, CircuitBreakerOpenError, circuit_breaker
    from utils.adaptive_timeout import AdaptiveTimeoutManager, get_timeout_manager, calculate_timeout
    from utils.error_classifier import ErrorClassifier, classify_error, ErrorType, ErrorSeverity, get_error_classifier
    from utils.smart_retry import SmartRetryManager, RetryStrategy, get_retry_manager, smart_retry
    from utils.timeout_manager import timeout_manager, timeout_context, TimeoutException
    from utils.error_recovery import error_handler, retry_with_backoff, safe_operation
    from utils.progress_tracker import ProgressTracker, ParallelProgressTracker
    ROBUSTNESS_UTILITIES_AVAILABLE = True
    logger.debug("Robustness utilities loaded successfully")
except ImportError as e:
    logger.warning(f"Robustness utility modules not available: {e}")
    ROBUSTNESS_UTILITIES_AVAILABLE = False
    
    # Fallback definitions for when robustness utilities are not available
    class CircuitBreaker:
        def __init__(self, *args, **kwargs):
            pass
        def call(self, func, *args, **kwargs):
            return func(*args, **kwargs)
    
    class CircuitBreakerOpenError(Exception):
        pass
    
    class AdaptiveTimeoutManager:
        def __init__(self, *args, **kwargs):
            pass
        def calculate_timeout(self, *args, **kwargs):
            return 60.0
    
    class ErrorClassifier:
        def __init__(self, *args, **kwargs):
            pass
        def classify_error(self, *args, **kwargs):
            return None
    
    class SmartRetryManager:
        def __init__(self, *args, **kwargs):
            pass
        def exponential_backoff(self, func, *args, **kwargs):
            return type('RetryResult', (), {'success': True, 'result': func()})()
    
    # Fallback functions
    def get_timeout_manager():
        return AdaptiveTimeoutManager()
    
    def get_error_classifier():
        return ErrorClassifier()
    
    def get_retry_manager():
        return SmartRetryManager()
    
    def calculate_timeout(*args, **kwargs):
        return 60.0
    
    def classify_error(*args, **kwargs):
        return None

# Check if Essentia was built with TensorFlow support (only once)
_essentia_tf_support_checked = False


def check_essentia_tf_support():
    """Check if Essentia has TensorFlow support for MusiCNN."""
    logger.debug("Checking Essentia TensorFlow support")
    try:
        import essentia.standard as es
        logger.debug("Essentia standard module imported successfully")

        # Try to import TensorFlowPredictMusiCNN
        try:
            from essentia.standard import TensorflowPredictMusiCNN
            logger.debug("TensorflowPredictMusiCNN import successful")
            logger.info("Essentia TensorFlow support: AVAILABLE")
            return True
        except ImportError as e:
            logger.debug(f"TensorflowPredictMusiCNN import failed: {e}")
            logger.warning(f"Essentia TensorFlow support: NOT AVAILABLE - {e}")
            logger.warning(
                "MusiCNN embeddings will not work. Install Essentia with TensorFlow support.")
            return False
    except ImportError as e:
        logger.error(f"Essentia import failed: {e}")
        return False


# Run the check once at module import
check_essentia_tf_support()

# Configure Numba to use fallback mode to avoid compilation errors
# Only disable CUDA JIT, keep CPU JIT for librosa
numba.config.CUDA_DISABLE_JIT = True

# Configure TensorFlow to suppress warnings (only if TensorFlow is used)
try:
    import tensorflow as tf
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.get_logger().setLevel('ERROR')
except ImportError:
    pass

# Lean metadata fields for playlisting
LEAN_FIELDS = [
    'artist', 'title', 'album', 'year', 'genre', 'tracknumber', 'discnumber',
    'composer', 'lyricist', 'arranger', 'publisher', 'label', 'isrc',
    'musicbrainz_id', 'isrc', 'mb_artist_id', 'mb_album_id',
    'release_date', 'country', 'work', 'composer', 'bpm',
    'genre_lastfm', 'album_lastfm', 'listeners', 'playcount', 'wiki', 'album_mbid', 'artist_mbid'
]


def filter_metadata(meta):
    """Filter metadata to keep only lean fields relevant for playlisting."""
    return {k: v for k, v in meta.items() if k in LEAN_FIELDS and v is not None and v != ''}


def safe_json_dumps(obj):
    """Safely serialize object to JSON, handling NumPy types."""
    if obj is None:
        return None
    try:
        return json.dumps(obj)
    except TypeError as e:
        if "not JSON serializable" in str(e):
            # Convert NumPy types to Python native types
            import numpy as np
            if isinstance(obj, dict):
                converted = {}
                for k, v in obj.items():
                    if isinstance(v, np.floating):
                        converted[k] = float(v)
                    elif isinstance(v, np.integer):
                        converted[k] = int(v)
                    elif isinstance(v, np.ndarray):
                        converted[k] = v.tolist()
                    elif isinstance(v, dict):
                        converted[k] = safe_json_dumps(v)
                    else:
                        converted[k] = v
                return json.dumps(converted)
            elif isinstance(obj, np.ndarray):
                return json.dumps(obj.tolist())
            elif isinstance(obj, np.floating):
                return json.dumps(float(obj))
            elif isinstance(obj, np.integer):
                return json.dumps(int(obj))
            else:
                return json.dumps(str(obj))
        else:
            raise e


def convert_to_python_types(obj):
    """Convert NumPy types to Python native types."""
    if isinstance(obj, dict):
        return {k: convert_to_python_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_python_types(v) for v in obj]
    elif hasattr(obj, 'dtype'):  # NumPy array or scalar
        if hasattr(obj, 'tolist'):
            return obj.tolist()
        else:
            return obj.item()
    else:
        return obj


def validate_and_convert_features(features):
    """Validate and convert features to ensure they're JSON serializable."""
    if not isinstance(features, dict):
        return None

    converted = {}
    for key, value in features.items():
        try:
            # Convert NumPy types to Python native types
            converted_value = convert_to_python_types(value)
            
            # Test JSON serialization
            json.dumps(converted_value)
            converted[key] = converted_value
        except (TypeError, ValueError) as e:
            logger.warning(f"Feature {key} failed validation: {e}")
            # Use a safe default value
            if isinstance(value, (list, tuple)):
                converted[key] = []
            elif isinstance(value, (int, float)):
                converted[key] = 0.0
            else:
                converted[key] = None

    return converted


class DatabaseConnectionPool:
    """Thread-safe database connection pool for SQLite."""
    
    def __init__(self, db_path: str, max_connections: int = 10):
        self.db_path = db_path
        self.max_connections = max_connections
        self._connections = []
        self._lock = threading.Lock()
        self._initialized = False
        
    def _initialize_pool(self):
        """Initialize the connection pool."""
        if self._initialized:
            return
            
        with self._lock:
            if self._initialized:
                return
                
            logger.debug(f"Initializing database connection pool with {self.max_connections} connections")
            for _ in range(self.max_connections):
                conn = sqlite3.connect(self.db_path, timeout=600)
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA busy_timeout=30000")
                conn.execute("PRAGMA synchronous=NORMAL")
                conn.execute("PRAGMA cache_size=10000")
                conn.execute("PRAGMA temp_store=MEMORY")
                conn.execute("PRAGMA mmap_size=268435456")  # 256MB memory mapping
                conn.execute("PRAGMA page_size=4096")
                conn.execute("PRAGMA auto_vacuum=INCREMENTAL")
                conn.execute("PRAGMA locking_mode=EXCLUSIVE")  # Better for concurrent access
                conn.execute("PRAGMA foreign_keys=OFF")  # Disable for performance
                self._connections.append(conn)
            self._initialized = True
            logger.debug("Database connection pool initialized")
    
    @contextmanager
    def get_connection(self):
        """Get a database connection from the pool."""
        self._initialize_pool()
        
        conn = None
        try:
            with self._lock:
                if not self._connections:
                    # Create a new connection if pool is empty
                    conn = sqlite3.connect(self.db_path, timeout=600)
                    conn.execute("PRAGMA journal_mode=WAL")
                    conn.execute("PRAGMA busy_timeout=30000")
                    conn.execute("PRAGMA synchronous=NORMAL")
                    conn.execute("PRAGMA cache_size=10000")
                    conn.execute("PRAGMA temp_store=MEMORY")
                    conn.execute("PRAGMA mmap_size=268435456")  # 256MB memory mapping
                    conn.execute("PRAGMA page_size=4096")
                    conn.execute("PRAGMA auto_vacuum=INCREMENTAL")
                    conn.execute("PRAGMA locking_mode=EXCLUSIVE")  # Better for concurrent access
                    conn.execute("PRAGMA foreign_keys=OFF")  # Disable for performance
                else:
                    conn = self._connections.pop()
            
            yield conn
        finally:
            if conn:
                # Return connection to pool (outside the lock to prevent deadlock)
                try:
                    with self._lock:
                        if len(self._connections) < self.max_connections:
                            self._connections.append(conn)
                        else:
                            conn.close()
                except Exception as e:
                    logger.debug(f"Error returning connection to pool: {e}")
                    try:
                        conn.close()
                    except:
                        pass
    
    def close_all(self):
        """Close all connections in the pool."""
        with self._lock:
            for conn in self._connections:
                try:
                    conn.close()
                except:
                    pass
            self._connections.clear()
            self._initialized = False


class TimeoutException(Exception):
    """Custom exception for timeout errors."""
    pass


def timeout(seconds=60, error_message="Processing timed out"):
    """Timeout decorator for long-running operations."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            def _handle_timeout(signum, frame):
                raise TimeoutException(error_message)
            
            import signal
            old_handler = signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
            return result
        return wrapper
    return decorator


def safe_essentia_call(func, *args, **kwargs):
    """Safely call Essentia functions with error handling."""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger.warning(f"Essentia call failed: {str(e)}")
        return None


class AudioAnalyzer:
    """Analyze audio files and extract features for playlist generation."""

    # Version identifier for tracking updates - added dynamic resource usage display
    VERSION = "4.20.0"

    def __init__(self, cache_file: str = None, library: str = None, music: str = None) -> None:
        """Initialize the AudioAnalyzer.

        Args:
            cache_file (str, optional): Path to the cache database file. Defaults to None.
            library (str, optional): Music library directory for path normalization.
            music (str, optional): Container music directory for path normalization.
        """
        logger.info(f"Initializing AudioAnalyzer version {self.VERSION}")
        self.timeout_seconds = 120
        cache_dir = os.getenv('CACHE_DIR', '/app/cache')
        self.cache_file = cache_file or os.path.join(
            cache_dir, 'audio_analysis.db')
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        
        # Initialize connection pool instead of single connection
        self.db_pool = DatabaseConnectionPool(self.cache_file)
        self._init_db()
        self.cleanup_database()  # Clean up immediately on init
        
        # Initialize robustness features
        if ROBUSTNESS_UTILITIES_AVAILABLE:
            self._init_robustness_features()
        
        # Set MusicBrainz user agent
        musicbrainzngs.set_useragent(
            "PlaylistGenerator", "1.0", "noreply@example.com")
        self.library = library
        self.music = music
        # Load TensorFlow models
        # self.vggish_model = self._load_vggish_model() # Removed VGGish model loading

    def _init_robustness_features(self):
        """Initialize robustness features for enhanced error handling and recovery."""
        logger.info("Initializing robustness features")
        
        # Initialize circuit breakers for different operations
        self.circuit_breakers = {
            'audio_processing': CircuitBreaker(
                failure_threshold=3,
                recovery_timeout=30,
                name='audio_processing'
            ),
            'database_operations': CircuitBreaker(
                failure_threshold=5,
                recovery_timeout=60,
                name='database_operations'
            ),
            'network_operations': CircuitBreaker(
                failure_threshold=3,
                recovery_timeout=120,
                name='network_operations'
            ),
            'feature_extraction': CircuitBreaker(
                failure_threshold=2,
                recovery_timeout=45,
                name='feature_extraction'
            )
        }
        
        # Initialize timeout manager
        self.timeout_manager = get_timeout_manager()
        
        # Initialize error classifier
        self.error_classifier = get_error_classifier()
        
        # Initialize retry manager
        self.retry_manager = get_retry_manager()
        
        # Register circuit breakers globally
        for name, breaker in self.circuit_breakers.items():
            register_circuit_breaker(name, breaker)
        
        logger.info(f"Initialized {len(self.circuit_breakers)} circuit breakers")
        logger.debug("Robustness features initialized successfully")

    # Removed _load_vggish_model

    # Removed _audio_to_mel_spectrogram

    def _extract_musicnn_embedding(self, audio_path):
        """Extract MusiCNN embedding and auto-tags using Essentia's TensorflowPredictMusiCNN and MonoLoader, matching the official tutorial."""
        logger.info(
            f"Starting MusiCNN embedding extraction for {os.path.basename(audio_path)}")
        try:
            import essentia.standard as es
            import numpy as np
            import json
            import sys
            
            # Check Python version compatibility
            if sys.version_info < (3, 6):
                logger.warning("Python version < 3.6 detected, MusicNN may have compatibility issues")
                return {'skipped': True, 'reason': 'python_version_too_old'}
            
            # Check TensorFlow availability
            try:
                import tensorflow as tf
                logger.debug(f"TensorFlow version: {tf.__version__}")
            except ImportError:
                logger.warning("TensorFlow not available, MusicNN will be skipped")
                return {'skipped': True, 'reason': 'tensorflow_not_available'}
            except Exception as e:
                logger.warning(f"TensorFlow import error: {str(e)}")
                return {'skipped': True, 'reason': 'tensorflow_import_error'}

            model_path = os.getenv(
                'MUSICNN_MODEL_PATH',
                '/app/feature_extraction/models/musicnn/msd-musicnn-1.pb'
            )
            json_path = os.getenv(
                'MUSICNN_JSON_PATH',
                '/app/feature_extraction/models/musicnn/msd-musicnn-1.json'
            )

            logger.info(f"MusiCNN model path: {model_path}")
            logger.info(f"MusiCNN JSON metadata path: {json_path}")

            if not os.path.exists(model_path):
                logger.warning(f"MusiCNN model not found at {model_path}")
                return {'skipped': True, 'reason': 'model_missing'}
            if not os.path.exists(json_path):
                logger.warning(
                    f"MusiCNN JSON metadata not found at {json_path}")
                return {'skipped': True, 'reason': 'json_missing'}

            logger.info("Loading MusiCNN tag names from JSON metadata")
            # Load tag names from JSON
            with open(json_path, 'r') as json_file:
                metadata = json.load(json_file)
            tag_names = metadata.get('classes', [])
            logger.info(
                f"Loaded {len(tag_names)} tag names from MusiCNN metadata")

            # Get output layer for embeddings
            output_layer = 'model/dense_1/BiasAdd'
            logger.info(f"Default MusiCNN output layer: {output_layer}")
            if 'schema' in metadata and 'outputs' in metadata['schema']:
                logger.info("Searching for embeddings output layer in schema")
                for output in metadata['schema']['outputs']:
                    if 'description' in output and output['description'] == 'embeddings':
                        output_layer = output['name']
                        logger.info(
                            f"Found embeddings output layer: {output_layer}")
                        break

            logger.info("Loading audio using Essentia MonoLoader at 16kHz")
            # Load audio using MonoLoader at 16kHz (tutorial pattern)
            audio = es.MonoLoader(filename=audio_path, sampleRate=16000)()
            logger.info(
                f"Loaded audio: {len(audio)} samples at 16kHz ({len(audio)/16000:.2f}s)")

            logger.info("Initializing MusiCNN for activations (auto-tagging)")
            # Run MusiCNN for activations (auto-tagging)
            try:
                musicnn = es.TensorflowPredictMusiCNN(graphFilename=model_path)
                logger.info("Running MusiCNN activations analysis")
                activations = musicnn(audio)  # shape: [time, tags]
            except Exception as e:
                logger.warning(f"MusiCNN activations initialization failed: {str(e)}")
                return {'skipped': True, 'reason': 'musicnn_initialization_failed'}
            
            # Handle different return types from MusicNN
            if isinstance(activations, list):
                logger.info(f"MusiCNN activations returned as list with {len(activations)} elements")
                # Convert list to numpy array if needed
                activations = np.array(activations)
            elif hasattr(activations, 'shape'):
                logger.info(f"MusiCNN activations shape: {activations.shape}")
            else:
                logger.warning(f"Unexpected activations type: {type(activations)}")
                return {'skipped': True, 'reason': 'unexpected_activations_type'}
            
            # Validate activations shape
            if len(activations.shape) != 2:
                logger.warning(f"Unexpected activations shape: {activations.shape}, expected 2D array")
                return {'skipped': True, 'reason': 'invalid_activations_shape'}

            tag_probs = activations.mean(axis=0)
            logger.info(
                f"Calculated tag probabilities: {len(tag_probs)} tags")
            # Convert NumPy types to Python native types for JSON serialization
            tags = dict(zip(tag_names, [float(prob) for prob in tag_probs]))
            logger.info(
                f"Top 5 predicted tags: {sorted(tags.items(), key=lambda x: x[1], reverse=True)[:5]}")

            logger.info(
                f"Initializing MusiCNN for embeddings using output layer: {output_layer}")
            # Run MusiCNN for embeddings (using correct output layer)
            try:
                musicnn_emb = es.TensorflowPredictMusiCNN(
                    graphFilename=model_path, output=output_layer)
                logger.info("Running MusiCNN embeddings analysis")
                embeddings = musicnn_emb(audio)
            except Exception as e:
                logger.warning(f"MusiCNN embeddings initialization failed: {str(e)}")
                return {'skipped': True, 'reason': 'musicnn_embeddings_failed'}
            
            # Handle different return types from MusicNN for embeddings
            if isinstance(embeddings, list):
                logger.info(f"MusiCNN embeddings returned as list with {len(embeddings)} elements")
                # Convert list to numpy array if needed
                embeddings = np.array(embeddings)
            elif hasattr(embeddings, 'shape'):
                logger.info(f"MusiCNN embeddings shape: {embeddings.shape}")
            else:
                logger.warning(f"Unexpected embeddings type: {type(embeddings)}")
                return {'skipped': True, 'reason': 'unexpected_embeddings_type'}
            
            # Validate embeddings shape
            if len(embeddings.shape) != 2:
                logger.warning(f"Unexpected embeddings shape: {embeddings.shape}, expected 2D array")
                return {'skipped': True, 'reason': 'invalid_embeddings_shape'}

            embedding = np.mean(embeddings, axis=0)
            logger.info(
                f"Calculated mean embedding: {len(embedding)} dimensions")
            logger.info(
                f"Embedding statistics: min={embedding.min():.3f}, max={embedding.max():.3f}, mean={embedding.mean():.3f}")

            result = {
                'embedding': embedding.tolist(),
                'tags': tags
            }
            logger.info("Successfully extracted MusiCNN embedding and tags")
            logger.info(
                f"MusiCNN extraction completed: {len(tags)} tags, {len(embedding)} embedding dimensions")
            return result
        except Exception as e:
            logger.warning(
                f"MusiCNN embedding/tag extraction failed: {str(e)}")
            logger.warning(f"Exception type: {type(e).__name__}")
            
            # Handle specific Python version compatibility issues
            if "flush" in str(e) and "print()" in str(e):
                logger.warning("Detected Python version compatibility issue with print() flush argument")
                return {'skipped': True, 'reason': 'python_version_compatibility'}
            
            import traceback
            logger.warning(f"Full traceback: {traceback.format_exc()}")
            return None

    def _init_db(self):
        """Initialize database tables using connection pool."""
        with self.db_pool.get_connection() as conn:
            conn.execute("""
            CREATE TABLE IF NOT EXISTS audio_features (
                file_hash TEXT PRIMARY KEY,
                file_path TEXT NOT NULL,
                duration REAL,
                bpm REAL,
                beat_confidence REAL,
                centroid REAL,
                loudness REAL,
                danceability REAL,
                key TEXT,
                scale TEXT,
                key_strength REAL,
                onset_rate REAL,
                zcr REAL,
                mfcc JSON,
                chroma JSON,
                spectral_contrast REAL,
                spectral_flatness REAL,
                spectral_rolloff REAL,
                musicnn_embedding JSON,
                musicnn_tags JSON,
                musicnn_skipped INTEGER DEFAULT 0,
                last_modified REAL,
                last_analyzed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata JSON,
                failed INTEGER DEFAULT 0
            )
            """)
            
            # Check if musicnn_skipped column exists, if not add it
            cursor = conn.execute("PRAGMA table_info(audio_features)")
            columns = [row[1] for row in cursor.fetchall()]
            if 'musicnn_skipped' not in columns:
                logger.info("Adding missing 'musicnn_skipped' column to audio_features table")
                conn.execute("ALTER TABLE audio_features ADD COLUMN musicnn_skipped INTEGER DEFAULT 0")
                logger.debug("Added musicnn_skipped column to audio_features table")

            # Create file discovery state table
            conn.execute("""
            CREATE TABLE IF NOT EXISTS file_discovery_state (
                file_path TEXT PRIMARY KEY,
                file_hash TEXT,
                file_size INTEGER,
                last_modified REAL,
                discovered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_seen_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'active'
            )
            """)

            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_file_path ON audio_features(file_path)")

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
            logger.warning(
                f"Couldn't get file stats for {library_path}: {str(e)}")
            return {
                'file_hash': hashlib.md5(library_path.encode()).hexdigest(),
                'last_modified': 0,
                'file_path': library_path
            }

    def _get_file_hash(self, filepath):
        """Get file hash for file discovery tracking."""
        import hashlib
        import traceback

        logger.debug(
            f"DISCOVERY: _get_file_hash called with filepath: {filepath}")
        # For container paths, use directly; for host paths, convert to container
        if filepath.startswith('/music'):
            # Already a container path
            container_path = filepath
        else:
            # Host path - convert to container
            container_path = self._normalize_to_library_path(filepath)

        logger.debug(
            f"DISCOVERY: _get_file_hash using container path: {container_path}")
        try:
            stat = os.stat(container_path)
            logger.debug(
                f"DISCOVERY: _get_file_hash successful stat for: {container_path}")
            return f"{os.path.basename(container_path)}_{stat.st_size}_{stat.st_mtime}"
        except Exception as e:
            logger.warning(
                f"Couldn't get file hash for {container_path}: {str(e)}")
            return hashlib.md5(container_path.encode()).hexdigest()

    def _normalize_to_library_path(self, path):
        """Convert any path to container path for file operations."""
        logger.debug(
            f"DISCOVERY: _normalize_to_library_path called with path: {path}")
        logger.debug(
            f"DISCOVERY: self.library: {self.library}, self.music: {self.music}")

        # If already a container path, return as is
        if path.startswith('/music'):
            logger.debug(f"DISCOVERY: Path is already container path: {path}")
            return path

        # Convert host path to container path
        if self.library and self.music:
            path_converter = PathConverter(self.library, self.music)
            if path.startswith(self.library):
                logger.debug(f"DISCOVERY: Converting host path to container")
                result = path_converter.host_to_container(path)
                logger.debug(f"DISCOVERY: Converted {path} -> {result}")
                return result
            else:
                logger.debug(
                    f"DISCOVERY: Path unclear, assuming host path and converting to container")
                result = path_converter.host_to_container(path)
                logger.debug(f"DISCOVERY: Converted {path} -> {result}")
                return result

        logger.debug(
            f"DISCOVERY: No library/music paths, returning normpath: {os.path.normpath(path)}")
        return os.path.normpath(path)

    def _safe_audio_load(self, audio_path):
        """Safely load audio file using Essentia."""
        logger.debug(f"Loading audio file: {os.path.basename(audio_path)}")
        try:
            import essentia.standard as es
            logger.debug("Initializing Essentia MonoLoader")
            loader = es.MonoLoader(filename=audio_path, sampleRate=44100)
            logger.debug("Running audio loading")
            audio = loader()
            logger.debug(
                f"Successfully loaded audio: {len(audio)} samples at 44.1kHz ({len(audio)/44100:.2f}s)")
            return audio
        except Exception as e:
            logger.error(f"Audio loading failed for {audio_path}: {str(e)}")
            return None

    def _extract_rhythm_features(self, audio, audio_path=None, metadata=None):
        """Extract rhythm features from audio with fallback to external APIs."""
        logger.info("Extracting rhythm features...")
        
        # Check if file is too large for Essentia rhythm extraction
        # Files larger than 150M samples (~5.7 hours at 44kHz) often cause buffer overflow
        if len(audio) > 150000000:
            logger.warning(f"File too large for rhythm extraction ({len(audio)} samples), skipping to external BPM lookup")
            external_bpm = self._get_external_bpm(audio_path, metadata)
            if external_bpm is not None:
                logger.info(f"Using external BPM for large file: {external_bpm:.1f}")
                return {'bpm': float(external_bpm)}
            else:
                logger.warning("No external BPM available for large file, using failed marker -1.0")
                return {'bpm': -1.0}
        
        try:
            logger.debug("Initializing Essentia RhythmExtractor algorithm")
            rhythm_algo = es.RhythmExtractor2013()
            logger.debug("Running rhythm analysis on audio")
            
            # Check if we're in a worker thread (signal only works in main thread)
            import threading
            is_main_thread = threading.current_thread() is threading.main_thread()
            
            # Add timeout protection for large files (only in main thread)
            if is_main_thread and len(audio) > 100000000:  # More than 100M samples
                import signal
                def timeout_handler(signum, frame):
                    raise TimeoutException("Rhythm extraction timed out")
                
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(600)  # 10 minutes for large files
                logger.debug("Set 10-minute timeout for large file rhythm extraction")
            
            try:
                rhythm_result = rhythm_algo(audio)
                logger.debug("Rhythm analysis completed successfully")
            except MemoryError as me:
                logger.error(f"Rhythm extraction failed due to memory error: {me}")
                raise
            except Exception as e:
                logger.error(f"Rhythm extraction failed with error: {e}")
                raise
            finally:
                # Cancel timeout (only in main thread)
                if is_main_thread and len(audio) > 100000000:
                    signal.alarm(0)
            logger.debug(f"Rhythm result type: {type(rhythm_result)}")
            logger.debug(
                f"Rhythm result length: {len(rhythm_result) if isinstance(rhythm_result, tuple) else 'not tuple'}")
            logger.debug(f"Rhythm result: {rhythm_result}")

            # Handle different return types from Essentia
            if isinstance(rhythm_result, tuple):
                # Try to get BPM from the first element
                if len(rhythm_result) > 0:
                    bpm = rhythm_result[0]
                    logger.debug(f"Extracted BPM from tuple[0]: {bpm}")
                else:
                    logger.warning("Empty rhythm result tuple")
                    bpm = -1.0  # Special marker for failed BPM extraction
            else:
                # Single value return
                bpm = rhythm_result
                logger.debug(f"Extracted BPM from single value: {bpm}")

            # Ensure BPM is a valid number
            try:
                bpm = float(bpm)
                if not np.isfinite(bpm) or bpm <= 0:
                    logger.warning(f"Invalid BPM value: {bpm}, using failed marker")
                    bpm = -1.0  # Special marker for failed BPM extraction
            except (ValueError, TypeError):
                logger.warning(
                    f"Could not convert BPM to float: {bpm}, using failed marker")
                bpm = -1.0  # Special marker for failed BPM extraction

            logger.debug(f"Final BPM: {bpm}")
            logger.info(f"Rhythm extraction completed: BPM = {bpm:.1f}")
            return {'bpm': float(bpm)}
        except TimeoutException as te:
            logger.error(f"Rhythm extraction timed out: {str(te)}")
            raise  # Re-raise to be caught by the main extraction function
        except Exception as e:
            logger.warning(f"Rhythm extraction failed: {str(e)}")
            logger.debug(
                f"Rhythm extraction error details: {type(e).__name__}")
            import traceback
            logger.debug(
                f"Rhythm extraction full traceback: {traceback.format_exc()}")
            
            # Try to get BPM from external APIs as fallback
            external_bpm = self._get_external_bpm(audio_path, metadata)
            if external_bpm is not None:
                logger.info(f"Using external BPM fallback: {external_bpm:.1f}")
                return {'bpm': float(external_bpm)}
            else:
                logger.warning("No external BPM available, using failed marker -1.0")
                return {'bpm': -1.0}  # Special marker for failed BPM extraction

    def _get_external_bpm(self, audio_path, metadata):
        """Get BPM from external APIs when local extraction fails."""
        logger.debug(f"External BPM lookup called with audio_path: {audio_path}")
        logger.debug(f"External BPM lookup called with metadata: {metadata}")
        
        if not audio_path:
            logger.debug("Missing audio_path for external BPM lookup")
            return None
            
        # Extract artist and title from metadata
        artist = metadata.get('artist') or metadata.get('mb_artist') if metadata else None
        title = metadata.get('title') or metadata.get('mb_title') if metadata else None
        
        # If metadata is empty, try to extract from filename
        if not artist or not title:
            logger.debug("No artist/title in metadata, trying filename parsing")
            try:
                filename = os.path.basename(audio_path)
                # Remove extension
                name_without_ext = os.path.splitext(filename)[0]
                
                # Try to parse "Artist  - Title" format
                if '  - ' in name_without_ext:
                    parts = name_without_ext.split('  - ', 1)
                    if len(parts) == 2:
                        artist = parts[0].strip()
                        title = parts[1].strip()
                        logger.debug(f"Extracted from filename - artist: '{artist}', title: '{title}'")
                    else:
                        logger.debug("Could not parse artist/title from filename")
                else:
                    logger.debug("Filename does not match expected format")
            except Exception as e:
                logger.debug(f"Filename parsing failed: {e}")
        
        logger.debug(f"Final artist: '{artist}', title: '{title}'")
        
        if not artist or not title:
            logger.debug("No artist/title available for external BPM lookup")
            return None
            
        logger.info(f"Attempting external BPM lookup for: {artist} - {title}")
        
        # Try MusicBrainz first
        try:
            logger.debug(f"Calling MusicBrainz lookup for: {artist} - {title}")
            mb_data = self._musicbrainz_lookup(artist, title)
            logger.debug(f"MusicBrainz lookup returned: {list(mb_data.keys()) if mb_data else 'None'}")
            if mb_data and 'bpm' in mb_data and mb_data['bpm']:
                bpm = float(mb_data['bpm'])
                if 60 <= bpm <= 200:  # Valid BPM range
                    logger.info(f"Found BPM in MusicBrainz: {bpm:.1f}")
                    return bpm
                else:
                    logger.debug(f"BPM from MusicBrainz ({bpm:.1f}) outside valid range (60-200)")
            else:
                logger.debug(f"No BPM found in MusicBrainz data: {mb_data.get('bpm', 'Not present')}")
        except Exception as e:
            logger.debug(f"MusicBrainz BPM lookup failed: {e}")
            logger.debug(f"MusicBrainz lookup exception type: {type(e).__name__}")
            import traceback
            logger.debug(f"MusicBrainz lookup full traceback: {traceback.format_exc()}")
            
        # Try LastFM as fallback
        try:
            lf_data = self._lastfm_lookup(artist, title)
            if lf_data and 'bpm' in lf_data and lf_data['bpm']:
                bpm = float(lf_data['bpm'])
                if 60 <= bpm <= 200:  # Valid BPM range
                    logger.info(f"Found BPM in LastFM: {bpm:.1f}")
                    return bpm
        except Exception as e:
            logger.debug(f"LastFM BPM lookup failed: {e}")
            
        logger.debug("No external BPM found")
        return None

    def _extract_spectral_features(self, audio):
        """Extract spectral features from audio."""
        logger.info("Extracting spectral features...")
        try:
            logger.debug(
                "Initializing Essentia SpectralCentroidTime algorithm")
            centroid_algo = es.SpectralCentroidTime()
            logger.debug("Running spectral centroid analysis on audio")
            
            # Check if we're in a worker thread (signal only works in main thread)
            import threading
            is_main_thread = threading.current_thread() is threading.main_thread()
            
            # Add timeout protection for large files (only in main thread)
            if is_main_thread and len(audio) > 100000000:  # More than 100M samples
                import signal
                def timeout_handler(signum, frame):
                    raise TimeoutException("Spectral extraction timed out")
                
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(600)  # 10 minutes for large files
                logger.debug("Set 10-minute timeout for large file spectral extraction")
            
            try:
                centroid_values = centroid_algo(audio)
                logger.debug("Spectral analysis completed successfully")
            except MemoryError as me:
                logger.error(f"Spectral extraction failed due to memory error: {me}")
                raise
            except Exception as e:
                logger.error(f"Spectral extraction failed with error: {e}")
                raise
            finally:
                # Cancel timeout (only in main thread)
                if is_main_thread and len(audio) > 100000000:
                    signal.alarm(0)
            logger.debug(
                f"Spectral centroid values shape: {np.array(centroid_values).shape if hasattr(centroid_values, 'shape') else type(centroid_values)}")

            centroid_mean = float(np.nanmean(centroid_values)) if isinstance(
                centroid_values, (list, np.ndarray)) else float(centroid_values)
            logger.debug(
                f"Calculated mean spectral centroid: {centroid_mean:.1f}")
            logger.info(
                f"Spectral features completed: centroid = {centroid_mean:.1f}Hz")
            return {'spectral_centroid': centroid_mean}
        except TimeoutException as te:
            logger.error(f"Spectral extraction timed out: {str(te)}")
            raise  # Re-raise to be caught by the main extraction function
        except Exception as e:
            logger.warning(f"Spectral features extraction failed: {str(e)}")
            logger.debug(
                f"Spectral features extraction error details: {type(e).__name__}")
            return {'spectral_centroid': 0.0}

    def _extract_loudness(self, audio):
        """Extract loudness features from audio."""
        logger.info("Extracting loudness features...")
        try:
            logger.debug("Initializing Essentia RMS algorithm")
            rms_algo = es.RMS()
            logger.debug("Running RMS analysis on audio")
            
            # Check if we're in a worker thread (signal only works in main thread)
            import threading
            is_main_thread = threading.current_thread() is threading.main_thread()
            
            # Add timeout protection for large files (only in main thread)
            if is_main_thread and len(audio) > 100000000:  # More than 100M samples
                import signal
                def timeout_handler(signum, frame):
                    raise TimeoutException("Loudness extraction timed out")
                
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(600)  # 10 minutes for large files
                logger.debug("Set 10-minute timeout for large file loudness extraction")
            
            try:
                rms_values = rms_algo(audio)
                logger.debug("Loudness analysis completed successfully")
            except MemoryError as me:
                logger.error(f"Loudness extraction failed due to memory error: {me}")
                raise
            except Exception as e:
                logger.error(f"Loudness extraction failed with error: {e}")
                raise
            finally:
                # Cancel timeout (only in main thread)
                if is_main_thread and len(audio) > 100000000:
                    signal.alarm(0)
            logger.debug(
                f"RMS values shape: {np.array(rms_values).shape if hasattr(rms_values, 'shape') else type(rms_values)}")

            rms_mean = float(np.nanmean(rms_values)) if isinstance(
                rms_values, (list, np.ndarray)) else float(rms_values)
            logger.debug(f"Calculated mean RMS: {rms_mean:.3f}")
            logger.info(f"Loudness extraction completed: RMS = {rms_mean:.3f}")
            return {'rms': rms_mean}
        except TimeoutException as te:
            logger.error(f"Loudness extraction timed out: {str(te)}")
            raise  # Re-raise to be caught by the main extraction function
        except Exception as e:
            logger.warning(f"Loudness extraction failed: {str(e)}")
            logger.debug(
                f"Loudness extraction error details: {type(e).__name__}")
            return {'rms': 0.0}

    def _extract_danceability(self, audio):
        """Extract danceability features from audio."""
        logger.info("Extracting danceability features...")
        try:
            logger.debug("Initializing Essentia Danceability algorithm")
            dance_algo = es.Danceability()
            logger.debug("Running danceability analysis on audio")
            dance_result = dance_algo(audio)
            logger.debug(f"Danceability result type: {type(dance_result)}")
            logger.debug(f"Danceability result: {dance_result}")

            # Handle different return types from Essentia
            if isinstance(dance_result, tuple):
                logger.debug(f"Danceability tuple length: {len(dance_result)}")
                for i, item in enumerate(dance_result):
                    logger.debug(
                        f"Danceability tuple[{i}]: {type(item)} = {item}")
                if len(dance_result) >= 1:
                    dance_values = dance_result[0]
                    logger.debug(
                        f"Extracted danceability from tuple[0]: {dance_values}")
                else:
                    logger.warning("Empty danceability result tuple")
                    dance_values = [0.0]
            else:
                dance_values = dance_result
                logger.debug(
                    f"Extracted danceability from single value: {dance_values}")

            # Handle numpy arrays
            if isinstance(dance_values, np.ndarray):
                if dance_values.size == 1:
                    dance_mean = float(dance_values.item())
                    logger.debug(
                        f"Converted single-element array to float: {dance_mean}")
                else:
                    dance_mean = float(np.nanmean(dance_values))
                    logger.debug(f"Calculated mean from array: {dance_mean}")
            elif isinstance(dance_values, (list, tuple)):
                dance_mean = float(np.nanmean(dance_values))
                logger.debug(f"Calculated mean from list/tuple: {dance_mean}")
            else:
                dance_mean = float(dance_values)
                logger.debug(f"Converted to float: {dance_mean}")

            # Ensure danceability is a valid number and normalize if needed
            if not np.isfinite(dance_mean):
                logger.warning(
                    f"Invalid danceability value (non-finite): {dance_mean}, using default")
                dance_mean = 0.0
            elif dance_mean < 0:
                logger.warning(
                    f"Negative danceability value: {dance_mean}, using default")
                dance_mean = 0.0
            elif dance_mean > 1:
                # The algorithm might return values in a different scale
                # Try to normalize or use as-is if it's reasonable
                if dance_mean <= 10:  # If it's in 0-10 scale, normalize
                    dance_mean = dance_mean / 10.0
                    logger.debug(
                        f"Normalized danceability from 0-10 scale: {dance_mean:.3f}")
                elif dance_mean <= 100:  # If it's in 0-100 scale, normalize
                    dance_mean = dance_mean / 100.0
                    logger.debug(
                        f"Normalized danceability from 0-100 scale: {dance_mean:.3f}")
                else:
                    logger.warning(
                        f"Danceability value out of expected range: {dance_mean}, using default")
                    dance_mean = 0.0

            logger.debug(f"Final danceability: {dance_mean}")
            logger.info(f"Danceability extraction completed: {dance_mean:.3f}")
            return {'danceability': dance_mean}
        except Exception as e:
            logger.warning(f"Danceability extraction failed: {str(e)}")
            logger.debug(
                f"Danceability extraction error details: {type(e).__name__}")
            import traceback
            logger.debug(
                f"Danceability extraction full traceback: {traceback.format_exc()}")
            return {'danceability': 0.0}

    def _extract_key(self, audio):
        """Extract key features from audio."""
        logger.info("Extracting key features...")
        try:
            logger.debug("Initializing Essentia KeyExtractor algorithm")
            key_algo = es.KeyExtractor()
            logger.debug("Running key analysis on audio")
            key, scale, strength = key_algo(audio)
            logger.debug(f"Extracted key: {key} {scale}, strength: {strength}")
            logger.info(
                f"Key extraction completed: {key} {scale} (strength: {strength:.3f})")
            return {'key': key, 'scale': scale, 'key_strength': float(strength)}
        except Exception as e:
            logger.warning(f"Key extraction failed: {str(e)}")
            logger.debug(f"Key extraction error details: {type(e).__name__}")
            return {'key': 'C', 'scale': 'major', 'key_strength': 0.0}

    def _extract_onset_rate(self, audio):
        """Extract onset rate features from audio."""
        logger.info("Extracting onset rate features...")
        try:
            logger.debug("Initializing Essentia OnsetRate algorithm")
            onset_algo = es.OnsetRate()
            logger.debug("Running onset rate analysis on audio")
            onset_result = onset_algo(audio)
            logger.debug(f"Onset rate result type: {type(onset_result)}")
            logger.debug(f"Onset rate result: {onset_result}")

            # Handle different return types from Essentia
            if isinstance(onset_result, tuple):
                if len(onset_result) >= 1:
                    onset_rate = onset_result[0]
                    logger.debug(
                        f"Extracted onset rate from tuple[0]: {onset_rate}")
                else:
                    logger.warning("Empty onset rate result tuple")
                    onset_rate = 0.0
            else:
                onset_rate = onset_result
                logger.debug(
                    f"Extracted onset rate from single value: {onset_rate}")

            # Handle numpy arrays
            if isinstance(onset_rate, np.ndarray):
                if onset_rate.size == 1:
                    onset_rate = float(onset_rate.item())
                    logger.debug(
                        f"Converted single-element array to float: {onset_rate}")
                else:
                    onset_rate = float(np.nanmean(onset_rate))
                    logger.debug(f"Calculated mean from array: {onset_rate}")
            else:
                # Convert to float
                onset_rate = float(onset_rate)
                logger.debug(f"Converted to float: {onset_rate}")

            # Ensure onset rate is a valid number
            if not np.isfinite(onset_rate) or onset_rate < 0:
                logger.warning(
                    f"Invalid onset rate value: {onset_rate}, using default")
                onset_rate = 0.0

            logger.debug(f"Final onset rate: {onset_rate}")
            logger.info(
                f"Onset rate extraction completed: {onset_rate:.2f} onsets/sec")
            return {'onset_rate': float(onset_rate)}
        except Exception as e:
            logger.warning(f"Onset rate extraction failed: {str(e)}")
            logger.debug(
                f"Onset rate extraction error details: {type(e).__name__}")
            import traceback
            logger.debug(
                f"Onset rate extraction full traceback: {traceback.format_exc()}")
            return {'onset_rate': 0.0}

    def _extract_zcr(self, audio):
        """Extract zero crossing rate features from audio."""
        logger.info("Extracting zero crossing rate features...")
        try:
            logger.debug("Initializing Essentia ZeroCrossingRate algorithm")
            zcr_algo = es.ZeroCrossingRate()
            logger.debug("Running zero crossing rate analysis on audio")
            zcr_values = zcr_algo(audio)
            logger.debug(
                f"ZCR values shape: {np.array(zcr_values).shape if hasattr(zcr_values, 'shape') else type(zcr_values)}")

            zcr_mean = float(np.nanmean(zcr_values)) if isinstance(
                zcr_values, (list, np.ndarray)) else float(zcr_values)
            logger.debug(f"Calculated mean ZCR: {zcr_mean:.3f}")
            logger.info(
                f"Zero crossing rate extraction completed: {zcr_mean:.3f}")
            return {'zcr': zcr_mean}
        except Exception as e:
            logger.warning(f"Zero crossing rate extraction failed: {str(e)}")
            logger.debug(
                f"Zero crossing rate extraction error details: {type(e).__name__}")
            return {'zcr': 0.0}

    def _extract_mfcc(self, audio, num_coeffs=13):
        """Extract MFCC coefficients from audio."""
        logger.info("Extracting MFCC coefficients...")
        try:
            logger.debug("Initializing Essentia MFCC algorithm")
            mfcc_algo = es.MFCC(numberCoefficients=num_coeffs)
            logger.debug("Running MFCC analysis on audio")

            # Check available memory before processing
            import psutil
            memory_info = psutil.virtual_memory()
            logger.debug(f"Available memory before MFCC: {memory_info.available / (1024**3):.1f}GB")
            logger.debug(f"Audio samples: {len(audio)}, estimated memory needed: {len(audio) * 8 / (1024**3):.1f}GB")

            # Check if we're in a worker thread (signal only works in main thread)
            import threading
            is_main_thread = threading.current_thread() is threading.main_thread()
            
            # For very large files, use a timeout (only in main thread)
            if is_main_thread and len(audio) > 100000000:  # More than 100M samples (~2 hours at 44kHz)
                import signal
                def timeout_handler(signum, frame):
                    raise TimeoutException("MFCC extraction timed out")
                
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(900)  # 15 minutes
                logger.debug(
                    "Set 15-minute timeout for large file MFCC extraction")

            try:
                logger.debug(f"Starting MFCC computation for {len(audio)} samples")
                
                # Check if audio data is valid
                if len(audio) == 0:
                    logger.error("MFCC extraction failed: Audio data is empty")
                    raise ValueError("Audio data is empty")
                
                # Check for NaN or infinite values
                import numpy as np
                if hasattr(audio, 'dtype') and np.issubdtype(audio.dtype, np.floating):
                    if np.any(np.isnan(audio)) or np.any(np.isinf(audio)):
                        logger.error("MFCC extraction failed: Audio contains NaN or infinite values")
                        raise ValueError("Audio contains NaN or infinite values")
                
                _, mfcc_coeffs = mfcc_algo(audio)
                logger.debug("MFCC computation completed successfully")
            except MemoryError as me:
                logger.error(f"MFCC extraction failed due to memory error: {me}")
                memory_info = psutil.virtual_memory()
                logger.error(f"Memory after failure: {memory_info.available / (1024**3):.1f}GB available")
                raise
            except Exception as e:
                logger.error(f"MFCC extraction failed with error: {e}")
                logger.error(f"Error type: {type(e).__name__}")
                import traceback
                logger.error(f"MFCC error traceback: {traceback.format_exc()}")
                raise
            finally:
                # Cancel timeout (only in main thread)
                if is_main_thread and len(audio) > 100000000:
                    signal.alarm(0)
            logger.debug(f"MFCC coefficients type: {type(mfcc_coeffs)}")
            logger.debug(
                f"MFCC coefficients shape: {np.array(mfcc_coeffs).shape if hasattr(mfcc_coeffs, 'shape') else 'no shape'}")

            # Handle different return types from Essentia MFCC
            if isinstance(mfcc_coeffs, (list, np.ndarray)):
                if len(mfcc_coeffs) > 0:
                    # If it's a 2D array (time x coefficients)
                    if hasattr(mfcc_coeffs, 'shape') and len(mfcc_coeffs.shape) == 2:
                        mfcc_mean = np.mean(mfcc_coeffs, axis=0).tolist()
                        logger.debug(
                            f"Calculated mean MFCC coefficients from 2D array: {len(mfcc_mean)} values")
                    # If it's a 1D array (single frame)
                    else:
                        mfcc_mean = np.array(mfcc_coeffs).tolist()
                        logger.debug(
                            f"Using single frame MFCC coefficients: {len(mfcc_mean)} values")
                else:
                    logger.warning("MFCC coefficients array is empty")
                    mfcc_mean = [0.0] * num_coeffs
            else:
                logger.warning(
                    f"Unexpected MFCC coefficients type: {type(mfcc_coeffs)}")
                mfcc_mean = [0.0] * num_coeffs

            # Ensure we have the right number of coefficients
            if len(mfcc_mean) != num_coeffs:
                logger.warning(
                    f"MFCC coefficients length mismatch: got {len(mfcc_mean)}, expected {num_coeffs}")
                if len(mfcc_mean) < num_coeffs:
                    # Pad with zeros
                    mfcc_mean.extend([0.0] * (num_coeffs - len(mfcc_mean)))
                else:
                    # Truncate
                    mfcc_mean = mfcc_mean[:num_coeffs]

            logger.debug(f"Final MFCC coefficients: {len(mfcc_mean)} values")
            logger.debug(
                f"MFCC coefficient range: min={min(mfcc_mean):.3f}, max={max(mfcc_mean):.3f}")
            logger.info(
                f"MFCC extraction completed: {len(mfcc_mean)} coefficients")
            return {'mfcc': mfcc_mean}
        except TimeoutException as te:
            logger.error(f"MFCC extraction timed out: {str(te)}")
            raise  # Re-raise to be caught by the main extraction function
        except Exception as e:
            logger.warning(f"MFCC extraction failed: {str(e)}")
            logger.debug(f"MFCC extraction error details: {type(e).__name__}")
            import traceback
            logger.debug(
                f"MFCC extraction full traceback: {traceback.format_exc()}")
            return {'mfcc': [0.0] * num_coeffs}

    def _extract_chroma(self, audio):
        """Extract chroma features from audio using HPCP."""
        logger.info("Extracting chroma features...")
        try:
            # Set up parameters
            frame_size = 2048
            hop_size = 1024
            logger.debug(
                f"Chroma extraction parameters: frame_size={frame_size}, hop_size={hop_size}")

            # Initialize algorithms
            logger.debug(
                "Initializing Essentia algorithms for chroma extraction")
            window = es.Windowing(type='blackmanharris62')
            spectrum = es.Spectrum()
            spectral_peaks = es.SpectralPeaks()
            hpcp = es.HPCP()

            logger.debug("Running chroma extraction pipeline")
            # Process audio in frames
            chroma_values = []
            frame_count = 0

            for frame in es.FrameGenerator(audio, frameSize=frame_size, hopSize=hop_size):
                frame_count += 1
                if frame_count % 100 == 0:
                    logger.debug(
                        f"Processed {frame_count} frames for chroma extraction")

                try:
                    logger.debug(
                        f"Processing frame {frame_count} of length {len(frame)}")
                    windowed = window(frame)
                    spec = spectrum(windowed)
                    frequencies, magnitudes = spectral_peaks(spec)

                    # Check if we have valid spectral peaks
                    if len(frequencies) > 0 and len(magnitudes) > 0:
                        # Ensure frequencies and magnitudes are numpy arrays
                        freq_array = np.array(frequencies)
                        mag_array = np.array(magnitudes)

                        # Check for valid frequency and magnitude ranges
                        if len(freq_array) > 0 and len(mag_array) > 0 and np.any(mag_array > 0):
                            hpcp_value = hpcp(freq_array, mag_array)
                            if hpcp_value is not None and len(hpcp_value) == 12:
                                chroma_values.append(hpcp_value)
                                logger.debug(
                                    f"Frame {frame_count}: valid HPCP value with {len(hpcp_value)} dimensions")
                        else:
                            logger.debug(
                                f"Frame {frame_count}: no valid frequencies/magnitudes")
                    else:
                        logger.debug(
                            f"Frame {frame_count}: no spectral peaks found")

                except Exception as frame_error:
                    logger.debug(
                        f"Frame {frame_count} processing error: {frame_error}")
                    continue

            logger.debug(
                f"Extracted {len(chroma_values)} chroma frames from {frame_count} total frames")

            # Calculate global average
            if chroma_values:
                chroma_avg = np.mean(chroma_values, axis=0).tolist()
                logger.debug(
                    f"Calculated mean chroma features: {len(chroma_avg)} values")
                logger.debug(
                    f"Chroma feature range: min={min(chroma_avg):.3f}, max={max(chroma_avg):.3f}")
                logger.info(
                    f"Chroma extraction completed: {len(chroma_avg)} features from {frame_count} frames")
                return {'chroma': chroma_avg}
            else:
                logger.debug(
                    "No valid chroma values calculated, returning default")
                logger.info(
                    "Chroma extraction completed: using default values (no valid frames)")
                return {'chroma': [0.0] * 12}

        except Exception as e:
            logger.warning(f"Chroma extraction failed: {str(e)}")
            logger.debug(
                f"Chroma extraction error details: {type(e).__name__}")
            import traceback
            logger.debug(
                f"Chroma extraction full traceback: {traceback.format_exc()}")
            return {'chroma': [0.0] * 12}

    def _extract_spectral_contrast(self, audio):
        """Extract spectral contrast features from audio."""
        logger.debug("Starting spectral contrast extraction with Essentia")
        try:
            # Use frame-by-frame processing for spectral contrast
            frame_size = 2048
            hop_size = 1024
            logger.debug(
                f"Spectral contrast parameters: frame_size={frame_size}, hop_size={hop_size}")

            logger.debug(
                "Initializing Essentia algorithms for spectral contrast")
            window = es.Windowing(type='hann')
            spectrum = es.Spectrum()
            spectral_peaks = es.SpectralPeaks()

            contrast_list = []
            frame_count = 0

            logger.debug("Running spectral contrast analysis frame by frame")
            # Process audio frame by frame
            for frame in es.FrameGenerator(audio, frameSize=frame_size, hopSize=hop_size, startFromZero=True):
                frame_count += 1
                if frame_count % 100 == 0:
                    logger.debug(
                        f"Processed {frame_count} frames for spectral contrast")

                try:
                    spec = spectrum(window(frame))
                    freqs, mags = spectral_peaks(spec)

                    if len(freqs) > 0 and len(mags) > 0:
                        # Ensure mags is a numpy array and has valid values
                        mags_array = np.array(mags)
                        if len(mags_array) > 0 and np.any(mags_array > 0):
                            # Calculate spectral contrast manually
                            # Sort magnitudes and find valleys
                            sorted_mags = np.sort(mags_array)
                            # Ensure at least 1 element
                            third = max(1, len(sorted_mags) // 3)
                            valleys = sorted_mags[:third]  # Bottom third
                            peaks = sorted_mags[-third:]   # Top third

                            if len(peaks) > 0 and len(valleys) > 0:
                                contrast = float(
                                    np.mean(peaks) - np.mean(valleys))
                                contrast_list.append(contrast)
                                logger.debug(
                                    f"Frame {frame_count}: contrast={contrast:.3f}")
                            else:
                                logger.debug(
                                    f"Frame {frame_count}: insufficient peaks/valleys for contrast")
                        else:
                            logger.debug(
                                f"Frame {frame_count}: no valid magnitudes")
                    else:
                        logger.debug(
                            f"Frame {frame_count}: no spectral peaks found")

                except Exception as frame_error:
                    logger.debug(
                        f"Frame {frame_count} processing error: {frame_error}")
                    continue

            logger.debug(
                f"Processed {frame_count} frames, calculated contrast for {len(contrast_list)} frames")
            # Return mean contrast across all frames
            if contrast_list:
                contrast_mean = float(np.mean(contrast_list))
                logger.debug(
                    f"Calculated mean spectral contrast: {contrast_mean:.3f}")
                return {'spectral_contrast': contrast_mean}
            else:
                logger.debug(
                    "No valid contrast values calculated, returning 0.0")
                return {'spectral_contrast': 0.0}

        except Exception as e:
            logger.warning(f"Spectral contrast extraction failed: {str(e)}")
            logger.debug(
                f"Spectral contrast extraction error details: {type(e).__name__}")
            import traceback
            logger.debug(
                f"Spectral contrast full traceback: {traceback.format_exc()}")
            return {'spectral_contrast': 0.0}

    def _extract_spectral_flatness(self, audio):
        """Extract spectral flatness from audio using frame-by-frame processing."""
        logger.info("Extracting spectral flatness...")
        try:
            # Use frame-by-frame processing for spectral flatness
            frame_size = 2048
            hop_size = 1024
            logger.debug(
                f"Spectral flatness parameters: frame_size={frame_size}, hop_size={hop_size}")

            logger.debug(
                "Initializing Essentia algorithms for spectral flatness")
            window = es.Windowing(type='hann')
            spectrum = es.Spectrum()

            flatness_list = []
            frame_count = 0

            logger.debug("Running spectral flatness analysis frame by frame")
            # Process audio frame by frame
            for frame in es.FrameGenerator(audio, frameSize=frame_size, hopSize=hop_size, startFromZero=True):
                frame_count += 1
                if frame_count % 100 == 0:
                    logger.debug(
                        f"Processed {frame_count} frames for spectral flatness")

                try:
                    spec = spectrum(window(frame))
                    # Calculate spectral flatness manually
                    # Spectral flatness = geometric_mean / arithmetic_mean
                    if len(spec) > 0 and np.any(spec > 0):
                        # Avoid log(0) by adding small epsilon
                        eps = 1e-10
                        spec_safe = spec + eps
                        geometric_mean = np.exp(np.mean(np.log(spec_safe)))
                        arithmetic_mean = np.mean(spec_safe)
                        flatness = geometric_mean / arithmetic_mean if arithmetic_mean > 0 else 0.0
                        flatness_list.append(float(flatness))
                        logger.debug(
                            f"Frame {frame_count}: flatness={flatness:.3f}")
                    else:
                        logger.debug(
                            f"Frame {frame_count}: no valid spectrum for flatness")
                except Exception as frame_error:
                    logger.debug(
                        f"Frame {frame_count} processing error: {frame_error}")
                    continue

            logger.debug(
                f"Processed {frame_count} frames, calculated flatness for {len(flatness_list)} frames")
            # Return mean flatness across all frames
            if flatness_list:
                flatness_mean = float(np.mean(flatness_list))
                logger.debug(
                    f"Calculated mean spectral flatness: {flatness_mean:.3f}")
                logger.info(
                    f"Spectral flatness completed: {flatness_mean:.3f} from {frame_count} frames")
                return {'spectral_flatness': flatness_mean}
            else:
                logger.debug(
                    "No valid flatness values calculated, returning 0.0")
                logger.info(
                    "Spectral flatness completed: using default value (no valid frames)")
                return {'spectral_flatness': 0.0}

        except Exception as e:
            logger.warning(f"Spectral flatness extraction failed: {str(e)}")
            logger.debug(
                f"Spectral flatness extraction error details: {type(e).__name__}")
            return {'spectral_flatness': 0.0}

    def _extract_spectral_rolloff(self, audio):
        """Extract spectral rolloff from audio using frame-by-frame processing."""
        logger.info("Extracting spectral rolloff...")
        try:
            # Use frame-by-frame processing for spectral rolloff
            frame_size = 2048
            hop_size = 1024
            logger.debug(
                f"Spectral rolloff parameters: frame_size={frame_size}, hop_size={hop_size}")

            logger.debug(
                "Initializing Essentia algorithms for spectral rolloff")
            window = es.Windowing(type='hann')
            spectrum = es.Spectrum()

            rolloff_list = []
            frame_count = 0

            logger.debug("Running spectral rolloff analysis frame by frame")
            # Process audio frame by frame
            for frame in es.FrameGenerator(audio, frameSize=frame_size, hopSize=hop_size, startFromZero=True):
                frame_count += 1
                if frame_count % 100 == 0:
                    logger.debug(
                        f"Processed {frame_count} frames for spectral rolloff")

                try:
                    spec = spectrum(window(frame))
                    # Calculate spectral rolloff manually
                    # Spectral rolloff is the frequency below which 85% of the energy is contained
                    if len(spec) > 0 and np.any(spec > 0):
                        # Calculate cumulative energy
                        energy = spec ** 2
                        total_energy = np.sum(energy)
                        if total_energy > 0:
                            cumulative_energy = np.cumsum(energy)
                            # Find frequency below which 85% of energy is contained
                            threshold = 0.85 * total_energy
                            rolloff_idx = np.where(
                                cumulative_energy >= threshold)[0]
                            if len(rolloff_idx) > 0:
                                # Convert to frequency (assuming 44.1kHz sample rate)
                                # Nyquist frequency
                                rolloff_freq = (
                                    rolloff_idx[0] / len(spec)) * 22050
                                rolloff_list.append(float(rolloff_freq))
                                logger.debug(
                                    f"Frame {frame_count}: rolloff={rolloff_freq:.1f}Hz")
                            else:
                                logger.debug(
                                    f"Frame {frame_count}: no rolloff found")
                        else:
                            logger.debug(
                                f"Frame {frame_count}: no energy for rolloff")
                    else:
                        logger.debug(
                            f"Frame {frame_count}: no valid spectrum for rolloff")
                except Exception as frame_error:
                    logger.debug(
                        f"Frame {frame_count} processing error: {frame_error}")
                    continue

            logger.debug(
                f"Processed {frame_count} frames, calculated rolloff for {len(rolloff_list)} frames")
            # Return mean rolloff across all frames
            if rolloff_list:
                rolloff_mean = float(np.mean(rolloff_list))
                logger.debug(
                    f"Calculated mean spectral rolloff: {rolloff_mean:.1f}Hz")
                logger.info(
                    f"Spectral rolloff completed: {rolloff_mean:.1f}Hz from {frame_count} frames")
                return {'spectral_rolloff': rolloff_mean}
            else:
                logger.debug(
                    "No valid rolloff values calculated, returning 0.0")
                logger.info(
                    "Spectral rolloff completed: using default value (no valid frames)")
                return {'spectral_rolloff': 0.0}

        except Exception as e:
            logger.warning(f"Spectral rolloff extraction failed: {str(e)}")
            logger.debug(
                f"Spectral rolloff extraction error details: {type(e).__name__}")
            return {'spectral_rolloff': 0.0}

    def _musicbrainz_lookup(self, artist, title):
        try:
            # Step 1: Search for the recording with exact match
            result = musicbrainzngs.search_recordings(
                artist=artist, recording=title, limit=1)
            
            # If no results, try with just the title (more flexible search)
            if not result.get('recording-list') or len(result['recording-list']) == 0:
                logger.debug(f"No exact match found for {artist} - {title}, trying title-only search")
                result = musicbrainzngs.search_recordings(
                    recording=title, limit=1)
            
            if not result.get('recording-list') or len(result['recording-list']) == 0:
                logger.debug(
                    f"No MusicBrainz results found for {artist} - {title}")
                return {}

            rec = result['recording-list'][0]
            mbid = rec['id']
            logger.debug(f"Found MusicBrainz recording: {mbid}")

            # Step 2: Lookup by MBID for full info - get everything available
            rec_full = musicbrainzngs.get_recording_by_id(
                mbid,
                includes=[
                    'artists', 'releases', 'tags', 'isrcs', 'work-rels', 'artist-credits',
                    'artist-rels', 'aliases', 'recording-rels'
                ]
            )['recording']

            # Extract all available data, then filter to lean fields
            all_mb_data = {}

            # Safe extraction with bounds checking
            try:
                if 'artist-credit' in rec_full and rec_full['artist-credit'] and len(rec_full['artist-credit']) > 0:
                    all_mb_data['artist'] = rec_full['artist-credit'][0]['artist']['name']
                    all_mb_data['mb_artist_id'] = rec_full['artist-credit'][0]['artist']['id']
                else:
                    all_mb_data['artist'] = None
                    all_mb_data['mb_artist_id'] = None
            except (KeyError, IndexError) as e:
                logger.debug(f"Error extracting artist info: {e}")
                all_mb_data['artist'] = None
                all_mb_data['mb_artist_id'] = None

            all_mb_data['title'] = rec_full.get('title', '')
            all_mb_data['musicbrainz_id'] = rec_full.get('id', '')

            # Safe release info extraction
            try:
                if 'releases' in rec_full and rec_full['releases'] and len(rec_full['releases']) > 0:
                    release = rec_full['releases'][0]
                    all_mb_data['album'] = release.get('title')
                    all_mb_data['release_date'] = release.get('date')
                    all_mb_data['country'] = release.get('country')
                    all_mb_data['mb_album_id'] = release.get('id')

                    # Safe label info extraction
                    try:
                        if 'label-info-list' in release and release['label-info-list'] and len(release['label-info-list']) > 0:
                            all_mb_data['label'] = release['label-info-list'][0]['label']['name']
                        else:
                            all_mb_data['label'] = None
                    except (KeyError, IndexError) as e:
                        logger.debug(f"Error extracting label info: {e}")
                        all_mb_data['label'] = None
                else:
                    all_mb_data['album'] = None
                    all_mb_data['release_date'] = None
                    all_mb_data['country'] = None
                    all_mb_data['mb_album_id'] = None
                    all_mb_data['label'] = None
            except (KeyError, IndexError) as e:
                logger.debug(f"Error extracting release info: {e}")
                all_mb_data['album'] = None
                all_mb_data['release_date'] = None
                all_mb_data['country'] = None
                all_mb_data['mb_album_id'] = None
                all_mb_data['label'] = None

            # Safe ISRC extraction
            try:
                if 'isrcs' in rec_full and rec_full['isrcs'] and len(rec_full['isrcs']) > 0:
                    all_mb_data['isrc'] = rec_full['isrcs'][0]
                else:
                    all_mb_data['isrc'] = None
            except (KeyError, IndexError) as e:
                logger.debug(f"Error extracting ISRC: {e}")
                all_mb_data['isrc'] = None

            # Safe genre extraction
            try:
                if 'tags' in rec_full and rec_full['tags'] and len(rec_full['tags']) > 0:
                    all_mb_data['genre'] = [tag['name']
                                            for tag in rec_full['tags']]
                else:
                    all_mb_data['genre'] = None
            except (KeyError, IndexError) as e:
                logger.debug(f"Error extracting genres: {e}")
                all_mb_data['genre'] = None

                        # Safe work info extraction
            try:
                if 'work-relation-list' in rec_full and rec_full['work-relation-list'] and len(rec_full['work-relation-list']) > 0:
                    work = rec_full['work-relation-list'][0]['work']
                    all_mb_data['work'] = work.get('title')
                    
                    # Try to get BPM from work attributes if available
                    if 'attributes' in work:
                        logger.debug(f"Work has {len(work['attributes'])} attributes")
                        for attr in work['attributes']:
                            logger.debug(f"Checking attribute: {attr.get('type', 'unknown')} = {attr.get('value', 'unknown')}")
                            if attr.get('type') == 'bpm' and attr.get('value'):
                                try:
                                    bpm = float(attr['value'])
                                    if 60 <= bpm <= 200:  # Valid BPM range
                                        all_mb_data['bpm'] = bpm
                                        logger.debug(f"Found BPM in MusicBrainz work: {bpm}")
                                    else:
                                        logger.debug(f"BPM from work attribute ({bpm}) outside valid range")
                                except (ValueError, TypeError) as e:
                                    logger.debug(f"Could not convert BPM value '{attr.get('value')}': {e}")
                    else:
                        logger.debug("No attributes found in work")
                    
                    # Also check for BPM in recording attributes
                    if 'attributes' in rec_full:
                        logger.debug(f"Recording has {len(rec_full['attributes'])} attributes")
                        for attr in rec_full['attributes']:
                            logger.debug(f"Checking recording attribute: {attr.get('type', 'unknown')} = {attr.get('value', 'unknown')}")
                            if attr.get('type') == 'bpm' and attr.get('value'):
                                try:
                                    bpm = float(attr['value'])
                                    if 60 <= bpm <= 200:  # Valid BPM range
                                        all_mb_data['bpm'] = bpm
                                        logger.debug(f"Found BPM in MusicBrainz recording: {bpm}")
                                    else:
                                        logger.debug(f"BPM from recording attribute ({bpm}) outside valid range")
                                except (ValueError, TypeError) as e:
                                    logger.debug(f"Could not convert BPM value '{attr.get('value')}': {e}")
                    else:
                        logger.debug("No attributes found in recording")
                    
                    all_mb_data['composer'] = work.get('composer')
                else:
                    all_mb_data['work'] = None
                    all_mb_data['composer'] = None
            except (KeyError, IndexError) as e:
                logger.debug(f"Error extracting work info: {e}")
                all_mb_data['work'] = None
                all_mb_data['composer'] = None

            # Try to extract BPM from tags if not found in attributes
            if 'bpm' not in all_mb_data and 'tags' in all_mb_data and all_mb_data['tags']:
                logger.debug("Checking tags for BPM information")
                for tag in all_mb_data['tags']:
                    if isinstance(tag, str) and 'bpm' in tag.lower():
                        try:
                            # Extract BPM from tag like "120 bpm" or "bpm 120"
                            import re
                            bpm_match = re.search(r'(\d{2,3})\s*bpm', tag.lower())
                            if bpm_match:
                                bpm = float(bpm_match.group(1))
                                if 60 <= bpm <= 200:
                                    all_mb_data['bpm'] = bpm
                                    logger.debug(f"Found BPM in tag: {bpm}")
                                    break
                        except (ValueError, TypeError):
                            pass
            
            # Filter to lean fields only
            filtered_data = filter_metadata(all_mb_data)
            logger.debug(
                f"MusicBrainz lookup completed: {len(filtered_data)} fields")
            return filtered_data

        except Exception as e:
            logger.warning(f"MusicBrainz lookup failed: {str(e)}")
            logger.debug(
                f"MusicBrainz lookup error details: {type(e).__name__}")
            return {}

    def _lastfm_lookup(self, artist, title):
        api_key = os.getenv('LASTFM_API_KEY')
        if not api_key:
            logger.debug(
                "LASTFM_API_KEY not set; skipping Last.fm enrichment.")
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
            logger.debug(f"Making Last.fm API request for: {artist} - {title}")
            resp = requests.get(url, params=params, timeout=10)
            logger.debug(
                f"Last.fm API returned status {resp.status_code} for {artist} - {title}")
            resp.raise_for_status()
            data = resp.json()
            track = data.get('track', {})

            # Extract all available Last.fm data with safe indexing
            all_lastfm_data = {}

            # Safe genre extraction
            try:
                toptags = track.get('toptags', {})
                tag_list = toptags.get('tag', [])
                if tag_list and isinstance(tag_list, list):
                    all_lastfm_data['genre_lastfm'] = [
                        t['name'] for t in tag_list if isinstance(t, dict) and 'name' in t]
                else:
                    all_lastfm_data['genre_lastfm'] = None
            except (KeyError, IndexError, TypeError) as e:
                logger.debug(f"Error extracting Last.fm genres: {e}")
                all_lastfm_data['genre_lastfm'] = None

            # Safe album extraction
            try:
                album_info = track.get('album', {})
                if album_info and isinstance(album_info, dict):
                    all_lastfm_data['album_lastfm'] = album_info.get('title')
                    all_lastfm_data['album_mbid'] = album_info.get('mbid')
                    all_lastfm_data['album_artist'] = album_info.get('artist')
                    all_lastfm_data['album_url'] = album_info.get('url')
                    all_lastfm_data['album_image'] = album_info.get('image')
                else:
                    all_lastfm_data['album_lastfm'] = None
                    all_lastfm_data['album_mbid'] = None
                    all_lastfm_data['album_artist'] = None
                    all_lastfm_data['album_url'] = None
                    all_lastfm_data['album_image'] = None
            except (KeyError, IndexError, TypeError) as e:
                logger.debug(f"Error extracting Last.fm album info: {e}")
                all_lastfm_data['album_lastfm'] = None
                all_lastfm_data['album_mbid'] = None
                all_lastfm_data['album_artist'] = None
                all_lastfm_data['album_url'] = None
                all_lastfm_data['album_image'] = None

            # Safe artist extraction
            try:
                artist_info = track.get('artist', {})
                if artist_info and isinstance(artist_info, dict):
                    all_lastfm_data['artist_mbid'] = artist_info.get('mbid')
                else:
                    all_lastfm_data['artist_mbid'] = None
            except (KeyError, IndexError, TypeError) as e:
                logger.debug(f"Error extracting Last.fm artist info: {e}")
                all_lastfm_data['artist_mbid'] = None

            # Safe wiki extraction
            try:
                wiki_info = track.get('wiki', {})
                if wiki_info and isinstance(wiki_info, dict):
                    all_lastfm_data['wiki'] = wiki_info.get('summary')
                else:
                    all_lastfm_data['wiki'] = None
            except (KeyError, IndexError, TypeError) as e:
                logger.debug(f"Error extracting Last.fm wiki info: {e}")
                all_lastfm_data['wiki'] = None

            # Safe @attr extraction
            try:
                attr_info = track.get('@attr', {})
                if attr_info and isinstance(attr_info, dict):
                    all_lastfm_data['rank'] = attr_info.get('rank')
                else:
                    all_lastfm_data['rank'] = None
            except (KeyError, IndexError, TypeError) as e:
                logger.debug(f"Error extracting Last.fm @attr info: {e}")
                all_lastfm_data['rank'] = None

            # Simple field extractions
            all_lastfm_data['listeners'] = track.get('listeners')
            all_lastfm_data['playcount'] = track.get('playcount')
            all_lastfm_data['duration'] = track.get('duration')
            all_lastfm_data['url'] = track.get('url')
            
            # Note: LastFM API doesn't provide BPM data directly
            # We could potentially use tags to infer BPM ranges, but it's not reliable
            # For now, we'll skip BPM from LastFM and rely on MusicBrainz
            all_lastfm_data['mbid'] = track.get('mbid')
            all_lastfm_data['streamable'] = track.get('streamable')
            all_lastfm_data['userplaycount'] = track.get('userplaycount')
            all_lastfm_data['userloved'] = track.get('userloved')

            logger.debug(
                f"Extracted {len(all_lastfm_data)} Last.fm fields for {artist} - {title}")

            # Filter to only lean fields for database
            return {k: v for k, v in all_lastfm_data.items() if k in LEAN_FIELDS and v is not None and v != ''}
        except requests.exceptions.RequestException as e:
            logger.warning(
                f"Last.fm API request failed for {artist} - {title}: {e}")
            return {}
        except (KeyError, IndexError, TypeError) as e:
            logger.warning(
                f"Last.fm data parsing failed for {artist} - {title}: {e}")
            return {}
        except Exception as e:
            logger.warning(
                f"Last.fm lookup failed for {artist} - {title}: {e}")
            import traceback
            logger.debug(
                f"Last.fm lookup error traceback: {traceback.format_exc()}")
            return {}

    def extract_features(self, audio_path: str, force_reextract: bool = False) -> Optional[tuple]:
        """Extract features from an audio file.

        Args:
            audio_path (str): Path to the audio file.
            force_reextract (bool): If True, bypass the cache and re-extract features.

        Returns:
            Optional[tuple]: (features dict, db_write_success bool, file_hash str) or None on failure.
        """
        # Use file discovery to check if file should be excluded
        from .file_discovery import FileDiscovery
        file_discovery = FileDiscovery()
        if file_discovery._is_in_excluded_directory(audio_path):
            logger.warning(
                f"Skipping file in excluded directory: {audio_path}")
            return None, False, None

        try:
            file_info = self._get_file_info(audio_path)

            # Dynamic timeout based on file size
            try:
                file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
                if file_size_mb > 100:  # Very large files (>100MB)
                    self.timeout_seconds = 600  # 10 minutes
                elif file_size_mb > 50:  # Large files (>50MB)
                    self.timeout_seconds = 300  # 5 minutes
                else:
                    self.timeout_seconds = 180  # 3 minutes for normal files
                logger.debug(
                    f"Set timeout to {self.timeout_seconds}s for {file_size_mb:.1f}MB file")
            except:
                self.timeout_seconds = 180  # Fallback timeout
            if not force_reextract:
                cached_features = self._get_cached_features(file_info)
                if cached_features:
                    logger.info(
                        f"Using cached features for {file_info['file_path']}")
                    return cached_features, True, file_info['file_hash']
            audio = self._safe_audio_load(audio_path)
            if audio is None:
                logger.warning(f"Audio loading failed for {audio_path}")
                self._mark_failed(file_info)
                return None, False, None
            features = self._extract_all_features(audio_path, audio)
            # Only mark as failed if features is None (complete failure)
            # If features is a dict, even with default values, it's a partial success
            if features is None:
                logger.error(f"Feature extraction failed for {audio_path}")
                self._mark_failed(file_info)
                return None, False, None
            # Validate that we have at least some basic features
            if not isinstance(features, dict) or len(features) == 0:
                logger.error(
                    f"Feature extraction returned invalid result for {audio_path}")
                self._mark_failed(file_info)
                return None, False, None
            db_write_success = self._save_features_to_db(
                file_info, features, failed=0)
            logger.debug(
                f"Database save result for {file_info['file_path']}: {db_write_success}")
            if db_write_success:
                logger.info(f"DB WRITE: {file_info['file_path']}")
            else:
                logger.error(f"DB WRITE FAILED: {file_info['file_path']}")
                self._mark_failed(file_info)
                return None, False, None
            logger.debug(
                f"Returning successful result for {file_info['file_path']}")
            return features, db_write_success, file_info['file_hash']
        except TimeoutException as te:
            logger.error(f"Timeout processing {audio_path}: {str(te)}")
            self._mark_failed(self._get_file_info(audio_path))
            return None, False, None
        except Exception as e:
            logger.error(f"Error processing {audio_path}: {str(e)}")
            import traceback
            logger.warning(traceback.format_exc())
            self._mark_failed(self._get_file_info(audio_path))
            return None, False, None

    def _get_cached_features(self, file_info):
        """Get cached features using connection pool."""
        with self.db_pool.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            SELECT duration, bpm, beat_confidence, centroid,
                   loudness, danceability, key, scale, onset_rate, zcr,
                   mfcc, chroma, spectral_contrast, spectral_flatness, spectral_rolloff, musicnn_embedding, musicnn_tags, musicnn_skipped, metadata
            FROM audio_features
            WHERE file_hash = ? AND last_modified >= ?
            """, (file_info['file_hash'], file_info['last_modified']))

            row = cursor.fetchone()
            if row:
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
                    'musicnn_tags': json.loads(row[16]) if row[16] else {},
                    'musicnn_skipped': row[17],
                    'metadata': json.loads(row[18]) if row[18] else {},
                    'filepath': file_info['file_path'],
                    'filename': os.path.basename(file_info['file_path'])
                }
        return None

    def _extract_all_features(self, audio_path, audio):
        # Initialize with default values
        features = {}

        logger.info(
            f"Starting feature extraction for {os.path.basename(audio_path)}")

        # Log memory usage at start of feature extraction
        try:
            import psutil
            memory_info = psutil.virtual_memory()
            memory_percent = memory_info.percent
            used_gb = memory_info.used / (1024**3)
            available_gb = memory_info.available / (1024**3)
            logger.debug(f"Memory usage at start of feature extraction: {memory_percent:.1f}% ({used_gb:.1f}GB used, {available_gb:.1f}GB available)")
            
            # Check if memory is critical (only if memory-aware processing is enabled)
            memory_aware = os.getenv('MEMORY_AWARE', 'false').lower() == 'true'
            is_memory_critical = memory_aware and memory_percent > 85
            if is_memory_critical:
                logger.warning(f"Memory usage critical ({memory_percent:.1f}%), will skip memory-intensive features")
        except Exception as e:
            logger.debug(f"Could not get memory info: {e}")
            is_memory_critical = False

        # Input validation
        if audio is None:
            logger.error("Audio is None, cannot extract features")
            return None
        if not hasattr(audio, '__len__'):
            logger.error(
                "Audio does not have length attribute, cannot extract features")
            return None

        logger.info(f"Audio loaded: {len(audio)} samples")

        # Check if this is an extremely large file (>200M samples ~4.5 hours at 44kHz)
        is_extremely_large = len(audio) > 200000000
        if is_extremely_large:
            logger.warning(
                f"Extremely large file detected ({len(audio)} samples), skipping some features")
        
        # Check if file is too large for MFCC (>100M samples ~2.3 hours at 44kHz)
        is_too_large_for_mfcc = len(audio) > 100000000
        if is_too_large_for_mfcc:
            logger.warning(
                f"File too large for MFCC extraction ({len(audio)} samples), skipping MFCC")
        
        # Check if file is extremely large for any processing (>500M samples ~11.3 hours at 44kHz)
        is_extremely_large_for_processing = len(audio) > 500000000
        if is_extremely_large_for_processing:
            logger.warning(
                f"File extremely large for processing ({len(audio)} samples), using minimal features only")

        # Duration calculation
        try:
            sample_rate = 44100  # Default sample rate
            audio_length = len(audio)
            duration = audio_length / sample_rate
            features['duration'] = float(duration)
            logger.info(f"Duration: {features['duration']:.2f}s")
        except Exception as e:
            logger.warning(f"Duration calculation failed: {str(e)}")
            features['duration'] = 0.0

        # Extract rhythm features (skip for extremely large files)
        if is_extremely_large_for_processing:
            logger.warning("Skipping rhythm extraction for extremely large file")
            features['bpm'] = -1.0  # Special marker for failed BPM extraction
        else:
            try:
                # Get metadata for external BPM lookup if local extraction fails
                metadata = {}
                try:
                    audio_file = MutagenFile(audio_path, easy=True)
                    if audio_file:
                        logger.debug(f"Audio file tags available: {list(audio_file.keys())}")
                        for tag in ['title', 'artist', 'album', 'date', 'genre']:
                            if tag in audio_file:
                                metadata[tag] = str(
                                    audio_file[tag][0]) if audio_file[tag] else None
                                logger.debug(f"Extracted {tag}: '{metadata[tag]}'")
                            else:
                                logger.debug(f"Tag '{tag}' not found in audio file")
                    else:
                        logger.debug("No audio file tags found")
                except Exception as e:
                    logger.debug(f"File tag extraction failed: {str(e)}")
                
                logger.debug(f"Final metadata for external BPM lookup: {metadata}")
                
                rhythm_result = self._extract_rhythm_features(audio, audio_path, metadata)
                features['bpm'] = rhythm_result['bpm']
                logger.info(f"Rhythm: BPM = {features['bpm']:.1f}")
            except TimeoutException as te:
                logger.error(f"Rhythm extraction timed out, skipping file: {str(te)}")
                return None  # Skip the entire file
            except Exception as e:
                logger.warning(f"Rhythm extraction failed: {str(e)}")
                features['bpm'] = -1.0  # Special marker for failed BPM extraction

        # Extract spectral features (skip for extremely large files)
        if is_extremely_large_for_processing:
            logger.warning("Skipping spectral extraction for extremely large file")
            features['centroid'] = 0.0
        else:
            try:
                spectral_features = self._extract_spectral_features(audio)
                features['centroid'] = spectral_features['spectral_centroid']
                logger.info(f"Spectral: centroid = {features['centroid']:.1f}Hz")
            except TimeoutException as te:
                logger.error(f"Spectral extraction timed out, skipping file: {str(te)}")
                return None  # Skip the entire file
            except Exception as e:
                logger.warning(f"Spectral extraction failed: {str(e)}")
                features['centroid'] = 0.0

        # Extract loudness (skip for extremely large files)
        if is_extremely_large_for_processing:
            logger.warning("Skipping loudness extraction for extremely large file")
            features['loudness'] = 0.0
        else:
            try:
                loudness_result = self._extract_loudness(audio)
                features['loudness'] = loudness_result['rms']
                logger.info(f"Loudness: RMS = {features['loudness']:.3f}")
            except TimeoutException as te:
                logger.error(f"Loudness extraction timed out, skipping file: {str(te)}")
                return None  # Skip the entire file
            except Exception as e:
                logger.warning(f"Loudness extraction failed: {str(e)}")
                features['loudness'] = 0.0

        # Extract danceability
        try:
            dance_result = self._extract_danceability(audio)
            features['danceability'] = dance_result['danceability']
            logger.info(f"Danceability: {features['danceability']:.3f}")
        except Exception as e:
            logger.warning(f"Danceability extraction failed: {str(e)}")
            features['danceability'] = 0.0

        # Extract key
        try:
            key_result = self._extract_key(audio)
            features['key'] = key_result['key']
            features['scale'] = key_result['scale']
            features['key_strength'] = key_result['key_strength']
            logger.info(
                f"Key: {features['key']} {features['scale']} (strength: {features['key_strength']:.3f})")
        except Exception as e:
            logger.warning(f"Key extraction failed: {str(e)}")
            features['key'] = 'C'
            features['scale'] = 'major'
            features['key_strength'] = 0.0

        # Extract onset rate
        try:
            onset_result = self._extract_onset_rate(audio)
            features['onset_rate'] = onset_result['onset_rate']
            logger.info(f"Onset rate: {features['onset_rate']:.2f} onsets/sec")
        except Exception as e:
            logger.warning(f"Onset rate extraction failed: {str(e)}")
            features['onset_rate'] = 0.0

        # Extract zero crossing rate
        try:
            zcr_result = self._extract_zcr(audio)
            features['zcr'] = zcr_result['zcr']
            logger.info(f"Zero crossing rate: {features['zcr']:.3f}")
        except Exception as e:
            logger.warning(f"Zero crossing rate extraction failed: {str(e)}")
            features['zcr'] = 0.0

        # Extract MFCC (skip for extremely large files or when memory is critical)
        if is_extremely_large or is_too_large_for_mfcc or is_memory_critical:
            if is_memory_critical:
                logger.warning("Skipping MFCC extraction due to critical memory usage")
            else:
                logger.warning("Skipping MFCC extraction for extremely large file to avoid memory issues")
            features['mfcc'] = [0.0] * 13
        else:
            try:
                mfcc_result = self._extract_mfcc(audio)
                features['mfcc'] = mfcc_result['mfcc']
                logger.info(f"MFCC: {len(features['mfcc'])} coefficients")
            except TimeoutException as te:
                logger.error(f"MFCC extraction timed out, skipping file: {str(te)}")
                return None  # Skip the entire file
            except Exception as e:
                logger.warning(f"MFCC extraction failed: {str(e)}")
                features['mfcc'] = [0.0] * 13

        # For extremely large files or when memory is critical, skip some features to avoid timeouts/memory issues
        if not is_extremely_large and not is_memory_critical:
            # Extract chroma
            try:
                chroma_result = self._extract_chroma(audio)
                features['chroma'] = chroma_result['chroma']
                logger.info(f"Chroma: {len(features['chroma'])} features")
            except Exception as e:
                logger.warning(f"Chroma extraction failed: {str(e)}")
                features['chroma'] = [0.0] * 12

            # Extract spectral contrast
            try:
                contrast_result = self._extract_spectral_contrast(audio)
                features['spectral_contrast'] = contrast_result['spectral_contrast']
                logger.info(
                    f"Spectral contrast: {features['spectral_contrast']:.3f}")
            except Exception as e:
                logger.warning(
                    f"Spectral contrast extraction failed: {str(e)}")
                features['spectral_contrast'] = 0.0

            # Extract spectral flatness
            try:
                flatness_result = self._extract_spectral_flatness(audio)
                features['spectral_flatness'] = flatness_result['spectral_flatness']
                logger.info(
                    f"Spectral flatness: {features['spectral_flatness']:.3f}")
            except Exception as e:
                logger.warning(
                    f"Spectral flatness extraction failed: {str(e)}")
                features['spectral_flatness'] = 0.0

            # Extract spectral rolloff
            try:
                rolloff_result = self._extract_spectral_rolloff(audio)
                features['spectral_rolloff'] = rolloff_result['spectral_rolloff']
                logger.info(
                    f"Spectral rolloff: {features['spectral_rolloff']:.1f}Hz")
            except Exception as e:
                logger.warning(f"Spectral rolloff extraction failed: {str(e)}")
                features['spectral_rolloff'] = 0.0
        else:
            # For extremely large files or when memory is critical, use default values for skipped features
            if is_memory_critical:
                logger.warning("Skipping chroma, spectral contrast, flatness, and rolloff due to critical memory usage")
            else:
                logger.warning("Skipping chroma, spectral contrast, flatness, and rolloff for extremely large file")
            features['chroma'] = [0.0] * 12
            features['spectral_contrast'] = 0.0
            features['spectral_flatness'] = 0.0
            features['spectral_rolloff'] = 0.0

        # Extract MusiCNN embedding (if available)
        try:
            musicnn_result = self._extract_musicnn_embedding(audio_path)
            if musicnn_result and not musicnn_result.get('skipped'):
                features['musicnn_embedding'] = musicnn_result['embedding']
                features['musicnn_tags'] = musicnn_result['tags']
                features['musicnn_skipped'] = 0
                logger.info(
                    f"MusiCNN: {len(features['musicnn_embedding'])} dimensions, {len(features['musicnn_tags'])} tags")
            else:
                if musicnn_result and musicnn_result.get('skipped'):
                    logger.info(f"MusiCNN: skipped ({musicnn_result.get('reason', 'unknown')})")
                    features['musicnn_skipped'] = 1
                else:
                    logger.info("MusiCNN: not available (models missing)")
                    features['musicnn_skipped'] = 1
                features['musicnn_embedding'] = []
                features['musicnn_tags'] = {}
        except Exception as e:
            logger.warning(f"MusiCNN extraction failed: {str(e)}")
            features['musicnn_embedding'] = []
            features['musicnn_tags'] = {}
            features['musicnn_skipped'] = 1

        # Metadata enrichment
        logger.info("Enriching metadata...")
        try:
            # Extract basic metadata from file
            metadata = {}

            # Try to get metadata from file tags
            try:
                audio_file = MutagenFile(audio_path, easy=True)
                if audio_file:
                    for tag in ['title', 'artist', 'album', 'date', 'genre']:
                        if tag in audio_file:
                            metadata[tag] = str(
                                audio_file[tag][0]) if audio_file[tag] else None
                    logger.info(f"Extracted metadata from file: {metadata}")
                else:
                    logger.info("No audio file metadata found")
            except Exception as e:
                logger.debug(f"File tag extraction failed: {str(e)}")
                logger.info(f"Could not extract metadata from file: {str(e)}")

            # Enrich with MusicBrainz data
            if metadata.get('artist') and metadata.get('title'):
                logger.info(
                    f"Looking up MusicBrainz data for {metadata['artist']} - {metadata['title']}")
                mb_data = self._musicbrainz_lookup(
                    metadata['artist'], metadata['title'])
                if mb_data:
                    metadata.update(mb_data)
                    logger.info(
                        f"MusicBrainz: found {len(mb_data)} additional fields")
                else:
                    logger.info("MusicBrainz: no additional data found")
            else:
                logger.info(f"No artist/title found for enrichment. Metadata: {metadata}")

            # Enrich with Last.fm data
            if metadata.get('artist') and metadata.get('title'):
                logger.info(
                    f"Looking up Last.fm data for {metadata['artist']} - {metadata['title']}")
                lfm_data = self._lastfm_lookup(
                    metadata['artist'], metadata['title'])
                if lfm_data:
                    metadata.update(lfm_data)
                    logger.info(
                        f"Last.fm: found {len(lfm_data)} additional fields")
                else:
                    logger.info("Last.fm: no additional data found")
            else:
                logger.warning(f"Skipping Last.fm enrichment - no artist/title available")

            features['metadata'] = metadata
            logger.info(
                f"Metadata enrichment completed: {len(metadata)} fields")

        except Exception as e:
            logger.warning(f"Metadata enrichment failed: {str(e)}")
            features['metadata'] = {}

        logger.info(
            f"Feature extraction completed: {len(features)} features extracted")
        return features

    def _mark_failed(self, file_info):
        """Mark file as failed using connection pool with robust error handling."""
        # Ensure file_path is container path
        file_info = dict(file_info)
        file_info['file_path'] = self._normalize_to_library_path(
            file_info['file_path'])
        logger.warning(
            f"Marking file as failed: {file_info['file_path']} (hash: {file_info['file_hash']})")
        
        def _mark_failed_operation():
            """Inner function for the mark failed operation that can be retried."""
            with self.db_pool.get_connection() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO audio_features (file_hash, file_path, last_modified, metadata, failed) VALUES (?, ?, ?, ?, 1)",
                    (file_info['file_hash'], file_info['file_path'],
                     file_info['last_modified'], '{}')
                )
            return True
        
        # Use robust error handling with retry logic
        if ROBUSTNESS_UTILITIES_AVAILABLE:
            try:
                # Use circuit breaker for database operations
                result = self.circuit_breakers['database_operations'].call(_mark_failed_operation)
                return result
            except CircuitBreakerOpenError:
                logger.error(f"Database circuit breaker is OPEN for marking failed: {file_info['file_path']}")
                return False
            except Exception as e:
                # Classify the error
                error_info = self.error_classifier.classify_error(e, {
                    'file_path': file_info['file_path'],
                    'operation': 'mark_failed',
                    'file_size_mb': os.path.getsize(file_info['file_path']) / (1024 * 1024) if os.path.exists(file_info['file_path']) else 0
                })
                
                logger.warning(f"Mark failed error classified as {error_info.error_type.value} "
                             f"(severity: {error_info.severity.value})")
                
                # Apply retry strategy if retryable
                if error_info.retryable and self.error_classifier.should_retry(error_info, 0):
                    logger.info(f"Retrying mark failed for {file_info['file_path']}")
                    retry_result = self.retry_manager.exponential_backoff(
                        _mark_failed_operation, max_attempts=3, timeout=15
                    )
                    return retry_result.success
                else:
                    logger.error(f"Mark failed operation failed and is not retryable: {str(e)}")
                    return False
        else:
            # Fallback to original implementation without robustness features
            try:
                return _mark_failed_operation()
            except Exception as e:
                logger.error(f"Error marking file as failed: {str(e)}")
                return False

    def _save_features_to_db(self, file_info, features, failed=0):
        """Save features to database using connection pool with robust error handling."""
        # Ensure file_path is container path
        file_info = dict(file_info)
        file_info['file_path'] = self._normalize_to_library_path(
            file_info['file_path'])

        def _save_operation():
            """Inner function for the save operation that can be retried."""
            # Validate and convert all features to proper Python types
            features = validate_and_convert_features(features)
            logger.debug(
                f"Validated features for {file_info['file_path']}: {len(features)} features")

            # Prepare values with proper type conversion
            values_tuple = (
                file_info['file_hash'],
                file_info['file_path'],
                features.get('duration', 0.0),
                features.get('bpm', 0.0),
                features.get('beat_confidence', 0.0),
                features.get('centroid', 0.0),
                features.get('loudness', 0.0),
                features.get('danceability', 0.0),
                features.get('key', ''),  # Changed from 0 to '' for string
                features.get('scale', ''),  # Changed from 0 to '' for string
                features.get('key_strength', 0.0),  # Added key_strength
                features.get('onset_rate', 0.0),
                features.get('zcr', 0.0),
                safe_json_dumps(features.get('mfcc', [])),
                safe_json_dumps(features.get('chroma', [])),
                features.get('spectral_contrast', 0.0),
                features.get('spectral_flatness', 0.0),
                features.get('spectral_rolloff', 0.0),
                safe_json_dumps(features.get('musicnn_embedding')),
                # Added musicnn_tags
                safe_json_dumps(features.get('musicnn_tags', {})),
                features.get('musicnn_skipped', 0),
                file_info['last_modified'],
                safe_json_dumps(features.get('metadata', {})),
                failed
            )

            # Debug: log the values for troubleshooting
            logger.debug(f"Values count: {len(values_tuple)}")
            for i, val in enumerate(values_tuple):
                logger.debug(f"Value {i}: {type(val)} = {val}")

            with self.db_pool.get_connection() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO audio_features (
                        file_hash, file_path, duration, bpm, beat_confidence, centroid, loudness, danceability, key, scale, key_strength, onset_rate, zcr,
                        mfcc, chroma, spectral_contrast, spectral_flatness, spectral_rolloff, musicnn_embedding, musicnn_tags, musicnn_skipped,
                        last_modified, metadata, failed
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    values_tuple
                )
            logger.debug(
                f"Successfully saved features to database for {file_info['file_path']}")
            logger.info(
                f"Database save completed successfully for {file_info['file_path']}")
            return True

        # Use robust error handling with retry logic
        if ROBUSTNESS_UTILITIES_AVAILABLE:
            try:
                # Use circuit breaker for database operations
                result = self.circuit_breakers['database_operations'].call(_save_operation)
                return result
            except CircuitBreakerOpenError:
                logger.error(f"Database circuit breaker is OPEN for {file_info['file_path']}")
                return False
            except Exception as e:
                # Classify the error
                error_info = self.error_classifier.classify_error(e, {
                    'file_path': file_info['file_path'],
                    'operation': 'database_save',
                    'file_size_mb': os.path.getsize(file_info['file_path']) / (1024 * 1024) if os.path.exists(file_info['file_path']) else 0
                })
                
                logger.warning(f"Database save error classified as {error_info.error_type.value} "
                             f"(severity: {error_info.severity.value})")
                
                # Apply retry strategy if retryable
                if error_info.retryable and self.error_classifier.should_retry(error_info, 0):
                    logger.info(f"Retrying database save for {file_info['file_path']}")
                    retry_result = self.retry_manager.exponential_backoff(
                        _save_operation, max_attempts=3, timeout=30
                    )
                    return retry_result.success
                else:
                    logger.error(f"Database save failed and is not retryable: {str(e)}")
                    return False
        else:
            # Fallback to original implementation without robustness features
            try:
                return _save_operation()
            except Exception as e:
                logger.error(f"Error saving features to DB: {str(e)}")
                import traceback
                logger.error(f"Database save error traceback: {traceback.format_exc()}")
                return False

    def get_all_features(self, include_failed=False):
        """Get all features using connection pool."""
        try:
            with self.db_pool.get_connection() as conn:
                cursor = conn.cursor()
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

    def get_all_audio_files(self, music_dir='/music'):
        """Get all audio files from the filesystem using FileDiscovery."""
        from .file_discovery import FileDiscovery

        file_discovery = FileDiscovery(music_dir=music_dir, audio_db=self)
        return file_discovery.discover_files()

    def get_all_tracks(self):
        """Get all tracks from the database (non-failed)."""
        try:
            with self.db_pool.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT COUNT(*) FROM audio_features WHERE failed=0")
                return cursor.fetchone()[0]
        except Exception as e:
            logger.error(f"Error getting track count: {str(e)}")
            return 0

    def cleanup_database(self) -> List[str]:
        """Remove entries for files that no longer exist.

        Returns:
            List[str]: List of file paths that were removed from the database.
        """
        try:
            with self.db_pool.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT file_path FROM audio_features")
                db_files = [row[0] for row in cursor.fetchall()]

                missing_files = [f for f in db_files if not os.path.exists(f)]

                if missing_files:
                    logger.info(
                        f"Cleaning up {len(missing_files)} missing files from database")
                    placeholders = ','.join(['?'] * len(missing_files))
                    cursor.execute(
                        f"DELETE FROM audio_features WHERE file_path IN ({placeholders})",
                        missing_files
                    )
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
                meta = {k: (v[0] if isinstance(v, list) and v else v)
                        for k, v in audiofile.items()}
        except Exception as e:
            logger.warning(
                f"Mutagen tag extraction failed for enrichment: {e}")
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
            missing_fields = [field for field in [
                'genre', 'year', 'album'] if not meta.get(field)]
            if missing_fields:
                lastfm_tags = self._lastfm_lookup(artist, title)
                for field in missing_fields:
                    v = lastfm_tags.get(field)
                    if v and not meta.get(field):
                        meta[field] = v
                        updated_fields.append(f"lastfm:{field}")
        # Update only the metadata column, keep failed=1
        try:
            with self.db_pool.get_connection() as conn:
                conn.execute(
                    "UPDATE audio_features SET metadata = ?, failed = 1 WHERE file_hash = ?",
                    (json.dumps(meta), file_info['file_hash'])
                )
            logger.info(
                f"Enriched metadata for failed file {file_info['file_path']} (fields updated: {updated_fields})")
        except Exception as e:
            logger.error(
                f"Error updating metadata for failed file {file_info['file_path']}: {e}")

    def validate_essential_fields(self, features):
        """Check if essential features are present and valid"""
        if not isinstance(features, dict):
            return False, "Features must be a dictionary"

        essential_fields = {
            'duration': (float, lambda x: x > 0, "Duration must be positive"),
            'bpm': (float, lambda x: 0 <= x <= 300, "BPM must be between 0-300"),
            'centroid': (float, lambda x: x >= 0, "Centroid must be non-negative"),
            'loudness': (float, lambda x: x >= 0, "Loudness must be non-negative"),
            'danceability': (float, lambda x: 0 <= x <= 1, "Danceability must be between 0-1"),
            'key': (str, lambda x: x in ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B', ''], "Invalid key"),
            'scale': (str, lambda x: x in ['major', 'minor', ''], "Invalid scale"),
            'onset_rate': (float, lambda x: x >= 0, "Onset rate must be non-negative"),
            'zcr': (float, lambda x: x >= 0, "ZCR must be non-negative"),
            'mfcc': (list, lambda x: len(x) == 13, "MFCC must have 13 coefficients"),
            'chroma': (list, lambda x: len(x) == 12, "Chroma must have 12 coefficients"),
            'spectral_flatness': (float, lambda x: 0 <= x <= 1, "Spectral flatness must be between 0-1"),
            'spectral_rolloff': (float, lambda x: x >= 0, "Spectral rolloff must be non-negative"),
            'spectral_contrast': (float, lambda x: x >= 0, "Spectral contrast must be non-negative")
        }

        for field, (expected_type, validator, error_msg) in essential_fields.items():
            if field not in features:
                return False, f"Missing essential field: {field}"
            if not isinstance(features[field], expected_type):
                return False, f"Invalid type for {field}: expected {expected_type}, got {type(features[field])}"
            if not validator(features[field]):
                return False, f"{error_msg}: {features[field]}"

        return True, "All essential fields valid"

    def validate_optional_fields(self, features):
        """Check if optional features are present and valid"""
        optional_fields = {
            'musicnn_embedding': (list, lambda x: len(x) > 0 if x else True, "MusiCNN embedding must be non-empty list"),
            'musicnn_tags': (dict, lambda x: isinstance(x, dict), "MusiCNN tags must be dictionary"),
            'metadata': (dict, lambda x: isinstance(x, dict), "Metadata must be dictionary"),
            'beat_confidence': (float, lambda x: 0 <= x <= 1, "Beat confidence must be between 0-1"),
            'key_strength': (float, lambda x: 0 <= x <= 1, "Key strength must be between 0-1")
        }

        warnings = []
        for field, (expected_type, validator, error_msg) in optional_fields.items():
            if field in features:
                if not isinstance(features[field], expected_type):
                    warnings.append(f"Invalid type for {field}: {error_msg}")
                elif not validator(features[field]):
                    warnings.append(f"Invalid value for {field}: {error_msg}")

        return warnings

    def validate_feature_quality(self, features):
        """Progressive validation with quality scoring"""
        if not features:
            return 0, "No features provided"

        score = 0
        max_score = 100

        # Essential features (60 points)
        essential_valid, essential_msg = self.validate_essential_fields(
            features)
        if essential_valid:
            score += 60
            logger.debug("Essential features validation passed")
        else:
            logger.warning(
                f"Essential features validation failed: {essential_msg}")
            return score, essential_msg

        # Optional features (40 points)
        optional_warnings = self.validate_optional_fields(features)
        optional_score = max(0, 40 - len(optional_warnings)
                             * 5)  # -5 points per warning
        score += optional_score

        if optional_warnings:
            logger.debug(f"Optional field warnings: {optional_warnings}")

        # Quality assessment
        if score >= 90:
            quality = 'excellent'
        elif score >= 70:
            quality = 'good'
        elif score >= 50:
            quality = 'acceptable'
        else:
            quality = 'poor'

        logger.info(f"Feature quality score: {score}/{max_score} ({quality})")
        return score, quality

    def retry_analysis_with_backoff(self, file_path, max_attempts=3):
        """Retry analysis with exponential backoff"""
        logger.info(
            f"Starting retry analysis for {file_path} (max {max_attempts} attempts)")

        for attempt in range(max_attempts):
            try:
                logger.info(
                    f"Attempt {attempt + 1}/{max_attempts} for {file_path}")
                result = self.extract_features(file_path, force_reextract=True)

                if result and result[0]:  # features exist
                    features, db_success, file_hash = result
                    quality_score, quality = self.validate_feature_quality(
                        features)

                    if quality in ['excellent', 'good', 'acceptable'] and db_success:
                        logger.info(
                            f"Retry successful for {file_path} (quality: {quality})")
                        return True, features
                    else:
                        logger.warning(
                            f"Retry attempt {attempt + 1} failed quality check for {file_path} (quality: {quality})")
                else:
                    logger.warning(
                        f"Retry attempt {attempt + 1} failed for {file_path}")

            except Exception as e:
                logger.warning(
                    f"Retry attempt {attempt + 1} failed for {file_path}: {e}")

            if attempt < max_attempts - 1:
                wait_time = 2 ** attempt  # Exponential backoff: 1, 2, 4 seconds
                logger.info(
                    f"Waiting {wait_time}s before next attempt for {file_path}")
                time.sleep(wait_time)

        logger.error(
            f"All {max_attempts} retry attempts failed for {file_path}")
        return False, None

    def get_files_needing_analysis(self, music_dir='/music'):
        """Phase 1: File Discovery - Update database with current file state"""
        logger.info(f"DISCOVERY: Phase 1 - File Discovery in {music_dir}")

        # Step 1: Scan filesystem for current audio files
        current_files = set()
        audio_extensions = {'.mp3', '.flac',
                            '.wav', '.ogg', '.m4a', '.aac', '.wma'}

        for root, dirs, files in os.walk(music_dir):
            # Skip failed_files directory
            if 'failed_files' in root:
                continue

            for file in files:
                if any(file.lower().endswith(ext) for ext in audio_extensions):
                    file_path = os.path.join(root, file)
                    current_files.add(file_path)

        logger.info(
            f"DISCOVERY: Found {len(current_files)} audio files in filesystem")

        # Step 2: Update file discovery state (adds/removes/updates accordingly)
        self.update_file_discovery_state(list(current_files))

        # Step 3: Get files that need analysis based on discovery state
        files_needing_analysis = self._get_files_for_analysis_from_discovery()

        logger.info(
            f"DISCOVERY: Phase 1 complete - {len(files_needing_analysis)} files need analysis")
        return files_needing_analysis

    def _get_files_for_analysis_from_discovery(self):
        """Phase 2: Analysis Selection - Check discovery state for files to process"""
        logger.info(f"DISCOVERY: Phase 2 - Analysis Selection")

        with self.db_pool.get_connection() as conn:
            # Get files that were added or changed in discovery phase
            cursor = conn.execute("""
                SELECT file_path, file_size, last_modified 
                FROM file_discovery_state 
                WHERE status = 'active'
            """)

            discovery_files = {row[0]: (row[1], row[2])
                               for row in cursor.fetchall()}

            # Get files already analyzed in audio_features table
            cursor = conn.execute("""
                SELECT file_path, last_modified 
                FROM audio_features 
                WHERE failed = 0
            """)
            analyzed_files = {row[0]: row[1] for row in cursor.fetchall()}

        logger.debug(
            f"DISCOVERY: Found {len(analyzed_files)} files in audio_features table")
        if analyzed_files:
            sample_paths = list(analyzed_files.keys())[:3]
            logger.debug(
                f"DISCOVERY: Sample paths from audio_features: {sample_paths}")

        # Determine which files need analysis
        files_needing_analysis = []

        logger.debug(
            f"DISCOVERY: Comparing {len(discovery_files)} discovery files with {len(analyzed_files)} analyzed files")

        for file_path, (file_size, discovery_mtime) in discovery_files.items():
            logger.debug(f"DISCOVERY: Checking file: {file_path}")
            if file_path not in analyzed_files:
                # New file - needs analysis
                files_needing_analysis.append((file_path, 'new'))
                logger.debug(
                    f"DISCOVERY: New file needs analysis: {file_path}")
            elif os.path.exists(file_path):
                # Check if file was modified since last analysis
                current_mtime = os.path.getmtime(file_path)
                if current_mtime > analyzed_files[file_path]:
                    files_needing_analysis.append((file_path, 'modified'))
                    logger.debug(
                        f"DISCOVERY: Modified file needs analysis: {file_path}")

        # Clean up removed files from audio_features table
        removed_count = self._cleanup_removed_files_from_analysis(
            discovery_files.keys())
        if removed_count > 0:
            logger.info(
                f"DISCOVERY: Cleaned up {removed_count} removed files from analysis table")

        logger.info(
            f"DISCOVERY: Analysis needed for {len(files_needing_analysis)} files")
        return files_needing_analysis

    def _cleanup_removed_files_from_analysis(self, current_files):
        """Remove entries from audio_features table for files that no longer exist"""
        try:
            current_file_set = set(current_files)

            with self.db_pool.get_connection() as conn:
                # Get all files in audio_features table
                cursor = conn.execute("SELECT file_path FROM audio_features")
                db_files = [row[0] for row in cursor.fetchall()]

                removed_count = 0
                for db_file in db_files:
                    if db_file not in current_file_set:
                        # File no longer exists - remove from analysis table
                        conn.execute(
                            "DELETE FROM audio_features WHERE file_path = ?", (db_file,))
                        removed_count += 1
                        logger.debug(
                            f"DISCOVERY: Removed from analysis table: {db_file}")

            return removed_count

        except Exception as e:
            logger.error(f"DISCOVERY: Error cleaning up removed files: {e}")
            return 0

    def validate_cached_features(self):
        """Quick validation of all cached files"""
        logger.info("Validating cached features...")

        with self.db_pool.get_connection() as conn:
            cursor = conn.execute("""
                SELECT file_path, duration, bpm, centroid, loudness, danceability, 
                       key, scale, onset_rate, zcr, mfcc, chroma, 
                       spectral_flatness, spectral_rolloff, spectral_contrast
                FROM audio_features 
                WHERE failed = 0
            """)

            invalid_files = []
            for row in cursor.fetchall():
                file_path = row[0]
                features = {
                    'duration': row[1],
                    'bpm': row[2],
                    'centroid': row[3],
                    'loudness': row[4],
                    'danceability': row[5],
                    'key': row[6],
                    'scale': row[7],
                    'onset_rate': row[8],
                    'zcr': row[9],
                    'mfcc': json.loads(row[10]) if row[10] else [],
                    'chroma': json.loads(row[11]) if row[11] else [],
                    'spectral_flatness': row[12],
                    'spectral_rolloff': row[13],
                    'spectral_contrast': row[14]
                }

                # Quick validation
                if not self.is_valid_feature_set(features):
                    invalid_files.append(file_path)

        logger.info(
            f"Found {len(invalid_files)} files with invalid cached features")
        return invalid_files

    def is_valid_feature_set(self, features):
        """Quick validation of essential features"""
        try:
            # Check essential fields exist and have reasonable values
            if not features.get('duration') or features['duration'] <= 0:
                return False
            if not features.get('bpm') or features['bpm'] < 0 or features['bpm'] > 300:
                return False
            if not features.get('centroid') or features['centroid'] < 0:
                return False
            if not features.get('mfcc') or len(features['mfcc']) != 13:
                return False
            if not features.get('chroma') or len(features['chroma']) != 12:
                return False

            return True
        except Exception:
            return False

    def get_failed_files_from_db(self):
        """Get all failed files from database"""
        from .file_discovery import FileDiscovery

        with self.db_pool.get_connection() as conn:
            cursor = conn.execute("""
                SELECT file_path, last_analyzed 
                FROM audio_features 
                WHERE failed = 1
            """)

            failed_files = []
            file_discovery = FileDiscovery(audio_db=self)

            for row in cursor.fetchall():
                file_path, last_analyzed = row
                if os.path.exists(file_path):
                    # Use file discovery to validate the file
                    if not file_discovery._is_in_excluded_directory(file_path):
                        failed_files.append(file_path)
                    else:
                        logger.debug(
                            f"Skipping file already in failed directory: {file_path}")
                else:
                    logger.warning(f"Failed file no longer exists: {file_path}")

        logger.info(f"Found {len(failed_files)} failed files to retry")
        return failed_files

    def get_files_with_skipped_musicnn(self):
        """Get all files that had MusicNN extraction skipped."""
        try:
            with self.db_pool.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT file_path, file_hash, last_modified, metadata
                    FROM audio_features
                    WHERE musicnn_skipped = 1 AND failed = 0
                """)
                return [{
                    'file_path': row[0],
                    'file_hash': row[1],
                    'last_modified': row[2],
                    'metadata': json.loads(row[3]) if row[3] else {}
                } for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error getting files with skipped MusicNN: {e}")
            return []

    def get_invalid_files_from_db(self):
        """Get all files that need to be reanalyzed (failed files + files with skipped MusicNN)."""
        try:
            # Get failed files
            failed_files = self.get_failed_files_from_db()
            
            # Get files with skipped MusicNN
            skipped_musicnn_files = self.get_files_with_skipped_musicnn()
            
            # Combine and deduplicate
            all_invalid_files = set()
            
            # Add failed files
            for file_path in failed_files:
                all_invalid_files.add(file_path)
            
            # Add files with skipped MusicNN
            for file_info in skipped_musicnn_files:
                all_invalid_files.add(file_info['file_path'])
            
            logger.info(f"Found {len(failed_files)} failed files and {len(skipped_musicnn_files)} files with skipped MusicNN")
            logger.info(f"Total invalid files to reanalyze: {len(all_invalid_files)}")
            
            # Check if MusicNN models are available for reanalysis
            musicnn_available = self._check_musicnn_availability()
            if skipped_musicnn_files and not musicnn_available:
                logger.warning("MusicNN models are not available - files with skipped MusicNN will remain skipped")
            
            return list(all_invalid_files)
        except Exception as e:
            logger.error(f"Error getting invalid files from DB: {str(e)}")
            return []

    def get_only_failed_files_from_db(self):
        """Get only files that are marked as failed (excluding MusicNN skipped files)."""
        try:
            with self.db_pool.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT file_path, file_hash, last_modified, metadata
                    FROM audio_features
                    WHERE failed = 1
                """)
                return [{
                    'file_path': row[0],
                    'file_hash': row[1],
                    'last_modified': row[2],
                    'metadata': json.loads(row[3]) if row[3] else {}
                } for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error getting only failed files from DB: {str(e)}")
            return []

    def _check_musicnn_availability(self):
        """Check if MusicNN models are available."""
        try:
            model_path = os.getenv(
                'MUSICNN_MODEL_PATH',
                '/app/feature_extraction/models/musicnn/msd-musicnn-1.pb'
            )
            json_path = os.getenv(
                'MUSICNN_JSON_PATH',
                '/app/feature_extraction/models/musicnn/msd-musicnn-1.json'
            )
            
            model_exists = os.path.exists(model_path)
            json_exists = os.path.exists(json_path)
            
            if model_exists and json_exists:
                logger.debug("MusicNN models are available")
                return True
            else:
                logger.debug(f"MusicNN models missing - model: {model_exists}, json: {json_exists}")
                return False
        except Exception as e:
            logger.debug(f"Error checking MusicNN availability: {str(e)}")
            return False

    def unmark_as_failed(self, file_path):
        """Remove failed status from a file"""
        try:
            with self.db_pool.get_connection() as conn:
                conn.execute("""
                    UPDATE audio_features 
                    SET failed = 0, last_analyzed = CURRENT_TIMESTAMP 
                    WHERE file_path = ?
                """, (file_path,))
            logger.info(f"Unmarked {file_path} as failed")
            return True
        except Exception as e:
            logger.error(f"Error unmarking {file_path} as failed: {e}")
            return False

    def unmark_musicnn_skipped(self, file_path):
        """Remove MusicNN skipped status from a file"""
        try:
            with self.db_pool.get_connection() as conn:
                conn.execute("""
                    UPDATE audio_features 
                    SET musicnn_skipped = 0, last_analyzed = CURRENT_TIMESTAMP 
                    WHERE file_path = ?
                """, (file_path,))
            logger.info(f"Unmarked {file_path} as MusicNN skipped")
            return True
        except Exception as e:
            logger.error(f"Error unmarking {file_path} as MusicNN skipped: {e}")
            return False

    def move_to_failed_directory(self, file_path):
        """Move a file to the failed_files directory"""
        try:
            failed_dir = '/music/failed_files'
            os.makedirs(failed_dir, exist_ok=True)

            filename = os.path.basename(file_path)
            failed_path = os.path.join(failed_dir, filename)

            # Handle duplicate filenames
            counter = 1
            original_failed_path = failed_path
            while os.path.exists(failed_path):
                name, ext = os.path.splitext(original_failed_path)
                failed_path = f"{name}_{counter}{ext}"
                counter += 1

            os.rename(file_path, failed_path)
            logger.info(f"Moved {file_path} to {failed_path}")

            # Update database to reflect the move
            with self.db_pool.get_connection() as conn:
                conn.execute("""
                    UPDATE audio_features 
                    SET file_path = ?, failed = 1 
                    WHERE file_path = ?
                """, (failed_path, file_path))

            return True
        except Exception as e:
            logger.error(f"Error moving {file_path} to failed directory: {e}")
            return False

    def update_file_discovery_state(self, file_paths: List[str]):
        """Update file discovery state in database - only changed files."""
        logger.debug(
            f"DISCOVERY: Starting file discovery state update for {len(file_paths)} files")
        try:
            with self.db_pool.get_connection() as conn:
                # Get existing files from database
                cursor = conn.execute("""
                    SELECT file_path, file_hash, file_size, last_modified 
                    FROM file_discovery_state 
                    WHERE status = 'active'
                """)
                existing_files = {row[0]: (row[1], row[2], row[3])
                                  for row in cursor.fetchall()}

                added_count = 0
                updated_count = 0
                unchanged_count = 0

                # Process current files
                for file_path in file_paths:
                    if os.path.exists(file_path):
                        try:
                            stat = os.stat(file_path)
                            current_hash = self._get_file_hash(file_path)
                            current_size = stat.st_size
                            current_mtime = stat.st_mtime

                            if file_path in existing_files:
                                # File exists in database - check if changed
                                old_hash, old_size, old_mtime = existing_files[file_path]

                                if (current_hash != old_hash or
                                    current_size != old_size or
                                        current_mtime != old_mtime):
                                    # File changed - update it
                                    conn.execute("""
                                        UPDATE file_discovery_state 
                                        SET file_hash = ?, file_size = ?, last_modified = ?, last_seen_at = CURRENT_TIMESTAMP
                                        WHERE file_path = ?
                                    """, (current_hash, current_size, current_mtime, file_path))
                                    updated_count += 1
                                    logger.debug(
                                        f"DISCOVERY: Updated {file_path}")
                                else:
                                    # File unchanged - just update last_seen_at
                                    conn.execute("""
                                        UPDATE file_discovery_state 
                                        SET last_seen_at = CURRENT_TIMESTAMP
                                        WHERE file_path = ?
                                    """, (file_path,))
                                    unchanged_count += 1
                            else:
                                # New file - insert it
                                conn.execute("""
                                    INSERT INTO file_discovery_state 
                                    (file_path, file_hash, file_size, last_modified, last_seen_at, status)
                                    VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, 'active')
                                """, (file_path, current_hash, current_size, current_mtime))
                                added_count += 1
                                logger.debug(f"DISCOVERY: Added {file_path}")

                        except Exception as e:
                            logger.warning(
                                f"DISCOVERY: Could not update state for {file_path}: {e}")

                # Mark files that no longer exist as removed
                current_file_set = set(file_paths)
                removed_count = 0
                for existing_file in existing_files:
                    if existing_file not in current_file_set:
                        conn.execute("""
                            UPDATE file_discovery_state 
                            SET status = 'removed', last_seen_at = CURRENT_TIMESTAMP
                            WHERE file_path = ?
                        """, (existing_file,))
                        removed_count += 1
                        logger.debug(f"DISCOVERY: Removed {existing_file}")

            logger.info(
                f"DISCOVERY: File discovery state updated - Added: {added_count}, Updated: {updated_count}, Unchanged: {unchanged_count}, Removed: {removed_count}")

            # Verify the update worked
            with self.db_pool.get_connection() as conn:
                cursor = conn.execute(
                    "SELECT COUNT(*) FROM file_discovery_state WHERE status = 'active'")
                active_count = cursor.fetchone()[0]
                logger.debug(
                    f"DISCOVERY: Active files in discovery state: {active_count}")

        except Exception as e:
            logger.error(
                f"DISCOVERY: Error updating file discovery state: {e}")
            import traceback
            logger.error(f"DISCOVERY: Traceback: {traceback.format_exc()}")

    def get_file_discovery_changes(self) -> Tuple[List[str], List[str], List[str]]:
        """Get added, removed, and unchanged files from database."""
        try:
            with self.db_pool.get_connection() as conn:
                cursor = conn.execute("""
                    SELECT file_path, status 
                    FROM file_discovery_state 
                    WHERE status IN ('active', 'removed')
                """)

                active_files = []
                removed_files = []

                for row in cursor.fetchall():
                    file_path, status = row
                    if status == 'active':
                        active_files.append(file_path)
                    elif status == 'removed':
                        removed_files.append(file_path)

            # Get unchanged files (active files that still exist on disk)
            unchanged_files = []
            for file_path in active_files:
                if os.path.exists(file_path):
                    unchanged_files.append(file_path)

            # Added files are active files that weren't in the previous run
            # This is a simplified approach - in practice, you'd track this more precisely
            added_files = [f for f in active_files if f not in unchanged_files]

            logger.info(
                f"DISCOVERY: File discovery changes: Added={len(added_files)}, Removed={len(removed_files)}, Unchanged={len(unchanged_files)}")
            return added_files, removed_files, unchanged_files

        except Exception as e:
            logger.error(
                f"DISCOVERY: Error getting file discovery changes: {e}")
            return [], [], []

    def cleanup_file_discovery_state(self):
        """Remove entries for files that no longer exist."""
        try:
            with self.db_pool.get_connection() as conn:
                cursor = conn.execute(
                    "SELECT file_path FROM file_discovery_state")
                files_to_check = [row[0] for row in cursor.fetchall()]

                removed_count = 0
                for file_path in files_to_check:
                    if not os.path.exists(file_path):
                        conn.execute(
                            "DELETE FROM file_discovery_state WHERE file_path = ?", (file_path,))
                        removed_count += 1

            if removed_count > 0:
                logger.info(
                    f"DISCOVERY: Cleaned up {removed_count} non-existent files from discovery state")

        except Exception as e:
            logger.error(
                f"DISCOVERY: Error cleaning up file discovery state: {e}")

    def get_file_sizes_from_db(self, file_paths: List[str]) -> Dict[str, int]:
        """Get file sizes from database for the given file paths."""
        logger.debug(
            f"DISCOVERY: Requesting file sizes for {len(file_paths)} files from database")
        try:
            # First check if there's any data in the table
            with self.db_pool.get_connection() as conn:
                cursor = conn.execute(
                    "SELECT COUNT(*) FROM file_discovery_state")
                total_count = cursor.fetchone()[0]
                logger.debug(
                    f"DISCOVERY: Total records in file_discovery_state: {total_count}")

                if total_count == 0:
                    logger.warning(
                        "DISCOVERY: No data in file_discovery_state table")
                    return {}

                cursor = conn.execute("""
                    SELECT file_path, file_size 
                    FROM file_discovery_state 
                    WHERE file_path IN ({})
                """.format(','.join(['?' for _ in file_paths])), file_paths)

                file_sizes = {}
                for row in cursor.fetchall():
                    file_path, file_size = row
                    file_sizes[file_path] = file_size

                logger.debug(
                    f"DISCOVERY: Retrieved file sizes for {len(file_sizes)} files from database")
                return file_sizes

        except Exception as e:
            logger.error(
                f"DISCOVERY: Error getting file sizes from database: {e}")
            import traceback
            logger.error(f"DISCOVERY: Traceback: {traceback.format_exc()}")
            return {}
    
    def cleanup(self):
        """Clean up resources, including database connection pool."""
        try:
            if hasattr(self, 'db_pool'):
                self.db_pool.close_all()
                logger.debug("Database connection pool closed")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()


# Remove module-level instantiation to prevent deadlock during import
# audio_analyzer = AudioAnalyzer()
