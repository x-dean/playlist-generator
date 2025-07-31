"""
Configuration settings for the Playlista application.

This module contains all configuration classes and their default values,
extracted from the scattered environment variables and hard-coded values
throughout the codebase.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging

from ..exceptions import ConfigurationError


@dataclass
class LoggingConfig:
    """Configuration for logging settings."""
    
    level: str = field(default_factory=lambda: os.getenv('LOG_LEVEL', 'DEBUG'))
    log_dir: Path = field(default_factory=lambda: Path(os.getenv('LOG_DIR', '/app/logs')))
    log_file_prefix: str = field(default_factory=lambda: os.getenv('LOG_FILE_PREFIX', 'playlista'))
    colored_output: bool = field(default_factory=lambda: os.getenv('COLORED_OUTPUT', 'true').lower() == 'true')
    file_logging: bool = field(default_factory=lambda: os.getenv('FILE_LOGGING', 'true').lower() == 'true')
    console_logging: bool = field(default_factory=lambda: os.getenv('CONSOLE_LOGGING', 'true').lower() == 'true')
    
    # TensorFlow and Essentia logging
    tensorflow_log_level: str = "2"  # Hide INFO and WARNING, show only ERROR
    essentia_logging_level: str = "error"
    essentia_stream_logging: str = "none"
    
    # Troubleshooting settings
    verbose_output: bool = field(default_factory=lambda: os.getenv('VERBOSE_OUTPUT', 'true').lower() == 'true')
    show_progress: bool = field(default_factory=lambda: os.getenv('SHOW_PROGRESS', 'true').lower() == 'true')
    log_memory_usage: bool = field(default_factory=lambda: os.getenv('LOG_MEMORY_USAGE', 'true').lower() == 'true')
    log_performance: bool = field(default_factory=lambda: os.getenv('LOG_PERFORMANCE', 'true').lower() == 'true')
    max_log_files: int = field(default_factory=lambda: int(os.getenv('MAX_LOG_FILES', '10')))
    log_file_size_mb: int = field(default_factory=lambda: int(os.getenv('LOG_FILE_SIZE_MB', '50')))
    
    def __post_init__(self):
        """Validate and normalize configuration after initialization."""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate log level
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if self.level.upper() not in valid_levels:
            raise ValueError(f"Invalid log level: {self.level}. Must be one of {valid_levels}")


@dataclass
class DatabaseConfig:
    """Configuration for database settings."""
    
    cache_dir: Path = field(default_factory=lambda: Path(os.getenv('CACHE_DIR', '/app/cache')))
    cache_file: str = "audio_analysis.db"
    playlist_db_file: str = "playlist.db"
    
    # SQLite settings
    pragma_cache_size: int = 10000
    pragma_journal_mode: str = "WAL"
    pragma_synchronous: str = "NORMAL"
    
    def __post_init__(self):
        """Validate and normalize configuration after initialization."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_dir / self.cache_file
        self.playlist_db_path = self.cache_dir / self.playlist_db_file


@dataclass
class MemoryConfig:
    """Configuration for memory management."""
    
    memory_limit_gb: float = field(default_factory=lambda: float(os.getenv('MEMORY_LIMIT_GB', '6.0')))
    rss_limit_gb: float = 6.0
    memory_aware: bool = field(default_factory=lambda: os.getenv('MEMORY_AWARE', 'false').lower() == 'true')
    low_memory_mode: bool = False
    
    # Memory pressure thresholds
    memory_pressure_threshold: float = 0.8  # 80% memory usage
    memory_pressure_pause_seconds: int = 30
    
    def __post_init__(self):
        """Validate and normalize configuration after initialization."""
        if self.memory_limit_gb <= 0:
            raise ValueError("Memory limit must be positive")
        if self.rss_limit_gb <= 0:
            raise ValueError("RSS limit must be positive")


@dataclass
class ProcessingConfig:
    """Configuration for processing settings."""
    
    # Worker configuration
    default_workers: Optional[int] = None  # Will be set to CPU count / 2
    max_workers: Optional[int] = None  # Will be set to CPU count
    batch_size_multiplier: int = 10
    max_batch_size: int = 100
    
    # File processing thresholds
    large_file_threshold_mb: int = field(default_factory=lambda: int(os.getenv('LARGE_FILE_THRESHOLD', '50')))
    very_large_file_threshold_mb: int = 100
    extremely_large_file_threshold_mb: int = 500
    
    # Timeout settings
    file_timeout_minutes: int = 10
    batch_timeout_minutes: int = 30
    max_retries: int = 3
    
    # Audio processing limits
    max_audio_samples: int = 150_000_000  # ~5.7 hours at 44kHz
    max_samples_for_mfcc: int = 100_000_000  # ~2.3 hours at 44kHz
    max_samples_for_processing: int = 500_000_000  # ~11.3 hours at 44kHz
    
    # Sample rate settings
    default_sample_rate: int = 44100
    
    def __post_init__(self):
        """Validate and normalize configuration after initialization."""
        import multiprocessing as mp
        
        if self.default_workers is None:
            self.default_workers = max(1, mp.cpu_count() // 2)
        
        if self.max_workers is None:
            self.max_workers = mp.cpu_count()
        
        if self.large_file_threshold_mb <= 0:
            raise ValueError("Large file threshold must be positive")


@dataclass
class AudioAnalysisConfig:
    """Configuration for audio analysis settings."""
    
    # Processing mode
    fast_mode: bool = field(default_factory=lambda: os.getenv('FAST_MODE', 'false').lower() == 'true')
    
    # Feature extraction settings
    extract_bpm: bool = True
    extract_mfcc: bool = True
    extract_chroma: bool = True
    extract_musicnn: bool = True
    extract_spectral: bool = True
    
    # Model paths
    model_dir: Path = field(default_factory=lambda: Path(os.getenv('MODEL_DIR', '/app/feature_extraction/models')))
    musicnn_model_path: Path = field(default_factory=lambda: Path(os.getenv('MUSICNN_MODEL_PATH', '/app/feature_extraction/models/msd-musicnn-1.pb')))
    musicnn_json_path: Path = field(default_factory=lambda: Path(os.getenv('MUSICNN_JSON_PATH', '/app/feature_extraction/models/musicnn/msd-musicnn-1.json')))
    
    # BPM extraction settings
    bpm_min: int = 60
    bpm_max: int = 200
    bpm_confidence_threshold: float = 0.5
    
    # Feature normalization
    normalize_danceability: bool = True
    normalize_centroid: bool = True
    
    def __post_init__(self):
        """Validate and normalize configuration after initialization."""
        if not self.model_dir.exists():
            self.model_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class PlaylistConfig:
    """Configuration for playlist generation settings."""
    
    # Playlist size constraints
    min_tracks_per_playlist: int = 10
    max_tracks_per_playlist: int = 500
    default_num_playlists: int = 8
    
    # Tag-based playlist settings
    min_tracks_per_genre: int = field(default_factory=lambda: int(os.getenv('MIN_TRACKS_PER_GENRE', '10')))
    
    # Time-based playlist settings
    time_slots: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        'Morning': {'min_bpm': 50, 'max_bpm': 100, 'min_centroid': 1000},
        'Afternoon': {'min_bpm': 80, 'max_bpm': 130, 'min_centroid': 1500},
        'Evening': {'min_bpm': 70, 'max_bpm': 120, 'max_centroid': 3500},
        'Night': {'min_bpm': 60, 'max_bpm': 90, 'max_centroid': 2000}
    })
    
    # K-means settings
    kmeans_n_clusters: int = 8
    kmeans_max_iter: int = 300
    kmeans_tolerance: float = 1e-4
    
    # Feature grouping settings
    feature_bins: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        'Upbeat': {'min_bpm': 120, 'max_bpm': 150, 'description': 'Energetic and lively tracks'},
        'Fast': {'min_bpm': 150, 'max_bpm': 180, 'description': 'High-energy dance music'},
        'Dark': {'min_centroid': 0, 'max_centroid': 1000, 'description': 'Deep and atmospheric'},
        'Warm': {'min_centroid': 1000, 'max_centroid': 2000, 'description': 'Rich and full-bodied sound'}
    })
    
    def __post_init__(self):
        """Validate and normalize configuration after initialization."""
        if self.min_tracks_per_playlist <= 0:
            raise ValueError("Minimum tracks per playlist must be positive")
        if self.max_tracks_per_playlist <= self.min_tracks_per_playlist:
            raise ValueError("Maximum tracks must be greater than minimum")


@dataclass
class FileDiscoveryConfig:
    """Configuration for file discovery settings."""
    
    # Tracking method
    use_hash_tracking: bool = field(default_factory=lambda: os.getenv('USE_HASH_TRACKING', 'true').lower() == 'true')
    
    # File filtering
    min_file_size_kb: int = field(default_factory=lambda: int(os.getenv('MIN_FILE_SIZE_KB', '1')))
    valid_extensions: tuple = field(default_factory=lambda: tuple(os.getenv('VALID_EXTENSIONS', '.mp3,.wav,.flac,.ogg,.m4a,.aac,.opus').split(',')))
    
    # Progress reporting
    progress_interval: int = field(default_factory=lambda: int(os.getenv('PROGRESS_INTERVAL', '100')))
    
    def __post_init__(self):
        """Validate and normalize configuration after initialization."""
        if self.min_file_size_kb < 0:
            raise ValueError("Minimum file size must be non-negative")


@dataclass
class ExternalAPIConfig:
    """Configuration for external API integrations."""
    
    # MusicBrainz settings
    musicbrainz_user_agent: str = "Playlista/1.0"
    musicbrainz_rate_limit: float = 1.0  # requests per second
    
    # Last.fm settings
    lastfm_api_key: str = field(default_factory=lambda: os.getenv('LASTFM_API_KEY', '9fd1f789ebdf1297e6aa1590a13d85e0'))
    lastfm_rate_limit: float = 2.0  # requests per second
    
    # Spotify settings (future)
    spotify_client_id: str = field(default_factory=lambda: os.getenv('SPOTIFY_CLIENT_ID', ''))
    spotify_client_secret: str = field(default_factory=lambda: os.getenv('SPOTIFY_CLIENT_SECRET', ''))
    
    # Discogs settings (future)
    discogs_user_token: str = field(default_factory=lambda: os.getenv('DISCOGS_USER_TOKEN', ''))
    
    def __post_init__(self):
        """Validate and normalize configuration after initialization."""
        if not self.lastfm_api_key:
            logging.warning("Last.fm API key not provided - tag-based playlists may be limited")


@dataclass
class UnifiedProcessingConfig:
    """Unified configuration for discovery and analysis processing."""
    
    # File extensions - unified across all services
    audio_extensions: List[str] = field(default_factory=lambda: [
        '.mp3', '.flac', '.wav', '.m4a', '.ogg', '.opus', '.aac', '.wma', '.aiff', '.alac'
    ])
    
    # File size thresholds - unified across all services
    min_file_size_bytes: int = 1024  # 1KB minimum
    large_file_threshold_mb: int = field(default_factory=lambda: int(os.getenv('LARGE_FILE_THRESHOLD', '50')))
    very_large_file_threshold_mb: int = 100
    extremely_large_file_threshold_mb: int = 500
    
    # Processing options - unified across all services
    parallel_processing: bool = True
    max_workers: Optional[int] = None
    batch_size: Optional[int] = None
    timeout_seconds: int = 300
    
    # Memory management - unified across all services
    memory_limit_gb: float = field(default_factory=lambda: float(os.getenv('MEMORY_LIMIT_GB', '6.0')))
    memory_aware: bool = field(default_factory=lambda: os.getenv('MEMORY_AWARE', 'false').lower() == 'true')
    memory_pressure_threshold: float = 0.8
    
    # Cache settings - unified across all services
    cache_enabled: bool = True
    cache_ttl_seconds: Optional[int] = 3600  # 1 hour default
    cache_max_size_mb: int = 1000  # 1GB cache limit
    
    # Error handling - unified across all services
    max_retries: int = 3
    retry_delay_seconds: int = 5
    fail_fast: bool = False
    
    def __post_init__(self):
        """Validate and normalize configuration after initialization."""
        import multiprocessing as mp
        
        if self.max_workers is None:
            self.max_workers = max(1, mp.cpu_count() // 2)
        
        if self.batch_size is None:
            self.batch_size = max(10, self.max_workers * 2)
        
        if self.memory_limit_gb <= 0:
            raise ValueError("Memory limit must be positive")
        
        if self.large_file_threshold_mb <= 0:
            raise ValueError("Large file threshold must be positive")
    
    def get_audio_extensions_set(self) -> set:
        """Get audio extensions as a set for efficient lookup."""
        return {ext.lower() for ext in self.audio_extensions}
    
    def is_valid_audio_file(self, file_path: Path) -> bool:
        """Check if a file is a valid audio file based on unified criteria."""
        try:
            # Check extension
            if file_path.suffix.lower() not in self.get_audio_extensions_set():
                return False
            
            # Check file exists and is readable
            if not file_path.exists() or not file_path.is_file():
                return False
            
            # Check file size
            file_size = file_path.stat().st_size
            if file_size < self.min_file_size_bytes:
                return False
            
            return True
            
        except Exception:
            return False
    
    def get_file_size_category(self, file_size_bytes: int) -> str:
        """Categorize file by size."""
        file_size_mb = file_size_bytes / (1024 * 1024)
        
        if file_size_mb < 5:
            return "small"
        elif file_size_mb < self.large_file_threshold_mb:
            return "medium"
        elif file_size_mb < self.very_large_file_threshold_mb:
            return "large"
        else:
            return "very_large"


@dataclass
class AppConfig:
    """Main application configuration."""
    
    # Core paths
    host_library_path: Path = field(default_factory=lambda: Path(os.getenv('HOST_LIBRARY_PATH', '/root/music/library')))
    music_path: Path = Path('/music')
    output_dir: Path = field(default_factory=lambda: Path(os.getenv('OUTPUT_DIR', '/app/playlists')))
    
    # Sub-configurations
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    audio_analysis: AudioAnalysisConfig = field(default_factory=AudioAnalysisConfig)
    playlist: PlaylistConfig = field(default_factory=PlaylistConfig)
    external_api: ExternalAPIConfig = field(default_factory=ExternalAPIConfig)
    file_discovery: FileDiscoveryConfig = field(default_factory=FileDiscoveryConfig)
    
    # FIXED: Add unified processing configuration
    unified_processing: UnifiedProcessingConfig = field(default_factory=UnifiedProcessingConfig)
    
    # Application settings
    app_name: str = "Playlista"
    app_version: str = "1.0.0"
    debug_mode: bool = field(default_factory=lambda: os.getenv('DEBUG', 'false').lower() == 'true')
    
    def __post_init__(self):
        """Validate and normalize configuration after initialization."""
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate container paths only (host paths are not accessible in Docker)
        if not self.music_path.exists():
            logging.warning(f"Music path does not exist: {self.music_path}")
        
        # FIXED: Synchronize configurations to prevent mismatches
        self._synchronize_configurations()
    
    def _synchronize_configurations(self):
        """Synchronize configurations to ensure consistency."""
        # Sync file extensions across all services
        unified_extensions = self.unified_processing.get_audio_extensions_set()
        
        # Update file discovery config
        self.file_discovery.file_extensions = list(unified_extensions)
        
        # Update processing config
        self.processing.large_file_threshold_mb = self.unified_processing.large_file_threshold_mb
        self.processing.max_workers = self.unified_processing.max_workers
        self.processing.batch_size = self.unified_processing.batch_size
        
        # Update memory config
        self.memory.memory_limit_gb = self.unified_processing.memory_limit_gb
        self.memory.memory_aware = self.unified_processing.memory_aware
        self.memory.memory_pressure_threshold = self.unified_processing.memory_pressure_threshold
        
        # Update audio analysis config
        self.audio_analysis.fast_mode = self.unified_processing.parallel_processing
        
        logging.info("Configuration synchronization completed")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization."""
        return {
            'app_name': self.app_name,
            'app_version': self.app_version,
            'debug_mode': self.debug_mode,
            'music_path': str(self.music_path),
            'output_dir': str(self.output_dir),
            'host_library_path': str(self.host_library_path),
            'logging': self.logging.__dict__,
            'database': self.database.__dict__,
            'memory': self.memory.__dict__,
            'processing': self.processing.__dict__,
            'audio_analysis': self.audio_analysis.__dict__,
            'playlist': self.playlist.__dict__,
            'external_api': self.external_api.__dict__,
            'file_discovery': self.file_discovery.__dict__,
            'unified_processing': self.unified_processing.__dict__
        }


def load_config() -> AppConfig:
    """Load application configuration from environment variables."""
    return AppConfig()


def validate_config(config: AppConfig) -> None:
    """Validate the configuration and raise exceptions for invalid settings."""
    # This will be called during AppConfig.__post_init__
    pass 