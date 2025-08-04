"""
Configuration loader for Playlist Generator Simple.
Handles loading and caching of configuration from multiple sources.
"""

import os
# Configure TensorFlow logging BEFORE any imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Hide INFO and WARNING, show only ERROR
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimization messages
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU to avoid GPU-related warnings
import json
import time
from typing import Dict, Any, Optional
from pathlib import Path

# Import logging
from .logging_setup import get_logger, log_universal

logger = get_logger('playlista.config_loader')


class ConfigLoader:
    """
    Configuration loader with caching support.
    
    Features:
    - Multiple configuration sources (file, environment, defaults)
    - Intelligent caching to avoid repeated file reads
    - Configuration validation and type conversion
    - Environment variable overrides
    """
    
    def __init__(self):
        """Initialize configuration loader."""
        self._config_cache = {}
        self._cache_timestamps = {}
        self._cache_duration = 300  # 5 minutes cache duration
        
        # Default configuration paths
        self.config_paths = [
            'playlista.conf',
            'config/playlista.conf',
            '/app/playlista.conf',
            os.path.expanduser('~/.playlista.conf')
        ]
        
        log_universal('INFO', 'Config', 'ConfigLoader initialized')
    
    def _get_cache_key(self, config_type: str) -> str:
        """Generate cache key for configuration type."""
        return f"config:{config_type}"
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached configuration is still valid."""
        if cache_key not in self._cache_timestamps:
            return False
        
        cache_age = time.time() - self._cache_timestamps[cache_key]
        return cache_age < self._cache_duration
    
    def _update_cache(self, cache_key: str, config_data: Dict[str, Any]):
        """Update configuration cache."""
        self._config_cache[cache_key] = config_data
        self._cache_timestamps[cache_key] = time.time()
        log_universal('DEBUG', 'Config', f'Updated cache for: {cache_key}')
    
    def _load_config_file(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Load configuration from file.
        
        Args:
            file_path: Path to configuration file
            
        Returns:
            Configuration dictionary or None on failure
        """
        try:
            if not os.path.exists(file_path):
                return None
            
            # Check file modification time for cache invalidation
            file_mtime = os.path.getmtime(file_path)
            cache_key = f"file:{file_path}"
            
            # Check if we have a valid cached version
            if (cache_key in self._cache_timestamps and 
                self._cache_timestamps[cache_key] >= file_mtime):
                log_universal('DEBUG', 'Config', f'Using cached config from: {file_path}')
                return self._config_cache.get(cache_key)
            
            log_universal('DEBUG', 'Config', f'Loading configuration from: {file_path}')
            
            config = {}
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    
                    # Skip comments and empty lines
                    if not line or line.startswith('#'):
                        continue
                    
                    # Parse key=value pairs
                    if '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        
                        # Remove quotes if present
                        if (value.startswith('"') and value.endswith('"')) or \
                           (value.startswith("'") and value.endswith("'")):
                            value = value[1:-1]
                        
                        # Convert value types
                        config[key] = self._convert_value_type(value)
            
            # Cache the loaded configuration
            self._update_cache(cache_key, config)
            
            log_universal('INFO', 'Config', f'Loaded {len(config)} settings from: {file_path}')
            return config
            
        except Exception as e:
            log_universal('ERROR', 'Config', f'Failed to load config from {file_path}: {e}')
            return None
    
    def _convert_value_type(self, value: str) -> Any:
        """
        Convert string value to appropriate type.
        
        Args:
            value: String value from config file
            
        Returns:
            Converted value with appropriate type
        """
        # Boolean values
        if value.lower() in ('true', 'yes', 'on', '1'):
            return True
        elif value.lower() in ('false', 'no', 'off', '0'):
            return False
        
        # Integer values
        try:
            if '.' not in value:  # Avoid converting floats to ints
                return int(value)
        except ValueError:
            pass
        
        # Float values
        try:
            return float(value)
        except ValueError:
            pass
        
        # String values (default)
        return value
    
    def _load_environment_overrides(self) -> Dict[str, Any]:
        """
        Load configuration overrides from environment variables.
        
        Returns:
            Dictionary of environment-based configuration
        """
        env_config = {}
        
        # Common configuration environment variables
        env_mappings = {
            'MUSIC_PATH': 'MUSIC_PATH',
            'DB_PATH': 'DB_PATH',
            'LOG_LEVEL': 'LOG_LEVEL',
            'LOG_FILE': 'LOG_FILE',
            'CACHE_ENABLED': 'CACHE_ENABLED',
            'CACHE_EXPIRY_HOURS': 'CACHE_EXPIRY_HOURS',
            'ANALYSIS_TIMEOUT': 'ANALYSIS_TIMEOUT',
            'WORKERS': 'WORKERS',
            'BATCH_SIZE': 'BATCH_SIZE',
            'MUSICBRAINZ_ENABLED': 'MUSICBRAINZ_ENABLED',
            'LASTFM_ENABLED': 'LASTFM_ENABLED',
            'LASTFM_API_KEY': 'LASTFM_API_KEY',
            'SAMPLE_RATE': 'SAMPLE_RATE',
            'HOP_SIZE': 'HOP_SIZE',
            'FRAME_SIZE': 'FRAME_SIZE',
            'EXTRACT_RHYTHM': 'EXTRACT_RHYTHM',
            'EXTRACT_SPECTRAL': 'EXTRACT_SPECTRAL',
            'EXTRACT_LOUDNESS': 'EXTRACT_LOUDNESS',
            'EXTRACT_KEY': 'EXTRACT_KEY',
            'EXTRACT_MFCC': 'EXTRACT_MFCC',
            'EXTRACT_MUSICNN': 'EXTRACT_MUSICNN',
            'MUSICNN_MODEL_PATH': 'MUSICNN_MODEL_PATH',
            'MUSICNN_JSON_PATH': 'MUSICNN_JSON_PATH',
            'MUSICNN_TIMEOUT_SECONDS': 'MUSICNN_TIMEOUT_SECONDS',
            'EXTRACT_CHROMA': 'EXTRACT_CHROMA',
            'FORCE_REANALYSIS': 'FORCE_REANALYSIS',
            'DB_CACHE_DEFAULT_EXPIRY_HOURS': 'DB_CACHE_DEFAULT_EXPIRY_HOURS',
            'DB_CACHE_CLEANUP_FREQUENCY_HOURS': 'DB_CACHE_CLEANUP_FREQUENCY_HOURS',
            'DB_CACHE_MAX_SIZE_MB': 'DB_CACHE_MAX_SIZE_MB',
            'DB_CLEANUP_RETENTION_DAYS': 'DB_CLEANUP_RETENTION_DAYS',
            'DB_FAILED_ANALYSIS_RETENTION_DAYS': 'DB_FAILED_ANALYSIS_RETENTION_DAYS',
            'DB_STATISTICS_RETENTION_DAYS': 'DB_STATISTICS_RETENTION_DAYS',
            'DB_CONNECTION_TIMEOUT_SECONDS': 'DB_CONNECTION_TIMEOUT_SECONDS',
            'DB_MAX_RETRY_ATTEMPTS': 'DB_MAX_RETRY_ATTEMPTS',
            'DB_BATCH_SIZE': 'DB_BATCH_SIZE',
            'DB_STATISTICS_COLLECTION_FREQUENCY_HOURS': 'DB_STATISTICS_COLLECTION_FREQUENCY_HOURS',
            'DB_AUTO_CLEANUP_ENABLED': 'DB_AUTO_CLEANUP_ENABLED',
            'DB_AUTO_CLEANUP_FREQUENCY_HOURS': 'DB_AUTO_CLEANUP_FREQUENCY_HOURS',
            'DB_BACKUP_ENABLED': 'DB_BACKUP_ENABLED',
            'DB_BACKUP_FREQUENCY_HOURS': 'DB_BACKUP_FREQUENCY_HOURS',
            'DB_BACKUP_RETENTION_DAYS': 'DB_BACKUP_RETENTION_DAYS',
            'DB_PERFORMANCE_MONITORING_ENABLED': 'DB_PERFORMANCE_MONITORING_ENABLED',
            'DB_QUERY_TIMEOUT_SECONDS': 'DB_QUERY_TIMEOUT_SECONDS',
            'DB_MAX_CONNECTIONS': 'DB_MAX_CONNECTIONS',
            'DB_WAL_MODE_ENABLED': 'DB_WAL_MODE_ENABLED',
            'DB_SYNCHRONOUS_MODE': 'DB_SYNCHRONOUS_MODE'
        }
        
        for env_var, config_key in env_mappings.items():
            if env_var in os.environ:
                value = os.environ[env_var]
                env_config[config_key] = self._convert_value_type(value)
        
        if env_config:
            log_universal('DEBUG', 'Config', f'Loaded {len(env_config)} environment overrides')
        
        return env_config
    
    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration values.
        
        Returns:
            Dictionary of default configuration values
        """
        return {
            # File paths
            'MUSIC_PATH': '/music',
            'DB_PATH': '/app/cache/playlista.db',
            'LOG_FILE': '/app/logs/playlista.log',
            'FAILED_FILES_DIR': '/app/cache/failed_dir',
            
            # Logging
            'LOG_LEVEL': 'INFO',
            'LOG_FORMAT': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'LOG_COLORED': True,
            'LOG_BUFFERED': True,
            
            # Analysis settings
            'ANALYSIS_TIMEOUT': 600,
            'WORKERS': 4,
            'BATCH_SIZE': 100,
            'SAMPLE_RATE': 44100,
            'HOP_SIZE': 512,
            'FRAME_SIZE': 2048,
            
            # Feature extraction
            'EXTRACT_RHYTHM': True,
            'EXTRACT_SPECTRAL': True,
            'EXTRACT_LOUDNESS': True,
            'EXTRACT_KEY': True,
            'EXTRACT_MFCC': True,
            'EXTRACT_MUSICNN': True,
            'MUSICNN_MODEL_PATH': '/app/models/msd-musicnn-1.pb',
            'MUSICNN_JSON_PATH': '/app/models/msd-musicnn-1.json',
            'MUSICNN_TIMEOUT_SECONDS': 60,
            'EXTRACT_CHROMA': True,
            'FORCE_REANALYSIS': False,
            
            # Caching
            'CACHE_ENABLED': True,
            'CACHE_EXPIRY_HOURS': 168,  # 1 week
            
            # External APIs
            'MUSICBRAINZ_ENABLED': True,
            'LASTFM_ENABLED': True,
            'LASTFM_API_KEY': '',
            
            # Database settings
            'DB_CACHE_DEFAULT_EXPIRY_HOURS': 24,
            'DB_CACHE_CLEANUP_FREQUENCY_HOURS': 24,
            'DB_CACHE_MAX_SIZE_MB': 100,
            'DB_CLEANUP_RETENTION_DAYS': 30,
            'DB_FAILED_ANALYSIS_RETENTION_DAYS': 7,
            'DB_STATISTICS_RETENTION_DAYS': 90,
            'DB_CONNECTION_TIMEOUT_SECONDS': 30,
            'DB_MAX_RETRY_ATTEMPTS': 3,
            'DB_BATCH_SIZE': 100,
            'DB_STATISTICS_COLLECTION_FREQUENCY_HOURS': 24,
            'DB_AUTO_CLEANUP_ENABLED': True,
            'DB_AUTO_CLEANUP_FREQUENCY_HOURS': 168,  # 1 week
            'DB_BACKUP_ENABLED': True,
            'DB_BACKUP_FREQUENCY_HOURS': 168,  # 1 week
            'DB_BACKUP_RETENTION_DAYS': 30,
            'DB_PERFORMANCE_MONITORING_ENABLED': True,
            'DB_QUERY_TIMEOUT_SECONDS': 60,
            'DB_MAX_CONNECTIONS': 10,
            'DB_WAL_MODE_ENABLED': True,
            'DB_SYNCHRONOUS_MODE': 'NORMAL'
        }
    
    def get_config(self, force_reload: bool = False) -> Dict[str, Any]:
        """
        Get complete configuration with caching.
        
        Args:
            force_reload: Force reload configuration (ignore cache)
            
        Returns:
            Complete configuration dictionary
        """
        cache_key = self._get_cache_key('complete')
        
        # Check cache first (unless forced reload)
        if not force_reload and self._is_cache_valid(cache_key):
            log_universal('DEBUG', 'Config', 'Using cached configuration')
            return self._config_cache[cache_key]
        
        log_universal('INFO', 'Config', 'Loading configuration')
        print("CONFIG_DEBUG: Starting configuration loading")  # Direct print for debugging
        
        # Start with defaults
        config = self._get_default_config()
        
        # Load from configuration files (in order of priority)
        for config_path in self.config_paths:
            file_config = self._load_config_file(config_path)
            if file_config:
                config.update(file_config)
                log_universal('INFO', 'Config', f'Applied config from: {config_path}')
                print(f"CONFIG_DEBUG: Applied config from: {config_path}")  # Direct print for debugging
                # Log the MUSICNN_HALF_TRACK_THRESHOLD_MB value for debugging
                if 'MUSICNN_HALF_TRACK_THRESHOLD_MB' in file_config:
                    log_universal('INFO', 'Config', f'  MUSICNN_HALF_TRACK_THRESHOLD_MB = {file_config["MUSICNN_HALF_TRACK_THRESHOLD_MB"]}')
                    print(f"CONFIG_DEBUG: MUSICNN_HALF_TRACK_THRESHOLD_MB = {file_config['MUSICNN_HALF_TRACK_THRESHOLD_MB']}")  # Direct print for debugging
        
        # Apply environment overrides (highest priority)
        env_config = self._load_environment_overrides()
        config.update(env_config)
        
        # Cache the complete configuration
        self._update_cache(cache_key, config)
        
        log_universal('INFO', 'Config', f'Configuration loaded with {len(config)} settings')
        return config
    
    def get_audio_analysis_config(self) -> Dict[str, Any]:
        """
        Get audio analysis specific configuration.
        
        Returns:
            Audio analysis configuration dictionary
        """
        cache_key = self._get_cache_key('audio_analysis')
        
        if self._is_cache_valid(cache_key):
            return self._config_cache[cache_key]
        
        config = self.get_config()
        
        # Extract audio analysis specific settings
        audio_config = {
            'SAMPLE_RATE': config.get('SAMPLE_RATE', 44100),
            'HOP_SIZE': config.get('HOP_SIZE', 512),
            'FRAME_SIZE': config.get('FRAME_SIZE', 2048),
            'TIMEOUT_SECONDS': config.get('ANALYSIS_TIMEOUT', 600),
            'EXTRACT_RHYTHM': config.get('EXTRACT_RHYTHM', True),
            'EXTRACT_SPECTRAL': config.get('EXTRACT_SPECTRAL', True),
            'EXTRACT_LOUDNESS': config.get('EXTRACT_LOUDNESS', True),
            'EXTRACT_KEY': config.get('EXTRACT_KEY', True),
            'EXTRACT_MFCC': config.get('EXTRACT_MFCC', True),
            'EXTRACT_MUSICNN': config.get('EXTRACT_MUSICNN', True),
            'MUSICNN_MODEL_PATH': config.get('MUSICNN_MODEL_PATH', '/app/models/msd-musicnn-1.pb'),
            'MUSICNN_JSON_PATH': config.get('MUSICNN_JSON_PATH', '/app/models/msd-musicnn-1.json'),
            'MUSICNN_TIMEOUT_SECONDS': config.get('MUSICNN_TIMEOUT_SECONDS', 60),
            'MUSICNN_HALF_TRACK_THRESHOLD_MB': config.get('MUSICNN_HALF_TRACK_THRESHOLD_MB', 50),
            'EXTRACT_CHROMA': config.get('EXTRACT_CHROMA', True),
            'EXTRACT_DANCEABILITY': config.get('EXTRACT_DANCEABILITY', True),
            'EXTRACT_ONSET_RATE': config.get('EXTRACT_ONSET_RATE', True),
            'EXTRACT_ZCR': config.get('EXTRACT_ZCR', True),
            'EXTRACT_SPECTRAL_CONTRAST': config.get('EXTRACT_SPECTRAL_CONTRAST', True),
            'FORCE_REANALYSIS': config.get('FORCE_REANALYSIS', False),
            'CACHE_ENABLED': config.get('CACHE_ENABLED', True),
            'CACHE_EXPIRY_HOURS': config.get('CACHE_EXPIRY_HOURS', 168),
            # Long audio configuration
            'LONG_AUDIO_ENABLED': config.get('LONG_AUDIO_ENABLED', True),
            'LONG_AUDIO_DURATION_THRESHOLD_MINUTES': config.get('LONG_AUDIO_DURATION_THRESHOLD_MINUTES', 20),
            'LONG_AUDIO_SIMPLIFIED_FEATURES': config.get('LONG_AUDIO_SIMPLIFIED_FEATURES', True),
            'LONG_AUDIO_CATEGORIES': config.get('LONG_AUDIO_CATEGORIES', 'long_mix,podcast,radio,compilation'),
            'LONG_AUDIO_SKIP_DETAILED_ANALYSIS': config.get('LONG_AUDIO_SKIP_DETAILED_ANALYSIS', True),
            'LONG_AUDIO_EXTRACT_BASIC_FEATURES': config.get('LONG_AUDIO_EXTRACT_BASIC_FEATURES', True),
            # Large file handling
            'MAX_AUDIO_FILE_SIZE_MB': config.get('MAX_AUDIO_FILE_SIZE_MB', 500),
            'LARGE_FILE_WARNING_THRESHOLD_MB': config.get('LARGE_FILE_WARNING_THRESHOLD_MB', 100),
            'SKIP_LARGE_FILES': config.get('SKIP_LARGE_FILES', True),
            # Parallel processing options
            'USE_THREADED_PROCESSING': config.get('USE_THREADED_PROCESSING', False),
            'THREADED_WORKERS_DEFAULT': config.get('THREADED_WORKERS_DEFAULT', 4)
        }
        
        self._update_cache(cache_key, audio_config)
        
        # Log the final MUSICNN_HALF_TRACK_THRESHOLD_MB value for debugging
        if 'MUSICNN_HALF_TRACK_THRESHOLD_MB' in audio_config:
            log_universal('INFO', 'Config', f'Audio analysis config: MUSICNN_HALF_TRACK_THRESHOLD_MB = {audio_config["MUSICNN_HALF_TRACK_THRESHOLD_MB"]}')
            print(f"CONFIG_DEBUG: Final audio analysis config: MUSICNN_HALF_TRACK_THRESHOLD_MB = {audio_config['MUSICNN_HALF_TRACK_THRESHOLD_MB']}")  # Direct print for debugging
        
        return audio_config
    
    def get_file_discovery_config(self) -> Dict[str, Any]:
        """
        Get file discovery specific configuration.
        
        Returns:
            File discovery configuration dictionary
        """
        cache_key = self._get_cache_key('file_discovery')
        
        if self._is_cache_valid(cache_key):
            return self._config_cache[cache_key]
        
        config = self.get_config()
        
        # Extract file discovery specific settings
        # Note: Music, failed, and database paths are fixed Docker paths
        # Note: Exclude directories are fixed and not configurable
        file_discovery_config = {
            'MIN_FILE_SIZE_BYTES': config.get('MIN_FILE_SIZE_BYTES', 10240),
            'MAX_FILE_SIZE_BYTES': config.get('MAX_FILE_SIZE_BYTES', None),
            'VALID_EXTENSIONS': config.get('VALID_EXTENSIONS', ['.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac', '.opus']),
            'HASH_ALGORITHM': config.get('HASH_ALGORITHM', 'md5'),
            'MAX_RETRY_COUNT': config.get('MAX_RETRY_COUNT', 3),
            'ENABLE_RECURSIVE_SCAN': config.get('ENABLE_RECURSIVE_SCAN', True),
            'ENABLE_DETAILED_LOGGING': config.get('ENABLE_DETAILED_LOGGING', True)
        }
        
        # Convert string extensions to list if needed
        if isinstance(file_discovery_config['VALID_EXTENSIONS'], str):
            extensions = [ext.strip() for ext in file_discovery_config['VALID_EXTENSIONS'].split(',')]
            # Ensure extensions start with dot
            valid_extensions = []
            for ext in extensions:
                if not ext.startswith('.'):
                    ext = '.' + ext
                valid_extensions.append(ext.lower())
            file_discovery_config['VALID_EXTENSIONS'] = valid_extensions
        
        self._update_cache(cache_key, file_discovery_config)
        return file_discovery_config
    
    def get_database_config(self) -> Dict[str, Any]:
        """
        Get database specific configuration.
        
        Returns:
            Database configuration dictionary
        """
        cache_key = self._get_cache_key('database')
        
        if self._is_cache_valid(cache_key):
            return self._config_cache[cache_key]
        
        config = self.get_config()
        
        # Extract database specific settings
        db_config = {
            'DB_PATH': config.get('DB_PATH', '/app/cache/playlista.db'),
            'DB_CACHE_DEFAULT_EXPIRY_HOURS': config.get('DB_CACHE_DEFAULT_EXPIRY_HOURS', 24),
            'DB_CACHE_CLEANUP_FREQUENCY_HOURS': config.get('DB_CACHE_CLEANUP_FREQUENCY_HOURS', 24),
            'DB_CACHE_MAX_SIZE_MB': config.get('DB_CACHE_MAX_SIZE_MB', 100),
            'DB_CLEANUP_RETENTION_DAYS': config.get('DB_CLEANUP_RETENTION_DAYS', 30),
            'DB_FAILED_ANALYSIS_RETENTION_DAYS': config.get('DB_FAILED_ANALYSIS_RETENTION_DAYS', 7),
            'DB_STATISTICS_RETENTION_DAYS': config.get('DB_STATISTICS_RETENTION_DAYS', 90),
            'DB_CONNECTION_TIMEOUT_SECONDS': config.get('DB_CONNECTION_TIMEOUT_SECONDS', 30),
            'DB_MAX_RETRY_ATTEMPTS': config.get('DB_MAX_RETRY_ATTEMPTS', 3),
            'DB_BATCH_SIZE': config.get('DB_BATCH_SIZE', 100),
            'DB_STATISTICS_COLLECTION_FREQUENCY_HOURS': config.get('DB_STATISTICS_COLLECTION_FREQUENCY_HOURS', 24),
            'DB_AUTO_CLEANUP_ENABLED': config.get('DB_AUTO_CLEANUP_ENABLED', True),
            'DB_AUTO_CLEANUP_FREQUENCY_HOURS': config.get('DB_AUTO_CLEANUP_FREQUENCY_HOURS', 168),
            'DB_BACKUP_ENABLED': config.get('DB_BACKUP_ENABLED', True),
            'DB_BACKUP_FREQUENCY_HOURS': config.get('DB_BACKUP_FREQUENCY_HOURS', 168),
            'DB_BACKUP_RETENTION_DAYS': config.get('DB_BACKUP_RETENTION_DAYS', 30),
            'DB_PERFORMANCE_MONITORING_ENABLED': config.get('DB_PERFORMANCE_MONITORING_ENABLED', True),
            'DB_QUERY_TIMEOUT_SECONDS': config.get('DB_QUERY_TIMEOUT_SECONDS', 60),
            'DB_MAX_CONNECTIONS': config.get('DB_MAX_CONNECTIONS', 10),
            'DB_WAL_MODE_ENABLED': config.get('DB_WAL_MODE_ENABLED', True),
            'DB_SYNCHRONOUS_MODE': config.get('DB_SYNCHRONOUS_MODE', 'NORMAL')
        }
        
        self._update_cache(cache_key, db_config)
        return db_config
    
    def get_external_api_config(self) -> Dict[str, Any]:
        """
        Get external API specific configuration.
        
        Returns:
            External API configuration dictionary
        """
        cache_key = self._get_cache_key('external_api')
        
        if self._is_cache_valid(cache_key):
            return self._config_cache[cache_key]
        
        config = self.get_config()
        
        # Extract external API specific settings
        api_config = {
            'MUSICBRAINZ_ENABLED': config.get('MUSICBRAINZ_ENABLED', True),
            'LASTFM_ENABLED': config.get('LASTFM_ENABLED', True),
            'DISCOGS_ENABLED': config.get('DISCOGS_ENABLED', False),
            'SPOTIFY_ENABLED': config.get('SPOTIFY_ENABLED', False),
            'LASTFM_API_KEY': config.get('LASTFM_API_KEY', ''),
            'DISCOGS_API_KEY': config.get('DISCOGS_API_KEY', ''),
            'DISCOGS_USER_TOKEN': config.get('DISCOGS_USER_TOKEN', ''),
            'SPOTIFY_CLIENT_ID': config.get('SPOTIFY_CLIENT_ID', ''),
            'SPOTIFY_CLIENT_SECRET': config.get('SPOTIFY_CLIENT_SECRET', ''),
            'MUSICBRAINZ_USER_AGENT': config.get('MUSICBRAINZ_USER_AGENT', 'Playlista/1.0'),
            'MUSICBRAINZ_RATE_LIMIT': config.get('MUSICBRAINZ_RATE_LIMIT', 1.0),
            'LASTFM_RATE_LIMIT': config.get('LASTFM_RATE_LIMIT', 2.0),
            'DISCOGS_RATE_LIMIT': config.get('DISCOGS_RATE_LIMIT', 1.0),
            'SPOTIFY_RATE_LIMIT': config.get('SPOTIFY_RATE_LIMIT', 1.0)
        }
        
        self._update_cache(cache_key, api_config)
        return api_config
    
    def get_logging_config(self) -> Dict[str, Any]:
        """
        Get logging specific configuration.
        
        Returns:
            Logging configuration dictionary
        """
        cache_key = self._get_cache_key('logging')
        
        if self._is_cache_valid(cache_key):
            return self._config_cache[cache_key]
        
        config = self.get_config()
        
        # Extract logging specific settings
        logging_config = {
            'LOG_LEVEL': config.get('LOG_LEVEL', 'INFO'),
            'LOG_FILE': config.get('LOG_FILE', '/app/logs/playlista.log'),
            'LOG_FORMAT': config.get('LOG_FORMAT', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
            'LOG_COLORED': config.get('LOG_COLORED', True),
            'LOG_BUFFERED': config.get('LOG_BUFFERED', True)
        }
        
        self._update_cache(cache_key, logging_config)
        return logging_config
    
    def get_resource_config(self) -> Dict[str, Any]:
        """
        Get resource management specific configuration.
        
        Returns:
            Resource configuration dictionary
        """
        cache_key = self._get_cache_key('resource')
        
        if self._is_cache_valid(cache_key):
            return self._config_cache[cache_key]
        
        config = self.get_config()
        
        # Extract resource management specific settings
        resource_config = {
            'MEMORY_LIMIT_GB': config.get('MEMORY_LIMIT_GB', 8.0),
            'CONTAINER_MEMORY_LIMIT_GB': config.get('CONTAINER_MEMORY_LIMIT_GB', None),
            'CPU_THRESHOLD_PERCENT': config.get('CPU_THRESHOLD_PERCENT', 80),
            'DISK_THRESHOLD_PERCENT': config.get('DISK_THRESHOLD_PERCENT', 90),
            'MONITORING_INTERVAL_SECONDS': config.get('MONITORING_INTERVAL_SECONDS', 5),
            'RESOURCE_HISTORY_SIZE': config.get('RESOURCE_HISTORY_SIZE', 100),
            'RESOURCE_ALERT_THRESHOLD_PERCENT': config.get('RESOURCE_ALERT_THRESHOLD_PERCENT', 85),
            'RESOURCE_AUTO_CLEANUP_ENABLED': config.get('RESOURCE_AUTO_CLEANUP_ENABLED', True),
            'RESOURCE_CALLBACK_ENABLED': config.get('RESOURCE_CALLBACK_ENABLED', True),
            'RESOURCE_PERFORMANCE_MONITORING': config.get('RESOURCE_PERFORMANCE_MONITORING', True),
            'RESOURCE_MEMORY_LIMIT_GB': config.get('RESOURCE_MEMORY_LIMIT_GB', 8.0),
            'RESOURCE_CPU_LIMIT_PERCENT': config.get('RESOURCE_CPU_LIMIT_PERCENT', 80),
            'RESOURCE_LOG_LEVEL': config.get('RESOURCE_LOG_LEVEL', 'INFO'),
            'RESOURCE_MONITORING_ENABLED': config.get('RESOURCE_MONITORING_ENABLED', True)
        }
        
        self._update_cache(cache_key, resource_config)
        return resource_config
    
    def get_playlist_config(self) -> Dict[str, Any]:
        """
        Get playlist generation specific configuration.
        
        Returns:
            Playlist configuration dictionary
        """
        cache_key = self._get_cache_key('playlist')
        
        if self._is_cache_valid(cache_key):
            return self._config_cache[cache_key]
        
        config = self.get_config()
        
        # Extract playlist generation specific settings
        playlist_config = {
            'PLAYLIST_MAX_TRACKS': config.get('PLAYLIST_MAX_TRACKS', 50),
            'PLAYLIST_MIN_TRACKS': config.get('PLAYLIST_MIN_TRACKS', 10),
            'PLAYLIST_TARGET_DURATION_MINUTES': config.get('PLAYLIST_TARGET_DURATION_MINUTES', 60),
            'PLAYLIST_SIMILARITY_THRESHOLD': config.get('PLAYLIST_SIMILARITY_THRESHOLD', 0.7),
            'PLAYLIST_DIVERSITY_FACTOR': config.get('PLAYLIST_DIVERSITY_FACTOR', 0.3),
            'PLAYLIST_ENERGY_VARIATION': config.get('PLAYLIST_ENERGY_VARIATION', 0.2),
            'PLAYLIST_TEMPO_VARIATION': config.get('PLAYLIST_TEMPO_VARIATION', 0.15),
            'PLAYLIST_KEY_COMPATIBILITY': config.get('PLAYLIST_KEY_COMPATIBILITY', True),
            'PLAYLIST_GENRE_MIXING': config.get('PLAYLIST_GENRE_MIXING', True),
            'PLAYLIST_ARTIST_VARIETY': config.get('PLAYLIST_ARTIST_VARIETY', True),
            'PLAYLIST_YEAR_RANGE': config.get('PLAYLIST_YEAR_RANGE', 10),
            'PLAYLIST_ENERGY_PROGRESSION': config.get('PLAYLIST_ENERGY_PROGRESSION', True),
            'PLAYLIST_TEMPO_PROGRESSION': config.get('PLAYLIST_TEMPO_PROGRESSION', True),
            'PLAYLIST_KEY_PROGRESSION': config.get('PLAYLIST_KEY_PROGRESSION', True),
            'PLAYLIST_QUALITY_THRESHOLD': config.get('PLAYLIST_QUALITY_THRESHOLD', 0.6),
            'PLAYLIST_CACHE_ENABLED': config.get('PLAYLIST_CACHE_ENABLED', True),
            'PLAYLIST_CACHE_EXPIRY_HOURS': config.get('PLAYLIST_CACHE_EXPIRY_HOURS', 24)
        }
        
        self._update_cache(cache_key, playlist_config)
        return playlist_config
    
    def clear_cache(self):
        """Clear all configuration caches."""
        self._config_cache.clear()
        self._cache_timestamps.clear()
        log_universal('INFO', 'Config', 'Configuration cache cleared')
    
    def reload_config(self) -> Dict[str, Any]:
        """
        Force reload configuration from all sources.
        
        Returns:
            Reloaded configuration dictionary
        """
        self.clear_cache()
        return self.get_config()


# Global configuration loader instance
config_loader = ConfigLoader() 
