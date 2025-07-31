"""
Configuration loader for Playlist Generator Simple.
Reads from plain text configuration file and supports environment variable overrides.
"""

import os
import logging
from typing import Dict, Any, Optional
from pathlib import Path

# Import universal logging
from .logging_setup import get_logger, log_universal

logger = get_logger(__name__)


class ConfigLoader:
    """
    Loads configuration from plain text file with environment variable support.
    """
    
    def __init__(self, config_file: str = "playlista.conf"):
        """
        Initialize configuration loader.
        
        Args:
            config_file: Path to configuration file
        """
        self.config_file = config_file
        self.config = {}
        
    def load_config(self) -> Dict[str, Any]:
        """
        Load configuration from file and environment variables.
        
        Returns:
            Dictionary with configuration settings
        """
        log_universal('INFO', 'Config', f'Loading configuration from: {self.config_file}')
        
        # Load from file
        file_config = self._load_from_file()
        
        # Override with environment variables
        env_config = self._load_from_environment()
        
        # Merge configurations (env vars override file)
        self.config = {**file_config, **env_config}
        
        # Validate configuration
        validation_result = self._validate_configuration()
        if not validation_result['valid']:
            log_universal('WARNING', 'Config', 'Configuration validation warnings:')
            for warning in validation_result['warnings']:
                log_universal('WARNING', 'Config', f'  {warning}')
        
        log_universal('INFO', 'Config', 'Configuration loaded successfully')
        
        return self.config
    
    def _load_from_file(self) -> Dict[str, Any]:
        """
        Load configuration from plain text file.
        
        Returns:
            Dictionary with configuration from file
        """
        config = {}
        
        if not os.path.exists(self.config_file):
            log_universal('WARNING', 'Config', f'Configuration file not found: {self.config_file}')
            return config
        
        try:
            with open(self.config_file, 'r') as f:
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
                        
                        # Remove inline comments
                        if '#' in value:
                            value = value.split('#')[0].strip()
                        
                        # Convert value types
                        config[key] = self._convert_value(value)
        
        except Exception as e:
            log_universal('ERROR', 'Config', f'Error reading configuration file: {e}')
        
        return config
    
    def _load_from_environment(self) -> Dict[str, Any]:
        """
        Load configuration from environment variables.
        
        Returns:
            Dictionary with configuration from environment
        """
        config = {}
        
        # Define environment variable mappings
        env_mappings = {
            'LOG_LEVEL': 'LOG_LEVEL',
            'LOG_CONSOLE_ENABLED': 'LOG_CONSOLE_ENABLED',
            'LOG_FILE_ENABLED': 'LOG_FILE_ENABLED',
            'LOG_COLORED_OUTPUT': 'LOG_COLORED_OUTPUT',
            'LOG_FILE_COLORED_OUTPUT': 'LOG_FILE_COLORED_OUTPUT',
            'LOG_FILE_PREFIX': 'LOG_FILE_PREFIX',
            'LOG_FILE_SIZE_MB': 'LOG_FILE_SIZE_MB',
            'LOG_MAX_FILES': 'LOG_MAX_FILES',
            'LOG_FILE_ENCODING': 'LOG_FILE_ENCODING',
            'LOG_CONSOLE_FORMAT': 'LOG_CONSOLE_FORMAT',
            'LOG_CONSOLE_DATE_FORMAT': 'LOG_CONSOLE_DATE_FORMAT',
            'LOG_FILE_FORMAT': 'LOG_FILE_FORMAT',
            'LOG_FILE_INCLUDE_EXTRA_FIELDS': 'LOG_FILE_INCLUDE_EXTRA_FIELDS',
            'LOG_FILE_INCLUDE_EXCEPTION_DETAILS': 'LOG_FILE_INCLUDE_EXCEPTION_DETAILS',
    
            'LOG_FUNCTION_CALLS_ENABLED': 'LOG_FUNCTION_CALLS_ENABLED',
            'LOG_ENVIRONMENT_MONITORING': 'LOG_ENVIRONMENT_MONITORING',
            'LOG_SIGNAL_HANDLING_ENABLED': 'LOG_SIGNAL_HANDLING_ENABLED',
            'LOG_SIGNAL_CYCLE_LEVELS': 'LOG_SIGNAL_CYCLE_LEVELS',
            'MIN_FILE_SIZE_BYTES': 'MIN_FILE_SIZE_BYTES',
            'VALID_EXTENSIONS': 'VALID_EXTENSIONS',
            'HASH_ALGORITHM': 'HASH_ALGORITHM',
            'MAX_RETRY_COUNT': 'MAX_RETRY_COUNT',
            'ENABLE_RECURSIVE_SCAN': 'ENABLE_RECURSIVE_SCAN',
            'EXCLUDE_PATTERNS': 'EXCLUDE_PATTERNS',
            'INCLUDE_PATTERNS': 'INCLUDE_PATTERNS',
            'ENABLE_DETAILED_LOGGING': 'ENABLE_DETAILED_LOGGING',
            # External API settings
            'LASTFM_API_KEY': 'LASTFM_API_KEY',
            'MUSICBRAINZ_USER_AGENT': 'MUSICBRAINZ_USER_AGENT',
            'MUSICBRAINZ_RATE_LIMIT': 'MUSICBRAINZ_RATE_LIMIT',
            'LASTFM_RATE_LIMIT': 'LASTFM_RATE_LIMIT',
            # Database settings
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
                config[config_key] = self._convert_value(value)
                log_universal('DEBUG', 'Config', f"Loaded from environment: {config_key} = {config[config_key]}")
        
        return config
    
    def _convert_value(self, value: str) -> Any:
        """
        Convert string value to appropriate type.
        
        Args:
            value: String value to convert
            
        Returns:
            Converted value
        """
        # Handle environment variable substitution
        if value.startswith('${') and value.endswith('}'):
            env_var = value[2:-1]  # Remove ${ and }
            env_value = os.getenv(env_var)
            if env_value is not None:
                value = env_value
            else:
                log_universal('WARNING', 'Config', f'Environment variable {env_var} not found, using literal value')
        
        # Handle boolean values
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # Handle integer values
        try:
            return int(value)
        except ValueError:
            pass
        
        # Handle float values
        try:
            return float(value)
        except ValueError:
            pass
        
        # Handle list values (comma-separated)
        if ',' in value:
            return [item.strip() for item in value.split(',') if item.strip()]
        
        # Return as string
        return value
    
    def get_file_discovery_config(self) -> Dict[str, Any]:
        """
        Get file discovery specific configuration.
        
        Returns:
            Dictionary with file discovery settings
        """
        if not self.config:
            self.load_config()
        
        # Extract file discovery related settings
        file_discovery_config = {}
        
        file_discovery_keys = [
            'MIN_FILE_SIZE_BYTES',
            'VALID_EXTENSIONS',
            'HASH_ALGORITHM',
            'MAX_RETRY_COUNT',
            'ENABLE_RECURSIVE_SCAN',
            'EXCLUDE_PATTERNS',
            'INCLUDE_PATTERNS',
            'LOG_LEVEL',
            'ENABLE_DETAILED_LOGGING'
        ]
        
        for key in file_discovery_keys:
            if key in self.config:
                file_discovery_config[key] = self.config[key]
        
        return file_discovery_config
    
    def get_logging_config(self) -> Dict[str, Any]:
        """
        Get logging specific configuration.
        
        Returns:
            Dictionary with logging settings
        """
        if not self.config:
            self.load_config()
        
        # Extract logging related settings
        logging_config = {}
        
        logging_keys = [
            'LOG_LEVEL',
            'LOG_CONSOLE_ENABLED',
            'LOG_FILE_ENABLED',
            'LOG_COLORED_OUTPUT',
            'LOG_FILE_COLORED_OUTPUT',
            'LOG_FILE_PREFIX',
            'LOG_FILE_SIZE_MB',
            'LOG_MAX_FILES',
            'LOG_FILE_ENCODING',
            'LOG_CONSOLE_FORMAT',
            'LOG_CONSOLE_DATE_FORMAT',
            'LOG_FILE_FORMAT',
            'LOG_FILE_INCLUDE_EXTRA_FIELDS',
            'LOG_FILE_INCLUDE_EXCEPTION_DETAILS',
    
            'LOG_FUNCTION_CALLS_ENABLED',
            'LOG_ENVIRONMENT_MONITORING',
            'LOG_SIGNAL_HANDLING_ENABLED',
            'LOG_SIGNAL_CYCLE_LEVELS'
        ]
        
        for key in logging_keys:
            if key in self.config:
                logging_config[key] = self.config[key]
        
        return logging_config
    
    def get_database_config(self) -> Dict[str, Any]:
        """
        Get database specific configuration.
        
        Returns:
            Dictionary with database settings
        """
        if not self.config:
            self.load_config()
        
        # Extract database related settings
        database_config = {}
        
        database_keys = [
            'DB_CACHE_DEFAULT_EXPIRY_HOURS',
            'DB_CACHE_CLEANUP_FREQUENCY_HOURS',
            'DB_CACHE_MAX_SIZE_MB',
            'DB_CLEANUP_RETENTION_DAYS',
            'DB_FAILED_ANALYSIS_RETENTION_DAYS',
            'DB_STATISTICS_RETENTION_DAYS',
            'DB_CONNECTION_TIMEOUT_SECONDS',
            'DB_MAX_RETRY_ATTEMPTS',
            'DB_BATCH_SIZE',
            'DB_STATISTICS_COLLECTION_FREQUENCY_HOURS',
            'DB_AUTO_CLEANUP_ENABLED',
            'DB_AUTO_CLEANUP_FREQUENCY_HOURS',
            'DB_BACKUP_ENABLED',
            'DB_BACKUP_FREQUENCY_HOURS',
            'DB_BACKUP_RETENTION_DAYS',
            'DB_PERFORMANCE_MONITORING_ENABLED',
            'DB_QUERY_TIMEOUT_SECONDS',
            'DB_MAX_CONNECTIONS',
            'DB_WAL_MODE_ENABLED',
            'DB_SYNCHRONOUS_MODE'
        ]
        
        for key in database_keys:
            if key in self.config:
                database_config[key] = self.config[key]
        
        return database_config

    def get_analysis_config(self) -> Dict[str, Any]:
        """
        Get analysis specific configuration.
        
        Returns:
            Dictionary with analysis settings
        """
        if not self.config:
            self.load_config()
        
        # Extract analysis related settings
        analysis_config = {}
        
        analysis_keys = [
            'MUSIC_PATH',
            'BIG_FILE_SIZE_MB',
            'ANALYSIS_TIMEOUT_SECONDS',
            'MEMORY_THRESHOLD_PERCENT',
            'SEQUENTIAL_TIMEOUT_SECONDS',
            'PARALLEL_TIMEOUT_SECONDS',
            'AUDIO_SAMPLE_RATE',
            'AUDIO_HOP_SIZE',
            'AUDIO_FRAME_SIZE',
            'FORCE_REEXTRACT',
            'INCLUDE_FAILED_FILES',
            'MAX_WORKERS',
            'WORKER_TIMEOUT_SECONDS',
            'ANALYSIS_CACHE_ENABLED',
            'ANALYSIS_CACHE_EXPIRY_HOURS',
            'ANALYSIS_RETRY_ATTEMPTS',
            'ANALYSIS_RETRY_DELAY_SECONDS',
            'ANALYSIS_PROGRESS_REPORTING',
            'ANALYSIS_STATISTICS_COLLECTION',
            'ANALYSIS_CLEANUP_ENABLED',
            'EXTRACT_MUSICNN',
            'MUSICNN_MODEL_PATH',
            'MUSICNN_JSON_PATH',
            'MUSICNN_TIMEOUT_SECONDS',
            'MAX_FULL_ANALYSIS_SIZE_MB',
            'MIN_FULL_ANALYSIS_SIZE_MB',
            'MIN_MEMORY_FOR_FULL_ANALYSIS_GB',
            'MEMORY_BUFFER_GB',
            'MAX_CPU_FOR_FULL_ANALYSIS_PERCENT',
            'CPU_CHECK_INTERVAL_SECONDS',
            'PARALLEL_MAX_FILE_SIZE_MB',
            'PARALLEL_MIN_MEMORY_GB',
            'PARALLEL_MAX_CPU_PERCENT',
            'SEQUENTIAL_MAX_FILE_SIZE_MB',
            'SEQUENTIAL_MIN_MEMORY_GB',
            'SEQUENTIAL_MAX_CPU_PERCENT',
            'SMART_ANALYSIS_ENABLED',
            'ANALYSIS_TYPE_FALLBACK',
            'RESOURCE_MONITORING_ENABLED'
        ]
        
        for key in analysis_keys:
            if key in self.config:
                analysis_config[key] = self.config[key]
        
        return analysis_config

    def get_resource_config(self) -> Dict[str, Any]:
        """
        Get resource management specific configuration.
        
        Returns:
            Dictionary with resource settings
        """
        if not self.config:
            self.load_config()
        
        # Extract resource related settings
        resource_config = {}
        
        resource_keys = [
            'MEMORY_LIMIT_GB',
            'CPU_THRESHOLD_PERCENT',
            'DISK_THRESHOLD_PERCENT',
            'MONITORING_INTERVAL_SECONDS',
            'MEMORY_PER_WORKER_GB',
            'MAX_WORKERS',
            'WORKER_TIMEOUT_SECONDS',
            'RESOURCE_MONITORING_ENABLED',
            'MEMORY_CLEANUP_ENABLED',
            'CPU_THROTTLING_ENABLED',
            'DISK_CLEANUP_ENABLED',
            'RESOURCE_HISTORY_SIZE',
            'RESOURCE_STATISTICS_COLLECTION',
            'RESOURCE_ALERT_THRESHOLD_PERCENT',
            'RESOURCE_AUTO_CLEANUP_ENABLED',
            'RESOURCE_CALLBACK_ENABLED',
            'RESOURCE_LOG_LEVEL',
            'RESOURCE_PERFORMANCE_MONITORING',
            'RESOURCE_MEMORY_LIMIT_GB',
            'RESOURCE_CPU_LIMIT_PERCENT'
        ]
        
        for key in resource_keys:
            if key in self.config:
                resource_config[key] = self.config[key]
        
        return resource_config
    
    def get_playlist_config(self) -> Dict[str, Any]:
        """
        Get playlist generation specific configuration.
        
        Returns:
            Dictionary with playlist settings
        """
        if not self.config:
            self.load_config()
        
        # Extract playlist related settings
        playlist_config = {}
        
        playlist_keys = [
            'DEFAULT_PLAYLIST_SIZE',
            'MIN_TRACKS_PER_GENRE',
            'MAX_PLAYLISTS',
            'SIMILARITY_THRESHOLD',
            'DIVERSITY_THRESHOLD',
            'PLAYLIST_OUTPUT_DIR',
            'PLAYLIST_GENERATION_ENABLED',
            'PLAYLIST_SAVE_METADATA',
            'PLAYLIST_OPTIMIZATION_ENABLED',
            'PLAYLIST_DEFAULT_METHOD',
            'PLAYLIST_KMEANS_CLUSTERS',
            'PLAYLIST_SIMILARITY_ALGORITHM',
            'PLAYLIST_TIME_SLOTS',
            'PLAYLIST_TAG_PRIORITY',
            'PLAYLIST_CACHE_ENABLED',
            'PLAYLIST_FEATURE_GROUPS',
            'PLAYLIST_MIXED_WEIGHTS'
        ]
        
        for key in playlist_keys:
            if key in self.config:
                playlist_config[key] = self.config[key]
        
        return playlist_config
    
    def get_external_api_config(self) -> Dict[str, Any]:
        """
        Get external API specific configuration.
        
        Returns:
            Dictionary with external API settings
        """
        if not self.config:
            self.load_config()
        
        # Extract external API related settings
        external_api_config = {}
        
        external_api_keys = [
            'EXTERNAL_API_ENABLED',
            'MUSICBRAINZ_ENABLED',
            'LASTFM_ENABLED',
            'MUSICBRAINZ_USER_AGENT',
            'MUSICBRAINZ_RATE_LIMIT',
            'LASTFM_API_KEY',
            'LASTFM_RATE_LIMIT',
            'METADATA_ENRICHMENT_ENABLED',
            'METADATA_ENRICHMENT_TIMEOUT',
            'METADATA_ENRICHMENT_MAX_TAGS',
            'METADATA_ENRICHMENT_RETRY_COUNT'
        ]
        
        for key in external_api_keys:
            if key in self.config:
                external_api_config[key] = self.config[key]
        
        return external_api_config
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        if not self.config:
            self.load_config()
        
        return self.config.get(key, default)
    
    def _validate_configuration(self) -> Dict[str, Any]:
        """
        Validate configuration settings.
        
        Returns:
            Dictionary with validation result and warnings
        """
        warnings = []
        valid = True
        
        # Required fields
        required_fields = {
            'MUSIC_PATH': 'Music directory path',
            'DB_PATH': 'Database file path',
            'LOG_LEVEL': 'Logging level'
        }
        
        for field, description in required_fields.items():
            if field not in self.config:
                warnings.append(f"Missing required field: {field} ({description})")
                valid = False
        
        # Validate numeric fields
        numeric_fields = {
            'ANALYSIS_TIMEOUT_SECONDS': (1, 3600),
            'ANALYSIS_RETRY_ATTEMPTS': (0, 10),
            'DB_CACHE_MAX_SIZE_MB': (1, 1000),
            'LOG_FILE_SIZE_MB': (1, 100)
        }
        
        for field, (min_val, max_val) in numeric_fields.items():
            if field in self.config:
                try:
                    value = float(self.config[field])
                    if value < min_val or value > max_val:
                        warnings.append(f"{field} value {value} is outside valid range [{min_val}, {max_val}]")
                        valid = False
                except (ValueError, TypeError):
                    warnings.append(f"{field} value '{self.config[field]}' is not a valid number")
                    valid = False
        
        # Validate boolean fields
        boolean_fields = [
            'EXTERNAL_API_ENABLED',
            'MUSICBRAINZ_ENABLED',
            'LASTFM_ENABLED',
            'METADATA_ENRICHMENT_ENABLED',
            'LOG_CONSOLE_ENABLED',
            'LOG_FILE_ENABLED',
            'ANALYSIS_CACHE_ENABLED'
        ]
        
        for field in boolean_fields:
            if field in self.config:
                value = self.config[field]
                if not isinstance(value, bool):
                    warnings.append(f"{field} should be boolean, got {type(value).__name__}")
                    valid = False
        
        # Validate log level
        if 'LOG_LEVEL' in self.config:
            valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
            if self.config['LOG_LEVEL'] not in valid_levels:
                warnings.append(f"LOG_LEVEL '{self.config['LOG_LEVEL']}' is not valid. Use one of: {valid_levels}")
                valid = False
        
        # Validate file paths exist (if not Docker paths)
        path_fields = ['MUSIC_PATH', 'DB_PATH']
        for field in path_fields:
            if field in self.config:
                path = self.config[field]
                if not path.startswith('/app/') and not path.startswith('/music') and not path.startswith('/root/'):
                    # Only check non-Docker paths
                    if not os.path.exists(os.path.dirname(path)):
                        warnings.append(f"Directory for {field} does not exist: {os.path.dirname(path)}")
        
        return {
            'valid': valid,
            'warnings': warnings
        }


# Global config loader instance
config_loader = ConfigLoader() 
