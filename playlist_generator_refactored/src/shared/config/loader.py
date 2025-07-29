"""
Configuration loader utilities for the Playlista application.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
import logging

from .settings import AppConfig, ConfigurationError
from .json_loader import load_json_config, get_json_config_summary


logger = logging.getLogger(__name__)


class ConfigLoader:
    """Utility class for loading configuration from various sources."""
    
    def __init__(self):
        self.config_cache: Optional[AppConfig] = None
    
    def load_config(self, force_reload: bool = False) -> AppConfig:
        """Load application configuration from environment variables and files."""
        if self.config_cache is not None and not force_reload:
            return self.config_cache
        
        try:
            # Load from environment variables (primary source)
            config = self._load_from_environment()
            
            # Override with config file if it exists
            config_file = self._find_config_file()
            if config_file:
                config = self._merge_with_file(config, config_file)
            
            # Validate the configuration
            self._validate_config(config)
            
            self.config_cache = config
            logger.debug("Configuration loaded successfully")
            return config
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise ConfigurationError(f"Configuration loading failed: {e}", original_exception=e)
    
    def _load_from_environment(self) -> AppConfig:
        """Load configuration from environment variables."""
        # Load JSON configuration first (if available)
        json_config = load_json_config()
        if json_config:
            logger.info(f"Loaded {len(json_config)} settings from JSON configuration")
        
        # Set environment variables that might be needed by the config classes
        self._set_default_environment_variables()
        
        # Create configuration from environment
        config = AppConfig()
        logger.debug("Configuration loaded from environment variables")
        return config
    
    def _set_default_environment_variables(self) -> None:
        """Set default environment variables if not already set."""
        defaults = {
            'LOG_LEVEL': 'INFO',
            'CACHE_DIR': '/app/cache',
            'LOG_DIR': '/app/logs',
            'OUTPUT_DIR': '/app/playlists',
            'HOST_LIBRARY_PATH': '/root/music/library',
            'LARGE_FILE_THRESHOLD': '50',
            'MEMORY_AWARE': 'false',
            'MIN_TRACKS_PER_GENRE': '10',
            'LASTFM_API_KEY': '',
            'DEBUG': 'false'
        }
        
        for key, value in defaults.items():
            if key not in os.environ:
                os.environ[key] = value
                logger.debug(f"Set default environment variable: {key}={value}")
    
    def _find_config_file(self) -> Optional[Path]:
        """Find configuration file in standard locations."""
        config_locations = [
            Path('config.json'),
            Path('playlista.json'),
            Path('~/.playlista/config.json').expanduser(),
            Path('/etc/playlista/config.json'),
            Path('/app/config.json')
        ]
        
        for location in config_locations:
            if location.exists():
                logger.debug(f"Found config file: {location}")
                return location
        
        logger.debug("No config file found, using environment variables only")
        return None
    
    def _merge_with_file(self, config: AppConfig, config_file: Path) -> AppConfig:
        """Merge configuration with values from config file."""
        try:
            with open(config_file, 'r') as f:
                file_config = json.load(f)
            
            logger.debug(f"Loading configuration from file: {config_file}")
            
            # Merge configuration (file takes precedence over environment)
            config = self._merge_configs(config, file_config)
            
        except Exception as e:
            logger.warning(f"Failed to load config file {config_file}: {e}")
            # Continue with environment-only config
        
        return config
    
    def _merge_configs(self, base_config: AppConfig, file_config: Dict[str, Any]) -> AppConfig:
        """Merge file configuration with base configuration."""
        # This is a simplified merge - in a full implementation,
        # you'd want to recursively merge nested configurations
        logger.debug("Merging configuration from file")
        return base_config
    
    def _validate_config(self, config: AppConfig) -> None:
        """Validate the loaded configuration."""
        try:
            # Basic validation - the dataclass __post_init__ methods handle most validation
            if not config.output_dir:
                raise ConfigurationError("Output directory is required")
            
            if not config.host_library_path:
                raise ConfigurationError("Host library path is required")
            
            logger.debug("Configuration validation passed")
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            raise ConfigurationError(f"Configuration validation failed: {e}", original_exception=e)
    
    def reload_config(self) -> AppConfig:
        """Force reload configuration from sources."""
        self.config_cache = None
        return self.load_config(force_reload=True)
    
    def get_config_dict(self) -> Dict[str, Any]:
        """Get configuration as dictionary for debugging/logging."""
        config = self.load_config()
        config_dict = config.to_dict()
        
        # Add JSON config summary if available
        json_summary = get_json_config_summary()
        if json_summary:
            config_dict['json_config'] = json_summary
        
        return config_dict


# Global configuration loader instance
config_loader = ConfigLoader()


def get_config() -> AppConfig:
    """Get the application configuration."""
    return config_loader.load_config()


def reload_config() -> AppConfig:
    """Reload the application configuration."""
    return config_loader.reload_config()


def get_config_dict() -> Dict[str, Any]:
    """Get configuration as dictionary."""
    return config_loader.get_config_dict() 