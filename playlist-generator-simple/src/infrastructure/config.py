"""
Configuration management for Playlist Generator.
Handles loading, validation, and type conversion of configuration.
"""

import os
import json
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from pathlib import Path


@dataclass
class DatabaseConfig:
    """Database configuration."""
    path: str
    timeout: int = 30
    max_connections: int = 10
    
    def __post_init__(self):
        if self.timeout <= 0:
            raise ValueError("Database timeout must be positive")
        if self.max_connections <= 0:
            raise ValueError("Max connections must be positive")


@dataclass
class AudioAnalysisConfig:
    """Audio analysis configuration."""
    sample_rate: int = 44100
    hop_size: int = 512
    frame_size: int = 2048
    extract_musicnn: bool = True
    extract_rhythm: bool = True
    extract_spectral: bool = True
    extract_loudness: bool = True
    extract_key: bool = True
    extract_mfcc: bool = True
    
    def __post_init__(self):
        if self.sample_rate <= 0:
            raise ValueError("Sample rate must be positive")
        if self.hop_size <= 0:
            raise ValueError("Hop size must be positive")
        if self.frame_size <= 0:
            raise ValueError("Frame size must be positive")


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    console_enabled: bool = True
    file_enabled: bool = True
    file_path: str = "logs/playlista.log"
    max_file_size_mb: int = 50
    max_files: int = 10
    
    def __post_init__(self):
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.level not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")


@dataclass
class ExternalAPIConfig:
    """External API configuration."""
    musicbrainz_enabled: bool = True
    musicbrainz_user_agent: str = "playlista/1.0"
    lastfm_enabled: bool = True
    lastfm_api_key: Optional[str] = None
    rate_limit_requests_per_minute: int = 60
    
    def __post_init__(self):
        if self.rate_limit_requests_per_minute <= 0:
            raise ValueError("Rate limit must be positive")


@dataclass
class AppConfig:
    """Main application configuration."""
    database: DatabaseConfig
    audio_analysis: AudioAnalysisConfig
    logging: LoggingConfig
    external_api: ExternalAPIConfig
    music_path: str = "/music"
    cache_path: str = "/app/cache"
    playlists_path: str = "/app/playlists"
    
    def __post_init__(self):
        if not self.music_path:
            raise ValueError("Music path is required")


class ConfigurationService:
    """Service for managing application configuration."""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path or self._find_config_file()
        self._config: Optional[AppConfig] = None
    
    def get_config(self) -> AppConfig:
        """Get application configuration."""
        if self._config is None:
            self._config = self._load_config()
        return self._config
    
    def reload_config(self) -> AppConfig:
        """Reload configuration from file."""
        self._config = self._load_config()
        return self._config
    
    def _find_config_file(self) -> str:
        """Find configuration file."""
        possible_paths = [
            "playlista.conf",
            "config/playlista.conf",
            "/app/playlista.conf",
            os.path.expanduser("~/.playlista.conf")
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        # Return default path if none found
        return "playlista.conf"
    
    def _load_config(self) -> AppConfig:
        """Load configuration from file."""
        # Load environment variables
        env_config = self._load_environment_config()
        
        # Load file configuration
        file_config = self._load_file_config()
        
        # Merge configurations (env overrides file)
        merged_config = {**file_config, **env_config}
        
        # Create configuration objects
        return self._create_config_objects(merged_config)
    
    def _load_environment_config(self) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        config = {}
        
        # Database
        if os.getenv("DB_PATH"):
            config["database_path"] = os.getenv("DB_PATH")
        if os.getenv("DB_TIMEOUT"):
            config["database_timeout"] = int(os.getenv("DB_TIMEOUT"))
        
        # Audio analysis
        if os.getenv("AUDIO_SAMPLE_RATE"):
            config["audio_sample_rate"] = int(os.getenv("AUDIO_SAMPLE_RATE"))
        if os.getenv("AUDIO_HOP_SIZE"):
            config["audio_hop_size"] = int(os.getenv("AUDIO_HOP_SIZE"))
        
        # Logging
        if os.getenv("LOG_LEVEL"):
            config["log_level"] = os.getenv("LOG_LEVEL")
        if os.getenv("LOG_CONSOLE_ENABLED"):
            config["log_console_enabled"] = os.getenv("LOG_CONSOLE_ENABLED").lower() == "true"
        
        # External APIs
        if os.getenv("LASTFM_API_KEY"):
            config["lastfm_api_key"] = os.getenv("LASTFM_API_KEY")
        if os.getenv("MUSICBRAINZ_USER_AGENT"):
            config["musicbrainz_user_agent"] = os.getenv("MUSICBRAINZ_USER_AGENT")
        
        # Paths
        if os.getenv("MUSIC_PATH"):
            config["music_path"] = os.getenv("MUSIC_PATH")
        if os.getenv("CACHE_PATH"):
            config["cache_path"] = os.getenv("CACHE_PATH")
        
        return config
    
    def _load_file_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        if not os.path.exists(self.config_path):
            return {}
        
        config = {}
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
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
                        
                        # Convert value type
                        config[key] = self._convert_value(value)
        
        except Exception as e:
            raise ValueError(f"Error loading config from {self.config_path}: {e}")
        
        return config
    
    def _convert_value(self, value: str) -> Any:
        """Convert string value to appropriate type."""
        # Boolean
        if value.lower() in ['true', 'yes', '1']:
            return True
        if value.lower() in ['false', 'no', '0']:
            return False
        
        # Integer
        try:
            return int(value)
        except ValueError:
            pass
        
        # Float
        try:
            return float(value)
        except ValueError:
            pass
        
        # String (default)
        return value
    
    def _create_config_objects(self, config: Dict[str, Any]) -> AppConfig:
        """Create configuration objects from dictionary."""
        # Database config
        db_config = DatabaseConfig(
            path=config.get("database_path", "/app/cache/playlista.db"),
            timeout=config.get("database_timeout", 30),
            max_connections=config.get("database_max_connections", 10)
        )
        
        # Audio analysis config
        audio_config = AudioAnalysisConfig(
            sample_rate=config.get("audio_sample_rate", 44100),
            hop_size=config.get("audio_hop_size", 512),
            frame_size=config.get("audio_frame_size", 2048),
            extract_musicnn=config.get("audio_extract_musicnn", True),
            extract_rhythm=config.get("audio_extract_rhythm", True),
            extract_spectral=config.get("audio_extract_spectral", True),
            extract_loudness=config.get("audio_extract_loudness", True),
            extract_key=config.get("audio_extract_key", True),
            extract_mfcc=config.get("audio_extract_mfcc", True)
        )
        
        # Logging config
        log_config = LoggingConfig(
            level=config.get("log_level", "INFO"),
            console_enabled=config.get("log_console_enabled", True),
            file_enabled=config.get("log_file_enabled", True),
            file_path=config.get("log_file_path", "logs/playlista.log"),
            max_file_size_mb=config.get("log_max_file_size_mb", 50),
            max_files=config.get("log_max_files", 10)
        )
        
        # External API config
        api_config = ExternalAPIConfig(
            musicbrainz_enabled=config.get("musicbrainz_enabled", True),
            musicbrainz_user_agent=config.get("musicbrainz_user_agent", "playlista/1.0"),
            lastfm_enabled=config.get("lastfm_enabled", True),
            lastfm_api_key=config.get("lastfm_api_key"),
            rate_limit_requests_per_minute=config.get("api_rate_limit", 60)
        )
        
        return AppConfig(
            database=db_config,
            audio_analysis=audio_config,
            logging=log_config,
            external_api=api_config,
            music_path=config.get("music_path", "/music"),
            cache_path=config.get("cache_path", "/app/cache"),
            playlists_path=config.get("playlists_path", "/app/playlists")
        ) 