"""
JSON Configuration Loader for Playlista

This module loads configuration from a JSON file and converts it to environment variables
that can be used by the existing settings system.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ConfigValue:
    """Represents a configuration value with metadata."""
    value: Any
    description: str
    type: str
    default: Any
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    options: Optional[list] = None


class JSONConfigLoader:
    """Loads configuration from JSON files and converts to environment variables."""
    
    def __init__(self, config_file_path: Optional[str] = None):
        """Initialize the JSON config loader.
        
        Args:
            config_file_path: Path to the JSON configuration file
        """
        self.config_file_path = config_file_path or "/app/config/playlista_config.json"
        self.config_cache: Optional[Dict[str, Any]] = None
    
    def load_config(self, force_reload: bool = False) -> Dict[str, Any]:
        """Load configuration from JSON file.
        
        Args:
            force_reload: Force reload from file even if cached
            
        Returns:
            Dictionary of configuration values
        """
        if self.config_cache is not None and not force_reload:
            return self.config_cache
        
        try:
            config_path = Path(self.config_file_path)
            if not config_path.exists():
                logger.debug(f"Config file not found: {config_path}")
                return {}
            
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            # Extract settings from the nested structure
            settings = self._extract_settings(config_data)
            
            # Convert to environment variables
            self._set_environment_variables(settings)
            
            self.config_cache = settings
            logger.info(f"Loaded configuration from {config_path}")
            return settings
            
        except Exception as e:
            logger.error(f"Failed to load JSON config: {e}")
            return {}
    
    def _extract_settings(self, config_data: Dict[str, Any]) -> Dict[str, ConfigValue]:
        """Extract settings from the nested JSON structure.
        
        Args:
            config_data: The loaded JSON configuration data
            
        Returns:
            Dictionary of setting names to ConfigValue objects
        """
        settings = {}
        
        if 'playlista_config' not in config_data:
            logger.warning("No 'playlista_config' section found in JSON")
            return settings
        
        playlista_config = config_data['playlista_config']
        
        # Iterate through all sections
        for section_name, section_data in playlista_config.items():
            if section_name == 'description' or section_name == 'version' or section_name == 'last_updated':
                continue
                
            if 'settings' not in section_data:
                continue
                
            section_settings = section_data['settings']
            
            # Extract each setting
            for setting_name, setting_data in section_settings.items():
                if not isinstance(setting_data, dict) or 'value' not in setting_data:
                    continue
                
                config_value = ConfigValue(
                    value=setting_data['value'],
                    description=setting_data.get('description', ''),
                    type=setting_data.get('type', 'string'),
                    default=setting_data.get('default', setting_data['value']),
                    min_value=setting_data.get('min'),
                    max_value=setting_data.get('max'),
                    options=setting_data.get('options')
                )
                
                settings[setting_name] = config_value
        
        return settings
    
    def _set_environment_variables(self, settings: Dict[str, ConfigValue]) -> None:
        """Convert configuration values to environment variables.
        
        Args:
            settings: Dictionary of setting names to ConfigValue objects
        """
        for setting_name, config_value in settings.items():
            # Convert value to string for environment variable
            if isinstance(config_value.value, bool):
                env_value = str(config_value.value).lower()
            elif isinstance(config_value.value, (int, float)):
                env_value = str(config_value.value)
            elif isinstance(config_value.value, dict):
                # For complex objects, we might need special handling
                env_value = json.dumps(config_value.value)
            else:
                env_value = str(config_value.value)
            
            # Set environment variable if not already set
            if setting_name not in os.environ:
                os.environ[setting_name] = env_value
                logger.debug(f"Set environment variable: {setting_name}={env_value}")
    
    def validate_config(self, settings: Dict[str, ConfigValue]) -> list:
        """Validate configuration values.
        
        Args:
            settings: Dictionary of setting names to ConfigValue objects
            
        Returns:
            List of validation errors
        """
        errors = []
        
        for setting_name, config_value in settings.items():
            # Type validation
            if config_value.type == 'integer':
                try:
                    int(config_value.value)
                except (ValueError, TypeError):
                    errors.append(f"{setting_name}: Value must be an integer")
                    continue
            
            elif config_value.type == 'float':
                try:
                    float(config_value.value)
                except (ValueError, TypeError):
                    errors.append(f"{setting_name}: Value must be a float")
                    continue
            
            elif config_value.type == 'boolean':
                if not isinstance(config_value.value, bool):
                    errors.append(f"{setting_name}: Value must be a boolean")
                    continue
            
            # Range validation
            if config_value.min_value is not None:
                if config_value.value < config_value.min_value:
                    errors.append(f"{setting_name}: Value {config_value.value} is below minimum {config_value.min_value}")
            
            if config_value.max_value is not None:
                if config_value.value > config_value.max_value:
                    errors.append(f"{setting_name}: Value {config_value.value} is above maximum {config_value.max_value}")
            
            # Options validation
            if config_value.options is not None:
                if config_value.value not in config_value.options:
                    errors.append(f"{setting_name}: Value {config_value.value} is not in allowed options {config_value.options}")
        
        return errors
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get a summary of the loaded configuration.
        
        Returns:
            Dictionary with configuration summary
        """
        if self.config_cache is None:
            self.load_config()
        
        summary = {
            'config_file': self.config_file_path,
            'total_settings': len(self.config_cache),
            'sections': {},
            'validation_errors': self.validate_config(self.config_cache)
        }
        
        # Group settings by type
        for setting_name, config_value in self.config_cache.items():
            section = setting_name.split('_')[0] if '_' in setting_name else 'other'
            if section not in summary['sections']:
                summary['sections'][section] = []
            summary['sections'][section].append({
                'name': setting_name,
                'value': config_value.value,
                'type': config_value.type,
                'description': config_value.description
            })
        
        return summary
    
    def reload_config(self) -> Dict[str, Any]:
        """Force reload configuration from file.
        
        Returns:
            Dictionary of configuration values
        """
        self.config_cache = None
        return self.load_config(force_reload=True)


# Global JSON config loader instance
json_config_loader = JSONConfigLoader()


def load_json_config(config_file_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from JSON file.
    
    Args:
        config_file_path: Optional path to config file
        
    Returns:
        Dictionary of configuration values
    """
    if config_file_path:
        loader = JSONConfigLoader(config_file_path)
        return loader.load_config()
    else:
        return json_config_loader.load_config()


def get_json_config_summary() -> Dict[str, Any]:
    """Get summary of JSON configuration.
    
    Returns:
        Configuration summary dictionary
    """
    return json_config_loader.get_config_summary()


def reload_json_config() -> Dict[str, Any]:
    """Reload JSON configuration.
    
    Returns:
        Dictionary of configuration values
    """
    return json_config_loader.reload_config() 