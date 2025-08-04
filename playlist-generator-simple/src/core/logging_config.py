"""
Centralized logging configuration for Playlist Generator.

This module provides standardized logging configurations and patterns
for different use cases throughout the application.
"""

import os
from typing import Dict, Any
from pathlib import Path


class LoggingPatterns:
    """Standard logging message patterns and components."""
    
    # Standard component names for consistent logging
    COMPONENTS = {
        'CLI': 'CLI',
        'AUDIO_ANALYZER': 'Audio',
        'DATABASE': 'Database',
        'CONFIG': 'Config',
        'PLAYLIST': 'Playlist',
        'ANALYSIS': 'Analysis',
        'RESOURCE_MANAGER': 'Resource',
        'SYSTEM': 'System',
        'FILE_DISCOVERY': 'FileDiscovery',
        'ENRICHMENT': 'Enrichment',
        'EXPORT': 'Export',
        'PIPELINE': 'Pipeline',
        'SEQUENTIAL': 'Sequential',
        'PARALLEL': 'Parallel',
        'STREAMING': 'Streaming',
        'API_MUSICBRAINZ': 'MB API',
        'API_LASTFM': 'LF API',
        'API_GENERIC': 'API',
        'ESSENTIA': 'Essentia',
        'LIBROSA': 'Librosa',
        'TENSORFLOW': 'TensorFlow',
        'CACHE': 'Cache',
        'FILE': 'File',
    }
    
    # Standard log level mappings for different scenarios
    LEVEL_MAPPINGS = {
        'development': 'DEBUG',
        'testing': 'INFO',
        'production': 'WARNING',
        'verbose': 'DEBUG',
        'quiet': 'ERROR',
    }


class LoggingConfigurations:
    """Pre-defined logging configurations for different scenarios."""
    
    @staticmethod
    def get_development_config() -> Dict[str, Any]:
        """Development configuration with full logging and colors."""
        return {
            'level': 'DEBUG',
            'console_enabled': True,
            'file_enabled': True,
            'structured_logging': False,
            'colored_output': True,
            'log_dir': 'logs',
            'log_file_prefix': 'playlista_dev',
            'max_file_size_mb': 10,
            'backup_count': 5,
            'performance_sampling': False,
            'external_library_level': 'INFO',  # More verbose for debugging
        }
    
    @staticmethod
    def get_production_config() -> Dict[str, Any]:
        """Production configuration with structured logging and minimal console output."""
        return {
            'level': 'INFO',
            'console_enabled': True,
            'file_enabled': True,
            'structured_logging': True,
            'colored_output': False,
            'log_dir': '/app/logs' if os.path.exists('/app') else 'logs',
            'log_file_prefix': 'playlista',
            'max_file_size_mb': 100,
            'backup_count': 20,
            'performance_sampling': True,
            'sample_rate': 0.05,  # 5% sampling for debug messages
            'external_library_level': 'ERROR',
        }
    
    @staticmethod
    def get_testing_config() -> Dict[str, Any]:
        """Testing configuration with file-only logging."""
        return {
            'level': 'INFO',
            'console_enabled': False,
            'file_enabled': True,
            'structured_logging': True,
            'colored_output': False,
            'log_dir': 'test_logs',
            'log_file_prefix': 'playlista_test',
            'max_file_size_mb': 5,
            'backup_count': 3,
            'performance_sampling': False,
            'external_library_level': 'ERROR',
        }
    
    @staticmethod
    def get_cli_config(verbose_level: int = 0) -> Dict[str, Any]:
        """CLI configuration based on verbosity level."""
        base_config = {
            'console_enabled': True,
            'file_enabled': True,
            'structured_logging': False,
            'colored_output': True,
            'log_dir': 'logs',
            'log_file_prefix': 'playlista_cli',
            'max_file_size_mb': 20,
            'backup_count': 10,
            'performance_sampling': False,
            'external_library_level': 'WARNING',
        }
        
        # Adjust level based on verbosity
        if verbose_level >= 3:
            base_config.update({
                'level': 'DEBUG',
                'external_library_level': 'DEBUG',
                'performance_sampling': False,
            })
        elif verbose_level == 2:
            base_config.update({
                'level': 'DEBUG',
                'external_library_level': 'INFO',
            })
        elif verbose_level == 1:
            base_config.update({
                'level': 'INFO',
                'external_library_level': 'WARNING',
            })
        else:
            base_config.update({
                'level': 'WARNING',
                'external_library_level': 'ERROR',
            })
        
        return base_config
    
    @staticmethod
    def get_api_config() -> Dict[str, Any]:
        """API server configuration with structured logging."""
        return {
            'level': 'INFO',
            'console_enabled': True,
            'file_enabled': True,
            'structured_logging': True,
            'colored_output': False,
            'log_dir': '/app/logs' if os.path.exists('/app') else 'logs',
            'log_file_prefix': 'playlista_api',
            'max_file_size_mb': 50,
            'backup_count': 15,
            'performance_sampling': True,
            'sample_rate': 0.1,
            'external_library_level': 'ERROR',
        }
    
    @staticmethod
    def get_config_by_environment() -> Dict[str, Any]:
        """Get configuration based on environment variables."""
        env = os.environ.get('ENVIRONMENT', 'development').lower()
        
        config_map = {
            'development': LoggingConfigurations.get_development_config,
            'production': LoggingConfigurations.get_production_config,
            'testing': LoggingConfigurations.get_testing_config,
            'api': LoggingConfigurations.get_api_config,
        }
        
        return config_map.get(env, LoggingConfigurations.get_development_config)()


class StandardLogMessages:
    """Standard log message templates for consistency."""
    
    @staticmethod
    def startup_message(component: str, version: str = None) -> str:
        """Standard startup message."""
        version_info = f" v{version}" if version else ""
        return f"{component}{version_info} starting up"
    
    @staticmethod
    def shutdown_message(component: str) -> str:
        """Standard shutdown message."""
        return f"{component} shutting down"
    
    @staticmethod
    def operation_start(operation: str, target: str = None) -> str:
        """Standard operation start message."""
        target_info = f" for {target}" if target else ""
        return f"Starting {operation}{target_info}"
    
    @staticmethod
    def operation_complete(operation: str, target: str = None, duration: float = None) -> str:
        """Standard operation completion message."""
        target_info = f" for {target}" if target else ""
        duration_info = f" in {duration:.3f}s" if duration is not None else ""
        return f"Completed {operation}{target_info}{duration_info}"
    
    @staticmethod
    def operation_failed(operation: str, target: str = None, error: str = None) -> str:
        """Standard operation failure message."""
        target_info = f" for {target}" if target else ""
        error_info = f": {error}" if error else ""
        return f"Failed {operation}{target_info}{error_info}"
    
    @staticmethod
    def resource_usage(component: str, **metrics) -> str:
        """Standard resource usage message."""
        metric_strs = [f"{k}={v}" for k, v in metrics.items()]
        return f"{component} resource usage: {', '.join(metric_strs)}"
    
    @staticmethod
    def api_call(api_name: str, operation: str, success: bool, duration: float = None) -> str:
        """Standard API call message."""
        status = "SUCCESS" if success else "FAILED"
        duration_info = f" ({duration:.3f}s)" if duration is not None else ""
        return f"{api_name} {operation}: {status}{duration_info}"
    
    @staticmethod
    def cache_operation(operation: str, key: str, hit: bool = None) -> str:
        """Standard cache operation message."""
        if hit is not None:
            status = "HIT" if hit else "MISS"
            return f"Cache {operation} for '{key}': {status}"
        return f"Cache {operation} for '{key}'"
    
    @staticmethod
    def database_operation(operation: str, table: str, success: bool, count: int = None) -> str:
        """Standard database operation message."""
        status = "SUCCESS" if success else "FAILED"
        count_info = f" ({count} rows)" if count is not None else ""
        return f"Database {operation} on {table}: {status}{count_info}"


def get_logging_config(
    environment: str = None,
    verbose_level: int = 0,
    config_overrides: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Get a logging configuration based on environment and parameters.
    
    Args:
        environment: Environment name ('development', 'production', 'testing', 'api')
        verbose_level: Verbosity level for CLI applications (0-3)
        config_overrides: Dictionary of configuration overrides
    
    Returns:
        Complete logging configuration
    """
    # Determine environment
    if environment is None:
        environment = os.environ.get('ENVIRONMENT', 'development').lower()
    
    # Get base configuration
    if environment == 'cli' or verbose_level > 0:
        config = LoggingConfigurations.get_cli_config(verbose_level)
    else:
        config_map = {
            'development': LoggingConfigurations.get_development_config,
            'production': LoggingConfigurations.get_production_config,
            'testing': LoggingConfigurations.get_testing_config,
            'api': LoggingConfigurations.get_api_config,
        }
        config_func = config_map.get(environment, LoggingConfigurations.get_development_config)
        config = config_func()
    
    # Apply overrides
    if config_overrides:
        config.update(config_overrides)
    
    # Ensure log directory exists
    log_dir = Path(config['log_dir'])
    log_dir.mkdir(parents=True, exist_ok=True)
    
    return config


def create_logger_name(module_name: str, component: str = None) -> str:
    """
    Create a standardized logger name.
    
    Args:
        module_name: Python module name (e.g., __name__)
        component: Optional component identifier
    
    Returns:
        Standardized logger name
    """
    # Convert module path to logger name
    if module_name.startswith('src.'):
        module_name = module_name[4:]  # Remove 'src.' prefix
    
    base_name = f"playlista.{module_name}"
    
    if component:
        return f"{base_name}.{component}"
    
    return base_name