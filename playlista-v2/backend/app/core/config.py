"""
Configuration management for Playlista v2
"""

import os
from functools import lru_cache
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # API Settings
    api_title: str = "Playlista v2 API"
    api_version: str = "2.0.0"
    debug: bool = Field(default=False, description="Debug mode")
    
    # Server Settings
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    workers: int = Field(default=1, description="Number of workers")
    
    # CORS Settings
    allowed_origins: List[str] = Field(
        default=["http://localhost:3000", "http://127.0.0.1:3000"],
        description="Allowed CORS origins"
    )
    
    # Database Settings
    database_url: str = Field(
        default="postgresql+asyncpg://playlista:playlista@localhost:5432/playlista_v2",
        description="Database connection URL"
    )
    database_pool_size: int = Field(default=20, description="Database connection pool size")
    database_max_overflow: int = Field(default=0, description="Database max overflow connections")
    
    # Redis Settings
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL"
    )
    redis_max_connections: int = Field(default=20, description="Redis connection pool size")
    
    # Logging Settings
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(default="json", description="Log format (json/text)")
    
    # Analysis Settings
    analysis_workers: int = Field(default=4, description="Number of analysis workers")
    analysis_batch_size: int = Field(default=10, description="Analysis batch size")
    analysis_timeout: int = Field(default=300, description="Analysis timeout in seconds")
    max_file_size_mb: int = Field(default=500, description="Maximum file size in MB")
    
    # Machine Learning Settings
    ml_model_path: str = Field(default="./models", description="ML models directory")
    enable_gpu: bool = Field(default=True, description="Enable GPU acceleration")
    torch_device: str = Field(default="auto", description="PyTorch device (auto/cpu/cuda)")
    
    # Cache Settings
    cache_ttl: int = Field(default=3600, description="Cache TTL in seconds")
    feature_cache_size: int = Field(default=10000, description="Feature cache size")
    
    # File Storage Settings
    music_library_path: str = Field(
        default="./music",
        description="Music library directory"
    )
    temp_directory: str = Field(default="./temp", description="Temporary files directory")
    max_concurrent_uploads: int = Field(default=5, description="Max concurrent file uploads")
    
    # Playlist Generation Settings
    default_playlist_size: int = Field(default=25, description="Default playlist size")
    max_playlist_size: int = Field(default=500, description="Maximum playlist size")
    similarity_threshold: float = Field(default=0.7, description="Similarity threshold")
    
    # Performance Settings
    enable_async_analysis: bool = Field(default=True, description="Enable async analysis")
    enable_parallel_processing: bool = Field(default=True, description="Enable parallel processing")
    memory_limit_gb: int = Field(default=8, description="Memory limit in GB")
    
    # Security Settings
    secret_key: str = Field(
        default="your-secret-key-change-in-production",
        description="Secret key for JWT tokens"
    )
    access_token_expire_minutes: int = Field(
        default=30,
        description="Access token expiration in minutes"
    )
    
    class Config:
        env_file = ".env"
        env_prefix = "PLAYLISTA_"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get application settings (cached)"""
    return Settings()


# Create settings instance
settings = get_settings()
