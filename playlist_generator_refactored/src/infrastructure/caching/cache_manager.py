"""
Cache manager for in-memory and file-based caching.

This module provides a unified caching interface for storing
analysis results, API responses, and other frequently accessed data.
"""

import logging
import json
import pickle
import hashlib
from pathlib import Path
from typing import Any, Optional, Dict, List, Union
from datetime import datetime, timedelta
from functools import wraps
import threading

from shared.config import get_config
from shared.exceptions import CacheError


class CacheEntry:
    """Represents a cache entry with metadata."""
    
    def __init__(self, key: str, value: Any, ttl_seconds: Optional[int] = None):
        """Initialize cache entry."""
        self.key = key
        self.value = value
        self.created_at = datetime.now()
        self.ttl_seconds = ttl_seconds
        self.access_count = 0
        self.last_accessed = datetime.now()
    
    def is_expired(self) -> bool:
        """Check if entry is expired."""
        if self.ttl_seconds is None:
            return False
        return datetime.now() > self.created_at + timedelta(seconds=self.ttl_seconds)
    
    def access(self):
        """Mark entry as accessed."""
        self.access_count += 1
        self.last_accessed = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'key': self.key,
            'value': self.value,
            'created_at': self.created_at.isoformat(),
            'ttl_seconds': self.ttl_seconds,
            'access_count': self.access_count,
            'last_accessed': self.last_accessed.isoformat()
        }


class CacheManager:
    """Unified cache manager for in-memory and file-based caching."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize cache manager."""
        self.config = get_config()
        self.cache_dir = cache_dir or self.config.database.cache_dir / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        self._memory_cache: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        
        # Cache statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
    
    def _generate_key(self, *args, **kwargs) -> str:
        """Generate a cache key from arguments."""
        key_data = {
            'args': args,
            'kwargs': sorted(kwargs.items())
        }
        key_string = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache."""
        with self._lock:
            if key in self._memory_cache:
                entry = self._memory_cache[key]
                
                if entry.is_expired():
                    del self._memory_cache[key]
                    self._evictions += 1
                    self._misses += 1
                    return default
                
                entry.access()
                self._hits += 1
                return entry.value
            
            # Try file cache
            file_value = self._get_from_file(key)
            if file_value is not None:
                self._hits += 1
                return file_value
            
            self._misses += 1
            return default
    
    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        """Set value in cache."""
        with self._lock:
            entry = CacheEntry(key, value, ttl_seconds)
            self._memory_cache[key] = entry
            
            # Also save to file cache for persistence
            self._save_to_file(key, value, ttl_seconds)
    
    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        with self._lock:
            # Remove from memory cache
            if key in self._memory_cache:
                del self._memory_cache[key]
            
            # Remove from file cache
            return self._delete_from_file(key)
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._memory_cache.clear()
            self._clear_file_cache()
            self.logger.info("Cache cleared")
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        with self._lock:
            if key in self._memory_cache:
                entry = self._memory_cache[key]
                if entry.is_expired():
                    del self._memory_cache[key]
                    return False
                return True
            
            return self._exists_in_file(key)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0
            
            return {
                'hits': self._hits,
                'misses': self._misses,
                'evictions': self._evictions,
                'hit_rate_percent': round(hit_rate, 2),
                'memory_entries': len(self._memory_cache),
                'file_entries': self._count_file_entries()
            }
    
    def cleanup_expired(self) -> int:
        """Remove expired entries from cache."""
        with self._lock:
            expired_keys = []
            for key, entry in self._memory_cache.items():
                if entry.is_expired():
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self._memory_cache[key]
            
            self.logger.info(f"Cleaned up {len(expired_keys)} expired entries")
            return len(expired_keys)
    
    def _get_from_file(self, key: str) -> Optional[Any]:
        """Get value from file cache."""
        try:
            cache_file = self.cache_dir / f"{key}.cache"
            if not cache_file.exists():
                return None
            
            # Check if file is expired
            metadata_file = self.cache_dir / f"{key}.meta"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                created_at = datetime.fromisoformat(metadata['created_at'])
                ttl_seconds = metadata.get('ttl_seconds')
                
                if ttl_seconds and datetime.now() > created_at + timedelta(seconds=ttl_seconds):
                    self._delete_from_file(key)
                    return None
            
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
                
        except Exception as e:
            self.logger.warning(f"Failed to read from file cache for key {key}: {e}")
            return None
    
    def _save_to_file(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        """Save value to file cache."""
        try:
            cache_file = self.cache_dir / f"{key}.cache"
            metadata_file = self.cache_dir / f"{key}.meta"
            
            # Save value
            with open(cache_file, 'wb') as f:
                pickle.dump(value, f)
            
            # Save metadata
            metadata = {
                'created_at': datetime.now().isoformat(),
                'ttl_seconds': ttl_seconds
            }
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f)
                
        except Exception as e:
            self.logger.warning(f"Failed to save to file cache for key {key}: {e}")
    
    def _delete_from_file(self, key: str) -> bool:
        """Delete value from file cache."""
        try:
            cache_file = self.cache_dir / f"{key}.cache"
            metadata_file = self.cache_dir / f"{key}.meta"
            
            deleted = False
            if cache_file.exists():
                cache_file.unlink()
                deleted = True
            
            if metadata_file.exists():
                metadata_file.unlink()
            
            return deleted
            
        except Exception as e:
            self.logger.warning(f"Failed to delete from file cache for key {key}: {e}")
            return False
    
    def _exists_in_file(self, key: str) -> bool:
        """Check if key exists in file cache."""
        cache_file = self.cache_dir / f"{key}.cache"
        return cache_file.exists()
    
    def _clear_file_cache(self) -> None:
        """Clear all file cache entries."""
        try:
            for cache_file in self.cache_dir.glob("*.cache"):
                cache_file.unlink()
            
            for metadata_file in self.cache_dir.glob("*.meta"):
                metadata_file.unlink()
                
        except Exception as e:
            self.logger.warning(f"Failed to clear file cache: {e}")
    
    def _count_file_entries(self) -> int:
        """Count file cache entries."""
        try:
            return len(list(self.cache_dir.glob("*.cache")))
        except Exception:
            return 0


def cached(ttl_seconds: Optional[int] = None, key_prefix: str = ""):
    """Decorator for caching function results."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get cache manager instance
            cache_manager = CacheManager()
            
            # Generate cache key
            cache_key = f"{key_prefix}:{func.__name__}:{cache_manager._generate_key(*args, **kwargs)}"
            
            # Try to get from cache
            cached_result = cache_manager.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache_manager.set(cache_key, result, ttl_seconds)
            
            return result
        return wrapper
    return decorator


# Global cache manager instance
_cache_manager: Optional[CacheManager] = None


def get_cache_manager() -> CacheManager:
    """Get global cache manager instance."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager 