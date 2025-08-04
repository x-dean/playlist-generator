"""
Memory-efficient cache layer for audio analysis results and computed features.
Implements LRU cache with size limits and TTL for optimal memory usage.
"""

import time
import threading
import hashlib
import pickle
import gzip
from typing import Any, Dict, Optional, Tuple, List
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timedelta

from .logging_setup import get_logger, log_universal

logger = get_logger('playlista.memory_cache')


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    value: Any
    created_at: float
    last_accessed: float
    access_count: int
    size_bytes: int
    compressed: bool = False
    
    def is_expired(self, ttl_seconds: int) -> bool:
        """Check if entry has expired."""
        return (time.time() - self.created_at) > ttl_seconds
    
    def is_stale(self, max_age_seconds: int) -> bool:
        """Check if entry is stale based on last access."""
        return (time.time() - self.last_accessed) > max_age_seconds
    
    def mark_accessed(self):
        """Mark entry as recently accessed."""
        self.last_accessed = time.time()
        self.access_count += 1


class MemoryCache:
    """
    Thread-safe LRU cache with size limits, TTL, and compression.
    Optimized for audio analysis results and computed features.
    """
    
    def __init__(
        self,
        max_size_mb: int = 256,
        max_entries: int = 1000,
        default_ttl_seconds: int = 3600,
        compression_threshold_bytes: int = 1024,
        cleanup_interval_seconds: int = 300
    ):
        """
        Initialize memory cache.
        
        Args:
            max_size_mb: Maximum cache size in MB
            max_entries: Maximum number of entries
            default_ttl_seconds: Default TTL for entries
            compression_threshold_bytes: Compress entries larger than this
            cleanup_interval_seconds: Cleanup interval
        """
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.max_entries = max_entries
        self.default_ttl_seconds = default_ttl_seconds
        self.compression_threshold_bytes = compression_threshold_bytes
        self.cleanup_interval_seconds = cleanup_interval_seconds
        
        # Cache storage
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        
        # Statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'compressions': 0,
            'decompressions': 0,
            'current_size_bytes': 0,
            'current_entries': 0,
            'cleanup_runs': 0
        }
        
        # Background cleanup
        self._cleanup_timer = None
        self._start_cleanup_timer()
        
        log_universal('INFO', 'MemoryCache', f'Initialized cache: {max_size_mb}MB, {max_entries} entries')
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            if key not in self._cache:
                self.stats['misses'] += 1
                return None
            
            entry = self._cache[key]
            
            # Check expiration
            if entry.is_expired(self.default_ttl_seconds):
                self._remove_entry(key)
                self.stats['misses'] += 1
                return None
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            entry.mark_accessed()
            
            # Decompress if needed
            value = entry.value
            if entry.compressed:
                try:
                    value = pickle.loads(gzip.decompress(value))
                    self.stats['decompressions'] += 1
                except Exception as e:
                    log_universal('WARNING', 'MemoryCache', f'Decompression failed for {key}: {e}')
                    self._remove_entry(key)
                    self.stats['misses'] += 1
                    return None
            
            self.stats['hits'] += 1
            return value
    
    def put(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> bool:
        """Put value in cache."""
        if ttl_seconds is None:
            ttl_seconds = self.default_ttl_seconds
        
        with self._lock:
            # Calculate size
            try:
                serialized_value = pickle.dumps(value)
                value_size = len(serialized_value)
                
                # Compress if large enough
                compressed = False
                if value_size > self.compression_threshold_bytes:
                    try:
                        compressed_value = gzip.compress(serialized_value, compresslevel=6)
                        if len(compressed_value) < value_size * 0.8:  # Only if significant compression
                            serialized_value = compressed_value
                            value_size = len(compressed_value)
                            compressed = True
                            self.stats['compressions'] += 1
                    except Exception as e:
                        log_universal('WARNING', 'MemoryCache', f'Compression failed for {key}: {e}')
                
                # Check if value is too large
                if value_size > self.max_size_bytes * 0.5:  # Max 50% of total cache size
                    log_universal('WARNING', 'MemoryCache', f'Value too large for cache: {key} ({value_size} bytes)')
                    return False
                
                # Remove existing entry if present
                if key in self._cache:
                    self._remove_entry(key)
                
                # Ensure space
                while (
                    (self.stats['current_size_bytes'] + value_size > self.max_size_bytes) or
                    (self.stats['current_entries'] >= self.max_entries)
                ):
                    if not self._evict_lru():
                        log_universal('WARNING', 'MemoryCache', 'Failed to evict entries for new value')
                        return False
                
                # Create and store entry
                entry = CacheEntry(
                    value=serialized_value if compressed else value,
                    created_at=time.time(),
                    last_accessed=time.time(),
                    access_count=0,
                    size_bytes=value_size,
                    compressed=compressed
                )
                
                self._cache[key] = entry
                self.stats['current_size_bytes'] += value_size
                self.stats['current_entries'] += 1
                
                return True
                
            except Exception as e:
                log_universal('ERROR', 'MemoryCache', f'Failed to cache value for {key}: {e}')
                return False
    
    def remove(self, key: str) -> bool:
        """Remove entry from cache."""
        with self._lock:
            if key in self._cache:
                self._remove_entry(key)
                return True
            return False
    
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self.stats['current_size_bytes'] = 0
            self.stats['current_entries'] = 0
            log_universal('INFO', 'MemoryCache', 'Cache cleared')
    
    def _remove_entry(self, key: str):
        """Remove entry and update stats."""
        if key in self._cache:
            entry = self._cache.pop(key)
            self.stats['current_size_bytes'] -= entry.size_bytes
            self.stats['current_entries'] -= 1
    
    def _evict_lru(self) -> bool:
        """Evict least recently used entry."""
        if not self._cache:
            return False
        
        # Get LRU key (first in OrderedDict)
        lru_key = next(iter(self._cache))
        self._remove_entry(lru_key)
        self.stats['evictions'] += 1
        
        log_universal('DEBUG', 'MemoryCache', f'Evicted LRU entry: {lru_key}')
        return True
    
    def cleanup_expired(self):
        """Remove expired and stale entries."""
        with self._lock:
            expired_keys = []
            current_time = time.time()
            
            for key, entry in self._cache.items():
                if (entry.is_expired(self.default_ttl_seconds) or 
                    entry.is_stale(self.default_ttl_seconds * 2)):
                    expired_keys.append(key)
            
            for key in expired_keys:
                self._remove_entry(key)
            
            if expired_keys:
                log_universal('INFO', 'MemoryCache', f'Cleaned up {len(expired_keys)} expired entries')
            
            self.stats['cleanup_runs'] += 1
    
    def _start_cleanup_timer(self):
        """Start background cleanup timer."""
        def cleanup_worker():
            try:
                self.cleanup_expired()
            except Exception as e:
                log_universal('ERROR', 'MemoryCache', f'Cleanup error: {e}')
            finally:
                # Schedule next cleanup
                self._cleanup_timer = threading.Timer(
                    self.cleanup_interval_seconds,
                    cleanup_worker
                )
                self._cleanup_timer.daemon = True
                self._cleanup_timer.start()
        
        self._cleanup_timer = threading.Timer(
            self.cleanup_interval_seconds,
            cleanup_worker
        )
        self._cleanup_timer.daemon = True
        self._cleanup_timer.start()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            hit_rate = 0.0
            total_requests = self.stats['hits'] + self.stats['misses']
            if total_requests > 0:
                hit_rate = self.stats['hits'] / total_requests
            
            return {
                **self.stats,
                'hit_rate': hit_rate,
                'size_mb': self.stats['current_size_bytes'] / (1024 * 1024),
                'utilization_percent': (self.stats['current_entries'] / self.max_entries) * 100,
                'memory_utilization_percent': (self.stats['current_size_bytes'] / self.max_size_bytes) * 100
            }
    
    def get_key_for_file_analysis(self, file_path: str, force_reanalysis: bool = False) -> str:
        """Generate cache key for file analysis."""
        import os
        
        # Include file modification time and size in key
        try:
            stat = os.stat(file_path)
            key_data = f"{file_path}:{stat.st_mtime}:{stat.st_size}:{force_reanalysis}"
            return hashlib.md5(key_data.encode()).hexdigest()
        except:
            # Fallback to simple hash
            key_data = f"{file_path}:{force_reanalysis}"
            return hashlib.md5(key_data.encode()).hexdigest()
    
    def shutdown(self):
        """Shutdown cache and cleanup resources."""
        if self._cleanup_timer:
            self._cleanup_timer.cancel()
        
        with self._lock:
            self._cache.clear()
            self.stats['current_size_bytes'] = 0
            self.stats['current_entries'] = 0
        
        log_universal('INFO', 'MemoryCache', 'Cache shutdown complete')


# Global cache instances
_analysis_cache = None
_feature_cache = None
_cache_lock = threading.Lock()


def get_analysis_cache() -> MemoryCache:
    """Get or create analysis results cache."""
    global _analysis_cache
    
    with _cache_lock:
        if _analysis_cache is None:
            _analysis_cache = MemoryCache(
                max_size_mb=128,
                max_entries=500,
                default_ttl_seconds=3600  # 1 hour
            )
        return _analysis_cache


def get_feature_cache() -> MemoryCache:
    """Get or create feature computation cache."""
    global _feature_cache
    
    with _cache_lock:
        if _feature_cache is None:
            _feature_cache = MemoryCache(
                max_size_mb=64,
                max_entries=1000,
                default_ttl_seconds=1800  # 30 minutes
            )
        return _feature_cache