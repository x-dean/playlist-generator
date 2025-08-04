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
            
            # Check if expired
            if entry.is_expired(self.default_ttl_seconds):
                self._remove_entry(key)
                self.stats['misses'] += 1
                return None
            
            # Mark as accessed and move to end (LRU)
            entry.mark_accessed()
            self._cache.move_to_end(key)
            
            self.stats['hits'] += 1
            
            # Decompress if needed
            if entry.compressed:
                try:
                    value = pickle.loads(gzip.decompress(entry.value))
                    self.stats['decompressions'] += 1
                    return value
                except Exception as e:
                    log_universal('ERROR', 'MemoryCache', f'Decompression failed for {key}: {e}')
                    self._remove_entry(key)
                    return None
            
            return entry.value
    
    def put(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> bool:
        """
        Put value in cache with optional TTL.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: Optional TTL override
            
        Returns:
            True if successfully cached, False otherwise
        """
        try:
            # Serialize value
            serialized = pickle.dumps(value)
            size_bytes = len(serialized)
            
            # Compress if above threshold
            compressed = False
            if size_bytes > self.compression_threshold_bytes:
                try:
                    serialized = gzip.compress(serialized)
                    size_bytes = len(serialized)
                    compressed = True
                    self.stats['compressions'] += 1
                except Exception as e:
                    log_universal('WARNING', 'MemoryCache', f'Compression failed for {key}: {e}')
            
            with self._lock:
                # Remove existing entry if present
                if key in self._cache:
                    self._remove_entry(key)
                
                # Check if we need to evict entries
                while (self.stats['current_size_bytes'] + size_bytes > self.max_size_bytes or 
                       self.stats['current_entries'] >= self.max_entries):
                    if not self._evict_lru():
                        log_universal('WARNING', 'MemoryCache', f'Failed to evict entries for {key}')
                        return False
                
                # Create new entry
                entry = CacheEntry(
                    value=serialized,
                    created_at=time.time(),
                    last_accessed=time.time(),
                    access_count=1,
                    size_bytes=size_bytes,
                    compressed=compressed
                )
                
                # Add to cache
                self._cache[key] = entry
                self.stats['current_size_bytes'] += size_bytes
                self.stats['current_entries'] += 1
                
                return True
                
        except Exception as e:
            log_universal('ERROR', 'MemoryCache', f'Failed to cache {key}: {e}')
            return False
    
    def remove(self, key: str) -> bool:
        """Remove entry from cache."""
        with self._lock:
            if key in self._cache:
                self._remove_entry(key)
                return True
            return False
    
    def clear(self):
        """Clear all entries from cache."""
        with self._lock:
            self._cache.clear()
            self.stats['current_size_bytes'] = 0
            self.stats['current_entries'] = 0
            log_universal('INFO', 'MemoryCache', 'Cache cleared')
    
    def _remove_entry(self, key: str):
        """Remove entry and update statistics."""
        if key in self._cache:
            entry = self._cache[key]
            self.stats['current_size_bytes'] -= entry.size_bytes
            self.stats['current_entries'] -= 1
            del self._cache[key]
    
    def _evict_lru(self) -> bool:
        """Evict least recently used entry."""
        if not self._cache:
            return False
        
        # Remove oldest entry
        oldest_key = next(iter(self._cache))
        self._remove_entry(oldest_key)
        self.stats['evictions'] += 1
        return True
    
    def cleanup_expired(self):
        """Remove expired entries from cache."""
        with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired(self.default_ttl_seconds)
            ]
            
            for key in expired_keys:
                self._remove_entry(key)
            
            if expired_keys:
                log_universal('INFO', 'MemoryCache', f'Cleaned up {len(expired_keys)} expired entries')
            
            self.stats['cleanup_runs'] += 1
    
    def _start_cleanup_timer(self):
        """Start background cleanup timer."""
        def cleanup_worker():
            while True:
                try:
                    time.sleep(self.cleanup_interval_seconds)
                    self.cleanup_expired()
                except Exception as e:
                    log_universal('ERROR', 'MemoryCache', f'Cleanup worker error: {e}')
        
        import threading
        cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        cleanup_thread.start()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self.stats['hits'] + self.stats['misses']
            hit_rate = (self.stats['hits'] / total_requests * 100) if total_requests > 0 else 0
            
            return {
                **self.stats,
                'hit_rate': hit_rate,
                'size_mb': self.stats['current_size_bytes'] / (1024 * 1024),
                'max_size_mb': self.max_size_bytes / (1024 * 1024),
                'utilization_percent': (self.stats['current_size_bytes'] / self.max_size_bytes * 100) if self.max_size_bytes > 0 else 0
            }
    
    def get_key_for_file_analysis(self, file_path: str, force_reanalysis: bool = False) -> str:
        """Generate cache key for file analysis."""
        # Create hash of file path and force flag
        key_data = f"{file_path}:{force_reanalysis}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def shutdown(self):
        """Shutdown cache and cleanup resources."""
        try:
            self.clear()
            log_universal('INFO', 'MemoryCache', 'Cache shutdown complete')
        except Exception as e:
            log_universal('ERROR', 'MemoryCache', f'Error during shutdown: {e}')


# Global cache instances
_analysis_cache = None
_feature_cache = None


def get_analysis_cache() -> MemoryCache:
    """Get or create the global analysis cache instance."""
    global _analysis_cache
    
    if _analysis_cache is None:
        # Larger cache for analysis results
        _analysis_cache = MemoryCache(
            max_size_mb=512,  # 512MB for analysis results
            max_entries=2000,
            default_ttl_seconds=7200,  # 2 hours
            compression_threshold_bytes=2048
        )
    
    return _analysis_cache


def get_feature_cache() -> MemoryCache:
    """Get or create the global feature cache instance."""
    global _feature_cache
    
    if _feature_cache is None:
        # Smaller cache for computed features
        _feature_cache = MemoryCache(
            max_size_mb=256,  # 256MB for features
            max_entries=1000,
            default_ttl_seconds=3600,  # 1 hour
            compression_threshold_bytes=1024
        )
    
    return _feature_cache