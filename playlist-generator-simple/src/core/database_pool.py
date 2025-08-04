"""
Database connection pool for improved performance and resource management.
Implements connection pooling, query optimization, and batch operations.
"""

import sqlite3
import threading
import time
import queue
from typing import Dict, Any, List, Optional, ContextManager
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime

from .logging_setup import get_logger, log_universal

logger = get_logger('playlista.database_pool')


@dataclass
class PoolConfig:
    """Configuration for database connection pool."""
    min_connections: int = 2
    max_connections: int = 10
    connection_timeout: int = 30
    idle_timeout: int = 300  # 5 minutes
    max_retries: int = 3
    enable_wal_mode: bool = True
    enable_foreign_keys: bool = True
    cache_size: int = -64000  # 64MB cache
    journal_mode: str = "WAL"
    synchronous: str = "NORMAL"


class PooledConnection:
    """Wrapper for pooled database connections."""
    
    def __init__(self, connection: sqlite3.Connection, created_at: float):
        self.connection = connection
        self.created_at = created_at
        self.last_used = created_at
        self.in_use = False
        self.transaction_active = False
    
    def is_expired(self, idle_timeout: int) -> bool:
        """Check if connection has been idle too long."""
        return (time.time() - self.last_used) > idle_timeout
    
    def mark_used(self):
        """Mark connection as recently used."""
        self.last_used = time.time()


class DatabaseConnectionPool:
    """
    Thread-safe database connection pool with automatic management.
    """
    
    def __init__(self, db_path: str, config: PoolConfig = None):
        self.db_path = db_path
        self.config = config or PoolConfig()
        
        # Thread safety
        self._lock = threading.RLock()
        self._connections: List[PooledConnection] = []
        self._connection_queue = queue.Queue(maxsize=self.config.max_connections)
        
        # Statistics
        self.stats = {
            'connections_created': 0,
            'connections_reused': 0,
            'connections_expired': 0,
            'total_queries': 0,
            'failed_queries': 0,
            'pool_hits': 0,
            'pool_misses': 0
        }
        
        # Initialize minimum connections
        self._initialize_pool()
        
        log_universal('INFO', 'DatabasePool', f'Initialized connection pool: {self.db_path}')
        log_universal('INFO', 'DatabasePool', f'Pool config: min={self.config.min_connections}, max={self.config.max_connections}')
    
    def _initialize_pool(self):
        """Initialize the connection pool with minimum connections."""
        with self._lock:
            for _ in range(self.config.min_connections):
                conn = self._create_connection()
                if conn:
                    pooled_conn = PooledConnection(conn, time.time())
                    self._connections.append(pooled_conn)
                    try:
                        self._connection_queue.put_nowait(pooled_conn)
                    except queue.Full:
                        conn.close()
    
    def _create_connection(self) -> Optional[sqlite3.Connection]:
        """Create a new database connection with optimizations."""
        try:
            conn = sqlite3.connect(
                self.db_path,
                timeout=self.config.connection_timeout,
                check_same_thread=False
            )
            
            # Enable row factory for better data access
            conn.row_factory = sqlite3.Row
            
            # Apply performance optimizations
            cursor = conn.cursor()
            
            if self.config.enable_foreign_keys:
                cursor.execute("PRAGMA foreign_keys = ON")
            
            cursor.execute(f"PRAGMA cache_size = {self.config.cache_size}")
            cursor.execute(f"PRAGMA journal_mode = {self.config.journal_mode}")
            cursor.execute(f"PRAGMA synchronous = {self.config.synchronous}")
            cursor.execute("PRAGMA temp_store = MEMORY")
            cursor.execute("PRAGMA mmap_size = 268435456")  # 256MB
            
            conn.commit()
            
            self.stats['connections_created'] += 1
            return conn
            
        except Exception as e:
            log_universal('ERROR', 'DatabasePool', f'Failed to create connection: {e}')
            return None
    
    @contextmanager
    def get_connection(self) -> ContextManager[sqlite3.Connection]:
        """Get a connection from the pool (context manager)."""
        pooled_conn = None
        try:
            # Try to get from queue first (fast path)
            try:
                pooled_conn = self._connection_queue.get_nowait()
                pooled_conn.mark_used()
                pooled_conn.in_use = True
                self.stats['pool_hits'] += 1
                yield pooled_conn.connection
                return
            except queue.Empty:
                self.stats['pool_misses'] += 1
            
            # Slow path: find available connection or create new one
            with self._lock:
                # Look for available connection
                for pc in self._connections:
                    if not pc.in_use and not pc.is_expired(self.config.idle_timeout):
                        pc.mark_used()
                        pc.in_use = True
                        self.stats['connections_reused'] += 1
                        pooled_conn = pc
                        break
                
                # Create new connection if none available and under limit
                if pooled_conn is None and len(self._connections) < self.config.max_connections:
                    conn = self._create_connection()
                    if conn:
                        pooled_conn = PooledConnection(conn, time.time())
                        pooled_conn.in_use = True
                        self._connections.append(pooled_conn)
                
                # If still no connection, wait for one to become available
                if pooled_conn is None:
                    try:
                        pooled_conn = self._connection_queue.get(timeout=5)
                        pooled_conn.mark_used()
                        pooled_conn.in_use = True
                    except queue.Empty:
                        raise RuntimeError("Unable to get database connection: pool exhausted")
            
            yield pooled_conn.connection
            
        finally:
            if pooled_conn:
                pooled_conn.in_use = False
                # Return to queue if not expired
                if not pooled_conn.is_expired(self.config.idle_timeout):
                    try:
                        self._connection_queue.put_nowait(pooled_conn)
                    except queue.Full:
                        pass  # Connection will be cleaned up later
    
    def execute_query(self, query: str, params: tuple = None, fetch_one: bool = False, fetch_all: bool = True):
        """Execute a query with connection pooling."""
        self.stats['total_queries'] += 1
        
        for attempt in range(self.config.max_retries):
            try:
                with self.get_connection() as conn:
                    cursor = conn.cursor()
                    
                    if params:
                        cursor.execute(query, params)
                    else:
                        cursor.execute(query)
                    
                    if fetch_one:
                        return cursor.fetchone()
                    elif fetch_all:
                        return cursor.fetchall()
                    else:
                        conn.commit()
                        return cursor.rowcount
                        
            except Exception as e:
                self.stats['failed_queries'] += 1
                if attempt == self.config.max_retries - 1:
                    log_universal('ERROR', 'DatabasePool', f'Query failed after {self.config.max_retries} attempts: {e}')
                    raise
                time.sleep(0.1 * (attempt + 1))  # Exponential backoff
    
    def execute_batch(self, query: str, params_list: List[tuple]):
        """Execute batch operations efficiently."""
        self.stats['total_queries'] += len(params_list)
        
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.executemany(query, params_list)
                conn.commit()
                return cursor.rowcount
        except Exception as e:
            self.stats['failed_queries'] += len(params_list)
            log_universal('ERROR', 'DatabasePool', f'Batch operation failed: {e}')
            raise
    
    def cleanup_expired_connections(self):
        """Remove expired connections from the pool."""
        with self._lock:
            expired_connections = []
            
            for pc in self._connections[:]:  # Copy list for safe iteration
                if pc.is_expired(self.config.idle_timeout) and not pc.in_use:
                    expired_connections.append(pc)
                    self._connections.remove(pc)
            
            for pc in expired_connections:
                try:
                    pc.connection.close()
                    self.stats['connections_expired'] += 1
                except:
                    pass
            
            if expired_connections:
                log_universal('INFO', 'DatabasePool', f'Cleaned up {len(expired_connections)} expired connections')
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        with self._lock:
            return {
                **self.stats,
                'active_connections': len([pc for pc in self._connections if pc.in_use]),
                'idle_connections': len([pc for pc in self._connections if not pc.in_use]),
                'total_connections': len(self._connections),
                'queue_size': self._connection_queue.qsize()
            }
    
    def close_all(self):
        """Close all connections in the pool."""
        with self._lock:
            while not self._connection_queue.empty():
                try:
                    self._connection_queue.get_nowait()
                except queue.Empty:
                    break
            
            for pc in self._connections:
                try:
                    pc.connection.close()
                except:
                    pass
            
            self._connections.clear()
            log_universal('INFO', 'DatabasePool', 'All connections closed')


# Global pool instance
_pool_instance = None
_pool_lock = threading.Lock()


def get_database_pool(db_path: str, config: PoolConfig = None) -> DatabaseConnectionPool:
    """Get or create the global database pool instance."""
    global _pool_instance
    
    with _pool_lock:
        if _pool_instance is None or _pool_instance.db_path != db_path:
            if _pool_instance:
                _pool_instance.close_all()
            _pool_instance = DatabaseConnectionPool(db_path, config)
        
        return _pool_instance