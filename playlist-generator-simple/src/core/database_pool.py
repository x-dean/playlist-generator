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
            conn = sqlite3.connect(self.db_path, timeout=self.config.connection_timeout)
            
            # Enable WAL mode for better concurrent access
            if self.config.enable_wal_mode:
                conn.execute("PRAGMA journal_mode=WAL")
            
            # Enable foreign keys
            if self.config.enable_foreign_keys:
                conn.execute("PRAGMA foreign_keys=ON")
            
            # Set cache size for better performance
            conn.execute(f"PRAGMA cache_size={self.config.cache_size}")
            
            # Set synchronous mode
            conn.execute(f"PRAGMA synchronous={self.config.synchronous}")
            
            # Enable memory-mapped I/O for better performance
            conn.execute("PRAGMA mmap_size=268435456")  # 256MB
            
            # Set temp store to memory for better performance
            conn.execute("PRAGMA temp_store=MEMORY")
            
            self.stats['connections_created'] += 1
            return conn
            
        except Exception as e:
            log_universal('ERROR', 'DatabasePool', f'Failed to create connection: {e}')
            return None
    
    @contextmanager
    def get_connection(self) -> ContextManager[sqlite3.Connection]:
        """
        Get a database connection from the pool.
        Automatically returns connection to pool when done.
        """
        connection = None
        try:
            # Try to get existing connection from pool
            try:
                pooled_conn = self._connection_queue.get(timeout=1.0)
                if pooled_conn.is_expired(self.config.idle_timeout):
                    # Connection expired, create new one
                    pooled_conn.connection.close()
                    connection = self._create_connection()
                    if connection:
                        pooled_conn = PooledConnection(connection, time.time())
                        self.stats['connections_expired'] += 1
                    else:
                        raise Exception("Failed to create new connection")
                else:
                    connection = pooled_conn.connection
                    pooled_conn.mark_used()
                    self.stats['pool_hits'] += 1
            except queue.Empty:
                # No available connections, create new one if under limit
                with self._lock:
                    if len(self._connections) < self.config.max_connections:
                        connection = self._create_connection()
                        if connection:
                            pooled_conn = PooledConnection(connection, time.time())
                            self._connections.append(pooled_conn)
                        else:
                            raise Exception("Failed to create new connection")
                    else:
                        # Wait for available connection
                        pooled_conn = self._connection_queue.get(timeout=5.0)
                        connection = pooled_conn.connection
                        pooled_conn.mark_used()
                
                self.stats['pool_misses'] += 1
            
            yield connection
            
        except Exception as e:
            log_universal('ERROR', 'DatabasePool', f'Error getting connection: {e}')
            raise
        finally:
            # Return connection to pool
            if connection:
                try:
                    # Reset connection state
                    connection.rollback()
                    self._connection_queue.put_nowait(PooledConnection(connection, time.time()))
                except queue.Full:
                    # Pool is full, close connection
                    connection.close()
                except Exception as e:
                    log_universal('ERROR', 'DatabasePool', f'Error returning connection to pool: {e}')
                    connection.close()
    
    def execute_query(self, query: str, params: tuple = None, fetch_one: bool = False, fetch_all: bool = True):
        """
        Execute a query with automatic connection management.
        
        Args:
            query: SQL query string
            params: Query parameters
            fetch_one: Return single row
            fetch_all: Return all rows
            
        Returns:
            Query results
        """
        self.stats['total_queries'] += 1
        
        for attempt in range(self.config.max_retries):
            try:
                with self.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(query, params or ())
                    
                    if fetch_one:
                        result = cursor.fetchone()
                    elif fetch_all:
                        result = cursor.fetchall()
                    else:
                        result = None
                    
                    conn.commit()
                    return result
                    
            except Exception as e:
                self.stats['failed_queries'] += 1
                log_universal('ERROR', 'DatabasePool', f'Query failed (attempt {attempt + 1}): {e}')
                
                if attempt == self.config.max_retries - 1:
                    raise
                
                # Exponential backoff
                time.sleep(2 ** attempt)
    
    def execute_batch(self, query: str, params_list: List[tuple]):
        """
        Execute batch of queries with single connection.
        
        Args:
            query: SQL query string
            params_list: List of parameter tuples
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(query, params_list)
            conn.commit()
    
    def cleanup_expired_connections(self):
        """Clean up expired connections from the pool."""
        with self._lock:
            expired_connections = [
                conn for conn in self._connections
                if conn.is_expired(self.config.idle_timeout) and not conn.in_use
            ]
            
            for conn in expired_connections:
                try:
                    conn.connection.close()
                    self._connections.remove(conn)
                    self.stats['connections_expired'] += 1
                except Exception as e:
                    log_universal('ERROR', 'DatabasePool', f'Error cleaning up expired connection: {e}')
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        with self._lock:
            return {
                **self.stats,
                'total_connections': len(self._connections),
                'available_connections': self._connection_queue.qsize(),
                'pool_size': self.config.max_connections,
                'min_connections': self.config.min_connections
            }
    
    def close_all(self):
        """Close all connections in the pool."""
        with self._lock:
            for conn in self._connections:
                try:
                    conn.connection.close()
                except Exception as e:
                    log_universal('ERROR', 'DatabasePool', f'Error closing connection: {e}')
            
            self._connections.clear()
            log_universal('INFO', 'DatabasePool', 'All connections closed')


# Global pool instance
_pool_instance = None


def get_database_pool(db_path: str, config: PoolConfig = None) -> DatabaseConnectionPool:
    """Get or create the global database pool instance."""
    global _pool_instance
    
    if _pool_instance is None:
        _pool_instance = DatabaseConnectionPool(db_path, config)
    
    return _pool_instance