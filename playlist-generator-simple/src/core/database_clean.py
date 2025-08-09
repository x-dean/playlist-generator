"""
Clean Database Interface - PostgreSQL Only.

Single, clean database interface without fallback complexity.
All applications use PostgreSQL for consistent, web-ready architecture.
"""

from .postgresql_manager import PostgreSQLManager, get_postgresql_manager
from .logging_setup import log_universal

# Clean aliases - no more confusion
DatabaseManager = PostgreSQLManager
get_db_manager = get_postgresql_manager
get_database_manager = get_postgresql_manager

# Log the clean architecture choice
log_universal('INFO', 'Database', 'Using clean PostgreSQL-only architecture')

__all__ = [
    'DatabaseManager',
    'get_db_manager', 
    'get_database_manager',
    'PostgreSQLManager',
    'get_postgresql_manager'
]
