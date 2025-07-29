"""
Database-related exceptions.
"""

from .base import PlaylistaException
from typing import Optional, Any, Dict


class DatabaseError(PlaylistaException):
    """Base exception for database errors."""
    
    def __init__(
        self,
        message: str,
        database_path: Optional[str] = None,
        table_name: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.database_path = database_path
        self.table_name = table_name
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/serialization."""
        base_dict = super().to_dict()
        base_dict.update({
            'database_path': self.database_path,
            'table_name': self.table_name
        })
        return base_dict


class DatabaseConnectionError(DatabaseError):
    """Raised when database connection fails."""
    
    def __init__(
        self,
        message: str,
        connection_string: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.connection_string = connection_string
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/serialization."""
        base_dict = super().to_dict()
        base_dict.update({
            'connection_string': self.connection_string
        })
        return base_dict


class DatabaseQueryError(DatabaseError):
    """Raised when a database query fails."""
    
    def __init__(
        self,
        message: str,
        query: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.query = query
        self.parameters = parameters
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/serialization."""
        base_dict = super().to_dict()
        base_dict.update({
            'query': self.query,
            'parameters': self.parameters
        })
        return base_dict


class DatabaseMigrationError(DatabaseError):
    """Raised when database migration fails."""
    
    def __init__(
        self,
        message: str,
        migration_version: Optional[str] = None,
        migration_script: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.migration_version = migration_version
        self.migration_script = migration_script
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/serialization."""
        base_dict = super().to_dict()
        base_dict.update({
            'migration_version': self.migration_version,
            'migration_script': self.migration_script
        })
        return base_dict 