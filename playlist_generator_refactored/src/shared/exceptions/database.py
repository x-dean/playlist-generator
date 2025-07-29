"""
Database related exceptions.
"""

from .base import PlaylistaException
from typing import Optional, Any, Dict


class DatabaseError(PlaylistaException):
    """Base exception for database operations."""
    pass


class DatabaseConnectionError(DatabaseError):
    """Exception raised when database connection fails."""
    
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
    """Exception raised when database query fails."""
    
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
    """Exception raised when database migration fails."""
    
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


class EntityNotFoundError(DatabaseError):
    """Exception raised when an entity is not found in the database."""
    pass 