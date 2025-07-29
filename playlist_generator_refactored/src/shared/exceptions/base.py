"""
Base exception classes for the Playlista application.
"""

from typing import Optional, Any, Dict


class PlaylistaException(Exception):
    """Base exception for all Playlista application errors."""
    
    def __init__(
        self,
        message: str,
        details: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None
    ):
        super().__init__(message)
        self.message = message
        self.details = details
        self.context = context or {}
        self.original_exception = original_exception
    
    def __str__(self) -> str:
        """Return a formatted error message."""
        if self.details:
            return f"{self.message}: {self.details}"
        return self.message
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/serialization."""
        return {
            'type': self.__class__.__name__,
            'message': self.message,
            'details': self.details,
            'context': self.context,
            'original_exception': str(self.original_exception) if self.original_exception else None
        }


class ConfigurationError(PlaylistaException):
    """Raised when there's an error with application configuration."""
    
    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        config_value: Optional[Any] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.config_key = config_key
        self.config_value = config_value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/serialization."""
        base_dict = super().to_dict()
        base_dict.update({
            'config_key': self.config_key,
            'config_value': str(self.config_value) if self.config_value is not None else None
        })
        return base_dict


class ValidationError(PlaylistaException):
    """Raised when data validation fails."""
    
    def __init__(
        self,
        message: str,
        field_name: Optional[str] = None,
        field_value: Optional[Any] = None,
        validation_rule: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.field_name = field_name
        self.field_value = field_value
        self.validation_rule = validation_rule
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/serialization."""
        base_dict = super().to_dict()
        base_dict.update({
            'field_name': self.field_name,
            'field_value': str(self.field_value) if self.field_value is not None else None,
            'validation_rule': self.validation_rule
        })
        return base_dict 