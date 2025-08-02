"""
API layer for Playlist Generator.
REST API implementation using FastAPI.
"""

from .routes import router
from .models import *

__all__ = [
    'router'
] 