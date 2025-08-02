"""
Domain layer for Playlist Generator.
Contains core business entities, value objects, and domain interfaces.
"""

from .entities import Track, TrackMetadata, AnalysisResult, Playlist
from .interfaces import ITrackRepository, IAnalysisRepository, IPlaylistRepository, IAudioAnalyzer
from .exceptions import DomainException, TrackNotFoundException, AnalysisFailedException, PlaylistGenerationException

__all__ = [
    'Track',
    'TrackMetadata', 
    'AnalysisResult',
    'Playlist',
    'ITrackRepository',
    'IAnalysisRepository',
    'IPlaylistRepository',
    'IAudioAnalyzer',
    'DomainException',
    'TrackNotFoundException',
    'AnalysisFailedException',
    'PlaylistGenerationException'
] 