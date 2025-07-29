"""
Metadata entity representing music metadata from various sources.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List
from uuid import UUID, uuid4

from shared.exceptions import ValidationError


@dataclass
class Metadata:
    """
    Represents music metadata from various sources (ID3 tags, MusicBrainz, Last.fm, etc.).
    
    This entity encapsulates all metadata information about a music track,
    including artist, album, title, genre, and other descriptive information.
    """
    
    # Core identification
    audio_file_id: UUID = field()
    id: UUID = field(default_factory=uuid4)
    
    # Basic track information
    title: Optional[str] = None
    artist: Optional[str] = None
    album: Optional[str] = None
    album_artist: Optional[str] = None
    track_number: Optional[int] = None
    total_tracks: Optional[int] = None
    disc_number: Optional[int] = None
    total_discs: Optional[int] = None
    year: Optional[int] = None
    genre: Optional[str] = None
    composer: Optional[str] = None
    conductor: Optional[str] = None
    performer: Optional[str] = None
    
    # Extended metadata
    lyrics: Optional[str] = None
    comment: Optional[str] = None
    language: Optional[str] = None
    copyright: Optional[str] = None
    publisher: Optional[str] = None
    isrc: Optional[str] = None  # International Standard Recording Code
    barcode: Optional[str] = None
    
    # MusicBrainz IDs
    musicbrainz_track_id: Optional[str] = None
    musicbrainz_artist_id: Optional[str] = None
    musicbrainz_album_id: Optional[str] = None
    musicbrainz_release_group_id: Optional[str] = None
    
    # Last.fm data
    lastfm_tags: List[str] = field(default_factory=list)
    lastfm_playcount: Optional[int] = None
    lastfm_rating: Optional[float] = None
    
    # Discogs data
    discogs_artist_id: Optional[str] = None
    discogs_release_id: Optional[str] = None
    discogs_master_id: Optional[str] = None
    
    # Spotify data (if available)
    spotify_track_id: Optional[str] = None
    spotify_artist_id: Optional[str] = None
    spotify_album_id: Optional[str] = None
    spotify_popularity: Optional[int] = None
    
    # Custom tags and ratings
    custom_tags: List[str] = field(default_factory=list)
    user_rating: Optional[float] = None  # 1.0 to 5.0
    play_count: int = 0
    last_played: Optional[datetime] = None
    date_added: datetime = field(default_factory=datetime.now)
    
    # Source information
    source: str = "id3"  # id3, musicbrainz, lastfm, discogs, spotify
    confidence: Optional[float] = None  # 0.0 to 1.0
    last_updated: datetime = field(default_factory=datetime.now)
    
    # Additional metadata
    additional_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate metadata after initialization."""
        if not self.audio_file_id:
            raise ValidationError("Audio file ID is required")
        
        # Validate year if provided
        if self.year is not None and (self.year < 1900 or self.year > datetime.now().year + 1):
            raise ValidationError(f"Invalid year: {self.year}")
        
        # Validate track number if provided
        if self.track_number is not None and self.track_number < 1:
            raise ValidationError(f"Invalid track number: {self.track_number}")
        
        # Validate user rating if provided
        if self.user_rating is not None and not (1.0 <= self.user_rating <= 5.0):
            raise ValidationError(f"User rating must be between 1.0 and 5.0, got {self.user_rating}")
        
        # Validate confidence if provided
        if self.confidence is not None and not (0.0 <= self.confidence <= 1.0):
            raise ValidationError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")
    
    @property
    def display_title(self) -> str:
        """Get display title (title or filename)."""
        if self.title:
            return self.title
        return "Unknown Title"
    
    @property
    def display_artist(self) -> str:
        """Get display artist (artist or album_artist)."""
        if self.artist:
            return self.artist
        elif self.album_artist:
            return self.album_artist
        return "Unknown Artist"
    
    @property
    def display_album(self) -> str:
        """Get display album."""
        if self.album:
            return self.album
        return "Unknown Album"
    
    @property
    def full_title(self) -> str:
        """Get full title with artist."""
        artist = self.display_artist
        title = self.display_title
        return f"{artist} - {title}"
    
    @property
    def decade(self) -> Optional[str]:
        """Get decade from year."""
        if self.year is None:
            return None
        decade_start = (self.year // 10) * 10
        return f"{decade_start}s"
    
    @property
    def has_lyrics(self) -> bool:
        """Check if track has lyrics."""
        return bool(self.lyrics and self.lyrics.strip())
    
    @property
    def is_compilation(self) -> bool:
        """Check if this is a compilation album."""
        return self.album_artist is not None and self.artist is not None and self.album_artist != self.artist
    
    @property
    def genre_tags(self) -> List[str]:
        """Get all genre-related tags."""
        tags = []
        if self.genre:
            tags.append(self.genre)
        tags.extend(self.lastfm_tags)
        tags.extend(self.custom_tags)
        return list(set(tags))  # Remove duplicates
    
    def add_tag(self, tag: str) -> None:
        """Add a custom tag."""
        if tag and tag not in self.custom_tags:
            self.custom_tags.append(tag)
            self.last_updated = datetime.now()
    
    def remove_tag(self, tag: str) -> None:
        """Remove a custom tag."""
        if tag in self.custom_tags:
            self.custom_tags.remove(tag)
            self.last_updated = datetime.now()
    
    def increment_play_count(self) -> None:
        """Increment play count and update last played."""
        self.play_count += 1
        self.last_played = datetime.now()
        self.last_updated = datetime.now()
    
    def set_user_rating(self, rating: float) -> None:
        """Set user rating."""
        if not (1.0 <= rating <= 5.0):
            raise ValidationError(f"Rating must be between 1.0 and 5.0, got {rating}")
        self.user_rating = rating
        self.last_updated = datetime.now()
    
    def update_from_source(self, source: str, data: Dict[str, Any], confidence: Optional[float] = None) -> None:
        """Update metadata from external source."""
        # Update basic fields if not already set or if new data has higher confidence
        for field, value in data.items():
            if hasattr(self, field) and value is not None:
                current_value = getattr(self, field)
                if current_value is None or (confidence and confidence > (self.confidence or 0)):
                    setattr(self, field, value)
        
        # Update source information
        self.source = source
        if confidence is not None:
            self.confidence = confidence
        self.last_updated = datetime.now()
    
    def merge_with(self, other: 'Metadata') -> None:
        """Merge with another metadata object."""
        # Merge basic fields (prefer non-None values)
        for field in ['title', 'artist', 'album', 'album_artist', 'genre', 'composer']:
            current_value = getattr(self, field)
            other_value = getattr(other, field)
            if current_value is None and other_value is not None:
                setattr(self, field, other_value)
        
        # Merge lists
        self.lastfm_tags = list(set(self.lastfm_tags + other.lastfm_tags))
        self.custom_tags = list(set(self.custom_tags + other.custom_tags))
        
        # Merge additional metadata
        self.additional_metadata.update(other.additional_metadata)
        
        # Update timestamp
        self.last_updated = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'id': str(self.id),
            'audio_file_id': str(self.audio_file_id),
            'title': self.title,
            'artist': self.artist,
            'album': self.album,
            'album_artist': self.album_artist,
            'track_number': self.track_number,
            'total_tracks': self.total_tracks,
            'disc_number': self.disc_number,
            'total_discs': self.total_discs,
            'year': self.year,
            'genre': self.genre,
            'composer': self.composer,
            'conductor': self.conductor,
            'performer': self.performer,
            'lyrics': self.lyrics,
            'comment': self.comment,
            'language': self.language,
            'copyright': self.copyright,
            'publisher': self.publisher,
            'isrc': self.isrc,
            'barcode': self.barcode,
            'musicbrainz_track_id': self.musicbrainz_track_id,
            'musicbrainz_artist_id': self.musicbrainz_artist_id,
            'musicbrainz_album_id': self.musicbrainz_album_id,
            'musicbrainz_release_group_id': self.musicbrainz_release_group_id,
            'lastfm_tags': self.lastfm_tags,
            'lastfm_playcount': self.lastfm_playcount,
            'lastfm_rating': self.lastfm_rating,
            'discogs_artist_id': self.discogs_artist_id,
            'discogs_release_id': self.discogs_release_id,
            'discogs_master_id': self.discogs_master_id,
            'spotify_track_id': self.spotify_track_id,
            'spotify_artist_id': self.spotify_artist_id,
            'spotify_album_id': self.spotify_album_id,
            'spotify_popularity': self.spotify_popularity,
            'custom_tags': self.custom_tags,
            'user_rating': self.user_rating,
            'play_count': self.play_count,
            'last_played': self.last_played.isoformat() if self.last_played else None,
            'date_added': self.date_added.isoformat(),
            'source': self.source,
            'confidence': self.confidence,
            'last_updated': self.last_updated.isoformat(),
            'additional_metadata': self.additional_metadata,
            'display_title': self.display_title,
            'display_artist': self.display_artist,
            'display_album': self.display_album,
            'full_title': self.full_title,
            'decade': self.decade,
            'has_lyrics': self.has_lyrics,
            'is_compilation': self.is_compilation,
            'genre_tags': self.genre_tags
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Metadata':
        """Create Metadata from dictionary."""
        # Convert string dates back to datetime objects
        for date_field in ['last_played', 'date_added', 'last_updated']:
            if date_field in data and isinstance(data[date_field], str):
                data[date_field] = datetime.fromisoformat(data[date_field])
        
        # Convert string IDs back to UUID objects
        if 'id' in data and isinstance(data['id'], str):
            data['id'] = UUID(data['id'])
        if 'audio_file_id' in data and isinstance(data['audio_file_id'], str):
            data['audio_file_id'] = UUID(data['audio_file_id'])
        
        return cls(**data)
    
    def __str__(self) -> str:
        """String representation."""
        return f"Metadata(id={self.id}, title={self.display_title}, artist={self.display_artist})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return (f"Metadata(id={self.id}, audio_file_id={self.audio_file_id}, "
                f"title={self.title}, artist={self.artist}, album={self.album})") 