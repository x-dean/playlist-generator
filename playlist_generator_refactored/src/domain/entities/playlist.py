"""
Playlist entity representing a generated playlist.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List
from uuid import UUID, uuid4

from shared.exceptions import PlaylistValidationError


@dataclass
class Playlist:
    """
    Represents a generated playlist in the music analysis domain.
    
    This entity encapsulates a collection of tracks with their features,
    metadata, and playlist-specific information.
    """
    
    # Core identification
    name: str = field()
    id: UUID = field(default_factory=uuid4)
    description: Optional[str] = None
    
    # Track information
    track_ids: List[UUID] = field(default_factory=list)
    track_paths: List[str] = field(default_factory=list)
    
    # Playlist characteristics
    playlist_type: str = "mixed"  # kmeans, time_based, tag_based, feature_group, cache, advanced
    generation_method: str = "kmeans"
    target_size: Optional[int] = None
    actual_size: int = field(init=False)
    
    # Musical characteristics (aggregated from tracks)
    average_bpm: Optional[float] = None
    dominant_key: Optional[str] = None
    average_energy: Optional[float] = None
    average_danceability: Optional[float] = None
    average_valence: Optional[float] = None
    
    # Genre and mood information
    genres: List[str] = field(default_factory=list)
    moods: List[str] = field(default_factory=list)
    decades: List[str] = field(default_factory=list)
    
    # Quality metrics
    coherence_score: Optional[float] = None  # 0.0 to 1.0
    diversity_score: Optional[float] = None  # 0.0 to 1.0
    overall_quality: Optional[float] = None  # 0.0 to 1.0
    
    # Generation metadata
    created_date: datetime = field(default_factory=datetime.now)
    last_modified: datetime = field(default_factory=datetime.now)
    generation_time_ms: Optional[float] = None
    generation_parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Usage statistics
    play_count: int = 0
    last_played: Optional[datetime] = None
    user_rating: Optional[float] = None  # 1.0 to 5.0
    user_notes: Optional[str] = None
    
    # Export information
    export_formats: List[str] = field(default_factory=list)  # m3u, pls, xspf, etc.
    export_path: Optional[str] = None
    
    def __post_init__(self):
        """Validate playlist after initialization."""
        if not self.name:
            raise PlaylistValidationError("Playlist name is required")
        
        # Set actual size
        self.actual_size = len(self.track_ids)
        
        # Validate target size if provided
        if self.target_size is not None and self.target_size < 1:
            raise PlaylistValidationError(f"Target size must be at least 1, got {self.target_size}")
        
        # Validate user rating if provided
        if self.user_rating is not None and not (1.0 <= self.user_rating <= 5.0):
            raise PlaylistValidationError(f"User rating must be between 1.0 and 5.0, got {self.user_rating}")
        
        # Validate quality scores if provided
        for score_name, score in [('coherence_score', self.coherence_score),
                                ('diversity_score', self.diversity_score),
                                ('overall_quality', self.overall_quality)]:
            if score is not None and not (0.0 <= score <= 1.0):
                raise PlaylistValidationError(f"{score_name} must be between 0.0 and 1.0, got {score}")
    
    @property
    def is_empty(self) -> bool:
        """Check if playlist is empty."""
        return self.actual_size == 0
    
    @property
    def is_full(self) -> bool:
        """Check if playlist has reached target size."""
        if self.target_size is None:
            return False
        return self.actual_size >= self.target_size
    
    @property
    def has_target_size(self) -> bool:
        """Check if playlist has a target size."""
        return self.target_size is not None
    
    @property
    def remaining_capacity(self) -> int:
        """Get remaining capacity."""
        if self.target_size is None:
            return float('inf')
        return max(0, self.target_size - self.actual_size)
    
    @property
    def duration_minutes(self) -> Optional[float]:
        """Get estimated duration in minutes."""
        # This would need to be calculated from track durations
        # For now, return None
        return None
    
    @property
    def dominant_genre(self) -> Optional[str]:
        """Get the most common genre."""
        if not self.genres:
            return None
        from collections import Counter
        genre_counts = Counter(self.genres)
        return genre_counts.most_common(1)[0][0]
    
    @property
    def dominant_mood(self) -> Optional[str]:
        """Get the most common mood."""
        if not self.moods:
            return None
        from collections import Counter
        mood_counts = Counter(self.moods)
        return mood_counts.most_common(1)[0][0]
    
    @property
    def dominant_decade(self) -> Optional[str]:
        """Get the most common decade."""
        if not self.decades:
            return None
        from collections import Counter
        decade_counts = Counter(self.decades)
        return decade_counts.most_common(1)[0][0]
    
    @property
    def tempo_category(self) -> Optional[str]:
        """Get tempo category based on average BPM."""
        if self.average_bpm is None:
            return None
        
        if self.average_bpm < 60:
            return "largo"
        elif self.average_bpm < 76:
            return "adagio"
        elif self.average_bpm < 108:
            return "andante"
        elif self.average_bpm < 168:
            return "allegro"
        else:
            return "presto"
    
    @property
    def energy_category(self) -> Optional[str]:
        """Get energy category."""
        if self.average_energy is None:
            return None
        
        if self.average_energy < 0.3:
            return "low"
        elif self.average_energy < 0.7:
            return "medium"
        else:
            return "high"
    
    def add_track(self, track_id: UUID, track_path: str) -> bool:
        """Add a track to the playlist."""
        if self.is_full:
            return False
        
        if track_id not in self.track_ids:
            self.track_ids.append(track_id)
            self.track_paths.append(track_path)
            self.actual_size = len(self.track_ids)
            self.last_modified = datetime.now()
            return True
        
        return False
    
    def remove_track(self, track_id: UUID) -> bool:
        """Remove a track from the playlist."""
        if track_id in self.track_ids:
            index = self.track_ids.index(track_id)
            self.track_ids.pop(index)
            self.track_paths.pop(index)
            self.actual_size = len(self.track_ids)
            self.last_modified = datetime.now()
            return True
        
        return False
    
    def clear_tracks(self) -> None:
        """Clear all tracks from the playlist."""
        self.track_ids.clear()
        self.track_paths.clear()
        self.actual_size = 0
        self.last_modified = datetime.now()
    
    def reorder_tracks(self, new_order: List[UUID]) -> bool:
        """Reorder tracks in the playlist."""
        if set(new_order) != set(self.track_ids):
            return False
        
        # Create new track_paths list based on new order
        new_track_paths = []
        for track_id in new_order:
            if track_id in self.track_ids:
                index = self.track_ids.index(track_id)
                new_track_paths.append(self.track_paths[index])
        
        self.track_ids = new_order
        self.track_paths = new_track_paths
        self.last_modified = datetime.now()
        return True
    
    def increment_play_count(self) -> None:
        """Increment play count and update last played."""
        self.play_count += 1
        self.last_played = datetime.now()
        self.last_modified = datetime.now()
    
    def set_user_rating(self, rating: float) -> None:
        """Set user rating."""
        if not (1.0 <= rating <= 5.0):
            raise PlaylistValidationError(f"Rating must be between 1.0 and 5.0, got {rating}")
        self.user_rating = rating
        self.last_modified = datetime.now()
    
    def add_export_format(self, format_name: str) -> None:
        """Add an export format."""
        if format_name not in self.export_formats:
            self.export_formats.append(format_name)
            self.last_modified = datetime.now()
    
    def calculate_quality_metrics(self, track_features: List[Dict[str, Any]]) -> None:
        """Calculate quality metrics based on track features."""
        if not track_features:
            return
        
        # Calculate coherence (similarity between tracks)
        coherence_scores = []
        for i in range(len(track_features) - 1):
            for j in range(i + 1, len(track_features)):
                # Simple similarity based on BPM and key
                track1, track2 = track_features[i], track_features[j]
                
                bpm_similarity = 0.0
                if track1.get('bpm') and track2.get('bpm'):
                    bpm_diff = abs(track1['bpm'] - track2['bpm'])
                    bpm_similarity = max(0, 1 - (bpm_diff / 60))  # Normalize by 60 BPM
                
                key_similarity = 1.0 if track1.get('key') == track2.get('key') else 0.0
                
                # Average similarity
                similarity = (bpm_similarity + key_similarity) / 2
                coherence_scores.append(similarity)
        
        if coherence_scores:
            self.coherence_score = sum(coherence_scores) / len(coherence_scores)
        
        # Calculate diversity (variety in the playlist)
        unique_genres = len(set(track.get('genre', '') for track in track_features if track.get('genre')))
        unique_artists = len(set(track.get('artist', '') for track in track_features if track.get('artist')))
        
        genre_diversity = min(1.0, unique_genres / max(1, len(track_features)))
        artist_diversity = min(1.0, unique_artists / max(1, len(track_features)))
        
        self.diversity_score = (genre_diversity + artist_diversity) / 2
        
        # Overall quality (balance of coherence and diversity)
        if self.coherence_score is not None and self.diversity_score is not None:
            self.overall_quality = (self.coherence_score + self.diversity_score) / 2
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the playlist."""
        return {
            'id': str(self.id),
            'name': self.name,
            'description': self.description,
            'playlist_type': self.playlist_type,
            'generation_method': self.generation_method,
            'target_size': self.target_size,
            'actual_size': self.actual_size,
            'average_bpm': self.average_bpm,
            'dominant_key': self.dominant_key,
            'average_energy': self.average_energy,
            'dominant_genre': self.dominant_genre,
            'dominant_mood': self.dominant_mood,
            'coherence_score': self.coherence_score,
            'diversity_score': self.diversity_score,
            'overall_quality': self.overall_quality,
            'play_count': self.play_count,
            'user_rating': self.user_rating,
            'created_date': self.created_date.isoformat(),
            'last_modified': self.last_modified.isoformat()
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'id': str(self.id),
            'name': self.name,
            'description': self.description,
            'track_ids': [str(track_id) for track_id in self.track_ids],
            'track_paths': self.track_paths,
            'playlist_type': self.playlist_type,
            'generation_method': self.generation_method,
            'target_size': self.target_size,
            'actual_size': self.actual_size,
            'average_bpm': self.average_bpm,
            'dominant_key': self.dominant_key,
            'average_energy': self.average_energy,
            'average_danceability': self.average_danceability,
            'average_valence': self.average_valence,
            'genres': self.genres,
            'moods': self.moods,
            'decades': self.decades,
            'coherence_score': self.coherence_score,
            'diversity_score': self.diversity_score,
            'overall_quality': self.overall_quality,
            'created_date': self.created_date.isoformat(),
            'last_modified': self.last_modified.isoformat(),
            'generation_time_ms': self.generation_time_ms,
            'generation_parameters': self.generation_parameters,
            'play_count': self.play_count,
            'last_played': self.last_played.isoformat() if self.last_played else None,
            'user_rating': self.user_rating,
            'user_notes': self.user_notes,
            'export_formats': self.export_formats,
            'export_path': self.export_path,
            'is_empty': self.is_empty,
            'is_full': self.is_full,
            'has_target_size': self.has_target_size,
            'remaining_capacity': self.remaining_capacity,
            'dominant_genre': self.dominant_genre,
            'dominant_mood': self.dominant_mood,
            'dominant_decade': self.dominant_decade,
            'tempo_category': self.tempo_category,
            'energy_category': self.energy_category
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Playlist':
        """Create Playlist from dictionary."""
        # Convert string dates back to datetime objects
        for date_field in ['created_date', 'last_modified', 'last_played']:
            if date_field in data and isinstance(data[date_field], str):
                data[date_field] = datetime.fromisoformat(data[date_field])
        
        # Convert string IDs back to UUID objects
        if 'id' in data and isinstance(data['id'], str):
            data['id'] = UUID(data['id'])
        if 'track_ids' in data and isinstance(data['track_ids'], list):
            data['track_ids'] = [UUID(track_id) if isinstance(track_id, str) else track_id 
                               for track_id in data['track_ids']]
        
        return cls(**data)
    
    def __str__(self) -> str:
        """String representation."""
        return f"Playlist(id={self.id}, name={self.name}, size={self.actual_size})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return (f"Playlist(id={self.id}, name={self.name}, type={self.playlist_type}, "
                f"size={self.actual_size}/{self.target_size})") 