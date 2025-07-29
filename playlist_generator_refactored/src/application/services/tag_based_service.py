"""
Tag-based playlist generation service.
Ports the original TagBasedPlaylistGenerator to the new architecture.
"""

import logging
from typing import List, Dict, Any, Optional
from collections import defaultdict, Counter
from datetime import datetime

from domain.entities.audio_file import AudioFile
from domain.entities.playlist import Playlist
from infrastructure.external_apis.musicbrainz_client import MusicBrainzClient
from infrastructure.external_apis.lastfm_client import LastFMClient
from shared.exceptions import PlaylistGenerationError

class TagBasedService:
    """Service for tag-based playlist generation using genre, decade, and mood."""
    
    def __init__(self, min_tracks_per_genre: int = 10, min_subgroup_size: int = 10):
        self.min_tracks_per_genre = min_tracks_per_genre
        self.min_subgroup_size = min_subgroup_size
        self.logger = logging.getLogger(__name__)
        
        # Initialize external API clients
        self.musicbrainz_client = MusicBrainzClient()
        self.lastfm_client = LastFMClient()
    
    def generate_tag_based_playlists(self, audio_files: List[AudioFile]) -> List[Playlist]:
        """Generate playlists based on tags (genre, decade, mood)."""
        self.logger.info(f"Generating tag-based playlists from {len(audio_files)} tracks")
        
        try:
            # Enrich metadata for all tracks
            enriched_files = self._enrich_metadata(audio_files)
            
            # Group by genre
            genre_playlists = self._group_by_genre(enriched_files)
            
            # Group by decade
            decade_playlists = self._group_by_decade(enriched_files)
            
            # Group by mood
            mood_playlists = self._group_by_mood(enriched_files)
            
            # Combine all playlists
            all_playlists = genre_playlists + decade_playlists + mood_playlists
            
            self.logger.info(f"Generated {len(all_playlists)} tag-based playlists")
            return all_playlists
            
        except Exception as e:
            self.logger.error(f"Tag-based playlist generation failed: {e}")
            raise PlaylistGenerationError(f"Tag-based generation failed: {e}")
    
    def _enrich_metadata(self, audio_files: List[AudioFile]) -> List[AudioFile]:
        """Enrich metadata using external APIs."""
        enriched_files = []
        
        for audio_file in audio_files:
            try:
                # Get basic metadata
                metadata = audio_file.external_metadata.get('metadata', {})
                artist = metadata.get('artist')
                title = metadata.get('title')
                
                if not artist or not title:
                    continue
                
                # Try MusicBrainz enrichment
                musicbrainz_track = self.musicbrainz_client.search_track(title, artist)
                if musicbrainz_track:
                    # Extract year from release date
                    if musicbrainz_track.release_date:
                        metadata['year'] = musicbrainz_track.release_date[:4]
                    
                    # Extract genres from tags
                    if musicbrainz_track.tags:
                        metadata['genre'] = musicbrainz_track.tags
                    
                    # Extract album information
                    if musicbrainz_track.album and musicbrainz_track.album != "Unknown":
                        metadata['album'] = musicbrainz_track.album
                
                # Try Last.fm enrichment if needed
                if not metadata.get('genre'):
                    lastfm_track = self.lastfm_client.get_track_info(title, artist)
                    if lastfm_track and lastfm_track.tags:
                        metadata['genre'] = lastfm_track.tags
                
                # Update audio file metadata
                audio_file.external_metadata['metadata'] = metadata
                enriched_files.append(audio_file)
                
            except Exception as e:
                self.logger.warning(f"Failed to enrich metadata for {audio_file.file_path}: {e}")
                enriched_files.append(audio_file)
        
        return enriched_files
    
    def _group_by_genre(self, audio_files: List[AudioFile]) -> List[Playlist]:
        """Group tracks by genre and create playlists."""
        genre_groups = defaultdict(list)
        
        for audio_file in audio_files:
            metadata = audio_file.external_metadata.get('metadata', {})
            genres = metadata.get('genre', [])
            
            if not genres:
                continue
            
            # Use primary genre
            primary_genre = genres[0] if isinstance(genres, list) else genres
            normalized_genre = self._normalize_genre(primary_genre)
            
            if normalized_genre:
                genre_groups[normalized_genre].append(audio_file)
        
        # Create playlists for genres with enough tracks
        playlists = []
        for genre, tracks in genre_groups.items():
            if len(tracks) >= self.min_tracks_per_genre:
                playlist = self._create_genre_playlist(genre, tracks)
                playlists.append(playlist)
        
        return playlists
    
    def _group_by_decade(self, audio_files: List[AudioFile]) -> List[Playlist]:
        """Group tracks by decade and create playlists."""
        decade_groups = defaultdict(list)
        
        for audio_file in audio_files:
            metadata = audio_file.external_metadata.get('metadata', {})
            year = metadata.get('year')
            
            if not year:
                continue
            
            decade = self._get_decade(year)
            if decade:
                decade_groups[decade].append(audio_file)
        
        # Create playlists for decades with enough tracks
        playlists = []
        for decade, tracks in decade_groups.items():
            if len(tracks) >= self.min_subgroup_size:
                playlist = self._create_decade_playlist(decade, tracks)
                playlists.append(playlist)
        
        return playlists
    
    def _group_by_mood(self, audio_files: List[AudioFile]) -> List[Playlist]:
        """Group tracks by mood and create playlists."""
        mood_groups = defaultdict(list)
        
        for audio_file in audio_files:
            mood = self._get_mood(audio_file)
            if mood:
                mood_groups[mood].append(audio_file)
        
        # Create playlists for moods with enough tracks
        playlists = []
        for mood, tracks in mood_groups.items():
            if len(tracks) >= self.min_subgroup_size:
                playlist = self._create_mood_playlist(mood, tracks)
                playlists.append(playlist)
        
        return playlists
    
    def _normalize_genre(self, genre: str) -> str:
        """Normalize genre name."""
        if not genre:
            return ""
        
        # Replace underscores/hyphens with spaces, title case, strip
        normalized = genre.replace('_', ' ').replace('-', ' ').title().strip()
        return normalized
    
    def _get_decade(self, year: str) -> str:
        """Get decade from year."""
        try:
            year_int = int(year[:4])
            decade_start = (year_int // 10) * 10
            return f"{decade_start}s"
        except (ValueError, TypeError):
            return ""
    
    def _get_mood(self, audio_file: AudioFile) -> str:
        """Get mood based on audio features."""
        features = audio_file.external_metadata.get('features', {})
        
        bpm = features.get('bpm', 0)
        danceability = features.get('danceability', 0)
        centroid = features.get('centroid', 0)
        
        # Simple mood classification
        if bpm > 140 and danceability > 0.7:
            return "Energetic"
        elif bpm < 80 and danceability < 0.3:
            return "Chill"
        elif centroid > 3000:
            return "Bright"
        elif centroid < 1000:
            return "Dark"
        else:
            return "Balanced"
    
    def _create_genre_playlist(self, genre: str, tracks: List[AudioFile]) -> Playlist:
        """Create a genre-based playlist."""
        track_ids = [track.id for track in tracks]
        track_paths = [str(track.file_path) for track in tracks]
        
        return Playlist(
            name=f"{genre} Mix",
            description=f"Genre-based playlist featuring {genre} music",
            track_ids=track_ids,
            track_paths=track_paths,
            generation_method="tag_based",
            playlist_type="genre",
            genres=[genre],
            created_date=datetime.now()
        )
    
    def _create_decade_playlist(self, decade: str, tracks: List[AudioFile]) -> Playlist:
        """Create a decade-based playlist."""
        track_ids = [track.id for track in tracks]
        track_paths = [str(track.file_path) for track in tracks]
        
        return Playlist(
            name=f"{decade} Hits",
            description=f"Decade-based playlist featuring {decade} music",
            track_ids=track_ids,
            track_paths=track_paths,
            generation_method="tag_based",
            playlist_type="decade",
            decades=[decade],
            created_date=datetime.now()
        )
    
    def _create_mood_playlist(self, mood: str, tracks: List[AudioFile]) -> Playlist:
        """Create a mood-based playlist."""
        track_ids = [track.id for track in tracks]
        track_paths = [str(track.file_path) for track in tracks]
        
        return Playlist(
            name=f"{mood} Vibes",
            description=f"Mood-based playlist featuring {mood} music",
            track_ids=track_ids,
            track_paths=track_paths,
            generation_method="tag_based",
            playlist_type="mood",
            moods=[mood],
            created_date=datetime.now()
        ) 