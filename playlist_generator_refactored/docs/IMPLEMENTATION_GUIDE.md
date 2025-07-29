# Quick Implementation Guide: Phase 1 - Tag-Based Playlist Generation

## Step 1: Create Tag-Based Service

### File: `src/application/services/tag_based_service.py`

```python
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
                enriched_metadata = self.musicbrainz_client.enrich_track_metadata(
                    artist, title, metadata
                )
                
                # Try Last.fm enrichment if needed
                if not enriched_metadata.get('genre'):
                    lastfm_metadata = self.lastfm_client.enrich_track_metadata(
                        artist, title, enriched_metadata
                    )
                    enriched_metadata.update(lastfm_metadata)
                
                # Update audio file metadata
                audio_file.external_metadata['metadata'] = enriched_metadata
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
```

## Step 2: Update Playlist Generation Service

### Add to `src/application/services/playlist_generation_service.py`

```python
# Add import at the top
from .tag_based_service import TagBasedService

# Add to __init__ method
def __init__(self):
    # ... existing code ...
    self.tag_based_service = TagBasedService()

# Add new method
def _generate_tag_based_playlist(self, request: PlaylistGenerationRequest) -> Playlist:
    """Generate playlist using tag-based selection."""
    self.logger.info("Generating playlist using tag-based selection")
    
    try:
        # Use tag-based service to generate playlists
        playlists = self.tag_based_service.generate_tag_based_playlists(request.audio_files)
        
        if not playlists:
            self.logger.warning("No tag-based playlists generated, using random")
            return self._generate_random_playlist(request)
        
        # Select the best playlist or combine multiple
        selected_playlist = playlists[0]  # Simple selection for now
        
        # Limit to requested size
        if request.playlist_size and len(selected_playlist.track_ids) > request.playlist_size:
            selected_playlist.track_ids = selected_playlist.track_ids[:request.playlist_size]
            selected_playlist.track_paths = selected_playlist.track_paths[:request.playlist_size]
        
        return selected_playlist
        
    except Exception as e:
        self.logger.error(f"Tag-based generation failed: {e}")
        return self._generate_random_playlist(request)

# Update the generate_playlist method to include tag-based
elif request.method == PlaylistGenerationMethod.TAG_BASED:
    playlist = self._generate_tag_based_playlist(request)
```

## Step 3: Update Configuration

### Add to `src/shared/config/settings.py`

```python
@dataclass
class TagBasedConfig:
    """Configuration for tag-based playlist generation."""
    min_tracks_per_genre: int = 10
    min_subgroup_size: int = 10
    large_group_threshold: int = 40
    
    # External API settings
    musicbrainz_user_agent: str = "Playlista/1.0"
    lastfm_api_key: str = field(default_factory=lambda: os.getenv('LASTFM_API_KEY', ''))

# Add to AppConfig
tag_based: TagBasedConfig = field(default_factory=TagBasedConfig)
```

## Step 4: Testing

### Create test file: `test_tag_based_generation.py`

```python
#!/usr/bin/env python3
"""Test tag-based playlist generation."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from application.services.playlist_generation_service import PlaylistGenerationService
from application.dtos.playlist_generation import PlaylistGenerationRequest, PlaylistGenerationMethod
from domain.entities.audio_file import AudioFile

def test_tag_based_generation():
    """Test tag-based playlist generation."""
    print("🧪 Testing tag-based playlist generation...")
    
    try:
        # Create service
        service = PlaylistGenerationService()
        
        # Create mock audio files with metadata
        audio_files = []
        for i in range(10):
            audio_file = AudioFile(file_path=Path(f"/music/audio/test{i}.mp3"))
            audio_file.external_metadata = {
                'metadata': {
                    'artist': f'Artist {i}',
                    'title': f'Track {i}',
                    'genre': ['Rock'] if i < 5 else ['Pop'],
                    'year': f'199{i % 10}'
                },
                'features': {
                    'bpm': 120 + (i * 10),
                    'danceability': 0.5 + (i * 0.1),
                    'centroid': 2000 + (i * 500)
                }
            }
            audio_files.append(audio_file)
        
        # Create request
        request = PlaylistGenerationRequest(
            audio_files=audio_files,
            method=PlaylistGenerationMethod.TAG_BASED,
            playlist_size=5
        )
        
        # Generate playlist
        response = service.generate_playlist(request)
        
        if response.status == "completed":
            print(f"✅ Tag-based playlist generated successfully!")
            print(f"   - Name: {response.playlist.name}")
            print(f"   - Size: {len(response.playlist.track_ids)}")
            print(f"   - Method: {response.playlist.generation_method}")
            return True
        else:
            print(f"❌ Tag-based playlist generation failed: {response.status}")
            return False
            
    except Exception as e:
        print(f"❌ Tag-based playlist generation failed: {e}")
        return False

if __name__ == "__main__":
    success = test_tag_based_generation()
    sys.exit(0 if success else 1)
```

## Step 5: Run Test

```bash
docker-compose -f docker-compose.real-test.yaml run --rm playlista-real-test python test_tag_based_generation.py
```

## Next Steps

1. **Implement Phase 2**: Cache-based playlist generation
2. **Add more sophisticated genre/mood detection**
3. **Implement playlist merging for large collections**
4. **Add configuration for minimum thresholds**
5. **Create comprehensive unit tests**

This implementation provides a solid foundation for tag-based playlist generation that can be extended with more sophisticated features from the original implementation. 