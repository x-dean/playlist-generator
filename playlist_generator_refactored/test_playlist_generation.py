#!/usr/bin/env python3
"""
Test script for real PlaylistGenerationService implementation.
"""

import sys
import logging
from pathlib import Path
import tempfile
import os
from uuid import uuid4

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from shared.config import get_config
from infrastructure.logging import setup_logging
from application.services.playlist_generation_service import PlaylistGenerationService
from application.dtos.playlist_generation import PlaylistGenerationRequest, PlaylistGenerationMethod
from domain.entities.audio_file import AudioFile
from domain.entities.feature_set import FeatureSet
from domain.entities.metadata import Metadata


def create_mock_audio_files():
    """Create mock audio files with realistic features for testing."""
    audio_files = []
    
    # Create diverse mock tracks
    mock_tracks = [
        {
            'title': 'Creep',
            'artist': 'Radiohead',
            'album': 'Pablo Honey',
            'duration': 240,
            'bpm': 92,
            'energy': 0.6,
            'danceability': 0.4,
            'valence': 0.3,
            'acousticness': 0.7,
            'instrumentalness': 0.1,
            'speechiness': 0.05,
            'liveness': 0.2,
            'loudness': -12.5
        },
        {
            'title': 'Bohemian Rhapsody',
            'artist': 'Queen',
            'album': 'A Night at the Opera',
            'duration': 354,
            'bpm': 72,
            'energy': 0.8,
            'danceability': 0.3,
            'valence': 0.5,
            'acousticness': 0.3,
            'instrumentalness': 0.2,
            'speechiness': 0.1,
            'liveness': 0.4,
            'loudness': -8.2
        },
        {
            'title': 'Billie Jean',
            'artist': 'Michael Jackson',
            'album': 'Thriller',
            'duration': 294,
            'bpm': 117,
            'energy': 0.9,
            'danceability': 0.9,
            'valence': 0.7,
            'acousticness': 0.1,
            'instrumentalness': 0.05,
            'speechiness': 0.08,
            'liveness': 0.3,
            'loudness': -6.8
        },
        {
            'title': 'Hotel California',
            'artist': 'Eagles',
            'album': 'Hotel California',
            'duration': 391,
            'bpm': 75,
            'energy': 0.5,
            'danceability': 0.2,
            'valence': 0.4,
            'acousticness': 0.8,
            'instrumentalness': 0.3,
            'speechiness': 0.03,
            'liveness': 0.1,
            'loudness': -14.2
        },
        {
            'title': 'Stairway to Heaven',
            'artist': 'Led Zeppelin',
            'album': 'Led Zeppelin IV',
            'duration': 482,
            'bpm': 63,
            'energy': 0.7,
            'danceability': 0.1,
            'valence': 0.3,
            'acousticness': 0.6,
            'instrumentalness': 0.4,
            'speechiness': 0.02,
            'liveness': 0.2,
            'loudness': -11.8
        },
        {
            'title': 'Imagine',
            'artist': 'John Lennon',
            'album': 'Imagine',
            'duration': 183,
            'bpm': 76,
            'energy': 0.3,
            'danceability': 0.2,
            'valence': 0.6,
            'acousticness': 0.9,
            'instrumentalness': 0.1,
            'speechiness': 0.04,
            'liveness': 0.1,
            'loudness': -16.5
        },
        {
            'title': 'Smells Like Teen Spirit',
            'artist': 'Nirvana',
            'album': 'Nevermind',
            'duration': 301,
            'bpm': 117,
            'energy': 0.9,
            'danceability': 0.6,
            'valence': 0.2,
            'acousticness': 0.2,
            'instrumentalness': 0.1,
            'speechiness': 0.1,
            'liveness': 0.4,
            'loudness': -7.2
        },
        {
            'title': 'Wonderwall',
            'artist': 'Oasis',
            'album': '(What\'s the Story) Morning Glory?',
            'duration': 258,
            'bpm': 87,
            'energy': 0.6,
            'danceability': 0.5,
            'valence': 0.5,
            'acousticness': 0.5,
            'instrumentalness': 0.1,
            'speechiness': 0.06,
            'liveness': 0.2,
            'loudness': -10.1
        }
    ]
    
    for i, track in enumerate(mock_tracks):
        # Create feature set
        feature_set = FeatureSet(
            audio_file_id=uuid4(),
            bpm=track['bpm'],
            energy=track['energy'],
            danceability=track['danceability'],
            valence=track['valence'],
            acousticness=track['acousticness'],
            instrumentalness=track['instrumentalness'],
            speechiness=track['speechiness'],
            liveness=track['liveness'],
            loudness=track['loudness']
        )
        
        # Create metadata
        metadata = Metadata(
            audio_file_id=str(uuid4()),
            title=track['title'],
            artist=track['artist'],
            album=track['album']
        )
        
        # Create audio file
        audio_file = AudioFile(
            file_path=Path(f"/music/{track['title'].replace(' ', '_')}.mp3"),
            id=uuid4(),
            file_size_bytes=1024 * 1024 * 5,  # 5MB
            duration_seconds=track['duration'],
            bitrate_kbps=320,
            sample_rate_hz=44100,
            channels=2
        )
        
        # Add feature set and metadata to external metadata
        audio_file.external_metadata['feature_set'] = feature_set
        audio_file.external_metadata['metadata'] = metadata
        
        audio_files.append(audio_file)
    
    return audio_files


def test_playlist_generation():
    """Test the real PlaylistGenerationService implementation."""
    
    print("🧪 Starting playlist generation test...")
    
    # Setup logging
    config = get_config()
    setup_logging(config)
    logger = logging.getLogger(__name__)
    
    logger.info("🧪 Testing real PlaylistGenerationService implementation...")
    
    try:
        # Initialize service
        logger.info("📦 Initializing PlaylistGenerationService...")
        playlist_service = PlaylistGenerationService()
        
        logger.info("✅ Service initialized successfully")
        
        # Create mock audio files
        logger.info("🎵 Creating mock audio files...")
        audio_files = create_mock_audio_files()
        print(f"🎵 Created {len(audio_files)} mock audio files")
        
        # Test different generation methods
        methods = [
            PlaylistGenerationMethod.KMEANS,
            PlaylistGenerationMethod.SIMILARITY,
            PlaylistGenerationMethod.FEATURE_BASED,
            PlaylistGenerationMethod.RANDOM,
            PlaylistGenerationMethod.TIME_BASED
        ]
        
        for method in methods:
            print(f"\n🎵 Testing {method.value} playlist generation...")
            
            # Create generation request
            request = PlaylistGenerationRequest(
                audio_files=audio_files,
                method=method,
                playlist_size=5
            )
            
            # Generate playlist
            response = playlist_service.generate_playlist(request)
            
            print(f"📊 Status: {response.status}")
            print(f"📊 Processing time: {response.processing_time_ms}ms")
            print(f"📊 Method used: {response.method_used}")
            
            if response.playlist:
                playlist = response.playlist
                print(f"🎵 Playlist: {playlist.name}")
                print(f"🎵 Description: {playlist.description}")
                print(f"🎵 Tracks: {len(playlist.track_ids)}")
                
                # Show track details
                for i, track_id in enumerate(playlist.track_ids[:3]):  # Show first 3 tracks
                    # Find the corresponding audio file
                    for audio_file in audio_files:
                        if audio_file.id == track_id:
                            metadata = audio_file.external_metadata.get('metadata', {})
                            artist = getattr(metadata, 'artist', 'Unknown')
                            title = getattr(metadata, 'title', 'Unknown')
                            print(f"  {i+1}. {artist} - {title}")
                            break
                
                if response.quality_metrics:
                    metrics = response.quality_metrics
                    print(f"📊 Quality Metrics:")
                    print(f"  - Diversity: {metrics.diversity_score:.3f}")
                    print(f"  - Coherence: {metrics.coherence_score:.3f}")
                    print(f"  - Balance: {metrics.balance_score:.3f}")
                    print(f"  - Overall: {metrics.overall_score:.3f}")
            
            if response.error_message:
                print(f"❌ Error: {response.error_message}")
        
        logger.info("🎉 Playlist generation test completed successfully!")
        print("🎉 Test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_playlist_generation()
    sys.exit(0 if success else 1) 