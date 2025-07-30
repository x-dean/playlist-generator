#!/usr/bin/env python3
"""
Test script for playlist generation functionality.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from core.playlist_generator import PlaylistGenerator, PlaylistGenerationMethod
from core.database import DatabaseManager
from core.logging_setup import get_logger

logger = get_logger('test.playlist_generation')


def test_playlist_generation():
    """Test playlist generation functionality."""
    logger.info("🧪 Testing playlist generation")
    
    try:
        # Create playlist generator
        playlist_generator = PlaylistGenerator()
        
        # Test with different methods
        methods = ['random', 'kmeans', 'similarity', 'time_based']
        
        for method in methods:
            logger.info(f"🎵 Testing {method} playlist generation")
            
            try:
                # Generate playlists
                playlists = playlist_generator.generate_playlists(
                    method=method,
                    num_playlists=2,
                    playlist_size=10
                )
                
                if playlists:
                    logger.info(f"✅ {method}: Generated {len(playlists)} playlists")
                    for name, playlist in playlists.items():
                        logger.info(f"   {name}: {playlist.size} tracks")
                else:
                    logger.warning(f"⚠️ {method}: No playlists generated")
                    
            except Exception as e:
                logger.error(f"❌ {method}: Failed - {e}")
        
        # Test playlist saving
        logger.info("💾 Testing playlist saving")
        test_playlists = {
            'Test_Playlist_1': playlist_generator.random_generator.generate([], 1, 5)[0],
            'Test_Playlist_2': playlist_generator.random_generator.generate([], 1, 5)[0]
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            success = playlist_generator.save_playlists(test_playlists, temp_dir)
            if success:
                logger.info(f"✅ Playlists saved to {temp_dir}")
                
                # Check if files were created
                files = list(Path(temp_dir).glob('*.m3u'))
                logger.info(f"📁 Created {len(files)} playlist files")
            else:
                logger.error("❌ Failed to save playlists")
        
        logger.info("✅ Playlist generation tests completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Playlist generation test failed: {e}")
        return False


def test_database_integration():
    """Test database integration for playlist generation."""
    logger.info("🗄️ Testing database integration")
    
    try:
        # Create database manager
        db_manager = DatabaseManager()
        
        # Test getting analyzed tracks
        tracks = db_manager.get_analyzed_tracks()
        logger.info(f"📁 Found {len(tracks)} analyzed tracks in database")
        
        # Test getting cached playlists
        cached_playlists = db_manager.get_cached_playlists()
        logger.info(f"📁 Found {len(cached_playlists)} cached playlists")
        
        # Test getting track features
        if tracks:
            first_track = tracks[0]
            features = db_manager.get_track_features(first_track['filepath'])
            if features:
                logger.info(f"✅ Retrieved features for {first_track['filename']}")
            else:
                logger.warning(f"⚠️ No features found for {first_track['filename']}")
        
        logger.info("✅ Database integration tests completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Database integration test failed: {e}")
        return False


def main():
    """Main test function."""
    logger.info("🧪 Starting playlist generation tests")
    
    # Test database integration
    db_success = test_database_integration()
    
    # Test playlist generation
    playlist_success = test_playlist_generation()
    
    if db_success and playlist_success:
        logger.info("✅ All tests passed!")
        return 0
    else:
        logger.error("❌ Some tests failed!")
        return 1


if __name__ == "__main__":
    exit(main()) 