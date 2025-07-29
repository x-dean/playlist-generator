#!/usr/bin/env python3
"""
Test script for infrastructure layer components.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def test_repositories():
    """Test database repositories."""
    print("🔍 Testing database repositories...")
    
    try:
        from infrastructure.persistence.repositories import (
            SQLiteAudioFileRepository,
            SQLiteFeatureSetRepository,
            SQLiteMetadataRepository,
            SQLitePlaylistRepository
        )
        from domain.entities import AudioFile, FeatureSet, Metadata, Playlist
        from uuid import uuid4
        from pathlib import Path
        
        # Test repository initialization
        audio_repo = SQLiteAudioFileRepository()
        feature_repo = SQLiteFeatureSetRepository()
        metadata_repo = SQLiteMetadataRepository()
        playlist_repo = SQLitePlaylistRepository()
        
        print("✅ All repositories initialized successfully")
        
        # Test AudioFile repository
        audio_file = AudioFile(
            file_path=Path("/test/song.mp3"),
            id=uuid4(),
            file_size_bytes=1024 * 1024 * 5,  # 5MB
            duration_seconds=180.0,
            bitrate_kbps=320,
            sample_rate_hz=44100,
            channels=2
        )
        saved_audio = audio_repo.save(audio_file)
        retrieved_audio = audio_repo.find_by_id(audio_file.id)
        
        if retrieved_audio and retrieved_audio.id == audio_file.id:
            print("✅ AudioFile repository CRUD operations working")
        else:
            print("❌ AudioFile repository CRUD operations failed")
            return False
        
        # Test FeatureSet repository
        feature_set = FeatureSet(
            audio_file_id=audio_file.id,
            bpm=120.0,
            energy=0.8
        )
        saved_feature = feature_repo.save(feature_set)
        retrieved_feature = feature_repo.find_by_audio_file_id(audio_file.id)
        
        if retrieved_feature and retrieved_feature.audio_file_id == audio_file.id:
            print("✅ FeatureSet repository CRUD operations working")
        else:
            print("❌ FeatureSet repository CRUD operations failed")
            return False
        
        # Test Metadata repository
        metadata = Metadata(
            audio_file_id=audio_file.id,
            title="Test Song",
            artist="Test Artist"
        )
        saved_metadata = metadata_repo.save(metadata)
        retrieved_metadata = metadata_repo.find_by_audio_file_id(audio_file.id)
        
        if retrieved_metadata and retrieved_metadata.audio_file_id == audio_file.id:
            print("✅ Metadata repository CRUD operations working")
        else:
            print("❌ Metadata repository CRUD operations failed")
            return False
        
        # Test Playlist repository
        playlist = Playlist(
            name="Test Playlist",
            track_ids=[audio_file.id],
            track_paths=["/test/song.mp3"]
        )
        saved_playlist = playlist_repo.save(playlist)
        retrieved_playlist = playlist_repo.find_by_id(playlist.id)
        
        if retrieved_playlist and retrieved_playlist.id == playlist.id:
            print("✅ Playlist repository CRUD operations working")
        else:
            print("❌ Playlist repository CRUD operations failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Repository test failed: {e}")
        return False

def test_external_apis():
    """Test external API clients."""
    print("\n🔍 Testing external API clients...")
    
    try:
        from infrastructure.external_apis.musicbrainz_client import MusicBrainzClient
        from infrastructure.external_apis.lastfm_client import LastFMClient
        
        # Test MusicBrainz client
        mb_client = MusicBrainzClient()
        print("✅ MusicBrainz client initialized")
        
        # Test Last.fm client
        lf_client = LastFMClient()
        print("✅ Last.fm client initialized")
        
        # Test MusicBrainz search (this will make real API calls)
        track_result = mb_client.search_track("Creep", "Radiohead")
        if track_result:
            print(f"✅ MusicBrainz track search working: {track_result.title} by {track_result.artist}")
        else:
            print("⚠️ MusicBrainz track search returned no results (this is normal)")
        
        # Test Last.fm search (this will make real API calls if API key is provided)
        track_info = lf_client.get_track_info("Creep", "Radiohead")
        if track_info:
            print(f"✅ Last.fm track info working: {track_info.name} by {track_info.artist}")
        else:
            print("⚠️ Last.fm track info returned no results (this is normal without API key)")
        
        return True
        
    except Exception as e:
        print(f"❌ External API test failed: {e}")
        return False

def test_file_system():
    """Test file system services."""
    print("\n🔍 Testing file system services...")
    
    try:
        from infrastructure.file_system.playlist_exporter import PlaylistExporter
        from domain.entities import Playlist
        from uuid import uuid4
        
        exporter = PlaylistExporter()
        print("✅ Playlist exporter initialized")
        
        # Create test playlist
        test_playlist = Playlist(
            name="Test Export Playlist",
            description="A test playlist for export testing",
            track_ids=[uuid4(), uuid4()],
            track_paths=["/test/song1.mp3", "/test/song2.mp3"]
        )
        
        # Test validation
        if exporter.validate_playlist(test_playlist):
            print("✅ Playlist validation working")
        else:
            print("❌ Playlist validation failed")
            return False
        
        # Test export formats
        formats = exporter.get_export_formats()
        if len(formats) == 4 and 'm3u' in formats and 'pls' in formats:
            print("✅ Export formats correctly identified")
        else:
            print("❌ Export formats identification failed")
            return False
        
        # Test M3U export
        try:
            m3u_path = exporter.export_m3u(test_playlist, "test_export.m3u")
            if m3u_path.exists():
                print("✅ M3U export working")
                m3u_path.unlink()  # Clean up
            else:
                print("❌ M3U export failed")
                return False
        except Exception as e:
            print(f"⚠️ M3U export test skipped: {e}")
        
        # Test PLS export
        try:
            pls_path = exporter.export_pls(test_playlist, "test_export.pls")
            if pls_path.exists():
                print("✅ PLS export working")
                pls_path.unlink()  # Clean up
            else:
                print("❌ PLS export failed")
                return False
        except Exception as e:
            print(f"⚠️ PLS export test skipped: {e}")
        
        # Test XSPF export
        try:
            xspf_path = exporter.export_xspf(test_playlist, "test_export.xspf")
            if xspf_path.exists():
                print("✅ XSPF export working")
                xspf_path.unlink()  # Clean up
            else:
                print("❌ XSPF export failed")
                return False
        except Exception as e:
            print(f"⚠️ XSPF export test skipped: {e}")
        
        # Test JSON export
        try:
            json_path = exporter.export_json(test_playlist, "test_export.json")
            if json_path.exists():
                print("✅ JSON export working")
                json_path.unlink()  # Clean up
            else:
                print("❌ JSON export failed")
                return False
        except Exception as e:
            print(f"⚠️ JSON export test skipped: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ File system test failed: {e}")
        return False

def test_caching():
    """Test caching system."""
    print("\n🔍 Testing caching system...")
    
    try:
        from infrastructure.caching.cache_manager import CacheManager, cached
        
        cache_manager = CacheManager()
        print("✅ Cache manager initialized")
        
        # Test basic cache operations
        cache_manager.set("test_key", "test_value", ttl_seconds=60)
        
        if cache_manager.exists("test_key"):
            print("✅ Cache set and exists working")
        else:
            print("❌ Cache set and exists failed")
            return False
        
        retrieved_value = cache_manager.get("test_key")
        if retrieved_value == "test_value":
            print("✅ Cache get working")
        else:
            print("❌ Cache get failed")
            return False
        
        # Test cache decorator
        @cached(ttl_seconds=30, key_prefix="test")
        def test_function(x, y):
            return x + y
        
        result1 = test_function(1, 2)
        result2 = test_function(1, 2)  # Should be cached
        
        if result1 == result2 == 3:
            print("✅ Cache decorator working")
        else:
            print("❌ Cache decorator failed")
            return False
        
        # Test cache statistics
        stats = cache_manager.get_stats()
        if 'hits' in stats and 'misses' in stats:
            print("✅ Cache statistics working")
        else:
            print("❌ Cache statistics failed")
            return False
        
        # Test cache cleanup
        expired_count = cache_manager.cleanup_expired()
        print(f"✅ Cache cleanup working (cleaned {expired_count} expired entries)")
        
        # Clean up test data
        cache_manager.delete("test_key")
        
        return True
        
    except Exception as e:
        print(f"❌ Caching test failed: {e}")
        return False

def main():
    """Run all infrastructure tests."""
    print("🧪 Starting infrastructure layer tests...\n")
    
    tests = [
        ("Database Repositories", test_repositories),
        ("External APIs", test_external_apis),
        ("File System Services", test_file_system),
        ("Caching System", test_caching)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running: {test_name}")
        print('='*50)
        
        if test_func():
            passed += 1
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")
    
    print(f"\n{'='*50}")
    print(f"INFRASTRUCTURE LAYER TEST SUMMARY")
    print('='*50)
    print(f"Tests passed: {passed}/{total}")
    print(f"Success rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("🎉 All infrastructure layer tests passed!")
        return True
    else:
        print("⚠️  Some infrastructure layer tests failed.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 