#!/usr/bin/env python3
"""
Test script for the database manager.
Verifies playlist operations, caching, analysis results, and statistics.
"""

import os
import sys
import tempfile
import shutil
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from core.logging_setup import setup_logging, cleanup_logging
from core.database import DatabaseManager


def cleanup_database():
    """Clean up database handlers."""
    cleanup_logging()


def test_database_initialization():
    """Test database initialization."""
    print("Testing Database Initialization...")
    
    temp_dir = tempfile.mkdtemp()
    try:
        # Setup logging
        setup_logging(
            log_level='DEBUG',
            log_dir=temp_dir,
            log_file_prefix='test_db',
            console_logging=True,
            file_logging=False
        )
        
        # Create database with temporary path
        db_path = os.path.join(temp_dir, 'test_playlista.db')
        db = DatabaseManager(db_path)
        
        # Check if database file was created
        if os.path.exists(db_path):
            print(f"Database file created: {db_path}")
            print(f"Database size: {os.path.getsize(db_path)} bytes")
        else:
            print("Database file not created")
            return False
        
        # Test database statistics
        stats = db.get_database_statistics()
        print(f"Database statistics: {stats}")
        
        print("Database initialization test completed")
        return True
        
    finally:
        cleanup_database()
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_playlist_operations():
    """Test playlist save, retrieve, and delete operations."""
    print("\nTesting Playlist Operations...")
    
    temp_dir = tempfile.mkdtemp()
    try:
        # Setup logging
        setup_logging(
            log_level='INFO',
            log_dir=temp_dir,
            console_logging=True,
            file_logging=False
        )
        
        # Create database
        db_path = os.path.join(temp_dir, 'test_playlista.db')
        db = DatabaseManager(db_path)
        
        # Test data
        playlist_name = "Test Playlist"
        tracks = [
            "/music/song1.mp3",
            "/music/song2.wav",
            "/music/song3.flac"
        ]
        description = "A test playlist for database testing"
        features = {
            "bpm_range": [120, 140],
            "key": "C major",
            "energy": "high"
        }
        metadata = {
            "created_by": "test_user",
            "tags": ["test", "database"]
        }
        
        # Test save playlist
        print("Saving playlist...")
        success = db.save_playlist(
            name=playlist_name,
            tracks=tracks,
            description=description,
            features=features,
            metadata=metadata
        )
        
        if success:
            print("Playlist saved successfully")
        else:
            print("Failed to save playlist")
            return False
        
        # Test get playlist
        print("Retrieving playlist...")
        playlist = db.get_playlist(playlist_name)
        
        if playlist:
            print(f"Retrieved playlist: {playlist['name']}")
            print(f"Tracks: {len(playlist['tracks'])}")
            print(f"Description: {playlist['description']}")
            print(f"Features: {playlist['features']}")
        else:
            print("Failed to retrieve playlist")
            return False
        
        # Test get all playlists
        print("Retrieving all playlists...")
        all_playlists = db.get_all_playlists()
        print(f"Total playlists: {len(all_playlists)}")
        
        # Test delete playlist
        print("Deleting playlist...")
        delete_success = db.delete_playlist(playlist_name)
        
        if delete_success:
            print("Playlist deleted successfully")
        else:
            print("Failed to delete playlist")
            return False
        
        # Verify deletion
        deleted_playlist = db.get_playlist(playlist_name)
        if deleted_playlist is None:
            print("Playlist deletion verified")
        else:
            print("Playlist still exists after deletion")
            return False
        
        print("Playlist operations test completed")
        return True
        
    finally:
        cleanup_database()
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_analysis_results():
    """Test analysis results save and retrieve operations."""
    print("\nTesting Analysis Results Operations...")
    
    temp_dir = tempfile.mkdtemp()
    try:
        # Setup logging
        setup_logging(
            log_level='INFO',
            log_dir=temp_dir,
            console_logging=True,
            file_logging=False
        )
        
        # Create database
        db_path = os.path.join(temp_dir, 'test_playlista.db')
        db = DatabaseManager(db_path)
        
        # Test data
        file_path = "/music/test_song.mp3"
        filename = "test_song.mp3"
        file_size_bytes = 1024000
        file_hash = "abc123def456"
        analysis_data = {
            "bpm": 128.5,
            "key": "C",
            "energy": 0.8,
            "danceability": 0.7,
            "valence": 0.6
        }
        metadata = {
            "analyzer_version": "1.0.0",
            "analysis_duration": 2.5
        }
        
        # Test save analysis result
        print("Saving analysis result...")
        success = db.save_analysis_result(
            file_path=file_path,
            filename=filename,
            file_size_bytes=file_size_bytes,
            file_hash=file_hash,
            analysis_data=analysis_data,
            metadata=metadata
        )
        
        if success:
            print("Analysis result saved successfully")
        else:
            print("Failed to save analysis result")
            return False
        
        # Test get analysis result
        print("Retrieving analysis result...")
        result = db.get_analysis_result(file_path)
        
        if result:
            print(f"Retrieved analysis for: {result['filename']}")
            print(f"BPM: {result['analysis_data']['bpm']}")
            print(f"Key: {result['analysis_data']['key']}")
        else:
            print("Failed to retrieve analysis result")
            return False
        
        # Test get all analysis results
        print("Retrieving all analysis results...")
        all_results = db.get_all_analysis_results()
        print(f"Total analysis results: {len(all_results)}")
        
        print("Analysis results test completed")
        return True
        
    finally:
        cleanup_database()
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_cache_operations():
    """Test cache save, retrieve, and cleanup operations."""
    print("\nTesting Cache Operations...")
    
    temp_dir = tempfile.mkdtemp()
    try:
        # Setup logging
        setup_logging(
            log_level='INFO',
            log_dir=temp_dir,
            console_logging=True,
            file_logging=False
        )
        
        # Create database
        db_path = os.path.join(temp_dir, 'test_playlista.db')
        db = DatabaseManager(db_path)
        
        # Test data
        cache_key = "test_cache_key"
        cache_value = {
            "feature_vector": [0.1, 0.2, 0.3, 0.4, 0.5],
            "timestamp": time.time(),
            "source": "test_analyzer"
        }
        
        # Test save cache
        print("Saving cache entry...")
        success = db.save_cache(
            key=cache_key,
            value=cache_value,
            expires_hours=1
        )
        
        if success:
            print("Cache entry saved successfully")
        else:
            print("Failed to save cache entry")
            return False
        
        # Test get cache
        print("Retrieving cache entry...")
        retrieved_value = db.get_cache(cache_key)
        
        if retrieved_value:
            print(f"Retrieved cache entry: {retrieved_value['source']}")
            print(f"Feature vector length: {len(retrieved_value['feature_vector'])}")
        else:
            print("Failed to retrieve cache entry")
            return False
        
        # Test cache cleanup with immediate expiration
        print("Testing cache cleanup...")
        # Create another cache entry with immediate expiration
        db.save_cache("expired_key", {"data": "expired"}, expires_hours=0)
        
        # Wait a moment to ensure expiration
        time.sleep(0.1)
        
        cleaned_count = db.cleanup_cache(max_age_hours=0)  # Clean up immediately
        print(f"Cleaned up {cleaned_count} cache entries")
        
        # Verify cleanup - expired entry should be gone, but original should remain
        expired_value = db.get_cache("expired_key")
        original_value = db.get_cache(cache_key)
        
        if expired_value is None and original_value is not None:
            print("Cache cleanup verified - expired entry removed, original preserved")
        else:
            print("Cache cleanup verification failed")
            return False
        
        print("Cache operations test completed")
        return True
        
    finally:
        cleanup_database()
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_tags_operations():
    """Test tags save and retrieve operations."""
    print("\nTesting Tags Operations...")
    
    temp_dir = tempfile.mkdtemp()
    try:
        # Setup logging
        setup_logging(
            log_level='INFO',
            log_dir=temp_dir,
            console_logging=True,
            file_logging=False
        )
        
        # Create database
        db_path = os.path.join(temp_dir, 'test_playlista.db')
        db = DatabaseManager(db_path)
        
        # Test data
        file_path = "/music/test_song.mp3"
        tags = {
            "genre": ["rock", "alternative"],
            "mood": ["energetic", "upbeat"],
            "artist": "Test Artist",
            "album": "Test Album",
            "year": 2023
        }
        source = "musicbrainz"
        confidence = 0.85
        
        # Test save tags
        print("Saving tags...")
        success = db.save_tags(
            file_path=file_path,
            tags=tags,
            source=source,
            confidence=confidence
        )
        
        if success:
            print("Tags saved successfully")
        else:
            print("Failed to save tags")
            return False
        
        # Test get tags
        print("Retrieving tags...")
        tags_data = db.get_tags(file_path)
        
        if tags_data:
            print(f"Retrieved tags from: {tags_data['source']}")
            print(f"Confidence: {tags_data['confidence']}")
            print(f"Genres: {tags_data['tags']['genre']}")
        else:
            print("Failed to retrieve tags")
            return False
        
        print("Tags operations test completed")
        return True
        
    finally:
        cleanup_database()
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_failed_analysis():
    """Test failed analysis operations."""
    print("\nTesting Failed Analysis Operations...")
    
    temp_dir = tempfile.mkdtemp()
    try:
        # Setup logging
        setup_logging(
            log_level='INFO',
            log_dir=temp_dir,
            console_logging=True,
            file_logging=False
        )
        
        # Create database
        db_path = os.path.join(temp_dir, 'test_playlista.db')
        db = DatabaseManager(db_path)
        
        # Test data
        file_path = "/music/corrupted_song.mp3"
        filename = "corrupted_song.mp3"
        error_message = "File is corrupted or unsupported format"
        
        # Test mark analysis failed
        print("Marking analysis as failed...")
        success = db.mark_analysis_failed(
            file_path=file_path,
            filename=filename,
            error_message=error_message
        )
        
        if success:
            print("Analysis marked as failed successfully")
        else:
            print("Failed to mark analysis as failed")
            return False
        
        # Test get failed analysis files
        print("Retrieving failed analysis files...")
        failed_files = db.get_failed_analysis_files(max_retries=3)
        
        if failed_files:
            print(f"Found {len(failed_files)} failed analysis files")
            for failed_file in failed_files:
                print(f"  - {failed_file['filename']}: {failed_file['error_message']}")
        else:
            print("No failed analysis files found")
            return False
        
        print("Failed analysis test completed")
        return True
        
    finally:
        cleanup_database()
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_statistics():
    """Test statistics operations."""
    print("\nTesting Statistics Operations...")
    
    temp_dir = tempfile.mkdtemp()
    try:
        # Setup logging
        setup_logging(
            log_level='INFO',
            log_dir=temp_dir,
            console_logging=True,
            file_logging=False
        )
        
        # Create database
        db_path = os.path.join(temp_dir, 'test_playlista.db')
        db = DatabaseManager(db_path)
        
        # Test save statistics
        print("Saving statistics...")
        success1 = db.save_statistic("analysis", "total_files", 150)
        success2 = db.save_statistic("analysis", "successful_files", 145)
        success3 = db.save_statistic("analysis", "failed_files", 5)
        success4 = db.save_statistic("playlists", "total_playlists", 12)
        success5 = db.save_statistic("playlists", "total_tracks", 1200)
        
        if all([success1, success2, success3, success4, success5]):
            print("Statistics saved successfully")
        else:
            print("Failed to save some statistics")
            return False
        
        # Test get statistics
        print("Retrieving statistics...")
        stats = db.get_statistics(hours=24)
        
        if stats:
            print(f"Retrieved {len(stats)} statistic categories")
            for category, category_stats in stats.items():
                print(f"  {category}: {len(category_stats)} entries")
        else:
            print("No statistics found")
            return False
        
        # Test get database statistics
        print("Retrieving database statistics...")
        db_stats = db.get_database_statistics()
        
        if db_stats:
            print("Database statistics:")
            for key, value in db_stats.items():
                print(f"  {key}: {value}")
        else:
            print("No database statistics found")
            return False
        
        print("Statistics test completed")
        return True
        
    finally:
        cleanup_database()
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_cleanup_and_export():
    """Test cleanup and export operations."""
    print("\nTesting Cleanup and Export Operations...")
    
    temp_dir = tempfile.mkdtemp()
    try:
        # Setup logging
        setup_logging(
            log_level='INFO',
            log_dir=temp_dir,
            console_logging=True,
            file_logging=False
        )
        
        # Create database
        db_path = os.path.join(temp_dir, 'test_playlista.db')
        db = DatabaseManager(db_path)
        
        # Add some test data
        db.save_playlist("Test Playlist", ["/music/song1.mp3"])
        db.save_cache("test_key", {"data": "test_value"})
        db.save_statistic("test", "value", 42)
        
        # Test cleanup
        print("Testing data cleanup...")
        cleanup_results = db.cleanup_old_data(days=0)  # Clean up immediately
        
        if cleanup_results:
            print(f"Cleanup results: {cleanup_results}")
        else:
            print("No cleanup performed")
        
        # Test export
        print("Testing database export...")
        export_path = os.path.join(temp_dir, 'exported_db.db')
        export_success = db.export_database(export_path)
        
        if export_success and os.path.exists(export_path):
            print(f"Database exported successfully to: {export_path}")
            print(f"Export file size: {os.path.getsize(export_path)} bytes")
        else:
            print("Failed to export database")
            return False
        
        print("Cleanup and export test completed")
        return True
        
    finally:
        cleanup_database()
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    """Run all database tests."""
    print("Testing Database Manager")
    print("=" * 50)
    
    tests = [
        ("Database Initialization", test_database_initialization),
        ("Playlist Operations", test_playlist_operations),
        ("Analysis Results", test_analysis_results),
        ("Cache Operations", test_cache_operations),
        ("Tags Operations", test_tags_operations),
        ("Failed Analysis", test_failed_analysis),
        ("Statistics", test_statistics),
        ("Cleanup and Export", test_cleanup_and_export)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                print(f"✅ {test_name} PASSED")
                passed += 1
            else:
                print(f"❌ {test_name} FAILED")
                failed += 1
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
            failed += 1
    
    print(f"\n{'='*50}")
    print(f"Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All database tests passed!")
    else:
        print(f"⚠️ {failed} test(s) failed")
    
    print("\nDatabase Manager Features:")
    print("   Playlist storage and retrieval")
    print("   Analysis results caching")
    print("   File metadata storage")
    print("   Tags and enrichment data")
    print("   Failed analysis tracking")
    print("   Statistics and reporting")
    print("   Cache management")
    print("   Data cleanup and export")


if __name__ == "__main__":
    main() 