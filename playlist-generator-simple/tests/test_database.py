#!/usr/bin/env python3
"""
Database testing script for Playlist Generator Simple.
Tests all database functionality and verifies implementation.
"""

import os
import sys
import tempfile
import shutil
import json
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from core.database import DatabaseManager
from core.config_loader import config_loader

def test_database_initialization():
    """Test database initialization."""
    print("Testing database initialization...")
    
    # Create temporary database
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, 'test.db')
    
    try:
        # Initialize database
        db_manager = DatabaseManager(db_path)
        
        # Check if tables exist
        with db_manager._get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            expected_tables = ['tracks', 'tags', 'playlists', 'playlist_tracks', 
                             'analysis_cache', 'discovery_cache', 'cache', 'statistics']
            
            for table in expected_tables:
                if table not in tables:
                    print(f"❌ Missing table: {table}")
                    return False
                else:
                    print(f"✅ Table exists: {table}")
        
        print("✅ Database initialization test passed")
        return True
        
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        return False
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)

def test_analysis_result_storage():
    """Test analysis result storage."""
    print("\nTesting analysis result storage...")
    
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, 'test.db')
    
    try:
        db_manager = DatabaseManager(db_path)
        
        # Test data
        file_path = "/music/test_song.mp3"
        filename = "test_song.mp3"
        file_size_bytes = 1024000
        file_hash = "abc123def456"
        
        analysis_data = {
            'duration': 180.5,
            'bpm': 120.0,
            'key': 'C',
            'mode': 'major',
            'loudness': -12.5,
            'danceability': 0.8,
            'energy': 0.7,
            'analysis_type': 'full'
        }
        
        metadata = {
            'title': 'Test Song',
            'artist': 'Test Artist',
            'album': 'Test Album',
            'genre': 'Pop',
            'year': 2023
        }
        
        # Save analysis result
        success = db_manager.save_analysis_result(
            file_path, filename, file_size_bytes, file_hash,
            analysis_data, metadata
        )
        
        if not success:
            print("❌ Failed to save analysis result")
            return False
        
        # Retrieve analysis result
        result = db_manager.get_analysis_result(file_path)
        
        if not result:
            print("❌ Failed to retrieve analysis result")
            return False
        
        # Verify data
        if (result['title'] == 'Test Song' and 
            result['artist'] == 'Test Artist' and
            result['bpm'] == 120.0):
            print("✅ Analysis result storage test passed")
            return True
        else:
            print("❌ Retrieved data doesn't match saved data")
            return False
            
    except Exception as e:
        print(f"❌ Analysis result storage test failed: {e}")
        return False
    finally:
        shutil.rmtree(temp_dir)

def test_playlist_operations():
    """Test playlist operations."""
    print("\nTesting playlist operations...")
    
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, 'test.db')
    
    try:
        db_manager = DatabaseManager(db_path)
        
        # First, add a track
        file_path = "/music/test_song.mp3"
        db_manager.save_analysis_result(
            file_path, "test_song.mp3", 1024000, "abc123",
            {'duration': 180.5, 'bpm': 120.0}, 
            {'title': 'Test Song', 'artist': 'Test Artist'}
        )
        
        # Create playlist
        playlist_name = "Test Playlist"
        tracks = [file_path]
        description = "A test playlist"
        
        success = db_manager.save_playlist(
            playlist_name, tracks, description, 'manual'
        )
        
        if not success:
            print("❌ Failed to save playlist")
            return False
        
        # Retrieve playlist
        playlist = db_manager.get_playlist(playlist_name)
        
        if not playlist:
            print("❌ Failed to retrieve playlist")
            return False
        
        if (playlist['name'] == playlist_name and 
            playlist['description'] == description and
            len(playlist['tracks']) == 1):
            print("✅ Playlist operations test passed")
            return True
        else:
            print("❌ Retrieved playlist doesn't match saved playlist")
            return False
            
    except Exception as e:
        print(f"❌ Playlist operations test failed: {e}")
        return False
    finally:
        shutil.rmtree(temp_dir)

def test_cache_operations():
    """Test cache operations."""
    print("\nTesting cache operations...")
    
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, 'test.db')
    
    try:
        db_manager = DatabaseManager(db_path)
        
        # Test data
        cache_key = "test_cache_key"
        cache_value = {"test": "data", "number": 42}
        cache_type = "test"
        
        # Save to cache
        success = db_manager.save_cache(cache_key, cache_value, cache_type, 1)
        
        if not success:
            print("❌ Failed to save to cache")
            return False
        
        # Retrieve from cache
        retrieved = db_manager.get_cache(cache_key)
        
        if not retrieved:
            print("❌ Failed to retrieve from cache")
            return False
        
        if retrieved == cache_value:
            print("✅ Cache operations test passed")
            return True
        else:
            print("❌ Retrieved cache value doesn't match saved value")
            return False
            
    except Exception as e:
        print(f"❌ Cache operations test failed: {e}")
        return False
    finally:
        shutil.rmtree(temp_dir)

def test_failed_analysis_tracking():
    """Test failed analysis tracking."""
    print("\nTesting failed analysis tracking...")
    
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, 'test.db')
    
    try:
        db_manager = DatabaseManager(db_path)
        
        # Mark analysis as failed
        file_path = "/music/failed_song.mp3"
        filename = "failed_song.mp3"
        error_message = "Test error message"
        
        success = db_manager.mark_analysis_failed(file_path, filename, error_message)
        
        if not success:
            print("❌ Failed to mark analysis as failed")
            return False
        
        # Get failed files
        failed_files = db_manager.get_failed_analysis_files()
        
        if not failed_files:
            print("❌ No failed files found")
            return False
        
        failed_file = failed_files[0]
        if (failed_file['file_path'] == file_path and 
            failed_file['filename'] == filename and
            failed_file['error_message'] == error_message):
            print("✅ Failed analysis tracking test passed")
            return True
        else:
            print("❌ Failed file data doesn't match")
            return False
            
    except Exception as e:
        print(f"❌ Failed analysis tracking test failed: {e}")
        return False
    finally:
        shutil.rmtree(temp_dir)

def test_database_statistics():
    """Test database statistics."""
    print("\nTesting database statistics...")
    
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, 'test.db')
    
    try:
        db_manager = DatabaseManager(db_path)
        
        # Add some test data
        db_manager.save_analysis_result(
            "/music/song1.mp3", "song1.mp3", 1024000, "hash1",
            {'duration': 180.5}, {'title': 'Song 1', 'artist': 'Artist 1'}
        )
        
        db_manager.save_analysis_result(
            "/music/song2.mp3", "song2.mp3", 2048000, "hash2",
            {'duration': 240.0}, {'title': 'Song 2', 'artist': 'Artist 2'}
        )
        
        # Get statistics
        stats = db_manager.get_database_statistics()
        
        if not stats:
            print("❌ Failed to get database statistics")
            return False
        
        # Check if we have the expected statistics
        if 'total_tracks' in stats and stats['total_tracks'] >= 2:
            print("✅ Database statistics test passed")
            return True
        else:
            print("❌ Database statistics don't match expected values")
            return False
            
    except Exception as e:
        print(f"❌ Database statistics test failed: {e}")
        return False
    finally:
        shutil.rmtree(temp_dir)

def test_database_management():
    """Test database management functions."""
    print("\nTesting database management functions...")
    
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, 'test.db')
    
    try:
        db_manager = DatabaseManager(db_path)
        
        # Test integrity check
        integrity_ok = db_manager.check_integrity()
        if not integrity_ok:
            print("❌ Database integrity check failed")
            return False
        
        # Test database size
        size_info = db_manager.get_database_size()
        if not size_info['exists']:
            print("❌ Database size check failed")
            return False
        
        # Test backup creation
        backup_path = db_manager.create_backup()
        if not os.path.exists(backup_path):
            print("❌ Database backup creation failed")
            return False
        
        # Test vacuum
        vacuum_ok = db_manager.vacuum_database()
        if not vacuum_ok:
            print("❌ Database vacuum failed")
            return False
        
        print("✅ Database management test passed")
        return True
        
    except Exception as e:
        print(f"❌ Database management test failed: {e}")
        return False
    finally:
        shutil.rmtree(temp_dir)

def test_web_ui_queries():
    """Test web UI optimized queries."""
    print("\nTesting web UI queries...")
    
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, 'test.db')
    
    try:
        db_manager = DatabaseManager(db_path)
        
        # Add test data
        for i in range(5):
            db_manager.save_analysis_result(
                f"/music/song{i}.mp3", f"song{i}.mp3", 1024000, f"hash{i}",
                {'duration': 180.5 + i, 'bpm': 120.0 + i}, 
                {'title': f'Song {i}', 'artist': f'Artist {i}', 'genre': 'Pop'}
            )
        
        # Test web UI queries
        tracks = db_manager.get_tracks_for_web_ui(limit=3)
        if len(tracks) >= 3:
            print("✅ Web UI tracks query test passed")
        else:
            print("❌ Web UI tracks query failed")
            return False
        
        dashboard_data = db_manager.get_web_ui_dashboard_data()
        if 'total_tracks' in dashboard_data:
            print("✅ Web UI dashboard query test passed")
            return True
        else:
            print("❌ Web UI dashboard query failed")
            return False
            
    except Exception as e:
        print(f"❌ Web UI queries test failed: {e}")
        return False
    finally:
        shutil.rmtree(temp_dir)

def main():
    """Run all database tests."""
    print("=" * 60)
    print("DATABASE IMPLEMENTATION TEST SUITE")
    print("=" * 60)
    
    tests = [
        test_database_initialization,
        test_analysis_result_storage,
        test_playlist_operations,
        test_cache_operations,
        test_failed_analysis_tracking,
        test_database_statistics,
        test_database_management,
        test_web_ui_queries
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
    
    print("\n" + "=" * 60)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("🎉 All database tests passed! Database implementation is working correctly.")
        return 0
    else:
        print("❌ Some database tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 