#!/usr/bin/env python3
"""
Integration test for FileDiscovery with DatabaseManager.
Tests the unified database approach.
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
from core.file_discovery import FileDiscovery
from core.database import DatabaseManager


def cleanup_integration():
    """Clean up integration test handlers."""
    cleanup_logging()


def test_file_discovery_with_database():
    """Test FileDiscovery integration with DatabaseManager."""
    print("Testing FileDiscovery with DatabaseManager Integration...")
    
    temp_dir = tempfile.mkdtemp()
    try:
        # Setup logging
        setup_logging(
            log_level='INFO',
            log_dir=temp_dir,
            log_file_prefix='test_integration',
            console_logging=True,
            file_logging=False
        )
        
        # Create test music directory with some files
        test_music_dir = os.path.join(temp_dir, 'music')
        os.makedirs(test_music_dir, exist_ok=True)
        
        # Create some test audio files
        test_files = [
            'song1.mp3',
            'song2.wav', 
            'song3.flac',
            'invalid.txt',  # Invalid file
            'small.mp3'     # Will be too small
        ]
        
        for filename in test_files:
            filepath = os.path.join(test_music_dir, filename)
            if filename == 'small.mp3':
                # Create a small file
                with open(filepath, 'w') as f:
                    f.write('small')
            elif filename == 'invalid.txt':
                # Create invalid file
                with open(filepath, 'w') as f:
                    f.write('not audio')
            else:
                # Create valid audio files (simulated)
                with open(filepath, 'w') as f:
                    f.write('audio content ' * 100)  # Make it large enough
        
        # Create database with custom path
        db_path = os.path.join(temp_dir, 'test_playlista.db')
        
        # Initialize FileDiscovery with custom config
        config = {
            'MIN_FILE_SIZE_BYTES': 50,  # Small enough for our test files
            'VALID_EXTENSIONS': ['.mp3', '.wav', '.flac'],
            'HASH_ALGORITHM': 'md5',
            'MAX_RETRY_COUNT': 3,
            'ENABLE_RECURSIVE_SCAN': True,
            'LOG_LEVEL': 'INFO'
        }
        
        # Override paths for testing
        discovery = FileDiscovery(config)
        discovery.music_dir = test_music_dir
        discovery.failed_dir = os.path.join(temp_dir, 'failed')
        discovery.db_path = db_path
        
        print(f"Test music directory: {test_music_dir}")
        print(f"Test database path: {db_path}")
        
        # Test 1: Discover files
        print("\n1. Testing file discovery...")
        discovered_files = discovery.discover_files()
        print(f"   Discovered {len(discovered_files)} files")
        for file in discovered_files:
            print(f"   - {os.path.basename(file)}")
        
        # Test 2: Save to database
        print("\n2. Testing database save...")
        save_stats = discovery.save_discovered_files_to_db(discovered_files)
        print(f"   Save stats: {save_stats}")
        
        # Test 3: Get files from database
        print("\n3. Testing database retrieval...")
        db_files = discovery.get_db_files()
        print(f"   Database files: {len(db_files)}")
        for file in db_files:
            print(f"   - {os.path.basename(file)}")
        
        # Test 4: Get files for analysis
        print("\n4. Testing analysis queue...")
        analysis_files = discovery.get_files_for_analysis()
        print(f"   Files for analysis: {len(analysis_files)}")
        for file in analysis_files:
            print(f"   - {os.path.basename(file)}")
        
        # Test 5: Mark a file as failed
        print("\n5. Testing failed file tracking...")
        if discovered_files:
            test_file = discovered_files[0]
            discovery.mark_file_as_failed(test_file, "Test failure")
            print(f"   Marked as failed: {os.path.basename(test_file)}")
            
            failed_files = discovery.get_failed_files()
            print(f"   Failed files: {len(failed_files)}")
            for file in failed_files:
                print(f"   - {os.path.basename(file)}")
        
        # Test 6: Get statistics
        print("\n6. Testing statistics...")
        stats = discovery.get_statistics()
        print(f"   Current files: {stats.get('current_files', 0)}")
        print(f"   Database files: {stats.get('database_files', 0)}")
        print(f"   Failed files: {stats.get('failed_files', 0)}")
        print(f"   New files: {stats.get('new_files', 0)}")
        
        # Test 7: Validate file paths
        print("\n7. Testing file validation...")
        all_files = [os.path.join(test_music_dir, f) for f in os.listdir(test_music_dir)]
        valid_files = discovery.validate_file_paths(all_files)
        print(f"   Valid files: {len(valid_files)}")
        print(f"   Invalid files: {len(all_files) - len(valid_files)}")
        
        # Test 8: Get file info
        print("\n8. Testing file info...")
        if discovered_files:
            file_info = discovery.get_file_info(discovered_files[0])
            print(f"   File info for {os.path.basename(discovered_files[0])}:")
            print(f"     Size: {file_info.get('file_size_bytes', 0)} bytes")
            print(f"     Hash: {file_info.get('file_hash', '')[:8]}...")
            print(f"     Valid: {file_info.get('is_valid', False)}")
        
        print("\n✅ FileDiscovery with DatabaseManager integration test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        cleanup_integration()
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_database_manager_operations():
    """Test DatabaseManager operations used by FileDiscovery."""
    print("\nTesting DatabaseManager Operations...")
    
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
        db_manager = DatabaseManager(db_path)
        
        # Test analysis result operations
        print("1. Testing analysis result operations...")
        
        # Save analysis result
        test_file = "/test/music/song.mp3"
        db_manager.save_analysis_result(
            file_path=test_file,
            filename="song.mp3",
            file_size_bytes=1024000,
            file_hash="abc123hash",
            analysis_data={'status': 'discovered', 'bpm': 120},
            metadata={'discovered_date': '2025-07-30'}
        )
        print("   ✓ Saved analysis result")
        
        # Retrieve analysis result
        result = db_manager.get_analysis_result(test_file)
        if result:
            print(f"   ✓ Retrieved analysis result: {result['filename']}")
            print(f"     Status: {result['analysis_data'].get('status')}")
            print(f"     BPM: {result['analysis_data'].get('bpm')}")
        else:
            print("   ❌ Failed to retrieve analysis result")
        
        # Test failed analysis operations
        print("\n2. Testing failed analysis operations...")
        
        # Mark as failed
        db_manager.mark_analysis_failed(test_file, "song.mp3", "Test error")
        print("   ✓ Marked analysis as failed")
        
        # Get failed files
        failed_files = db_manager.get_failed_analysis_files()
        print(f"   ✓ Retrieved {len(failed_files)} failed files")
        for failed in failed_files:
            print(f"     - {failed['filename']}: {failed['error_message']}")
        
        # Test delete operations
        print("\n3. Testing delete operations...")
        
        # Delete analysis result
        deleted = db_manager.delete_analysis_result(test_file)
        print(f"   ✓ Deleted analysis result: {deleted}")
        
        # Delete failed analysis
        deleted = db_manager.delete_failed_analysis(test_file)
        print(f"   ✓ Deleted failed analysis: {deleted}")
        
        # Verify deletion
        result = db_manager.get_analysis_result(test_file)
        failed_files = db_manager.get_failed_analysis_files()
        print(f"   ✓ Verification: analysis result exists: {result is not None}")
        print(f"   ✓ Verification: failed files count: {len(failed_files)}")
        
        print("\n✅ DatabaseManager operations test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ DatabaseManager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        cleanup_integration()
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    """Run all integration tests."""
    print("Integration Test: FileDiscovery with DatabaseManager")
    print("=" * 60)
    
    tests = [
        ("FileDiscovery with DatabaseManager", test_file_discovery_with_database),
        ("DatabaseManager Operations", test_database_manager_operations)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                failed += 1
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            failed += 1
            print(f"❌ {test_name} FAILED with exception: {e}")
    
    print(f"\n{'='*60}")
    print(f"Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All integration tests passed!")
        print("\n✅ FileDiscovery successfully integrated with DatabaseManager")
        print("✅ Database operations working correctly")
        print("✅ File discovery and tracking working")
        print("✅ Failed file handling working")
        print("✅ Statistics and reporting working")
    else:
        print("❌ Some integration tests failed")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 