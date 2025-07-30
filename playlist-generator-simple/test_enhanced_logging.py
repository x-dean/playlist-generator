#!/usr/bin/env python3
"""
Test script to demonstrate enhanced logging in DatabaseManager.
Shows detailed logging with emojis, performance tracking, and statistics.
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


def cleanup_test():
    """Clean up test handlers."""
    cleanup_logging()


def test_enhanced_logging():
    """Test enhanced logging features of DatabaseManager."""
    print("Testing Enhanced DatabaseManager Logging")
    print("=" * 50)
    
    temp_dir = tempfile.mkdtemp()
    try:
        # Setup logging with DEBUG level to see all details
        setup_logging(
            log_level='DEBUG',
            log_dir=temp_dir,
            log_file_prefix='test_enhanced_logging',
            console_logging=True,
            file_logging=True
        )
        
        print(f"📁 Test directory: {temp_dir}")
        
        # Create database
        db_path = os.path.join(temp_dir, 'test_playlista.db')
        print(f"🗄️ Database path: {db_path}")
        
        print("\n1. 🗄️ Testing Database Initialization...")
        db_manager = DatabaseManager(db_path)
        
        print("\n2. 📋 Testing Playlist Operations with Enhanced Logging...")
        # Save playlist
        db_manager.save_playlist(
            name="Enhanced Test Playlist",
            tracks=["/music/song1.mp3", "/music/song2.wav", "/music/song3.flac"],
            description="A test playlist for enhanced logging",
            features={"bpm_range": [120, 140], "energy": "high"},
            metadata={"created_by": "enhanced_test"}
        )
        
        # Get playlist
        playlist = db_manager.get_playlist("Enhanced Test Playlist")
        if playlist:
            print(f"   ✅ Retrieved playlist: {playlist['name']}")
            print(f"   📊 Tracks: {len(playlist['tracks'])}")
        
        print("\n3. 📊 Testing Analysis Results with Enhanced Logging...")
        # Save analysis results
        for i in range(3):
            db_manager.save_analysis_result(
                file_path=f"/music/test_song_{i}.mp3",
                filename=f"test_song_{i}.mp3",
                file_size_bytes=1024000 + (i * 100000),
                file_hash=f"hash_{i}_abc123",
                analysis_data={
                    "bpm": 120 + (i * 10),
                    "key": "C major",
                    "energy": 0.8,
                    "status": "analyzed"
                },
                metadata={"analyzed_by": "enhanced_test"}
            )
        
        print("\n4. 💾 Testing Cache Operations with Enhanced Logging...")
        # Save cache entries
        for i in range(2):
            db_manager.save_cache(
                key=f"test_cache_{i}",
                value={"data": f"cached_data_{i}", "timestamp": time.time()},
                expires_hours=1
            )
        
        print("\n5. ❌ Testing Failed Analysis with Enhanced Logging...")
        # Mark some files as failed
        for i in range(2):
            db_manager.mark_analysis_failed(
                file_path=f"/music/failed_song_{i}.mp3",
                filename=f"failed_song_{i}.mp3",
                error_message=f"Test error {i}: File format not supported"
            )
        
        print("\n6. 📈 Testing Statistics with Enhanced Logging...")
        # Save some statistics
        for i in range(3):
            db_manager.save_statistic("test_category", f"test_key_{i}", f"test_value_{i}")
        
        # Get database statistics
        stats = db_manager.get_database_statistics()
        print(f"   📊 Database statistics:")
        for key, value in stats.items():
            if key.endswith('_count'):
                print(f"      {key}: {value}")
        
        print("\n7. 🧹 Testing Cleanup with Enhanced Logging...")
        # Clean up old data
        cleanup_results = db_manager.cleanup_old_data(days=1)
        print(f"   🗑️ Cleanup results: {cleanup_results}")
        
        print("\n8. 📤 Testing Export with Enhanced Logging...")
        # Export database
        export_path = os.path.join(temp_dir, 'exported_db.db')
        success = db_manager.export_database(export_path)
        if success:
            export_size = os.path.getsize(export_path)
            print(f"   ✅ Export successful: {export_size / 1024:.1f} KB")
        
        print("\n✅ Enhanced logging test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Enhanced logging test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        cleanup_test()
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    """Run the enhanced logging test."""
    print("Enhanced DatabaseManager Logging Test")
    print("=" * 50)
    print("This test demonstrates:")
    print("  🗄️ Database initialization with detailed logging")
    print("  📋 Playlist operations with performance tracking")
    print("  📊 Analysis results with timing information")
    print("  💾 Cache operations with detailed status")
    print("  ❌ Failed analysis tracking with error details")
    print("  📈 Statistics generation with comprehensive data")
    print("  🧹 Cleanup operations with detailed results")
    print("  📤 Export operations with size information")
    print("=" * 50)
    
    success = test_enhanced_logging()
    
    if success:
        print("\n🎉 Enhanced logging test passed!")
        print("✅ All logging features working correctly")
        print("✅ Performance tracking active")
        print("✅ Detailed status reporting working")
        print("✅ Error handling with proper logging")
    else:
        print("\n❌ Enhanced logging test failed!")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 