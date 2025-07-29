#!/usr/bin/env python3
"""
Test script to check database issues and see why record count is stuck at 290.
"""

import os
import sys
import sqlite3
import logging

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_database():
    """Test the database to see what's happening."""
    try:
        # Connect to the database
        cache_dir = os.getenv('CACHE_DIR', '/app/cache')
        db_path = os.path.join(cache_dir, 'audio_analysis.db')
        
        logger.info(f"Testing database: {db_path}")
        
        if not os.path.exists(db_path):
            logger.error(f"Database file does not exist: {db_path}")
            return False
        
        conn = sqlite3.connect(db_path, timeout=60)
        
        # Check basic info
        cursor = conn.execute("SELECT COUNT(*) FROM audio_features")
        total_records = cursor.fetchone()[0]
        logger.info(f"Total records: {total_records}")
        
        # Check failed records
        cursor = conn.execute("SELECT COUNT(*) FROM audio_features WHERE failed = 1")
        failed_records = cursor.fetchone()[0]
        logger.info(f"Failed records: {failed_records}")
        
        # Check successful records
        cursor = conn.execute("SELECT COUNT(*) FROM audio_features WHERE failed = 0")
        successful_records = cursor.fetchone()[0]
        logger.info(f"Successful records: {successful_records}")
        
        # Check recent records
        cursor = conn.execute("""
            SELECT file_path, failed, last_analyzed 
            FROM audio_features 
            ORDER BY last_analyzed DESC 
            LIMIT 10
        """)
        recent_records = cursor.fetchall()
        logger.info("Recent records:")
        for record in recent_records:
            logger.info(f"  {record[0]} - Failed: {record[1]} - Last analyzed: {record[2]}")
        
        # Check table schema
        cursor = conn.execute("PRAGMA table_info(audio_features)")
        columns = cursor.fetchall()
        logger.info(f"Table has {len(columns)} columns:")
        for col in columns:
            logger.info(f"  {col[1]} ({col[2]})")
        
        # Test a simple insert
        test_hash = "test_insert_hash"
        test_path = "/test/insert.mp3"
        
        logger.info("Testing database insert...")
        with conn:
            conn.execute("""
                INSERT OR REPLACE INTO audio_features (
                    file_hash, file_path, duration, bpm, failed
                ) VALUES (?, ?, ?, ?, ?)
            """, (test_hash, test_path, 0.0, 0.0, 1))
            
            # Check if insert worked
            cursor = conn.execute("SELECT COUNT(*) FROM audio_features")
            new_total = cursor.fetchone()[0]
            logger.info(f"Records after test insert: {new_total}")
            
            if new_total > total_records:
                logger.info("Database insert test successful!")
            else:
                logger.warning("Database insert test failed - record count didn't increase")
            
            # Clean up test record
            conn.execute("DELETE FROM audio_features WHERE file_hash = ?", (test_hash,))
        
        conn.close()
        return True
        
    except Exception as e:
        logger.error(f"Database test failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def test_audio_analyzer():
    """Test the AudioAnalyzer class."""
    try:
        from music_analyzer.feature_extractor import AudioAnalyzer
        
        logger.info("Testing AudioAnalyzer...")
        analyzer = AudioAnalyzer()
        
        # Test database stats
        stats = analyzer.get_database_stats()
        logger.info(f"AudioAnalyzer database stats: {stats}")
        
        # Test connectivity
        if analyzer.test_database_connectivity():
            logger.info("AudioAnalyzer database connectivity test passed")
        else:
            logger.error("AudioAnalyzer database connectivity test failed")
        
        return True
        
    except Exception as e:
        logger.error(f"AudioAnalyzer test failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def main():
    """Main test function."""
    logger.info("Starting database tests...")
    
    # Test basic database operations
    if test_database():
        logger.info("Basic database test passed")
    else:
        logger.error("Basic database test failed")
    
    # Test AudioAnalyzer
    if test_audio_analyzer():
        logger.info("AudioAnalyzer test passed")
    else:
        logger.error("AudioAnalyzer test failed")

if __name__ == "__main__":
    main() 