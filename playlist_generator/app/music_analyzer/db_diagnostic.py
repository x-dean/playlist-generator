#!/usr/bin/env python3
"""
Database diagnostic script for the playlist generator.
This script helps identify and fix database issues.
"""

import os
import sqlite3
import json
import logging
from typing import Dict, List, Any

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatabaseDiagnostic:
    """Diagnostic tool for database issues."""
    
    def __init__(self, cache_file: str = None):
        cache_dir = os.getenv('CACHE_DIR', '/app/cache')
        self.cache_file = cache_file or os.path.join(cache_dir, 'audio_analysis.db')
        self.conn = None
        
    def connect(self):
        """Connect to the database."""
        try:
            self.conn = sqlite3.connect(self.cache_file, timeout=60)
            logger.info(f"Connected to database: {self.cache_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            return False
    
    def check_table_exists(self, table_name: str) -> bool:
        """Check if a table exists."""
        try:
            cursor = self.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?", 
                (table_name,)
            )
            return cursor.fetchone() is not None
        except Exception as e:
            logger.error(f"Error checking table existence: {e}")
            return False
    
    def get_table_schema(self, table_name: str) -> List[Dict[str, Any]]:
        """Get the schema of a table."""
        try:
            cursor = self.conn.execute(f"PRAGMA table_info({table_name})")
            columns = []
            for row in cursor.fetchall():
                columns.append({
                    'cid': row[0],
                    'name': row[1],
                    'type': row[2],
                    'notnull': row[3],
                    'dflt_value': row[4],
                    'pk': row[5]
                })
            return columns
        except Exception as e:
            logger.error(f"Error getting table schema: {e}")
            return []
    
    def check_data_integrity(self) -> Dict[str, Any]:
        """Check data integrity of the audio_features table."""
        results = {
            'total_records': 0,
            'failed_records': 0,
            'null_durations': 0,
            'invalid_bpm': 0,
            'missing_metadata': 0,
            'corrupted_json': 0
        }
        
        try:
            # Count total records
            cursor = self.conn.execute("SELECT COUNT(*) FROM audio_features")
            results['total_records'] = cursor.fetchone()[0]
            
            # Count failed records
            cursor = self.conn.execute("SELECT COUNT(*) FROM audio_features WHERE failed = 1")
            results['failed_records'] = cursor.fetchone()[0]
            
            # Check for null durations
            cursor = self.conn.execute("SELECT COUNT(*) FROM audio_features WHERE duration IS NULL")
            results['null_durations'] = cursor.fetchone()[0]
            
            # Check for invalid BPM values
            cursor = self.conn.execute("SELECT COUNT(*) FROM audio_features WHERE bpm < 0 OR bpm > 300")
            results['invalid_bpm'] = cursor.fetchone()[0]
            
            # Check for missing metadata
            cursor = self.conn.execute("SELECT COUNT(*) FROM audio_features WHERE metadata IS NULL OR metadata = ''")
            results['missing_metadata'] = cursor.fetchone()[0]
            
            # Check for corrupted JSON
            cursor = self.conn.execute("SELECT file_path, mfcc, chroma, musicnn_embedding, musicnn_tags FROM audio_features LIMIT 100")
            corrupted_count = 0
            for row in cursor.fetchall():
                for json_field in row[1:]:  # Skip file_path
                    if json_field:
                        try:
                            json.loads(json_field)
                        except json.JSONDecodeError:
                            corrupted_count += 1
                            break
            results['corrupted_json'] = corrupted_count
            
        except Exception as e:
            logger.error(f"Error checking data integrity: {e}")
        
        return results
    
    def repair_database(self) -> bool:
        """Attempt to repair common database issues."""
        try:
            logger.info("Starting database repair...")
            
            # Check if audio_features table exists
            if not self.check_table_exists('audio_features'):
                logger.error("audio_features table does not exist!")
                return False
            
            # Get current schema
            schema = self.get_table_schema('audio_features')
            column_names = [col['name'] for col in schema]
            
            # Define required columns
            required_columns = {
                'file_hash': 'TEXT',
                'file_path': 'TEXT',
                'duration': 'REAL',
                'bpm': 'REAL',
                'beat_confidence': 'REAL',
                'centroid': 'REAL',
                'loudness': 'REAL',
                'danceability': 'REAL',
                'key': 'TEXT',
                'scale': 'TEXT',
                'key_strength': 'REAL',
                'onset_rate': 'REAL',
                'zcr': 'REAL',
                'mfcc': 'JSON',
                'chroma': 'JSON',
                'spectral_contrast': 'REAL',
                'spectral_flatness': 'REAL',
                'spectral_rolloff': 'REAL',
                'musicnn_embedding': 'JSON',
                'musicnn_tags': 'JSON',
                'musicnn_skipped': 'INTEGER',
                'last_modified': 'REAL',
                'last_analyzed': 'TIMESTAMP',
                'metadata': 'JSON',
                'failed': 'INTEGER'
            }
            
            # Add missing columns
            with self.conn:
                for col_name, col_type in required_columns.items():
                    if col_name not in column_names:
                        try:
                            self.conn.execute(f"ALTER TABLE audio_features ADD COLUMN {col_name} {col_type}")
                            logger.info(f"Added missing column: {col_name}")
                        except Exception as e:
                            logger.warning(f"Failed to add column {col_name}: {e}")
            
            # Fix data issues
            with self.conn:
                # Fix null durations
                self.conn.execute("UPDATE audio_features SET duration = 0.0 WHERE duration IS NULL")
                
                # Fix invalid BPM values
                self.conn.execute("UPDATE audio_features SET bpm = 0.0 WHERE bpm < 0 OR bpm > 300")
                
                # Fix empty metadata
                self.conn.execute("UPDATE audio_features SET metadata = '{}' WHERE metadata IS NULL OR metadata = ''")
                
                # Fix corrupted JSON fields
                self.conn.execute("UPDATE audio_features SET mfcc = '[]' WHERE mfcc IS NULL OR mfcc = ''")
                self.conn.execute("UPDATE audio_features SET chroma = '[]' WHERE chroma IS NULL OR chroma = ''")
                self.conn.execute("UPDATE audio_features SET musicnn_embedding = '[]' WHERE musicnn_embedding IS NULL OR musicnn_embedding = ''")
                self.conn.execute("UPDATE audio_features SET musicnn_tags = '{}' WHERE musicnn_tags IS NULL OR musicnn_tags = ''")
            
            logger.info("Database repair completed")
            return True
            
        except Exception as e:
            logger.error(f"Error repairing database: {e}")
            return False
    
    def run_diagnostics(self):
        """Run comprehensive database diagnostics."""
        logger.info("Starting database diagnostics...")
        
        if not self.connect():
            return False
        
        # Check table existence
        if not self.check_table_exists('audio_features'):
            logger.error("audio_features table does not exist!")
            return False
        
        # Get schema
        schema = self.get_table_schema('audio_features')
        logger.info(f"audio_features table has {len(schema)} columns")
        
        # Check data integrity
        integrity_results = self.check_data_integrity()
        logger.info("Data integrity check results:")
        for key, value in integrity_results.items():
            logger.info(f"  {key}: {value}")
        
        # Attempt repair if issues found
        total_issues = sum(integrity_results.values()) - integrity_results['total_records']
        if total_issues > 0:
            logger.info(f"Found {total_issues} issues, attempting repair...")
            if self.repair_database():
                logger.info("Repair completed successfully")
            else:
                logger.error("Repair failed")
        
        self.conn.close()
        return True

def main():
    """Main diagnostic function."""
    diagnostic = DatabaseDiagnostic()
    diagnostic.run_diagnostics()

if __name__ == "__main__":
    main() 