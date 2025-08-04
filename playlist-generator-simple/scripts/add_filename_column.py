#!/usr/bin/env python3
"""
Script to add filename column to existing analysis_cache table.
"""

import sqlite3
import os
import sys

def add_filename_column(db_path: str) -> bool:
    """Add filename column to analysis_cache table."""
    print(f"Adding filename column to database: {db_path}")
    
    if not os.path.exists(db_path):
        print(f"Database file not found: {db_path}")
        return False
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if analysis_cache table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='analysis_cache'")
        if not cursor.fetchone():
            print("analysis_cache table does not exist. Nothing to do.")
            return True
        
        # Check if filename column exists
        cursor.execute("PRAGMA table_info(analysis_cache)")
        columns = [col[1] for col in cursor.fetchall()]
        
        if 'filename' not in columns:
            print("Adding filename column to analysis_cache table...")
            cursor.execute("ALTER TABLE analysis_cache ADD COLUMN filename TEXT")
            
            # Update existing records with filename from file_path
            cursor.execute("""
                UPDATE analysis_cache 
                SET filename = SUBSTR(file_path, LENGTH(file_path) - LENGTH(REPLACE(file_path, '/', '')) + 1)
                WHERE filename IS NULL
            """)
            
            conn.commit()
            print("Successfully added filename column to analysis_cache table")
        else:
            print("Filename column already exists in analysis_cache table")
        
        return True
        
    except Exception as e:
        print(f"Error adding filename column: {e}")
        return False
    finally:
        conn.close()

def main():
    """Main function."""
    if len(sys.argv) != 2:
        print("Usage: python add_filename_column.py <database_path>")
        print("Example: python add_filename_column.py cache/playlista.db")
        sys.exit(1)
    
    db_path = sys.argv[1]
    
    if add_filename_column(db_path):
        print("Filename column addition completed successfully!")
        sys.exit(0)
    else:
        print("Filename column addition failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 