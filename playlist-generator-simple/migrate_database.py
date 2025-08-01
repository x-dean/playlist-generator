#!/usr/bin/env python3
"""
Database migration script for simplified schema.
"""

import sqlite3
import json
import os
import sys
from typing import Dict, Any, Optional

def migrate_database(db_path: str) -> bool:
    """
    Migrate existing database to new simplified schema.
    
    Args:
        db_path: Path to the database file
        
    Returns:
        True if migration successful, False otherwise
    """
    print(f"Migrating database: {db_path}")
    
    if not os.path.exists(db_path):
        print(f"Database file not found: {db_path}")
        return False
    
    # Create backup
    backup_path = f"{db_path}.backup"
    print(f"Creating backup: {backup_path}")
    
    try:
        with open(db_path, 'rb') as src, open(backup_path, 'wb') as dst:
            dst.write(src.read())
    except Exception as e:
        print(f"Failed to create backup: {e}")
        return False
    
    # Connect to database
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
    except Exception as e:
        print(f"Failed to connect to database: {e}")
        return False
    
    try:
        # Check if new schema already exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='tracks'")
        if cursor.fetchone():
            print("New schema already exists. Skipping migration.")
            return True
        
        # Read the new schema
        schema_path = "database_schema.sql"
        if not os.path.exists(schema_path):
            print(f"Schema file not found: {schema_path}")
            return False
        
        with open(schema_path, 'r') as f:
            schema_sql = f.read()
        
        # Execute schema creation
        print("Creating new schema...")
        cursor.executescript(schema_sql)
        
        # Check if old table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='analysis_results'")
        if not cursor.fetchone():
            print("No existing analysis_results table found. Migration complete.")
            conn.commit()
            return True
        
        # Migrate existing data
        print("Migrating existing data...")
        migrate_existing_data(cursor)
        
        conn.commit()
        print("Migration completed successfully!")
        return True
        
    except Exception as e:
        print(f"Migration failed: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()

def migrate_existing_data(cursor: sqlite3.Cursor) -> None:
    """
    Migrate data from old analysis_results table to new schema.
    """
    # Get all existing records
    cursor.execute("SELECT * FROM analysis_results")
    rows = cursor.fetchall()
    
    print(f"Found {len(rows)} existing records to migrate")
    
    for i, row in enumerate(rows):
        try:
            migrate_single_record(cursor, row, i + 1)
            if (i + 1) % 100 == 0:
                print(f"Migrated {i + 1} records...")
        except Exception as e:
            print(f"Failed to migrate record {i + 1}: {e}")
            continue

def migrate_single_record(cursor: sqlite3.Cursor, row: tuple, record_num: int) -> None:
    """
    Migrate a single record from old format to new schema.
    """
    # Parse the old JSON data
    try:
        data = json.loads(row[1])  # Assuming JSON data is in second column
    except (json.JSONDecodeError, IndexError):
        print(f"Invalid JSON data in record {record_num}")
        return
    
    # Extract basic file info
    file_path = data.get('file_path', '')
    filename = data.get('filename', '')
    file_size_bytes = data.get('file_size_bytes', 0)
    file_hash = data.get('file_hash', '')
    
    # Extract metadata
    metadata = data.get('metadata', {})
    title = metadata.get('title', 'Unknown')
    artist = metadata.get('artist', 'Unknown')
    album = metadata.get('album')
    track_number = metadata.get('track_number')
    genre = metadata.get('genre')
    year = metadata.get('year')
    
    # Extract analysis data
    analysis_data = data.get('analysis_data', {})
    duration = analysis_data.get('duration')
    bpm = analysis_data.get('bpm')
    key = analysis_data.get('key')
    mode = analysis_data.get('mode')
    loudness = analysis_data.get('loudness')
    danceability = analysis_data.get('danceability')
    energy = analysis_data.get('energy')
    
    # Determine analysis type and category
    analysis_type = analysis_data.get('analysis_type', 'full')
    long_audio_category = analysis_data.get('long_audio_category')
    
    # Insert into new tracks table
    cursor.execute("""
        INSERT OR REPLACE INTO tracks (
            file_path, file_hash, filename, file_size_bytes, analysis_date,
            title, artist, album, track_number, genre, year, duration,
            bpm, key, mode, loudness, danceability, energy,
            analysis_type, long_audio_category
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        file_path, file_hash, filename, file_size_bytes, 'now',
        title, artist, album, track_number, genre, year, duration,
        bpm, key, mode, loudness, danceability, energy,
        analysis_type, long_audio_category
    ))
    
    track_id = cursor.lastrowid
    
    # Migrate tags if available
    if 'tags' in data:
        migrate_tags(cursor, track_id, data['tags'])

def migrate_tags(cursor: sqlite3.Cursor, track_id: int, tags: Dict[str, Any]) -> None:
    """
    Migrate tags to new schema.
    """
    for source, tag_data in tags.items():
        if isinstance(tag_data, dict):
            for tag_name, tag_value in tag_data.items():
                cursor.execute("""
                    INSERT OR REPLACE INTO tags (track_id, source, tag_name, tag_value)
                    VALUES (?, ?, ?, ?)
                """, (track_id, source, tag_name, str(tag_value)))
        elif isinstance(tag_data, list):
            for tag_name in tag_data:
                cursor.execute("""
                    INSERT OR REPLACE INTO tags (track_id, source, tag_name)
                    VALUES (?, ?, ?)
                """, (track_id, source, tag_name))

def main():
    """Main migration function."""
    if len(sys.argv) != 2:
        print("Usage: python migrate_database.py <database_path>")
        sys.exit(1)
    
    db_path = sys.argv[1]
    
    if migrate_database(db_path):
        print("Migration completed successfully!")
        sys.exit(0)
    else:
        print("Migration failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 