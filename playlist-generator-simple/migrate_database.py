#!/usr/bin/env python3
"""
Database migration script to upgrade from JSON blob to normalized schema.
"""

import sqlite3
import json
import os
import sys
from typing import Dict, Any, Optional

def migrate_database(db_path: str) -> bool:
    """
    Migrate existing database to new normalized schema.
    
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
        except Exception as e:
            print(f"Failed to migrate record {i + 1}: {e}")
            continue
    
    print("Data migration completed")

def migrate_single_record(cursor: sqlite3.Cursor, row: tuple, record_num: int) -> None:
    """
    Migrate a single record from old to new schema.
    """
    # Parse old row structure
    file_path, filename, file_size_bytes, file_hash, analysis_data, metadata, artist, album, title, genre, year, long_audio_category, analysis_date = row
    
    # Parse JSON data
    try:
        analysis_dict = json.loads(analysis_data) if analysis_data else {}
        metadata_dict = json.loads(metadata) if metadata else {}
    except json.JSONDecodeError:
        print(f"Invalid JSON in record {record_num}, skipping")
        return
    
    # Insert into tracks table
    cursor.execute("""
        INSERT INTO tracks (
            file_path, filename, file_size_bytes, file_hash, analysis_date,
            title, artist, album, genre, year, long_audio_category,
            bpm, key, mode, loudness, danceability, energy,
            rhythm_confidence, key_confidence, duration
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        file_path, filename, file_size_bytes, file_hash, analysis_date,
        title, artist, album, genre, year, long_audio_category,
        analysis_dict.get('bpm'), analysis_dict.get('key'), analysis_dict.get('mode'),
        analysis_dict.get('loudness'), analysis_dict.get('danceability'), analysis_dict.get('energy'),
        analysis_dict.get('confidence'), analysis_dict.get('key_confidence'), metadata_dict.get('duration')
    ))
    
    track_id = cursor.lastrowid
    
    # Migrate external API data
    migrate_external_data(cursor, track_id, metadata_dict)
    
    # Migrate tags
    migrate_tags(cursor, track_id, metadata_dict)
    
    # Migrate spectral features
    migrate_spectral_features(cursor, track_id, analysis_dict)
    
    # Migrate loudness features
    migrate_loudness_features(cursor, track_id, analysis_dict)
    
    # Migrate MFCC features
    migrate_mfcc_features(cursor, track_id, analysis_dict)
    
    # Migrate MusiCNN features
    migrate_musicnn_features(cursor, track_id, analysis_dict)
    
    # Migrate chroma features
    migrate_chroma_features(cursor, track_id, analysis_dict)
    
    # Migrate rhythm features
    migrate_rhythm_features(cursor, track_id, analysis_dict)
    
    # Migrate advanced features
    migrate_advanced_features(cursor, track_id, analysis_dict)
    
    if record_num % 10 == 0:
        print(f"Migrated {record_num} records...")

def migrate_external_data(cursor: sqlite3.Cursor, track_id: int, metadata: Dict[str, Any]) -> None:
    """Migrate external API data."""
    if 'musicbrainz_id' in metadata:
        cursor.execute("""
            INSERT INTO external_metadata (
                track_id, source, musicbrainz_id, musicbrainz_artist_id, 
                musicbrainz_album_id, release_date
            ) VALUES (?, ?, ?, ?, ?, ?)
        """, (
            track_id, 'musicbrainz',
            metadata.get('musicbrainz_id'),
            metadata.get('musicbrainz_artist_id'),
            metadata.get('musicbrainz_album_id'),
            metadata.get('release_date')
        ))

def migrate_tags(cursor: sqlite3.Cursor, track_id: int, metadata: Dict[str, Any]) -> None:
    """Migrate tags from metadata."""
    # MusicBrainz tags
    if 'musicbrainz_tags' in metadata:
        for tag in metadata['musicbrainz_tags']:
            cursor.execute("""
                INSERT INTO tags (track_id, source, tag_name)
                VALUES (?, ?, ?)
            """, (track_id, 'musicbrainz', tag))
    
    # Last.fm tags
    if 'lastfm_tags' in metadata:
        for tag in metadata['lastfm_tags']:
            cursor.execute("""
                INSERT INTO tags (track_id, source, tag_name)
                VALUES (?, ?, ?)
            """, (track_id, 'lastfm', tag))

def migrate_spectral_features(cursor: sqlite3.Cursor, track_id: int, analysis: Dict[str, Any]) -> None:
    """Migrate spectral features."""
    if any(key in analysis for key in ['spectral_centroid', 'spectral_rolloff', 'spectral_flatness']):
        cursor.execute("""
            INSERT INTO spectral_features (
                track_id, spectral_centroid, spectral_rolloff, spectral_flatness
            ) VALUES (?, ?, ?, ?)
        """, (
            track_id,
            analysis.get('spectral_centroid'),
            analysis.get('spectral_rolloff'),
            analysis.get('spectral_flatness')
        ))

def migrate_loudness_features(cursor: sqlite3.Cursor, track_id: int, analysis: Dict[str, Any]) -> None:
    """Migrate loudness features."""
    if 'loudness' in analysis:
        cursor.execute("""
            INSERT INTO loudness_features (
                track_id, integrated_loudness
            ) VALUES (?, ?)
        """, (track_id, analysis.get('loudness')))

def migrate_mfcc_features(cursor: sqlite3.Cursor, track_id: int, analysis: Dict[str, Any]) -> None:
    """Migrate MFCC features."""
    if 'mfcc' in analysis:
        cursor.execute("""
            INSERT INTO mfcc_features (
                track_id, mfcc_coefficients, mfcc_bands, mfcc_std
            ) VALUES (?, ?, ?, ?)
        """, (
            track_id,
            json.dumps(analysis.get('mfcc')),
            json.dumps(analysis.get('mfcc_bands')),
            json.dumps(analysis.get('mfcc_std'))
        ))

def migrate_musicnn_features(cursor: sqlite3.Cursor, track_id: int, analysis: Dict[str, Any]) -> None:
    """Migrate MusiCNN features."""
    if 'musicnn_embedding' in analysis:
        cursor.execute("""
            INSERT INTO musicnn_features (
                track_id, embedding, tags
            ) VALUES (?, ?, ?)
        """, (
            track_id,
            json.dumps(analysis.get('musicnn_embedding')),
            json.dumps(analysis.get('musicnn_tags'))
        ))

def migrate_chroma_features(cursor: sqlite3.Cursor, track_id: int, analysis: Dict[str, Any]) -> None:
    """Migrate chroma features."""
    if 'chroma' in analysis:
        cursor.execute("""
            INSERT INTO chroma_features (
                track_id, chroma_mean, chroma_std
            ) VALUES (?, ?, ?)
        """, (
            track_id,
            json.dumps(analysis.get('chroma')),
            json.dumps(analysis.get('chroma_std'))
        ))

def migrate_rhythm_features(cursor: sqlite3.Cursor, track_id: int, analysis: Dict[str, Any]) -> None:
    """Migrate rhythm features."""
    if any(key in analysis for key in ['bpm', 'estimates', 'bpm_intervals']):
        cursor.execute("""
            INSERT INTO rhythm_features (
                track_id, bpm_estimates, bpm_intervals, external_bpm
            ) VALUES (?, ?, ?, ?)
        """, (
            track_id,
            json.dumps(analysis.get('estimates')),
            json.dumps(analysis.get('bpm_intervals')),
            analysis.get('external_bpm')
        ))

def migrate_advanced_features(cursor: sqlite3.Cursor, track_id: int, analysis: Dict[str, Any]) -> None:
    """Migrate advanced features."""
    if any(key in analysis for key in ['onset_rate', 'zero_crossing_rate']):
        cursor.execute("""
            INSERT INTO advanced_features (
                track_id, onset_rate, zero_crossing_rate
            ) VALUES (?, ?, ?)
        """, (
            track_id,
            analysis.get('onset_rate'),
            analysis.get('zero_crossing_rate')
        ))

if __name__ == "__main__":
    # Default database path
    db_path = "cache/playlista.db"
    
    if len(sys.argv) > 1:
        db_path = sys.argv[1]
    
    success = migrate_database(db_path)
    sys.exit(0 if success else 1) 