#!/usr/bin/env python3
"""
Database Migration Script for Playlist Generator Simple
Migrates existing databases to the optimized schema.
"""

import sqlite3
import json
import os
import sys
from pathlib import Path
from datetime import datetime

def get_db_path():
    """Get database path."""
    if os.path.exists('/app/cache'):
        return '/app/cache/playlista.db'
    else:
        base_path = Path(__file__).resolve().parents[1]
        return str(base_path / 'cache' / 'playlista.db')

def check_table_exists(cursor, table_name):
    """Check if table exists."""
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
    return cursor.fetchone() is not None

def get_table_columns(cursor, table_name):
    """Get table columns."""
    cursor.execute("PRAGMA table_info(?)", (table_name,))
    return [row[1] for row in cursor.fetchall()]

def migrate_database():
    """Migrate database to optimized schema."""
    db_path = get_db_path()
    
    if not os.path.exists(db_path):
        print(f"Database not found: {db_path}")
        return False
    
    print(f"Migrating database: {db_path}")
    
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # Check if tracks table exists
            if not check_table_exists(cursor, 'tracks'):
                print("No tracks table found. Database may be empty.")
                return True
            
            # Get current tracks table columns
            current_columns = get_table_columns(cursor, 'tracks')
            print(f"Current tracks table columns: {len(current_columns)}")
            
            # Check if migration is needed
            new_columns = [
                'id', 'file_path', 'file_hash', 'filename', 'file_size_bytes',
                'analysis_date', 'created_at', 'updated_at', 'status', 'analysis_status',
                'retry_count', 'error_message', 'title', 'artist', 'album', 'track_number',
                'genre', 'year', 'duration', 'bitrate', 'sample_rate', 'channels',
                'bpm', 'key', 'mode', 'loudness', 'energy', 'danceability', 'valence',
                'acousticness', 'instrumentalness', 'composer', 'mood', 'style',
                'analysis_type', 'long_audio_category',
                # Rhythm features (Essentia)
                'rhythm_confidence', 'bpm_estimates', 'bpm_intervals', 'external_bpm',
                # Spectral features (Essentia)
                'spectral_centroid', 'spectral_flatness', 'spectral_rolloff', 'spectral_bandwidth',
                'spectral_contrast_mean', 'spectral_contrast_std',
                # Loudness features (Essentia)
                'dynamic_complexity', 'loudness_range', 'dynamic_range',
                # Key features (Essentia)
                'scale', 'key_strength', 'key_confidence',
                # MFCC features (Essentia)
                'mfcc_coefficients', 'mfcc_bands', 'mfcc_std', 'mfcc_delta', 'mfcc_delta2',
                # MusiCNN features
                'embedding', 'embedding_std', 'embedding_min', 'embedding_max', 'tags', 'musicnn_skipped',
                # Chroma features (Essentia)
                'chroma_mean', 'chroma_std',
                # Extended features (JSON for flexibility)
                'rhythm_features', 'spectral_features', 'mfcc_features', 'musicnn_features', 'spotify_features'
            ]
            
            # Check if all new columns exist
            missing_columns = [col for col in new_columns if col not in current_columns]
            
            if not missing_columns:
                print("Database already has optimized schema.")
                return True
            
            print(f"Missing columns: {missing_columns}")
            
            # Create backup
            backup_path = f"{db_path}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            print(f"Creating backup: {backup_path}")
            
            # Copy database to backup
            with sqlite3.connect(backup_path) as backup_conn:
                conn.backup(backup_conn)
            
            # Add missing columns
            for column in missing_columns:
                if column == 'id':
                    continue  # Primary key already exists
                
                # Determine column type
                if column in ['file_path', 'file_hash', 'filename', 'title', 'artist', 'album', 'genre', 'key', 'mode', 'composer', 'mood', 'style', 'analysis_type', 'long_audio_category', 'error_message']:
                    column_type = 'TEXT'
                elif column in ['file_size_bytes', 'retry_count', 'track_number', 'year', 'bitrate', 'sample_rate', 'channels']:
                    column_type = 'INTEGER'
                elif column in ['duration', 'bpm', 'loudness', 'energy', 'danceability', 'valence', 'acousticness', 'instrumentalness']:
                    column_type = 'REAL'
                elif column in ['analysis_date', 'created_at', 'updated_at']:
                    column_type = 'TIMESTAMP'
                else:
                    column_type = 'TEXT'  # JSON fields
                
                try:
                    cursor.execute(f"ALTER TABLE tracks ADD COLUMN {column} {column_type}")
                    print(f"Added column: {column}")
                except sqlite3.OperationalError as e:
                    if "duplicate column name" in str(e):
                        print(f"Column {column} already exists")
                    else:
                        raise
            
            # Remove unused tables if they exist
            unused_tables = ['file_metadata', 'analysis_statistics', 'failed_analysis']
            for table in unused_tables:
                if check_table_exists(cursor, table):
                    cursor.execute(f"DROP TABLE {table}")
                    print(f"Dropped unused table: {table}")
            
            # Update cache table structure if needed
            if check_table_exists(cursor, 'cache'):
                cache_columns = get_table_columns(cursor, 'cache')
                if 'cache_type' not in cache_columns:
                    cursor.execute("ALTER TABLE cache ADD COLUMN cache_type TEXT DEFAULT 'general'")
                    print("Added cache_type column to cache table")
            
            # Create views if they don't exist
            views = [
                ('track_complete', """
                    CREATE VIEW track_complete AS
                    SELECT 
                        t.*,
                        GROUP_CONCAT(DISTINCT tg.tag_name || ':' || tg.tag_value) as all_tags
                    FROM tracks t
                    LEFT JOIN tags tg ON t.id = tg.track_id
                    GROUP BY t.id
                """),
                ('track_summary', """
                    CREATE VIEW track_summary AS
                    SELECT 
                        id, file_path, filename, title, artist, album, genre, year, duration,
                        bpm, key, mode, energy, danceability, status, analysis_date
                    FROM tracks
                    WHERE status = 'analyzed'
                """),
                ('audio_analysis_complete', """
                    CREATE VIEW audio_analysis_complete AS
                    SELECT 
                        id, file_path, title, artist,
                        bpm, key, mode, loudness, energy, danceability, valence, acousticness, instrumentalness,
                        rhythm_features, spectral_features, mfcc_features, musicnn_features, spotify_features
                    FROM tracks
                    WHERE status = 'analyzed'
                """),
                ('failed_analysis_summary', """
                    CREATE VIEW failed_analysis_summary AS
                    SELECT 
                        cache_key, cache_value, created_at
                    FROM cache
                    WHERE cache_type = 'failed_analysis'
                """),
                ('playlist_features', """
                    CREATE VIEW playlist_features AS
                    SELECT 
                        t.id, t.title, t.artist, t.album, t.genre, t.year,
                        t.bpm, t.key, t.mode, t.energy, t.danceability, t.valence, t.acousticness,
                        pt.position
                    FROM tracks t
                    JOIN playlist_tracks pt ON t.id = pt.track_id
                    WHERE t.status = 'analyzed'
                """),
                ('playlist_summary', """
                    CREATE VIEW playlist_summary AS
                    SELECT 
                        p.id, p.name, p.description, p.generation_method, p.track_count,
                        p.created_at, p.updated_at,
                        COUNT(pt.track_id) as actual_track_count
                    FROM playlists p
                    LEFT JOIN playlist_tracks pt ON p.id = pt.playlist_id
                    GROUP BY p.id
                """),
                ('genre_analysis', """
                    CREATE VIEW genre_analysis AS
                    SELECT 
                        genre,
                        COUNT(*) as track_count,
                        AVG(bpm) as avg_bpm,
                        AVG(energy) as avg_energy,
                        AVG(danceability) as avg_danceability,
                        AVG(duration) as avg_duration
                    FROM tracks
                    WHERE status = 'analyzed' AND genre IS NOT NULL
                    GROUP BY genre
                """),
                ('statistics_summary', """
                    CREATE VIEW statistics_summary AS
                    SELECT 
                        category,
                        metric_name,
                        AVG(metric_value) as avg_value,
                        MAX(metric_value) as max_value,
                        MIN(metric_value) as min_value,
                        COUNT(*) as record_count
                    FROM statistics
                    GROUP BY category, metric_name
                """)
            ]
            
            for view_name, view_sql in views:
                try:
                    cursor.execute(view_sql)
                    print(f"Created view: {view_name}")
                except sqlite3.OperationalError as e:
                    if "already exists" in str(e):
                        print(f"View {view_name} already exists")
                    else:
                        print(f"Error creating view {view_name}: {e}")
            
            # Create indexes if they don't exist
            indexes = [
                ('idx_tracks_file_path', 'CREATE INDEX idx_tracks_file_path ON tracks(file_path)'),
                ('idx_tracks_status', 'CREATE INDEX idx_tracks_status ON tracks(status)'),
                ('idx_tracks_analysis_status', 'CREATE INDEX idx_tracks_analysis_status ON tracks(analysis_status)'),
                ('idx_tracks_artist', 'CREATE INDEX idx_tracks_artist ON tracks(artist)'),
                ('idx_tracks_title', 'CREATE INDEX idx_tracks_title ON tracks(title)'),
                ('idx_tracks_artist_title', 'CREATE INDEX idx_tracks_artist_title ON tracks(artist, title)'),
                ('idx_tracks_genre', 'CREATE INDEX idx_tracks_genre ON tracks(genre)'),
                ('idx_tracks_year', 'CREATE INDEX idx_tracks_year ON tracks(year)'),
                ('idx_tracks_bpm', 'CREATE INDEX idx_tracks_bpm ON tracks(bpm)'),
                ('idx_tracks_key', 'CREATE INDEX idx_tracks_key ON tracks(key)'),
                ('idx_tracks_energy', 'CREATE INDEX idx_tracks_energy ON tracks(energy)'),
                ('idx_tracks_danceability', 'CREATE INDEX idx_tracks_danceability ON tracks(danceability)'),
                ('idx_tracks_analysis_date', 'CREATE INDEX idx_tracks_analysis_date ON tracks(analysis_date)'),
                ('idx_tracks_genre_year', 'CREATE INDEX idx_tracks_genre_year ON tracks(genre, year)'),
                ('idx_tracks_bpm_energy', 'CREATE INDEX idx_tracks_bpm_energy ON tracks(bpm, energy)'),
                ('idx_tracks_key_mode', 'CREATE INDEX idx_tracks_key_mode ON tracks(key, mode)'),
                ('idx_tracks_artist_album', 'CREATE INDEX idx_tracks_artist_album ON tracks(artist, album)')
            ]
            
            for index_name, index_sql in indexes:
                try:
                    cursor.execute(index_sql)
                    print(f"Created index: {index_name}")
                except sqlite3.OperationalError as e:
                    if "already exists" in str(e):
                        print(f"Index {index_name} already exists")
                    else:
                        print(f"Error creating index {index_name}: {e}")
            
            conn.commit()
            print("Database migration completed successfully!")
            return True
            
    except Exception as e:
        print(f"Migration failed: {e}")
        return False

def main():
    """Main function."""
    if len(sys.argv) > 1 and sys.argv[1] == '--help':
        print("Database Migration Script")
        print("Usage: python migrate_database.py")
        print("Migrates existing database to optimized schema.")
        return
    
    success = migrate_database()
    if success:
        print("Migration completed successfully!")
        sys.exit(0)
    else:
        print("Migration failed!")
        sys.exit(1)

if __name__ == '__main__':
    main() 