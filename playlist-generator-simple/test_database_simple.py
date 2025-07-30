#!/usr/bin/env python3
"""
Simple test for database metadata functionality.
Tests the new separate fields without full system dependencies.
"""

import sqlite3
import json
import os
from pathlib import Path

def test_database_metadata():
    """Test the new metadata fields in the database."""
    print("🧪 Testing database metadata fields...")
    
    # Create a test database
    test_db_path = "test_metadata.db"
    
    # Remove existing test database
    if os.path.exists(test_db_path):
        os.remove(test_db_path)
    
    # Create database with new schema
    with sqlite3.connect(test_db_path) as conn:
        cursor = conn.cursor()
        
        # Create analysis_results table with new fields
        cursor.execute("""
            CREATE TABLE analysis_results (
                file_path TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                file_size_bytes INTEGER NOT NULL,
                file_hash TEXT NOT NULL,
                analysis_data TEXT NOT NULL,
                metadata TEXT,
                artist TEXT,
                album TEXT,
                title TEXT,
                genre TEXT,
                year INTEGER,
                analysis_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_checked TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Test data
        test_tracks = [
            {
                'file_path': '/test/music/track1.mp3',
                'filename': 'track1.mp3',
                'file_size_bytes': 1024000,
                'file_hash': 'abc123',
                'analysis_data': {'bpm': 120, 'danceability': 0.8},
                'metadata': {
                    'artist': 'Radiohead',
                    'album': 'OK Computer',
                    'title': 'Paranoid Android',
                    'genre': 'Alternative Rock',
                    'year': 1997
                }
            },
            {
                'file_path': '/test/music/track2.mp3',
                'filename': 'track2.mp3',
                'file_size_bytes': 2048000,
                'file_hash': 'def456',
                'analysis_data': {'bpm': 140, 'danceability': 0.9},
                'metadata': {
                    'artist': 'Radiohead',
                    'album': 'The Bends',
                    'title': 'High and Dry',
                    'genre': 'Alternative Rock',
                    'year': 1995
                }
            },
            {
                'file_path': '/test/music/track3.mp3',
                'filename': 'track3.mp3',
                'file_size_bytes': 1536000,
                'file_hash': 'ghi789',
                'analysis_data': {'bpm': 110, 'danceability': 0.7},
                'metadata': {
                    'artist': 'Nirvana',
                    'album': 'Nevermind',
                    'title': 'Smells Like Teen Spirit',
                    'genre': 'Grunge',
                    'year': 1991
                }
            }
        ]
        
        # Insert test data
        print("📝 Inserting test tracks...")
        for track in test_tracks:
            analysis_json = json.dumps(track['analysis_data'])
            metadata_json = json.dumps(track['metadata'])
            
            # Extract metadata fields for efficient querying
            artist = track['metadata'].get('artist')
            album = track['metadata'].get('album')
            title = track['metadata'].get('title')
            genre = track['metadata'].get('genre')
            year = track['metadata'].get('year')
            
            cursor.execute("""
                INSERT INTO analysis_results 
                (file_path, filename, file_size_bytes, file_hash, 
                 analysis_data, metadata, artist, album, title, genre, year)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (track['file_path'], track['filename'], track['file_size_bytes'], 
                 track['file_hash'], analysis_json, metadata_json, 
                 artist, album, title, genre, year))
            
            print(f"✅ Inserted: {track['metadata']['title']} - {track['metadata']['artist']}")
        
        conn.commit()
        
        # Test queries
        print("\n🔍 Testing metadata queries...")
        
        # Test artist query
        print("\n🎤 Testing artist query (Radiohead):")
        cursor.execute("""
            SELECT file_path, filename, artist, album, title, genre, year, metadata
            FROM analysis_results 
            WHERE artist LIKE ? AND artist IS NOT NULL
            ORDER BY album, title
        """, ('%Radiohead%',))
        
        radiohead_tracks = cursor.fetchall()
        print(f"Found {len(radiohead_tracks)} tracks by Radiohead:")
        for row in radiohead_tracks:
            print(f"  - {row[4]} ({row[3]}, {row[6]})")
        
        # Test album query
        print("\n💿 Testing album query (OK Computer):")
        cursor.execute("""
            SELECT file_path, filename, artist, album, title, genre, year, metadata
            FROM analysis_results 
            WHERE album LIKE ? AND album IS NOT NULL
            ORDER BY title
        """, ('%OK Computer%',))
        
        ok_computer_tracks = cursor.fetchall()
        print(f"Found {len(ok_computer_tracks)} tracks from OK Computer:")
        for row in ok_computer_tracks:
            print(f"  - {row[4]} by {row[2]} ({row[6]})")
        
        # Test genre query
        print("\n🎵 Testing genre query (Alternative Rock):")
        cursor.execute("""
            SELECT file_path, filename, artist, album, title, genre, year, metadata
            FROM analysis_results 
            WHERE genre LIKE ? AND genre IS NOT NULL
            ORDER BY artist, album, title
        """, ('%Alternative Rock%',))
        
        alt_rock_tracks = cursor.fetchall()
        print(f"Found {len(alt_rock_tracks)} Alternative Rock tracks:")
        for row in alt_rock_tracks:
            print(f"  - {row[4]} by {row[2]} ({row[3]})")
        
        # Test year query
        print("\n📅 Testing year query (1997):")
        cursor.execute("""
            SELECT file_path, filename, artist, album, title, genre, year, metadata
            FROM analysis_results 
            WHERE year = ? AND year IS NOT NULL
            ORDER BY artist, album, title
        """, (1997,))
        
        tracks_1997 = cursor.fetchall()
        print(f"Found {len(tracks_1997)} tracks from 1997:")
        for row in tracks_1997:
            print(f"  - {row[4]} by {row[2]} ({row[3]})")
        
        # Test unique values
        print("\n📊 Testing unique value queries:")
        
        cursor.execute("""
            SELECT DISTINCT artist 
            FROM analysis_results 
            WHERE artist IS NOT NULL AND artist != ''
            ORDER BY artist
        """)
        artists = [row[0] for row in cursor.fetchall()]
        print(f"🎤 All artists ({len(artists)}): {', '.join(artists)}")
        
        cursor.execute("""
            SELECT DISTINCT album 
            FROM analysis_results 
            WHERE album IS NOT NULL AND album != ''
            ORDER BY album
        """)
        albums = [row[0] for row in cursor.fetchall()]
        print(f"💿 All albums ({len(albums)}): {', '.join(albums)}")
        
        cursor.execute("""
            SELECT DISTINCT genre 
            FROM analysis_results 
            WHERE genre IS NOT NULL AND genre != ''
            ORDER BY genre
        """)
        genres = [row[0] for row in cursor.fetchall()]
        print(f"🎵 All genres ({len(genres)}): {', '.join(genres)}")
    
    # Clean up
    os.remove(test_db_path)
    print("\n✅ All metadata querying tests completed successfully!")
    print("✅ Database schema supports efficient artist/album/genre/year queries!")

if __name__ == "__main__":
    test_database_metadata() 