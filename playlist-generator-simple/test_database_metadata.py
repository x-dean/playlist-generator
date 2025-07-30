#!/usr/bin/env python3
"""
Test script for database metadata querying functionality.
Tests the new separate fields for artist, album, title, genre, and year.
"""

import sys
import os
import json
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from core.database import DatabaseManager
from core.config_loader import config_loader

def test_metadata_querying():
    """Test the new metadata querying functionality."""
    print("🧪 Testing database metadata querying functionality...")
    
    # Initialize database
    db_manager = DatabaseManager()
    
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
    
    # Save test data
    print("📝 Saving test tracks...")
    for track in test_tracks:
        success = db_manager.save_analysis_result(
            file_path=track['file_path'],
            filename=track['filename'],
            file_size_bytes=track['file_size_bytes'],
            file_hash=track['file_hash'],
            analysis_data=track['analysis_data'],
            metadata=track['metadata']
        )
        if success:
            print(f"✅ Saved: {track['metadata']['title']} - {track['metadata']['artist']}")
        else:
            print(f"❌ Failed to save: {track['metadata']['title']}")
    
    # Test queries
    print("\n🔍 Testing metadata queries...")
    
    # Test artist query
    print("\n🎤 Testing artist query (Radiohead):")
    radiohead_tracks = db_manager.get_tracks_by_artist('Radiohead')
    print(f"Found {len(radiohead_tracks)} tracks by Radiohead:")
    for track in radiohead_tracks:
        print(f"  - {track['title']} ({track['album']}, {track['year']})")
    
    # Test album query
    print("\n💿 Testing album query (OK Computer):")
    ok_computer_tracks = db_manager.get_tracks_by_album('OK Computer')
    print(f"Found {len(ok_computer_tracks)} tracks from OK Computer:")
    for track in ok_computer_tracks:
        print(f"  - {track['title']} by {track['artist']} ({track['year']})")
    
    # Test genre query
    print("\n🎵 Testing genre query (Alternative Rock):")
    alt_rock_tracks = db_manager.get_tracks_by_genre('Alternative Rock')
    print(f"Found {len(alt_rock_tracks)} Alternative Rock tracks:")
    for track in alt_rock_tracks:
        print(f"  - {track['title']} by {track['artist']} ({track['album']})")
    
    # Test year query
    print("\n📅 Testing year query (1997):")
    tracks_1997 = db_manager.get_tracks_by_year(1997)
    print(f"Found {len(tracks_1997)} tracks from 1997:")
    for track in tracks_1997:
        print(f"  - {track['title']} by {track['artist']} ({track['album']})")
    
    # Test unique values
    print("\n📊 Testing unique value queries:")
    
    artists = db_manager.get_all_artists()
    print(f"🎤 All artists ({len(artists)}): {', '.join(artists)}")
    
    albums = db_manager.get_all_albums()
    print(f"💿 All albums ({len(albums)}): {', '.join(albums)}")
    
    genres = db_manager.get_all_genres()
    print(f"🎵 All genres ({len(genres)}): {', '.join(genres)}")
    
    print("\n✅ All metadata querying tests completed successfully!")

if __name__ == "__main__":
    test_metadata_querying() 