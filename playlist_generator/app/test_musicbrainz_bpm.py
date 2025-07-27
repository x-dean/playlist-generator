#!/usr/bin/env python3
"""
Test script to verify MusicBrainz BPM fallback functionality.
"""

import os
import sys
import logging
from musicbrainzngs import set_useragent

# Add the app directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from music_analyzer.feature_extractor import AudioAnalyzer

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# Set MusicBrainz user agent
set_useragent("playlista-test", "1.0", "test@example.com")

def test_musicbrainz_bpm():
    """Test MusicBrainz BPM lookup functionality."""
    
    # Initialize AudioAnalyzer
    analyzer = AudioAnalyzer()
    
    # Test cases with known artists/titles
    test_cases = [
        ("Daft Punk", "Get Lucky"),
        ("The Weeknd", "Blinding Lights"),
        ("Calvin Harris", "This Is What You Came For"),
        ("Avicii", "Wake Me Up"),
        ("David Guetta", "Titanium"),
    ]
    
    print("Testing MusicBrainz BPM lookup...")
    print("=" * 50)
    
    for artist, title in test_cases:
        print(f"\nTesting: {artist} - {title}")
        
        # Create dummy metadata
        metadata = {
            'artist': artist,
            'title': title
        }
        
        # Test the external BPM lookup
        try:
            bpm = analyzer._get_external_bpm("/dummy/path.mp3", metadata)
            if bpm is not None:
                print(f"✅ Found BPM: {bpm:.1f}")
            else:
                print("❌ No BPM found")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n" + "=" * 50)
    print("Test completed!")

if __name__ == "__main__":
    test_musicbrainz_bpm() 