#!/usr/bin/env python3
"""
Simple test to check MusicBrainz lookup and see what data is returned.
"""

import os
import sys
import logging
from musicbrainzngs import set_useragent, search_recordings, get_recording_by_id

# Add the app directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# Set MusicBrainz user agent
set_useragent("playlista-simple", "1.0", "simple@example.com")

def test_musicbrainz_simple():
    """Test basic MusicBrainz lookup."""
    
    print("Testing basic MusicBrainz lookup...")
    print("=" * 50)
    
    # Test with a well-known track
    artist = "Daft Punk"
    title = "Get Lucky"
    
    print(f"Searching for: {artist} - {title}")
    
    try:
        # Step 1: Search for the recording
        result = search_recordings(artist=artist, recording=title, limit=1)
        print(f"Search result: {result}")
        
        if not result.get('recording-list') or len(result['recording-list']) == 0:
            print("❌ No recordings found")
            return
        
        rec = result['recording-list'][0]
        mbid = rec['id']
        print(f"✅ Found recording: {mbid}")
        print(f"Recording title: {rec.get('title', 'Unknown')}")
        
        # Step 2: Get full recording info
        rec_full = get_recording_by_id(
            mbid,
            includes=['artists', 'releases', 'tags', 'work-rels']
        )['recording']
        
        print(f"✅ Got full recording info")
        print(f"Available keys: {list(rec_full.keys())}")
        
        # Check for work relations
        if 'work-relation-list' in rec_full:
            work_rels = rec_full['work-relation-list']
            print(f"✅ Found {len(work_rels)} work relations")
            
            for i, work_rel in enumerate(work_rels):
                work = work_rel['work']
                print(f"Work {i+1}: {work.get('title', 'Unknown')}")
                
                if 'attributes' in work:
                    attrs = work['attributes']
                    print(f"  Work has {len(attrs)} attributes:")
                    for attr in attrs:
                        print(f"    {attr.get('type', 'unknown')} = {attr.get('value', 'unknown')}")
                else:
                    print("  Work has no attributes")
        else:
            print("❌ No work relations found")
        
        # Check for recording attributes
        if 'attributes' in rec_full:
            attrs = rec_full['attributes']
            print(f"✅ Recording has {len(attrs)} attributes:")
            for attr in attrs:
                print(f"  {attr.get('type', 'unknown')} = {attr.get('value', 'unknown')}")
        else:
            print("❌ Recording has no attributes")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    test_musicbrainz_simple() 