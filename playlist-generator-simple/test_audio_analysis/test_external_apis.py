#!/usr/bin/env python3
"""
Test script to verify enhanced external API logging.
"""

import sys
import os
sys.path.insert(0, '/app/src')

def test_external_apis():
    """Test external APIs with enhanced logging."""
    print("=" * 60)
    print("TESTING EXTERNAL APIS WITH ENHANCED LOGGING")
    print("=" * 60)
    
    try:
        # Import the external APIs module
        from core.external_apis import metadata_enrichment_service, MusicBrainzClient, LastFMClient
        
        print(" Successfully imported external APIs module")
        
        # Test MusicBrainz client
        print("\n Testing MusicBrainz client...")
        mb_client = MusicBrainzClient()
        print(f" MusicBrainz client initialized")
        
        # Test Last.fm client
        print("\n Testing Last.fm client...")
        lf_client = LastFMClient()
        print(f" Last.fm client initialized")
        
        # Test metadata enrichment service
        print("\n Testing metadata enrichment service...")
        print(f" Metadata enrichment service available: {metadata_enrichment_service.is_available()}")
        
        # Test with sample metadata
        print("\n Testing with sample metadata...")
        sample_metadata = {
            'title': 'Bohemian Rhapsody',
            'artist': 'Queen'
        }
        
        print(f"Input metadata: {sample_metadata}")
        enriched = metadata_enrichment_service.enrich_metadata(sample_metadata)
        print(f"Enriched metadata keys: {list(enriched.keys())}")
        
        print("\n External APIs test completed successfully!")
        
    except Exception as e:
        print(f" Error testing external APIs: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_external_apis() 