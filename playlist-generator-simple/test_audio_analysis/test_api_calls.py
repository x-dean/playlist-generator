#!/usr/bin/env python3
"""
Test script to force external API calls and show enhanced logging.
"""

import sys
import os
import logging
sys.path.insert(0, '/app/src')

# Configure logging to see external API calls
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_api_calls():
    """Test external API calls with enhanced logging."""
    print("=" * 60)
    print("TESTING EXTERNAL API CALLS WITH ENHANCED LOGGING")
    print("=" * 60)
    
    try:
        from core.external_apis import metadata_enrichment_service
        
        print(" Successfully imported metadata enrichment service")
        
        # Test with metadata that will trigger API calls
        test_metadata = {
            'title': 'Bohemian Rhapsody',
            'artist': 'Queen'
        }
        
        print(f"\n Testing with metadata: {test_metadata}")
        print(" This should trigger MusicBrainz and Last.fm API calls...")
        
        # This should trigger the API calls and show enhanced logging
        enriched = metadata_enrichment_service.enrich_metadata(test_metadata)
        
        print(f"\n Enriched metadata keys: {list(enriched.keys())}")
        
        if enriched != test_metadata:
            print(" Metadata was enriched!")
            for key, value in enriched.items():
                if key not in test_metadata:
                    print(f"  + {key}: {value}")
        else:
            print("ℹ️ No enrichment occurred")
        
        print("\n External API calls test completed successfully!")
        
    except Exception as e:
        print(f" Error testing external API calls: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_api_calls() 