#!/usr/bin/env python3
"""
Test script to verify metadata enrichment with enhanced logging.
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

def test_metadata_enrichment():
    """Test metadata enrichment with enhanced logging."""
    print("=" * 60)
    print("TESTING METADATA ENRICHMENT WITH ENHANCED LOGGING")
    print("=" * 60)
    
    try:
        # Import the audio analyzer
        from core.audio_analyzer import AudioAnalyzer
        
        print(" Successfully imported AudioAnalyzer")
        
        # Initialize the analyzer
        analyzer = AudioAnalyzer()
        print(" AudioAnalyzer initialized")
        
        # Test with a sample file that has metadata
        test_file = "/music/Alex Warren - Ordinary.mp3"
        
        if os.path.exists(test_file):
            print(f" Testing with file: {test_file}")
            
            # Extract metadata only
            print("\n Extracting metadata...")
            metadata = analyzer._extract_metadata(test_file)
            print(f" Extracted metadata: {metadata}")
            
            # Test enrichment
            print("\n Testing metadata enrichment...")
            from core.external_apis import metadata_enrichment_service
            
            enriched = metadata_enrichment_service.enrich_metadata(metadata)
            print(f" Enriched metadata keys: {list(enriched.keys())}")
            
            if enriched != metadata:
                print(" Metadata was enriched!")
                for key, value in enriched.items():
                    if key not in metadata:
                        print(f"  + {key}: {value}")
            else:
                print("ℹ️ No enrichment occurred")
                
        else:
            print(f" Test file not found: {test_file}")
            print("Testing with dummy metadata...")
            
            # Test with dummy metadata
            dummy_metadata = {
                'title': 'Bohemian Rhapsody',
                'artist': 'Queen'
            }
            
            from core.external_apis import metadata_enrichment_service
            enriched = metadata_enrichment_service.enrich_metadata(dummy_metadata)
            print(f" Enriched metadata keys: {list(enriched.keys())}")
        
        print("\n Metadata enrichment test completed successfully!")
        
    except Exception as e:
        print(f" Error testing metadata enrichment: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_metadata_enrichment() 