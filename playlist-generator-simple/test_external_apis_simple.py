#!/usr/bin/env python3
"""
Simple test script for external API integration.

This script tests the MusicBrainz and Last.fm API clients
without depending on the full audio analysis system.
"""

import sys
import os
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Mock the logging setup to avoid dependencies
class MockLogger:
    def __init__(self, name):
        self.name = name
    
    def debug(self, msg):
        print(f"DEBUG: {msg}")
    
    def info(self, msg):
        print(f"INFO: {msg}")
    
    def warning(self, msg):
        print(f"WARNING: {msg}")
    
    def error(self, msg):
        print(f"ERROR: {msg}")

def get_logger(name):
    return MockLogger(name)

# Mock the config loader
class MockConfigLoader:
    def get_external_api_config(self):
        return {
            'EXTERNAL_API_ENABLED': True,
            'MUSICBRAINZ_ENABLED': True,
            'LASTFM_ENABLED': True,
            'MUSICBRAINZ_USER_AGENT': 'playlista-simple/1.0',
            'MUSICBRAINZ_RATE_LIMIT': 1,
            'LASTFM_API_KEY': '9fd1f789ebdf1297e6aa1590a13d85e0',
            'LASTFM_RATE_LIMIT': 2,
            'METADATA_ENRICHMENT_ENABLED': True,
            'METADATA_ENRICHMENT_TIMEOUT': 30,
            'METADATA_ENRICHMENT_MAX_TAGS': 15,
            'METADATA_ENRICHMENT_RETRY_COUNT': 3
        }

# Create mock config_loader
config_loader = MockConfigLoader()

# Now import the external APIs module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'core'))

# Import the external APIs module directly
import external_apis


def test_musicbrainz_client():
    """Test MusicBrainz client functionality."""
    print("🔍 Testing MusicBrainz client...")
    
    try:
        client = external_apis.MusicBrainzClient()
        
        # Test search for a well-known track
        track = client.search_track("Bohemian Rhapsody", "Queen")
        
        if track:
            print(f"✅ Found track: {track.artist} - {track.title}")
            print(f"   Album: {track.album}")
            print(f"   Release date: {track.release_date}")
            print(f"   Tags: {track.tags[:5] if track.tags else 'None'}")
            return True
        else:
            print("❌ No track found")
            return False
            
    except Exception as e:
        print(f"❌ MusicBrainz test failed: {e}")
        return False


def test_lastfm_client():
    """Test Last.fm client functionality."""
    print("🔍 Testing Last.fm client...")
    
    try:
        client = external_apis.LastFMClient()
        
        # Test getting track info
        track = client.get_track_info("Bohemian Rhapsody", "Queen")
        
        if track:
            print(f"✅ Found track: {track.artist} - {track.name}")
            print(f"   Play count: {track.play_count}")
            print(f"   Listeners: {track.listeners}")
            print(f"   Tags: {track.tags[:5] if track.tags else 'None'}")
            return True
        else:
            print("❌ No track found")
            return False
            
    except Exception as e:
        print(f"❌ Last.fm test failed: {e}")
        return False


def test_metadata_enrichment():
    """Test metadata enrichment service."""
    print("🔍 Testing metadata enrichment service...")
    
    try:
        service = external_apis.MetadataEnrichmentService()
        
        # Test metadata enrichment
        metadata = {
            'title': 'Bohemian Rhapsody',
            'artist': 'Queen',
            'album': 'A Night at the Opera'
        }
        
        enriched = service.enrich_metadata(metadata)
        
        print(f"✅ Original metadata: {list(metadata.keys())}")
        print(f"✅ Enriched metadata: {list(enriched.keys())}")
        
        # Show new fields
        new_fields = set(enriched.keys()) - set(metadata.keys())
        if new_fields:
            print(f"✅ Added fields: {list(new_fields)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Metadata enrichment test failed: {e}")
        return False


def test_api_availability():
    """Test API availability."""
    print("🔍 Testing API availability...")
    
    try:
        service = external_apis.MetadataEnrichmentService()
        available = service.is_available()
        
        if available:
            print("✅ External APIs are available")
            return True
        else:
            print("❌ No external APIs are available")
            return False
            
    except Exception as e:
        print(f"❌ API availability test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🚀 Starting external API integration tests...")
    print("=" * 50)
    
    tests = [
        ("API Availability", test_api_availability),
        ("MusicBrainz Client", test_musicbrainz_client),
        ("Last.fm Client", test_lastfm_client),
        ("Metadata Enrichment", test_metadata_enrichment)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name}...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! External API integration is working correctly.")
        return 0
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 