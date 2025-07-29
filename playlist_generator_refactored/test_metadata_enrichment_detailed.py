#!/usr/bin/env python3
"""
Detailed test script for real MetadataEnrichmentService implementation.
"""

import sys
import logging
from pathlib import Path
import tempfile
import os
from uuid import uuid4

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from shared.config import get_config
from infrastructure.logging import setup_logging
from application.services.metadata_enrichment_service import MetadataEnrichmentService
from application.dtos.metadata_enrichment import MetadataEnrichmentRequest, EnrichmentSource
from domain.entities.metadata import Metadata


def test_metadata_enrichment_detailed():
    """Test the real MetadataEnrichmentService implementation with detailed output."""
    
    print("🧪 Starting detailed metadata enrichment test...")
    
    # Setup logging with DEBUG level
    config = get_config()
    config.logging.level = "DEBUG"
    setup_logging(config)
    logger = logging.getLogger(__name__)
    
    logger.info("🧪 Testing real MetadataEnrichmentService implementation...")
    
    try:
        # Initialize service
        logger.info("📦 Initializing MetadataEnrichmentService...")
        enrichment_service = MetadataEnrichmentService()
        
        logger.info("✅ Service initialized successfully")
        
        # Create test audio file ID
        test_audio_file_id = uuid4()
        print(f"🎵 Test audio file ID: {test_audio_file_id}")
        
        # Create enrichment request
        logger.info("🔍 Creating enrichment request...")
        request = MetadataEnrichmentRequest(
            audio_file_ids=[test_audio_file_id],
            sources=[EnrichmentSource.MUSICBRAINZ, EnrichmentSource.LASTFM]
        )
        
        # Perform enrichment
        logger.info("🚀 Starting metadata enrichment...")
        print("🚀 Calling enrich_metadata...")
        response = enrichment_service.enrich_metadata(request)
        print("✅ enrich_metadata completed")
        
        # Print detailed results
        print(f"\n📊 ENRICHMENT RESULTS:")
        print(f"📊 Status: {response.status}")
        print(f"📊 Processing time: {response.processing_time_ms}ms")
        print(f"📊 Total files: {response.total_files}")
        print(f"📊 Successful files: {response.successful_files}")
        print(f"📊 Failed files: {response.failed_files}")
        print(f"📊 Errors: {len(response.errors)}")
        
        if response.enriched_metadata:
            enriched = response.enriched_metadata[0]
            print(f"\n🎵 ENRICHED METADATA:")
            print(f"🎵 Title: {enriched.title}")
            print(f"🎵 Artist: {enriched.artist}")
            print(f"🎵 Album: {enriched.album}")
            print(f"🎵 Genre: {enriched.genre}")
            print(f"🎵 Year: {enriched.year}")
            print(f"🎵 Confidence: {enriched.confidence}")
            print(f"🎵 Source: {enriched.source}")
            
            # Show MusicBrainz data
            if enriched.musicbrainz_track_id:
                print(f"🎵 MusicBrainz track ID: {enriched.musicbrainz_track_id}")
            if enriched.musicbrainz_artist_id:
                print(f"🎵 MusicBrainz artist ID: {enriched.musicbrainz_artist_id}")
            if enriched.musicbrainz_album_id:
                print(f"🎵 MusicBrainz album ID: {enriched.musicbrainz_album_id}")
            
            # Show Last.fm data
            if enriched.lastfm_tags:
                print(f"🎵 Last.fm tags: {enriched.lastfm_tags}")
            if enriched.lastfm_playcount:
                print(f"🎵 Last.fm play count: {enriched.lastfm_playcount}")
            if enriched.lastfm_rating:
                print(f"🎵 Last.fm rating: {enriched.lastfm_rating}")
        
        if response.errors:
            print(f"\n❌ ERRORS:")
            for error in response.errors:
                print(f"❌ {error}")
        
        logger.info("🎉 Metadata enrichment test completed successfully!")
        print("🎉 Test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_metadata_enrichment_detailed()
    sys.exit(0 if success else 1) 