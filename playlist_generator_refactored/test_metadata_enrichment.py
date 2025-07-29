#!/usr/bin/env python3
"""
Test script for real MetadataEnrichmentService implementation.
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


def test_metadata_enrichment():
    """Test the real MetadataEnrichmentService implementation."""
    
    print("🧪 Starting metadata enrichment test...")
    
    # Setup logging
    config = get_config()
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
        
        logger.info(f"📊 Enrichment result: {response.status}")
        logger.info(f"📊 Processing time: {response.processing_time_ms}ms")
        logger.info(f"📊 Errors: {len(response.errors)}")
        logger.info(f"📊 Successful files: {response.successful_files}")
        logger.info(f"📊 Failed files: {response.failed_files}")
        
        if response.enriched_metadata:
            enriched = response.enriched_metadata[0]
            logger.info(f"🎵 Enriched confidence: {enriched.confidence}")
            logger.info(f"🎵 Enriched source: {enriched.source}")
            
            # Show MusicBrainz data
            if enriched.musicbrainz_track_id:
                logger.info(f"🎵 MusicBrainz track ID: {enriched.musicbrainz_track_id}")
            if enriched.musicbrainz_artist_id:
                logger.info(f"🎵 MusicBrainz artist ID: {enriched.musicbrainz_artist_id}")
            
            # Show Last.fm data
            if enriched.lastfm_tags:
                logger.info(f"🎵 Last.fm tags: {enriched.lastfm_tags}")
            if enriched.lastfm_playcount:
                logger.info(f"🎵 Last.fm play count: {enriched.lastfm_playcount}")
        
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
    success = test_metadata_enrichment()
    sys.exit(0 if success else 1) 