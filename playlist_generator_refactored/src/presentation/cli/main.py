#!/usr/bin/env python3
"""
Main CLI entry point for the refactored playlista application.
"""

import sys
import logging
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from shared.config import get_config
from shared.exceptions import PlaylistaException
from infrastructure.logging import setup_logging
from application.services.audio_analysis_service import AudioAnalysisService
from application.services.file_discovery_service import FileDiscoveryService


def main():
    """Main CLI entry point."""
    try:
        # Setup logging
        config = get_config()
        setup_logging(config.logging)
        
        logger = logging.getLogger(__name__)
        logger.info("🎵 Playlista Refactored - Starting up...")
        
        # Initialize services
        file_discovery = FileDiscoveryService()
        audio_analysis = AudioAnalysisService()
        
        logger.info("✅ Services initialized successfully")
        
        # For now, just show that everything is working
        logger.info("🚀 Ready for real implementation!")
        logger.info("📁 FileDiscoveryService: Ready")
        logger.info("🎼 AudioAnalysisService: Ready")
        
        return 0
        
    except PlaylistaException as e:
        logging.error(f"Playlista error: {e}")
        return 1
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 