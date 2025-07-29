#!/usr/bin/env python3
"""
Quick test for logging infrastructure.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from shared.config import get_config
from infrastructure.logging import setup_logging, get_logger

def main():
    print("🚀 Quick Logging Test")
    print("=" * 30)
    
    # Setup logging
    config = get_config()
    logger = setup_logging(config)
    
    # Test basic logging
    logger.info("Logging infrastructure is working!")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    print("\n🎉 Logging test completed successfully!")
    return True

if __name__ == "__main__":
    main() 