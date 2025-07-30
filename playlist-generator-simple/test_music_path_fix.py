#!/usr/bin/env python3
"""
Test script to verify that the music directory path fix is working correctly.
This script tests the file discovery functionality with the corrected paths.
"""

import os
import sys
import logging

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from core.file_discovery import FileDiscovery
from core.config_loader import ConfigLoader

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_music_directory_path():
    """Test that the music directory path is correctly configured."""
    logger.info("🔍 Testing music directory path configuration...")
    
    try:
        # Initialize file discovery
        file_discovery = FileDiscovery()
        
        # Check the configured music directory
        music_dir = file_discovery.music_dir
        logger.info(f"📁 Configured music directory: {music_dir}")
        
        # Check if the directory exists
        if os.path.exists(music_dir):
            logger.info(f"✅ Music directory exists: {music_dir}")
            
            # List files in the directory
            files = os.listdir(music_dir)
            logger.info(f"📄 Found {len(files)} files in music directory:")
            for file in files:
                logger.info(f"   - {file}")
                
            # Test file discovery
            discovered_files = file_discovery.discover_files()
            logger.info(f"🎵 Discovered {len(discovered_files)} audio files")
            
            if discovered_files:
                logger.info("✅ File discovery is working correctly!")
                return True
            else:
                logger.warning("⚠️ No audio files discovered - check file extensions")
                return False
        else:
            logger.error(f"❌ Music directory does not exist: {music_dir}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error testing music directory: {e}")
        return False

def test_configuration():
    """Test that the configuration is loading correctly."""
    logger.info("🔍 Testing configuration loading...")
    
    try:
        # Load configuration
        config_loader = ConfigLoader()
        config = config_loader.load_config()
        
        # Check music path configuration
        music_path = config.get('MUSIC_PATH', '/music')
        logger.info(f"📋 Configured MUSIC_PATH: {music_path}")
        
        # Check MusiCNN model paths
        musicnn_model_path = config.get('MUSICNN_MODEL_PATH', '/app/models/musicnn_model.pb')
        musicnn_json_path = config.get('MUSICNN_JSON_PATH', '/app/models/musicnn_features.json')
        
        logger.info(f"📋 MusiCNN model path: {musicnn_model_path}")
        logger.info(f"📋 MusiCNN JSON path: {musicnn_json_path}")
        
        # Check if model files exist (optional)
        if os.path.exists(musicnn_model_path):
            logger.info("✅ MusiCNN model file exists")
        else:
            logger.warning("⚠️ MusiCNN model file not found (this is expected if models are not provided)")
            
        if os.path.exists(musicnn_json_path):
            logger.info("✅ MusiCNN JSON file exists")
        else:
            logger.warning("⚠️ MusiCNN JSON file not found (this is expected if models are not provided)")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error testing configuration: {e}")
        return False

def main():
    """Run all tests."""
    logger.info("🚀 Starting music directory path fix tests...")
    
    # Test configuration
    config_ok = test_configuration()
    
    # Test music directory
    music_ok = test_music_directory_path()
    
    # Summary
    logger.info("📊 Test Results:")
    logger.info(f"   Configuration: {'✅ PASS' if config_ok else '❌ FAIL'}")
    logger.info(f"   Music Directory: {'✅ PASS' if music_ok else '❌ FAIL'}")
    
    if config_ok and music_ok:
        logger.info("🎉 All tests passed! The music directory path fix is working correctly.")
        return 0
    else:
        logger.error("❌ Some tests failed. Please check the configuration.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 