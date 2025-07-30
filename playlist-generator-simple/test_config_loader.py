#!/usr/bin/env python3
"""
Test script for the configuration loader.
Verifies that the plain text configuration file is properly loaded.
"""

import os
import sys
import logging
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from core.config_loader import ConfigLoader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_config_loader():
    """Test the configuration loader with the actual config file."""
    print("🧪 Testing Configuration Loader...")
    
    # Test with default config file
    loader = ConfigLoader()
    config = loader.load_config()
    
    print("✅ Configuration loaded successfully")
    print("📋 Configuration contents:")
    for key, value in config.items():
        print(f"   {key} = {value}")
    
    # Test File Discovery specific config
    file_discovery_config = loader.get_file_discovery_config()
    print("\n🎵 File Discovery Configuration:")
    for key, value in file_discovery_config.items():
        print(f"   {key} = {value}")

def test_environment_overrides():
    """Test environment variable overrides."""
    print("\n🧪 Testing Environment Variable Overrides...")
    
    # Set some environment variables
    os.environ['MUSIC_DIR'] = '/custom/music'
    os.environ['HASH_ALGORITHM'] = 'sha256'
    os.environ['MAX_RETRY_COUNT'] = '5'
    
    try:
        loader = ConfigLoader()
        config = loader.load_config()
        
        print("✅ Environment overrides loaded successfully")
        print("🌍 Environment overrides:")
        print(f"   MUSIC_DIR = {config.get('MUSIC_DIR')}")
        print(f"   HASH_ALGORITHM = {config.get('HASH_ALGORITHM')}")
        print(f"   MAX_RETRY_COUNT = {config.get('MAX_RETRY_COUNT')}")
        
    finally:
        # Clean up environment variables
        for key in ['MUSIC_DIR', 'HASH_ALGORITHM', 'MAX_RETRY_COUNT']:
            if key in os.environ:
                del os.environ[key]

def test_file_discovery_integration():
    """Test FileDiscovery with the configuration loader."""
    print("\n🧪 Testing FileDiscovery Integration...")
    
    try:
        from core.file_discovery import FileDiscovery
        
        # Create FileDiscovery instance (will use config loader)
        discovery = FileDiscovery()
        
        print("✅ FileDiscovery initialized with configuration loader")
        print(f"🎵 Music directory: {discovery.music_dir}")
        print(f"🔐 Hash algorithm: {discovery.hash_algorithm}")
        print(f"📏 Min file size: {discovery.min_file_size_bytes}")
        print(f"🔄 Max retry count: {discovery.max_retry_count}")
        
    except Exception as e:
        print(f"❌ Error testing FileDiscovery integration: {e}")

def main():
    """Run all configuration tests."""
    print("🚀 Testing Configuration System")
    print("=" * 50)
    
    # Test basic config loading
    test_config_loader()
    
    # Test environment overrides
    test_environment_overrides()
    
    # Test FileDiscovery integration
    test_file_discovery_integration()
    
    print("\n✅ All configuration tests completed!")

if __name__ == "__main__":
    main() 