#!/usr/bin/env python3
"""
Basic test script to verify the CLI entry point works with original arguments.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def test_cli_help():
    """Test that the CLI shows help correctly."""
    print("🧪 Testing CLI help...")
    
    try:
        # Import the CLI
        from presentation.cli.cli_interface import CLIInterface
        
        cli = CLIInterface()
        
        # Test help display
        print("✅ CLI interface imported successfully")
        
        # Test argument parser creation
        parser = cli._create_argument_parser()
        print("✅ Argument parser created successfully")
        
        # Test help text
        help_text = parser.format_help()
        if "Playlista" in help_text:
            print("✅ Help text contains expected content")
        else:
            print("❌ Help text missing expected content")
            
        return True
        
    except Exception as e:
        print(f"❌ CLI help test failed: {e}")
        return False

def test_argument_parsing():
    """Test that original arguments are parsed correctly."""
    print("🧪 Testing argument parsing...")
    
    try:
        from presentation.cli.cli_interface import CLIInterface
        
        cli = CLIInterface()
        parser = cli._create_argument_parser()
        
        # Test original arguments
        test_args = [
            ['--help'],
            ['--analyze', '--workers', '4'],
            ['--generate_only', '--playlist_method', 'kmeans'],
            ['--update'],
            ['--status'],
            ['--pipeline'],
            ['--fast_mode', '--memory_aware', '--memory_limit', '2GB'],
            ['--failed', '--force'],
            ['--no_cache'],
            ['--min_tracks_per_genre', '15']
        ]
        
        for args in test_args:
            try:
                parsed = parser.parse_args(args)
                print(f"✅ Parsed arguments: {args}")
            except Exception as e:
                print(f"❌ Failed to parse arguments {args}: {e}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Argument parsing test failed: {e}")
        return False

def test_service_initialization():
    """Test that all services initialize correctly."""
    print("🧪 Testing service initialization...")
    
    try:
        from application.services.audio_analysis_service import AudioAnalysisService
        from application.services.file_discovery_service import FileDiscoveryService
        from application.services.metadata_enrichment_service import MetadataEnrichmentService
        from application.services.playlist_generation_service import PlaylistGenerationService
        
        # Initialize services
        discovery = FileDiscoveryService()
        analysis = AudioAnalysisService()
        enrichment = MetadataEnrichmentService()
        playlist = PlaylistGenerationService()
        
        print("✅ All services initialized successfully")
        return True
        
    except Exception as e:
        print(f"❌ Service initialization test failed: {e}")
        return False

def test_config_loading():
    """Test that configuration loads correctly."""
    print("🧪 Testing configuration loading...")
    
    try:
        from shared.config import get_config
        
        config = get_config()
        
        # Check that config has expected attributes
        expected_attrs = ['logging', 'processing', 'memory', 'database']
        for attr in expected_attrs:
            if hasattr(config, attr):
                print(f"✅ Config has {attr} section")
            else:
                print(f"❌ Config missing {attr} section")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_dto_creation():
    """Test that DTOs can be created correctly."""
    print("🧪 Testing DTO creation...")
    
    try:
        from application.dtos.audio_analysis import AudioAnalysisRequest
        from application.dtos.file_discovery import FileDiscoveryRequest
        from application.dtos.metadata_enrichment import MetadataEnrichmentRequest
        from application.dtos.playlist_generation import PlaylistGenerationRequest
        from pathlib import Path
        
        # Test DTO creation
        analysis_request = AudioAnalysisRequest(
            file_paths=[Path('/test/music')],
            fast_mode=True,
            parallel_processing=True,
            max_workers=4
        )
        
        discovery_request = FileDiscoveryRequest(
            search_paths=[Path('/test/music')],
            recursive=True
        )
        
        enrichment_request = MetadataEnrichmentRequest(
            search_paths=[Path('/test/music')],
            use_musicbrainz=True
        )
        
        playlist_request = PlaylistGenerationRequest(
            method='kmeans',
            playlist_size=20,
            num_playlists=8
        )
        
        print("✅ All DTOs created successfully")
        return True
        
    except Exception as e:
        print(f"❌ DTO creation test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Starting Basic CLI Tests")
    print("=" * 50)
    
    tests = [
        test_cli_help,
        test_argument_parsing,
        test_service_initialization,
        test_config_loading,
        test_dto_creation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Basic CLI functionality is working.")
        return 0
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 