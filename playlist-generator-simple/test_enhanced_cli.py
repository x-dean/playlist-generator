#!/usr/bin/env python3
"""
Test script for the Enhanced CLI functionality.
Demonstrates all available commands and variants.
"""

import os
import sys
import subprocess
import tempfile
import shutil
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def run_command(cmd, description=""):
    """Run a command and display the result."""
    print(f"\n{'='*60}")
    print(f"🔧 Testing: {description}")
    print(f"📝 Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        print(f"✅ Exit code: {result.returncode}")
        if result.stdout:
            print("📤 Output:")
            print(result.stdout)
        if result.stderr:
            print("⚠️ Errors:")
            print(result.stderr)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("⏰ Command timed out")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_help_commands():
    """Test help and documentation commands."""
    print("\n🎯 Testing Help and Documentation Commands")
    
    # Test main help
    run_command(["python", "src/enhanced_cli.py"], "Main help")
    
    # Test command-specific help
    run_command(["python", "src/enhanced_cli.py", "analyze", "--help"], "Analyze help")
    run_command(["python", "src/enhanced_cli.py", "playlist", "--help"], "Playlist help")
    run_command(["python", "src/enhanced_cli.py", "pipeline", "--help"], "Pipeline help")

def test_configuration_commands():
    """Test configuration management commands."""
    print("\n🎯 Testing Configuration Commands")
    
    # Test config command
    run_command(["python", "src/enhanced_cli.py", "config"], "Show configuration")
    run_command(["python", "src/enhanced_cli.py", "config", "--json"], "Show JSON configuration")
    run_command(["python", "src/enhanced_cli.py", "config", "--validate"], "Validate configuration")

def test_playlist_methods():
    """Test playlist generation methods."""
    print("\n🎯 Testing Playlist Generation Methods")
    
    # Test playlist methods command
    run_command(["python", "src/enhanced_cli.py", "playlist-methods"], "List playlist methods")
    
    # Test different playlist generation methods
    methods = ["kmeans", "similarity", "random", "time_based", "tag_based", "cache_based", "feature_group", "mixed", "all"]
    
    for method in methods[:3]:  # Test first 3 methods to avoid long execution
        run_command([
            "python", "src/enhanced_cli.py", "playlist",
            "--method", method,
            "--num-playlists", "2",
            "--playlist-size", "5"
        ], f"Generate playlists using {method} method")

def test_statistics_commands():
    """Test statistics and monitoring commands."""
    print("\n🎯 Testing Statistics and Monitoring Commands")
    
    # Test stats command
    run_command(["python", "src/enhanced_cli.py", "stats"], "Show basic statistics")
    run_command(["python", "src/enhanced_cli.py", "stats", "--detailed"], "Show detailed statistics")
    run_command(["python", "src/enhanced_cli.py", "stats", "--memory-usage"], "Show memory usage")
    
    # Test status command
    run_command(["python", "src/enhanced_cli.py", "status"], "Show system status")
    run_command(["python", "src/enhanced_cli.py", "status", "--detailed"], "Show detailed status")
    
    # Test cleanup command
    run_command(["python", "src/enhanced_cli.py", "cleanup", "--max-retries", "3"], "Cleanup failed files")

def test_discovery_commands():
    """Test file discovery commands."""
    print("\n🎯 Testing File Discovery Commands")
    
    # Create a temporary test directory with some files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create some dummy files
        test_files = [
            "test1.mp3",
            "test2.flac", 
            "test3.wav",
            "subdir/test4.mp3"
        ]
        
        for file_path in test_files:
            full_path = os.path.join(temp_dir, file_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, 'w') as f:
                f.write("dummy content")
        
        # Test discover command
        run_command([
            "python", "src/enhanced_cli.py", "discover",
            "--path", temp_dir,
            "--recursive"
        ], "Discover files recursively")
        
        run_command([
            "python", "src/enhanced_cli.py", "discover",
            "--path", temp_dir,
            "--extensions", "mp3,flac",
            "--recursive"
        ], "Discover files with specific extensions")

def test_analysis_commands():
    """Test analysis commands with different options."""
    print("\n🎯 Testing Analysis Commands")
    
    # Test basic analysis (will likely fail without real music files, but shows command structure)
    run_command([
        "python", "src/enhanced_cli.py", "analyze",
        "--music-path", "/nonexistent/path"
    ], "Basic analysis (expected to fail)")
    
    # Test analysis with different options
    run_command([
        "python", "src/enhanced_cli.py", "analyze",
        "--music-path", "/nonexistent/path",
        "--fast-mode",
        "--parallel",
        "--workers", "2"
    ], "Analysis with fast mode and parallel processing")
    
    run_command([
        "python", "src/enhanced_cli.py", "analyze",
        "--music-path", "/nonexistent/path",
        "--memory-aware",
        "--memory-limit", "1GB"
    ], "Analysis with memory awareness")

def test_export_commands():
    """Test export functionality."""
    print("\n🎯 Testing Export Commands")
    
    # Test export with different formats
    formats = ["m3u", "pls", "xspf", "json"]
    
    for format_type in formats:
        run_command([
            "python", "src/enhanced_cli.py", "export",
            "--playlist-file", "nonexistent.json",
            "--format", format_type
        ], f"Export in {format_type} format")

def test_pipeline_commands():
    """Test pipeline commands."""
    print("\n🎯 Testing Pipeline Commands")
    
    # Test pipeline command
    run_command([
        "python", "src/enhanced_cli.py", "pipeline",
        "--music-path", "/nonexistent/path",
        "--force"
    ], "Pipeline with force option")
    
    run_command([
        "python", "src/enhanced_cli.py", "pipeline",
        "--music-path", "/nonexistent/path",
        "--generate",
        "--export"
    ], "Pipeline with generate and export")

def test_monitoring_commands():
    """Test monitoring commands."""
    print("\n🎯 Testing Monitoring Commands")
    
    # Test monitor command (short duration)
    run_command([
        "python", "src/enhanced_cli.py", "monitor",
        "--duration", "5"
    ], "Monitor resources for 5 seconds")

def test_metadata_enrichment():
    """Test metadata enrichment commands."""
    print("\n🎯 Testing Metadata Enrichment Commands")
    
    # Test enrich command
    run_command([
        "python", "src/enhanced_cli.py", "enrich",
        "--path", "/nonexistent/path",
        "--musicbrainz"
    ], "Enrich metadata with MusicBrainz")
    
    run_command([
        "python", "src/enhanced_cli.py", "enrich",
        "--audio-ids", "1,2,3",
        "--lastfm"
    ], "Enrich specific audio IDs with Last.fm")

def test_processing_modes():
    """Test different processing modes and options."""
    print("\n🎯 Testing Processing Modes")
    
    # Test different processing options
    processing_tests = [
        (["--fast-mode"], "Fast mode"),
        (["--parallel", "--workers", "2"], "Parallel processing"),
        (["--sequential"], "Sequential processing"),
        (["--memory-aware", "--memory-limit", "512MB"], "Memory-aware processing"),
        (["--low-memory", "--rss-limit-gb", "2.0"], "Low memory mode"),
        (["--force", "--no-cache"], "Force re-analysis with no cache"),
        (["--failed"], "Re-analyze failed files"),
        (["--large-file-threshold", "100"], "Large file threshold"),
        (["--batch-size", "50"], "Batch processing"),
        (["--timeout", "600"], "Custom timeout")
    ]
    
    for options, description in processing_tests:
        run_command([
            "python", "src/enhanced_cli.py", "analyze",
            "--music-path", "/nonexistent/path"
        ] + options, f"Analysis with {description}")

def test_logging_options():
    """Test different logging levels and options."""
    print("\n🎯 Testing Logging Options")
    
    # Test different log levels
    log_levels = ["DEBUG", "INFO", "WARNING", "ERROR"]
    
    for level in log_levels:
        run_command([
            "python", "src/enhanced_cli.py", "stats",
            "--log-level", level
        ], f"Statistics with {level} logging")
    
    # Test verbose and quiet options
    run_command([
        "python", "src/enhanced_cli.py", "stats",
        "--verbose"
    ], "Statistics with verbose output")
    
    run_command([
        "python", "src/enhanced_cli.py", "stats",
        "--quiet"
    ], "Statistics with quiet output")

def main():
    """Run all CLI tests."""
    print("🎵 Enhanced CLI Test Suite")
    print("=" * 60)
    print("This script tests all available CLI commands and variants.")
    print("Note: Some commands will fail without real music files, but this")
    print("demonstrates the command structure and options available.")
    print("=" * 60)
    
    # Test all command categories
    test_help_commands()
    test_configuration_commands()
    test_playlist_methods()
    test_statistics_commands()
    test_discovery_commands()
    test_analysis_commands()
    test_export_commands()
    test_pipeline_commands()
    test_monitoring_commands()
    test_metadata_enrichment()
    test_processing_modes()
    test_logging_options()
    
    print("\n" + "=" * 60)
    print("✅ Enhanced CLI Test Suite Complete")
    print("=" * 60)
    print("\n📋 Summary of Available Commands:")
    print("  • analyze - Audio file analysis with multiple processing modes")
    print("  • stats - Statistics and monitoring")
    print("  • playlist - Playlist generation with 13 different methods")
    print("  • discover - File discovery with filtering options")
    print("  • enrich - Metadata enrichment from external APIs")
    print("  • export - Export playlists in multiple formats")
    print("  • status - System and database status")
    print("  • pipeline - Complete analysis and generation pipeline")
    print("  • config - Configuration management")
    print("  • monitor - Real-time resource monitoring")
    print("  • cleanup - Cleanup failed analysis")
    print("  • test-audio - Test audio analyzer")
    print("  • playlist-methods - List available playlist methods")
    
    print("\n🚀 Key Features:")
    print("  • Memory-aware processing with Docker optimization")
    print("  • Fast mode (3-5x faster processing)")
    print("  • Parallel and sequential processing modes")
    print("  • Advanced audio features (MusiCNN, emotional features)")
    print("  • Multiple playlist generation methods")
    print("  • Database management and validation")
    print("  • Metadata enrichment from external APIs")
    print("  • Export functionality in multiple formats")
    
    print("\n📖 For detailed usage:")
    print("  python src/enhanced_cli.py --help")
    print("  python src/enhanced_cli.py <command> --help")

if __name__ == "__main__":
    main() 