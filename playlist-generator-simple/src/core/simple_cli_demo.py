"""
Simple CLI Demo - Shows the "1 for all" SingleAnalyzer in action.

This demonstrates how simple audio analysis can be with the SingleAnalyzer.
"""

import os
import sys
from pathlib import Path
from typing import List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.single_analyzer import analyze_file, analyze_files, get_single_analyzer
from core.logging_setup import setup_logging, log_universal


def demo_single_file():
    """Demo analyzing a single file."""
    print("\n🎵 Single File Analysis Demo")
    print("=" * 40)
    
    # Example file path (would be real file in practice)
    file_path = "/music/example.mp3"
    
    print(f"Analyzing: {file_path}")
    
    # The magic - one function call does everything
    result = analyze_file(file_path)
    
    if result['success']:
        print(f"✅ Success!")
        print(f"   Duration: {result['audio_features'].get('duration', 'N/A')}s")
        print(f"   Method: {result['analysis_method']}")
        print(f"   Time: {result['analysis_time']:.2f}s")
        
        # Show detected metadata
        if 'title' in result['metadata']:
            print(f"   Title: {result['metadata']['title']}")
        if 'artist' in result['metadata']:
            print(f"   Artist: {result['metadata']['artist']}")
    else:
        print(f"❌ Failed: {result['error']}")


def demo_batch_analysis():
    """Demo analyzing multiple files."""
    print("\n🎶 Batch Analysis Demo")
    print("=" * 40)
    
    # Example file list (would be real files in practice)
    files = [
        "/music/song1.mp3",
        "/music/song2.flac", 
        "/music/song3.wav",
        "/music/large_file.mp3"
    ]
    
    print(f"Analyzing {len(files)} files...")
    
    # The magic - one function call handles everything
    results = analyze_files(files)
    
    print(f"✅ Batch Complete!")
    print(f"   Total files: {results['total_files']}")
    print(f"   Success: {results['success_count']}")
    print(f"   Failed: {results['failed_count']}")
    print(f"   Success rate: {results['success_rate']:.1f}%")
    print(f"   Total time: {results['total_time']:.2f}s")
    print(f"   Throughput: {results['throughput']:.1f} files/sec")


def demo_analyzer_stats():
    """Demo getting analyzer statistics."""
    print("\n📊 Analyzer Statistics Demo")
    print("=" * 40)
    
    analyzer = get_single_analyzer()
    stats = analyzer.get_statistics()
    
    print("Analyzer Configuration:")
    pipeline_config = stats.get('pipeline_config', {})
    print(f"   Workers: {pipeline_config.get('workers', 'N/A')}")
    print(f"   Optimized range: {pipeline_config.get('optimized_range', 'N/A')}")
    print(f"   Timeout: {pipeline_config.get('timeout', 'N/A')}s")
    
    print("\nDatabase Statistics:")
    print(f"   Total analyzed: {stats.get('total_files_analyzed', 0)}")
    print(f"   Total failed: {stats.get('total_files_failed', 0)}")
    print(f"   Success rate: {stats.get('success_rate', 0):.1f}%")


def demo_different_file_sizes():
    """Demo how different file sizes are handled automatically."""
    print("\n📏 File Size Handling Demo")
    print("=" * 40)
    
    # Simulate different file sizes
    test_cases = [
        ("small_file.mp3", 2.1),      # < 5MB - basic analysis
        ("medium_file.flac", 25.3),   # 5-200MB - optimized pipeline  
        ("large_file.wav", 250.8)     # > 200MB - simplified analysis
    ]
    
    analyzer = get_single_analyzer()
    
    for filename, size_mb in test_cases:
        print(f"\nFile: {filename} ({size_mb}MB)")
        
        if analyzer._should_use_optimized_pipeline(size_mb):
            method = "OptimizedPipeline (chunk-based)"
        elif size_mb < 5:
            method = "Basic analysis (tempo detection)"
        else:
            method = "Simplified analysis (duration only)"
            
        print(f"   → Automatic method: {method}")


def main():
    """Main demo function."""
    # Setup logging
    setup_logging()
    
    print("🎯 SingleAnalyzer - The '1 for All' Audio Analysis Demo")
    print("=" * 60)
    print("\nThis demo shows how ONE ANALYZER handles everything automatically:")
    print("• Automatic file size detection and optimization")
    print("• Built-in metadata extraction and enrichment") 
    print("• Intelligent caching and database integration")
    print("• Batch processing with optimal workers")
    print("• Complete feature extraction (Essentia + MusiCNN)")
    print("• Progress tracking and statistics")
    
    # Run demos
    demo_different_file_sizes()
    demo_single_file()
    demo_batch_analysis()
    demo_analyzer_stats()
    
    print("\n" + "=" * 60)
    print("🎉 Demo Complete!")
    print("\nKey Points:")
    print("• Just call analyze_file() or analyze_files()")
    print("• Everything else happens automatically")
    print("• No configuration needed")
    print("• One analyzer handles all file sizes optimally")
    print("• Maximum performance with minimum complexity")
    print("\nThis is TRUE '1 for All' audio analysis! ✅")


if __name__ == "__main__":
    main()

