"""
Simple CLI interface for analysis components in Playlist Generator Simple.
Provides commands to test and use the analysis functionality.
"""

import os
import sys
import argparse
from pathlib import Path

# Add core to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core'))

from core.analysis_manager import AnalysisManager
from core.resource_manager import ResourceManager
from core.audio_analyzer import AudioAnalyzer
from core.database import DatabaseManager
from core.playlist_generator import PlaylistGenerator, PlaylistGenerationMethod
from core.logging_setup import get_logger, setup_logging
from core.config_loader import config_loader

# Initialize logging system from config
config = config_loader.load_config()
logging_config = config_loader.get_logging_config()

# Use config values or defaults
log_level = logging_config.get('LOG_LEVEL', 'INFO')
log_file_prefix = logging_config.get('LOG_FILE_PREFIX', 'playlista_analysis')
console_logging = logging_config.get('LOG_CONSOLE_ENABLED', True)
file_logging = logging_config.get('LOG_FILE_ENABLED', True)
colored_output = logging_config.get('LOG_COLORED_OUTPUT', True)
max_log_files = logging_config.get('LOG_MAX_FILES', 10)
log_file_size_mb = logging_config.get('LOG_FILE_SIZE_MB', 50)
log_file_format = logging_config.get('LOG_FILE_FORMAT', 'text')
log_file_encoding = logging_config.get('LOG_FILE_ENCODING', 'utf-8')

setup_logging(
    log_level=log_level,
    log_file_prefix=log_file_prefix,
    console_logging=console_logging,
    file_logging=file_logging,
    colored_output=colored_output,
    max_log_files=max_log_files,
    log_file_size_mb=log_file_size_mb,
    log_file_format=log_file_format,
    log_file_encoding=log_file_encoding
)

logger = get_logger('playlista.analysis_cli')


def analyze_files(args):
    """Analyze audio files using the analysis manager."""
    logger.info("Starting file analysis")
    
    try:
        # Create analysis manager
        analysis_manager = AnalysisManager()
        
        # Select files for analysis
        files = analysis_manager.select_files_for_analysis(
            music_path=args.music_path,
            force_reextract=args.force,
            include_failed=args.include_failed
        )
        
        if not files:
            logger.warning("️ No files found for analysis")
            return
        
        logger.info(f"Found {len(files)} files to analyze")
        
        # Analyze files
        results = analysis_manager.analyze_files(
            files=files,
            force_reextract=args.force,
            max_workers=args.workers
        )
        
        # Display results
        logger.info("Analysis Results:")
        logger.info(f"  Success: {results['success_count']}")
        logger.info(f"  Failed: {results['failed_count']}")
        logger.info(f"  Total time: {results['total_time']:.2f}s")
        
        if 'big_files_processed' in results:
            logger.info(f"  Large files: {results['big_files_processed']}")
        if 'small_files_processed' in results:
            logger.info(f"  Small files: {results['small_files_processed']}")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")


def show_statistics(args):
    """Show analysis and resource statistics."""
    logger.info("Showing statistics")
    
    try:
        # Create managers
        analysis_manager = AnalysisManager()
        resource_manager = ResourceManager()
        db_manager = DatabaseManager()
        
        # Analysis statistics
        analysis_stats = analysis_manager.get_analysis_statistics()
        logger.info("Analysis Statistics:")
        for key, value in analysis_stats.items():
            if key != 'database_stats':
                logger.info(f"  {key}: {value}")
        
        # Resource statistics
        resource_stats = resource_manager.get_resource_statistics(minutes=60)
        logger.info("Resource Statistics (last hour):")
        for category, stats in resource_stats.items():
            if category != 'period_minutes' and category != 'data_points':
                logger.info(f"  {category}:")
                for metric, value in stats.items():
                    logger.info(f"    {metric}: {value:.2f}")
        
        # Database statistics
        db_stats = db_manager.get_database_statistics()
        logger.info("Database Statistics:")
        for key, value in db_stats.items():
            logger.info(f"  {key}: {value}")
        
    except Exception as e:
        logger.error(f"Error getting statistics: {e}")


def test_audio_analyzer(args):
    """Test the audio analyzer with a specific file."""
    logger.info("Testing audio analyzer")
    
    if not args.file:
        logger.error("Please specify a file with --file")
        return
    
    if not os.path.exists(args.file):
        logger.error(f"File not found: {args.file}")
        return
    
    try:
        # Create audio analyzer
        audio_analyzer = AudioAnalyzer()
        
        # Extract features
        result = audio_analyzer.extract_features(args.file, force_reextract=args.force)
        
        if result:
            logger.info("Feature extraction successful")
            logger.info(f"Features extracted: {len(result.get('features', {}))}")
            
            # Show some key features
            features = result.get('features', {})
            if 'bpm' in features:
                logger.info(f"BPM: {features['bpm']}")
            if 'loudness' in features:
                logger.info(f"Loudness: {features['loudness']}")
            if 'key' in features:
                logger.info(f"Key: {features['key']}")
            
            # Show metadata
            metadata = result.get('metadata', {})
            if metadata:
                logger.info("Metadata:")
                for key, value in metadata.items():
                    logger.info(f"  {key}: {value}")
        else:
            logger.error("Feature extraction failed")
        
    except Exception as e:
        logger.error(f"Audio analyzer test failed: {e}")


def monitor_resources(args):
    """Monitor system resources in real-time."""
    logger.info("Starting resource monitoring")
    
    try:
        # Create resource manager
        resource_manager = ResourceManager()
        
        # Start monitoring
        resource_manager.start_monitoring()
        
        logger.info("Resource monitoring started (press Ctrl+C to stop)")
        logger.info("Monitoring interval: {} seconds".format(
            resource_manager.monitoring_interval
        ))
        
        # Monitor for specified duration or until interrupted
        import time
        try:
            if args.duration:
                time.sleep(args.duration)
            else:
                # Monitor indefinitely
                while True:
                    time.sleep(5)
                    resources = resource_manager.get_current_resources()
                    if 'memory' in resources:
                        memory = resources['memory']
                        logger.info(f"Memory: {memory['used_gb']:.2f}GB used ({memory['percent']:.1f}%)")
                    if 'cpu_percent' in resources:
                        logger.info(f"️ CPU: {resources['cpu_percent']:.1f}%")
        except KeyboardInterrupt:
            logger.info("Stopping resource monitoring")
        finally:
            resource_manager.stop_monitoring()
        
    except Exception as e:
        logger.error(f"Resource monitoring failed: {e}")


def cleanup_failed(args):
    """Clean up failed analysis entries."""
    logger.info("Cleaning up failed analysis entries")
    
    try:
        # Create analysis manager
        analysis_manager = AnalysisManager()
        
        # Clean up failed analysis
        cleaned_count = analysis_manager.cleanup_failed_analysis(
            max_retries=args.max_retries
        )
        
        logger.info(f"Cleaned up {cleaned_count} failed analysis entries")
        
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")


def generate_playlists(args):
    """Generate playlists using analyzed tracks."""
    logger.info("Starting playlist generation")
    
    try:
        # Create playlist generator
        playlist_generator = PlaylistGenerator()
        
        # Generate playlists
        playlists = playlist_generator.generate_playlists(
            method=args.method,
            num_playlists=args.num_playlists,
            playlist_size=args.playlist_size
        )
        
        if not playlists:
            logger.warning("️ No playlists generated")
            return
        
        # Display results
        logger.info("Playlist Generation Results:")
        logger.info(f"  Total playlists: {len(playlists)}")
        
        for name, playlist in playlists.items():
            logger.info(f"  {name}: {playlist.size} tracks")
        
        # Save playlists if requested
        if args.save:
            output_dir = args.output_dir or 'playlists'
            success = playlist_generator.save_playlists(playlists, output_dir)
            if success:
                logger.info(f"Playlists saved to {output_dir}")
            else:
                logger.error(f"Failed to save playlists")
        
        # Show statistics
        stats = playlist_generator.get_playlist_statistics(playlists)
        logger.info("Playlist Statistics:")
        logger.info(f"  Total tracks: {stats.get('total_tracks', 0)}")
        logger.info(f"  Average size: {stats.get('average_playlist_size', 0):.1f}")
        logger.info(f"  Size range: {stats.get('min_playlist_size', 0)} - {stats.get('max_playlist_size', 0)}")
        
    except Exception as e:
        logger.error(f"Playlist generation failed: {e}")


def list_playlist_methods(args):
    """List available playlist generation methods."""
    logger.info("Available playlist generation methods:")
    
    methods = [
        ("kmeans", "K-means clustering based on audio features"),
        ("similarity", "Similarity-based track selection"),
        ("random", "Random track selection"),
        ("time_based", "Time-based scheduling (morning, afternoon, etc.)"),
        ("tag_based", "Tag-based selection using metadata"),
        ("cache_based", "Cache-based selection using previous playlists"),
        ("feature_group", "Feature group-based selection"),
        ("mixed", "Combination of multiple methods"),
        ("all", "Generate playlists using all available methods")
    ]
    
    for method, description in methods:
        logger.info(f"  {method}: {description}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Playlist Generator Simple - Analysis CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analysis_cli.py analyze --music-path /path/to/music
  python analysis_cli.py stats
  python analysis_cli.py test-audio --file /path/to/audio.mp3
  python analysis_cli.py monitor --duration 60
  python analysis_cli.py cleanup --max-retries 3
  python analysis_cli.py playlist --method kmeans --num-playlists 5 --save
  python analysis_cli.py playlist-methods
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze audio files')
    analyze_parser.add_argument('--music-path', default='/music', 
                               help='Path to music directory')
    analyze_parser.add_argument('--force', action='store_true',
                               help='Force re-analysis of all files')
    analyze_parser.add_argument('--include-failed', action='store_true',
                               help='Include previously failed files')
    analyze_parser.add_argument('--workers', type=int, default=None,
                               help='Maximum number of workers')
    analyze_parser.set_defaults(func=analyze_files)
    
    # Statistics command
    stats_parser = subparsers.add_parser('stats', help='Show statistics')
    stats_parser.set_defaults(func=show_statistics)
    
    # Test audio analyzer command
    test_parser = subparsers.add_parser('test-audio', help='Test audio analyzer')
    test_parser.add_argument('--file', required=True,
                            help='Audio file to test')
    test_parser.add_argument('--force', action='store_true',
                            help='Force re-extraction')
    test_parser.set_defaults(func=test_audio_analyzer)
    
    # Monitor resources command
    monitor_parser = subparsers.add_parser('monitor', help='Monitor system resources')
    monitor_parser.add_argument('--duration', type=int, default=None,
                               help='Duration to monitor in seconds')
    monitor_parser.set_defaults(func=monitor_resources)
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Clean up failed analysis')
    cleanup_parser.add_argument('--max-retries', type=int, default=3,
                               help='Maximum retry count to keep')
    cleanup_parser.set_defaults(func=cleanup_failed)
    
    # Generate playlists command
    playlist_parser = subparsers.add_parser('playlist', help='Generate playlists')
    playlist_parser.add_argument('--method', default='all',
                                choices=['kmeans', 'similarity', 'random', 'time_based', 
                                       'tag_based', 'cache_based', 'feature_group', 'mixed', 'all'],
                                help='Playlist generation method')
    playlist_parser.add_argument('--num-playlists', type=int, default=8,
                                help='Number of playlists to generate')
    playlist_parser.add_argument('--playlist-size', type=int, default=20,
                                help='Size of each playlist')
    playlist_parser.add_argument('--save', action='store_true',
                                help='Save playlists to files')
    playlist_parser.add_argument('--output-dir', default='playlists',
                                help='Output directory for playlists')
    playlist_parser.set_defaults(func=generate_playlists)
    
    # List playlist methods command
    methods_parser = subparsers.add_parser('playlist-methods', help='List available playlist methods')
    methods_parser.set_defaults(func=list_playlist_methods)
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Execute command
    try:
        args.func(args)
    except Exception as e:
        logger.error(f"Command failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 