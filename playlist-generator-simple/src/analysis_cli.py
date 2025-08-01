"""
Simple CLI interface for analysis components in Playlist Generator Simple.
Provides commands to test and use the analysis functionality.
"""

# Suppress external library logging BEFORE any imports
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU usage
os.environ['ESSENTIA_LOG_LEVEL'] = 'error'  # Suppress Essentia info/warnings
os.environ['MUSICEXTRACTOR_LOG_LEVEL'] = 'error'  # Suppress MusicExtractorSVM logs
os.environ['TENSORFLOW_LOG_LEVEL'] = '2'  # Suppress TensorFlow info/warnings
os.environ['LIBROSA_LOG_LEVEL'] = 'error'  # Suppress Librosa logs

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
from core.logging_setup import get_logger, setup_logging, log_universal
from core.config_loader import config_loader

# Check for verbose arguments before setting up logging
def check_verbose_args():
    """Check for verbose arguments in sys.argv and return appropriate log level."""
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-v', '--verbose', action='count', default=0)
    try:
        args, _ = parser.parse_known_args()
        if args.verbose > 0:
            verbosity_map = {
                1: 'INFO',
                2: 'DEBUG', 
                3: 'TRACE'
            }
            return verbosity_map.get(args.verbose, 'TRACE')
    except:
        pass
    return None  # Return None if no verbose flags, so we use config default

# Get initial log level considering verbose flags
verbose_level = check_verbose_args()

# --- Stage 1: Set initial log level based on CLI verbose flags immediately ---
# This ensures that any logs from config_loader.load_config() respect the CLI verbosity
if verbose_level:
    setup_logging(log_level=verbose_level, console_logging=True, file_logging=False, environment_monitoring=False)

# Now load the config. Logs from config_loader will now respect the verbose_level.
config = config_loader.load_config()
logging_config = config_loader.get_logging_config()

# Use config values or defaults, potentially overridden by CLI verbose_level
log_level_from_config = logging_config.get('LOG_LEVEL', 'INFO') # Get config's LOG_LEVEL
log_file_prefix = logging_config.get('LOG_FILE_PREFIX', 'playlista_analysis')
console_logging = logging_config.get('LOG_CONSOLE_ENABLED', True)
file_logging = logging_config.get('LOG_FILE_ENABLED', True)
colored_output = logging_config.get('LOG_COLORED_OUTPUT', True)
file_colored_output = logging_config.get('LOG_FILE_COLORED_OUTPUT', None)  # None means use colored_output
max_log_files = logging_config.get('LOG_MAX_FILES', 10)
log_file_size_mb = logging_config.get('LOG_FILE_SIZE_MB', 50)
log_file_format = logging_config.get('LOG_FILE_FORMAT', 'text')
log_file_encoding = logging_config.get('LOG_FILE_ENCODING', 'utf-8')

# --- Stage 2: Full logging setup with all config parameters ---
setup_logging(
    log_level=verbose_level if verbose_level is not None else log_level_from_config, # Prioritize CLI verbose_level
    log_file_prefix=log_file_prefix,
    console_logging=console_logging,
    file_logging=file_logging,
    colored_output=colored_output,
    file_colored_output=file_colored_output,
    max_log_files=max_log_files,
    log_file_size_mb=log_file_size_mb,
    log_file_format=log_file_format,
    log_file_encoding=log_file_encoding,
    environment_monitoring=verbose_level is None  # Disable env monitoring if verbose flags used
)

logger = get_logger('playlista.analysis_cli')


def analyze_files(args):
    """Analyze audio files using the analysis manager."""
    log_universal('INFO', 'CLI', "Starting file analysis")
    
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
            log_universal('WARNING', 'CLI', "No files found for analysis")
            return
        
        log_universal('INFO', 'CLI', f"Found {len(files)} files to analyze")
        
        # Analyze files
        results = analysis_manager.analyze_files(
            files=files,
            force_reextract=args.force,
            max_workers=args.workers
        )
        
        # Display results
        log_universal('INFO', 'CLI', "Analysis Results:")
        log_universal('INFO', 'CLI', f"  Success: {results['success_count']}")
        log_universal('INFO', 'CLI', f"  Failed: {results['failed_count']}")
        log_universal('INFO', 'CLI', f"  Total time: {results['total_time']:.2f}s")
        
        if 'big_files_processed' in results:
            log_universal('INFO', 'CLI', f"  Large files: {results['big_files_processed']}")
        if 'small_files_processed' in results:
            log_universal('INFO', 'CLI', f"  Small files: {results['small_files_processed']}")
        
    except Exception as e:
        log_universal('ERROR', 'CLI', f"Analysis failed: {e}")


def show_statistics(args):
    """Show analysis and resource statistics."""
    log_universal('INFO', 'CLI', "Showing statistics")
    
    try:
        # Create managers
        analysis_manager = AnalysisManager()
        resource_manager = ResourceManager()
        db_manager = DatabaseManager()
        
        # Analysis statistics
        analysis_stats = analysis_manager.get_analysis_statistics()
        log_universal('INFO', 'CLI', "Analysis Statistics:")
        for key, value in analysis_stats.items():
            if key != 'database_stats':
                log_universal('INFO', 'CLI', f"  {key}: {value}")
        
        # Resource statistics
        resource_stats = resource_manager.get_resource_statistics(minutes=60)
        log_universal('INFO', 'CLI', "Resource Statistics (last hour):")
        for category, stats in resource_stats.items():
            if category != 'period_minutes' and category != 'data_points':
                log_universal('INFO', 'CLI', f"  {category}:")
                for metric, value in stats.items():
                    log_universal('INFO', 'CLI', f"    {metric}: {value:.2f}")
        
        # Database statistics
        db_stats = db_manager.get_database_statistics()
        log_universal('INFO', 'CLI', "Database Statistics:")
        for key, value in db_stats.items():
            log_universal('INFO', 'CLI', f"  {key}: {value}")
        
    except Exception as e:
        log_universal('ERROR', 'CLI', f"Error getting statistics: {e}")


def test_audio_analyzer(args):
    """Test the audio analyzer with a specific file."""
    log_universal('INFO', 'CLI', "Testing audio analyzer")
    
    if not args.file:
        log_universal('ERROR', 'CLI', "Please specify a file with --file")
        return
    
    if not os.path.exists(args.file):
        log_universal('ERROR', 'CLI', f"File not found: {args.file}")
        return
    
    try:
        # Create audio analyzer
        audio_analyzer = AudioAnalyzer()
        
        # Extract features
        result = audio_analyzer.extract_features(args.file, force_reextract=args.force)
        
        if result:
            log_universal('INFO', 'CLI', "Feature extraction successful")
            log_universal('INFO', 'CLI', f"Features extracted: {len(result.get('features', {}))}")
            
            # Show some key features
            features = result.get('features', {})
            if 'bpm' in features:
                log_universal('INFO', 'CLI', f"BPM: {features['bpm']}")
            if 'loudness' in features:
                log_universal('INFO', 'CLI', f"Loudness: {features['loudness']}")
            if 'key' in features:
                log_universal('INFO', 'CLI', f"Key: {features['key']}")
            
            # Show metadata
            metadata = result.get('metadata', {})
            if metadata:
                log_universal('INFO', 'CLI', "Metadata:")
                for key, value in metadata.items():
                    log_universal('INFO', 'CLI', f"  {key}: {value}")
        else:
            log_universal('ERROR', 'CLI', "Feature extraction failed")
        
    except Exception as e:
        log_universal('ERROR', 'CLI', f"Audio analyzer test failed: {e}")


def monitor_resources(args):
    """Monitor system resources in real-time."""
    log_universal('INFO', 'CLI', "Starting resource monitoring")
    
    try:
        # Create resource manager
        resource_manager = ResourceManager()
        
        # Start monitoring
        resource_manager.start_monitoring()
        
        log_universal('INFO', 'CLI', "Resource monitoring started (press Ctrl+C to stop)")
        log_universal('INFO', 'CLI', "Monitoring interval: {} seconds".format(
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
                        log_universal('INFO', 'CLI', f"Memory: {memory['used_gb']:.2f}GB used ({memory['percent']:.1f}%)")
                    if 'cpu_percent' in resources:
                        log_universal('INFO', 'CLI', f"CPU: {resources['cpu_percent']:.1f}%")
        except KeyboardInterrupt:
            log_universal('INFO', 'CLI', "Stopping resource monitoring")
        finally:
            resource_manager.stop_monitoring()
        
    except Exception as e:
        log_universal('ERROR', 'CLI', f"Resource monitoring failed: {e}")


def cleanup_failed(args):
    """Clean up failed analysis entries."""
    log_universal('INFO', 'CLI', "Cleaning up failed analysis entries")
    
    try:
        # Create analysis manager
        analysis_manager = AnalysisManager()
        
        # Clean up failed analysis
        cleaned_count = analysis_manager.cleanup_failed_analysis(
            max_retries=args.max_retries
        )
        
        log_universal('INFO', 'CLI', f"Cleaned up {cleaned_count} failed analysis entries")
        
    except Exception as e:
        log_universal('ERROR', 'CLI', f"Cleanup failed: {e}")


def generate_playlists(args):
    """Generate playlists using analyzed tracks."""
    log_universal('INFO', 'CLI', "Starting playlist generation")
    
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
            log_universal('WARNING', 'CLI', "No playlists generated")
            return
        
        # Display results
        log_universal('INFO', 'CLI', "Playlist Generation Results:")
        log_universal('INFO', 'CLI', f"  Total playlists: {len(playlists)}")
        
        for name, playlist in playlists.items():
            log_universal('INFO', 'CLI', f"  {name}: {playlist.size} tracks")
        
        # Save playlists if requested
        if args.save:
            output_dir = args.output_dir or 'playlists'
            success = playlist_generator.save_playlists(playlists, output_dir)
            if success:
                log_universal('INFO', 'CLI', f"Playlists saved to {output_dir}")
            else:
                log_universal('ERROR', 'CLI', f"Failed to save playlists")
        
        # Show statistics
        stats = playlist_generator.get_playlist_statistics(playlists)
        log_universal('INFO', 'CLI', "Playlist Statistics:")
        log_universal('INFO', 'CLI', f"  Total tracks: {stats.get('total_tracks', 0)}")
        log_universal('INFO', 'CLI', f"  Average size: {stats.get('average_playlist_size', 0):.1f}")
        log_universal('INFO', 'CLI', f"  Size range: {stats.get('min_playlist_size', 0)} - {stats.get('max_playlist_size', 0)}")
        
    except Exception as e:
        log_universal('ERROR', 'CLI', f"Playlist generation failed: {e}")


def list_playlist_methods(args):
    """List available playlist generation methods."""
    log_universal('INFO', 'CLI', "Available playlist generation methods:")
    
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
        log_universal('INFO', 'CLI', f"  {method}: {description}")


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
    
    # Global verbose options
    parser.add_argument('-v', '--verbose', action='count', default=0,
                       help='Increase verbosity (-v: INFO, -vv: DEBUG, -vvv: TRACE)')
    
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
        log_universal('ERROR', 'CLI', f"Command failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 
