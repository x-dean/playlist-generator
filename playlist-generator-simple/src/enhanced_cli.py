"""
Main CLI interface for Playlist Generator Simple.
Provides all functionality: analysis, playlist generation, pipeline, export, stats, etc.
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
import json
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add the current directory to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

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
                3: 'DEBUG'  # Max out at DEBUG level
            }
            return verbosity_map.get(args.verbose, 'DEBUG')
    except:
        pass
    return None  # Return None if no verbose flags, so we use config default

# Get initial log level considering verbose flags
verbose_level = check_verbose_args()

# --- Stage 1: Set initial log level based on CLI verbose flags immediately ---
# This ensures that any logs from config_loader.load_config() respect the CLI verbosity
if verbose_level:
    setup_logging(log_level=verbose_level, console_logging=True, file_logging=False)

# Now load the config. Logs from config_loader will now respect the verbose_level.
config = config_loader.load_config()
logging_config = config_loader.get_logging_config()

# Use config values or defaults, potentially overridden by CLI verbose_level
log_level_from_config = logging_config.get('LOG_LEVEL', 'INFO') # Get config's LOG_LEVEL
log_file_prefix = logging_config.get('LOG_FILE_PREFIX', 'playlista')
console_logging = logging_config.get('LOG_CONSOLE_ENABLED', True)  # Default to console logging
file_logging = logging_config.get('LOG_FILE_ENABLED', True)
max_log_files = logging_config.get('LOG_MAX_FILES', 10)
log_file_size_mb = logging_config.get('LOG_FILE_SIZE_MB', 50)

# Force console logging when verbose flag is used
if verbose_level is not None:
    console_logging = True

# --- Stage 2: Full logging setup with all config parameters ---
setup_logging(
    log_level=verbose_level if verbose_level is not None else log_level_from_config, # Prioritize CLI verbose_level
    log_file_prefix=log_file_prefix,
    console_logging=console_logging,
    file_logging=file_logging
)

logger = get_logger('playlista.enhanced_cli')


class EnhancedCLI:
    """
    Enhanced CLI interface with all features from both simple and refactored versions.
    
    Features:
    - All analysis commands (analyze, stats, test-audio, monitor, cleanup)
    - All playlist generation methods (kmeans, similarity, random, time_based, etc.)
    - Discovery and pipeline commands
    - Metadata enrichment
    - Export functionality
    - System status and configuration
    - Memory-aware processing
    - Fast mode processing
    - Parallel/sequential processing
    """
    
    def __init__(self):
        """Initialize the enhanced CLI interface."""
        self.config = config_loader.load_config()
        
        # Use local database path for development
        db_path = os.path.join(os.path.dirname(__file__), '..', 'cache', 'playlista.db')
        self.db_manager = DatabaseManager(db_path=db_path)
        
        # Pass the database manager to other components
        self.analysis_manager = AnalysisManager(db_manager=self.db_manager)
        self.resource_manager = ResourceManager()
        self.playlist_generator = PlaylistGenerator()
        
        log_universal('INFO', 'CLI', 'Initializing Enhanced CLI')
    
    def run(self, args: Optional[List[str]] = None) -> int:
        """Run the enhanced CLI interface."""
        if args is None:
            args = sys.argv[1:]
        
        parser = self._create_argument_parser()
        parsed_args = parser.parse_args(args)
        
        # Handle help or missing command
        if not parsed_args.command:
            self._show_help()
            return 0
        
        # Add session header
        from core.logging_setup import log_session_header
        log_session_header(f"playlista {parsed_args.command}")
        
        # Route to appropriate handler
        try:
            if parsed_args.command == 'analyze':
                return self._handle_analyze(parsed_args)
            elif parsed_args.command == 'stats':
                return self._handle_stats(parsed_args)
            elif parsed_args.command == 'retry-failed':
                return self._handle_retry_failed(parsed_args)
            elif parsed_args.command == 'test-audio':
                return self._handle_test_audio(parsed_args)
            elif parsed_args.command == 'monitor':
                return self._handle_monitor(parsed_args)
            elif parsed_args.command == 'cleanup':
                return self._handle_cleanup(parsed_args)
            elif parsed_args.command == 'playlist':
                return self._handle_playlist(parsed_args)
            elif parsed_args.command == 'playlist-methods':
                return self._handle_playlist_methods(parsed_args)
            elif parsed_args.command == 'discover':
                return self._handle_discover(parsed_args)
            elif parsed_args.command == 'enrich':
                return self._handle_enrich(parsed_args)
            elif parsed_args.command == 'export':
                return self._handle_export(parsed_args)
            elif parsed_args.command == 'status':
                return self._handle_status(parsed_args)
            elif parsed_args.command == 'pipeline':
                return self._handle_pipeline(parsed_args)
            elif parsed_args.command == 'config':
                return self._handle_config(parsed_args)
            else:
                log_universal('ERROR', 'CLI', f"Unknown command: {parsed_args.command}")
                return 1
        except (ValueError, TypeError, OSError) as e:
            log_universal('ERROR', 'CLI', f"Command failed due to invalid input or system error: {e}")
            return 1
        except KeyboardInterrupt:
            log_universal('INFO', 'CLI', 'Operation cancelled by user')
            return 130
        except Exception as e:
            log_universal('ERROR', 'CLI', f"Unexpected error occurred: {e}")
            return 1
    
    def _create_argument_parser(self) -> argparse.ArgumentParser:
        """Create the argument parser with all options."""
        parser = argparse.ArgumentParser(
            description="""
Enhanced Playlist Generator Simple CLI

Features:
- Memory-aware processing with Docker optimization
- Fast mode for 3-5x faster processing
- Parallel and sequential processing modes
- Advanced audio features (MusiCNN, emotional features)
- Multiple playlist generation methods
- Database management and validation
- Metadata enrichment from external APIs
- Export functionality in multiple formats
            """,
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Analyze audio files
  playlista analyze --music-path /path/to/music --fast-mode --parallel
  
  # Memory-aware processing
  playlista analyze --music-path /path/to/music --memory-aware --memory-limit 2GB
  
  # Generate playlists
  playlista playlist --method kmeans --num-playlists 5 --playlist-size 20
  
  # Full pipeline
  playlista pipeline --music-path /path/to/music --force --failed
  
  # Show statistics
  playlista stats --detailed
  
  # Export playlists
  playlista export --playlist-file playlists.json --format m3u
            """
        )
        
        # Global verbose options
        parser.add_argument('-v', '--verbose', action='count', default=0,
                           help='Increase verbosity (-v: INFO, -vv: DEBUG, -vvv: DEBUG)')
        
        subparsers = parser.add_subparsers(dest='command', help='Available commands')
        
        # Analyze command
        analyze_parser = subparsers.add_parser('analyze', help='Analyze audio files')
        # Note: Music path is fixed to /music (Docker internal path)
        
        # Processing options
        analyze_parser.add_argument('--parallel', '-p', action='store_true', 
                                   help='Use parallel processing')
        analyze_parser.add_argument('--sequential', '-s', action='store_true', 
                                   help='Use sequential processing')
        analyze_parser.add_argument('--workers', '-w', type=int, 
                                   help='Number of worker processes')
        
        # Memory management
        analyze_parser.add_argument('--memory-limit', 
                                   help='Memory limit per worker (e.g., "2GB", "512MB")')
        analyze_parser.add_argument('--memory-aware', action='store_true', 
                                   help='Enable memory-aware processing')
        analyze_parser.add_argument('--rss-limit-gb', type=float, default=6.0, 
                                   help='Max total Python RSS (GB)')
        analyze_parser.add_argument('--low-memory', action='store_true', 
                                   help='Enable low memory mode')
        
        # Processing modes
        analyze_parser.add_argument('--fast-mode', action='store_true', 
                                   help='Enable fast mode (3-5x faster)')
        analyze_parser.add_argument('--force', '-f', action='store_true', 
                                   help='Force re-analysis')
        analyze_parser.add_argument('--no-cache', action='store_true', 
                                   help='Bypass cache')
        analyze_parser.add_argument('--failed', action='store_true', 
                                   help='Re-analyze only failed files')
        analyze_parser.add_argument('--include-failed', action='store_true', 
                                   help='Include previously failed files')
        analyze_parser.add_argument('--retry-failed', action='store_true', 
                                   help='Retry files that failed in current analysis run')
        
        # File handling
        analyze_parser.add_argument('--large-file-threshold', type=int, default=50, 
                                   help='Large file threshold (MB)')
        analyze_parser.add_argument('--batch-size', type=int, 
                                   help='Batch size for processing')
        analyze_parser.add_argument('--timeout', type=int, default=300, 
                                   help='Processing timeout in seconds')
        
        # Statistics command
        stats_parser = subparsers.add_parser('stats', help='Show statistics')
        stats_parser.add_argument('--detailed', '-d', action='store_true', 
                                 help='Show detailed statistics')
        stats_parser.add_argument('--failed-files', action='store_true', 
                                 help='Show failed files')
        stats_parser.add_argument('--memory-usage', action='store_true', 
                                 help='Show memory usage')
        
        # Final retry command
        retry_parser = subparsers.add_parser('retry-failed', help='Final retry of failed files')
        retry_parser.add_argument('--failed-dir', default='/music/failed',
                                 help='Directory to move permanently failed files')
        retry_parser.add_argument('--force', action='store_true',
                                 help='Force retry even if files were already moved')
        
        # Test audio analyzer command
        test_parser = subparsers.add_parser('test-audio', help='Test audio analyzer')
        test_parser.add_argument('--file', required=True,
                                help='Audio file to test')
        test_parser.add_argument('--force', action='store_true', 
                                help='Force re-extraction')
        
        # Monitor resources command
        monitor_parser = subparsers.add_parser('monitor', help='Monitor system resources')
        monitor_parser.add_argument('--duration', type=int, default=None, 
                                   help='Duration to monitor in seconds')
        
        # Cleanup command
        cleanup_parser = subparsers.add_parser('cleanup', help='Clean up failed analysis')
        cleanup_parser.add_argument('--max-retries', type=int, default=3, 
                                   help='Maximum retry count to keep')
        
        # Generate playlists command
        playlist_parser = subparsers.add_parser('playlist', help='Generate playlists')
        playlist_parser.add_argument('--method', '-m', default='all',
                                    choices=['kmeans', 'similarity', 'random', 'time_based', 
                                           'tag_based', 'cache_based', 'feature_group', 'mixed', 
                                           'all', 'ensemble', 'hierarchical', 'recommendation', 'mood_based'],
                                    help='Playlist generation method')
        playlist_parser.add_argument('--num-playlists', type=int, default=8,
                                    help='Number of playlists to generate')
        playlist_parser.add_argument('--playlist-size', '-s', type=int, default=20,
                                    help='Size of each playlist')
        playlist_parser.add_argument('--save', action='store_true',
                                    help='Save playlists to files')
        playlist_parser.add_argument('--output-dir', default='playlists',
                                    help='Output directory for playlists')
        playlist_parser.add_argument('--min-tracks-per-genre', type=int, default=10,
                                    help='Minimum tracks per genre (tags method)')
        
        # List playlist methods command
        methods_parser = subparsers.add_parser('playlist-methods', help='List available playlist methods')
        
        # Discover command
        discover_parser = subparsers.add_parser('discover', help='Discover audio files')
        # Note: Music path is fixed to /music (Docker internal path)
        discover_parser.add_argument('--recursive', '-r', action='store_true', 
                                    help='Search recursively')
        discover_parser.add_argument('--extensions', '-e', 
                                    help='File extensions to include (comma-separated)')
        discover_parser.add_argument('--exclude-dirs', 
                                    help='Directories to exclude (comma-separated)')
        discover_parser.add_argument('--min-size', type=int, 
                                    help='Minimum file size in bytes')
        discover_parser.add_argument('--max-size', type=int, 
                                    help='Maximum file size in bytes')
        
        # Enrich command
        enrich_parser = subparsers.add_parser('enrich', help='Enrich metadata')
        enrich_parser.add_argument('--audio-ids', 
                                  help='Comma-separated list of audio file IDs')
        # Note: Music path is fixed to /music (Docker internal path)
        enrich_parser.add_argument('--force', '-f', action='store_true', 
                                  help='Force re-enrichment')
        enrich_parser.add_argument('--musicbrainz', action='store_true', 
                                  help='Use MusicBrainz API')
        enrich_parser.add_argument('--lastfm', action='store_true', 
                                  help='Use Last.fm API')
        
        # Export command
        export_parser = subparsers.add_parser('export', help='Export playlists')
        export_parser.add_argument('--playlist-file', default='playlists/playlist.json', 
                                  help='Playlist file to export')
        export_parser.add_argument('--format', '-f', 
                                  choices=['m3u', 'pls', 'xspf', 'json', 'all'],
                                  default='m3u', help='Export format')
        export_parser.add_argument('--output-dir', 
                                  help='Output directory')
        
        # Status command
        status_parser = subparsers.add_parser('status', help='Show database status')
        status_parser.add_argument('--detailed', '-d', action='store_true', 
                                  help='Show detailed statistics')
        status_parser.add_argument('--failed-files', action='store_true', 
                                  help='Show failed files')
        status_parser.add_argument('--memory-usage', action='store_true', 
                                  help='Show memory usage')
        
        # Pipeline command
        pipeline_parser = subparsers.add_parser('pipeline', help='Run full pipeline')
        # Note: Music path is fixed to /music (Docker internal path)
        pipeline_parser.add_argument('--force', '-f', action='store_true', 
                                    help='Force re-analysis')
        pipeline_parser.add_argument('--failed', action='store_true', 
                                    help='Re-analyze failed files')
        pipeline_parser.add_argument('--include-failed', action='store_true', 
                                    help='Include previously failed files')
        pipeline_parser.add_argument('--generate', action='store_true', 
                                    help='Generate playlists after analysis')
        pipeline_parser.add_argument('--export', action='store_true', 
                                    help='Export playlists after generation')
        pipeline_parser.add_argument('--no-final-retry', action='store_true', 
                                    help='Skip final retry of failed files (enabled by default)')
        pipeline_parser.add_argument('--playlist-method', 
                                    choices=['kmeans', 'similarity', 'random', 'time_based', 
                                           'tag_based', 'cache_based', 'feature_group', 'mixed', 
                                           'all', 'ensemble', 'hierarchical', 'recommendation', 'mood_based'],
                                    default='all', help='Playlist generation method')
        pipeline_parser.add_argument('--num-playlists', type=int, default=5, 
                                    help='Number of playlists to generate')
        pipeline_parser.add_argument('--playlist-size', type=int, default=20, 
                                    help='Size of each playlist')
        pipeline_parser.add_argument('--export-format', 
                                    choices=['m3u', 'pls', 'xspf', 'json', 'all'],
                                    default='all', help='Export format')
        
        # Config command
        config_parser = subparsers.add_parser('config', help='Show configuration information')
        config_parser.add_argument('--json', action='store_true', 
                                  help='Show JSON configuration')
        config_parser.add_argument('--validate', action='store_true', 
                                  help='Validate configuration')
        config_parser.add_argument('--reload', action='store_true', 
                                  help='Reload configuration')
        
        # Global options
        for subparser in [analyze_parser, stats_parser, test_parser, monitor_parser, 
                         cleanup_parser, playlist_parser, methods_parser, discover_parser, 
                         enrich_parser, export_parser, status_parser, pipeline_parser, config_parser]:
            subparser.add_argument('--log-level', 
                                  choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                                  default='INFO', help='Set logging level')
            subparser.add_argument('--verbose', '-v', action='store_true', 
                                  help='Verbose output')
            subparser.add_argument('--quiet', '-q', action='store_true', 
                                  help='Quiet output')
        
        return parser
    
    def _show_help(self):
        """Show the main help screen."""
        help_text = """
 Enhanced Playlist Generator Simple CLI

 Available Commands:
  analyze          Analyze audio files for features
  stats            Show analysis and resource statistics
  test-audio       Test audio analyzer with a single file
  monitor          Monitor system resources
  cleanup          Clean up failed analysis
  playlist         Generate playlists using various methods
  playlist-methods List available playlist generation methods
  discover         Discover audio files in a directory
  enrich           Enrich metadata from external APIs
  export           Export playlists in different formats
  status           Show database and system status
  pipeline         Run full analysis and generation pipeline
  config           Show configuration information

 Key Features:
   Memory-aware processing with Docker optimization
   Fast mode processing (3-5x faster)
   Parallel and sequential processing
   Advanced audio features (MusiCNN, emotional features)
   Multiple playlist generation methods
   Database management and validation
   Metadata enrichment from external APIs
   Export functionality in multiple formats

 Quick Start:
  1. playlista analyze --music-path /path/to/music --fast-mode
  2. playlista playlist --method kmeans --num-playlists 5
  3. playlista export --playlist-file playlists.json --format m3u

 Memory Management:
  --memory-aware --memory-limit 2GB --rss-limit-gb 6.0

 Processing Modes:
  --fast-mode (3-5x faster)
  --parallel --workers 4
  --sequential (for debugging)

 For detailed help:
  playlista <command> --help
        """
        
        log_universal('INFO', 'CLI', help_text)
    
    def _handle_analyze(self, args) -> int:
        """Handle analyze command."""
        log_universal('INFO', 'Analysis', 'Starting file analysis')
        
        try:
            # Configure analysis based on arguments
            analysis_config = self._build_analysis_config(args)
            
            # Use Docker internal path for music library (mapped via docker-compose)
            music_path = '/music'  # Always use Docker internal path
            
            # Select files for analysis
            files = self.analysis_manager.select_files_for_analysis(
                music_path=music_path,  # Use Docker internal path
                force_reextract=args.force,
                include_failed=args.include_failed
            )
            
            if not files:
                log_universal('WARNING', 'Analysis', f'No files found for analysis in {music_path}')
                return 0
            
            log_universal('INFO', 'Analysis', f"Found {len(files)} files to analyze in {music_path}")
            
            # Analyze files
            results = self.analysis_manager.analyze_files(
                files=files,
                force_reextract=args.force,
                max_workers=args.workers
            )
            
            # Display results
            log_universal('INFO', 'Analysis', "Analysis Results:")
            log_universal('INFO', 'Analysis', f"  Success: {results.get('success_count', 0)}")
            log_universal('INFO', 'Analysis', f"  Failed: {results.get('failed_count', 0)}")
            log_universal('INFO', 'Analysis', f"  Total time: {results.get('total_time', 0):.2f}s")
            
            if 'big_files_processed' in results:
                log_universal('INFO', 'Analysis', f"  Large files: {results['big_files_processed']}")
            if 'small_files_processed' in results:
                log_universal('INFO', 'Analysis', f"  Small files: {results['small_files_processed']}")
            
            # Retry failed files from current analysis run if requested
            if args.retry_failed and results.get('failed_count', 0) > 0:
                log_universal('INFO', 'Analysis', "Retrying files that failed in current analysis run")
                retry_results = self.analysis_manager.retry_current_failed_files()
                log_universal('INFO', 'Analysis', f"Retry completed: {retry_results.get('successful', 0)} recovered, {retry_results.get('still_failed', 0)} still failed")
            
            return 0
            
        except FileNotFoundError as e:
            log_universal('ERROR', 'Analysis', f"Music directory not found: {e}")
            return 1
        except PermissionError as e:
            log_universal('ERROR', 'Analysis', f"Permission denied accessing music directory: {e}")
            return 1
        except OSError as e:
            log_universal('ERROR', 'Analysis', f"System error during analysis: {e}")
            return 1
        except Exception as e:
            log_universal('ERROR', 'Analysis', f"Unexpected error during analysis: {e}")
            return 1
    
    def _handle_stats(self, args) -> int:
        """Handle stats command."""
        log_universal('INFO', 'CLI', 'Showing statistics')
        
        try:
            # Analysis statistics
            analysis_stats = self.analysis_manager.get_analysis_statistics()
            log_universal('INFO', 'Analysis', "Analysis Statistics:")
            for key, value in analysis_stats.items():
                if key != 'database_stats':
                    log_universal('INFO', 'Analysis', f"  {key}: {value}")
            
            # Resource statistics
            resource_stats = self.resource_manager.get_resource_statistics(minutes=60)
            log_universal('INFO', 'Resource', "Resource Statistics (last hour):")
            for category, stats in resource_stats.items():
                if category != 'period_minutes' and category != 'data_points':
                    log_universal('INFO', 'Resource', f"  {category}:")
                    for metric, value in stats.items():
                        log_universal('INFO', 'Resource', f"    {metric}: {value:.2f}")
            
            # Database statistics
            db_stats = self.db_manager.get_database_statistics()
            log_universal('INFO', 'Database', "Database Statistics:")
            for key, value in db_stats.items():
                log_universal('INFO', 'Database', f"  {key}: {value}")
            
            # Detailed statistics if requested
            if args.detailed:
                self._show_detailed_statistics()
            
            # Failed files if requested
            if args.failed_files:
                self._show_failed_files()
            
            # Memory usage if requested
            if args.memory_usage:
                self._show_memory_usage()
            
            return 0
            
        except sqlite3.Error as e:
            log_universal('ERROR', 'Database', f"Database error getting statistics: {e}")
            return 1
        except OSError as e:
            log_universal('ERROR', 'System', f"System error getting statistics: {e}")
            return 1
        except Exception as e:
            log_universal('ERROR', 'CLI', f"Unexpected error getting statistics: {e}")
            return 1
    
    def _handle_retry_failed(self, args) -> int:
        """Handle retry-failed command."""
        log_universal('INFO', 'CLI', 'Starting final retry of failed files')
        
        try:
            # Update configuration with failed directory
            if args.failed_dir:
                self.analysis_manager.config['FAILED_FILES_DIR'] = args.failed_dir
            
            # Run final retry
            results = self.analysis_manager.final_retry_failed_files()
            
            # Display results
            log_universal('INFO', 'CLI', "Final Retry Results:")
            log_universal('INFO', 'CLI', f"  Files retried: {results.get('retried', 0)}")
            log_universal('INFO', 'CLI', f"  Successfully analyzed: {results.get('successful', 0)}")
            log_universal('INFO', 'CLI', f"  Moved to failed directory: {results.get('moved_to_failed_dir', 0)}")
            log_universal('INFO', 'CLI', f"  Total time: {results.get('total_time', 0):.2f}s")
            
            if results.get('successful', 0) > 0:
                log_universal('INFO', 'CLI', f"✅ Successfully recovered {results['successful']} files")
            
            if results.get('moved_to_failed_dir', 0) > 0:
                log_universal('INFO', 'CLI', f"📁 Moved {results['moved_to_failed_dir']} files to failed directory")
            
            return 0
            
        except Exception as e:
            log_universal('ERROR', 'CLI', f"Error during final retry: {e}")
            return 1

    def _handle_test_audio(self, args) -> int:
        """Handle test-audio command."""
        log_universal('INFO', 'Audio', f"Testing audio analyzer with file: {args.file}")
        
        try:
            # Test the audio analyzer
            analyzer = AudioAnalyzer()
            result = analyzer.analyze_file(args.file, force_reextract=args.force)
            
            if result:
                log_universal('INFO', 'Audio', "Audio analysis test successful")
                log_universal('INFO', 'Audio', f"  Features extracted: {len(result)}")
                for key, value in result.items():
                    if isinstance(value, (int, float)):
                        log_universal('INFO', 'Audio', f"  {key}: {value}")
                    elif isinstance(value, list) and len(value) <= 5:
                        log_universal('INFO', 'Audio', f"  {key}: {value}")
                    else:
                        log_universal('INFO', 'Audio', f"  {key}: <complex data>")
            else:
                log_universal('ERROR', 'Audio', "Audio analysis test failed")
                return 1
            
            return 0
            
        except (FileNotFoundError, PermissionError) as e:
            log_universal('ERROR', 'Audio', f"File access error during audio test: {e}")
            return 1
        except Exception as e:
            log_universal('ERROR', 'Audio', f"Audio analysis test failed: {e}")
            return 1
    
    def _handle_monitor(self, args) -> int:
        """Handle monitor command."""
        log_universal('INFO', 'Resource', 'Starting resource monitoring')
        
        try:
            # Start monitoring
            self.resource_manager.start_monitoring()
            
            # Monitor for specified duration or indefinitely
            if args.duration:
                import time
                log_universal('INFO', 'Resource', f"Monitoring for {args.duration} seconds")
                time.sleep(args.duration)
                self.resource_manager.stop_monitoring()
            else:
                log_universal('INFO', 'Resource', "Monitoring indefinitely (press Ctrl+C to stop)")
                try:
                    while True:
                        import time
                        time.sleep(5)
                except KeyboardInterrupt:
                    log_universal('INFO', 'Resource', "Monitoring stopped by user")
                    self.resource_manager.stop_monitoring()
            
            return 0
            
        except Exception as e:
            log_universal('ERROR', 'Resource', f"Monitoring failed: {e}")
            return 1
    
    def _handle_cleanup(self, args) -> int:
        """Handle cleanup command."""
        log_universal('INFO', 'CLI', 'Cleaning up failed analysis')
        
        try:
            # Clean up failed files using the analysis manager
            cleaned_count = self.analysis_manager.cleanup_failed_analysis(max_retries=args.max_retries)
            log_universal('INFO', 'CLI', f"Cleaned up {cleaned_count} failed files")
            
            return 0
            
        except Exception as e:
            log_universal('ERROR', 'CLI', f"Cleanup failed: {e}")
            return 1
    
    def _handle_playlist(self, args) -> int:
        """Handle playlist command."""
        log_universal('INFO', 'Playlist', f"Generating playlists using method: {args.method}")
        
        try:
            # Generate playlists
            playlists = self.playlist_generator.generate_playlists(
                method=args.method,
                num_playlists=args.num_playlists,
                playlist_size=args.playlist_size,
                min_tracks_per_genre=args.min_tracks_per_genre
            )
            
            log_universal('INFO', 'Playlist', f"Generated {len(playlists)} playlists")
            
            # Show playlist statistics
            stats = self.playlist_generator.get_playlist_statistics(playlists)
            log_universal('INFO', 'Playlist', "Playlist Statistics:")
            for key, value in stats.items():
                log_universal('INFO', 'Playlist', f"  {key}: {value}")
            
            # Save playlists if requested
            if args.save:
                success = self.playlist_generator.save_playlists(playlists, args.output_dir)
                if success:
                    log_universal('INFO', 'Playlist', f"Playlists saved to {args.output_dir}")
                else:
                    log_universal('ERROR', 'Playlist', "Failed to save playlists")
                    return 1
            
            return 0
            
        except Exception as e:
            log_universal('ERROR', 'Playlist', f"Playlist generation failed: {e}")
            return 1
    
    def _handle_playlist_methods(self, args) -> int:
        """Handle playlist-methods command."""
        log_universal('INFO', 'Playlist', "Available playlist generation methods:")
        
        methods = [
            ("kmeans", "K-means clustering based on audio features"),
            ("similarity", "Similarity-based selection using cosine similarity"),
            ("random", "Random selection for variety"),
            ("time_based", "Time-based scheduling for different times of day"),
            ("tag_based", "Tag-based selection using metadata tags"),
            ("cache_based", "Cache-based selection using previously generated playlists"),
            ("feature_group", "Feature group selection based on audio characteristics"),
            ("mixed", "Mixed approach combining multiple methods"),
            ("all", "All methods combined for comprehensive coverage"),
            ("ensemble", "Ensemble methods combining multiple algorithms"),
            ("hierarchical", "Hierarchical clustering for nested groupings"),
            ("recommendation", "Recommendation-based using collaborative filtering"),
            ("mood_based", "Mood-based generation using emotional features")
        ]
        
        for method, description in methods:
            log_universal('INFO', 'Playlist', f"  {method}: {description}")
        
        return 0
    
    def _handle_discover(self, args) -> int:
        """Handle discover command."""
        # Use Docker internal path for music library
        music_path = '/music'  # Always use Docker internal path
        log_universal('INFO', 'FileDiscovery', f"Discovering audio files in: {music_path}")
        
        try:
            # Use file discovery directly
            from core.file_discovery import FileDiscovery
            file_discovery = FileDiscovery()
            
            # Discover files from Docker internal path
            files = file_discovery.discover_files(music_path)
            
            log_universal('INFO', 'FileDiscovery', f"Discovered {len(files)} audio files")
            
            # Save discovered files to database
            if files:
                stats = file_discovery.save_discovered_files_to_db(files)
                log_universal('INFO', 'FileDiscovery', f"Database save results:")
                log_universal('INFO', 'FileDiscovery', f"  New files: {stats['new']}")
                log_universal('INFO', 'FileDiscovery', f"  Updated files: {stats['updated']}")
                log_universal('INFO', 'FileDiscovery', f"  Unchanged files: {stats['unchanged']}")
                log_universal('INFO', 'FileDiscovery', f"  Errors: {stats['errors']}")
            
            # Show file details
            for file_path in files[:10]:  # Show first 10 files
                log_universal('INFO', 'FileDiscovery', f"  {file_path}")
            
            if len(files) > 10:
                log_universal('INFO', 'FileDiscovery', f"  ... and {len(files) - 10} more files")
            
            return 0
            
        except Exception as e:
            log_universal('ERROR', 'FileDiscovery', f"File discovery failed: {e}")
            return 1
    
    def _handle_enrich(self, args) -> int:
        """Handle enrich command."""
        log_universal('INFO', 'Enrichment', 'Enriching metadata')
        
        try:
            # Use Docker internal path for music library
            music_path = '/music'  # Always use Docker internal path
            
            # Enrich specific audio files if provided
            if args.audio_ids:
                audio_ids = [id.strip() for id in args.audio_ids.split(',')]
                log_universal('INFO', 'Enrichment', f"Enriching {len(audio_ids)} specific audio files")
                # Implementation for specific file enrichment
                # This would need to be implemented in the enrichment module
                log_universal('INFO', 'Enrichment', "Specific file enrichment not yet implemented")
            else:
                # Enrich all files in the music directory
                log_universal('INFO', 'Enrichment', f"Enriching all files in {music_path}")
                # Implementation for bulk enrichment
                # This would need to be implemented in the enrichment module
                log_universal('INFO', 'Enrichment', "Bulk enrichment not yet implemented")
            
            return 0
            
        except Exception as e:
            log_universal('ERROR', 'Enrichment', f"Enrichment failed: {e}")
            return 1
    
    def _handle_export(self, args) -> int:
        """Handle export command."""
        log_universal('INFO', 'Export', f"Exporting playlists in {args.format} format")
        
        try:
            # Get playlists to export
            if args.playlist_file and os.path.exists(args.playlist_file):
                # Export from specific playlist file
                with open(args.playlist_file, 'r') as f:
                    playlists_data = json.load(f)
                playlists = playlists_data.get('playlists', [])
                log_universal('INFO', 'Export', f"Loading playlists from {args.playlist_file}")
            else:
                # Export all playlists from database
                playlists = self.db_manager.get_all_playlists()
                log_universal('INFO', 'Export', f"Loading {len(playlists)} playlists from database")
            
            if not playlists:
                log_universal('WARNING', 'Export', "No playlists found to export")
                return 1
            
            # Export playlists
            export_success = self._export_playlists(playlists, args.format, args.output_dir)
            
            if export_success:
                log_universal('INFO', 'Export', f"Successfully exported {len(playlists)} playlists in {args.format} format")
                return 0
            else:
                log_universal('ERROR', 'Export', "Export failed")
                return 1
            
        except Exception as e:
            log_universal('ERROR', 'Export', f"Export failed: {e}")
            return 1
    
    def _export_playlists(self, playlists: List[Dict], format_type: str, output_dir: str = None) -> bool:
        """Export playlists in the specified format."""
        try:
            if output_dir is None:
                output_dir = 'playlists'
            
            os.makedirs(output_dir, exist_ok=True)
            
            if format_type == 'all':
                formats = ['m3u', 'pls', 'xspf', 'json']
            else:
                formats = [format_type]
            
            exported_files = []
            
            for playlist in playlists:
                playlist_name = playlist.get('name', 'playlist')
                tracks = playlist.get('tracks', [])
                
                for fmt in formats:
                    filename = f"{playlist_name}.{fmt}"
                    filepath = os.path.join(output_dir, filename)
                    
                    if fmt == 'm3u':
                        success = self._export_m3u(playlist, filepath)
                    elif fmt == 'pls':
                        success = self._export_pls(playlist, filepath)
                    elif fmt == 'xspf':
                        success = self._export_xspf(playlist, filepath)
                    elif fmt == 'json':
                        success = self._export_json(playlist, filepath)
                    else:
                        log_universal('WARNING', 'Export', f"Unknown format: {fmt}")
                        continue
                    
                    if success:
                        exported_files.append(filepath)
                        log_universal('INFO', 'Export', f"Exported {playlist_name} to {filepath}")
            
            log_universal('INFO', 'Export', f"Exported {len(exported_files)} files to {output_dir}")
            return len(exported_files) > 0
            
        except Exception as e:
            log_universal('ERROR', 'Export', f"Export failed: {e}")
            return False
    
    def _export_m3u(self, playlist: Dict, filepath: str) -> bool:
        """Export playlist in M3U format."""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("#EXTM3U\n")
                f.write(f"# {playlist.get('name', 'Playlist')}\n")
                
                for track in playlist.get('tracks', []):
                    f.write(f"{track}\n")
            
            return True
        except Exception as e:
            log_universal('ERROR', 'Export', f"M3U export failed: {e}")
            return False
    
    def _export_pls(self, playlist: Dict, filepath: str) -> bool:
        """Export playlist in PLS format."""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("[playlist]\n")
                f.write(f"NumberOfEntries={len(playlist.get('tracks', []))}\n")
                
                for i, track in enumerate(playlist.get('tracks', []), 1):
                    f.write(f"File{i}={track}\n")
                    f.write(f"Title{i}={os.path.basename(track)}\n")
                    f.write(f"Length{i}=-1\n")
                
                f.write("Version=2\n")
            
            return True
        except Exception as e:
            log_universal('ERROR', 'Export', f"PLS export failed: {e}")
            return False
    
    def _export_xspf(self, playlist: Dict, filepath: str) -> bool:
        """Export playlist in XSPF format."""
        try:
            xspf_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<playlist xmlns="http://xspf.org/ns/0/" version="1">
    <title>{playlist.get('name', 'Playlist')}</title>
    <trackList>
"""
            
            for track in playlist.get('tracks', []):
                track_name = os.path.basename(track)
                xspf_content += f"""        <track>
            <title>{track_name}</title>
            <location>{track}</location>
        </track>
"""
            
            xspf_content += """    </trackList>
</playlist>"""
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(xspf_content)
            
            return True
        except Exception as e:
            log_universal('ERROR', 'Export', f"XSPF export failed: {e}")
            return False
    
    def _export_json(self, playlist: Dict, filepath: str) -> bool:
        """Export playlist in JSON format."""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(playlist, f, indent=2, ensure_ascii=False)
            
            return True
        except Exception as e:
            log_universal('ERROR', 'Export', f"JSON export failed: {e}")
            return False
    
    def _handle_status(self, args) -> int:
        """Handle status command."""
        log_universal('INFO', 'CLI', 'Showing system status')
        
        try:
            # Database status (using available methods)
            db_stats = self.db_manager.get_database_statistics()
            log_universal('INFO', 'Database', "Database Status:")
            for key, value in db_stats.items():
                log_universal('INFO', 'Database', f"  {key}: {value}")
            
            # System status
            system_status = self.resource_manager.get_current_resources()
            log_universal('INFO', 'System', "System Status:")
            log_universal('INFO', 'System', f"  Memory: {system_status['memory']['percent']:.1f}%")
            log_universal('INFO', 'System', f"  CPU: {system_status['cpu_percent']:.1f}%")
            log_universal('INFO', 'System', f"  Disk: {system_status['disk']['percent']:.1f}%")
            
            # Detailed status if requested
            if args.detailed:
                self._show_detailed_status()
            
            # Failed files if requested
            if args.failed_files:
                self._show_failed_files()
            
            # Memory usage if requested
            if args.memory_usage:
                self._show_memory_usage()
            
            return 0
            
        except Exception as e:
            log_universal('ERROR', 'CLI', f"Error getting status: {e}")
            return 1
    
    def _handle_pipeline(self, args) -> int:
        """Handle pipeline command - full analysis to playlist generation workflow."""
        log_universal('INFO', 'CLI', 'Running full pipeline')
        
        try:
            pipeline_results = {
                'analysis': {'success': False, 'files_processed': 0, 'failed': 0},
                'playlists': {'success': False, 'generated': 0},
                'export': {'success': False, 'files_created': 0},
                'retry': {'success': False, 'recovered': 0, 'moved': 0}
            }
            
            # Use Docker internal paths (mapped via docker-compose)
            music_path = '/music'  # Docker internal path for music library
            playlists_dir = '/app/playlists'  # Docker internal path for playlists
            cache_dir = '/app/cache'  # Docker internal path for cache
            
            # Step 1: Analysis
            log_universal('INFO', 'Pipeline', f"Step 1: Starting analysis from {music_path}")
            files = self.analysis_manager.select_files_for_analysis(
                music_path=music_path,  # Always use Docker internal path
                force_reextract=args.force,
                include_failed=args.include_failed
            )
            
            if files:
                analysis_results = self.analysis_manager.analyze_files(
                    files=files,
                    force_reextract=args.force
                )
                pipeline_results['analysis'] = {
                    'success': True,
                    'files_processed': analysis_results.get('success_count', 0),
                    'failed': analysis_results.get('failed_count', 0)
                }
                log_universal('INFO', 'Pipeline', f"Analysis completed: {pipeline_results['analysis']['files_processed']} files processed, {pipeline_results['analysis']['failed']} failed")
            else:
                log_universal('WARNING', 'Pipeline', f"No files found for analysis in {music_path}")
            
            # Step 2: Playlist Generation (if requested)
            if args.generate:
                log_universal('INFO', 'Pipeline', "Step 2: Generating playlists")
                try:
                    # Use pipeline arguments or defaults
                    method = getattr(args, 'playlist_method', 'all')
                    num_playlists = getattr(args, 'num_playlists', 5)
                    playlist_size = getattr(args, 'playlist_size', 20)
                    
                    playlists = self.playlist_generator.generate_playlists(
                        method=method,
                        num_playlists=num_playlists,
                        playlist_size=playlist_size
                    )
                    pipeline_results['playlists'] = {
                        'success': True,
                        'generated': len(playlists)
                    }
                    log_universal('INFO', 'Pipeline', f"Generated {len(playlists)} playlists using {method} method")
                    
                    # Save playlists to Docker internal path
                    if playlists:
                        save_success = self.playlist_generator.save_playlists(playlists, playlists_dir)
                        if save_success:
                            log_universal('INFO', 'Pipeline', f"Playlists saved to {playlists_dir}")
                        else:
                            log_universal('WARNING', 'Pipeline', f"Failed to save playlists to {playlists_dir}")
                            
                except Exception as e:
                    log_universal('ERROR', 'Pipeline', f"Playlist generation failed: {e}")
                    pipeline_results['playlists']['success'] = False
            
            # Step 3: Export (if requested and playlists exist)
            if args.export and pipeline_results['playlists']['success']:
                log_universal('INFO', 'Pipeline', "Step 3: Exporting playlists")
                try:
                    # Get saved playlists for export
                    saved_playlists = self.db_manager.get_all_playlists()
                    if saved_playlists:
                        # Use specified format or default to 'all'
                        export_format = getattr(args, 'export_format', 'all')
                        export_success = self._export_playlists(saved_playlists, export_format, playlists_dir)
                        pipeline_results['export'] = {
                            'success': export_success,
                            'files_created': len(saved_playlists) if export_success else 0
                        }
                        if export_success:
                            log_universal('INFO', 'Pipeline', f"Exported {len(saved_playlists)} playlists in {export_format} format to {playlists_dir}")
                        else:
                            log_universal('WARNING', 'Pipeline', "Export failed")
                    else:
                        log_universal('WARNING', 'Pipeline', "No playlists to export")
                except Exception as e:
                    log_universal('ERROR', 'Pipeline', f"Export failed: {e}")
                    pipeline_results['export']['success'] = False
            
            # Step 4: Final retry of failed files (if any)
            if not args.no_final_retry:
                log_universal('INFO', 'Pipeline', "Step 4: Final retry of failed files")
                try:
                    retry_results = self.analysis_manager.final_retry_failed_files()
                    pipeline_results['retry'] = {
                        'success': True,
                        'recovered': retry_results.get('successful', 0),
                        'moved': retry_results.get('moved_to_failed_dir', 0)
                    }
                    log_universal('INFO', 'Pipeline', f"Final retry completed: {pipeline_results['retry']['recovered']} recovered, {pipeline_results['retry']['moved']} moved to failed directory")
                except Exception as e:
                    log_universal('ERROR', 'Pipeline', f"Final retry failed: {e}")
                    pipeline_results['retry']['success'] = False
            
            # Pipeline summary
            log_universal('INFO', 'Pipeline', "Pipeline Summary:")
            log_universal('INFO', 'Pipeline', f"  Analysis: {'✓' if pipeline_results['analysis']['success'] else '✗'} ({pipeline_results['analysis']['files_processed']} files)")
            log_universal('INFO', 'Pipeline', f"  Playlists: {'✓' if pipeline_results['playlists']['success'] else '✗'} ({pipeline_results['playlists']['generated']} generated)")
            log_universal('INFO', 'Pipeline', f"  Export: {'✓' if pipeline_results['export']['success'] else '✗'} ({pipeline_results['export']['files_created']} files)")
            log_universal('INFO', 'Pipeline', f"  Retry: {'✓' if pipeline_results['retry']['success'] else '✗'} ({pipeline_results['retry']['recovered']} recovered)")
            
            return 0
            
        except Exception as e:
            log_universal('ERROR', 'Pipeline', f"Pipeline failed: {e}")
            return 1
    
    def _handle_config(self, args) -> int:
        """Handle config command."""
        log_universal('INFO', 'Config', 'Showing configuration information')
        
        try:
            # Show configuration
            if args.json:
                config_json = json.dumps(self.config, indent=2)
                log_universal('INFO', 'Config', config_json)
            else:
                log_universal('INFO', 'Config', "Configuration:")
                for key, value in self.config.items():
                    log_universal('INFO', 'Config', f"  {key}: {value}")
            
            # Validate configuration if requested
            if args.validate:
                is_valid = self._validate_configuration()
                if is_valid:
                    log_universal('INFO', 'Config', "Configuration is valid")
                else:
                    log_universal('ERROR', 'Config', "Configuration has issues")
                    return 1
            
            # Reload configuration if requested
            if args.reload:
                self.config = config_loader.load_config()
                log_universal('INFO', 'Config', "Configuration reloaded")
            
            return 0
            
        except Exception as e:
            log_universal('ERROR', 'Config', f"Configuration error: {e}")
            return 1
    
    def _build_analysis_config(self, args) -> Dict[str, Any]:
        """Build analysis configuration from arguments."""
        config = {}
        
        # Processing options
        if args.parallel:
            config['parallel'] = True
        if args.sequential:
            config['sequential'] = True
        if args.workers:
            config['workers'] = args.workers
        
        # Memory management
        if args.memory_aware:
            config['memory_aware'] = True
        if args.memory_limit:
            config['memory_limit'] = args.memory_limit
        if args.low_memory:
            config['low_memory'] = True
        if args.rss_limit_gb:
            config['rss_limit_gb'] = args.rss_limit_gb
        
        # Processing modes
        if args.fast_mode:
            config['fast_mode'] = True
        if args.no_cache:
            config['no_cache'] = True
        if args.failed:
            config['failed_only'] = True
        
        # File handling
        if args.large_file_threshold:
            config['large_file_threshold'] = args.large_file_threshold
        if args.batch_size:
            config['batch_size'] = args.batch_size
        if args.timeout:
            config['timeout'] = args.timeout
        
        return config
    
    def _show_detailed_statistics(self):
        """Show detailed statistics."""
        log_universal('INFO', 'CLI', "Detailed Statistics:")
        
        try:
            # Analysis statistics
            analysis_stats = self.analysis_manager.get_analysis_statistics()
            log_universal('INFO', 'Analysis', "Analysis Statistics:")
            for key, value in analysis_stats.items():
                if key != 'database_stats':
                    log_universal('INFO', 'Analysis', f"  {key}: {value}")
            
            # Database statistics
            db_stats = self.db_manager.get_database_statistics()
            log_universal('INFO', 'Database', "Database Statistics:")
            for key, value in db_stats.items():
                log_universal('INFO', 'Database', f"  {key}: {value}")
            
            # Playlist statistics
            playlists = self.db_manager.get_all_playlists()
            log_universal('INFO', 'Playlist', "Playlist Statistics:")
            log_universal('INFO', 'Playlist', f"  Total playlists: {len(playlists)}")
            
            if playlists:
                total_tracks = sum(len(p.get('tracks', [])) for p in playlists)
                avg_playlist_size = total_tracks / len(playlists)
                log_universal('INFO', 'Playlist', f"  Total tracks: {total_tracks}")
                log_universal('INFO', 'Playlist', f"  Average playlist size: {avg_playlist_size:.1f}")
                
                # Playlist size distribution
                sizes = [len(p.get('tracks', [])) for p in playlists]
                log_universal('INFO', 'Playlist', f"  Smallest playlist: {min(sizes)} tracks")
                log_universal('INFO', 'Playlist', f"  Largest playlist: {max(sizes)} tracks")
            
            # Resource statistics
            resource_stats = self.resource_manager.get_resource_statistics(minutes=60)
            log_universal('INFO', 'Resource', "Resource Statistics (last hour):")
            for category, stats in resource_stats.items():
                if category != 'period_minutes' and category != 'data_points':
                    log_universal('INFO', 'Resource', f"  {category}:")
                    for metric, value in stats.items():
                        log_universal('INFO', 'Resource', f"    {metric}: {value:.2f}")
                        
        except Exception as e:
            log_universal('ERROR', 'CLI', f"Error showing detailed statistics: {e}")
    
    def _show_failed_files(self):
        """Show failed files."""
        log_universal('INFO', 'CLI', "Failed Files:")
        
        try:
            failed_files = self.db_manager.get_failed_analysis_files()
            
            if not failed_files:
                log_universal('INFO', 'Failed', "No failed files found")
                return
            
            log_universal('INFO', 'Failed', f"Found {len(failed_files)} failed files:")
            
            for failed_file in failed_files:
                file_path = failed_file.get('file_path', 'Unknown')
                error_message = failed_file.get('error_message', 'Unknown error')
                retry_count = failed_file.get('retry_count', 0)
                last_attempt = failed_file.get('last_attempt', 'Unknown')
                
                log_universal('INFO', 'Failed', f"  File: {file_path}")
                log_universal('INFO', 'Failed', f"    Error: {error_message}")
                log_universal('INFO', 'Failed', f"    Retry count: {retry_count}")
                log_universal('INFO', 'Failed', f"    Last attempt: {last_attempt}")
                log_universal('INFO', 'Failed', "")  # Empty line for readability
                
        except Exception as e:
            log_universal('ERROR', 'CLI', f"Error showing failed files: {e}")
    
    def _show_memory_usage(self):
        """Show memory usage."""
        log_universal('INFO', 'CLI', "Memory Usage:")
        
        try:
            # Current memory usage
            current_resources = self.resource_manager.get_current_resources()
            memory_info = current_resources.get('memory', {})
            
            log_universal('INFO', 'Memory', f"  Total memory: {memory_info.get('total', 0):.1f} GB")
            log_universal('INFO', 'Memory', f"  Available memory: {memory_info.get('available', 0):.1f} GB")
            log_universal('INFO', 'Memory', f"  Used memory: {memory_info.get('used', 0):.1f} GB")
            log_universal('INFO', 'Memory', f"  Memory usage: {memory_info.get('percent', 0):.1f}%")
            
            # Memory history
            resource_stats = self.resource_manager.get_resource_statistics(minutes=60)
            memory_stats = resource_stats.get('memory', {})
            
            if memory_stats:
                log_universal('INFO', 'Memory', "Memory Statistics (last hour):")
                log_universal('INFO', 'Memory', f"  Average usage: {memory_stats.get('average_percent', 0):.1f}%")
                log_universal('INFO', 'Memory', f"  Peak usage: {memory_stats.get('max_percent', 0):.1f}%")
                log_universal('INFO', 'Memory', f"  Minimum usage: {memory_stats.get('min_percent', 0):.1f}%")
            
            # Process memory usage
            import psutil
            process = psutil.Process()
            process_memory = process.memory_info()
            log_universal('INFO', 'Memory', f"  Process memory: {process_memory.rss / 1024 / 1024:.1f} MB")
            
        except Exception as e:
            log_universal('ERROR', 'CLI', f"Error showing memory usage: {e}")
    
    def _show_detailed_status(self):
        """Show detailed status."""
        log_universal('INFO', 'CLI', "Detailed Status:")
        
        try:
            # System status
            current_resources = self.resource_manager.get_current_resources()
            
            log_universal('INFO', 'System', "System Status:")
            log_universal('INFO', 'System', f"  CPU usage: {current_resources.get('cpu_percent', 0):.1f}%")
            log_universal('INFO', 'System', f"  Memory usage: {current_resources.get('memory', {}).get('percent', 0):.1f}%")
            log_universal('INFO', 'System', f"  Disk usage: {current_resources.get('disk', {}).get('percent', 0):.1f}%")
            
            # Database status
            db_stats = self.db_manager.get_database_statistics()
            log_universal('INFO', 'Database', "Database Status:")
            log_universal('INFO', 'Database', f"  Database size: {db_stats.get('database_size_mb', 0):.1f} MB")
            log_universal('INFO', 'Database', f"  Total tracks: {db_stats.get('total_tracks', 0)}")
            log_universal('INFO', 'Database', f"  Analyzed tracks: {db_stats.get('analyzed_tracks', 0)}")
            log_universal('INFO', 'Database', f"  Failed tracks: {db_stats.get('failed_tracks', 0)}")
            log_universal('INFO', 'Database', f"  Total playlists: {db_stats.get('total_playlists', 0)}")
            
            # Analysis status
            analysis_stats = self.analysis_manager.get_analysis_statistics()
            log_universal('INFO', 'Analysis', "Analysis Status:")
            log_universal('INFO', 'Analysis', f"  Files processed: {analysis_stats.get('files_processed', 0)}")
            log_universal('INFO', 'Analysis', f"  Success rate: {analysis_stats.get('success_rate', 0):.1f}%")
            log_universal('INFO', 'Analysis', f"  Average processing time: {analysis_stats.get('avg_processing_time', 0):.2f}s")
            
            # Configuration status
            log_universal('INFO', 'Config', "Configuration Status:")
            log_universal('INFO', 'Config', f"  Music path: {self.config.get('MUSIC_PATH', '/music')}")
            log_universal('INFO', 'Config', f"  Cache enabled: {self.config.get('ANALYSIS_CACHE_ENABLED', True)}")
            log_universal('INFO', 'Config', f"  Fast mode: {self.config.get('FAST_MODE_ENABLED', False)}")
            log_universal('INFO', 'Config', f"  Memory aware: {self.config.get('MEMORY_AWARE_PROCESSING', True)}")
            
        except Exception as e:
            log_universal('ERROR', 'CLI', f"Error showing detailed status: {e}")
    
    def _validate_configuration(self) -> bool:
        """Validate configuration."""
        try:
            # Check for essential configuration sections
            essential_sections = [
                'MUSIC_PATH',
                'ANALYSIS_CACHE_ENABLED', 
                'LOG_LEVEL',
                'DB_PATH'
            ]
            
            missing_keys = []
            for key in essential_sections:
                if key not in self.config:
                    missing_keys.append(key)
            
            if missing_keys:
                log_universal('ERROR', 'Config', f"Missing required configuration keys: {', '.join(missing_keys)}")
                return False
            
            # Validate music path
            music_path = self.config.get('MUSIC_PATH', '/music')
            if not os.path.exists(music_path):
                log_universal('WARNING', 'Config', f"Music path does not exist: {music_path}")
            
            # Validate database path
            db_path = self.config.get('DB_PATH', '/app/cache/playlista.db')
            db_dir = os.path.dirname(db_path)
            if not os.path.exists(db_dir):
                log_universal('WARNING', 'Config', f"Database directory does not exist: {db_dir}")
            
            # Validate log level
            log_level = self.config.get('LOG_LEVEL', 'INFO')
            valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
            if log_level not in valid_log_levels:
                log_universal('ERROR', 'Config', f"Invalid log level: {log_level}")
                return False
            
            # Validate analysis settings
            analysis_cache = self.config.get('ANALYSIS_CACHE_ENABLED', True)
            if not isinstance(analysis_cache, bool):
                log_universal('ERROR', 'Config', "ANALYSIS_CACHE_ENABLED must be boolean")
                return False
            
            # Validate memory settings
            memory_limit = self.config.get('MEMORY_LIMIT_GB', 6.0)
            if not isinstance(memory_limit, (int, float)) or memory_limit <= 0:
                log_universal('ERROR', 'Config', "MEMORY_LIMIT_GB must be positive number")
                return False
            
            log_universal('INFO', 'Config', "Configuration validation passed")
            return True
            
        except Exception as e:
            log_universal('ERROR', 'Config', f"Configuration validation failed: {e}")
            return False


def main():
    """Main CLI entry point."""
    cli = EnhancedCLI()
    return cli.run()


if __name__ == "__main__":
    exit(main()) 