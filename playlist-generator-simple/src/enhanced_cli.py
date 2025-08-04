"""
Enhanced CLI interface for Playlist Generator Simple.
Integrates all variants from both simple and refactored versions.
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

from core.analysis_manager import AnalysisManager
from core.resource_manager import ResourceManager
from core.audio_analyzer import AudioAnalyzer
from core.database import DatabaseManager
from core.playlist_generator import PlaylistGenerator, PlaylistGenerationMethod
from core.unified_logging import get_logger, setup_logging, log_universal, log_session_start, log_session_end
from core.logging_config import get_logging_config, LoggingPatterns, StandardLogMessages
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
# This ensures that any logs from config_loader.get_config() respect the CLI verbosity
if verbose_level:
    setup_logging(log_level=verbose_level, console_logging=True, file_logging=False)

# Now load the config. Logs from config_loader will now respect the verbose_level.
config = config_loader.get_config()
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

# --- Stage 2: Full logging setup with unified system ---
verbose_count = 0
if verbose_level == 'DEBUG':
    verbose_count = 3
elif verbose_level == 'INFO':
    verbose_count = 1

# Create config overrides from legacy settings
config_overrides = {
    'console_enabled': console_logging,
    'file_enabled': file_logging,
    'log_file_prefix': log_file_prefix,
    'max_file_size_mb': log_file_size_mb,
    'backup_count': max_log_files,
}

# If verbose_level was set, override the config level
if verbose_level is not None:
    config_overrides['level'] = verbose_level

logging_config = get_logging_config(
    environment='cli',
    verbose_level=verbose_count,
    config_overrides=config_overrides
)

setup_logging(logging_config)

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
        self.config = config_loader.get_config()
        
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
            elif parsed_args.command == 'db':
                return self._handle_db(parsed_args)
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
        analyze_parser.add_argument('--music-path', default='/music', 
                                   help='Path to music directory')
        
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
        cleanup_parser.add_argument('--show-failed', action='store_true', 
                                   help='Show failed files before cleaning')
        cleanup_parser.add_argument('--move-to-failed-dir', action='store_true', 
                                   help='Move failed files to a directory')
        cleanup_parser.add_argument('--failed-dir', default='/music/failed',
                                 help='Directory to move permanently failed files')
        
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
        discover_parser.add_argument('--recursive', '-r', action='store_true', 
                                    help='Search recursively')
        discover_parser.add_argument('--extensions', '-e', 
                                    help='File extensions to include (comma-separated)')
        discover_parser.add_argument('--min-size', type=int, 
                                    help='Minimum file size in bytes')
        discover_parser.add_argument('--max-size', type=int, 
                                    help='Maximum file size in bytes')
        
        # Enrich command
        enrich_parser = subparsers.add_parser('enrich', help='Enrich metadata')
        enrich_parser.add_argument('--audio-ids', 
                                  help='Comma-separated list of audio file IDs')
        enrich_parser.add_argument('--path', default='/music', 
                                  help='Directory path to enrich all files')
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
        pipeline_parser.add_argument('--music-path', default='/music', 
                                    help='Directory path to process')
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
        
        # Config command
        config_parser = subparsers.add_parser('config', help='Show configuration information')
        config_parser.add_argument('--json', action='store_true', 
                                  help='Show JSON configuration')
        config_parser.add_argument('--validate', action='store_true', 
                                  help='Validate configuration')
        config_parser.add_argument('--reload', action='store_true', 
                                  help='Reload configuration')
        
        # Database commands
        db_parser = subparsers.add_parser('db', help='Database management')
        db_parser.add_argument('--init', action='store_true',
                              help='Initialize database schema')
        db_parser.add_argument('--migrate', action='store_true',
                              help='Migrate existing database')
        db_parser.add_argument('--backup', action='store_true',
                              help='Create database backup')
        db_parser.add_argument('--restore', action='store_true',
                              help='Restore database from backup')
        db_parser.add_argument('--integrity-check', action='store_true',
                              help='Check database integrity')
        db_parser.add_argument('--vacuum', action='store_true',
                              help='Vacuum database to reclaim space')
        db_parser.add_argument('--db-path', default='/app/cache/playlista.db',
                              help='Database file path')
        
        # Global options
        for subparser in [analyze_parser, stats_parser, test_parser, monitor_parser, 
                         cleanup_parser, playlist_parser, methods_parser, discover_parser, 
                         enrich_parser, export_parser, status_parser, pipeline_parser, config_parser, db_parser]:
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
            
            # Select files for analysis
            files = self.analysis_manager.select_files_for_analysis(
                music_path=args.music_path,
                force_reextract=args.force,
                include_failed=args.include_failed
            )
            
            if not files:
                log_universal('WARNING', 'Analysis', 'No files found for analysis')
                return 0
            
            log_universal('INFO', 'Analysis', f"Found {len(files)} files to analyze")
            
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
            # Get failed files statistics
            failed_files = self.db_manager.get_failed_analysis_files(max_retries=args.max_retries)
            log_universal('INFO', 'CLI', f"Found {len(failed_files)} failed files")
            
            if not failed_files:
                log_universal('INFO', 'CLI', 'No failed files to clean up')
                return 0
            
            # Show failed files if requested
            if args.show_failed:
                log_universal('INFO', 'CLI', 'Failed files:')
                for failed in failed_files:
                    log_universal('INFO', 'CLI', f"  {failed['filename']}: {failed['error_message']}")
            
            # Move failed files to failed directory if requested
            if args.move_to_failed_dir:
                failed_dir = args.failed_dir if hasattr(args, 'failed_dir') else None
                cleanup_stats = self.db_manager.cleanup_failed_files(failed_dir=failed_dir, max_retries=args.max_retries)
                
                if 'error' in cleanup_stats:
                    log_universal('ERROR', 'CLI', f"Failed to clean up files: {cleanup_stats['error']}")
                    return 1
                
                log_universal('INFO', 'CLI', f"Cleanup completed:")
                log_universal('INFO', 'CLI', f"  Moved: {cleanup_stats['moved_count']}")
                log_universal('INFO', 'CLI', f"  Skipped: {cleanup_stats['skipped_count']}")
                log_universal('INFO', 'CLI', f"  Errors: {cleanup_stats['error_count']}")
            else:
                # Just clean up old data without moving files
                cleaned_count = self.analysis_manager.cleanup_failed_analysis(max_retries=args.max_retries)
                log_universal('INFO', 'CLI', f"Cleaned up {cleaned_count} failed analysis entries")
            
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
        log_universal('INFO', 'FileDiscovery', "Discovering audio files in /music (fixed Docker path)")
        
        try:
            # Use file discovery directly
            from core.file_discovery import FileDiscovery
            file_discovery = FileDiscovery()
            
            # Process CLI arguments and override configuration
            config_overrides = {}
            
            # Process extensions argument
            if args.extensions:
                try:
                    extensions = [ext.strip() for ext in args.extensions.split(',')]
                    # Ensure extensions start with dot
                    valid_extensions = []
                    for ext in extensions:
                        if not ext.startswith('.'):
                            ext = '.' + ext
                        valid_extensions.append(ext.lower())
                    config_overrides['valid_extensions'] = valid_extensions
                    log_universal('INFO', 'FileDiscovery', f"Using custom extensions: {', '.join(valid_extensions)}")
                except Exception as e:
                    log_universal('ERROR', 'FileDiscovery', f"Invalid extensions format: {e}")
                    return 1
            
            # Process min size argument
            if args.min_size:
                if args.min_size < 0:
                    log_universal('ERROR', 'FileDiscovery', "Min size must be positive")
                    return 1
                config_overrides['min_file_size_bytes'] = args.min_size
                log_universal('INFO', 'FileDiscovery', f"Min file size: {args.min_size} bytes")
            
            # Process max size argument
            if args.max_size:
                if args.max_size < 0:
                    log_universal('ERROR', 'FileDiscovery', "Max size must be positive")
                    return 1
                config_overrides['max_file_size_bytes'] = args.max_size
                log_universal('INFO', 'FileDiscovery', f"Max file size: {args.max_size} bytes")
            
            # Process recursive argument
            if hasattr(args, 'recursive'):
                config_overrides['enable_recursive_scan'] = args.recursive
                log_universal('INFO', 'FileDiscovery', f"Recursive scan: {args.recursive}")
            
            # Apply configuration overrides
            if config_overrides:
                file_discovery.override_config(**config_overrides)
            
            # Discover files
            files = file_discovery.discover_files()
            
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
            # For now, just show a placeholder message
            # This would need to be implemented in the analysis manager
            log_universal('INFO', 'Enrichment', "Metadata enrichment not yet implemented")
            log_universal('INFO', 'Enrichment', "  This feature requires external API integration")
            log_universal('INFO', 'Enrichment', "  (MusicBrainz, Last.fm, etc.)")
            
            return 0
            
        except Exception as e:
            log_universal('ERROR', 'Enrichment', f"Metadata enrichment failed: {e}")
            return 1
    
    def _handle_export(self, args) -> int:
        """Handle export command."""
        log_universal('INFO', 'Export', f"Exporting playlists in {args.format} format")
        
        try:
            # For now, just show a placeholder message
            # This would need to be implemented in the playlist generator
            log_universal('INFO', 'Export', "Export functionality not yet implemented")
            log_universal('INFO', 'Export', "  This feature requires playlist export methods")
            log_universal('INFO', 'Export', f"  Format: {args.format}")
            log_universal('INFO', 'Export', f"  File: {args.playlist_file}")
            
            return 0
            
        except Exception as e:
            log_universal('ERROR', 'Export', f"Export failed: {e}")
            return 1
    
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
        """Handle pipeline command."""
        log_universal('INFO', 'CLI', 'Running full pipeline')
        
        try:
            # For now, just run analysis and playlist generation separately
            # This would need to be implemented as a full pipeline
            log_universal('INFO', 'Pipeline', "Full pipeline not yet implemented")
            log_universal('INFO', 'Pipeline', "  Running analysis and playlist generation separately")
            
            # Run analysis
            files = self.analysis_manager.select_files_for_analysis(
                music_path=args.music_path,
                force_reextract=args.force,
                include_failed=args.include_failed
            )
            
            if files:
                results = self.analysis_manager.analyze_files(
                    files=files,
                    force_reextract=args.force
                )
                log_universal('INFO', 'Analysis', f"Analysis completed: {results.get('success_count', 0)} files processed")
            
            # Generate playlists if requested
            if args.generate:
                playlists = self.playlist_generator.generate_playlists(
                    method='all',
                    num_playlists=5,
                    playlist_size=20
                )
                log_universal('INFO', 'Playlist', f"Generated {len(playlists)} playlists")
            
            # Export if requested
            if args.export:
                log_universal('INFO', 'Export', "Export functionality not yet implemented")
            
            # Final retry of failed files (if any)
            if not args.no_final_retry: # Only run final retry if not explicitly skipped
                log_universal('INFO', 'Pipeline', "Running final retry of failed files")
                retry_results = self.analysis_manager.final_retry_failed_files()
                log_universal('INFO', 'Pipeline', f"Final retry completed: {retry_results.get('successful', 0)} recovered, {retry_results.get('moved_to_failed_dir', 0)} moved to failed directory")
            
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
                self.config = config_loader.get_config()
                log_universal('INFO', 'Config', "Configuration reloaded")
            
            return 0
            
        except Exception as e:
            log_universal('ERROR', 'Config', f"Configuration error: {e}")
            return 1
    
    def _handle_db(self, args) -> int:
        """Handle database management commands."""
        log_universal('INFO', 'CLI', 'Database Management Commands')
        
        try:
            if args.init:
                log_universal('INFO', 'CLI', 'Initializing database schema...')
                self.db_manager.initialize_schema()
                log_universal('INFO', 'CLI', 'Database schema initialized.')
            elif args.migrate:
                log_universal('INFO', 'CLI', 'Migrating existing database...')
                self.db_manager.migrate_database()
                log_universal('INFO', 'CLI', 'Database migrated.')
            elif args.backup:
                log_universal('INFO', 'CLI', 'Creating database backup...')
                backup_path = self.db_manager.create_backup()
                log_universal('INFO', 'CLI', f'Database backed up to: {backup_path}')
            elif args.restore:
                log_universal('INFO', 'CLI', 'Restoring database from backup...')
                if args.db_path:
                    self.db_manager.restore_from_backup(args.db_path)
                    log_universal('INFO', 'CLI', f'Database restored from: {args.db_path}')
                else:
                    log_universal('ERROR', 'CLI', 'Please specify --db-path for restore.')
                    return 1
            elif args.integrity_check:
                log_universal('INFO', 'CLI', 'Checking database integrity...')
                is_ok = self.db_manager.check_integrity()
                if is_ok:
                    log_universal('INFO', 'CLI', 'Database integrity check passed.')
                else:
                    log_universal('ERROR', 'CLI', 'Database integrity check failed.')
                    return 1
            elif args.vacuum:
                log_universal('INFO', 'CLI', 'Vacuuming database...')
                self.db_manager.vacuum_database()
                log_universal('INFO', 'CLI', 'Database vacuumed.')
            else:
                log_universal('INFO', 'CLI', 'No specific database command provided. Use --help for details.')
            
            return 0
            
        except Exception as e:
            log_universal('ERROR', 'CLI', f"Database management failed: {e}")
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
        # Implementation for detailed statistics
        pass
    
    def _show_failed_files(self):
        """Show failed files."""
        log_universal('INFO', 'CLI', "Failed Files:")
        # Implementation for failed files
        pass
    
    def _show_memory_usage(self):
        """Show memory usage."""
        log_universal('INFO', 'CLI', "Memory Usage:")
        # Implementation for memory usage
        pass
    
    def _show_detailed_status(self):
        """Show detailed status."""
        log_universal('INFO', 'CLI', "Detailed Status:")
        # Implementation for detailed status
        pass
    
    def _validate_configuration(self) -> bool:
        """Validate configuration."""
        try:
            # Basic validation
            required_keys = ['MEMORY_LIMIT_GB', 'CPU_THRESHOLD_PERCENT']
            for key in required_keys:
                if key not in self.config:
                    log_universal('ERROR', 'Config', f"Missing required configuration key: {key}")
                    return False
            
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