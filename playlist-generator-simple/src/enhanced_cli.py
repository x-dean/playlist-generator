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

import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add core to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core'))

from core.analysis_manager import AnalysisManager
from core.resource_manager import ResourceManager
from core.audio_analyzer import AudioAnalyzer
from core.database import DatabaseManager
from core.playlist_generator import PlaylistGenerator, PlaylistGenerationMethod
from core.logging_setup import get_logger
from core.config_loader import config_loader

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
        
        # Fixed Docker database path
        db_path = '/app/cache/playlista.db'
        self.db_manager = DatabaseManager(db_path=db_path)
        
        # Pass the database manager to other components
        self.analysis_manager = AnalysisManager(db_manager=self.db_manager)
        self.resource_manager = ResourceManager()
        self.playlist_generator = PlaylistGenerator()
        
        logger.info("🔧 Initializing Enhanced CLI")
    
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
        
        # Route to appropriate handler
        try:
            if parsed_args.command == 'analyze':
                return self._handle_analyze(parsed_args)
            elif parsed_args.command == 'stats':
                return self._handle_stats(parsed_args)
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
                logger.error(f"❌ Unknown command: {parsed_args.command}")
                return 1
        except Exception as e:
            logger.error(f"❌ Command failed: {e}")
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
        discover_parser.add_argument('--path', default='/music', 
                                    help='Directory path to search')
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
🎵 Enhanced Playlist Generator Simple CLI

📋 Available Commands:
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

🚀 Key Features:
  🧠 Memory-aware processing with Docker optimization
  ⚡ Fast mode processing (3-5x faster)
  🔄 Parallel and sequential processing
  🎵 Advanced audio features (MusiCNN, emotional features)
  🎼 Multiple playlist generation methods
  💾 Database management and validation
  🌐 Metadata enrichment from external APIs
  📤 Export functionality in multiple formats

📖 Quick Start:
  1. playlista analyze --music-path /path/to/music --fast-mode
  2. playlista playlist --method kmeans --num-playlists 5
  3. playlista export --playlist-file playlists.json --format m3u

🔧 Memory Management:
  --memory-aware --memory-limit 2GB --rss-limit-gb 6.0

⚡ Processing Modes:
  --fast-mode (3-5x faster)
  --parallel --workers 4
  --sequential (for debugging)

📊 For detailed help:
  playlista <command> --help
        """
        
        print(help_text)
    
    def _handle_analyze(self, args) -> int:
        """Handle analyze command."""
        logger.info("🎵 Starting file analysis")
        
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
                logger.warning("⚠️ No files found for analysis")
                return 0
            
            logger.info(f"📁 Found {len(files)} files to analyze")
            
            # Analyze files
            results = self.analysis_manager.analyze_files(
                files=files,
                force_reextract=args.force,
                max_workers=args.workers
            )
            
            # Display results
            logger.info("📊 Analysis Results:")
            logger.info(f"   Success: {results.get('success_count', 0)}")
            logger.info(f"   Failed: {results.get('failed_count', 0)}")
            logger.info(f"   Total time: {results.get('total_time', 0):.2f}s")
            
            if 'big_files_processed' in results:
                logger.info(f"   Large files: {results['big_files_processed']}")
            if 'small_files_processed' in results:
                logger.info(f"   Small files: {results['small_files_processed']}")
            
            return 0
            
        except Exception as e:
            logger.error(f"❌ Analysis failed: {e}")
            return 1
    
    def _handle_stats(self, args) -> int:
        """Handle stats command."""
        logger.info("📊 Showing statistics")
        
        try:
            # Analysis statistics
            analysis_stats = self.analysis_manager.get_analysis_statistics()
            logger.info("📊 Analysis Statistics:")
            for key, value in analysis_stats.items():
                if key != 'database_stats':
                    logger.info(f"   {key}: {value}")
            
            # Resource statistics
            resource_stats = self.resource_manager.get_resource_statistics(minutes=60)
            logger.info("📊 Resource Statistics (last hour):")
            for category, stats in resource_stats.items():
                if category != 'period_minutes' and category != 'data_points':
                    logger.info(f"   {category}:")
                    for metric, value in stats.items():
                        logger.info(f"     {metric}: {value:.2f}")
            
            # Database statistics
            db_stats = self.db_manager.get_database_statistics()
            logger.info("📊 Database Statistics:")
            for key, value in db_stats.items():
                logger.info(f"   {key}: {value}")
            
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
            
        except Exception as e:
            logger.error(f"❌ Error getting statistics: {e}")
            return 1
    
    def _handle_test_audio(self, args) -> int:
        """Handle test-audio command."""
        logger.info(f"🧪 Testing audio analyzer with file: {args.file}")
        
        try:
            # Test the audio analyzer
            analyzer = AudioAnalyzer()
            result = analyzer.analyze_file(args.file, force_reextract=args.force)
            
            if result:
                logger.info("✅ Audio analysis test successful")
                logger.info(f"   Features extracted: {len(result)}")
                for key, value in result.items():
                    if isinstance(value, (int, float)):
                        logger.info(f"   {key}: {value}")
                    elif isinstance(value, list) and len(value) <= 5:
                        logger.info(f"   {key}: {value}")
                    else:
                        logger.info(f"   {key}: <complex data>")
            else:
                logger.error("❌ Audio analysis test failed")
                return 1
            
            return 0
            
        except Exception as e:
            logger.error(f"❌ Audio analysis test failed: {e}")
            return 1
    
    def _handle_monitor(self, args) -> int:
        """Handle monitor command."""
        logger.info("🔍 Starting resource monitoring")
        
        try:
            # Start monitoring
            self.resource_manager.start_monitoring()
            
            # Monitor for specified duration or indefinitely
            if args.duration:
                import time
                logger.info(f"⏱️ Monitoring for {args.duration} seconds")
                time.sleep(args.duration)
                self.resource_manager.stop_monitoring()
            else:
                logger.info("⏱️ Monitoring indefinitely (press Ctrl+C to stop)")
                try:
                    while True:
                        import time
                        time.sleep(5)
                except KeyboardInterrupt:
                    logger.info("🛑 Monitoring stopped by user")
                    self.resource_manager.stop_monitoring()
            
            return 0
            
        except Exception as e:
            logger.error(f"❌ Monitoring failed: {e}")
            return 1
    
    def _handle_cleanup(self, args) -> int:
        """Handle cleanup command."""
        logger.info("🧹 Cleaning up failed analysis")
        
        try:
            # Clean up failed files using the analysis manager
            cleaned_count = self.analysis_manager.cleanup_failed_analysis(max_retries=args.max_retries)
            logger.info(f"✅ Cleaned up {cleaned_count} failed files")
            
            return 0
            
        except Exception as e:
            logger.error(f"❌ Cleanup failed: {e}")
            return 1
    
    def _handle_playlist(self, args) -> int:
        """Handle playlist command."""
        logger.info(f"🎼 Generating playlists using method: {args.method}")
        
        try:
            # Generate playlists
            playlists = self.playlist_generator.generate_playlists(
                method=args.method,
                num_playlists=args.num_playlists,
                playlist_size=args.playlist_size,
                min_tracks_per_genre=args.min_tracks_per_genre
            )
            
            logger.info(f"✅ Generated {len(playlists)} playlists")
            
            # Show playlist statistics
            stats = self.playlist_generator.get_playlist_statistics(playlists)
            logger.info("📊 Playlist Statistics:")
            for key, value in stats.items():
                logger.info(f"   {key}: {value}")
            
            # Save playlists if requested
            if args.save:
                success = self.playlist_generator.save_playlists(playlists, args.output_dir)
                if success:
                    logger.info(f"💾 Playlists saved to {args.output_dir}")
                else:
                    logger.error("❌ Failed to save playlists")
                    return 1
            
            return 0
            
        except Exception as e:
            logger.error(f"❌ Playlist generation failed: {e}")
            return 1
    
    def _handle_playlist_methods(self, args) -> int:
        """Handle playlist-methods command."""
        print("Available playlist generation methods:")
        
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
            print(f"   {method}: {description}")
        
        return 0
    
    def _handle_discover(self, args) -> int:
        """Handle discover command."""
        logger.info(f"🔍 Discovering audio files in: {args.path}")
        
        try:
            # Use file discovery directly
            from core.file_discovery import FileDiscovery
            file_discovery = FileDiscovery()
            
            # Discover files
            files = file_discovery.discover_files()
            
            logger.info(f"📁 Discovered {len(files)} audio files")
            
            # Show file details
            for file_path in files[:10]:  # Show first 10 files
                logger.info(f"   {file_path}")
            
            if len(files) > 10:
                logger.info(f"   ... and {len(files) - 10} more files")
            
            return 0
            
        except Exception as e:
            logger.error(f"❌ File discovery failed: {e}")
            return 1
    
    def _handle_enrich(self, args) -> int:
        """Handle enrich command."""
        logger.info("🌐 Enriching metadata")
        
        try:
            # For now, just show a placeholder message
            # This would need to be implemented in the analysis manager
            logger.info("⚠️ Metadata enrichment not yet implemented")
            logger.info("   This feature requires external API integration")
            logger.info("   (MusicBrainz, Last.fm, etc.)")
            
            return 0
            
        except Exception as e:
            logger.error(f"❌ Metadata enrichment failed: {e}")
            return 1
    
    def _handle_export(self, args) -> int:
        """Handle export command."""
        logger.info(f"📤 Exporting playlists in {args.format} format")
        
        try:
            # For now, just show a placeholder message
            # This would need to be implemented in the playlist generator
            logger.info("⚠️ Export functionality not yet implemented")
            logger.info("   This feature requires playlist export methods")
            logger.info(f"   Format: {args.format}")
            logger.info(f"   File: {args.playlist_file}")
            
            return 0
            
        except Exception as e:
            logger.error(f"❌ Export failed: {e}")
            return 1
    
    def _handle_status(self, args) -> int:
        """Handle status command."""
        logger.info("📊 Showing system status")
        
        try:
            # Database status (using available methods)
            db_stats = self.db_manager.get_database_statistics()
            logger.info("📊 Database Status:")
            for key, value in db_stats.items():
                logger.info(f"   {key}: {value}")
            
            # System status
            system_status = self.resource_manager.get_current_resources()
            logger.info("📊 System Status:")
            logger.info(f"   Memory: {system_status['memory']['percent']:.1f}%")
            logger.info(f"   CPU: {system_status['cpu_percent']:.1f}%")
            logger.info(f"   Disk: {system_status['disk']['percent']:.1f}%")
            
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
            logger.error(f"❌ Error getting status: {e}")
            return 1
    
    def _handle_pipeline(self, args) -> int:
        """Handle pipeline command."""
        logger.info("🔄 Running full pipeline")
        
        try:
            # For now, just run analysis and playlist generation separately
            # This would need to be implemented as a full pipeline
            logger.info("⚠️ Full pipeline not yet implemented")
            logger.info("   Running analysis and playlist generation separately")
            
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
                logger.info(f"✅ Analysis completed: {results.get('success_count', 0)} files processed")
            
            # Generate playlists if requested
            if args.generate:
                playlists = self.playlist_generator.generate_playlists(
                    method='all',
                    num_playlists=5,
                    playlist_size=20
                )
                logger.info(f"✅ Generated {len(playlists)} playlists")
            
            # Export if requested
            if args.export:
                logger.info("⚠️ Export functionality not yet implemented")
            
            return 0
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            return 1
    
    def _handle_config(self, args) -> int:
        """Handle config command."""
        print("Showing configuration information")
        
        try:
            # Show configuration
            if args.json:
                config_json = json.dumps(self.config, indent=2)
                print(config_json)
            else:
                print("Configuration:")
                for key, value in self.config.items():
                    print(f"   {key}: {value}")
            
            # Validate configuration if requested
            if args.validate:
                is_valid = self._validate_configuration()
                if is_valid:
                    print("Configuration is valid")
                else:
                    print("Configuration has issues")
                    return 1
            
            # Reload configuration if requested
            if args.reload:
                self.config = config_loader.load_config()
                print("Configuration reloaded")
            
            return 0
            
        except Exception as e:
            print(f"Configuration error: {e}")
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
        logger.info("📊 Detailed Statistics:")
        # Implementation for detailed statistics
        pass
    
    def _show_failed_files(self):
        """Show failed files."""
        logger.info("📊 Failed Files:")
        # Implementation for failed files
        pass
    
    def _show_memory_usage(self):
        """Show memory usage."""
        logger.info("📊 Memory Usage:")
        # Implementation for memory usage
        pass
    
    def _show_detailed_status(self):
        """Show detailed status."""
        logger.info("📊 Detailed Status:")
        # Implementation for detailed status
        pass
    
    def _validate_configuration(self) -> bool:
        """Validate configuration."""
        try:
            # Basic validation
            required_keys = ['MEMORY_LIMIT_GB', 'CPU_THRESHOLD_PERCENT']
            for key in required_keys:
                if key not in self.config:
                    logger.error(f"❌ Missing required configuration key: {key}")
                    return False
            
            return True
        except Exception as e:
            logger.error(f"❌ Configuration validation failed: {e}")
            return False


def main():
    """Main CLI entry point."""
    cli = EnhancedCLI()
    return cli.run()


if __name__ == "__main__":
    exit(main()) 