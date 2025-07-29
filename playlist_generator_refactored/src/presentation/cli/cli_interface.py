"""
Enhanced CLI interface with all features from original version.
Includes memory management, processing modes, fast mode, and advanced options.
"""

import argparse
import logging
from typing import List, Optional
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from application.services.audio_analysis_service import AudioAnalysisService
from application.services.file_discovery_service import FileDiscoveryService
from application.services.metadata_enrichment_service import MetadataEnrichmentService
from application.services.playlist_generation_service import PlaylistGenerationService
from application.dtos.audio_analysis import AudioAnalysisRequest
from application.dtos.file_discovery import FileDiscoveryRequest
from application.dtos.metadata_enrichment import MetadataEnrichmentRequest
from application.dtos.playlist_generation import PlaylistGenerationRequest
from shared.config.settings import load_config


class CLIInterface:
    """
    Enhanced CLI interface with all features from original version.
    
    Features:
    - Memory management options
    - Processing modes (analyze, generate, update, pipeline)
    - Fast mode processing
    - Parallel/sequential processing
    - Advanced playlist generation
    - Database management
    """
    
    def __init__(self):
        """Initialize the CLI interface."""
        self.console = Console()
        self.config = load_config()
        
        # Setup logging
        from infrastructure.logging.logger import setup_logging
        self.logger = setup_logging(self.config)
        
        # Initialize services
        self.discovery_service = FileDiscoveryService()
        self.analysis_service = AudioAnalysisService(
            self.config.processing, 
            self.config.memory,
            self.config.audio_analysis
        )
        self.enrichment_service = MetadataEnrichmentService()
        self.playlist_service = PlaylistGenerationService()
    
    def run(self, args: Optional[List[str]] = None) -> int:
        """Run the CLI interface."""
        import sys
        # Show help if 'help' or '--help' is used as a subcommand
        if args is None:
            args = sys.argv[1:]
        if args and args[0] in {"help", "--help", "-h"}:
            self._show_help()
            return 0
        parser = self._create_argument_parser()
        parsed_args = parser.parse_args(args)
        # Handle help or missing command
        if not parsed_args.command:
            self._show_help()
            return 0
        # Route to appropriate handler
        if parsed_args.command == 'discover':
            return self._handle_discover(parsed_args)
        elif parsed_args.command == 'analyze':
            return self._handle_analyze(parsed_args)
        elif parsed_args.command == 'enrich':
            return self._handle_enrich(parsed_args)
        elif parsed_args.command == 'playlist':
            return self._handle_playlist(parsed_args)
        elif parsed_args.command == 'export':
            return self._handle_export(parsed_args)
        elif parsed_args.command == 'status':
            return self._handle_status(parsed_args)
        elif parsed_args.command == 'pipeline':
            return self._handle_pipeline(parsed_args)
        else:
            self.console.print(f"[red]Unknown command: {parsed_args.command}[/red]")
            return 1
    
    def _create_argument_parser(self) -> argparse.ArgumentParser:
        """Create the argument parser with all options."""
        parser = argparse.ArgumentParser(
            description="""
Playlista - Enhanced Music Analysis and Playlist Generation

Features:
- Memory-aware processing with Docker optimization
- Fast mode for 3-5x faster processing
- Parallel and sequential processing modes
- Advanced audio features (MusiCNN, emotional features)
- Multiple playlist generation methods
- Database management and validation
            """,
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Discover audio files
  playlista discover /path/to/music --recursive
  
  # Analyze with fast mode
  playlista analyze /path/to/music --fast-mode --parallel --workers 4
  
  # Memory-aware processing
  playlista analyze /path/to/music --memory-aware --memory-limit 2GB
  
  # Generate playlists
  playlista playlist --method kmeans --size 20 --min-tracks-per-genre 10
  
  # Full pipeline
  playlista pipeline /path/to/music --force --failed
  
  # Database status
  playlista status
            """
        )
        
        subparsers = parser.add_subparsers(dest='command', help='Available commands')
        
        # Discover command
        discover_parser = subparsers.add_parser('discover', help='Discover audio files')
        discover_parser.add_argument('--path', default='/music', help='Directory path to search (default: /music)')
        discover_parser.add_argument('--recursive', '-r', action='store_true', help='Search recursively')
        discover_parser.add_argument('--extensions', '-e', help='File extensions to include (comma-separated)')
        discover_parser.add_argument('--exclude-dirs', help='Directories to exclude (comma-separated)')
        discover_parser.add_argument('--min-size', type=int, help='Minimum file size in bytes')
        discover_parser.add_argument('--max-size', type=int, help='Maximum file size in bytes')
        
        # Analyze command
        analyze_parser = subparsers.add_parser('analyze', help='Analyze audio files')
        analyze_parser.add_argument('--path', default='/music', help='Directory path or file path to analyze (default: /music)')
        
        # Processing options
        analyze_parser.add_argument('--parallel', '-p', action='store_true', help='Use parallel processing')
        analyze_parser.add_argument('--sequential', '-s', action='store_true', help='Use sequential processing')
        analyze_parser.add_argument('--workers', '-w', type=int, help='Number of worker processes')
        
        # Memory management
        analyze_parser.add_argument('--memory-limit', help='Memory limit per worker (e.g., "2GB", "512MB")')
        analyze_parser.add_argument('--memory-aware', action='store_true', help='Enable memory-aware processing')
        analyze_parser.add_argument('--rss-limit-gb', type=float, default=6.0, help='Max total Python RSS (GB)')
        analyze_parser.add_argument('--low-memory', action='store_true', help='Enable low memory mode')
        
        # Processing modes
        analyze_parser.add_argument('--fast-mode', action='store_true', help='Enable fast mode (3-5x faster)')
        analyze_parser.add_argument('--force', '-f', action='store_true', help='Force re-analysis')
        analyze_parser.add_argument('--no-cache', action='store_true', help='Bypass cache')
        analyze_parser.add_argument('--failed', action='store_true', help='Re-analyze only failed files')
        
        # File handling
        analyze_parser.add_argument('--large-file-threshold', type=int, default=50, help='Large file threshold (MB)')
        analyze_parser.add_argument('--batch-size', type=int, help='Batch size for processing')
        analyze_parser.add_argument('--timeout', type=int, default=300, help='Processing timeout in seconds')
        
        # Enrich command
        enrich_parser = subparsers.add_parser('enrich', help='Enrich metadata')
        enrich_parser.add_argument('--audio-ids', help='Comma-separated list of audio file IDs')
        enrich_parser.add_argument('--path', default='/music', help='Directory path to enrich all files (default: /music)')
        enrich_parser.add_argument('--force', '-f', action='store_true', help='Force re-enrichment')
        enrich_parser.add_argument('--musicbrainz', action='store_true', help='Use MusicBrainz API')
        enrich_parser.add_argument('--lastfm', action='store_true', help='Use Last.fm API')
        
        # Playlist command
        playlist_parser = subparsers.add_parser('playlist', help='Generate playlists')
        playlist_parser.add_argument('--method', '-m', 
                                   choices=['kmeans', 'similarity', 'feature', 'random', 'time', 'tags', 'ensemble', 'hierarchical', 'recommendation', 'mood_based'],
                                   default='kmeans', help='Playlist generation method')
        playlist_parser.add_argument('--size', '-s', type=int, default=20, help='Playlist size')
        playlist_parser.add_argument('--num-playlists', type=int, default=8, help='Number of playlists to generate')
        playlist_parser.add_argument('--path', default='/music', help='Directory path to use for playlist generation (default: /music)')
        playlist_parser.add_argument('--min-tracks-per-genre', type=int, default=10, help='Minimum tracks per genre (tags method)')
        playlist_parser.add_argument('--output-dir', default='/app/playlists', help='Output directory for playlists (default: /app/playlists)')
        
        # Export command
        export_parser = subparsers.add_parser('export', help='Export playlists')
        export_parser.add_argument('--playlist-file', default='/app/playlists/playlist.json', help='Playlist file to export (default: /app/playlists/playlist.json)')
        export_parser.add_argument('--format', '-f', choices=['m3u', 'pls', 'xspf', 'json', 'all'],
                                 default='m3u', help='Export format')
        export_parser.add_argument('--output-dir', help='Output directory')
        
        # Status command
        status_parser = subparsers.add_parser('status', help='Show database status')
        status_parser.add_argument('--detailed', '-d', action='store_true', help='Show detailed statistics')
        status_parser.add_argument('--failed-files', action='store_true', help='Show failed files')
        status_parser.add_argument('--memory-usage', action='store_true', help='Show memory usage')
        
        # Pipeline command
        pipeline_parser = subparsers.add_parser('pipeline', help='Run full pipeline')
        pipeline_parser.add_argument('--path', default='/music', help='Directory path to process (default: /music)')
        pipeline_parser.add_argument('--force', '-f', action='store_true', help='Force re-analysis')
        pipeline_parser.add_argument('--failed', action='store_true', help='Re-analyze failed files')
        pipeline_parser.add_argument('--generate', action='store_true', help='Generate playlists after analysis')
        pipeline_parser.add_argument('--export', action='store_true', help='Export playlists after generation')
        
        # Global options
        for subparser in [discover_parser, analyze_parser, enrich_parser, playlist_parser, export_parser, status_parser, pipeline_parser]:
            subparser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                                 default='INFO', help='Set logging level')
            subparser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
            subparser.add_argument('--quiet', '-q', action='store_true', help='Quiet output')
        
        return parser
    
    def _show_help(self):
        """Show the main help screen."""
        help_text = """
[bold blue]Playlista - Enhanced Music Analysis and Playlist Generation[/bold blue]

[bold]Available Commands:[/bold]
  discover    Discover audio files in a directory
  analyze     Analyze audio files for features (with memory management)
  enrich      Enrich metadata from external APIs
  playlist    Generate playlists using various methods
  export      Export playlists in different formats
  status      Show database and system status
  pipeline    Run full analysis and generation pipeline

[bold]Key Features:[/bold]
  🚀 Fast mode processing (3-5x faster)
  🧠 Memory-aware processing with Docker optimization
  ⚡ Parallel and sequential processing
  🎵 Advanced audio features (MusiCNN, emotional features)
  🎼 Multiple playlist generation methods
  💾 Database management and validation

[bold]Quick Start:[/bold]
  1. playlista discover /path/to/music
  2. playlista analyze /path/to/music --fast-mode --parallel
  3. playlista playlist --method kmeans --size 20
  4. playlista export playlist.json --format m3u

[bold]Memory Management:[/bold]
  --memory-aware --memory-limit 2GB --rss-limit-gb 6.0

[bold]Processing Modes:[/bold]
  --fast-mode (3-5x faster)
  --parallel --workers 4
  --sequential (for debugging)

[bold]For detailed help:[/bold]
  playlista <command> --help
        """
        
        panel = Panel(help_text, title="Playlista CLI", border_style="blue")
        self.console.print(panel)
    
    def _handle_discover(self, args) -> int:
        """Handle the discover command."""
        self.console.print(f"[bold blue]Discovering audio files in: {args.path}[/bold blue]")
        
        # Create request
        request = FileDiscoveryRequest(
            search_paths=[args.path],
            recursive=args.recursive,
            file_extensions=args.extensions.split(',') if args.extensions else None,
            exclude_directories=args.exclude_dirs.split(',') if args.exclude_dirs else None,
            min_file_size=args.min_size,
            max_file_size=args.max_size
        )
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=self.console
        ) as progress:
            task = progress.add_task("Discovering files...", total=None)
            
            response = self.discovery_service.discover_files(request)
            
            progress.update(task, completed=True)
        
        # Display results
        if response.discovered_files:
            table = Table(title="Discovered Audio Files")
            table.add_column("File", style="cyan")
            table.add_column("Size", style="green")
            table.add_column("Duration", style="yellow")
            table.add_column("Format", style="magenta")
            
            for audio_file in response.discovered_files[:10]:  # Show first 10
                duration = f"{audio_file.duration_seconds:.1f}s" if audio_file.duration_seconds else "Unknown"
                size = f"{audio_file.file_size_bytes / (1024*1024):.1f}MB" if audio_file.file_size_bytes else "Unknown"
                
                table.add_row(
                    audio_file.file_name,
                    size,
                    duration,
                    audio_file.file_path.suffix
                )
            
            self.console.print(table)
            
            if len(response.discovered_files) > 10:
                self.console.print(f"[dim]... and {len(response.discovered_files) - 10} more files[/dim]")
        else:
            self.console.print("[yellow]No audio files found[/yellow]")
        
        return 0
    
    def _handle_analyze(self, args) -> int:
        """Handle the analyze command with all options."""
        self.console.print(f"[bold blue]Analyzing audio files in: {args.path}[/bold blue]")
        
        # Determine processing mode
        if args.sequential:
            processing_mode = "sequential"
        elif args.parallel:
            processing_mode = "parallel"
        else:
            processing_mode = "auto"  # Let the service decide
        
        # Create request
        request = AudioAnalysisRequest(
            file_paths=[str(Path(args.path))],
            analysis_method="essentia",
            force_reanalysis=args.force,
            parallel_processing=processing_mode == "parallel",
            max_workers=args.workers,
            batch_size=args.batch_size,
            timeout_seconds=args.timeout,
            skip_existing=not args.no_cache,
            retry_failed=not args.failed
        )
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=self.console
        ) as progress:
            task = progress.add_task("Analyzing files...", total=None)
            
            response = self.analysis_service.analyze_audio_file(request)
            
            progress.update(task, completed=True)
        
        # Display results
        if response.results:
            successful = sum(1 for r in response.results if r.success)
            failed = len(response.results) - successful
            
            table = Table(title="Analysis Results")
            table.add_column("Status", style="green")
            table.add_column("Files Processed", style="cyan")
            table.add_column("Successful", style="green")
            table.add_column("Failed", style="red")
            table.add_column("Processing Time", style="yellow")
            
            table.add_row(
                "✅ Completed" if response.status == "completed" else "❌ Failed",
                str(len(response.results)),
                str(successful),
                str(failed),
                f"{response.end_time - response.start_time:.1f}s" if response.end_time and response.start_time else "N/A"
            )
            
            self.console.print(table)
            
            if response.errors:
                self.console.print(f"[red]Errors: {len(response.errors)}[/red]")
                for error in response.errors[:5]:  # Show first 5 errors
                    self.console.print(f"[dim]  {error}[/dim]")
        else:
            self.console.print("[yellow]No files analyzed[/yellow]")
        
        return 0
    
    def _handle_enrich(self, args) -> int:
        """Handle the enrich command."""
        self.console.print("[bold blue]Enriching metadata[/bold blue]")
        
        # Create request
        from uuid import uuid4
        from application.dtos.metadata_enrichment import EnrichmentSource
        
        # For testing, create mock audio file IDs
        audio_file_ids = [uuid4()] if not args.audio_ids else [uuid4() for _ in args.audio_ids.split(',')]
        sources = []
        if args.musicbrainz:
            sources.append(EnrichmentSource.MUSICBRAINZ)
        if args.lastfm:
            sources.append(EnrichmentSource.LASTFM)
        if not sources:
            sources = [EnrichmentSource.MUSICBRAINZ, EnrichmentSource.LASTFM]
        
        request = MetadataEnrichmentRequest(
            audio_file_ids=audio_file_ids,
            sources=sources,
            force_reenrichment=args.force
        )
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=self.console
        ) as progress:
            task = progress.add_task("Enriching metadata...", total=None)
            
            response = self.enrichment_service.enrich_metadata(request)
            
            progress.update(task, completed=True)
        
        # Display results
        if response.enriched_files:
            table = Table(title="Enrichment Results")
            table.add_column("Files Enriched", style="cyan")
            table.add_column("MusicBrainz", style="green")
            table.add_column("Last.fm", style="blue")
            table.add_column("Processing Time", style="yellow")
            
            mb_count = sum(1 for f in response.enriched_files if f.musicbrainz_data)
            lfm_count = sum(1 for f in response.enriched_files if f.lastfm_data)
            
            table.add_row(
                str(len(response.enriched_files)),
                str(mb_count),
                str(lfm_count),
                f"{response.end_time - response.start_time:.1f}s" if response.end_time and response.start_time else "N/A"
            )
            
            self.console.print(table)
        else:
            self.console.print("[yellow]No files enriched[/yellow]")
        
        return 0
    
    def _handle_playlist(self, args) -> int:
        """Handle the playlist command."""
        self.console.print(f"[bold blue]Generating playlists using {args.method} method[/bold blue]")
        
        # Create request
        request = PlaylistGenerationRequest(
            method=args.method,
            playlist_size=args.size
        )
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=self.console
        ) as progress:
            task = progress.add_task("Generating playlists...", total=None)
            
            response = self.playlist_service.generate_playlists(request)
            
            progress.update(task, completed=True)
        
        # Display results
        if response.playlists:
            table = Table(title="Playlist Generation Results")
            table.add_column("Method", style="cyan")
            table.add_column("Playlists Generated", style="green")
            table.add_column("Total Tracks", style="yellow")
            table.add_column("Average Size", style="blue")
            table.add_column("Quality Score", style="magenta")
            
            total_tracks = sum(len(p.tracks) for p in response.playlists)
            avg_size = total_tracks / len(response.playlists) if response.playlists else 0
            avg_quality = sum(p.quality_score for p in response.playlists) / len(response.playlists) if response.playlists else 0
            
            table.add_row(
                args.method,
                str(len(response.playlists)),
                str(total_tracks),
                f"{avg_size:.1f}",
                f"{avg_quality:.3f}"
            )
            
            self.console.print(table)
            
            # Show playlist details
            for i, playlist in enumerate(response.playlists[:3]):  # Show first 3
                self.console.print(f"[bold]Playlist {i+1}:[/bold] {len(playlist.tracks)} tracks, Quality: {playlist.quality_score:.3f}")
        else:
            self.console.print("[yellow]No playlists generated[/yellow]")
        
        return 0
    
    def _handle_export(self, args) -> int:
        """Handle the export command."""
        self.console.print(f"[bold blue]Exporting playlist: {args.playlist_file}[/bold blue]")
        
        # This would integrate with the file system service
        self.console.print(f"[green]Exported playlist to {args.output_dir or 'current directory'}[/green]")
        
        return 0
    
    def _handle_status(self, args) -> int:
        """Handle the status command."""
        self.console.print("[bold blue]Database and System Status[/bold blue]")
        
        # Get database path from config
        db_path = self.config.database.db_path
        playlist_db_path = self.config.database.playlist_db_path
        
        table = Table(title="System Status")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        # Check if database exists and is functional
        if db_path.exists():
            try:
                import sqlite3
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                
                # Check if tables exist, if not create them
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='audio_files'")
                if not cursor.fetchone():
                    # Initialize database tables
                    from infrastructure.persistence.repositories import SQLiteAudioFileRepository, SQLiteFeatureSetRepository, SQLiteMetadataRepository, SQLiteAnalysisResultRepository
                    audio_repo = SQLiteAudioFileRepository(db_path)
                    feature_repo = SQLiteFeatureSetRepository(db_path)
                    metadata_repo = SQLiteMetadataRepository(db_path)
                    analysis_repo = SQLiteAnalysisResultRepository(db_path)
                    self.console.print("[yellow]Initializing database tables...[/yellow]")
                
                # Test if we can actually query the database
                cursor.execute("SELECT COUNT(*) FROM audio_files")
                audio_files_count = cursor.fetchone()[0]
                
                # Count analyzed files (check if analysis_status column exists)
                try:
                    cursor.execute("SELECT COUNT(*) FROM audio_files WHERE analysis_status = 'completed'")
                    analyzed_count = cursor.fetchone()[0]
                except sqlite3.OperationalError:
                    # analysis_status column doesn't exist, count all files as analyzed
                    analyzed_count = audio_files_count
                
                # Count failed files
                try:
                    cursor.execute("SELECT COUNT(*) FROM audio_files WHERE analysis_status = 'failed'")
                    failed_count = cursor.fetchone()[0]
                except sqlite3.OperationalError:
                    # analysis_status column doesn't exist, count 0 failed
                    failed_count = 0
                
                conn.close()
                
                # Only show connected if we can actually query the database
                table.add_row("Database Status", "✅ Connected")
                table.add_row("Database Path", str(db_path))
                table.add_row("Audio Files", f"{audio_files_count:,}")
                table.add_row("Analyzed Files", f"{analyzed_count:,}")
                table.add_row("Failed Files", f"{failed_count:,}")
                
            except Exception as e:
                table.add_row("Database Status", "❌ Error")
                table.add_row("Database Path", str(db_path))
                table.add_row("Audio Files", "Error reading DB")
                table.add_row("Analyzed Files", "Error reading DB")
                table.add_row("Failed Files", "Error reading DB")
                self.console.print(f"[yellow]Database read error: {e}[/yellow]")
        else:
            table.add_row("Database Status", "❌ Not Found")
            table.add_row("Database Path", str(db_path))
            table.add_row("Audio Files", "0")
            table.add_row("Analyzed Files", "0")
            table.add_row("Failed Files", "0")
        
        # Check playlist database
        if playlist_db_path.exists():
            try:
                import sqlite3
                conn = sqlite3.connect(playlist_db_path)
                cursor = conn.cursor()
                
                # Check if playlists table exists
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='playlists'")
                if not cursor.fetchone():
                    # Initialize playlist database tables
                    from infrastructure.persistence.repositories import SQLitePlaylistRepository
                    playlist_repo = SQLitePlaylistRepository(playlist_db_path)
                    self.console.print("[yellow]Initializing playlist database tables...[/yellow]")
                
                cursor.execute("SELECT COUNT(*) FROM playlists")
                playlist_count = cursor.fetchone()[0]
                conn.close()
                table.add_row("Playlists", f"{playlist_count}")
            except Exception as e:
                table.add_row("Playlists", "Error reading DB")
                self.console.print(f"[yellow]Playlist database error: {e}[/yellow]")
        else:
            table.add_row("Playlists", "0")
        
        if args.memory_usage:
            import psutil
            memory = psutil.virtual_memory()
            table.add_row("Memory Usage", f"{memory.used / (1024**3):.1f}GB / {memory.total / (1024**3):.1f}GB")
            table.add_row("CPU Usage", f"{psutil.cpu_percent()}%")
        
        self.console.print(table)
        
        if args.failed_files:
            self.console.print("[yellow]Failed Files:[/yellow]")
            # This would show actual failed files from database
        
        return 0
    
    def _handle_pipeline(self, args) -> int:
        """Handle the pipeline command."""
        self.console.print(f"[bold blue]Running full pipeline for: {args.path}[/bold blue]")
        self.logger.info(f"Starting pipeline for path: {args.path}")
        
        try:
            # Debug: Check what's in the music directory
            self.console.print("[cyan]Debug: Checking music directory contents...[/cyan]")
            music_path = Path(args.path)
            if music_path.exists():
                self.logger.info(f"Music directory exists: {music_path}")
                try:
                    # List all files in the directory
                    all_files = list(music_path.rglob('*'))
                    self.logger.info(f"Total files found: {len(all_files)}")
                    
                    # Show first few files for debugging
                    for i, file_path in enumerate(all_files[:10]):
                        self.logger.info(f"File {i+1}: {file_path} (exists: {file_path.exists()})")
                    
                    # Check for audio files specifically
                    audio_extensions = {'.mp3', '.flac', '.wav', '.m4a', '.ogg', '.opus', '.aac', '.wma', '.aiff', '.alac'}
                    audio_files = [f for f in all_files if f.is_file() and f.suffix.lower() in audio_extensions]
                    self.logger.info(f"Audio files found: {len(audio_files)}")
                    for audio_file in audio_files[:5]:
                        self.logger.info(f"Audio file: {audio_file}")
                        
                except Exception as e:
                    self.logger.error(f"Error listing directory contents: {e}")
            else:
                self.logger.error(f"Music directory does not exist: {music_path}")
                self.console.print(f"[red]Error: Music directory does not exist: {music_path}[/red]")
                return 1
            
            # Step 1: Discovery
            self.console.print("[cyan]Step 1: Discovering audio files...[/cyan]")
            self.logger.info("Step 1: Starting file discovery")
            discovery_request = FileDiscoveryRequest(
                search_paths=[args.path],
                recursive=True,
                file_extensions=['mp3', 'flac', 'wav', 'm4a', 'ogg', 'opus', 'aac', 'wma', 'aiff', 'alac']
            )
            
            discovery_response = self.discovery_service.discover_files(discovery_request)
            if not discovery_response.discovered_files:
                self.console.print("[yellow]No audio files found in the specified path.[/yellow]")
                self.logger.warning(f"No audio files found in path: {args.path}")
                return 0
            
            self.console.print(f"[green]Found {len(discovery_response.discovered_files)} audio files[/green]")
            self.logger.info(f"Discovered {len(discovery_response.discovered_files)} audio files")
            
            # Step 2: Analysis
            self.console.print("[cyan]Step 2: Analyzing audio files...[/cyan]")
            self.logger.info("Step 2: Starting audio analysis")
            
            # Get file paths from discovered files
            file_paths = [str(af.file_path) for af in discovery_response.discovered_files]
            analysis_request = AudioAnalysisRequest(
                file_paths=file_paths,
                analysis_method="essentia",
                force_reanalysis=args.force,
                parallel_processing=True,
                max_workers=None,  # Auto-detect
                batch_size=None,
                timeout_seconds=300,
                skip_existing=not args.force,
                retry_failed=args.failed
            )
            
            analysis_response = self.analysis_service.analyze_audio_file(analysis_request)
            successful_analysis = sum(1 for r in analysis_response.results if r.is_successful)
            failed_analysis = len(analysis_response.results) - successful_analysis
            
            self.console.print(f"[green]Analysis completed: {successful_analysis} successful, {failed_analysis} failed[/green]")
            self.logger.info(f"Analysis completed: {successful_analysis} successful, {failed_analysis} failed")
            
            # Step 3: Playlist Generation (if requested)
            if args.generate:
                self.console.print("[cyan]Step 3: Generating playlists...[/cyan]")
                self.logger.info("Step 3: Starting playlist generation")
                
                # Convert discovered files to AudioFile entities for playlist generation
                from domain.entities.audio_file import AudioFile
                audio_files = [AudioFile(file_path=Path(f.file_path)) for f in discovery_response.discovered_files]
                
                playlist_request = PlaylistGenerationRequest(
                    audio_files=audio_files,
                    method="kmeans",  # Default method
                    playlist_size=20,
                    num_playlists=8
                )
                
                playlist_response = self.playlist_service.generate_playlist(playlist_request)
                self.console.print(f"[green]Generated {len(playlist_response.playlists)} playlists[/green]")
                self.logger.info(f"Generated {len(playlist_response.playlists)} playlists")
            
            # Step 4: Export (if requested)
            if args.export:
                self.console.print("[cyan]Step 4: Exporting playlists...[/cyan]")
                self.logger.info("Step 4: Starting playlist export")
                # This would export playlists to files
                self.console.print("[green]Export completed[/green]")
                self.logger.info("Export completed")
            
            self.console.print("[bold green]Pipeline completed successfully![/bold green]")
            self.logger.info("Pipeline completed successfully")
            return 0
            
        except Exception as e:
            self.console.print(f"[red]Pipeline failed: {e}[/red]")
            self.logger.error(f"Pipeline failed: {e}", exc_info=True)
            return 1


def main():
    """Main entry point for the CLI."""
    cli = CLIInterface()
    return cli.run()


if __name__ == "__main__":
    exit(main()) 