import argparse
import logging
import threading
import queue
import time
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from rich.text import Text
from datetime import datetime
import os
from utils.logging_setup import setup_colored_logging
log_level = os.getenv('LOG_LEVEL', 'DEBUG')
import logging
logging.getLogger().setLevel(getattr(logging, log_level.upper(), logging.DEBUG))

logger = logging.getLogger(__name__)

from typing import Tuple, Dict, Any, List, Optional

# Remove queue-based log handler and consumer thread setup from this file

class PlaylistGeneratorCLI:
    """Rich CLI for the Playlist Generator application."""
    def __init__(self) -> None:
        """Initialize the CLI interface."""
        self.console = Console()
        self.progress = None
        self.start_time = None

    def create_progress(self, description: str, total: int) -> Tuple[Progress, int]:
        """Create a new progress bar"""
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=40),  # Fixed width
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),    # Estimated time remaining
            console=self.console
        )
        progress.start()
        task_id = progress.add_task(description, total=total)
        return progress, task_id

    def show_analysis_progress(self, total_files: int) -> Tuple[Progress, int]:
        """Show progress for audio analysis"""
        return self.create_progress("[cyan]Analyzing audio files...", total_files)

    def show_playlist_generation_progress(self, total_steps: int) -> Tuple[Progress, int]:
        """Show progress for playlist generation"""
        return self.create_progress("[green]Generating playlists...", total_steps)

    def show_playlist_stats(self, stats: Dict[str, Dict[str, Any]]):
        """Display playlist statistics in a rich table"""
        if not stats:
            self.console.print(Panel(
                "[yellow]No playlist statistics available[/yellow]",
                title="Playlist Statistics",
                border_style="yellow"
            ))
            return

        layout = Layout()
        layout.split_column(
            Layout(name="header"),
            Layout(name="stats"),
            Layout(name="keys")
        )

        # Header
        header = Panel(
            f"[bold green]Generated {len(stats)} Playlists[/bold green]",
            style="blue"
        )
        layout["header"].update(header)

        # Stats table
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Playlist", style="cyan", max_width=28, overflow="fold")
        table.add_column("Tracks", justify="right", style="green")
        table.add_column("Duration", justify="right", style="green")
        table.add_column("Avg BPM", justify="right", style="yellow")
        table.add_column("Danceability", justify="right", style="yellow")

        for name, playlist_stats in stats.items():
            try:
                display_name = name if len(name) <= 28 else name[:25] + "..."
                table.add_row(
                    display_name,
                    str(playlist_stats.get('track_count', 0)),
                    f"{playlist_stats.get('total_duration', 0)/60:.1f}m",
                    f"{playlist_stats.get('avg_bpm', 0):.0f}",
                    f"{playlist_stats.get('avg_danceability', 0):.2f}"
                )
            except (TypeError, KeyError, AttributeError) as e:
                logger.warning(f"Error adding stats for playlist {name}: {str(e)}")
                table.add_row(name, "Error", "Error", "Error", "Error")

        layout["stats"].update(table)

        # Key distribution
        key_tables = []
        for name, playlist_stats in stats.items():
            if not isinstance(playlist_stats, dict):
                continue

            key_distribution = playlist_stats.get('key_distribution', {})
            if not key_distribution:
                continue

            try:
                key_table = Table(
                    title=f"Key Distribution - {name[:25] + '...' if len(name) > 28 else name}",
                    show_header=True,
                    title_style="bold magenta"
                )
                key_table.add_column("Key", style="cyan")
                key_table.add_column("Count", justify="right", style="green")

                # Show top 5 keys
                for key, count in sorted(
                    key_distribution.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]:
                    key_table.add_row(str(key), str(count))

                key_tables.append(key_table)
            except Exception as e:
                logger.warning(f"Error creating key distribution table for {name}: {str(e)}")

        if key_tables:
            # Only show up to 3 key tables at once for readability
            keys_layout = Layout()
            keys_layout.split_column(*[Layout(table) for table in key_tables[:3]])
            layout["keys"].update(keys_layout)
            if len(key_tables) > 3:
                self.console.print(Panel(
                    f"[yellow]Only showing top 3 key distributions out of {len(key_tables)} playlists.[/yellow]",
                    border_style="yellow"
                ))
        else:
            layout["keys"].update(Panel(
                "[yellow]No key distribution data available[/yellow]",
                border_style="yellow"
            ))

        self.console.print(layout)

    def show_error(self, error: str, details: Optional[str] = None):
        """Display error message in a panel"""
        text = f"[bold red]Error:[/bold red] {error}"
        if details:
            text += f"\n\n[dim]{details}[/dim]"
        
        self.console.print(Panel(
            text,
            title="❌ Error",
            border_style="red"
        ))

    def show_warning(self, message: str):
        """Display warning message"""
        self.console.print(f"[yellow]⚠️  Warning:[/yellow] {message}")

    def show_success(self, message: str):
        """Display success message"""
        self.console.print(f"[green]✓[/green] {message}")

    def show_file_errors(self, failed_files: List[str]):
        """Display list of failed files"""
        if not failed_files:
            return

        table = Table(title="Failed Files", show_header=True, header_style="bold red")
        table.add_column("File", style="red")
        table.add_column("Status", style="yellow")

        for file in failed_files:
            table.add_row(file, "Failed to process")

        self.console.print(table)

    def show_session_summary(self, stats: Dict[str, Any]):
        """Display session summary"""
        duration = time.time() - self.start_time
        
        panel = Panel(
            f"""[bold green]Session Complete[/bold green]

Duration: {duration:.1f} seconds
Processed Files: {stats.get('processed_files', 0)}
Generated Playlists: {stats.get('total_playlists', 0)}
Failed Files: {stats.get('failed_files', 0)}
""",
            title="📊 Session Summary",
            border_style="blue"
        )
        
        self.console.print(panel)

    def prompt_continue(self, message: str) -> bool:
        """Prompt user for confirmation"""
        return self.console.input(f"\n{message} (y/n): ").lower().startswith('y')

    def update_status(self, message: str):
        """Update status message"""
        self.console.print(f"[blue]ℹ[/blue] {message}")

    def show_library_statistics(self, stats: dict):
        """Display library/database statistics in a rich table (no outer panel). Also show genre breakdown tables."""
        from rich.table import Table
        # Main stats table (match show_config style)
        table = Table(title="Status", show_header=True, header_style="bold magenta")
        table.add_column("Stat", style="cyan")
        table.add_column("Value", style="green")
        # Playlist Stats section
        table.add_row("─" * 30, "─" * 30)
        table.add_row(Text("Playlist Stats", style="bold magenta"), "")
        playlists_per_mode = stats.get('playlists_per_mode')
        if playlists_per_mode:
            for mode, count in playlists_per_mode.items():
                table.add_row(f"  Playlists ({mode})", str(count))
        if 'avg_playlist_size' in stats:
            table.add_row("  Avg Playlist Size", f"{stats['avg_playlist_size']:.2f}")
        if 'largest_playlist_size' in stats:
            table.add_row("  Largest Playlist Size", str(stats['largest_playlist_size']))
        if 'smallest_playlist_size' in stats:
            table.add_row("  Smallest Playlist Size", str(stats['smallest_playlist_size']))
        if 'playlists_with_0_tracks' in stats:
            table.add_row("  Playlists with 0 Tracks", str(stats['playlists_with_0_tracks']))
        # Track Stats section
        table.add_row("─" * 30, "─" * 30)
        table.add_row(Text("Track Stats", style="bold magenta"), "")
        table.add_row("  Total Tracks", str(stats.get('total_tracks', 0)))
        table.add_row("  Tracks with Tags", str(stats.get('tracks_with_tags', 0)))
        table.add_row("  Tracks with Year", str(stats.get('tracks_with_year', 0)))
        # Use deduplicated count for Tracks with Genre
        table.add_row("  Tracks with Genre", str(stats.get('tracks_with_real_genre', 0)))
        if 'skipped_failed' in stats:
            table.add_row("  Skipped (Failed) Files", str(stats['skipped_failed']))
        if 'tracks_not_in_any_playlist' in stats:
            table.add_row("  Tracks Not in Any Playlist", str(stats['tracks_not_in_any_playlist']))
        if 'tracks_in_multiple_playlists' in stats:
            table.add_row("  Tracks in Multiple Playlists", str(stats['tracks_in_multiple_playlists']))
        if 'tracks_with_multiple_genres' in stats:
            table.add_row("  Tracks with Multiple Genres", str(stats['tracks_with_multiple_genres']))
        # Genre Stats section
        table.add_row("─" * 30, "─" * 30)
        table.add_row(Text("Genre Stats", style="bold magenta"), "")
        genre_counts = stats.get('genre_counts', {})
        if 'tracks_with_real_genre' in stats:
            table.add_row("  Track with genres", str(stats['tracks_with_real_genre']))
        if 'tracks_with_no_real_genre' in stats:
            table.add_row("  Others (no genres)", str(stats['tracks_with_no_real_genre']))
            unique_genres = len([g for g in genre_counts if g not in ("Other", "UnknownGenre", "", None)])
            table.add_row("  Unique Genres", str(unique_genres))
            if unique_genres > 0:
                most_common = max(((g, c) for g, c in genre_counts.items() if g not in ("Other", "UnknownGenre", "", None)), key=lambda x: x[1])
                table.add_row("  Most Common Genre", f"{most_common[0]} ({most_common[1]})")
            if 'top_5_genres' in stats and stats['top_5_genres']:
                top_genres = ", ".join([f"{g} ({c})" for g, c in stats['top_5_genres']])
                table.add_row("  Top 5 Genres", top_genres)
        # File Stats section
        table.add_row("─" * 30, "─" * 30)
        table.add_row(Text("File Stats", style="bold magenta"), "")
        if 'unique_file_extensions' in stats:
            table.add_row("  Unique File Extensions", ", ".join(stats['unique_file_extensions']))
        if 'tracks_per_extension' in stats:
            ext_counts = ", ".join([f"{ext}: {count}" for ext, count in stats['tracks_per_extension'].items()])
            table.add_row("  Tracks per Extension", ext_counts)
        # Dates section
        table.add_row("─" * 30, "─" * 30)
        table.add_row(Text("Dates", style="bold magenta"), "")
        if 'last_analysis_date' in stats:
            table.add_row("  Last Analysis Date", str(stats['last_analysis_date']))
        if 'last_playlist_update_date' in stats:
            table.add_row("  Last Playlist Update Date", str(stats['last_playlist_update_date']))
        self.console.print(table)
        # Playlist membership histogram (if present)
        hist = stats.get('track_playlist_membership', {})
        if hist:
            hist_table = Table(title="Track Playlist Membership", show_header=True, header_style="bold magenta")
            hist_table.add_column("# Playlists", justify="right", style="cyan")
            hist_table.add_column("# Tracks", justify="right", style="green")
            for n in sorted(hist):
                label = str(n)
                if int(n) >= 3:
                    label = f"{n}+"
                hist_table.add_row(label, str(hist[n]))
            self.console.print(hist_table)

    def show_analysis_summary(self, stats: dict, processed_this_run: int, failed_this_run: int, total_found: int, total_in_db: int, total_failed: int):
        from rich.table import Table
        summary_table = Table(title="Summary after analysis", show_header=True, header_style="bold magenta")
        summary_table.add_column("Stat", style="cyan")
        summary_table.add_column("Value", style="green")
        summary_table.add_row("Total tracks found in directory", str(total_found))
        summary_table.add_row("Total tracks in database", str(total_in_db))
        summary_table.add_row("Total failed tracks (in db)", str(total_failed))
        summary_table.add_row("Processed this run", str(processed_this_run))
        summary_table.add_row("Failed this run", str(failed_this_run))
        self.console.print(summary_table)

class CLIContextManager:
    """Context manager for CLI progress tracking"""
    def __init__(self, cli: PlaylistGeneratorCLI, total: int, description: str):
        self.cli = cli
        self.total = total
        self.description = description
        self.progress = None
        self.task_id = None

    def __enter__(self) -> Tuple[Progress, int]:
        self.progress, self.task_id = self.cli.create_progress(self.description, self.total)
        return self.progress, self.task_id

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.progress:
            self.progress.stop() 

class CLI:
    """Command-line interface utilities."""
    
    def __init__(self):
        self.console = Console()
        logger.debug("Initialized CLI with Rich console")
    
    def show_analysis_summary(self, stats: Dict[str, Any], processed_this_run: int = 0, 
                            failed_this_run: int = 0, total_found: int = 0, 
                            total_in_db: int = 0, total_failed: int = 0):
        """Display analysis summary with statistics."""
        logger.debug("Displaying analysis summary")
        logger.debug(f"Summary stats: processed={processed_this_run}, failed={failed_this_run}, total_found={total_found}, total_in_db={total_in_db}, total_failed={total_failed}")
        
        try:
            # Create summary table
            table = Table(title="Analysis Summary", show_header=True, header_style="bold magenta")
            table.add_column("Metric", style="cyan", no_wrap=True)
            table.add_column("Value", style="green")
            
            table.add_row("Files Processed This Run", str(processed_this_run))
            table.add_row("Files Failed This Run", str(failed_this_run))
            table.add_row("Total Files Found", str(total_found))
            table.add_row("Total Files in Database", str(total_in_db))
            table.add_row("Total Failed Files", str(total_failed))
            
            if stats:
                table.add_row("Total Tracks", str(stats.get('total_tracks', 0)))
                table.add_row("Unique Tracks", str(stats.get('unique_tracks', 0)))
                table.add_row("Total Playlists", str(stats.get('total_playlists', 0)))
            
            self.console.print(table)
            logger.info("Analysis summary displayed successfully")
            
        except Exception as e:
            logger.error(f"Error displaying analysis summary: {str(e)}")
            import traceback
            logger.error(f"Analysis summary error traceback: {traceback.format_exc()}")
    
    def show_playlist_summary(self, playlists: Dict[str, Dict[str, Any]]):
        """Display playlist generation summary."""
        logger.debug(f"Displaying playlist summary for {len(playlists)} playlists")
        
        try:
            if not playlists:
                self.console.print(Panel("No playlists generated", style="yellow"))
                logger.warning("No playlists to display")
                return
            
            # Create playlist table
            table = Table(title="Generated Playlists", show_header=True, header_style="bold magenta")
            table.add_column("Playlist Name", style="cyan", no_wrap=True)
            table.add_column("Track Count", style="green", justify="right")
            table.add_column("Type", style="blue")
            
            for name, data in playlists.items():
                track_count = len(data.get('tracks', []))
                playlist_type = data.get('features', {}).get('type', 'unknown')
                table.add_row(name, str(track_count), playlist_type)
            
            self.console.print(table)
            logger.info(f"Playlist summary displayed for {len(playlists)} playlists")
            
        except Exception as e:
            logger.error(f"Error displaying playlist summary: {str(e)}")
            import traceback
            logger.error(f"Playlist summary error traceback: {traceback.format_exc()}")
    
    def show_library_stats(self, stats: Dict[str, Any]):
        """Display library statistics."""
        logger.debug("Displaying library statistics")
        logger.debug(f"Library stats: {stats}")
        
        try:
            if not stats:
                self.console.print(Panel("No library statistics available", style="yellow"))
                logger.warning("No library statistics to display")
                return
            
            # Create stats table
            table = Table(title="Library Statistics", show_header=True, header_style="bold magenta")
            table.add_column("Metric", style="cyan", no_wrap=True)
            table.add_column("Value", style="green")
            
            for key, value in stats.items():
                if isinstance(value, dict):
                    # Handle nested dictionaries
                    for sub_key, sub_value in value.items():
                        table.add_row(f"{key}.{sub_key}", str(sub_value))
                else:
                    table.add_row(key, str(value))
            
            self.console.print(table)
            logger.info("Library statistics displayed successfully")
            
        except Exception as e:
            logger.error(f"Error displaying library statistics: {str(e)}")
            import traceback
            logger.error(f"Library stats error traceback: {traceback.format_exc()}")
    
    def show_error(self, message: str, details: str = None):
        """Display error message."""
        logger.error(f"CLI Error: {message}")
        if details:
            logger.error(f"Error details: {details}")
        
        try:
            error_text = Text(message, style="red")
            if details:
                error_text.append(f"\n\nDetails: {details}", style="dim")
            
            self.console.print(Panel(error_text, title="Error", border_style="red"))
            
        except Exception as e:
            logger.error(f"Error displaying error message: {str(e)}")
    
    def show_success(self, message: str):
        """Display success message."""
        logger.info(f"CLI Success: {message}")
        
        try:
            self.console.print(Panel(message, title="Success", border_style="green"))
            
        except Exception as e:
            logger.error(f"Error displaying success message: {str(e)}")
    
    def show_warning(self, message: str):
        """Display warning message."""
        logger.warning(f"CLI Warning: {message}")
        
        try:
            self.console.print(Panel(message, title="Warning", border_style="yellow"))
            
        except Exception as e:
            logger.error(f"Error displaying warning message: {str(e)}")
    
    def show_info(self, message: str):
        """Display info message."""
        logger.info(f"CLI Info: {message}")
        
        try:
            self.console.print(Panel(message, title="Info", border_style="blue"))
            
        except Exception as e:
            logger.error(f"Error displaying info message: {str(e)}")
    
    def add_playlist_stats(self, name: str, stats: Dict[str, Any]):
        """Add playlist statistics to display."""
        logger.debug(f"Adding playlist stats for '{name}': {stats}")
        
        try:
            # Create stats table for the playlist
            table = Table(title=f"Playlist Statistics: {name}", show_header=True, header_style="bold magenta")
            table.add_column("Metric", style="cyan", no_wrap=True)
            table.add_column("Value", style="green")
            
            for key, value in stats.items():
                table.add_row(key, str(value))
            
            self.console.print(table)
            logger.debug(f"Playlist stats displayed for '{name}'")
            
        except Exception as e:
            logger.error(f"Error adding stats for playlist {name}: {str(e)}")
            import traceback
            logger.error(f"Add playlist stats error traceback: {traceback.format_exc()}")
    
    def create_key_distribution_table(self, name: str, key_data: Dict[str, int]):
        """Create and display key distribution table."""
        logger.debug(f"Creating key distribution table for '{name}'")
        logger.debug(f"Key data: {key_data}")
        
        try:
            if not key_data:
                logger.warning(f"No key data available for playlist '{name}'")
                return
            
            # Create key distribution table
            table = Table(title=f"Key Distribution: {name}", show_header=True, header_style="bold magenta")
            table.add_column("Key", style="cyan", no_wrap=True)
            table.add_column("Count", style="green", justify="right")
            table.add_column("Percentage", style="blue", justify="right")
            
            total = sum(key_data.values())
            for key, count in sorted(key_data.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total * 100) if total > 0 else 0
                table.add_row(key, str(count), f"{percentage:.1f}%")
            
            self.console.print(table)
            logger.debug(f"Key distribution table displayed for '{name}'")
            
        except Exception as e:
            logger.error(f"Error creating key distribution table for {name}: {str(e)}")
            import traceback
            logger.error(f"Key distribution table error traceback: {traceback.format_exc()}")
    
    def show_progress(self, current: int, total: int, description: str = "Processing"):
        """Show progress bar."""
        logger.debug(f"Progress: {current}/{total} - {description}")
        
        try:
            percentage = (current / total * 100) if total > 0 else 0
            progress_text = f"{description}: {current}/{total} ({percentage:.1f}%)"
            self.console.print(f"[cyan]{progress_text}[/cyan]")
            
        except Exception as e:
            logger.error(f"Error showing progress: {str(e)}")
    
    def show_help(self):
        """Display help information."""
        logger.debug("Displaying help information")
        
        try:
            help_text = """
            [bold]Playlist Generator CLI[/bold]
            
            [bold]Available Commands:[/bold]
            • analyze: Analyze audio files and extract features
            • playlist: Generate playlists from analyzed files
            • stats: Show library statistics
            • help: Show this help message
            
            [bold]Common Options:[/bold]
            • --library: Specify music library directory
            • --workers: Number of parallel workers
            • --force: Force re-analysis of files
            • --method: Playlist generation method (kmeans, time, cache, etc.)
            
            [bold]Examples:[/bold]
            • audiolyzer --library /music --analyze
            • audiolyzer --playlist --method kmeans
            • audiolyzer --stats
            """
            
            self.console.print(Panel(help_text, title="Help", border_style="blue"))
            logger.info("Help information displayed successfully")
            
        except Exception as e:
            logger.error(f"Error displaying help: {str(e)}")
            import traceback
            logger.error(f"Help display error traceback: {traceback.format_exc()}") 