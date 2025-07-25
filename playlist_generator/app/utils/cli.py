import logging
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from rich.text import Text
from datetime import datetime
import time
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger()
console = Console()

class PlaylistGeneratorCLI:
    """Rich CLI for the Playlist Generator application."""
    def __init__(self) -> None:
        """Initialize the CLI interface."""
        self.console = Console()
        self.progress = None
        self.start_time = None

    def start_session(self) -> None:
        """Start a new generation session with welcome message."""
        self.start_time = time.time()
        logo = '''[bold magenta]
   ____        _     _ _       _           
  / __ \ _   _| |__ (_) | ___ | |__   __ _ 
 / / _` | | | | '_ \| | |/ _ \| '_ \ / _` |
| | (_| | |_| | |_) | | | (_) | |_) | (_| |
 \ \__,_|\__,_|_.__/|_|_|\___/|_.__/ \__,_|
  \____/                                    
[/bold magenta]'''
        self.console.print(Panel(logo, title="🎵 Playlist Generator", border_style="magenta"))

    def show_config(self, config: dict) -> None:
        """Display current configuration.

        Args:
            config (dict): Configuration dictionary.
        """
        table = Table(title="Configuration", show_header=True, header_style="bold magenta")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="green")

        for key, value in config.items():
            table.add_row(key.replace('_', ' ').title(), str(value))

        self.console.print(table)

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