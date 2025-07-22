import logging
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from rich.text import Text
from datetime import datetime
import time
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)
console = Console()

class PlaylistGeneratorCLI:
    def __init__(self):
        self.console = Console()
        self.progress = None
        self.start_time = None

    def start_session(self):
        """Start a new generation session with welcome message"""
        self.start_time = time.time()
        self.console.print(Panel(
            "[bold green]Welcome to Playlist Generator[/bold green]\n\n"
            "This tool will analyze your music library and create personalized playlists "
            "based on audio characteristics and patterns.",
            title="🎵 Playlist Generator",
            border_style="blue"
        ))

    def show_config(self, config: Dict[str, Any]):
        """Display current configuration"""
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
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
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
Peak Memory Usage: {stats.get('peak_memory_mb', 0):.1f} MB
Peak CPU Usage: {stats.get('peak_cpu', 0):.1f}%
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