"""
Simple Progress Bar for Playlist Generator Simple.
Provides basic progress tracking using rich library.
"""

import time
from typing import Optional, Dict, Any, List
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .logging_setup import get_logger

logger = get_logger('playlista.progress')


class SimpleProgressBar:
    """
    Simple progress bar implementation using rich library.
    
    Features:
    - Basic progress tracking
    - Time elapsed display
    - File processing progress
    - Error handling
    """
    
    def __init__(self, show_progress: bool = True):
        """
        Initialize the progress bar.
        
        Args:
            show_progress: Whether to show progress bars (can be disabled for logging-only mode)
        """
        self.show_progress = show_progress
        self.console = Console()
        self.current_progress = None
        self.current_task = None
        
    def start_file_processing(self, total_files: int, description: str = "Processing files") -> None:
        """
        Start progress tracking for file processing.
        
        Args:
            total_files: Total number of files to process
            description: Description for the progress bar
        """
        if not self.show_progress:
            logger.info(f"🔄 {description}: {total_files} files")
            return
            
        self.current_progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=self.console
        )
        
        self.current_progress.start()
        self.current_task = self.current_progress.add_task(
            description, total=total_files
        )
        
    def update_file_progress(self, processed_files: int, current_file: str = None) -> None:
        """
        Update progress for file processing.
        
        Args:
            processed_files: Number of files processed so far
            current_file: Current file being processed (optional)
        """
        if not self.show_progress or not self.current_progress or not self.current_task:
            if current_file:
                logger.info(f"📁 Processing: {current_file}")
            return
            
        self.current_progress.update(self.current_task, completed=processed_files)
        
        if current_file:
            # Update description to show current file
            self.current_progress.update(
                self.current_task, 
                description=f"Processing: {current_file}"
            )
    
    def complete_file_processing(self, total_processed: int, success_count: int, failed_count: int) -> None:
        """
        Complete file processing and show results.
        
        Args:
            total_processed: Total number of files processed
            success_count: Number of successful files
            failed_count: Number of failed files
        """
        if not self.show_progress or not self.current_progress:
            logger.info(f"✅ Processing completed: {success_count} successful, {failed_count} failed")
            return
            
        # Update to 100% completion
        self.current_progress.update(self.current_task, completed=total_processed)
        
        # Stop progress bar
        self.current_progress.stop()
        
        # Show results
        results_table = Table(title="Processing Results")
        results_table.add_column("Metric", style="cyan")
        results_table.add_column("Count", style="green")
        
        results_table.add_row("Total Files", str(total_processed))
        results_table.add_row("Successful", str(success_count))
        results_table.add_row("Failed", str(failed_count))
        
        if total_processed > 0:
            success_rate = (success_count / total_processed) * 100
            results_table.add_row("Success Rate", f"{success_rate:.1f}%")
        
        self.console.print(results_table)
        
        # Clean up
        self.current_progress = None
        self.current_task = None
    
    def start_analysis(self, total_files: int, analysis_type: str = "Analysis") -> None:
        """
        Start progress tracking for analysis.
        
        Args:
            total_files: Total number of files to analyze
            analysis_type: Type of analysis (e.g., "Sequential", "Parallel")
        """
        if not self.show_progress:
            logger.info(f"🎵 Starting {analysis_type}: {total_files} files")
            return
            
        self.current_progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=self.console
        )
        
        self.current_progress.start()
        self.current_task = self.current_progress.add_task(
            f"{analysis_type} Progress", total=total_files
        )
    
    def update_analysis_progress(self, processed_files: int, current_file: str = None) -> None:
        """
        Update progress for analysis.
        
        Args:
            processed_files: Number of files analyzed so far
            current_file: Current file being analyzed (optional)
        """
        if not self.show_progress or not self.current_progress or not self.current_task:
            if current_file:
                logger.info(f"🎵 Analyzing: {current_file}")
            return
            
        self.current_progress.update(self.current_task, completed=processed_files)
        
        if current_file:
            # Update description to show current file
            self.current_progress.update(
                self.current_task, 
                description=f"Analyzing: {current_file}"
            )
    
    def complete_analysis(self, total_analyzed: int, success_count: int, failed_count: int, 
                         analysis_type: str = "Analysis") -> None:
        """
        Complete analysis and show results.
        
        Args:
            total_analyzed: Total number of files analyzed
            success_count: Number of successful analyses
            failed_count: Number of failed analyses
            analysis_type: Type of analysis completed
        """
        if not self.show_progress or not self.current_progress:
            logger.info(f"✅ {analysis_type} completed: {success_count} successful, {failed_count} failed")
            return
            
        # Update to 100% completion
        self.current_progress.update(self.current_task, completed=total_analyzed)
        
        # Stop progress bar
        self.current_progress.stop()
        
        # Show results
        results_table = Table(title=f"{analysis_type} Results")
        results_table.add_column("Metric", style="cyan")
        results_table.add_column("Count", style="green")
        
        results_table.add_row("Total Files", str(total_analyzed))
        results_table.add_row("Successful", str(success_count))
        results_table.add_row("Failed", str(failed_count))
        
        if total_analyzed > 0:
            success_rate = (success_count / total_analyzed) * 100
            results_table.add_row("Success Rate", f"{success_rate:.1f}%")
        
        self.console.print(results_table)
        
        # Clean up
        self.current_progress = None
        self.current_task = None
    
    def show_status(self, message: str, style: str = "blue") -> None:
        """
        Show a status message.
        
        Args:
            message: Status message to display
            style: Rich style for the message
        """
        if not self.show_progress:
            logger.info(message)
            return
            
        self.console.print(f"[{style}]{message}[/{style}]")
    
    def show_error(self, message: str) -> None:
        """
        Show an error message.
        
        Args:
            message: Error message to display
        """
        if not self.show_progress:
            logger.error(message)
            return
            
        self.console.print(f"[red]❌ {message}[/red]")
    
    def show_success(self, message: str) -> None:
        """
        Show a success message.
        
        Args:
            message: Success message to display
        """
        if not self.show_progress:
            logger.info(message)
            return
            
        self.console.print(f"[green]✅ {message}[/green]")
    
    def show_warning(self, message: str) -> None:
        """
        Show a warning message.
        
        Args:
            message: Warning message to display
        """
        if not self.show_progress:
            logger.warning(message)
            return
            
        self.console.print(f"[yellow]⚠️ {message}[/yellow]")


# Global progress bar instance
_progress_bar: Optional[SimpleProgressBar] = None


def get_progress_bar(show_progress: bool = True) -> SimpleProgressBar:
    """
    Get the global progress bar instance.
    
    Args:
        show_progress: Whether to show progress bars
        
    Returns:
        Progress bar instance
    """
    global _progress_bar
    if _progress_bar is None:
        _progress_bar = SimpleProgressBar(show_progress=show_progress)
    return _progress_bar


def set_progress_bar(progress_bar: SimpleProgressBar) -> None:
    """
    Set the global progress bar instance.
    
    Args:
        progress_bar: Progress bar instance to set
    """
    global _progress_bar
    _progress_bar = progress_bar 