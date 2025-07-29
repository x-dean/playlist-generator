"""
Progress reporting system for user feedback.

This module provides real-time progress reporting for long-running
operations like file discovery, analysis, and playlist generation.
"""

import logging
import time
from typing import Optional, Dict, Any, Callable, List
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
from threading import Lock

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from rich.layout import Layout


class ProgressStatus(Enum):
    """Status of a progress operation."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ProgressStep:
    """Represents a step in a multi-step operation."""
    name: str
    description: str
    status: ProgressStatus = ProgressStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    progress: float = 0.0  # 0.0 to 1.0
    total_items: Optional[int] = None
    processed_items: int = 0
    error_message: Optional[str] = None
    
    @property
    def duration(self) -> Optional[float]:
        """Get duration in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        elif self.start_time:
            return (datetime.now() - self.start_time).total_seconds()
        return None
    
    @property
    def is_complete(self) -> bool:
        """Check if step is complete."""
        return self.status in [ProgressStatus.COMPLETED, ProgressStatus.FAILED, ProgressStatus.CANCELLED]


class ProgressReporter:
    """Main progress reporter for user feedback."""
    
    def __init__(self, console: Optional[Console] = None):
        """Initialize the progress reporter."""
        self.console = console or Console()
        self.logger = logging.getLogger(__name__)
        self._lock = Lock()
        self._current_operation: Optional[Dict[str, Any]] = None
        self._steps: List[ProgressStep] = []
        self._callbacks: Dict[str, Callable] = {}
    
    def start_operation(self, operation_name: str, description: str) -> str:
        """Start a new operation and return operation ID."""
        with self._lock:
            operation_id = f"{operation_name}_{int(time.time())}"
            
            self._current_operation = {
                "id": operation_id,
                "name": operation_name,
                "description": description,
                "start_time": datetime.now(),
                "steps": []
            }
            
            self.console.print(f"[bold blue]Starting: {operation_name}[/bold blue]")
            self.console.print(f"[dim]{description}[/dim]")
            
            return operation_id
    
    def add_step(self, step_name: str, description: str, total_items: Optional[int] = None) -> str:
        """Add a step to the current operation."""
        with self._lock:
            step = ProgressStep(
                name=step_name,
                description=description,
                total_items=total_items
            )
            
            self._steps.append(step)
            step_id = f"step_{len(self._steps)}"
            
            self.console.print(f"[cyan]→ {step_name}[/cyan]")
            if description:
                self.console.print(f"[dim]  {description}[/dim]")
            
            return step_id
    
    def start_step(self, step_index: int) -> None:
        """Start a specific step."""
        with self._lock:
            if 0 <= step_index < len(self._steps):
                step = self._steps[step_index]
                step.status = ProgressStatus.IN_PROGRESS
                step.start_time = datetime.now()
                step.progress = 0.0
                
                self.console.print(f"[green]Started: {step.name}[/green]")
    
    def update_step_progress(self, step_index: int, progress: float, processed_items: Optional[int] = None) -> None:
        """Update progress for a specific step."""
        with self._lock:
            if 0 <= step_index < len(self._steps):
                step = self._steps[step_index]
                step.progress = max(0.0, min(1.0, progress))
                
                if processed_items is not None:
                    step.processed_items = processed_items
                
                # Call callback if registered
                callback_key = f"step_{step_index}_progress"
                if callback_key in self._callbacks:
                    self._callbacks[callback_key](step)
    
    def complete_step(self, step_index: int, error_message: Optional[str] = None) -> None:
        """Complete a specific step."""
        with self._lock:
            if 0 <= step_index < len(self._steps):
                step = self._steps[step_index]
                step.end_time = datetime.now()
                
                if error_message:
                    step.status = ProgressStatus.FAILED
                    step.error_message = error_message
                    self.console.print(f"[red]Failed: {step.name} - {error_message}[/red]")
                else:
                    step.status = ProgressStatus.COMPLETED
                    step.progress = 1.0
                    duration = step.duration
                    self.console.print(f"[green]Completed: {step.name}[/green]")
                    if duration:
                        self.console.print(f"[dim]  Duration: {duration:.1f}s[/dim]")
    
    def complete_operation(self, operation_id: str, error_message: Optional[str] = None) -> None:
        """Complete the current operation."""
        with self._lock:
            if self._current_operation and self._current_operation["id"] == operation_id:
                end_time = datetime.now()
                duration = (end_time - self._current_operation["start_time"]).total_seconds()
                
                if error_message:
                    self.console.print(f"[red]Operation failed: {error_message}[/red]")
                else:
                    self.console.print(f"[bold green]Operation completed successfully![/bold green]")
                    self.console.print(f"[dim]Total duration: {duration:.1f}s[/dim]")
                
                # Generate summary
                self._generate_summary()
    
    def _generate_summary(self) -> None:
        """Generate a summary of the operation."""
        if not self._steps:
            return
        
        table = Table(title="Operation Summary")
        table.add_column("Step", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Duration", style="yellow")
        table.add_column("Progress", style="blue")
        
        for step in self._steps:
            status_style = {
                ProgressStatus.COMPLETED: "green",
                ProgressStatus.FAILED: "red",
                ProgressStatus.IN_PROGRESS: "yellow",
                ProgressStatus.PENDING: "dim"
            }.get(step.status, "white")
            
            duration_str = f"{step.duration:.1f}s" if step.duration else "N/A"
            progress_str = f"{step.progress * 100:.1f}%" if step.progress is not None else "N/A"
            
            table.add_row(
                step.name,
                f"[{status_style}]{step.status.value}[/{status_style}]",
                duration_str,
                progress_str
            )
        
        self.console.print(table)
    
    def register_callback(self, callback_id: str, callback: Callable) -> None:
        """Register a callback for progress updates."""
        self._callbacks[callback_id] = callback
    
    def get_current_progress(self) -> Optional[Dict[str, Any]]:
        """Get current progress information."""
        with self._lock:
            if not self._current_operation:
                return None
            
            return {
                "operation": self._current_operation,
                "steps": [
                    {
                        "name": step.name,
                        "status": step.status.value,
                        "progress": step.progress,
                        "processed_items": step.processed_items,
                        "total_items": step.total_items,
                        "duration": step.duration,
                        "error_message": step.error_message
                    }
                    for step in self._steps
                ]
            }


class RichProgressReporter(ProgressReporter):
    """Rich-based progress reporter with live updates."""
    
    def __init__(self, console: Optional[Console] = None):
        """Initialize the rich progress reporter."""
        super().__init__(console)
        self._progress = None
        self._live = None
        self._task_ids = {}
    
    def start_operation(self, operation_name: str, description: str) -> str:
        """Start a new operation with rich progress display."""
        operation_id = super().start_operation(operation_name, description)
        
        # Create rich progress
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console
        )
        
        self._live = Live(self._progress, console=self.console, refresh_per_second=10)
        self._live.start()
        
        return operation_id
    
    def add_step(self, step_name: str, description: str, total_items: Optional[int] = None) -> str:
        """Add a step with rich progress tracking."""
        step_id = super().add_step(step_name, description, total_items)
        
        if self._progress:
            task_id = self._progress.add_task(
                description,
                total=total_items or 100,
                visible=False  # Start hidden
            )
            self._task_ids[step_id] = task_id
        
        return step_id
    
    def start_step(self, step_index: int) -> None:
        """Start a step with rich progress display."""
        super().start_step(step_index)
        
        step_id = f"step_{step_index}"
        if step_id in self._task_ids and self._progress:
            task_id = self._task_ids[step_id]
            self._progress.update(task_id, visible=True)
    
    def update_step_progress(self, step_index: int, progress: float, processed_items: Optional[int] = None) -> None:
        """Update step progress with rich display."""
        super().update_step_progress(step_index, progress, processed_items)
        
        step_id = f"step_{step_index}"
        if step_id in self._task_ids and self._progress:
            task_id = self._task_ids[step_id]
            
            if processed_items is not None:
                self._progress.update(task_id, completed=processed_items)
            else:
                # Convert progress (0.0-1.0) to completed items
                step = self._steps[step_index]
                total = step.total_items or 100
                completed = int(progress * total)
                self._progress.update(task_id, completed=completed)
    
    def complete_step(self, step_index: int, error_message: Optional[str] = None) -> None:
        """Complete a step with rich display."""
        super().complete_step(step_index, error_message)
        
        step_id = f"step_{step_index}"
        if step_id in self._task_ids and self._progress:
            task_id = self._task_ids[step_id]
            step = self._steps[step_index]
            
            if step.status == ProgressStatus.COMPLETED:
                self._progress.update(task_id, completed=step.total_items or 100)
            elif step.status == ProgressStatus.FAILED:
                self._progress.update(task_id, description=f"[red]{step.name} - Failed[/red]")
    
    def complete_operation(self, operation_id: str, error_message: Optional[str] = None) -> None:
        """Complete operation and stop rich display."""
        super().complete_operation(operation_id, error_message)
        
        if self._live:
            self._live.stop()
        
        if self._progress:
            self._progress.stop()
    
    def create_summary_panel(self) -> Panel:
        """Create a rich summary panel."""
        if not self._steps:
            return Panel("No steps completed", title="Summary")
        
        # Calculate statistics
        completed_steps = [s for s in self._steps if s.status == ProgressStatus.COMPLETED]
        failed_steps = [s for s in self._steps if s.status == ProgressStatus.FAILED]
        total_duration = sum(s.duration or 0 for s in self._steps if s.duration)
        
        summary_text = f"""
[bold]Operation Summary[/bold]

Steps: {len(self._steps)}
✅ Completed: {len(completed_steps)}
❌ Failed: {len(failed_steps)}
⏱️ Total Duration: {total_duration:.1f}s

[bold]Step Details:[/bold]
"""
        
        for i, step in enumerate(self._steps, 1):
            status_icon = {
                ProgressStatus.COMPLETED: "✅",
                ProgressStatus.FAILED: "❌",
                ProgressStatus.IN_PROGRESS: "🔄",
                ProgressStatus.PENDING: "⏳"
            }.get(step.status, "❓")
            
            duration_str = f"{step.duration:.1f}s" if step.duration else "N/A"
            progress_str = f"{step.progress * 100:.1f}%" if step.progress is not None else "N/A"
            
            summary_text += f"{status_icon} {step.name}: {progress_str} ({duration_str})\n"
            
            if step.error_message:
                summary_text += f"   [red]Error: {step.error_message}[/red]\n"
        
        return Panel(summary_text, title="Operation Summary", border_style="blue")


# Global progress reporter instance
_progress_reporter: Optional[RichProgressReporter] = None


def get_progress_reporter() -> RichProgressReporter:
    """Get the global progress reporter instance."""
    global _progress_reporter
    if _progress_reporter is None:
        _progress_reporter = RichProgressReporter()
    return _progress_reporter 