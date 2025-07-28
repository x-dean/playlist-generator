"""
Shared MusiCNN status for progress bar updates.
"""
import threading
import time
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.console import Console

# Shared status variable
musicnn_status = "Waiting for MusiCNN extraction..."
status_lock = threading.Lock()

# Use a shared console instance to prevent conflicts
try:
    # Try to get the existing console from the main script
    import sys
    if hasattr(sys, '_console_instance'):
        console = sys._console_instance
    else:
        console = Console()
        sys._console_instance = console
except:
    console = Console()

# Global progress bar for MusiCNN steps
musicnn_progress = None
musicnn_task = None
_initialized = False

# MusiCNN step definitions
MUSICNN_STEPS = [
    "Loading JSON metadata",
    "Loading audio file", 
    "Running activations",
    "Running embeddings",
    "Finalizing results"
]

def init_musicnn_progress():
    """Initialize the MusiCNN progress bar only once."""
    global musicnn_progress, musicnn_task, _initialized
    
    if _initialized and musicnn_progress is not None:
        return  # Already initialized and running
    
    # If we have a progress bar but it's not initialized, stop it first
    if musicnn_progress is not None and not _initialized:
        try:
            musicnn_progress.stop()
        except:
            pass
        musicnn_progress = None
        musicnn_task = None
    
    # Print a separator line to create space
    console.print("\n" + "="*80)
    console.print("[bold blue]MusiCNN Processing Status:[/bold blue]")
    console.print("="*80)
    
    musicnn_progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Step:"),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
        transient=False
    )
    musicnn_task = musicnn_progress.add_task("Waiting for files...", total=len(MUSICNN_STEPS))
    musicnn_progress.start()
    _initialized = True

def update_musicnn_status(new_status):
    """Update the MusiCNN status in the progress bar."""
    global musicnn_progress, musicnn_task, _initialized
    
    # Initialize if not already done
    if not _initialized:
        init_musicnn_progress()
    
    if musicnn_progress and musicnn_task:
        musicnn_progress.update(musicnn_task, description=new_status)

def update_musicnn_file_status(filename):
    """Update status for a new file being processed."""
    global musicnn_progress, musicnn_task, _initialized
    
    # Initialize if not already done
    if not _initialized:
        init_musicnn_progress()
    
    if musicnn_progress and musicnn_task:
        # Reset progress for new file
        musicnn_progress.reset(musicnn_task)
        musicnn_progress.update(musicnn_task, description=f"Starting: {filename}")

def update_musicnn_step_progress(step_index, description=None):
    """Update progress for a specific MusiCNN step."""
    global musicnn_progress, musicnn_task, _initialized
    
    # Initialize if not already done
    if not _initialized:
        init_musicnn_progress()
    
    if musicnn_progress and musicnn_task:
        if step_index < len(MUSICNN_STEPS):
            step_name = MUSICNN_STEPS[step_index]
            if description:
                step_name = f"{step_name}: {description}"
            # Update the description and advance the progress bar
            musicnn_progress.update(musicnn_task, description=step_name)
            # Advance to the next step (this fills the progress bar)
            musicnn_progress.advance(musicnn_task)

def finish_musicnn_processing():
    """Call this when all MusiCNN processing is complete."""
    global musicnn_progress, _initialized
    if musicnn_progress:
        musicnn_progress.stop()
        # Print a completion message
        console.print("[bold green]✓ MusiCNN processing completed[/bold green]\n")
        _initialized = False

def clear_musicnn_status():
    """Clear the MusiCNN progress bar (for individual file failures)."""
    global musicnn_progress, musicnn_task
    if musicnn_progress and musicnn_task:
        # Just reset the progress bar for the next file
        musicnn_progress.reset(musicnn_task)
        musicnn_progress.update(musicnn_task, description="Waiting for next file...")

def get_musicnn_status():
    """Get the current MusiCNN status text."""
    global musicnn_status
    with status_lock:
        return musicnn_status 