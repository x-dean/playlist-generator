"""
Shared MusiCNN status for simple console updates.
"""
import threading
import time
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

# MusiCNN step definitions
MUSICNN_STEPS = [
    "Loading JSON metadata",
    "Loading audio file", 
    "Running activations",
    "Running embeddings",
    "Finalizing results"
]

_initialized = False

def init_musicnn_progress():
    """Initialize the MusiCNN status display."""
    global _initialized
    
    if _initialized:
        return  # Already initialized
    
    # Print a separator line to create space
    console.print("\n" + "="*80)
    console.print("[bold blue]MusiCNN Processing Status:[/bold blue]")
    console.print("="*80)
    _initialized = True

def update_musicnn_status(new_status):
    """Update the MusiCNN status and print it on a single line."""
    global musicnn_status, _initialized
    
    # Initialize if not already done
    if not _initialized:
        init_musicnn_progress()
    
    with status_lock:
        musicnn_status = new_status
        # Print on the same line using carriage return
        console.print(f"\r🎵 MusiCNN: {new_status}", end='', flush=True)

def update_musicnn_file_status(filename):
    """Update status for a new file being processed."""
    update_musicnn_status(f"Processing: {filename}")

def update_musicnn_step_progress(step_index, description=None):
    """Update status for a specific MusiCNN step."""
    global _initialized
    
    # Initialize if not already done
    if not _initialized:
        init_musicnn_progress()
    
    if step_index < len(MUSICNN_STEPS):
        step_name = MUSICNN_STEPS[step_index]
        if description:
            step_name = f"{step_name}: {description}"
        update_musicnn_status(step_name)

def finish_musicnn_processing():
    """Call this when all MusiCNN processing is complete."""
    global _initialized
    console.print("\n[bold green]✓ MusiCNN processing completed[/bold green]\n")
    _initialized = False

def clear_musicnn_status():
    """Clear the MusiCNN status line."""
    console.print("\r" + " " * 80 + "\r", end='', flush=True)

def get_musicnn_status():
    """Get the current MusiCNN status text."""
    global musicnn_status
    with status_lock:
        return musicnn_status 