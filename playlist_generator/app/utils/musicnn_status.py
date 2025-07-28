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
console = Console()

# Global progress bar for MusiCNN steps
musicnn_progress = None
musicnn_task = None

def init_musicnn_progress():
    """Initialize the MusiCNN progress bar."""
    global musicnn_progress, musicnn_task
    musicnn_progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]MusiCNN:"),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True
    )
    musicnn_task = musicnn_progress.add_task("Waiting...", total=5)
    musicnn_progress.start()

def update_musicnn_status(new_status):
    """Update the MusiCNN status in the progress bar."""
    global musicnn_progress, musicnn_task
    if musicnn_progress and musicnn_task:
        musicnn_progress.update(musicnn_task, description=new_status)

def update_musicnn_file_status(filename):
    """Update status for a new file being processed."""
    global musicnn_progress, musicnn_task
    if musicnn_progress and musicnn_task:
        # Reset progress for new file
        musicnn_progress.reset(musicnn_task)
        musicnn_progress.update(musicnn_task, description=f"Processing: {filename}")

def update_musicnn_step_status(step, **kwargs):
    """Update status for a specific step with dynamic values."""
    step_messages = {
        'start': "Starting...",
        'loaded_json': f"JSON: {kwargs.get('tag_count', 0)} tags",
        'loaded_audio': f"Audio: {kwargs.get('duration', 0):.1f}s",
        'activations': "Activations...",
        'embeddings': "Embeddings...",
        'success': f"✓ {kwargs.get('tag_count', 0)} tags, {kwargs.get('embedding_dims', 0)} dims | {kwargs.get('top_tags', '')}",
        'failure': f"✗ {kwargs.get('error', 'Unknown error')}"
    }
    
    if step in step_messages:
        update_musicnn_status(step_messages[step])
        # Advance progress for completed steps
        if step in ['loaded_json', 'loaded_audio', 'activations', 'embeddings']:
            global musicnn_progress, musicnn_task
            if musicnn_progress and musicnn_task:
                musicnn_progress.advance(musicnn_task)

def clear_musicnn_status():
    """Clear the MusiCNN progress bar."""
    global musicnn_progress
    if musicnn_progress:
        musicnn_progress.stop()

def get_musicnn_status():
    """Get the current MusiCNN status text."""
    global musicnn_status
    with status_lock:
        return musicnn_status 