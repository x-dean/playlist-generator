"""
Shared MusiCNN status for live console updates.
"""
from rich.text import Text
import threading
import time

# Shared status variable
musicnn_status = Text("Waiting for MusiCNN extraction...")
status_lock = threading.Lock()

def update_musicnn_status(new_status):
    """Update the MusiCNN status text."""
    global musicnn_status
    with status_lock:
        musicnn_status.text = new_status

def get_musicnn_status():
    """Get the current MusiCNN status text."""
    global musicnn_status
    with status_lock:
        return musicnn_status.text

def run_musicnn_panel():
    """Run the MusiCNN status panel in a separate thread."""
    from rich.live import Live
    from rich.panel import Panel
    
    with Live(Panel(musicnn_status, title="MusiCNN Status"), refresh_per_second=4):
        while True:
            # Keep the panel alive and updating
            time.sleep(0.1) 