"""
Shared MusiCNN status for simple console updates.
"""
import threading
import time
import sys

# Shared status variable
musicnn_status = "Waiting for MusiCNN extraction..."
status_lock = threading.Lock()

def update_musicnn_status(new_status):
    """Update the MusiCNN status and print it on the same line."""
    global musicnn_status
    with status_lock:
        musicnn_status = new_status
        # Print on the same line using carriage return
        print(f"\r🎵 MusiCNN: {new_status}", end='', flush=True)

def update_musicnn_file_status(filename):
    """Update status for a new file being processed."""
    update_musicnn_status(f"Processing: {filename}")

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

def clear_musicnn_status():
    """Clear the MusiCNN status line."""
    print("\r" + " " * 80 + "\r", end='', flush=True)

def get_musicnn_status():
    """Get the current MusiCNN status text."""
    global musicnn_status
    with status_lock:
        return musicnn_status 