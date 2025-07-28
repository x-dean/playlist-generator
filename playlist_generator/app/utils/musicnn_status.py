"""
Shared MusiCNN status for simple console updates.
"""
import threading
import time

# Shared status variable
musicnn_status = "Waiting for MusiCNN extraction..."
status_lock = threading.Lock()

def update_musicnn_status(new_status):
    """Update the MusiCNN status and print it."""
    global musicnn_status
    with status_lock:
        musicnn_status = new_status
        # Print the status with a simple format that won't interfere
        print(f"\n🎵 MusiCNN: {new_status}")

def update_musicnn_file_status(filename):
    """Update status for a new file being processed."""
    update_musicnn_status(f"Processing: {filename}")

def update_musicnn_step_status(step, **kwargs):
    """Update status for a specific step with dynamic values."""
    step_messages = {
        'start': "Starting MusiCNN extraction...",
        'loaded_json': f"Loaded {kwargs.get('tag_count', 0)} tags from JSON",
        'loaded_audio': f"Loaded {kwargs.get('duration', 0):.1f}s audio at 16kHz",
        'activations': "Running MusiCNN activations...",
        'embeddings': "Running MusiCNN embeddings...",
        'success': f"Done: {kwargs.get('tag_count', 0)} tags, {kwargs.get('embedding_dims', 0)} dims | Top: {kwargs.get('top_tags', '')}",
        'failure': f"Failed: {kwargs.get('error', 'Unknown error')}"
    }
    
    if step in step_messages:
        update_musicnn_status(step_messages[step])

def get_musicnn_status():
    """Get the current MusiCNN status text."""
    global musicnn_status
    with status_lock:
        return musicnn_status 