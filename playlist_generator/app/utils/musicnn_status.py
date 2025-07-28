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

def get_musicnn_status():
    """Get the current MusiCNN status text."""
    global musicnn_status
    with status_lock:
        return musicnn_status

# Remove the run_musicnn_panel function since we're using simple prints 