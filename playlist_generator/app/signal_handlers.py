#!/usr/bin/env python3
"""
Signal handlers for the playlist generator application.
"""

import os
import sys
import signal
import logging
import threading
import psutil
from typing import Optional

logger = logging.getLogger(__name__)

# Global state for interrupt handling
_interrupt_requested = threading.Event()
_parallel_processing_active = threading.Event()


def signal_handler(signum, frame):
    """Handle interrupt signals gracefully."""
    import os
    
    # Check if this is a worker process (not the main process)
    if os.getpid() != 1:
        logger.debug(f"Child process {os.getpid()} received signal {signum}, exiting quietly")
        sys.exit(130)
    
    # Check if we're in parallel processing mode - if so, ignore the signal
    if _parallel_processing_active.is_set():
        logger.debug(f"Main process received signal {signum} during parallel processing, ignoring (likely from pool termination)")
        return  # Don't shutdown, just ignore the signal
    
    logger.warning(f"Main process received signal {signum}, initiating graceful shutdown...")
    print("\n🛑 Interrupt received! Initiating graceful shutdown...")
    
    # Set the interrupt flag
    _interrupt_requested.set()
    
    # Force cleanup of any running processes
    try:
        current_process = psutil.Process()
        children = current_process.children(recursive=True)
        if children:
            logger.info(f"Terminating {len(children)} child processes...")
            for child in children:
                try:
                    child.terminate()
                except:
                    pass
    except:
        pass
    
    sys.exit(130)


def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown."""
    # Register the signal handler FIRST
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def is_interrupt_requested() -> bool:
    """Check if an interrupt has been requested."""
    return _interrupt_requested.is_set()


def set_parallel_processing_active():
    """Mark that parallel processing is active."""
    _parallel_processing_active.set()


def clear_parallel_processing_active():
    """Mark that parallel processing is inactive."""
    _parallel_processing_active.clear()


def cleanup_child_processes():
    """Force cleanup of any running child processes."""
    try:
        current_process = psutil.Process()
        children = current_process.children(recursive=True)
        if children:
            logger.info(f"Terminating {len(children)} child processes...")
            for child in children:
                try:
                    child.terminate()
                except:
                    pass
    except Exception as e:
        logger.warning(f"Error during process cleanup: {e}")


def handle_keyboard_interrupt():
    """Handle keyboard interrupt gracefully."""
    logger.info("Application interrupted by user")
    print("\n🛑 Interrupted by user. Exiting...")
    
    cleanup_child_processes()
    sys.exit(130) 