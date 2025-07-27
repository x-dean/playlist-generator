import logging
import sys
import threading
import queue
import time
import os
import signal
from colorlog import ColoredFormatter

log_queue = None
log_consumer_thread = None
log_level_monitor_thread = None


def change_log_level(new_level):
    """Change log level at runtime."""
    try:
        level = getattr(logging, new_level.upper(), logging.INFO)
        root_logger = logging.getLogger()
        root_logger.setLevel(level)
        
        # Also update all other loggers
        for name in logging.root.manager.loggerDict:
            logger = logging.getLogger(name)
            logger.setLevel(level)
        
        # Log the change
        root_logger.info(f"Log level changed to: {new_level.upper()}")
        return True
    except Exception as e:
        print(f"Failed to change log level: {e}")
        return False


def monitor_log_level_changes():
    """Background thread that monitors LOG_LEVEL environment variable changes."""
    last_level = os.getenv('LOG_LEVEL', 'INFO')
    
    while True:
        try:
            # Get current level from environment
            current_level = os.getenv('LOG_LEVEL', 'INFO')
            
            if current_level != last_level:
                print(f"\n🔄 Environment LOG_LEVEL changed from {last_level} to {current_level}")
                if change_log_level(current_level):
                    print(f"✅ Log level updated to: {current_level}")
                else:
                    print(f"❌ Failed to update log level to: {current_level}")
                last_level = current_level
            
            time.sleep(2)  # Check every 2 seconds
            
        except Exception as e:
            print(f"Error in log level monitor: {e}")
            time.sleep(5)  # Wait longer on error


def start_log_level_monitor():
    """Start the background thread that monitors LOG_LEVEL changes."""
    global log_level_monitor_thread
    
    if log_level_monitor_thread is None or not log_level_monitor_thread.is_alive():
        log_level_monitor_thread = threading.Thread(
            target=monitor_log_level_changes, 
            daemon=True,
            name="LogLevelMonitor"
        )
        log_level_monitor_thread.start()
        print("📝 Log level monitor started - export LOG_LEVEL inside Docker to change level on the fly")
        print("   Example: docker exec -it <container> export LOG_LEVEL=DEBUG")


def setup_log_level_signal_handler():
    """Setup signal handler to change log level at runtime."""
    def signal_handler(signum, frame):
        current_level = os.getenv('LOG_LEVEL', 'INFO')
        levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        
        try:
            current_index = levels.index(current_level)
            next_index = (current_index + 1) % len(levels)
            new_level = levels[next_index]
            
            if change_log_level(new_level):
                print(f"\n🔄 Log level cycled to: {new_level}")
            else:
                print(f"\n❌ Failed to change log level")
        except Exception as e:
            print(f"\n❌ Error changing log level: {e}")
    
    # Use SIGUSR1 to cycle through log levels
    signal.signal(signal.SIGUSR1, signal_handler)
    print("📝 Log level control: Send SIGUSR1 signal to cycle through log levels")


def setup_colored_file_logging(logfile_path=None):
    """Setup colored logging to write to files only."""
    # Ensure log directory exists
    if logfile_path:
        log_dir = os.path.dirname(logfile_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

    # Create a direct file handler instead of queue-based
    if logfile_path:
        file_handler = logging.FileHandler(
            logfile_path, mode='a', encoding='utf-8')
    else:
        # If no logfile_path provided, create a default one or use a null handler
        log_dir = os.getenv('LOG_DIR', '/app/logs')
        os.makedirs(log_dir, exist_ok=True)
        date_str = time.strftime('%Y%m%d')
        default_logfile = os.path.join(log_dir, f'playlista_{date_str}.log')
        file_handler = logging.FileHandler(
            default_logfile, mode='a', encoding='utf-8')

    # Use colored formatter for file logging (colors will be preserved in file)
    color_formatter = ColoredFormatter(
        "%(log_color)s%(asctime)s %(levelname)-8s%(reset)s %(message)s",
        datefmt="%H:%M:%S",
        reset=True,
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        },
        style='%'
    )
    file_handler.setFormatter(color_formatter)

    # Clear all existing handlers and add only file handler
    root_logger = logging.getLogger()
    root_logger.handlers = [file_handler]

    # Set the log level from environment variable
    log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
    root_logger.setLevel(getattr(logging, log_level, logging.INFO))

    # Also clear handlers for all other loggers and set their level
    for name in logging.root.manager.loggerDict:
        logger = logging.getLogger(name)
        logger.handlers = []
        logger.propagate = True  # Ensure messages propagate to root logger
        logger.setLevel(getattr(logging, log_level, logging.INFO))


def setup_file_only_logging(logfile_path=None):
    """Setup logging to write only to files, nothing to terminal."""
    # Ensure log directory exists
    if logfile_path:
        log_dir = os.path.dirname(logfile_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

    # Create a direct file handler instead of queue-based
    if logfile_path:
        file_handler = logging.FileHandler(
            logfile_path, mode='a', encoding='utf-8')
    else:
        # If no logfile_path provided, create a default one
        log_dir = os.getenv('LOG_DIR', '/app/logs')
        os.makedirs(log_dir, exist_ok=True)
        date_str = time.strftime('%Y%m%d')
        default_logfile = os.path.join(log_dir, f'playlista_{date_str}.log')
        file_handler = logging.FileHandler(
            default_logfile, mode='a', encoding='utf-8')

    # Use plain formatter for file logging (no colors)
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S"
    )
    file_handler.setFormatter(formatter)

    # Clear all existing handlers and add only the file handler
    root_logger = logging.getLogger()
    root_logger.handlers = [file_handler]

    # Also clear handlers for all other loggers
    for name in logging.root.manager.loggerDict:
        logger = logging.getLogger(name)
        logger.handlers = []
        logger.propagate = True  # Ensure messages propagate to root logger


def setup_colored_logging():
    """Setup colored logging for terminal output (legacy function)."""
    logger = logging.getLogger()
    if not logger.hasHandlers():
        formatter = ColoredFormatter(
            "%(log_color)s%(asctime)s %(levelname)-8s%(reset)s %(message)s",
            datefmt="%H:%M:%S",
            reset=True,
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            },
            style='%'
        )
        handler = logging.StreamHandler(sys.stderr)  # Use stderr
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def setup_queue_colored_logging(logfile_path=None):
    """Legacy function - now redirects to colored file logging."""
    setup_colored_file_logging(logfile_path)

# Do NOT call setup_colored_logging() here.
