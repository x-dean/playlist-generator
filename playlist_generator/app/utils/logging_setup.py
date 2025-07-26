import logging
import sys
import threading
import queue
import time
import os
from colorlog import ColoredFormatter

log_queue = None
log_consumer_thread = None

def setup_colored_file_logging(logfile_path=None):
    """Setup colored logging to write to files only, nothing to terminal."""
    global log_queue, log_consumer_thread
    if log_queue is not None:
        return  # Already set up in this process
    
    # Ensure log directory exists
    if logfile_path:
        log_dir = os.path.dirname(logfile_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
    
    log_queue = queue.Queue()

    class QueueHandler(logging.Handler):
        def emit(self, record):
            try:
                msg = self.format(record)
                log_queue.put(msg)
            except Exception:
                pass

    def log_consumer():
        log_file = None
        if logfile_path:
            try:
                log_file = open(logfile_path, 'a', encoding='utf-8')
            except Exception as e:
                # Fallback to stderr if file can't be opened
                sys.stderr.write(f"Failed to open log file {logfile_path}: {e}\n")
                return
        
        while True:
            try:
                msg = log_queue.get(timeout=0.5)
                if log_file:
                    log_file.write(msg + '\n')
                    log_file.flush()
                # Do not print to terminal - completely silent
                time.sleep(0.01)  # Reduced sleep for better performance
            except queue.Empty:
                continue
            except Exception as e:
                # Only write to stderr if there's a critical logging error
                sys.stderr.write(f"Logging error: {e}\n")
                continue

    queue_handler = QueueHandler()
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
    queue_handler.setFormatter(color_formatter)
    
    # Clear all existing handlers and add only the queue handler
    root_logger = logging.getLogger()
    root_logger.handlers = [queue_handler]
    
    # Also clear handlers for all other loggers
    for name in logging.root.manager.loggerDict:
        logger = logging.getLogger(name)
        logger.handlers = []
        logger.propagate = True  # Ensure messages propagate to root logger

    log_consumer_thread = threading.Thread(
        target=log_consumer, daemon=True)
    log_consumer_thread.start()


def setup_file_only_logging(logfile_path=None):
    """Setup logging to write only to files, nothing to terminal."""
    global log_queue, log_consumer_thread
    if log_queue is not None:
        return  # Already set up in this process
    
    # Ensure log directory exists
    if logfile_path:
        log_dir = os.path.dirname(logfile_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
    
    log_queue = queue.Queue()

    class QueueHandler(logging.Handler):
        def emit(self, record):
            try:
                msg = self.format(record)
                log_queue.put(msg)
            except Exception:
                pass

    def log_consumer():
        log_file = None
        if logfile_path:
            try:
                log_file = open(logfile_path, 'a', encoding='utf-8')
            except Exception as e:
                # Fallback to stderr if file can't be opened
                sys.stderr.write(f"Failed to open log file {logfile_path}: {e}\n")
                return
        
        while True:
            try:
                msg = log_queue.get(timeout=0.5)
                if log_file:
                    log_file.write(msg + '\n')
                    log_file.flush()
                # Do not print to terminal - completely silent
                time.sleep(0.01)  # Reduced sleep for better performance
            except queue.Empty:
                continue
            except Exception as e:
                # Only write to stderr if there's a critical logging error
                sys.stderr.write(f"Logging error: {e}\n")
                continue

    queue_handler = QueueHandler()
    # Use plain formatter for file logging (no colors)
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S"
    )
    queue_handler.setFormatter(formatter)
    
    # Clear all existing handlers and add only the queue handler
    root_logger = logging.getLogger()
    root_logger.handlers = [queue_handler]
    
    # Also clear handlers for all other loggers
    for name in logging.root.manager.loggerDict:
        logger = logging.getLogger(name)
        logger.handlers = []
        logger.propagate = True  # Ensure messages propagate to root logger

    log_consumer_thread = threading.Thread(
        target=log_consumer, daemon=True)
    log_consumer_thread.start()


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