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
    # Ensure log directory exists
    if logfile_path:
        log_dir = os.path.dirname(logfile_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
    
    # Create a direct file handler instead of queue-based
    if logfile_path:
        file_handler = logging.FileHandler(logfile_path, mode='a', encoding='utf-8')
    else:
        # If no logfile_path provided, create a default one or use a null handler
        log_dir = os.getenv('LOG_DIR', '/app/logs')
        os.makedirs(log_dir, exist_ok=True)
        date_str = time.strftime('%Y%m%d')
        default_logfile = os.path.join(log_dir, f'playlista_{date_str}.log')
        file_handler = logging.FileHandler(default_logfile, mode='a', encoding='utf-8')
    
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
    
    # Clear all existing handlers and add only the file handler
    root_logger = logging.getLogger()
    root_logger.handlers = [file_handler]
    
    # Also clear handlers for all other loggers
    for name in logging.root.manager.loggerDict:
        logger = logging.getLogger(name)
        logger.handlers = []
        logger.propagate = True  # Ensure messages propagate to root logger


def setup_file_only_logging(logfile_path=None):
    """Setup logging to write only to files, nothing to terminal."""
    # Ensure log directory exists
    if logfile_path:
        log_dir = os.path.dirname(logfile_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
    
    # Create a direct file handler instead of queue-based
    if logfile_path:
        file_handler = logging.FileHandler(logfile_path, mode='a', encoding='utf-8')
    else:
        # If no logfile_path provided, create a default one
        log_dir = os.getenv('LOG_DIR', '/app/logs')
        os.makedirs(log_dir, exist_ok=True)
        date_str = time.strftime('%Y%m%d')
        default_logfile = os.path.join(log_dir, f'playlista_{date_str}.log')
        file_handler = logging.FileHandler(default_logfile, mode='a', encoding='utf-8')
    
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