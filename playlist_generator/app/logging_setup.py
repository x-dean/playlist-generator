#!/usr/bin/env python3
"""
Logging setup for the playlist generator application.
"""

import os
import sys
import logging
from datetime import datetime
from utils.logging_setup import setup_colored_file_logging, setup_log_level_signal_handler_direct, start_log_level_monitor


def setup_early_logging():
    """Setup early logging before imports."""
    # Create log file path and redirect stderr immediately
    log_dir = os.getenv('LOG_DIR', '/app/logs')
    os.makedirs(log_dir, exist_ok=True)
    date_str = datetime.now().strftime('%Y%m%d')
    log_file = os.path.join(log_dir, f'playlista_{date_str}.log')

    # Redirect stderr to log file to capture all early TensorFlow logs
    stderr_log = open(log_file, 'a')
    os.dup2(stderr_log.fileno(), 2)  # Redirect fd 2 (stderr) at the OS level
    sys.stderr = stderr_log  # Also update Python's sys.stderr
    
    return log_file


def setup_tensorflow_logging(log_file: str):
    """Setup TensorFlow logging to redirect to log file."""
    # Setup TensorFlow and Essentia logging AFTER log file is created
    import tensorflow as tf

    # Redirect TensorFlow Python output to log file
    tf_logger = tf.get_logger()
    tf_logger.handlers = []  # Remove default handlers
    tf_logger.addHandler(logging.FileHandler(log_file))

    # Redirect Essentia logs to log file
    import essentia
    essentia.log.infoActive = True
    essentia.log.warningActive = True
    essentia.log.errorActive = True

    essentia_logger = logging.getLogger('essentia')
    essentia_logger.handlers = []  # Remove default handlers
    essentia_logger.addHandler(logging.FileHandler(log_file))

    # Make both TensorFlow and Essentia respect the LOG_LEVEL environment variable
    log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
    tf_logger.setLevel(getattr(logging, log_level, logging.INFO))
    essentia_logger.setLevel(getattr(logging, log_level, logging.INFO))


def setup_application_logging(log_file: str):
    """Setup application logging."""
    # Setup colored logging to file only (log file already created above)
    setup_colored_file_logging(log_file)

    # Setup runtime log level control AFTER interrupt handlers
    setup_log_level_signal_handler_direct()
    start_log_level_monitor()

    logger = logging.getLogger(__name__)

    # Add session separator to distinguish between runs
    logger.info("=" * 80)
    logger.info(
        f"PLAYLISTA SESSION STARTED - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)

    # Set other loggers to appropriate levels
    import logging as pylogging
    pylogging.getLogger("musicbrainzngs").setLevel(pylogging.WARNING)

    return logger


def setup_environment_logging():
    """Setup environment-specific logging configuration."""
    # Control TensorFlow C++ backend logging (must be set before import)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Hide INFO and WARNING, show only ERROR
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU to avoid GPU-related warnings
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimization messages

    # Set Essentia logging environment variables
    os.environ["ESSENTIA_LOGGING_LEVEL"] = "error"
    os.environ["ESSENTIA_STREAM_LOGGING"] = "none" 