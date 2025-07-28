#!/usr/bin/env python3
"""
Main entry point for the playlist generator application.
"""

import os
import sys
import multiprocessing as mp
import traceback
from typing import Optional

# Add the app directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import our new modular components
from cli_parser import create_parser, parse_early_args, validate_args, setup_environment_vars
from logging_setup import setup_early_logging, setup_tensorflow_logging, setup_application_logging, setup_environment_logging
from signal_handlers import setup_signal_handlers
from app_runner import run_application


def main() -> None:
    """Main entry point for the Playlista CLI application."""
    # Setup early logging and environment
    setup_environment_logging()
    log_file = setup_early_logging()
    
    # Parse early arguments for logging level
    pre_args = parse_early_args()
    os.environ['LOG_LEVEL'] = pre_args.log_level
    
    # Setup multiprocessing
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass
    
    # Setup signal handlers
    setup_signal_handlers()
    
    # Setup TensorFlow logging
    setup_tensorflow_logging(log_file)
    
    # Setup application logging
    logger = setup_application_logging(log_file)
    
    # Create and parse arguments
    parser = create_parser()
    args = parser.parse_args()
    
    # Validate arguments
    try:
        validate_args(args)
    except ValueError as e:
        logger.error(f"Invalid arguments: {e}")
        sys.exit(1)
    
    # Setup environment variables
    setup_environment_vars(args)
    
    # Log parsed arguments
    logger.info(
        f"Parsed arguments: analyze={args.analyze}, generate_only={args.generate_only}, update={args.update}, failed={args.failed}, force={args.force}, playlist_method={args.playlist_method}, workers={args.workers}, large_file_threshold={args.large_file_threshold}MB, memory_aware={args.memory_aware}")

    # Run the application
    run_application(args)


if __name__ == "__main__":
    main() 