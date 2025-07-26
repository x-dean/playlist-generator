# music_analyzer/file_discovery_example.py
"""
Example usage of the FileDiscovery module.

This demonstrates how to use the FileDiscovery class to manage file discovery,
tracking changes, and feeding files to analyzers.
"""

from .file_discovery import FileDiscovery
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_usage():
    """Example of how to use the FileDiscovery module."""

    # Initialize file discovery
    file_discovery = FileDiscovery(
        music_dir='/music',
        failed_dir='/music/failed_files',
        cache_dir='/app/cache'
    )

    # Discover all files
    all_files = file_discovery.discover_files()
    logger.info(f"Discovered {len(all_files)} files")

    # Get file changes (compared to previous state)
    added, removed, unchanged = file_discovery.get_file_changes()
    logger.info(
        f"Added: {len(added)}, Removed: {len(removed)}, Unchanged: {len(unchanged)}")

    # Example: Get files for normal analysis
    db_files = {'/music/song1.mp3', '/music/song2.mp3'}  # Files already in DB
    failed_db_files = {'/music/problematic.mp3'}  # Files marked as failed

    files_to_analyze = file_discovery.get_files_for_analysis(
        db_files=db_files,
        failed_db_files=failed_db_files,
        force=False,
        failed_mode=False
    )

    logger.info(f"Files to analyze: {len(files_to_analyze)}")

    # Example: Get files for failed mode
    failed_files_to_retry = file_discovery.get_files_for_analysis(
        db_files=db_files,
        failed_db_files=failed_db_files,
        force=False,
        failed_mode=True
    )

    logger.info(f"Failed files to retry: {len(failed_files_to_retry)}")

    # Example: Validate file paths before processing
    valid_files = file_discovery.validate_file_paths(files_to_analyze)
    logger.info(f"Valid files: {len(valid_files)}")

    # Update state after processing
    file_discovery.update_state()

    return files_to_analyze, failed_files_to_retry


def example_with_analyzers():
    """Example showing how to integrate with analyzers."""

    file_discovery = FileDiscovery()

    # Get files for analysis
    db_files = set()  # Would come from your database
    failed_db_files = set()  # Would come from your database

    files_to_analyze = file_discovery.get_files_for_analysis(
        db_files=db_files,
        failed_db_files=failed_db_files,
        force=False,
        failed_mode=False
    )

    # Feed files to analyzers
    from .parallel import ParallelProcessor
    from .sequential import SequentialProcessor

    # Choose processor based on file count
    if len(files_to_analyze) > 10:
        processor = ParallelProcessor()
        logger.info("Using parallel processor")
    else:
        processor = SequentialProcessor()
        logger.info("Using sequential processor")

    # Process files
    for features, filepath, success in processor.process(files_to_analyze):
        if success:
            logger.info(f"Successfully processed: {filepath}")
        else:
            logger.warning(f"Failed to process: {filepath}")

    # Update file discovery state
    file_discovery.update_state()


if __name__ == "__main__":
    example_usage()
    example_with_analyzers()
