# music_analyzer/file_discovery.py
import os
import logging
import hashlib
import json
from typing import List, Dict, Set, Tuple, Optional
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class FileDiscovery:
    """Handles file discovery, exclusion logic, and change tracking for audio analysis."""

    def __init__(self, music_dir: str = '/music', failed_dir: str = '/music/failed_files',
                 cache_dir: str = '/app/cache', audio_db=None):
        self.music_dir = music_dir
        self.failed_dir = failed_dir
        self.cache_dir = cache_dir
        self.valid_extensions = (
            '.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac', '.opus')

        # Use provided audio_db or create new one
        if audio_db is None:
            from .audio_analyzer import AudioAnalyzer
            self.audio_db = AudioAnalyzer()
        else:
            self.audio_db = audio_db

        self.current_files = set()

    def _get_file_hash(self, filepath: str) -> str:
        """Generate a hash for a file based on path and modification time."""
        try:
            stat = os.stat(filepath)
            # Use path + modification time for hash
            content = f"{filepath}:{stat.st_mtime}:{stat.st_size}"
            return hashlib.md5(content.encode()).hexdigest()
        except Exception:
            return hashlib.md5(filepath.encode()).hexdigest()

    def _is_valid_audio_file(self, filepath: str) -> bool:
        """Check if a file is a valid audio file."""
        if not os.path.isfile(filepath):
            return False

        # Check file size (skip very small files)
        try:
            if os.path.getsize(filepath) < 1024:  # Less than 1KB
                return False
        except Exception:
            return False

        # Check extension
        file_lower = filepath.lower()
        return file_lower.endswith(self.valid_extensions)

    def _is_in_excluded_directory(self, filepath: str) -> bool:
        """Check if a file is in an excluded directory."""
        abs_path = os.path.abspath(filepath)
        failed_dir_abs = os.path.abspath(self.failed_dir)

        # Check if file is in failed directory
        if abs_path.startswith(failed_dir_abs):
            logger.debug(f"Excluding file in failed directory: {filepath}")
            return True

        return False

    def discover_files(self) -> List[str]:
        """Discover all valid audio files, excluding failed directory."""
        logger.info(f"Discovering audio files in {self.music_dir}")
        logger.debug(f"Excluding directory: {self.failed_dir}")

        discovered_files = []

        try:
            for root, dirs, files in os.walk(self.music_dir):
                abs_root = os.path.abspath(root)

                # Skip failed directory
                if abs_root.startswith(os.path.abspath(self.failed_dir)):
                    logger.debug(f"Skipping directory: {abs_root}")
                    continue

                for file in files:
                    filepath = os.path.join(root, file)

                    # Additional check for files in failed directory
                    if self._is_in_excluded_directory(filepath):
                        continue

                    if self._is_valid_audio_file(filepath):
                        discovered_files.append(filepath)
                    else:
                        logger.debug(f"Skipping invalid file: {filepath}")

        except Exception as e:
            logger.error(f"Error during file discovery: {e}")

        self.current_files = set(discovered_files)
        logger.info(f"Discovered {len(discovered_files)} valid audio files")
        return discovered_files

    def get_file_changes(self) -> Tuple[List[str], List[str], List[str]]:
        """Get added, removed, and unchanged files from database."""
        return self.audio_db.get_file_discovery_changes()

    def get_files_for_analysis(self, db_files: Set[str], failed_db_files: Set[str],
                               force: bool = False, failed_mode: bool = False) -> List[str]:
        """Get files that need analysis based on current state and database."""
        if failed_mode:
            # In failed mode, only process files that are failed but not in failed directory
            files_to_analyze = []
            for filepath in failed_db_files:
                if not self._is_in_excluded_directory(filepath):
                    files_to_analyze.append(filepath)
            logger.info(f"Failed mode: {len(files_to_analyze)} files to retry")
            return files_to_analyze

        # Get all discovered files
        all_files = self.discover_files()

        if force:
            # Force mode: process all files except those in failed directory
            files_to_analyze = [
                f for f in all_files if f not in failed_db_files]
            logger.info(
                f"Force mode: {len(files_to_analyze)} files to process")
        else:
            # Normal mode: only process new files
            files_to_analyze = [
                f for f in all_files if f not in db_files and f not in failed_db_files]
            logger.info(
                f"Normal mode: {len(files_to_analyze)} new files to process")

        return files_to_analyze

    def update_state(self):
        """Update the file discovery state in database."""
        logger.debug(
            f"DISCOVERY: FileDiscovery.update_state() called with {len(self.current_files)} files")
        self.audio_db.update_file_discovery_state(list(self.current_files))
        logger.debug("DISCOVERY: File discovery state updated in database")

    def get_file_info(self, filepath: str) -> Dict[str, any]:
        """Get detailed information about a file."""
        try:
            stat = os.stat(filepath)
            return {
                'filepath': filepath,
                'filename': os.path.basename(filepath),
                'size': stat.st_size,
                'modified': stat.st_mtime,
                'hash': self._get_file_hash(filepath)
            }
        except Exception as e:
            logger.error(f"Error getting file info for {filepath}: {e}")
            return {
                'filepath': filepath,
                'filename': os.path.basename(filepath),
                'size': 0,
                'modified': 0,
                'hash': ''
            }

    def cleanup_removed_files(self, db_files: Set[str]) -> List[str]:
        """Find files in database that no longer exist on disk."""
        self.audio_db.cleanup_file_discovery_state()

        removed_from_disk = []
        for filepath in db_files:
            if not os.path.exists(filepath):
                removed_from_disk.append(filepath)

        if removed_from_disk:
            logger.info(
                f"Found {len(removed_from_disk)} files in database that no longer exist on disk")

        return removed_from_disk

    def validate_file_paths(self, filepaths: List[str]) -> List[str]:
        """Validate that all file paths are accessible and not in excluded directories."""
        valid_files = []

        for filepath in filepaths:
            if not os.path.exists(filepath):
                logger.warning(f"File does not exist: {filepath}")
                continue

            if self._is_in_excluded_directory(filepath):
                logger.warning(f"File is in excluded directory: {filepath}")
                continue

            if not self._is_valid_audio_file(filepath):
                logger.warning(f"File is not a valid audio file: {filepath}")
                continue

            valid_files.append(filepath)

        logger.info(
            f"Validated {len(valid_files)} out of {len(filepaths)} files")
        return valid_files
