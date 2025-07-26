from tqdm import tqdm
import logging
from typing import List
from utils.logging_setup import setup_colored_logging
import os
import logging
import time
import traceback
from typing import List, Dict, Any, Optional
from .feature_extractor import AudioAnalyzer
from utils.path_converter import PathConverter

logger = logging.getLogger(__name__)


class SequentialProcessor:
    """Sequential processor for audio analysis (single-threaded)."""

    def __init__(self, audio_analyzer: AudioAnalyzer = None) -> None:
        self.failed_files: List[str] = []
        self.audio_analyzer = audio_analyzer

    def process(self, file_list: List[str], workers: int = None, force_reextract: bool = False) -> iter:
        """Process a list of files sequentially.

        Args:
            file_list (List[str]): List of file paths.
            workers (int, optional): Ignored for sequential processing.
            force_reextract (bool, optional): If True, bypass the cache for all files.

        Yields:
            tuple: (features, filepath, db_write_success) for each file.
        """
        yield from self._process_sequential(file_list, force_reextract=force_reextract)

    def _process_sequential(self, file_list: List[str], force_reextract: bool = False) -> iter:
        """Internal generator for sequential processing."""
        from utils.logging_setup import setup_queue_colored_logging
        setup_queue_colored_logging()
        import essentia
        essentia.log.infoActive = False
        essentia.log.warningActive = False

        for i, filepath in enumerate(file_list):
            try:
                logger.debug(f"SEQUENTIAL: Processing file {i+1}/{len(file_list)}: {filepath}")
                
                # Use file discovery to check if file should be excluded
                from .file_discovery import FileDiscovery
                file_discovery = FileDiscovery()
                if file_discovery._is_in_excluded_directory(filepath):
                    logger.warning(
                        f"Skipping file in excluded directory: {filepath}")
                    yield None, filepath, False
                    continue

                # Use the provided audio_analyzer or create a new one
                logger.debug(f"SEQUENTIAL: Calling extract_features for: {filepath}")
                if self.audio_analyzer:
                    analyzer = self.audio_analyzer
                else:
                    analyzer = AudioAnalyzer()
                features, db_write_success, file_hash = analyzer.extract_features(
                    filepath, force_reextract=force_reextract)
                logger.debug(f"SEQUENTIAL: extract_features result - features: {features is not None}, db_write: {db_write_success}")
                
                if features and db_write_success:
                    logger.debug(f"SEQUENTIAL: Success for {filepath}")
                    yield features, filepath, db_write_success
                else:
                    logger.warning(f"SEQUENTIAL: Failed for {filepath} - features: {features is not None}, db_write: {db_write_success}")
                    self.failed_files.append(filepath)
                    yield None, filepath, False

            except Exception as e:
                self.failed_files.append(filepath)
                logger.error(f"Error processing {filepath}: {str(e)}")
                yield None, filepath, False
