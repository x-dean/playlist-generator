from tqdm import tqdm
import logging
from typing import List
import os
import time
import traceback
from typing import List, Dict, Any, Optional
from .feature_extractor import AudioAnalyzer
from utils.path_converter import PathConverter
import multiprocessing as mp
import signal
import psutil
import threading
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FutureTimeoutError

logger = logging.getLogger(__name__)


class LargeFileProcessor:
    """Handles large files in a separate process with timeout and progress monitoring."""
    
    def __init__(self, timeout_seconds: int = 600):
        self.timeout_seconds = timeout_seconds
        self._stop_processing = threading.Event()
    
    def _extract_features_in_process(self, audio_path: str, force_reextract: bool = False) -> tuple:
        """Extract features in a separate process to avoid blocking the main process."""
        try:
            # Create a new AudioAnalyzer instance in the subprocess
            analyzer = AudioAnalyzer()
            result = analyzer.extract_features(audio_path, force_reextract=force_reextract)
            return result
        except Exception as e:
            logger.error(f"Process extraction failed for {audio_path}: {str(e)}")
            return None, False, None
    
    def process_large_file(self, audio_path: str, force_reextract: bool = False) -> tuple:
        """Process a large file in a separate process with timeout."""
        file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
        logger.info(f"Processing large file ({file_size_mb:.1f}MB) in separate process: {os.path.basename(audio_path)}")
        
        # Adjust timeout based on file size
        if file_size_mb > 200:
            timeout = 900  # 15 minutes for very large files
        elif file_size_mb > 100:
            timeout = 600  # 10 minutes for large files
        else:
            timeout = 300  # 5 minutes for medium files
        
        try:
            with ProcessPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self._extract_features_in_process, audio_path, force_reextract)
                result = future.result(timeout=timeout)
                logger.info(f"Large file processing completed: {os.path.basename(audio_path)}")
                return result
        except FutureTimeoutError:
            logger.error(f"Large file processing timed out after {timeout}s: {os.path.basename(audio_path)}")
            # Kill any remaining processes
            for proc in psutil.process_iter(['pid', 'name']):
                try:
                    if 'python' in proc.info['name'].lower():
                        proc.terminate()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            return None, False, None
        except Exception as e:
            logger.error(f"Large file processing failed: {os.path.basename(audio_path)} - {str(e)}")
            return None, False, None


class SequentialProcessor:
    """Sequential processor for audio analysis (single-threaded)."""

    def __init__(self, audio_analyzer: AudioAnalyzer = None) -> None:
        self.failed_files: List[str] = []
        self.audio_analyzer = audio_analyzer
        self.large_file_processor = LargeFileProcessor()

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
        import essentia
        # Essentia logging is now handled in main playlista script

        for i, filepath in enumerate(file_list):
            try:
                logger.debug(
                    f"SEQUENTIAL: Processing file {i+1}/{len(file_list)}: {filepath}")

                # Use file discovery to check if file should be excluded
                from .file_discovery import FileDiscovery
                file_discovery = FileDiscovery()
                if file_discovery._is_in_excluded_directory(filepath):
                    logger.warning(
                        f"Skipping file in excluded directory: {filepath}")
                    yield None, filepath, False
                    continue

                # Check if this is a large file that should be processed separately
                try:
                    file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
                    # Get threshold from environment or use default
                    threshold_mb = int(os.getenv('LARGE_FILE_THRESHOLD', '50'))
                    is_large_file = file_size_mb > threshold_mb
                except:
                    is_large_file = False
                
                if is_large_file:
                    logger.info(f"Large file detected ({file_size_mb:.1f}MB), using separate process: {os.path.basename(filepath)}")
                    features, db_write_success, file_hash = self.large_file_processor.process_large_file(
                        filepath, force_reextract=force_reextract)
                else:
                    # Use the provided audio_analyzer or create a new one
                    logger.debug(
                        f"SEQUENTIAL: Calling extract_features for: {filepath}")
                    if self.audio_analyzer:
                        analyzer = self.audio_analyzer
                    else:
                        analyzer = AudioAnalyzer()
                    
                    # Log that we're starting the extraction
                    logger.info(f"SEQUENTIAL: Starting feature extraction for {os.path.basename(filepath)}")
                    
                    features, db_write_success, file_hash = analyzer.extract_features(
                        filepath, force_reextract=force_reextract)
                
                logger.debug(
                    f"SEQUENTIAL: extract_features result - features: {features is not None}, db_write: {db_write_success}")

                if features and db_write_success:
                    logger.debug(f"SEQUENTIAL: Success for {filepath}")
                    yield features, filepath, db_write_success
                else:
                    logger.warning(
                        f"SEQUENTIAL: Failed for {filepath} - features: {features is not None}, db_write: {db_write_success}")
                    self.failed_files.append(filepath)
                    yield None, filepath, False

            except Exception as e:
                self.failed_files.append(filepath)
                logger.error(f"Error processing {filepath}: {str(e)}")
                yield None, filepath, False
