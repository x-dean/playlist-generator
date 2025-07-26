from tqdm import tqdm
import logging
from typing import List
from .feature_extractor import audio_analyzer
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

    def __init__(self) -> None:
        self.failed_files: List[str] = []

    def process(self, file_list: List[str], workers: int = None, stop_event=None, force_reextract: bool = False) -> iter:
        """Process a list of files sequentially.

        Args:
            file_list (List[str]): List of file paths.
            workers (int, optional): Ignored for sequential processing.
            stop_event (multiprocessing.Event, optional): Event to signal graceful shutdown.
            force_reextract (bool, optional): If True, bypass the cache for all files.

        Yields:
            tuple: (features, filepath, db_write_success) for each file.
        """
        yield from self._process_sequential(file_list, stop_event=stop_event, force_reextract=force_reextract)

    def _process_sequential(self, file_list: List[str], stop_event=None, force_reextract: bool = False) -> iter:
        """Internal generator for sequential processing."""
        from utils.logging_setup import setup_queue_colored_logging
        setup_queue_colored_logging()
        import essentia
        essentia.log.infoActive = False
        essentia.log.warningActive = False
        for filepath in file_list:
            if stop_event and stop_event.is_set():
                break
            try:
                from .feature_extractor import AudioAnalyzer
                audio_analyzer = AudioAnalyzer()
                features, db_write_success, file_hash = audio_analyzer.extract_features(
                    filepath, force_reextract=force_reextract)
                if features and db_write_success:
                    yield features, filepath, db_write_success
                else:
                    self.failed_files.append(filepath)
                    yield None, filepath, False
            except Exception as e:
                self.failed_files.append(filepath)
                logger.error(f"Error processing {filepath}: {str(e)}")
                yield None, filepath, False
