from tqdm import tqdm
import logging
from .feature_extractor import audio_analyzer

logger = logging.getLogger(__name__)

class SequentialProcessor:
    """Sequential processor for audio analysis (single-threaded)."""
    def __init__(self) -> None:
        self.failed_files: list[str] = []
    
    def process(self, file_list: list[str], workers: int = None, stop_event=None) -> iter:
        """Process a list of files sequentially.

        Args:
            file_list (list[str]): List of file paths.
            workers (int, optional): Ignored for sequential processing.
            stop_event (multiprocessing.Event, optional): Event to signal graceful shutdown.

        Yields:
            tuple: (features, filepath) for each file.
        """
        yield from self._process_sequential(file_list, stop_event=stop_event)

    def _process_sequential(self, file_list: list[str], stop_event=None) -> iter:
        """Internal generator for sequential processing."""
        for filepath in file_list:
            if stop_event and stop_event.is_set():
                break
            import os
            try:
                from .parallel import process_file_worker
                features, _, _ = process_file_worker(filepath)
                if features:
                    yield features, filepath
                else:
                    self.failed_files.append(filepath)
            except Exception as e:
                self.failed_files.append(filepath)
                logger.error(f"Error processing {filepath}: {str(e)}")