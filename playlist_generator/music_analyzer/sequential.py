from tqdm import tqdm
import logging
from .audio_analyzer import audio_analyzer
from .parallel import process_file_worker  # Reuse worker function

logger = logging.getLogger(__name__)

class SequentialProcessor:
    """Sequential processor for audio analysis (single-threaded)."""
    def __init__(self) -> None:
        self.failed_files: list[str] = []
    
    def process(self, file_list: list[str], workers: int = None) -> iter:
        """Process a list of files sequentially.

        Args:
            file_list (list[str]): List of file paths.
            workers (int, optional): Ignored for sequential processing.

        Yields:
            dict: Extracted features for each file.
        """
        yield from self._process_sequential(file_list)

    def _process_sequential(self, file_list: list[str]) -> iter:
        """Internal generator for sequential processing."""
        for filepath in file_list:
            import os
            filename = os.path.basename(filepath)
            try:
                size_mb = os.path.getsize(filepath) / (1024 * 1024)
            except Exception:
                size_mb = 0
            print(f"Processing: {filename} ({size_mb:.1f} MB)")
            try:
                features, _, _ = process_file_worker(filepath)
                if features:
                    yield features
                else:
                    self.failed_files.append(filepath)
            except Exception as e:
                self.failed_files.append(filepath)
                logger.error(f"Error processing {filepath}: {str(e)}")