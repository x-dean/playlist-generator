from tqdm import tqdm
import logging
from .feature_extractor import audio_analyzer

logger = logging.getLogger()

class SequentialProcessor:
    """Sequential processor for audio analysis (single-threaded)."""
    def __init__(self) -> None:
        self.failed_files: list[str] = []
    
    def process(self, file_list: list[str], workers: int = None, stop_event=None, force_reextract: bool = False) -> iter:
        """Process a list of files sequentially.

        Args:
            file_list (list[str]): List of file paths.
            workers (int, optional): Ignored for sequential processing.
            stop_event (multiprocessing.Event, optional): Event to signal graceful shutdown.
            force_reextract (bool, optional): If True, bypass the cache for all files.

        Yields:
            tuple: (features, filepath) for each file.
        """
        yield from self._process_sequential(file_list, stop_event=stop_event, force_reextract=force_reextract)

    def _process_sequential(self, file_list: list[str], stop_event=None, force_reextract: bool = False) -> iter:
        """Internal generator for sequential processing."""
        for filepath in file_list:
            if stop_event and stop_event.is_set():
                break
            import os
            try:
                from .feature_extractor import AudioAnalyzer
                audio_analyzer = AudioAnalyzer()
                features, _, _ = audio_analyzer.extract_features(filepath, force_reextract=force_reextract), filepath, True
                if features:
                    yield features, filepath
                else:
                    self.failed_files.append(filepath)
            except Exception as e:
                self.failed_files.append(filepath)
                logger.error(f"Error processing {filepath}: {str(e)}")