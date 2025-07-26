from tqdm import tqdm
import logging
from typing import List
from .feature_extractor import audio_analyzer
from utils.logging_setup import setup_colored_logging
import os
setup_colored_logging()
log_level = os.getenv('LOG_LEVEL', 'INFO')
import logging
logging.getLogger().setLevel(getattr(logging, log_level.upper(), logging.INFO))

logger = logging.getLogger()

class SequentialProcessor:
    """Sequential processor for audio analysis (single-threaded)."""
    def __init__(self) -> None:
        self.failed_files: List[str] = []
        logger.debug("Initialized SequentialProcessor")
    
    def process(self, file_list: List[str], workers: int = None, stop_event=None, force_reextract: bool = False) -> iter:
        """Process a list of files sequentially.

        Args:
            file_list (List[str]): List of file paths.
            workers (int, optional): Ignored for sequential processing.
            stop_event (multiprocessing.Event, optional): Event to signal graceful shutdown.
            force_reextract (bool, optional): If True, bypass the cache for all files.

        Yields:
            tuple: (features, filepath) for each file.
        """
        logger.info(f"Starting sequential processing of {len(file_list)} files")
        logger.debug(f"Sequential processing parameters: force_reextract={force_reextract}")
        
        yield from self._process_sequential(file_list, stop_event=stop_event, force_reextract=force_reextract)

    def _process_sequential(self, file_list: List[str], stop_event=None, force_reextract: bool = False) -> iter:
        """Internal generator for sequential processing."""
        logger.debug("Starting sequential processing loop")
        from utils.logging_setup import setup_queue_colored_logging
        setup_queue_colored_logging()
        import essentia
        essentia.log.infoActive = False
        essentia.log.warningActive = False
        
        processed_count = 0
        total_files = len(file_list)
        
        for filepath in file_list:
            processed_count += 1
            logger.debug(f"Processing file {processed_count}/{total_files}: {filepath}")
            
            if stop_event and stop_event.is_set():
                logger.info("Stop event received, stopping sequential processing")
                break
            try:
                from .feature_extractor import AudioAnalyzer
                audio_analyzer = AudioAnalyzer()
                logger.debug(f"Starting feature extraction for: {filepath}")
                features, db_write_success, file_hash = audio_analyzer.extract_features(filepath, force_reextract=force_reextract)
                
                if features:
                    logger.info(f"Successfully processed: {filepath}")
                    yield features, filepath
                else:
                    logger.warning(f"Feature extraction failed for: {filepath}")
                    self.failed_files.append(filepath)
            except Exception as e:
                logger.error(f"Error processing {filepath}: {str(e)}")
                import traceback
                logger.error(f"Processing error traceback: {traceback.format_exc()}")
                self.failed_files.append(filepath)
        
        logger.info(f"Sequential processing complete: {processed_count} files processed, {len(self.failed_files)} failed")