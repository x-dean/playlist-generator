import gc
from tqdm import tqdm
import logging
from .audio_analyzer import audio_analyzer
from .parallel import process_file_worker  # Reuse worker function

logger = logging.getLogger(__name__)

class SequentialProcessor:
    def __init__(self):
        self.failed_files = []
    
    def process(self, file_list, workers=None):
        yield from self._process_sequential(file_list)

    def _process_sequential(self, file_list):
        for filepath in file_list:
            try:
                features, _ = process_file_worker(filepath)
                if features:
                    yield features
                else:
                    self.failed_files.append(filepath)
            except Exception as e:
                self.failed_files.append(filepath)
                logger.error(f"Error processing {filepath}: {str(e)}")