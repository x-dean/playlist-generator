import gc
from tqdm import tqdm
import logging
from .audio_analyzer import audio_analyzer
from .parallel import process_file_worker  # Reuse worker function

logger = logging.getLogger(__name__)

class SequentialProcessor:
    def __init__(self):
        self.failed_files = []
    
    def process(self, file_list):
        return self._process_sequential(file_list)

    def _process_sequential(self, file_list):
        results = []
        with tqdm(file_list, desc="Analyzing files") as pbar:
            for filepath in pbar:
                try:
                    logger.info(f"Starting analysis for {filepath}")
                    if pbar.n % 10 == 0:
                        gc.collect()

                    features, _ = process_file_worker(filepath)
                    logger.info(f"Finished analysis for {filepath}")
                    if features:
                        results.append(features)
                    else:
                        self.failed_files.append(filepath)
                except Exception as e:
                    self.failed_files.append(filepath)
                    logger.error(f"Error processing {filepath}: {str(e)}")

                pbar.set_postfix_str(f"OK: {len(results)}, Failed: {len(self.failed_files)}")
        return results