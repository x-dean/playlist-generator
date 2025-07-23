# music_analyzer/parallel.py
import multiprocessing as mp
import os
import sys
from tqdm import tqdm
import logging
import time
from .audio_analyzer import AudioAnalyzer
from typing import Optional

logger = logging.getLogger(__name__)

def process_file_worker(filepath: str) -> Optional[tuple]:
    """Worker function to process a single audio file in parallel.

    Args:
        filepath (str): Path to the audio file.

    Returns:
        tuple | None: (features dict, filepath, db_write_success bool) or None on failure.
    """
    import os
    from .audio_analyzer import AudioAnalyzer
    audio_analyzer = AudioAnalyzer()
    max_retries = 2
    retry_count = 0
    backoff_time = 1  # Initial backoff time in seconds

    # Removed memory limit logic

    while retry_count <= max_retries:
        try:
            if not os.path.exists(filepath):
                logger.warning(f"File not found: {filepath}")
                return None, filepath, False
            
            if os.path.getsize(filepath) < 1024:
                logger.warning(f"Skipping small file: {filepath}")
                return None, filepath, False

            if not filepath.lower().endswith(('.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac')):
                logger.warning(f"Unsupported extension, skipping: {filepath}")
                return None, filepath, False

            result = audio_analyzer.extract_features(filepath)

            if result and result[0] is not None:
                features, db_write_success, _ = result
                for key in ['bpm', 'centroid', 'duration']:
                    if features.get(key) is None:
                        features[key] = 0.0
                logger.info(f"PROCESSED: {filepath}")
                return features, filepath, db_write_success
            
            # If we get here, result was None or features were None
            if retry_count < max_retries:
                retry_count += 1
                time.sleep(backoff_time)
                backoff_time *= 2  # Exponential backoff
                logger.debug(f"Retrying {filepath} (attempt {retry_count}/{max_retries})")
                continue
            
            logger.warning(f"Feature extraction failed for {filepath}")
            return None, filepath, False

        except Exception as e:
            if retry_count < max_retries:
                retry_count += 1
                time.sleep(backoff_time)
                backoff_time *= 2
                logger.debug(f"Error processing {filepath}, retrying (attempt {retry_count}/{max_retries}): {str(e)}")
                continue
            
            logger.error(f"Error processing {filepath} after {max_retries} retries: {str(e)}")
            logger.warning(f"FAIL: {filepath} (exception: {str(e)})")
            return None, filepath, False

class ParallelProcessor:
    """Parallel processor for batch audio analysis using multiprocessing."""
    def __init__(self) -> None:
        self.failed_files = []
        self.batch_size = int(os.getenv('BATCH_SIZE', '50'))
        self.max_retries = int(os.getenv('MAX_RETRIES', '3'))
        self.min_workers = 2
        self.max_workers = int(os.getenv('MAX_WORKERS', str(mp.cpu_count())))

    def process(self, file_list: list[str], workers: int = None) -> iter:
        """Process a list of files in parallel.

        Args:
            file_list (list[str]): List of file paths.
            workers (int, optional): Number of worker processes. Defaults to None.

        Yields:
            dict: Extracted features for each file.
        """
        if not file_list:
            return
        self.workers = max(self.min_workers, min(workers or self.max_workers, self.max_workers))
        self.batch_size = min(self.batch_size, len(file_list))
        yield from self._process_parallel(file_list)

    def _process_parallel(self, file_list):
        retries = 0
        remaining_files = file_list[:]

        while remaining_files and retries < self.max_retries:
            try:
                logger.info(f"Starting multiprocessing with {self.workers} workers (retry {retries})")
                ctx = mp.get_context('spawn')

                failed_in_batch = []
                for i in range(0, len(remaining_files), self.batch_size):
                    batch = remaining_files[i:i+self.batch_size]

                    with ctx.Pool(processes=self.workers) as pool:
                        for features, filepath, db_write_success in pool.imap_unordered(process_file_worker, batch):
                            if features and db_write_success:
                                yield features
                            else:
                                failed_in_batch.append(filepath)
                # Instead of removing processed files, retry only failed ones
                if failed_in_batch:
                    logger.info(f"Retrying {len(failed_in_batch)} failed files in next round")
                    remaining_files = failed_in_batch
                    self.failed_files.extend(failed_in_batch)
                else:
                    logger.info(f"Processing completed - yielded all successful, {len(self.failed_files)} failed")
                    return

            except (mp.TimeoutError, BrokenPipeError, ConnectionResetError) as e:
                logger.error(f"Multiprocessing error: {str(e)}")
                retries += 1
                if retries < self.max_retries:
                    # Reduce workers on retry to handle potential resource constraints
                    self.workers = max(self.min_workers, self.workers // 2)
                    logger.warning(f"Reducing workers to {self.workers} and retrying with {len(remaining_files)} remaining files")
                    time.sleep(2 ** retries)  # Exponential backoff
                else:
                    logger.error("Max retries reached, switching to sequential for remaining files")
                    # Process remaining files sequentially
                    for filepath in remaining_files:
                        features, _, db_write_success = process_file_worker(filepath)
                        if features and db_write_success:
                            yield features
                        else:
                            self.failed_files.append(filepath)
                    break

        return