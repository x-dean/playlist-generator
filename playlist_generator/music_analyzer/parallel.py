# music_analyzer/parallel.py
import multiprocessing as mp
import os
import sys
from tqdm import tqdm
import logging
import time
from .audio_analyzer import AudioAnalyzer
import psutil
import threading

logger = logging.getLogger(__name__)

# Remove worker_wrapper and all memory limit logic

def process_file_worker(filepath):
    import time
    print(f"[DEBUG] Minimal worker for {filepath}")
    time.sleep(2)
    # Return a dict with 'metadata' key for compatibility with main loop
    return {"filepath": filepath, "metadata": {"dummy": True}}, filepath, True

class ParallelProcessor:
    def __init__(self):
        self.failed_files = []
        self.batch_size = int(os.getenv('BATCH_SIZE', '50'))
        self.max_retries = int(os.getenv('MAX_RETRIES', '3'))
        self.min_workers = 2  # Ensure at least 2 workers
        self.max_workers = int(os.getenv('MAX_WORKERS', str(mp.cpu_count())))

    def process(self, file_list, workers=None):
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
                logger.info(f"Starting multiprocessing (retry {retries})")
                failed_in_batch = []
                with mp.Pool(processes=self.workers) as pool:
                    for features, filepath, db_write_success in pool.imap_unordered(process_file_worker, remaining_files):
                        if features and db_write_success:
                            yield features
                        else:
                            failed_in_batch.append(filepath)
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
                    logger.warning(f"Retrying with {len(remaining_files)} remaining files")
                    time.sleep(2 ** retries)
                else:
                    logger.error("Max retries reached, switching to sequential for remaining files")
                    for filepath in remaining_files:
                        features, _, db_write_success = process_file_worker(filepath)
                        if features and db_write_success:
                            yield features
                        else:
                            self.failed_files.append(filepath)
                    break
        return