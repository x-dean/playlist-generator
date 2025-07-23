# music_analyzer/parallel.py
import multiprocessing as mp
import os
import sys
from tqdm import tqdm
import logging
import time
from .audio_analyzer import AudioAnalyzer
import psutil

logger = logging.getLogger(__name__)

def process_file_worker(filepath):
    import os
    from .audio_analyzer import AudioAnalyzer
    audio_analyzer = AudioAnalyzer()
    max_retries = 2
    retry_count = 0
    backoff_time = 1  # Initial backoff time in seconds

    # Memory limit check (per worker)
    max_mem_mb = int(os.getenv('WORKER_MAX_MEM_MB', '2048'))
    process = psutil.Process(os.getpid())
    if process.memory_info().rss > max_mem_mb * 1024 * 1024:
        logger.warning(f"Worker memory exceeded {max_mem_mb}MB, skipping {filepath}")
        logger.info(f"SKIP: {filepath} (memory exceeded)")
        return None, filepath, False

    while retry_count <= max_retries:
        try:
            if not os.path.exists(filepath):
                logger.warning(f"File not found: {filepath}")
                logger.info(f"SKIP: {filepath} (not found)")
                return None, filepath, False
            
            if os.path.getsize(filepath) < 1024:
                logger.warning(f"Skipping small file: {filepath}")
                logger.info(f"SKIP: {filepath} (too small)")
                return None, filepath, False

            if not filepath.lower().endswith(('.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac')):
                logger.info(f"SKIP: {filepath} (unsupported extension)")
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
                logger.warning(f"Retrying {filepath} (attempt {retry_count}/{max_retries})")
                continue
            
            logger.info(f"FAIL: {filepath} (feature extraction failed)")
            return None, filepath, False

        except Exception as e:
            if retry_count < max_retries:
                retry_count += 1
                time.sleep(backoff_time)
                backoff_time *= 2
                logger.warning(f"Error processing {filepath}, retrying (attempt {retry_count}/{max_retries}): {str(e)}")
                continue
            
            logger.error(f"Error processing {filepath} after {max_retries} retries: {str(e)}", exc_info=True)
            logger.info(f"FAIL: {filepath} (exception: {str(e)})")
            return None, filepath, False

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
        import os
        max_memory_mb = int(os.getenv('MAX_MEMORY_MB', '8192'))
        worker_max_mem_mb = int(os.getenv('WORKER_MAX_MEM_MB', '2048'))
        pool = AdaptiveMemoryPool(max_memory_mb, worker_max_mem_mb)
        retries = 0
        remaining_files = file_list[:]

        while remaining_files and retries < self.max_retries:
            try:
                logger.info(f"Starting adaptive multiprocessing with memory cap {max_memory_mb}MB (retry {retries})")
                failed_in_batch = []
                for features, filepath, db_write_success in pool.run(remaining_files, process_file_worker):
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