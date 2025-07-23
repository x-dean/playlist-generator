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

# Top-level worker wrapper for multiprocessing (must be picklable)
def worker_wrapper(worker_func, file_path, conn, mem_limit_mb):
    print(f"[DEBUG] worker_wrapper started for {file_path}")
    import sys; sys.stdout.flush()
    result = worker_func(file_path, mem_limit_mb)
    conn.send(result)
    conn.close()

def process_file_worker(filepath, dynamic_mem_limit_mb=None):
    import os
    from .audio_analyzer import AudioAnalyzer
    print(f"[DEBUG] Worker started for {filepath}")
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

class AdaptiveMemoryPool:
    """
    Adaptive pool that launches worker processes only if enough memory is available.
    """
    def __init__(self, max_memory_mb, worker_max_mem_mb):
        self.max_memory_mb = max_memory_mb
        self.worker_max_mem_mb = worker_max_mem_mb
        self.active_workers = []
        self.lock = threading.Lock()

    def estimate_memory_for_file(self, file_path):
        try:
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            ext = os.path.splitext(file_path)[1].lower()
            # Use a higher multiplier for compressed formats
            if ext in ('.mp3', '.m4a', '.aac', '.ogg', '.opus'):
                return max(2048, size_mb * 15)  # 15x file size, min 2GB
            else:  # wav, flac, etc.
                return max(1024, size_mb * 2)   # 2x file size, min 1GB
        except Exception:
            return self.worker_max_mem_mb

    def can_launch_worker(self, file_path):
        import psutil
        current_memory = psutil.virtual_memory().used / (1024 * 1024)
        est_mem = self.estimate_memory_for_file(file_path)
        # Only launch if total memory usage + estimated for next file is below cap
        return (current_memory + est_mem) <= self.max_memory_mb

    def run(self, file_list, worker_func):
        import time
        import psutil
        results = []
        file_iter = iter(file_list)
        active_workers = []  # List of (process, est_mem)
        while True:
            # Clean up finished workers and update running memory
            for proc, mem in active_workers[:]:
                if not proc.is_alive():
                    proc.join()
                    active_workers.remove((proc, mem))
            # Calculate current total estimated memory
            total_est_mem = sum(mem for _, mem in active_workers)
            try:
                file_path = next(file_iter)
            except StopIteration:
                break
            est_mem = self.estimate_memory_for_file(file_path)
            print(f"[DEBUG] Considering file: {file_path}")
            print(f"[DEBUG] Estimated memory for file: {est_mem} MB")
            print(f"[DEBUG] Total estimated running memory: {total_est_mem} MB, Max allowed: {self.max_memory_mb} MB")
            while total_est_mem + est_mem > self.max_memory_mb:
                print(f"[DEBUG] Not enough memory to launch worker for {file_path}. Waiting...")
                time.sleep(1)
                for proc, mem in active_workers[:]:
                    if not proc.is_alive():
                        proc.join()
                        active_workers.remove((proc, mem))
                total_est_mem = sum(mem for _, mem in active_workers)
                print(f"[DEBUG] (Wait loop) Total estimated running memory: {total_est_mem} MB")
            print(f"[DEBUG] Launching worker for {file_path} with memory limit {est_mem} MB")
            # Launch worker with dynamic memory limit
            pconn, cconn = mp.Pipe()
            proc = mp.Process(target=worker_wrapper, args=(worker_func, file_path, cconn, est_mem))
            proc.start()
            active_workers.append((proc, est_mem))
            results.append(pconn)
        # Wait for all workers to finish
        for proc, _ in active_workers:
            proc.join()
        # Collect results
        for conn in results:
            try:
                yield conn.recv()
            except EOFError:
                yield (None, None, False)

class ParallelProcessor:
    def __init__(self):
        self.failed_files = []
        self.batch_size = int(os.getenv('BATCH_SIZE', '50'))
        self.max_retries = int(os.getenv('MAX_RETRIES', '3'))
        self.min_workers = 2
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