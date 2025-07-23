# music_analyzer/parallel.py
import multiprocessing as mp
import os
import sys
from tqdm import tqdm
import logging
import time
from .audio_analyzer import AudioAnalyzer
import psutil
import resource
import threading

logger = logging.getLogger(__name__)

# Top-level worker wrapper for multiprocessing (must be picklable)
def worker_wrapper(worker_func, file_path, conn, mem_limit_mb):
    result = worker_func(file_path, mem_limit_mb)
    conn.send(result)
    conn.close()

def process_file_worker(filepath, dynamic_mem_limit_mb=None):
    import os
    import resource
    from .audio_analyzer import AudioAnalyzer
    audio_analyzer = AudioAnalyzer()
    max_retries = 2
    retry_count = 0
    backoff_time = 1  # Initial backoff time in seconds

    # Set per-worker memory limit (Linux only)
    # If WORKER_MAX_MEM_MB_FORCE is set, use it as a hard override for all workers
    # Otherwise, use the dynamic value passed in (dynamic_mem_limit_mb)
    def set_memory_limit_mb(mb):
        soft = hard = int(mb) * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (soft, hard))
    try:
        forced = os.getenv('WORKER_MAX_MEM_MB_FORCE')
        if forced:
            set_memory_limit_mb(forced)
        elif dynamic_mem_limit_mb:
            set_memory_limit_mb(dynamic_mem_limit_mb)
        else:
            set_memory_limit_mb(int(os.getenv('WORKER_MAX_MEM_MB', '2048')))
    except Exception as e:
        # If resource is not available (e.g., on Windows), just continue
        pass

    # Memory limit check (per worker)
    max_mem_mb = int(os.getenv('WORKER_MAX_MEM_MB', '2048'))
    process = psutil.Process(os.getpid())
    if process.memory_info().rss > max_mem_mb * 1024 * 1024:
        logger.warning(f"Worker memory exceeded {max_mem_mb}MB, skipping {filepath}")
        return None, filepath, False

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
            return max(512, size_mb * 2)  # 2x file size, min 512MB
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
        while True:
            # Clean up finished workers
            with self.lock:
                for w in self.active_workers[:]:
                    if not w.is_alive():
                        w.join()
                        self.active_workers.remove(w)
            # Launch new workers if possible
            try:
                file_path = next(file_iter)
            except StopIteration:
                break
            est_mem = self.estimate_memory_for_file(file_path)
            current_used = psutil.virtual_memory().used / (1024 * 1024)
            current_avail = psutil.virtual_memory().available / (1024 * 1024)
            print(f"[DEBUG] Considering file: {file_path}")
            print(f"[DEBUG] Estimated memory for file: {est_mem} MB")
            print(f"[DEBUG] Current used: {current_used:.2f} MB, Available: {current_avail:.2f} MB, Max allowed: {self.max_memory_mb} MB")
            while not self.can_launch_worker(file_path):
                print(f"[DEBUG] Not enough memory to launch worker for {file_path}. Waiting...")
                time.sleep(1)
                with self.lock:
                    for w in self.active_workers[:]:
                        if not w.is_alive():
                            w.join()
                            self.active_workers.remove(w)
                current_used = psutil.virtual_memory().used / (1024 * 1024)
                current_avail = psutil.virtual_memory().available / (1024 * 1024)
                print(f"[DEBUG] (Wait loop) Used: {current_used:.2f} MB, Available: {current_avail:.2f} MB")
            print(f"[DEBUG] Launching worker for {file_path} with memory limit {est_mem} MB")
            # Launch worker with dynamic memory limit
            pconn, cconn = mp.Pipe()
            proc = mp.Process(target=worker_wrapper, args=(worker_func, file_path, cconn, est_mem))
            proc.start()
            self.active_workers.append(proc)
            results.append(pconn)
        # Wait for all workers to finish
        for w in self.active_workers:
            w.join()
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