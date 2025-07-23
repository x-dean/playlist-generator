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

def worker_wrapper(worker_func, file_path, conn, mem_limit_mb):
    print(f"[DEBUG] worker_wrapper started for {file_path}")
    import sys; sys.stdout.flush()
    result = worker_func(file_path, mem_limit_mb)
    conn.send(result)
    conn.close()

def process_file_worker(filepath, dynamic_mem_limit_mb=None):
    import time
    print(f"[DEBUG] Minimal worker for {filepath}")
    time.sleep(2)
    # Return a dict with 'metadata' key for compatibility with main loop
    return {"filepath": filepath, "metadata": {"dummy": True}}, filepath, True

def adaptive_parallel_process(file_list, worker_func, max_workers, timeout=120):
    active = []  # List of (proc, conn, file_path)
    file_iter = iter(file_list)
    while True:
        # Clean up finished workers and yield results
        for proc, conn, file_path in active[:]:
            if not proc.is_alive():
                proc.join()
                if conn.poll():
                    try:
                        yield conn.recv()
                    except EOFError:
                        yield (None, file_path, False)
                else:
                    yield (None, file_path, False)
                conn.close()
                active.remove((proc, conn, file_path))
        # Launch new workers if possible
        if len(active) < max_workers:
            try:
                file_path = next(file_iter)
            except StopIteration:
                break
            pconn, cconn = mp.Pipe()
            proc = mp.Process(target=worker_wrapper, args=(worker_func, file_path, cconn, None))
            proc.start()
            active.append((proc, pconn, file_path))
        else:
            time.sleep(0.1)
    # Wait for all to finish and yield their results
    for proc, conn, file_path in active:
        proc.join(timeout)
        if proc.is_alive():
            logger.warning(f"Timeout: killing process for {file_path}")
            proc.terminate()
            proc.join()
        if conn.poll():
            try:
                yield conn.recv()
            except EOFError:
                yield (None, file_path, False)
        else:
            yield (None, file_path, False)
        conn.close()

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
                logger.info(f"Starting adaptive multiprocessing (retry {retries})")
                failed_in_batch = []
                for features, filepath, db_write_success in adaptive_parallel_process(
                    remaining_files, process_file_worker, self.workers):
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