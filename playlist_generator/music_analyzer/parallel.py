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
    return {"status": "ok", "file": filepath}, filepath, True

def estimate_memory_for_file(file_path):
    try:
        size_mb = os.path.getsize(file_path) / (1024 * 1024)
        ext = os.path.splitext(file_path)[1].lower()
        if ext in ('.mp3', '.m4a', '.aac', '.ogg', '.opus'):
            return max(2048, size_mb * 15)  # 15x file size, min 2GB
        else:  # wav, flac, etc.
            return max(1024, size_mb * 2)   # 2x file size, min 1GB
    except Exception:
        return 2048

def adaptive_parallel_process(file_list, worker_func, max_workers, max_mem_mb, timeout=120):
    active = []  # List of (proc, conn, file_path, est_mem)
    mem_used = 0
    file_iter = iter(file_list)
    results = []
    while True:
        # Clean up finished workers
        for proc, conn, file_path, mem in active[:]:
            if not proc.is_alive():
                proc.join()
                conn.close()
                active.remove((proc, conn, file_path, mem))
                mem_used -= mem
        # Launch new workers if possible
        if len(active) < max_workers:
            try:
                file_path = next(file_iter)
            except StopIteration:
                break
            est_mem = estimate_memory_for_file(file_path)
            if mem_used + est_mem > max_mem_mb and len(active) > 0:
                time.sleep(0.5)
                continue
            # Always allow at least one worker
            pconn, cconn = mp.Pipe()
            proc = mp.Process(target=worker_wrapper, args=(worker_func, file_path, cconn, est_mem))
            proc.start()
            active.append((proc, pconn, file_path, est_mem))
            mem_used += est_mem
            results.append((proc, pconn, file_path))
        else:
            time.sleep(0.1)
    # Wait for all to finish
    for proc, conn, file_path, mem in active:
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
    # Collect results from already finished
    for proc, conn, file_path in results:
        if conn.poll():
            try:
                yield conn.recv()
            except EOFError:
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
        max_memory_mb = int(os.getenv('MAX_MEMORY_MB', '8192'))
        worker_max_mem_mb = int(os.getenv('WORKER_MAX_MEM_MB', '2048'))
        retries = 0
        remaining_files = file_list[:]
        while remaining_files and retries < self.max_retries:
            try:
                logger.info(f"Starting adaptive multiprocessing with memory cap {max_memory_mb}MB (retry {retries})")
                failed_in_batch = []
                for features, filepath, db_write_success in adaptive_parallel_process(
                    remaining_files, process_file_worker, self.workers, max_memory_mb):
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