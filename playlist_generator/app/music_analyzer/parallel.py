# music_analyzer/parallel.py
import multiprocessing as mp
import os
import sys
from tqdm import tqdm
import logging
import time
from .feature_extractor import AudioAnalyzer
from typing import Optional, List
import threading
import signal
import psutil

logger = logging.getLogger(__name__)


def timeout_handler(signum, frame):
    raise TimeoutException()


class TimeoutException(Exception):
    pass


def get_memory_aware_worker_count(max_workers: int = None, memory_limit_str: str = None) -> int:
    """Calculate optimal worker count based on available memory."""
    try:
        from utils.memory_monitor import MemoryMonitor
        monitor = MemoryMonitor()
        return monitor.get_optimal_worker_count(max_workers or mp.cpu_count(), memory_limit_str=memory_limit_str)
    except Exception as e:
        logger.warning(f"Could not determine memory-aware worker count: {e}")
        return max(1, min(max_workers or mp.cpu_count(), mp.cpu_count()))


def process_file_worker(filepath: str, status_queue: Optional[object] = None, force_reextract: bool = False, fast_mode: bool = False) -> Optional[tuple]:
    """Worker function to process a single audio file in parallel.

    Args:
        filepath (str): Path to the audio file.
        status_queue (multiprocessing.Queue, optional): Queue to notify main process of long-running files.
        force_reextract (bool): If True, bypass the cache for all files.
        fast_mode (bool): If True, use fast mode for 3-5x faster processing.

    Returns:
        Optional[tuple]: (features dict, filepath, db_write_success bool) or None on failure.
    """
    import essentia
    # Essentia logging is now handled in main playlista script
    
    # Add parallel worker identification
    import os
    worker_pid = os.getpid()
    logger.info(f"🔄 PARALLEL WORKER {worker_pid}: Starting analysis for {os.path.basename(filepath)}")

    import os
    import traceback
    from .feature_extractor import AudioAnalyzer
    import logging
    import os
    log_level = os.getenv('LOG_LEVEL', 'INFO')
    logging.getLogger().setLevel(getattr(logging, log_level.upper(), logging.INFO))
    audio_analyzer = AudioAnalyzer()

    max_retries = 2
    retry_count = 0
    backoff_time = 1  # Initial backoff time in seconds

    notified = {"shown": False}

    def notify_if_long():
        time.sleep(5)
        if not notified["shown"] and status_queue is not None:
            status_queue.put(filepath)
            notified["shown"] = True
    if status_queue is not None:
        notifier = threading.Thread(target=notify_if_long, daemon=True)
        notifier.start()

    # Set a timeout for processing each file (e.g., 5 minutes = 300 seconds)
    # Only set signal handler in main thread to avoid issues
    import threading
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(300)  # 5 minutes timeout

    while retry_count <= max_retries:
        try:
            # Use fast mode if enabled
            if fast_mode:
                logger.info(f"🔄 PARALLEL WORKER {worker_pid}: Using FAST MODE for {os.path.basename(filepath)}")
                result = audio_analyzer.extract_features_fast(filepath, force_reextract=force_reextract)
            else:
                result = audio_analyzer.extract_features(filepath, force_reextract=force_reextract)
            
            # Cancel the alarm
            signal.alarm(0)
            
            if result and result[0] is not None:
                features, db_write_success, _ = result
                logger.debug(
                    f"Worker result for {filepath}: features={features is not None}, db_write={db_write_success}")
                for key in ['bpm', 'centroid', 'duration']:
                    if features.get(key) is None:
                        features[key] = 0.0
                logger.info(f"PROCESSED: {filepath}")
                return features, filepath, db_write_success
            if retry_count < max_retries:
                retry_count += 1
                time.sleep(backoff_time)
                backoff_time *= 2  # Exponential backoff
                logger.debug(
                    f"Retrying {filepath} (attempt {retry_count}/{max_retries})")
                continue
            logger.warning(f"Feature extraction failed for {filepath}")
            return None, filepath, False
        except TimeoutException:
            logger.debug(f"TIMEOUT in worker for {os.path.basename(filepath)}")
            return None, filepath, False
        except Exception as e:
            logger.error(
                f"FATAL ERROR in worker for {os.path.basename(filepath)}: {e}\n{traceback.format_exc()}")
            return None, filepath, False
    notified["shown"] = True
    return None, filepath, False


class ParallelProcessor:
    """Parallel processor for batch audio analysis using multiprocessing."""

    def __init__(self, enforce_fail_limit: bool = False, retry_counter=None) -> None:
        self.failed_files = []
        self.batch_size = None  # Will be set dynamically based on worker count
        self.max_retries = int(os.getenv('MAX_RETRIES', '3'))
        self.min_workers = 2
        self.max_workers = int(os.getenv('MAX_WORKERS', str(mp.cpu_count())))
        self.enforce_fail_limit = enforce_fail_limit
        if retry_counter is None and enforce_fail_limit:
            manager = mp.Manager()
            self.retry_counter = manager.dict()
        else:
            self.retry_counter = retry_counter

    def process(self, file_list: List[str], workers: int = None, status_queue: Optional[object] = None, force_reextract: bool = False, enforce_fail_limit: bool = None, retry_counter=None, fast_mode: bool = False) -> iter:
        if enforce_fail_limit is not None:
            self.enforce_fail_limit = enforce_fail_limit
        if retry_counter is not None:
            self.retry_counter = retry_counter
        if not file_list:
            return
        
        # Use memory-aware worker calculation if MAX_WORKERS is not explicitly set
        if workers is None and os.getenv('MAX_WORKERS') is None:
            workers = get_memory_aware_worker_count()
        elif workers is None:
            workers = self.max_workers
            
        self.workers = max(self.min_workers, min(workers, self.max_workers))
        
        # Set batch size based on memory constraints
        batch_size_env = os.getenv('BATCH_SIZE')
        if batch_size_env:
            self.batch_size = int(batch_size_env)
        else:
            # Default: batch size should be much larger than workers to amortize pool creation overhead
            # Use 10 items per worker for optimal efficiency
            min_batch_size = self.workers * 10
            # But don't make batches too large to avoid memory issues
            max_batch_size = min(100, len(file_list))
            self.batch_size = min(min_batch_size, max_batch_size)
            
        logger.info(
            f"🔄 PARALLEL: Using {self.workers} workers with batch size {self.batch_size} (efficient batch processing)")
        if fast_mode:
            logger.info("🔄 PARALLEL: FAST MODE enabled for 3-5x faster processing")
        logger.info(f"🔄 PARALLEL: Starting multiprocessing pool for {len(file_list)} files")
        yield from self._process_parallel(file_list, status_queue, force_reextract=force_reextract, fast_mode=fast_mode)

    def _process_parallel(self, file_list, status_queue, force_reextract: bool = False, fast_mode: bool = False):
        retries = 0
        remaining_files = file_list[:]
        total_batches = (len(remaining_files) + self.batch_size - 1) // self.batch_size
        current_batch = 0
        
        logger.info(f"🔄 PARALLEL: Processing {len(remaining_files)} files in {total_batches} batches")
        
        while remaining_files and retries < self.max_retries:
            try:
                logger.info(
                    f"🔄 PARALLEL: Starting multiprocessing with {self.workers} workers (retry {retries})")
                ctx = mp.get_context('spawn')
                failed_in_batch = []
                enrich_later = []
                
                for i in range(0, len(remaining_files), self.batch_size):
                    current_batch += 1
                    batch = remaining_files[i:i+self.batch_size]
                    
                    logger.info(f"🔄 PARALLEL: Starting batch {current_batch}/{total_batches} with {len(batch)} files (batch size: {self.batch_size})")
                    
                    # Check memory before starting batch
                    try:
                        from utils.memory_monitor import MemoryMonitor
                        memory_monitor = MemoryMonitor()
                        memory_status = memory_monitor.check_memory_usage()
                        
                        # Store original batch size if not already stored
                        if not hasattr(self, 'original_batch_size'):
                            self.original_batch_size = self.batch_size
                        
                        if memory_status['is_critical']:
                            logger.warning("Memory critical, reducing batch size")
                            new_batch_size = max(self.workers, self.batch_size // 2)  # Don't go below workers count
                            if new_batch_size != self.batch_size:
                                logger.info(f"Reducing batch size from {self.batch_size} to {new_batch_size}")
                                self.batch_size = new_batch_size
                        elif memory_status['is_high']:
                            # Keep current batch size
                            logger.debug("Memory high, keeping current batch size")
                        else:
                            # Memory is good, restore original batch size
                            if self.batch_size < self.original_batch_size:
                                logger.info(f"Memory improved, restoring batch size from {self.batch_size} to {self.original_batch_size}")
                                self.batch_size = self.original_batch_size
                    except Exception as e:
                        logger.debug(f"Could not check memory: {e}")
                    
                    # Create pool for this batch
                    pool = None
                    try:
                        pool = ctx.Pool(processes=self.workers)
                        logger.debug(f"🔄 PARALLEL: Pool created with {self.workers} workers for batch of {len(batch)} files")
                        
                        # Check if workers are actually running
                        import psutil
                        pool_pids = [p.pid for p in pool._pool]
                        logger.debug(f"🔄 PARALLEL: Worker PIDs: {pool_pids}")
                        
                        # Verify pool is healthy
                        if not pool_pids:
                            logger.error("🔄 PARALLEL: Pool has no workers, recreating pool")
                            pool.terminate()
                            pool.join()
                            pool = ctx.Pool(processes=self.workers)
                            pool_pids = [p.pid for p in pool._pool]
                            logger.debug(f"🔄 PARALLEL: New worker PIDs: {pool_pids}")
                        
                        from functools import partial
                        worker_func = partial(
                            process_file_worker, status_queue=status_queue, force_reextract=force_reextract, fast_mode=fast_mode)

                        # Process results from the batch
                        logger.debug(f"🔄 PARALLEL: Starting to process batch results")
                        processed_in_batch = 0
                        batch_start_time = time.time()
                        
                        # Add timeout for batch processing
                        batch_timeout = 1800  # 30 minutes per batch (increased for larger batches)
                        
                        for features, filepath, db_write_success in pool.imap_unordered(worker_func, batch):
                            processed_in_batch += 1
                            logger.debug(f"🔄 PARALLEL: Got result {processed_in_batch}/{len(batch)} for {os.path.basename(filepath)}")
                            
                            # Check if batch is taking too long
                            batch_elapsed = time.time() - batch_start_time
                            if batch_elapsed > batch_timeout:
                                logger.error(f"🔄 PARALLEL: Batch timeout after {batch_elapsed:.1f}s - terminating pool")
                                pool.terminate()
                                pool.join()
                                raise TimeoutException(f"Batch processing timed out after {batch_elapsed:.1f}s")
                            
                            if self.enforce_fail_limit:
                                # Use in-memory retry counter
                                count = self.retry_counter.get(filepath, 0)
                                if count >= 3:
                                    # Mark as failed in DB
                                    import sqlite3
                                    conn = sqlite3.connect(
                                        os.getenv('CACHE_DIR', '/app/cache') + '/audio_analysis.db')
                                    cur = conn.cursor()
                                    cur.execute(
                                        "UPDATE audio_features SET failed = 1 WHERE file_path = ?", (filepath,))
                                    conn.commit()
                                    conn.close()
                                    logger.warning(
                                        f"File {filepath} failed 3 times in parallel mode. Skipping for the rest of this run.")
                                    continue
                                
                                # Check if this file has been stuck for too long
                                if batch_elapsed > 600:  # 10 minutes per file (increased for larger batches)
                                    logger.warning(f"File {os.path.basename(filepath)} stuck for {batch_elapsed:.1f}s - marking for retry")
                                    # Don't mark as failed yet, just skip for now
                                    continue

                            import sqlite3
                            conn = sqlite3.connect(
                                os.getenv('CACHE_DIR', '/app/cache') + '/audio_analysis.db')
                            cur = conn.cursor()
                            cur.execute(
                                "PRAGMA table_info(audio_features)")
                            columns = [row[1] for row in cur.fetchall()]
                            if features and db_write_success:
                                # On success, reset failed
                                cur.execute(
                                    "UPDATE audio_features SET failed = 0 WHERE file_path = ?", (filepath,))
                                conn.commit()
                                conn.close()
                                yield features, filepath, db_write_success
                            else:
                                if self.enforce_fail_limit:
                                    count = self.retry_counter.get(
                                        filepath, 0) + 1
                                    self.retry_counter[filepath] = count
                                    logger.warning(
                                        f"Retrying file {filepath} (retry count: {count})")
                                    if count >= 3:
                                        cur.execute(
                                            "UPDATE audio_features SET failed = 1 WHERE file_path = ?", (filepath,))
                                    conn.commit()
                                    conn.close()
                                    enrich_later.append(filepath)
                                    logger.warning(
                                        f"File {filepath} failed 3 times in parallel mode. Skipping for the rest of this run.")
                                    continue
                                else:
                                    cur.execute(
                                        "UPDATE audio_features SET failed = 1 WHERE file_path = ?", (filepath,))
                                    conn.commit()
                                    conn.close()
                                    yield None, filepath, False
                        
                        batch_time = time.time() - batch_start_time
                        logger.info(f"🔄 PARALLEL: Batch {current_batch}/{total_batches} completed in {batch_time:.1f}s")
                        
                    except KeyboardInterrupt:
                        logger.debug(
                            "KeyboardInterrupt received, terminating pool and exiting cleanly...")
                        if pool:
                            pool.terminate()
                            pool.join()
                        raise
                    except Exception as e:
                        logger.error(f"Error in parallel processing batch {current_batch}: {e}")
                        if pool:
                            pool.terminate()
                            pool.join()
                        raise
                    finally:
                        # Always terminate and join the pool
                        if pool:
                            pool.terminate()
                            pool.join()
                            logger.debug(f"🔄 PARALLEL: Pool terminated for batch {current_batch}")
                            
                            # Force memory cleanup after each batch
                            import gc
                            gc.collect()
                            logger.debug(f"🔄 PARALLEL: Memory cleanup completed after batch {current_batch}")
                            
                            # Check memory after cleanup and decide if we should pause
                            try:
                                from utils.memory_monitor import should_pause_between_batches, get_pause_duration_seconds
                                should_pause, reason = should_pause_between_batches()
                                
                                if should_pause:
                                    pause_duration = get_pause_duration_seconds()
                                    logger.warning(f"🔄 PARALLEL: Pausing {pause_duration}s between batches: {reason}")
                                    time.sleep(pause_duration)
                                else:
                                    logger.debug("🔄 PARALLEL: Memory status good, no pause needed")
                            except Exception as e:
                                logger.debug(f"Could not check memory after cleanup: {e}")
                                # Default pause if we can't check memory
                                time.sleep(3)
                    
                    # Log batch transition
                    logger.info(f"🔄 PARALLEL: Batch {current_batch}/{total_batches} transition complete")
                    
                    # Check for interrupt between batches
                    try:
                        from playlista import is_interrupt_requested
                        if is_interrupt_requested():
                            logger.warning("🔄 PARALLEL: Interrupt received between batches, stopping processing")
                            # Clean up any remaining processes
                            if pool:
                                pool.terminate()
                                pool.join()
                            return
                    except ImportError:
                        pass  # If we can't import the function, continue normally

                if enrich_later:
                    from music_analyzer.feature_extractor import AudioAnalyzer
                    analyzer = AudioAnalyzer(
                        os.getenv('CACHE_DIR', '/app/cache') + '/audio_analysis.db')
                    for filepath in enrich_later:
                        file_info = analyzer._get_file_info(filepath)
                        analyzer.enrich_metadata_for_failed_file(file_info)
                if failed_in_batch:
                    logger.info(
                        f"Retrying {len(failed_in_batch)} failed files in next round")
                    files_to_retry = []
                    for filepath in failed_in_batch:
                        if self.enforce_fail_limit:
                            count = self.retry_counter.get(filepath, 0)
                            if count < 3:
                                files_to_retry.append(filepath)
                    remaining_files = files_to_retry if self.enforce_fail_limit else []
                    self.failed_files.extend(files_to_retry)
                else:
                    logger.info(
                        f"Batch processing complete: {len(file_list)} files processed, {len(self.failed_files)} failed in total.")
                    return
            except (mp.TimeoutError, BrokenPipeError, ConnectionResetError) as e:
                logger.error(f"Multiprocessing error: {str(e)}")
                retries += 1
                if retries < self.max_retries:
                    self.workers = max(self.min_workers, self.workers // 2)
                    logger.warning(
                        f"Reducing workers to {self.workers} and retrying with {len(remaining_files)} remaining files")
                    time.sleep(2 ** retries)
                else:
                    logger.error(
                        "Max retries reached, switching to sequential for remaining files")
                    for filepath in remaining_files:
                        features, _, db_write_success = process_file_worker(
                            filepath, status_queue)
                        if features and db_write_success:
                            yield features, filepath, db_write_success
                        else:
                            self.failed_files.append(filepath)
                            yield None, filepath, False
                    return
