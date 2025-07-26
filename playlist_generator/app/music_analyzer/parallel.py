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

logger = logging.getLogger(__name__)

def timeout_handler(signum, frame):
    raise TimeoutException()

class TimeoutException(Exception):
    pass

class UserAbortException(Exception):
    """Raised when user aborts with Ctrl+C and we want to stop all processing."""
    pass

def process_file_worker(filepath: str, status_queue: Optional[object] = None, force_reextract: bool = False) -> Optional[tuple]:
    """Worker function to process a single audio file in parallel.

    Args:
        filepath (str): Path to the audio file.
        status_queue (multiprocessing.Queue, optional): Queue to notify main process of long-running files.

    Returns:
        Optional[tuple]: (features dict, filepath, db_write_success bool) or None on failure.
    """
    from utils.logging_setup import setup_queue_colored_logging
    setup_queue_colored_logging()
    import essentia
    essentia.log.infoActive = False
    essentia.log.warningActive = False
    
    import os
    import traceback
    from .feature_extractor import AudioAnalyzer
    from utils.logging_setup import setup_colored_logging
    setup_colored_logging()
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
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(300)
    try:
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
    except Exception:
        size_mb = 0

    while retry_count <= max_retries:
        try:
            if not os.path.exists(filepath):
                notified["shown"] = True
                logger.warning(f"File not found: {filepath}")
                from .feature_extractor import AudioAnalyzer
                audio_analyzer = AudioAnalyzer()
                file_info = audio_analyzer._get_file_info(filepath)
                audio_analyzer._mark_failed(file_info)
                return None, filepath, False
            if os.path.getsize(filepath) < 1024:
                notified["shown"] = True
                logger.warning(f"Skipping small file: {filepath}")
                from .feature_extractor import AudioAnalyzer
                audio_analyzer = AudioAnalyzer()
                file_info = audio_analyzer._get_file_info(filepath)
                audio_analyzer._mark_failed(file_info)
                return None, filepath, False
            if not filepath.lower().endswith(('.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac')):
                notified["shown"] = True
                logger.warning(f"Unsupported extension, skipping: {filepath}")
                from .feature_extractor import AudioAnalyzer
                audio_analyzer = AudioAnalyzer()
                file_info = audio_analyzer._get_file_info(filepath)
                audio_analyzer._mark_failed(file_info)
                return None, filepath, False
            result = None
            try:
                result = audio_analyzer.extract_features(filepath, force_reextract=force_reextract)
            except Exception as e:
                logger.debug(f"ERROR in worker for {os.path.basename(filepath)}: {e}\n{traceback.format_exc()}")
                return None, filepath, False
            finally:
                # Cancel the alarm
                signal.alarm(0)
                # If result is None or database write failed, mark as failed and return failure
                features, db_write_success, file_hash = (result if result is not None else (None, False, None))
                if not features or not db_write_success:
                    file_info = audio_analyzer._get_file_info(filepath)
                    audio_analyzer._mark_failed(file_info)
                    return None, filepath, False
            if result and result[0] is not None:
                features, db_write_success, _ = result
                for key in ['bpm', 'centroid', 'duration']:
                    if features.get(key) is None:
                        features[key] = 0.0
                logger.info(f"PROCESSED: {filepath}")
                return features, filepath, db_write_success
            if retry_count < max_retries:
                retry_count += 1
                time.sleep(backoff_time)
                backoff_time *= 2  # Exponential backoff
                logger.debug(f"Retrying {filepath} (attempt {retry_count}/{max_retries})")
                continue
            logger.warning(f"Feature extraction failed for {filepath}")
            return None, filepath, False
        except TimeoutException:
            logger.debug(f"TIMEOUT in worker for {os.path.basename(filepath)}")
            return None, filepath, False
        except Exception as e:
            logger.error(f"FATAL ERROR in worker for {os.path.basename(filepath)}: {e}\n{traceback.format_exc()}")
            return None, filepath, False
    notified["shown"] = True
    return None, filepath, False

class ParallelProcessor:
    """Parallel processor for batch audio analysis using multiprocessing."""
    def __init__(self, enforce_fail_limit: bool = False, retry_counter=None) -> None:
        self.failed_files = []
        self.batch_size = int(os.getenv('BATCH_SIZE', '50'))
        self.max_retries = int(os.getenv('MAX_RETRIES', '3'))
        self.min_workers = 2
        self.max_workers = int(os.getenv('MAX_WORKERS', str(mp.cpu_count())))
        self.enforce_fail_limit = enforce_fail_limit
        if retry_counter is None and enforce_fail_limit:
            manager = mp.Manager()
            self.retry_counter = manager.dict()
        else:
            self.retry_counter = retry_counter

    def process(self, file_list: List[str], workers: int = None, status_queue: Optional[object] = None, stop_event=None, force_reextract: bool = False, enforce_fail_limit: bool = None, retry_counter=None) -> iter:
        if enforce_fail_limit is not None:
            self.enforce_fail_limit = enforce_fail_limit
        if retry_counter is not None:
            self.retry_counter = retry_counter
        if not file_list:
            return
        self.workers = max(self.min_workers, min(workers or self.max_workers, self.max_workers))
        self.batch_size = min(self.batch_size, len(file_list))
        yield from self._process_parallel(file_list, status_queue, stop_event=stop_event, force_reextract=force_reextract)

    def _process_parallel(self, file_list, status_queue, stop_event=None, force_reextract: bool = False):
        retries = 0
        remaining_files = file_list[:]
        while remaining_files and retries < self.max_retries:
            try:
                logger.info(f"Starting multiprocessing with {self.workers} workers (retry {retries})")
                ctx = mp.get_context('spawn')
                failed_in_batch = []
                enrich_later = []
                for i in range(0, len(remaining_files), self.batch_size):
                    batch = remaining_files[i:i+self.batch_size]
                    if stop_event and stop_event.is_set():
                        break
                    with ctx.Pool(processes=self.workers) as pool:
                        try:
                            from functools import partial
                            worker_func = partial(process_file_worker, status_queue=status_queue, force_reextract=force_reextract)
                            for features, filepath, db_write_success in pool.imap_unordered(worker_func, batch):
                                if self.enforce_fail_limit:
                                    # Use in-memory retry counter
                                    count = self.retry_counter.get(filepath, 0)
                                    if count >= 3:
                                        # Mark as failed in DB
                                        import sqlite3
                                        conn = sqlite3.connect(os.getenv('CACHE_DIR', '/app/cache') + '/audio_analysis.db')
                                        cur = conn.cursor()
                                        cur.execute("UPDATE audio_features SET failed = 1 WHERE file_path = ?", (filepath,))
                                        conn.commit()
                                        conn.close()
                                        logger.warning(f"File {filepath} failed 3 times in parallel mode. Skipping for the rest of this run.")
                                        continue
                                if stop_event and stop_event.is_set():
                                    break
                                import sqlite3
                                conn = sqlite3.connect(os.getenv('CACHE_DIR', '/app/cache') + '/audio_analysis.db')
                                cur = conn.cursor()
                                cur.execute("PRAGMA table_info(audio_features)")
                                columns = [row[1] for row in cur.fetchall()]
                                if features and db_write_success:
                                    # On success, reset failed
                                    cur.execute("UPDATE audio_features SET failed = 0 WHERE file_path = ?", (filepath,))
                                    conn.commit()
                                    conn.close()
                                    yield features, filepath, db_write_success
                                else:
                                    if self.enforce_fail_limit:
                                        count = self.retry_counter.get(filepath, 0) + 1
                                        self.retry_counter[filepath] = count
                                        logger.warning(f"Retrying file {filepath} (retry count: {count})")
                                        if count >= 3:
                                            cur.execute("UPDATE audio_features SET failed = 1 WHERE file_path = ?", (filepath,))
                                        conn.commit()
                                        conn.close()
                                        enrich_later.append(filepath)
                                        logger.warning(f"File {filepath} failed 3 times in parallel mode. Skipping for the rest of this run.")
                                        continue
                                    else:
                                        cur.execute("UPDATE audio_features SET failed = 1 WHERE file_path = ?", (filepath,))
                                        conn.commit()
                                        conn.close()
                                        yield None, filepath, False
                        except KeyboardInterrupt:
                            logger.debug("KeyboardInterrupt received, terminating pool and exiting cleanly...")
                            pool.terminate()
                            pool.join()
                            raise UserAbortException()
                        finally:
                            pool.terminate()
                            pool.join()
                if enrich_later:
                    from music_analyzer.feature_extractor import AudioAnalyzer
                    analyzer = AudioAnalyzer(os.getenv('CACHE_DIR', '/app/cache') + '/audio_analysis.db')
                    for filepath in enrich_later:
                        file_info = analyzer._get_file_info(filepath)
                        analyzer.enrich_metadata_for_failed_file(file_info)
                if failed_in_batch:
                    logger.info(f"Retrying {len(failed_in_batch)} failed files in next round")
                    files_to_retry = []
                    for filepath in failed_in_batch:
                        if self.enforce_fail_limit:
                            count = self.retry_counter.get(filepath, 0)
                            if count < 3:
                                files_to_retry.append(filepath)
                    remaining_files = files_to_retry if self.enforce_fail_limit else []
                    self.failed_files.extend(files_to_retry)
                else:
                    logger.info(f"Batch processing complete: {len(file_list)} files processed, {len(self.failed_files)} failed in total.")
                    return
            except (mp.TimeoutError, BrokenPipeError, ConnectionResetError) as e:
                logger.error(f"Multiprocessing error: {str(e)}")
                retries += 1
                if retries < self.max_retries:
                    self.workers = max(self.min_workers, self.workers // 2)
                    logger.warning(f"Reducing workers to {self.workers} and retrying with {len(remaining_files)} remaining files")
                    time.sleep(2 ** retries)
                else:
                    logger.error("Max retries reached, switching to sequential for remaining files")
                    for filepath in remaining_files:
                        features, _, db_write_success = process_file_worker(filepath, status_queue)
                        if features and db_write_success:
                            yield features, filepath, db_write_success
                        else:
                            self.failed_files.append(filepath)
                            yield None, filepath, False
                    return