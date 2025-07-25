# music_analyzer/parallel.py
import multiprocessing as mp
import os
import sys
from tqdm import tqdm
import logging
import time
from .feature_extractor import AudioAnalyzer
from typing import Optional
import threading
import signal

logger = logging.getLogger()

def timeout_handler(signum, frame):
    raise TimeoutException()

class TimeoutException(Exception):
    pass

class UserAbortException(Exception):
    """Raised when user aborts with Ctrl+C and we want to stop all processing."""
    pass

def process_file_worker(filepath: str, status_queue: Optional[object] = None) -> Optional[tuple]:
    """Worker function to process a single audio file in parallel.

    Args:
        filepath (str): Path to the audio file.
        status_queue (multiprocessing.Queue, optional): Queue to notify main process of long-running files.

    Returns:
        Optional[tuple]: (features dict, filepath, db_write_success bool) or None on failure.
    """
    
    import os
    import traceback
    from .feature_extractor import AudioAnalyzer
    from utils.logging_setup import setup_colored_logging
    setup_colored_logging()
    import logging
    import os
    log_level = os.getenv('LOG_LEVEL', 'WARNING')
    logging.getLogger().setLevel(getattr(logging, log_level.upper(), logging.WARNING))
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
                result = audio_analyzer.extract_features(filepath)
            except Exception as e:
                logger.debug(f"ERROR in worker for {os.path.basename(filepath)}: {e}\n{traceback.format_exc()}")
                return None, filepath, False
            finally:
                signal.alarm(0)  # Cancel the alarm
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
    def __init__(self) -> None:
        self.failed_files = []
        self.batch_size = int(os.getenv('BATCH_SIZE', '50'))
        self.max_retries = int(os.getenv('MAX_RETRIES', '3'))
        self.min_workers = 2
        self.max_workers = int(os.getenv('MAX_WORKERS', str(mp.cpu_count())))

    def process(self, file_list: list[str], workers: int = None, status_queue: Optional[object] = None, stop_event=None) -> iter:
        """Process a list of files in parallel.

        Args:
            file_list (list[str]): List of file paths.
            workers (int, optional): Number of worker processes. Defaults to None.
            status_queue (multiprocessing.Queue, optional): Queue for long-running file notifications.
            stop_event (multiprocessing.Event, optional): Event to signal graceful shutdown.

        Yields:
            dict: Extracted features for each file.
        """
        if not file_list:
            return
        self.workers = max(self.min_workers, min(workers or self.max_workers, self.max_workers))
        self.batch_size = min(self.batch_size, len(file_list))
        yield from self._process_parallel(file_list, status_queue, stop_event=stop_event)

    def _process_parallel(self, file_list, status_queue, stop_event=None):
        retries = 0
        remaining_files = file_list[:]
        while remaining_files and retries < self.max_retries:
            try:
                logger.info(f"Starting multiprocessing with {self.workers} workers (retry {retries})")
                ctx = mp.get_context('spawn')
                failed_in_batch = []
                for i in range(0, len(remaining_files), self.batch_size):
                    batch = remaining_files[i:i+self.batch_size]
                    if stop_event and stop_event.is_set():
                        break
                    with ctx.Pool(processes=self.workers) as pool:
                        try:
                            from functools import partial
                            worker_func = partial(process_file_worker, status_queue=status_queue)
                            for features, filepath, db_write_success in pool.imap_unordered(worker_func, batch):
                                if stop_event and stop_event.is_set():
                                    break
                                import sqlite3
                                conn = sqlite3.connect(os.getenv('CACHE_DIR', '/app/cache') + '/audio_analysis.db')
                                cur = conn.cursor()
                                cur.execute("PRAGMA table_info(audio_features)")
                                columns = [row[1] for row in cur.fetchall()]
                                if 'fail_count' not in columns:
                                    cur.execute("ALTER TABLE audio_features ADD COLUMN fail_count INTEGER DEFAULT 0")
                                    conn.commit()
                                if features and db_write_success:
                                    # On success, reset fail_count and failed
                                    cur.execute("UPDATE audio_features SET fail_count = 0, failed = 0 WHERE file_path = ?", (filepath,))
                                    conn.commit()
                                    conn.close()
                                    yield features, filepath
                                else:
                                    # On failure, increment fail_count
                                    cur.execute("SELECT COALESCE(fail_count, 0) FROM audio_features WHERE file_path = ?", (filepath,))
                                    row = cur.fetchone()
                                    fail_count = row[0] if row else 0
                                    new_fail_count = fail_count + 1
                                    if new_fail_count >= 3:
                                        cur.execute("UPDATE audio_features SET fail_count = 0, failed = 1 WHERE file_path = ?", (filepath,))
                                        conn.commit()
                                        conn.close()
                                        logger.warning(f"File {filepath} failed 3 times in parallel mode. Skipping for the rest of this run and resetting fail_count.")
                                        continue  # skip for rest of run, keep failed=1
                                    else:
                                        cur.execute("UPDATE audio_features SET fail_count = ? WHERE file_path = ?", (new_fail_count, filepath))
                                        conn.commit()
                                        conn.close()
                                        failed_in_batch.append(filepath)
                        except KeyboardInterrupt:
                            logger.debug("KeyboardInterrupt received, terminating pool and exiting cleanly...")
                            pool.terminate()
                            pool.join()
                            raise UserAbortException()
                        finally:
                            # Always ensure pool is terminated and joined on exit
                            pool.terminate()
                            pool.join()
                if failed_in_batch:
                    logger.info(f"Retrying {len(failed_in_batch)} failed files in next round")
                    remaining_files = failed_in_batch
                    self.failed_files.extend(failed_in_batch)
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
                            yield features, filepath
                        else:
                            self.failed_files.append(filepath)
                    return