# music_analyzer/parallel.py
import multiprocessing as mp
import os
import sys
from tqdm import tqdm
import logging
import time
from .audio_analyzer import AudioAnalyzer
from typing import Optional
import threading
import signal

logger = logging.getLogger(__name__)

def timeout_handler(signum, frame):
    raise TimeoutException()

class TimeoutException(Exception):
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
    from .audio_analyzer import AudioAnalyzer
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
                from .audio_analyzer import AudioAnalyzer
                audio_analyzer = AudioAnalyzer()
                file_info = audio_analyzer._get_file_info(filepath)
                audio_analyzer._mark_failed(file_info)
                return None, filepath, False
            if os.path.getsize(filepath) < 1024:
                notified["shown"] = True
                logger.warning(f"Skipping small file: {filepath}")
                from .audio_analyzer import AudioAnalyzer
                audio_analyzer = AudioAnalyzer()
                file_info = audio_analyzer._get_file_info(filepath)
                audio_analyzer._mark_failed(file_info)
                return None, filepath, False
            if not filepath.lower().endswith(('.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac')):
                notified["shown"] = True
                logger.warning(f"Unsupported extension, skipping: {filepath}")
                from .audio_analyzer import AudioAnalyzer
                audio_analyzer = AudioAnalyzer()
                file_info = audio_analyzer._get_file_info(filepath)
                audio_analyzer._mark_failed(file_info)
                return None, filepath, False
            result = None
            try:
                result = audio_analyzer.extract_features(filepath)
            except Exception as e:
                print(f"ERROR in worker for {os.path.basename(filepath)}: {e}\n{traceback.format_exc()}")
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
            print(f"TIMEOUT in worker for {os.path.basename(filepath)}")
            return None, filepath, False
        except Exception as e:
            print(f"FATAL ERROR in worker for {os.path.basename(filepath)}: {e}\n{traceback.format_exc()}")
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

    def process(self, file_list: list[str], workers: int = None, status_queue: Optional[object] = None) -> iter:
        """Process a list of files in parallel.

        Args:
            file_list (list[str]): List of file paths.
            workers (int, optional): Number of worker processes. Defaults to None.
            status_queue (multiprocessing.Queue, optional): Queue for long-running file notifications.

        Yields:
            dict: Extracted features for each file.
        """
        if not file_list:
            return
        self.workers = max(self.min_workers, min(workers or self.max_workers, self.max_workers))
        self.batch_size = min(self.batch_size, len(file_list))
        yield from self._process_parallel(file_list, status_queue)

    def _process_parallel(self, file_list, status_queue):
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
                        # Use starmap to pass status_queue to each worker
                        from functools import partial
                        worker_func = partial(process_file_worker, status_queue=status_queue)
                        for features, filepath, db_write_success in pool.imap_unordered(worker_func, batch):
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
                    self.workers = max(self.min_workers, self.workers // 2)
                    logger.warning(f"Reducing workers to {self.workers} and retrying with {len(remaining_files)} remaining files")
                    time.sleep(2 ** retries)
                else:
                    logger.error("Max retries reached, switching to sequential for remaining files")
                    for filepath in remaining_files:
                        features, _, db_write_success = process_file_worker(filepath, status_queue)
                        if features and db_write_success:
                            yield features
                        else:
                            self.failed_files.append(filepath)
                    return