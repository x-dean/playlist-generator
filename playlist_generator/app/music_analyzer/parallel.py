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

logger = logging.getLogger()

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
    logger.debug(f"Worker starting processing for: {filepath}")
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
            logger.debug(f"Notifying main process about long-running file: {filepath}")
            status_queue.put(filepath)
            notified["shown"] = True
    if status_queue is not None:
        notifier = threading.Thread(target=notify_if_long, daemon=True)
        notifier.start()
        logger.debug(f"Started notification thread for: {filepath}")

    # Set a timeout for processing each file (e.g., 5 minutes = 300 seconds)
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(300)
    try:
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        logger.debug(f"File size: {size_mb:.2f} MB")
    except Exception:
        size_mb = 0
        logger.warning(f"Could not determine file size for: {filepath}")

    while retry_count <= max_retries:
        try:
            logger.debug(f"Processing attempt {retry_count + 1}/{max_retries + 1} for: {filepath}")
            
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
                logger.debug(f"Starting feature extraction for: {filepath}")
                result = audio_analyzer.extract_features(filepath, force_reextract=force_reextract)
                logger.debug(f"Feature extraction completed for: {filepath}")
            except Exception as e:
                logger.debug(f"ERROR in worker for {os.path.basename(filepath)}: {e}\n{traceback.format_exc()}")
                result = None
            finally:
                signal.alarm(0)  # Cancel the alarm
            if result and result[0]:
                notified["shown"] = True
                logger.info(f"PROCESSED: {filepath}")
                return result
            else:
                logger.warning(f"Feature extraction failed for {filepath}")
                if retry_count < max_retries:
                    logger.debug(f"Retrying {filepath} (attempt {retry_count + 1}/{max_retries})")
                    time.sleep(backoff_time)
                    retry_count += 1
                    backoff_time *= 2  # Exponential backoff
                else:
                    logger.debug(f"TIMEOUT in worker for {os.path.basename(filepath)}")
                    return None, filepath, False
        except TimeoutException:
            logger.debug(f"TIMEOUT in worker for {os.path.basename(filepath)}")
            return None, filepath, False
        except Exception as e:
            logger.error(f"FATAL ERROR in worker for {os.path.basename(filepath)}: {e}\n{traceback.format_exc()}")
            return None, filepath, False
    return None, filepath, False

class ParallelProcessor:
    """Parallel processor for audio analysis using multiprocessing."""
    def __init__(self, enforce_fail_limit: bool = False, retry_counter=None) -> None:
        self.enforce_fail_limit = enforce_fail_limit
        self.retry_counter = retry_counter
        self.failed_files = []
        self.workers = None
        logger.debug("Initialized ParallelProcessor")

    def process(self, file_list: List[str], workers: int = None, status_queue: Optional[object] = None, stop_event=None, force_reextract: bool = False, enforce_fail_limit: bool = None, retry_counter=None) -> iter:
        if enforce_fail_limit is not None:
            self.enforce_fail_limit = enforce_fail_limit
        if retry_counter is not None:
            self.retry_counter = retry_counter
        
        logger.info(f"Starting parallel processing with {len(file_list)} files")
        logger.debug(f"Parallel processing parameters: workers={workers}, force_reextract={force_reextract}, enforce_fail_limit={enforce_fail_limit}")
        
        if workers is None:
            workers = max(1, multiprocessing.cpu_count() // 2)
        self.workers = workers
        
        remaining_files = file_list[:]
        retries = 0
        max_retries = 3
        
        while remaining_files and retries < max_retries:
            logger.info(f"Starting multiprocessing with {self.workers} workers (retry {retries})")
            
            try:
                with multiprocessing.Pool(processes=self.workers) as pool:
                    logger.debug(f"Created process pool with {self.workers} workers")
                    
                    # Prepare arguments for each file
                    args_list = [(f, status_queue, force_reextract) for f in remaining_files]
                    logger.debug(f"Prepared {len(args_list)} tasks for processing")
                    
                    # Process files in parallel
                    results = []
                    for result in pool.imap_unordered(process_file_worker, args_list):
                        if stop_event and stop_event.is_set():
                            logger.info("Stop event received, terminating parallel processing")
                            pool.terminate()
                            pool.join()
                            return
                        
                        if result and result[0]:
                            logger.debug(f"Successful processing result: {result[1]}")
                            yield result
                        else:
                            failed_file = result[1] if result else "unknown"
                            logger.warning(f"File {failed_file} failed 3 times in parallel mode. Skipping for the rest of this run.")
                            self.failed_files.append(failed_file)
                    
                    logger.debug(f"Parallel processing completed for batch")
                    
            except KeyboardInterrupt:
                logger.debug("KeyboardInterrupt received, terminating pool and exiting cleanly...")
                if 'pool' in locals():
                    pool.terminate()
                    pool.join()
                return
            except Exception as e:
                logger.error(f"Multiprocessing error: {str(e)}")
                import traceback
                logger.error(f"Multiprocessing error traceback: {traceback.format_exc()}")
                
                # Reduce workers and retry
                if self.workers > 1:
                    self.workers = max(1, self.workers // 2)
                    logger.warning(f"Reducing workers to {self.workers} and retrying with {len(remaining_files)} remaining files")
                    retries += 1
                else:
                    logger.error("Max retries reached, switching to sequential for remaining files")
                    break
        
        # Process any remaining failed files
        if self.failed_files:
            logger.info(f"Retrying {len(self.failed_files)} failed files in next round")
            for failed_file in self.failed_files:
                try:
                    result = process_file_worker(failed_file, status_queue, force_reextract)
                    if result and result[0]:
                        logger.debug(f"Retry successful for: {failed_file}")
                        yield result
                    else:
                        logger.warning(f"Retry failed for: {failed_file}")
                        count = self.retry_counter.get(failed_file, 0) + 1 if self.retry_counter else 1
                        if self.retry_counter:
                            self.retry_counter[failed_file] = count
                        if count >= 3:
                            logger.warning(f"File {failed_file} failed 3 times in parallel mode. Skipping for the rest of this run.")
                except Exception as e:
                    logger.error(f"Error during retry processing: {e}")
        
        logger.info(f"Batch processing complete: {len(file_list)} files processed, {len(self.failed_files)} failed in total.")