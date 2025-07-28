from tqdm import tqdm
import logging
from typing import List
import os
import time
import traceback
from typing import List, Dict, Any, Optional
import essentia.standard as es
import tensorflow as tf
from music_analyzer.feature_extractor import AudioAnalyzer
import multiprocessing
import threading
import gc
from utils.memory_monitor import log_detailed_memory_info, check_memory_against_limit, check_total_python_rss_limit, get_total_python_rss_gb

logger = logging.getLogger(__name__)


def analyze_file_worker(audio_path, force_reextract, result_queue):
    """Worker function for analyzing audio files in a separate process."""
    import essentia.standard as es
    import tensorflow as tf
    from music_analyzer.feature_extractor import AudioAnalyzer
    cache_file = os.getenv('CACHE_FILE', '/app/cache/audio_analysis.db')
    library = os.getenv('HOST_LIBRARY_PATH', '/root/music/library')
    music = os.getenv('MUSIC_PATH', '/music')
    analyzer = AudioAnalyzer(cache_file=cache_file, library=library, music=music)
    result = analyzer.extract_features(audio_path, force_reextract=force_reextract)
    result_queue.put(result)


class MemoryMonitor:
    """Monitor system memory and trigger cleanup when needed."""
    
    def __init__(self, memory_threshold_percent=85):
        self.memory_threshold_percent = memory_threshold_percent
        self._stop_monitoring = threading.Event()
    
    def get_memory_usage(self):
        """Get current memory usage percentage."""
        try:
            import psutil
            memory = psutil.virtual_memory()
            return memory.percent, memory.used / (1024**3), memory.available / (1024**3)
        except:
            return 0, 0, 0
    
    def is_memory_critical(self):
        """Check if memory usage is critical."""
        usage_percent, used_gb, available_gb = self.get_memory_usage()
        return usage_percent > self.memory_threshold_percent
    
    def force_cleanup(self):
        """Force garbage collection and memory cleanup."""
        logger.warning("Memory usage critical, forcing cleanup...")
        gc.collect()
        time.sleep(1)  # Give system time to free memory
        
        # Log memory after cleanup (use total Python RSS)
        total_rss_gb = get_total_python_rss_gb()
        logger.info(f"Total Python RSS after cleanup: {total_rss_gb:.2f}GB")


class ThreadSafeAudioAnalyzer:
    """Thread-safe wrapper for AudioAnalyzer that creates new connections per thread."""
    
    def __init__(self, cache_file: str, library: str, music: str):
        self.cache_file = cache_file
        self.library = library
        self.music = music
        self._local = threading.local()
    
    def _get_analyzer(self):
        """Get or create an AudioAnalyzer instance for the current thread."""
        if not hasattr(self._local, 'analyzer'):
            self._local.analyzer = AudioAnalyzer(
                cache_file=self.cache_file,
                library=self.library,
                music=self.music
            )
        return self._local.analyzer
    
    def extract_features(self, audio_path: str, force_reextract: bool = False):
        """Extract features using thread-safe analyzer."""
        try:
            analyzer = self._get_analyzer()
            return analyzer.extract_features(audio_path, force_reextract=force_reextract)
        except Exception as e:
            logger.error(f"Thread-safe analyzer failed for {audio_path}: {str(e)}")
            return None, False, None
    
    def cleanup(self):
        """Clean up thread-local resources."""
        if hasattr(self._local, 'analyzer'):
            try:
                if hasattr(self._local.analyzer, 'conn'):
                    self._local.analyzer.conn.close()
            except:
                pass
            delattr(self._local, 'analyzer')


class LargeFileProcessor:
    """Handles large files in a separate process with timeout and memory monitoring."""
    
    def __init__(self, timeout_seconds: int = 600, memory_threshold_percent: int = 85):
        self.timeout_seconds = timeout_seconds
        self.memory_monitor = MemoryMonitor(memory_threshold_percent)
        self._stop_processing = threading.Event()
    
    def _extract_features_in_process(self, audio_path: str, force_reextract: bool = False) -> tuple:
        """Extract features in a separate process to avoid blocking the main process."""
        try:
            # Create a new AudioAnalyzer instance in the subprocess with minimal initialization
            # This avoids pickle issues with thread locks and other non-serializable objects
            cache_file = os.getenv('CACHE_FILE', '/app/cache/audio_analysis.db')
            library = os.getenv('HOST_LIBRARY_PATH', '/root/music/library')
            music = os.getenv('MUSIC_PATH', '/music')
            
            analyzer = AudioAnalyzer(cache_file=cache_file, library=library, music=music)
            result = analyzer.extract_features(audio_path, force_reextract=force_reextract)
            return result
        except Exception as e:
            logger.error(f"Process extraction failed for {audio_path}: {str(e)}")
            return None, False, None
    
    def _extract_features_with_timeout(self, audio_path: str, force_reextract: bool = False) -> tuple:
        """Extract features with timeout in the same process to avoid pickle issues."""
        try:
            # Use threading with timeout instead of multiprocessing
            import threading
            import queue
            import time
            
            result_queue = queue.Queue()
            exception_queue = queue.Queue()
            timeout_occurred = threading.Event()
            
            def extract_worker():
                analyzer = None
                try:
                    # Check if timeout occurred before starting
                    if timeout_occurred.is_set():
                        logger.debug(f"Timeout occurred before worker started for {os.path.basename(audio_path)}")
                        return
                    
                    # Create a thread-safe AudioAnalyzer instance in the worker thread
                    # This ensures SQLite connections are thread-specific
                    cache_file = os.getenv('CACHE_FILE', '/app/cache/audio_analysis.db')
                    library = os.getenv('HOST_LIBRARY_PATH', '/root/music/library')
                    music = os.getenv('MUSIC_PATH', '/music')
                    
                    # Use thread-safe analyzer
                    analyzer = ThreadSafeAudioAnalyzer(cache_file, library, music)
                    result = analyzer.extract_features(audio_path, force_reextract=force_reextract)
                    
                    # Only put result if timeout hasn't occurred
                    if not timeout_occurred.is_set():
                        result_queue.put(result)
                    else:
                        logger.debug(f"Timeout occurred during processing, discarding result for {os.path.basename(audio_path)}")
                        # Don't write to database if timeout occurred
                        if result and len(result) >= 3:
                            # Mark as failed in database to prevent partial writes
                            try:
                                file_info = analyzer._get_file_info(audio_path)
                                analyzer._mark_failed(file_info)
                                logger.debug(f"Marked timed-out file as failed: {os.path.basename(audio_path)}")
                            except Exception as e:
                                logger.debug(f"Could not mark timed-out file as failed: {e}")
                except Exception as e:
                    if not timeout_occurred.is_set():
                        exception_queue.put(e)
                finally:
                    # Clean up analyzer resources
                    if analyzer:
                        try:
                            analyzer.cleanup()
                        except:
                            pass
            
            # Start worker thread
            worker_thread = threading.Thread(target=extract_worker, daemon=True)
            worker_thread.start()
            
            # Wait for result with timeout
            try:
                result = result_queue.get(timeout=self.timeout_seconds)
                return result
            except queue.Empty:
                # Set timeout flag to prevent worker from putting results
                timeout_occurred.set()
                logger.error(f"Large file processing timed out after {self.timeout_seconds}s: {os.path.basename(audio_path)}")
                
                # Wait a bit for worker to finish cleanup
                worker_thread.join(timeout=2.0)
                
                # Double-check that no result was put in the queue during the timeout
                try:
                    # Try to get any result that might have been put during timeout
                    result = result_queue.get_nowait()
                    logger.warning(f"Received late result after timeout for {os.path.basename(audio_path)}, discarding")
                except queue.Empty:
                    pass  # No late result, which is expected
                
                return None, False, None
            except Exception as e:
                timeout_occurred.set()
                if not exception_queue.empty():
                    exc = exception_queue.get()
                    logger.error(f"Large file processing failed: {os.path.basename(audio_path)} - {str(exc)}")
                return None, False, None
                
        except Exception as e:
            logger.error(f"Large file processing failed: {os.path.basename(audio_path)} - {str(e)}")
            return None, False, None
    
    def process_large_file(self, audio_path: str, force_reextract: bool = False, rss_limit_gb: float = 6.0) -> tuple:
        """Process a large file with timeout and memory monitoring using a subprocess."""
        import threading
        import time
        import queue
        file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
        logger.info(f"Processing large file ({file_size_mb:.1f}MB) with timeout: {os.path.basename(audio_path)}")
        log_detailed_memory_info("BEFORE large file processing")
        abort_event = threading.Event()
        memory_high_event = threading.Event()
        memory_high_start = [None]
        result_queue = multiprocessing.Queue()
        def memory_monitor(proc):
            logger.info(f"Starting memory monitor for large file: {os.path.basename(audio_path)}")
            system_mem_warning_logged = [False]
            last_warning_time = [0]
            warning_interval = 60  # Only log warnings every 60 seconds
            while proc.is_alive():
                is_over_limit, status_msg = check_memory_against_limit(user_limit_gb=rss_limit_gb, user_limit_percent=80.0)
                is_rss_over, rss_msg = check_total_python_rss_limit(rss_limit_gb=rss_limit_gb)
                total_rss_gb = get_total_python_rss_gb()
                rss_threshold = 0.8 * rss_limit_gb
                current_time = time.time()
                
                if is_rss_over:
                    if not memory_high_event.is_set():
                        memory_high_event.set()
                        memory_high_start[0] = time.time()
                        logger.warning(f"Total Python RSS limit exceeded while analyzing {os.path.basename(audio_path)}: {rss_msg}. Starting 20s timer.")
                        log_detailed_memory_info("MEMORY HIGH")
                    elif time.time() - memory_high_start[0] > 20:
                        logger.error(f"Total Python RSS limit exceeded for >20s. Terminating analysis of {os.path.basename(audio_path)}: {rss_msg}")
                        log_detailed_memory_info("ABORTING DUE TO MEMORY")
                        abort_event.set()
                        proc.terminate()
                        break
                else:
                    # Only warn about system/cgroup memory if RSS is also near the limit
                    if is_over_limit and total_rss_gb > rss_threshold:
                        if not system_mem_warning_logged[0] or (current_time - last_warning_time[0] > warning_interval):
                            logger.warning(f"RSS near limit ({total_rss_gb:.2f}GB/{rss_limit_gb:.2f}GB) and system memory high: {status_msg}")
                            system_mem_warning_logged[0] = True
                            last_warning_time[0] = current_time
                    elif system_mem_warning_logged[0] and total_rss_gb < (rss_threshold * 0.9):  # Only clear when RSS drops significantly
                        logger.info("Memory warning cleared - RSS and system memory back to normal levels.")
                        system_mem_warning_logged[0] = False
                        last_warning_time[0] = 0  # Reset timer when warning clears
                    if memory_high_event.is_set():
                        logger.info(f"Memory usage dropped below limits during analysis of {os.path.basename(audio_path)}. Resetting timer.")
                        memory_high_event.clear()
                        memory_high_start[0] = None
                time.sleep(1)
        proc = multiprocessing.Process(target=globals()['analyze_file_worker'], args=(audio_path, force_reextract, result_queue))
        proc.start()
        monitor_thread = threading.Thread(target=memory_monitor, args=(proc,), daemon=True)
        monitor_thread.start()
        result = None
        try:
            proc.join(timeout=self.timeout_seconds)
            if proc.is_alive():
                logger.error(f"Large file processing timed out after {self.timeout_seconds}s: {os.path.basename(audio_path)}. Terminating process.")
                proc.terminate()
                proc.join(timeout=2.0)
                return None, False, None
            if abort_event.is_set():
                logger.error(f"Aborted large file analysis due to sustained high memory: {os.path.basename(audio_path)}")
                return None, False, None
            try:
                result = result_queue.get_nowait()
            except queue.Empty:
                logger.error(f"No result returned from analysis subprocess for {os.path.basename(audio_path)}")
                return None, False, None
            if result and result[0] and result[1]:
                logger.info(f"Large file processing completed: {os.path.basename(audio_path)}")
            log_detailed_memory_info("AFTER large file processing")
            return result
        except Exception as e:
            logger.error(f"Large file processing failed: {os.path.basename(audio_path)} - {str(e)}")
            return None, False, None
        finally:
            abort_event.set()
            monitor_thread.join(timeout=2.0)


class SequentialProcessor:
    """Sequential processor for audio analysis (single-threaded)."""

    def __init__(self, audio_analyzer: AudioAnalyzer = None, rss_limit_gb: float = 6.0) -> None:
        self.failed_files: List[str] = []
        self.skipped_files_due_to_memory: List[str] = []  # Track skipped files
        self.audio_analyzer = audio_analyzer
        self.large_file_processor = LargeFileProcessor()
        self.memory_monitor = MemoryMonitor()
        self.rss_limit_gb = rss_limit_gb

    def process(self, file_list: List[str], workers: int = None, force_reextract: bool = False) -> iter:
        """Process a list of files sequentially.

        Args:
            file_list (List[str]): List of file paths.
            workers (int, optional): Ignored for sequential processing.
            force_reextract (bool, optional): If True, bypass the cache for all files.

        Yields:
            tuple: (features, filepath, db_write_success) for each file.
        """
        yield from self._process_sequential(file_list, force_reextract=force_reextract)

    def _process_sequential(self, file_list: List[str], force_reextract: bool = False) -> iter:
        """Internal generator for sequential processing."""
        import essentia
        # Essentia logging is now handled in main playlista script

        for i, filepath in enumerate(file_list):
            try:
                logger.debug(
                    f"SEQUENTIAL: Processing file {i+1}/{len(file_list)}: {filepath}")

                # Log file size before processing
                try:
                    file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
                except Exception as e:
                    file_size_mb = 0
                logger.info(f"Preparing to analyze file: {os.path.basename(filepath)} ({file_size_mb:.1f}MB)")

                # Remove/relax the pre-check for memory before processing each file
                # (Let the memory monitor during processing handle abort/skip)

                # Use file discovery to check if file should be excluded
                from .file_discovery import FileDiscovery
                file_discovery = FileDiscovery()
                if file_discovery._is_in_excluded_directory(filepath):
                    logger.warning(
                        f"Skipping file in excluded directory: {filepath}")
                    yield None, filepath, False
                    continue

                # Check if this is a large file that should be processed separately
                try:
                    threshold_mb = int(os.getenv('LARGE_FILE_THRESHOLD', '50'))
                    is_large_file = file_size_mb > threshold_mb
                except:
                    is_large_file = False

                if is_large_file:
                    logger.info(f"Large file detected ({file_size_mb:.1f}MB), using separate process: {os.path.basename(filepath)}")
                    max_retries = 2
                    for attempt in range(max_retries):
                        features, db_write_success, file_hash = self.large_file_processor.process_large_file(
                            filepath, force_reextract=force_reextract, rss_limit_gb=self.rss_limit_gb)
                        if features is not None and db_write_success:
                            break  # Success!
                        else:
                            logger.warning(f"Large file analysis failed or hung for {filepath}, retrying ({attempt+1}/{max_retries})...")
                    else:
                        logger.error(f"Large file analysis failed after {max_retries} attempts for {filepath}. Skipping file.")
                        self.failed_files.append(filepath)
                        yield None, filepath, False
                        continue
                else:
                    # Use the provided audio_analyzer or create a new one
                    logger.debug(
                        f"SEQUENTIAL: Calling extract_features for: {filepath}")
                    if self.audio_analyzer:
                        analyzer = self.audio_analyzer
                    else:
                        analyzer = AudioAnalyzer()
                    # Log that we're starting the extraction
                    logger.info(f"SEQUENTIAL: Starting feature extraction for {os.path.basename(filepath)}")
                    features, db_write_success, file_hash = analyzer.extract_features(
                        filepath, force_reextract=force_reextract)

                # Log memory after processing
                usage_percent_after, used_gb_after, available_gb_after = self.memory_monitor.get_memory_usage()
                logger.info(f"Memory after processing: {usage_percent_after:.1f}% used, {used_gb_after:.1f}GB used, {available_gb_after:.1f}GB available")

                logger.debug(
                    f"SEQUENTIAL: extract_features result - features: {features is not None}, db_write: {db_write_success}")

                if features and db_write_success:
                    logger.debug(f"SEQUENTIAL: Success for {filepath}")
                    yield features, filepath, db_write_success
                else:
                    logger.warning(
                        f"SEQUENTIAL: Failed for {filepath} - features: {features is not None}, db_write: {db_write_success}")
                    self.failed_files.append(filepath)
                    yield None, filepath, False

            except Exception as e:
                self.failed_files.append(filepath)
                logger.error(f"Error processing {filepath}: {str(e)}")
                yield None, filepath, False
            
            # Note: Interrupt checking is handled in the main analysis loop
            # This allows the current file to complete but stops the next one
            
            # Force cleanup after each file to prevent memory buildup
            gc.collect()

        # At the end, log a summary of skipped files due to memory
        if self.skipped_files_due_to_memory:
            logger.warning(f"Summary: {len(self.skipped_files_due_to_memory)} files were skipped due to low memory:")
            for skipped in self.skipped_files_due_to_memory:
                logger.warning(f"  Skipped: {skipped}")
