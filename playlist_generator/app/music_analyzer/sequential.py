from tqdm import tqdm
import logging
from typing import List
import os
import time
import traceback
from typing import List, Dict, Any, Optional
from .feature_extractor import AudioAnalyzer
from utils.path_converter import PathConverter
import psutil
import threading
import gc

logger = logging.getLogger(__name__)


class MemoryMonitor:
    """Monitor system memory and trigger cleanup when needed."""
    
    def __init__(self, memory_threshold_percent=85):
        self.memory_threshold_percent = memory_threshold_percent
        self._stop_monitoring = threading.Event()
    
    def get_memory_usage(self):
        """Get current memory usage percentage."""
        try:
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
        
        # Log memory after cleanup
        usage_percent, used_gb, available_gb = self.get_memory_usage()
        logger.info(f"Memory after cleanup: {usage_percent:.1f}% ({used_gb:.1f}GB used, {available_gb:.1f}GB available)")


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
    
    def process_large_file(self, audio_path: str, force_reextract: bool = False) -> tuple:
        """Process a large file with timeout and memory monitoring."""
        
        file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
        logger.info(f"Processing large file ({file_size_mb:.1f}MB) with timeout: {os.path.basename(audio_path)}")
        
        # Check memory before starting
        usage_percent, used_gb, available_gb = self.memory_monitor.get_memory_usage()
        logger.info(f"Memory before processing: {usage_percent:.1f}% ({used_gb:.1f}GB used, {available_gb:.1f}GB available)")
        
        if self.memory_monitor.is_memory_critical():
            logger.error(f"Memory usage too high ({usage_percent:.1f}%) to process large file safely")
            return None, False, None
        
        # Adjust timeout based on file size and available memory
        if file_size_mb > 200:
            timeout = 1200  # 20 minutes for very large files
        elif file_size_mb > 100:
            timeout = 900   # 15 minutes for large files
        else:
            timeout = 600   # 10 minutes for medium files
        
        # Reduce timeout if memory is limited
        if available_gb < 2.0:  # Less than 2GB available
            timeout = min(timeout, 300)  # Max 5 minutes
            logger.warning(f"Limited memory available ({available_gb:.1f}GB), reducing timeout to {timeout}s")
        
        # Set timeout for this processing session
        self.timeout_seconds = timeout
        
        try:
            # Use the threading-based approach to avoid pickle issues
            result = self._extract_features_with_timeout(audio_path, force_reextract)
            if result and result[0] and result[1]:
                logger.info(f"Large file processing completed: {os.path.basename(audio_path)}")
            
            # Check for interrupt AFTER processing the current file
            # This allows the current file to complete but stops the next one
            try:
                import signal
                # This will raise KeyboardInterrupt if Ctrl+C was pressed
                signal.signal(signal.SIGINT, signal.default_int_handler)
            except KeyboardInterrupt:
                logger.warning("Interrupt received after completing large file")
                print(f"\n🛑 Interrupt received! Completed {os.path.basename(audio_path)}, stopping processing...")
                # Return the current result but signal to stop processing more files
                return result
            
            return result
        except Exception as e:
            logger.error(f"Large file processing failed: {os.path.basename(audio_path)} - {str(e)}")
            return None, False, None


class SequentialProcessor:
    """Sequential processor for audio analysis (single-threaded)."""

    def __init__(self, audio_analyzer: AudioAnalyzer = None) -> None:
        self.failed_files: List[str] = []
        self.audio_analyzer = audio_analyzer
        self.large_file_processor = LargeFileProcessor()
        self.memory_monitor = MemoryMonitor()

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

                # Check memory before processing each file
                usage_percent, used_gb, available_gb = self.memory_monitor.get_memory_usage()
                if self.memory_monitor.is_memory_critical():
                    logger.warning(f"Memory usage critical ({usage_percent:.1f}%), forcing cleanup before processing")
                    self.memory_monitor.force_cleanup()
                    
                    # Check again after cleanup
                    usage_percent, used_gb, available_gb = self.memory_monitor.get_memory_usage()
                    if self.memory_monitor.is_memory_critical():
                        logger.error(f"Memory still critical after cleanup ({usage_percent:.1f}%), skipping file")
                        self.failed_files.append(filepath)
                        yield None, filepath, False
                        continue

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
                    file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
                    # Get threshold from environment or use default
                    threshold_mb = int(os.getenv('LARGE_FILE_THRESHOLD', '50'))
                    is_large_file = file_size_mb > threshold_mb
                except:
                    is_large_file = False
                
                if is_large_file:
                    logger.info(f"Large file detected ({file_size_mb:.1f}MB), using separate process: {os.path.basename(filepath)}")
                    features, db_write_success, file_hash = self.large_file_processor.process_large_file(
                        filepath, force_reextract=force_reextract)
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
            
            # Check for interrupt AFTER processing the current file
            # This allows the current file to complete but stops the next one
            try:
                import signal
                # This will raise KeyboardInterrupt if Ctrl+C was pressed
                signal.signal(signal.SIGINT, signal.default_int_handler)
            except KeyboardInterrupt:
                logger.warning("Interrupt received after completing file")
                print(f"\n🛑 Interrupt received! Completed {os.path.basename(filepath)}, stopping sequential processing...")
                return  # Exit the generator
            
            # Force cleanup after each file to prevent memory buildup
            gc.collect()
