from tqdm import tqdm
import logging
from typing import List
import os
import time
import traceback
from typing import List, Dict, Any, Optional
from .feature_extractor import AudioAnalyzer
from utils.path_converter import PathConverter
import multiprocessing as mp
import signal
import psutil
import threading
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FutureTimeoutError
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


class LargeFileProcessor:
    """Handles large files in a separate process with timeout and memory monitoring."""
    
    def __init__(self, timeout_seconds: int = 600, memory_threshold_percent: int = 85):
        self.timeout_seconds = timeout_seconds
        self.memory_monitor = MemoryMonitor(memory_threshold_percent)
        self._stop_processing = threading.Event()
    
    def _extract_features_in_process(self, audio_path: str, force_reextract: bool = False) -> tuple:
        """Extract features in a separate process to avoid blocking the main process."""
        try:
            # Create a new AudioAnalyzer instance in the subprocess
            analyzer = AudioAnalyzer()
            result = analyzer.extract_features(audio_path, force_reextract=force_reextract)
            return result
        except Exception as e:
            logger.error(f"Process extraction failed for {audio_path}: {str(e)}")
            return None, False, None
    
    def _monitor_memory_and_kill_if_needed(self, process, timeout):
        """Monitor memory usage and kill process if memory is exhausted."""
        start_time = time.time()
        while process.is_alive() and (time.time() - start_time) < timeout:
            if self.memory_monitor.is_memory_critical():
                logger.error(f"Memory usage critical during large file processing, terminating process")
                try:
                    process.terminate()
                    process.wait(timeout=10)  # Wait for termination
                except:
                    process.kill()  # Force kill if terminate fails
                return False
            time.sleep(5)  # Check every 5 seconds
        return True
    
    def process_large_file(self, audio_path: str, force_reextract: bool = False) -> tuple:
        """Process a large file in a separate process with timeout and memory monitoring."""
        file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
        logger.info(f"Processing large file ({file_size_mb:.1f}MB) in separate process: {os.path.basename(audio_path)}")
        
        # Check memory before starting
        usage_percent, used_gb, available_gb = self.memory_monitor.get_memory_usage()
        logger.info(f"Memory before processing: {usage_percent:.1f}% ({used_gb:.1f}GB used, {available_gb:.1f}GB available)")
        
        if self.memory_monitor.is_memory_critical():
            logger.error(f"Memory usage too high ({usage_percent:.1f}%) to process large file safely")
            return None, False, None
        
        # Adjust timeout based on file size and available memory
        if file_size_mb > 200:
            timeout = 900  # 15 minutes for very large files
        elif file_size_mb > 100:
            timeout = 600  # 10 minutes for large files
        else:
            timeout = 300  # 5 minutes for medium files
        
        # Reduce timeout if memory is limited
        if available_gb < 2.0:  # Less than 2GB available
            timeout = min(timeout, 180)  # Max 3 minutes
            logger.warning(f"Limited memory available ({available_gb:.1f}GB), reducing timeout to {timeout}s")
        
        try:
            with ProcessPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self._extract_features_in_process, audio_path, force_reextract)
                
                # Monitor the process with memory checks
                start_time = time.time()
                while not future.done() and (time.time() - start_time) < timeout:
                    # Check memory every 10 seconds
                    if (time.time() - start_time) % 10 < 1:
                        if self.memory_monitor.is_memory_critical():
                            logger.error(f"Memory usage critical during processing, cancelling future")
                            future.cancel()
                            self.memory_monitor.force_cleanup()
                            return None, False, None
                    
                    time.sleep(1)
                
                if future.done():
                    result = future.result(timeout=1)  # Get result with short timeout
                    logger.info(f"Large file processing completed: {os.path.basename(audio_path)}")
                    return result
                else:
                    logger.error(f"Large file processing timed out after {timeout}s: {os.path.basename(audio_path)}")
                    future.cancel()
                    return None, False, None
                    
        except FutureTimeoutError:
            logger.error(f"Large file processing timed out after {timeout}s: {os.path.basename(audio_path)}")
            # Kill any remaining processes
            self._kill_python_processes()
            return None, False, None
        except Exception as e:
            logger.error(f"Large file processing failed: {os.path.basename(audio_path)} - {str(e)}")
            return None, False, None
    
    def _kill_python_processes(self):
        """Kill any remaining Python processes that might be hanging."""
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if 'python' in proc.info['name'].lower():
                        # Check if it's our process
                        cmdline = proc.info.get('cmdline', [])
                        if any('playlista' in arg for arg in cmdline):
                            logger.warning(f"Killing hanging Python process: {proc.info['name']} (PID: {proc.info['pid']})")
                            proc.terminate()
                            proc.wait(timeout=5)
                        elif any('essentia' in arg.lower() for arg in cmdline):
                            logger.warning(f"Killing hanging Essentia process: {proc.info['name']} (PID: {proc.info['pid']})")
                            proc.terminate()
                            proc.wait(timeout=5)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        except Exception as e:
            logger.error(f"Error killing processes: {str(e)}")


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
            
            # Force cleanup after each file to prevent memory buildup
            gc.collect()
