"""
Parallel processing infrastructure for audio analysis.
Designed for Docker environments with memory monitoring and timeout handling.
"""

import multiprocessing as mp
import os
import sys
import time
import logging
import threading
import signal
import psutil
from typing import List, Optional, Dict, Any, Callable, Tuple
from pathlib import Path
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, TimeoutError
import queue

from shared.exceptions import ProcessingError, TimeoutError as CustomTimeoutError
from shared.config.settings import ProcessingConfig, MemoryConfig
from shared.utils import get_file_size_mb


@dataclass
class ProcessingResult:
    """Result of a processing operation."""
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    processing_time: float = 0.0
    memory_usage_mb: float = 0.0


class MemoryMonitor:
    """Monitor system memory in Docker environment."""
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._stop_monitoring = threading.Event()
    
    def get_memory_usage(self) -> Tuple[float, float, float]:
        """Get current memory usage (percent, used_gb, available_gb)."""
        try:
            memory = psutil.virtual_memory()
            percent = memory.percent
            used_gb = memory.used / (1024**3)
            available_gb = memory.available / (1024**3)
            total_gb = memory.total / (1024**3)
            
            self.logger.debug(f"Memory calculation details:")
            self.logger.debug(f"  - Total memory: {total_gb:.1f}GB")
            self.logger.debug(f"  - Used memory: {used_gb:.1f}GB")
            self.logger.debug(f"  - Available memory: {available_gb:.1f}GB")
            self.logger.debug(f"  - Usage percentage: {percent:.1f}%")
            self.logger.debug(f"  - Calculation: used({used_gb:.1f}GB) / total({total_gb:.1f}GB) * 100 = {percent:.1f}%")
            
            return percent, used_gb, available_gb
        except Exception as e:
            self.logger.warning(f"Could not get memory info: {e}")
            return 0.0, 0.0, 0.0
    
    def is_memory_critical(self) -> bool:
        """Check if memory usage is critical."""
        usage_percent, used_gb, available_gb = self.get_memory_usage()
        return usage_percent > (self.config.memory_pressure_threshold * 100)
    
    def get_optimal_worker_count(self, max_workers: int, memory_limit_str: str = None) -> int:
        """Calculate optimal worker count based on available memory."""
        try:
            usage_percent, used_gb, available_gb = self.get_memory_usage()
            
            self.logger.info(f"Memory-aware worker calculation:")
            self.logger.info(f"  - Memory usage: {usage_percent:.1f}% ({used_gb:.1f}GB used, {available_gb:.1f}GB available)")
            self.logger.info(f"  - Max workers requested: {max_workers}")
            self.logger.info(f"  - Memory limit string: {memory_limit_str}")
            
            # If memory is critical, reduce workers
            if self.is_memory_critical():
                reduced_workers = max(1, max_workers // 2)
                self.logger.warning(f"Memory usage critical ({usage_percent:.1f}%), reducing workers from {max_workers} to {reduced_workers}")
                return reduced_workers
            
            # If memory limit is specified, calculate based on that
            if memory_limit_str:
                memory_limit_gb = self._parse_memory_limit(memory_limit_str)
                if memory_limit_gb > 0:
                    # Estimate memory per worker (rough estimate)
                    memory_per_worker_gb = 0.5  # Conservative estimate
                    optimal_workers = int(available_gb / memory_per_worker_gb)
                    final_workers = max(1, min(optimal_workers, max_workers))
                    self.logger.info(f"Memory limit calculation:")
                    self.logger.info(f"  - Memory limit: {memory_limit_gb:.1f}GB")
                    self.logger.info(f"  - Memory per worker estimate: {memory_per_worker_gb:.1f}GB")
                    self.logger.info(f"  - Optimal workers (available/memory_per_worker): {optimal_workers}")
                    self.logger.info(f"  - Final workers (capped at max): {final_workers}")
                    return final_workers
            
            # Default: use half of CPU count for memory safety
            default_workers = max(1, max_workers // 2)
            self.logger.info(f"Using default calculation (max_workers // 2): {default_workers}")
            return default_workers
            
        except Exception as e:
            self.logger.warning(f"Could not determine memory-aware worker count: {e}")
            fallback_workers = max(1, min(max_workers, mp.cpu_count()))
            self.logger.info(f"Using fallback worker count: {fallback_workers}")
            return fallback_workers
    
    def _parse_memory_limit(self, memory_limit_str: str) -> float:
        """Parse memory limit string (e.g., '2GB', '512MB')."""
        try:
            memory_limit_str = memory_limit_str.upper()
            if memory_limit_str.endswith('GB'):
                return float(memory_limit_str[:-2])
            elif memory_limit_str.endswith('MB'):
                return float(memory_limit_str[:-2]) / 1024
            else:
                return float(memory_limit_str)
        except Exception:
            return 0.0


class TimeoutHandler:
    """Handle timeouts for processing operations."""
    
    def __init__(self, timeout_seconds: int = 300):
        self.timeout_seconds = timeout_seconds
        self.logger = logging.getLogger(__name__)
    
    def timeout_handler(self, signum, frame):
        """Signal handler for timeout."""
        raise CustomTimeoutError(f"Processing timed out after {self.timeout_seconds} seconds")
    
    def process_with_timeout(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with timeout."""
        # Set up signal handler for timeout
        old_handler = signal.signal(signal.SIGALRM, self.timeout_handler)
        signal.alarm(self.timeout_seconds)
        
        try:
            result = func(*args, **kwargs)
            signal.alarm(0)  # Cancel alarm
            return result
        except CustomTimeoutError:
            raise
        except Exception as e:
            signal.alarm(0)  # Cancel alarm
            from shared.exceptions.processing import ProcessingError, ErrorSeverity, ErrorCategory
            raise ProcessingError(
                error_type="processing_failed",
                message=f"Processing failed: {e}",
                severity=ErrorSeverity.HIGH,
                category=ErrorCategory.PROCESSING
            )
        finally:
            signal.signal(signal.SIGALRM, old_handler)


class LargeFileProcessor:
    """Handle large files with special processing."""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.timeout_handler = TimeoutHandler(config.file_timeout_minutes * 60)
    
    def is_large_file(self, file_path: Path) -> bool:
        """Check if file is considered large."""
        file_size_mb = get_file_size_mb(file_path)
        if file_size_mb is None:
            return False
        return file_size_mb > self.config.large_file_threshold_mb
    
    def is_very_large_file(self, file_path: Path) -> bool:
        """Check if file is very large."""
        file_size_mb = get_file_size_mb(file_path)
        if file_size_mb is None:
            return False
        return file_size_mb > self.config.very_large_file_threshold_mb
    
    def process_large_file(self, file_path: Path, processor_func: Callable, *args, **kwargs) -> ProcessingResult:
        """Process large file with special handling."""
        start_time = time.time()
        
        # Get file size and initial memory
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        initial_memory = self._get_memory_usage_mb()
        
        self.logger.info(f"Processing large file: {file_path.name} ({file_size_mb:.1f}MB)")
        self.logger.info(f"  - Initial process memory: {initial_memory:.1f}MB")
        self.logger.info(f"  - Timeout: {self.timeout_handler.timeout_seconds}s")
        
        try:
            # Use timeout handler for large files
            result = self.timeout_handler.process_with_timeout(
                processor_func, file_path, *args, **kwargs
            )
            
            processing_time = time.time() - start_time
            final_memory = self._get_memory_usage_mb()
            memory_delta = final_memory - initial_memory
            
            self.logger.info(f"Large file processing completed: {file_path.name}")
            self.logger.info(f"  - Processing time: {processing_time:.1f}s")
            self.logger.info(f"  - Process memory: {initial_memory:.1f}MB → {final_memory:.1f}MB (Δ{memory_delta:+.1f}MB)")
            self.logger.info(f"  - Memory efficiency: {memory_delta/file_size_mb:.2f}MB memory per MB of file")
            
            return ProcessingResult(
                success=True,
                data=result,
                processing_time=processing_time,
                memory_usage_mb=final_memory
            )
            
        except CustomTimeoutError as e:
            processing_time = time.time() - start_time
            final_memory = self._get_memory_usage_mb()
            memory_delta = final_memory - initial_memory
            
            self.logger.error(f"Large file processing timed out: {file_path.name}")
            self.logger.error(f"  - Processing time: {processing_time:.1f}s")
            self.logger.error(f"  - Process memory: {initial_memory:.1f}MB → {final_memory:.1f}MB (Δ{memory_delta:+.1f}MB)")
            self.logger.error(f"  - Timeout limit: {self.timeout_handler.timeout_seconds}s")
            
            return ProcessingResult(
                success=False,
                error=f"Timeout: {e}",
                processing_time=processing_time
            )
        except Exception as e:
            processing_time = time.time() - start_time
            final_memory = self._get_memory_usage_mb()
            memory_delta = final_memory - initial_memory
            
            self.logger.error(f"Large file processing failed: {file_path.name} - {e}")
            self.logger.error(f"  - Processing time: {processing_time:.1f}s")
            self.logger.error(f"  - Process memory: {initial_memory:.1f}MB → {final_memory:.1f}MB (Δ{memory_delta:+.1f}MB)")
            
            return ProcessingResult(
                success=False,
                error=str(e),
                processing_time=processing_time
            )
    
    def _get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            rss_bytes = process.memory_info().rss
            rss_mb = rss_bytes / (1024 * 1024)
            
            self.logger.debug(f"Process memory calculation (LargeFileProcessor):")
            self.logger.debug(f"  - RSS bytes: {rss_bytes:,}")
            self.logger.debug(f"  - RSS MB: {rss_mb:.1f}MB")
            self.logger.debug(f"  - Calculation: {rss_bytes:,} bytes / (1024 * 1024) = {rss_mb:.1f}MB")
            
            return rss_mb
        except Exception as e:
            self.logger.warning(f"Could not get process memory info: {e}")
            return 0.0


class ParallelProcessor:
    """Parallel processing with memory awareness and Docker optimization."""
    
    def __init__(self, config: ProcessingConfig, memory_config: MemoryConfig):
        self.config = config
        self.memory_config = memory_config
        # Don't create MemoryMonitor here - it has unpickleable threading.Event
        self.large_file_processor = LargeFileProcessor(config)
        self.logger = logging.getLogger(__name__)
        
        # Docker-specific optimizations
        self._setup_docker_environment()
    
    def _setup_docker_environment(self):
        """Setup Docker-specific environment optimizations."""
        # Set multiprocessing start method for Docker
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass  # Already set
        
        # Set environment variables for better Docker performance
        os.environ['OMP_NUM_THREADS'] = '1'  # Prevent OpenMP conflicts
        os.environ['MKL_NUM_THREADS'] = '1'  # Prevent MKL conflicts
        
        self.logger.info("Docker environment optimized for parallel processing")
    
    def get_optimal_worker_count(self, max_workers: Optional[int] = None) -> int:
        """Get optimal number of workers based on memory and CPU."""
        if max_workers is None:
            max_workers = self.config.max_workers
        
        # Create MemoryMonitor when needed (not during initialization)
        memory_monitor = MemoryMonitor(self.memory_config)
        
        # Get memory-aware worker count
        optimal_workers = memory_monitor.get_optimal_worker_count(
            max_workers, 
            memory_limit_str=os.getenv('MEMORY_LIMIT')
        )
        
        self.logger.info(f"Using {optimal_workers} workers (max: {max_workers})")
        return optimal_workers
    
    def process_files_parallel(
        self, 
        file_paths: List[Path], 
        processor_func: Callable,
        max_workers: Optional[int] = None,
        timeout_minutes: Optional[int] = None,
        **kwargs
    ) -> List[ProcessingResult]:
        """Process files in parallel with memory monitoring."""
        
        if not file_paths:
            self.logger.info("No files to process")
            return []
        
        self.logger.info(f"Starting parallel processing decision:")
        self.logger.info(f"  - Total files: {len(file_paths)}")
        self.logger.info(f"  - Max workers requested: {max_workers}")
        self.logger.info(f"  - Timeout requested: {timeout_minutes} minutes")
        
        # Get optimal worker count
        workers = self.get_optimal_worker_count(max_workers)
        
        # Set timeout
        if timeout_minutes is None:
            timeout_minutes = self.config.batch_timeout_minutes
            self.logger.info(f"  - Using default timeout: {timeout_minutes} minutes")
        else:
            self.logger.info(f"  - Using specified timeout: {timeout_minutes} minutes")
        
        self.logger.info(f"Parallel processing configuration:")
        self.logger.info(f"  - Workers: {workers}")
        self.logger.info(f"  - Timeout: {timeout_minutes} minutes")
        self.logger.info(f"  - Batch size multiplier: {self.config.batch_size_multiplier}")
        
        results = []
        
        # Split files into batches for better memory management
        batch_size = min(len(file_paths), workers * self.config.batch_size_multiplier)
        batches = [file_paths[i:i + batch_size] for i in range(0, len(file_paths), batch_size)]
        
        self.logger.info(f"Batch configuration:")
        self.logger.info(f"  - Calculated batch size: {batch_size} files")
        self.logger.info(f"  - Total batches: {len(batches)}")
        self.logger.info(f"  - Reason: workers({workers}) * multiplier({self.config.batch_size_multiplier}) = {workers * self.config.batch_size_multiplier}")
        
        for batch_idx, batch in enumerate(batches):
            self.logger.info(f"Processing batch {batch_idx + 1}/{len(batches)}:")
            self.logger.info(f"  - Batch size: {len(batch)} files")
            self.logger.info(f"  - Files: {[f.name for f in batch[:3]]}{'...' if len(batch) > 3 else ''}")
            
            batch_start_time = time.time()
            batch_results = self._process_batch_parallel(
                batch, processor_func, workers, timeout_minutes, **kwargs
            )
            batch_time = time.time() - batch_start_time
            
            successful_in_batch = sum(1 for r in batch_results if r.success)
            failed_in_batch = len(batch_results) - successful_in_batch
            
            self.logger.info(f"Batch {batch_idx + 1} completed:")
            self.logger.info(f"  - Time: {batch_time:.1f}s")
            self.logger.info(f"  - Successful: {successful_in_batch}")
            self.logger.info(f"  - Failed: {failed_in_batch}")
            
            results.extend(batch_results)
            
            # Check memory after each batch
            memory_monitor = MemoryMonitor(self.memory_config)
            memory_usage = memory_monitor.get_memory_usage()
            self.logger.info(f"  - Memory after batch: {memory_usage[0]:.1f}%")
            
            if memory_monitor.is_memory_critical():
                self.logger.warning(f"Memory usage critical ({memory_usage[0]:.1f}%), pausing {self.memory_config.memory_pressure_pause_seconds}s before next batch")
                time.sleep(self.memory_config.memory_pressure_pause_seconds)
        
        total_successful = sum(1 for r in results if r.success)
        total_failed = len(results) - total_successful
        
        self.logger.info(f"Parallel processing completed:")
        self.logger.info(f"  - Total files: {len(file_paths)}")
        self.logger.info(f"  - Successful: {total_successful}")
        self.logger.info(f"  - Failed: {total_failed}")
        self.logger.info(f"  - Workers used: {workers}")
        self.logger.info(f"  - Batches processed: {len(batches)}")
        
        return results
    
    def _process_batch_parallel(
        self, 
        file_paths: List[Path], 
        processor_func: Callable,
        workers: int,
        timeout_minutes: int,
        **kwargs
    ) -> List[ProcessingResult]:
        """Process a batch of files in parallel."""
        
        self.logger.info(f"Starting batch parallel processing:")
        self.logger.info(f"  - Files in batch: {len(file_paths)}")
        self.logger.info(f"  - Workers: {workers}")
        self.logger.info(f"  - Timeout: {timeout_minutes} minutes")
        
        results = []
        successful_count = 0
        failed_count = 0
        timeout_count = 0
        
        with ProcessPoolExecutor(max_workers=workers) as executor:
            self.logger.info(f"Created ProcessPoolExecutor with {workers} workers")
            
            # Submit all tasks
            future_to_path = {
                executor.submit(self._process_single_file, path, processor_func, **kwargs): path
                for path in file_paths
            }
            
            self.logger.info(f"Submitted {len(future_to_path)} tasks to executor")
            
            # Collect results with timeout
            for i, future in enumerate(future_to_path):
                path = future_to_path[future]
                self.logger.debug(f"Collecting result {i+1}/{len(future_to_path)} for: {path.name}")
                
                try:
                    result = future.result(timeout=timeout_minutes * 60)
                    results.append(result)
                    
                    if result.success:
                        successful_count += 1
                        self.logger.debug(f"Success: {path.name} ({result.processing_time:.1f}s)")
                    else:
                        failed_count += 1
                        self.logger.warning(f"Failed: {path.name} - {result.error}")
                        
                except TimeoutError:
                    timeout_count += 1
                    self.logger.error(f"Processing timed out for: {path.name} (timeout: {timeout_minutes} minutes)")
                    results.append(ProcessingResult(
                        success=False,
                        error="Processing timed out",
                        processing_time=timeout_minutes * 60
                    ))
                except Exception as e:
                    failed_count += 1
                    self.logger.error(f"Processing failed for {path.name}: {e}")
                    results.append(ProcessingResult(
                        success=False,
                        error=str(e)
                    ))
        
        self.logger.info(f"Batch parallel processing completed:")
        self.logger.info(f"  - Total files: {len(file_paths)}")
        self.logger.info(f"  - Successful: {successful_count}")
        self.logger.info(f"  - Failed: {failed_count}")
        self.logger.info(f"  - Timeouts: {timeout_count}")
        self.logger.info(f"  - Workers used: {workers}")
        
        return results
    
    def _process_single_file(
        self, 
        file_path: Path, 
        processor_func: Callable,
        **kwargs
    ) -> ProcessingResult:
        """Process a single file with appropriate handling."""
        
        start_time = time.time()
        
        try:
            # Check if this is a large file
            if self.large_file_processor.is_large_file(file_path):
                return self.large_file_processor.process_large_file(
                    file_path, processor_func, **kwargs
                )
            
            # Regular processing
            result = processor_func(file_path, **kwargs)
            
            processing_time = time.time() - start_time
            memory_usage = self._get_memory_usage_mb()
            
            return ProcessingResult(
                success=True,
                data=result,
                processing_time=processing_time,
                memory_usage_mb=memory_usage
            )
            
        except Exception as e:
            self.logger.error(f"Processing failed for {file_path}: {e}")
            return ProcessingResult(
                success=False,
                error=str(e),
                processing_time=time.time() - start_time
            )
    
    def _get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            rss_bytes = process.memory_info().rss
            rss_mb = rss_bytes / (1024 * 1024)
            
            self.logger.debug(f"Process memory calculation (ParallelProcessor):")
            self.logger.debug(f"  - RSS bytes: {rss_bytes:,}")
            self.logger.debug(f"  - RSS MB: {rss_mb:.1f}MB")
            self.logger.debug(f"  - Calculation: {rss_bytes:,} bytes / (1024 * 1024) = {rss_mb:.1f}MB")
            
            return rss_mb
        except Exception as e:
            self.logger.warning(f"Could not get process memory info: {e}")
            return 0.0


class SequentialProcessor:
    """Sequential processing for debugging and low-memory scenarios."""
    
    def __init__(self, config: ProcessingConfig, memory_config: MemoryConfig):
        self.config = config
        self.memory_config = memory_config
        self.memory_monitor = MemoryMonitor(memory_config)
        self.large_file_processor = LargeFileProcessor(config)
        self.logger = logging.getLogger(__name__)
    
    def process_files_sequential(
        self, 
        file_paths: List[Path], 
        processor_func: Callable,
        **kwargs
    ) -> List[ProcessingResult]:
        """Process files sequentially with memory monitoring."""
        
        if not file_paths:
            self.logger.info("No files to process sequentially")
            return []
        
        self.logger.info(f"Starting sequential processing of {len(file_paths)} files")
        self.logger.info(f"Memory monitoring enabled: {self.memory_config.memory_aware}")
        self.logger.info(f"Memory pressure threshold: {self.memory_config.memory_pressure_threshold * 100:.1f}%")
        
        results = []
        successful_count = 0
        failed_count = 0
        
        for i, file_path in enumerate(file_paths):
            # Get file size
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            self.logger.info(f"Processing file {i + 1}/{len(file_paths)}: {file_path.name} ({file_size_mb:.1f}MB)")
            
            # Check memory before each file
            memory_usage = self.memory_monitor.get_memory_usage()
            self.logger.info(f"Memory before processing: {memory_usage[0]:.1f}% ({memory_usage[1]:.1f}GB used, {memory_usage[2]:.1f}GB available)")
            
            if self.memory_monitor.is_memory_critical():
                self.logger.warning(f"Memory usage critical ({memory_usage[0]:.1f}%), pausing {self.memory_config.memory_pressure_pause_seconds}s")
                time.sleep(self.memory_config.memory_pressure_pause_seconds)
                
                # Check memory again after pause
                memory_after_pause = self.memory_monitor.get_memory_usage()
                self.logger.info(f"Memory after pause: {memory_after_pause[0]:.1f}% ({memory_after_pause[1]:.1f}GB used, {memory_after_pause[2]:.1f}GB available)")
            
            # Process the file
            start_time = time.time()
            result = self._process_single_file(file_path, processor_func, **kwargs)
            processing_time = time.time() - start_time
            
            # Check memory after processing
            memory_after = self.memory_monitor.get_memory_usage()
            self.logger.info(f"Memory after processing: {memory_after[0]:.1f}% ({memory_after[1]:.1f}GB used, {memory_after[2]:.1f}GB available)")
            self.logger.info(f"Processing time: {processing_time:.1f}s")
            
            if result.success:
                successful_count += 1
                self.logger.info(f"File completed successfully: {file_path.name}")
            else:
                failed_count += 1
                self.logger.warning(f"File failed: {file_path.name} - {result.error}")
            
            results.append(result)
            
            # Log progress every 10 files
            if (i + 1) % 10 == 0:
                self.logger.info(f"Progress: {i + 1}/{len(file_paths)} files processed ({successful_count} successful, {failed_count} failed)")
        
        self.logger.info(f"Sequential processing completed:")
        self.logger.info(f"  - Total files: {len(file_paths)}")
        self.logger.info(f"  - Successful: {successful_count}")
        self.logger.info(f"  - Failed: {failed_count}")
        
        return results
    
    def _process_single_file(
        self, 
        file_path: Path, 
        processor_func: Callable,
        **kwargs
    ) -> ProcessingResult:
        """Process a single file sequentially."""
        
        start_time = time.time()
        
        # Get initial memory state
        initial_memory = self._get_memory_usage_mb()
        system_memory = self.memory_monitor.get_memory_usage()
        self.logger.debug(f"Starting file processing: {file_path.name}")
        self.logger.debug(f"  - Initial process memory: {initial_memory:.1f}MB")
        self.logger.debug(f"  - Initial system memory: {system_memory[0]:.1f}% ({system_memory[1]:.1f}GB used)")
        
        try:
            # Check if this is a large file
            if self.large_file_processor.is_large_file(file_path):
                self.logger.info(f"Large file detected: {file_path.name} - using large file processor")
                return self.large_file_processor.process_large_file(
                    file_path, processor_func, **kwargs
                )
            
            # Regular processing
            self.logger.debug(f"Processing file with regular processor: {file_path.name}")
            result = processor_func(file_path, **kwargs)
            
            processing_time = time.time() - start_time
            final_memory = self._get_memory_usage_mb()
            final_system_memory = self.memory_monitor.get_memory_usage()
            
            memory_delta = final_memory - initial_memory
            system_memory_delta = final_system_memory[1] - system_memory[1]
            
            self.logger.info(f"File processing completed: {file_path.name}")
            self.logger.info(f"  - Processing time: {processing_time:.1f}s")
            self.logger.info(f"  - Process memory: {initial_memory:.1f}MB → {final_memory:.1f}MB (Δ{memory_delta:+.1f}MB)")
            self.logger.info(f"  - System memory: {system_memory[0]:.1f}% → {final_system_memory[0]:.1f}% (Δ{system_memory_delta:+.1f}GB)")
            
            return ProcessingResult(
                success=True,
                data=result,
                processing_time=processing_time,
                memory_usage_mb=final_memory
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            final_memory = self._get_memory_usage_mb()
            final_system_memory = self.memory_monitor.get_memory_usage()
            
            memory_delta = final_memory - initial_memory
            system_memory_delta = final_system_memory[1] - system_memory[1]
            
            self.logger.error(f"Processing failed for {file_path.name}: {e}")
            self.logger.error(f"  - Processing time: {processing_time:.1f}s")
            self.logger.error(f"  - Process memory: {initial_memory:.1f}MB → {final_memory:.1f}MB (Δ{memory_delta:+.1f}MB)")
            self.logger.error(f"  - System memory: {system_memory[0]:.1f}% → {final_system_memory[0]:.1f}% (Δ{system_memory_delta:+.1f}GB)")
            
            return ProcessingResult(
                success=False,
                error=str(e),
                processing_time=processing_time
            )
    
    def _get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            rss_bytes = process.memory_info().rss
            rss_mb = rss_bytes / (1024 * 1024)
            
            self.logger.debug(f"Process memory calculation (SequentialProcessor):")
            self.logger.debug(f"  - RSS bytes: {rss_bytes:,}")
            self.logger.debug(f"  - RSS MB: {rss_mb:.1f}MB")
            self.logger.debug(f"  - Calculation: {rss_bytes:,} bytes / (1024 * 1024) = {rss_mb:.1f}MB")
            
            return rss_mb
        except Exception as e:
            self.logger.warning(f"Could not get process memory info: {e}")
            return 0.0 