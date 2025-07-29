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
            return memory.percent, memory.used / (1024**3), memory.available / (1024**3)
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
            
            # If memory is critical, reduce workers
            if self.is_memory_critical():
                self.logger.warning(f"Memory usage critical ({usage_percent:.1f}%), reducing workers")
                return max(1, max_workers // 2)
            
            # If memory limit is specified, calculate based on that
            if memory_limit_str:
                memory_limit_gb = self._parse_memory_limit(memory_limit_str)
                if memory_limit_gb > 0:
                    # Estimate memory per worker (rough estimate)
                    memory_per_worker_gb = 0.5  # Conservative estimate
                    optimal_workers = int(available_gb / memory_per_worker_gb)
                    return max(1, min(optimal_workers, max_workers))
            
            # Default: use half of CPU count for memory safety
            return max(1, max_workers // 2)
            
        except Exception as e:
            self.logger.warning(f"Could not determine memory-aware worker count: {e}")
            return max(1, min(max_workers, mp.cpu_count()))
    
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
            raise ProcessingError(f"Processing failed: {e}")
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
        return file_size_mb > self.config.large_file_threshold_mb
    
    def is_very_large_file(self, file_path: Path) -> bool:
        """Check if file is very large."""
        file_size_mb = get_file_size_mb(file_path)
        return file_size_mb > self.config.very_large_file_threshold_mb
    
    def process_large_file(self, file_path: Path, processor_func: Callable, *args, **kwargs) -> ProcessingResult:
        """Process large file with special handling."""
        start_time = time.time()
        
        try:
            self.logger.info(f"Processing large file: {file_path}")
            
            # Use timeout handler for large files
            result = self.timeout_handler.process_with_timeout(
                processor_func, file_path, *args, **kwargs
            )
            
            processing_time = time.time() - start_time
            memory_usage = self._get_memory_usage_mb()
            
            return ProcessingResult(
                success=True,
                data=result,
                processing_time=processing_time,
                memory_usage_mb=memory_usage
            )
            
        except CustomTimeoutError as e:
            self.logger.error(f"Large file processing timed out: {file_path}")
            return ProcessingResult(
                success=False,
                error=f"Timeout: {e}",
                processing_time=time.time() - start_time
            )
        except Exception as e:
            self.logger.error(f"Large file processing failed: {file_path} - {e}")
            return ProcessingResult(
                success=False,
                error=str(e),
                processing_time=time.time() - start_time
            )
    
    def _get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except Exception:
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
            return []
        
        # Get optimal worker count
        workers = self.get_optimal_worker_count(max_workers)
        
        # Set timeout
        if timeout_minutes is None:
            timeout_minutes = self.config.batch_timeout_minutes
        
        self.logger.info(f"Starting parallel processing of {len(file_paths)} files with {workers} workers")
        
        results = []
        
        # Split files into batches for better memory management
        batch_size = min(len(file_paths), workers * self.config.batch_size_multiplier)
        batches = [file_paths[i:i + batch_size] for i in range(0, len(file_paths), batch_size)]
        
        for batch_idx, batch in enumerate(batches):
            self.logger.info(f"Processing batch {batch_idx + 1}/{len(batches)} ({len(batch)} files)")
            
            batch_results = self._process_batch_parallel(
                batch, processor_func, workers, timeout_minutes, **kwargs
            )
            results.extend(batch_results)
            
            # Check memory after each batch
            memory_monitor = MemoryMonitor(self.memory_config)
            if memory_monitor.is_memory_critical():
                self.logger.warning("Memory usage critical, pausing before next batch")
                time.sleep(self.memory_config.memory_pressure_pause_seconds)
        
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
        
        results = []
        
        with ProcessPoolExecutor(max_workers=workers) as executor:
            # Submit all tasks
            future_to_path = {
                executor.submit(self._process_single_file, path, processor_func, **kwargs): path
                for path in file_paths
            }
            
            # Collect results with timeout
            for future in future_to_path:
                try:
                    result = future.result(timeout=timeout_minutes * 60)
                    results.append(result)
                except TimeoutError:
                    path = future_to_path[future]
                    self.logger.error(f"Processing timed out for: {path}")
                    results.append(ProcessingResult(
                        success=False,
                        error="Processing timed out",
                        processing_time=timeout_minutes * 60
                    ))
                except Exception as e:
                    path = future_to_path[future]
                    self.logger.error(f"Processing failed for {path}: {e}")
                    results.append(ProcessingResult(
                        success=False,
                        error=str(e)
                    ))
        
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
            return process.memory_info().rss / (1024 * 1024)
        except Exception:
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
            return []
        
        self.logger.info(f"Starting sequential processing of {len(file_paths)} files")
        
        results = []
        
        for i, file_path in enumerate(file_paths):
            self.logger.info(f"Processing file {i + 1}/{len(file_paths)}: {file_path.name}")
            
            # Check memory before each file
            if self.memory_monitor.is_memory_critical():
                self.logger.warning("Memory usage critical, pausing processing")
                time.sleep(self.memory_config.memory_pressure_pause_seconds)
            
            result = self._process_single_file(file_path, processor_func, **kwargs)
            results.append(result)
            
            # Log progress
            if (i + 1) % 10 == 0:
                self.logger.info(f"Processed {i + 1}/{len(file_paths)} files")
        
        return results
    
    def _process_single_file(
        self, 
        file_path: Path, 
        processor_func: Callable,
        **kwargs
    ) -> ProcessingResult:
        """Process a single file sequentially."""
        
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
            return process.memory_info().rss / (1024 * 1024)
        except Exception:
            return 0.0 