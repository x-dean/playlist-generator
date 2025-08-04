"""
Async Audio Processor for non-blocking audio analysis operations.
Converts synchronous audio processing to async patterns for better API performance.
"""

import asyncio
import functools
import time
from typing import Dict, Any, Optional, List, Callable
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

from .logging_setup import get_logger, log_universal
from .audio_analyzer import AudioAnalyzer
from .resource_manager import ResourceManager

logger = get_logger('playlista.async_audio')


class AsyncAudioProcessor:
    """
    Async wrapper for audio processing operations.
    Provides non-blocking audio analysis with proper resource management.
    """
    
    def __init__(self, max_workers: int = None, use_process_pool: bool = False):
        """
        Initialize async audio processor.
        
        Args:
            max_workers: Maximum number of worker threads/processes
            use_process_pool: If True, use process pool instead of thread pool
        """
        if max_workers is None:
            max_workers = min(4, mp.cpu_count())
        
        self.max_workers = max_workers
        self.use_process_pool = use_process_pool
        
        # Create executor
        if use_process_pool:
            self.executor = ProcessPoolExecutor(max_workers=max_workers)
        else:
            self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Resource management
        self.resource_manager = ResourceManager()
        
        # Processing stats
        self.stats = {
            'total_processed': 0,
            'successful_analyses': 0,
            'failed_analyses': 0,
            'average_processing_time': 0,
            'concurrent_operations': 0,
            'queue_size': 0
        }
        
        log_universal('INFO', 'AsyncAudio', f'Initialized async audio processor with {max_workers} workers')
        log_universal('INFO', 'AsyncAudio', f'Using {"process" if use_process_pool else "thread"} pool')
    
    async def analyze_track_async(
        self,
        file_path: str,
        force_reanalysis: bool = False,
        config: Dict[str, Any] = None,
        timeout: float = 300
    ) -> Optional[Dict[str, Any]]:
        """
        Analyze audio track asynchronously.
        
        Args:
            file_path: Path to audio file
            force_reanalysis: Force re-analysis even if cached
            config: Analysis configuration
            timeout: Maximum time to wait for analysis
            
        Returns:
            Analysis results or None if failed
        """
        start_time = time.time()
        self.stats['concurrent_operations'] += 1
        
        try:
            # Check resource availability before processing
            if not self._check_resource_availability():
                log_universal('WARNING', 'AsyncAudio', f'Insufficient resources for {file_path}')
                return None
            
            # Create analysis function
            analysis_func = functools.partial(
                self._analyze_track_sync,
                file_path=file_path,
                force_reanalysis=force_reanalysis,
                config=config
            )
            
            # Execute in thread/process pool
            loop = asyncio.get_event_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(self.executor, analysis_func),
                timeout=timeout
            )
            
            processing_time = time.time() - start_time
            
            if result:
                self.stats['successful_analyses'] += 1
                self._update_average_processing_time(processing_time)
                log_universal('INFO', 'AsyncAudio', f'Successfully analyzed {file_path} in {processing_time:.2f}s')
            else:
                self.stats['failed_analyses'] += 1
                log_universal('WARNING', 'AsyncAudio', f'Failed to analyze {file_path}')
            
            return result
            
        except asyncio.TimeoutError:
            self.stats['failed_analyses'] += 1
            log_universal('ERROR', 'AsyncAudio', f'Analysis timeout for {file_path} after {timeout}s')
            return None
        except Exception as e:
            self.stats['failed_analyses'] += 1
            log_universal('ERROR', 'AsyncAudio', f'Analysis error for {file_path}: {e}')
            return None
        finally:
            self.stats['concurrent_operations'] -= 1
            self.stats['total_processed'] += 1
    
    async def analyze_batch_async(
        self,
        file_paths: List[str],
        force_reanalysis: bool = False,
        config: Dict[str, Any] = None,
        max_concurrent: int = 3
    ) -> List[Optional[Dict[str, Any]]]:
        """
        Analyze multiple tracks concurrently with controlled concurrency.
        
        Args:
            file_paths: List of audio file paths
            force_reanalysis: Force re-analysis even if cached
            config: Analysis configuration
            max_concurrent: Maximum concurrent analyses
            
        Returns:
            List of analysis results
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def analyze_with_semaphore(file_path: str):
            async with semaphore:
                return await self.analyze_track_async(
                    file_path, force_reanalysis, config
                )
        
        # Create tasks for all files
        tasks = [
            analyze_with_semaphore(file_path)
            for file_path in file_paths
        ]
        
        # Execute with progress tracking
        results = []
        for i, task in enumerate(asyncio.as_completed(tasks)):
            result = await task
            results.append(result)
            
            # Log progress
            if (i + 1) % 10 == 0 or (i + 1) == len(file_paths):
                log_universal('INFO', 'AsyncAudio', f'Batch progress: {i + 1}/{len(file_paths)} completed')
        
        return results
    
    def _analyze_track_sync(
        self,
        file_path: str,
        force_reanalysis: bool = False,
        config: Dict[str, Any] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Synchronous analysis function for execution in thread/process pool.
        """
        try:
            # Create analyzer instance (thread-safe)
            analyzer = AudioAnalyzer()
            
            # Perform analysis
            result = analyzer.analyze_file(
                file_path,
                force_reextract=force_reanalysis,
                config=config
            )
            
            return result
            
        except Exception as e:
            log_universal('ERROR', 'AsyncAudio', f'Sync analysis error for {file_path}: {e}')
            return None
    
    def _check_resource_availability(self) -> bool:
        """Check if system has enough resources for analysis."""
        try:
            # Get current resource usage
            memory_percent = self.resource_manager.get_memory_usage_percent()
            cpu_percent = self.resource_manager.get_cpu_usage_percent()
            
            # Conservative thresholds for async processing
            memory_threshold = 85
            cpu_threshold = 80
            
            if memory_percent > memory_threshold:
                log_universal('WARNING', 'AsyncAudio', f'Memory usage too high: {memory_percent:.1f}%')
                return False
            
            if cpu_percent > cpu_threshold:
                log_universal('WARNING', 'AsyncAudio', f'CPU usage too high: {cpu_percent:.1f}%')
                return False
            
            return True
            
        except Exception:
            # If resource check fails, allow processing
            return True
    
    def _update_average_processing_time(self, processing_time: float):
        """Update average processing time with exponential smoothing."""
        alpha = 0.1  # Smoothing factor
        if self.stats['average_processing_time'] == 0:
            self.stats['average_processing_time'] = processing_time
        else:
            self.stats['average_processing_time'] = (
                alpha * processing_time + 
                (1 - alpha) * self.stats['average_processing_time']
            )
    
    async def get_processing_stats(self) -> Dict[str, Any]:
        """Get current processing statistics."""
        return {
            **self.stats,
            'executor_type': 'process' if self.use_process_pool else 'thread',
            'max_workers': self.max_workers,
            'memory_usage_percent': self.resource_manager.get_memory_usage_percent(),
            'cpu_usage_percent': self.resource_manager.get_cpu_usage_percent()
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Check processor health status."""
        try:
            # Test with a simple operation
            test_start = time.time()
            
            # Simple resource check
            memory_ok = self.resource_manager.get_memory_usage_percent() < 90
            cpu_ok = self.resource_manager.get_cpu_usage_percent() < 90
            
            response_time = time.time() - test_start
            
            status = "healthy" if (memory_ok and cpu_ok and response_time < 1.0) else "degraded"
            
            return {
                'status': status,
                'response_time_ms': response_time * 1000,
                'memory_ok': memory_ok,
                'cpu_ok': cpu_ok,
                'active_operations': self.stats['concurrent_operations'],
                'total_processed': self.stats['total_processed']
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'response_time_ms': 0,
                'memory_ok': False,
                'cpu_ok': False
            }
    
    def shutdown(self):
        """Shutdown the processor and cleanup resources."""
        try:
            self.executor.shutdown(wait=True)
            log_universal('INFO', 'AsyncAudio', 'Async audio processor shutdown complete')
        except Exception as e:
            log_universal('ERROR', 'AsyncAudio', f'Error during shutdown: {e}')


# Global processor instance
_processor_instance = None


def get_async_audio_processor(max_workers: int = None, use_process_pool: bool = False) -> AsyncAudioProcessor:
    """Get or create the global async audio processor instance."""
    global _processor_instance
    
    if _processor_instance is None:
        _processor_instance = AsyncAudioProcessor(max_workers, use_process_pool)
    
    return _processor_instance