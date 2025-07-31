"""
Parallel Analyzer for Playlist Generator Simple.
Processes smaller files in parallel for efficiency.
"""

import os
import time
import threading
import signal
import multiprocessing as mp
from typing import List, Dict, Any, Optional, Tuple, Iterator
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

# Import local modules
from .database import DatabaseManager
from .logging_setup import get_logger, log_function_call, log_performance, log_worker_performance, log_batch_processing_detailed, log_resource_usage
from .resource_manager import ResourceManager
from .progress_bar import get_progress_bar

logger = get_logger('playlista.parallel_analyzer')

# Constants
DEFAULT_TIMEOUT_SECONDS = 300  # 5 minutes for smaller files
DEFAULT_MEMORY_THRESHOLD_PERCENT = 85
DEFAULT_MAX_WORKERS = None  # Auto-determined


class TimeoutException(Exception):
    """Exception raised when analysis times out."""
    pass


def timeout_handler(signum, frame):
    """Signal handler for timeout."""
    raise TimeoutException("Analysis timed out")


class ParallelAnalyzer:
    """
    Analyzes smaller files in parallel for efficiency.
    
    Handles:
    - Parallel file processing with multiple workers
    - Memory-aware worker count calculation
    - Process pool management
    - Database integration
    - Error handling and recovery
    """
    
    def __init__(self, db_manager: DatabaseManager = None,
                 resource_manager: ResourceManager = None,
                 timeout_seconds: int = None,
                 memory_threshold_percent: int = None,
                 max_workers: int = None):
        """
        Initialize the parallel analyzer.
        
        Args:
            db_manager: Database manager instance (creates new one if None)
            resource_manager: Resource manager instance (creates new one if None)
            timeout_seconds: Timeout for analysis in seconds
            memory_threshold_percent: Memory threshold percentage
            max_workers: Maximum number of workers (auto-determined if None)
        """
        self.db_manager = db_manager or DatabaseManager()
        self.resource_manager = resource_manager or ResourceManager()
        
        # Analysis settings
        self.timeout_seconds = timeout_seconds or DEFAULT_TIMEOUT_SECONDS
        self.memory_threshold_percent = memory_threshold_percent or DEFAULT_MEMORY_THRESHOLD_PERCENT
        self.max_workers = max_workers or DEFAULT_MAX_WORKERS
        
        logger.info(f" Initializing ParallelAnalyzer")
        logger.debug(f" Timeout: {self.timeout_seconds}s, Memory threshold: {self.memory_threshold_percent}%")
        logger.info(f"ParallelAnalyzer initialized successfully")

    @log_function_call
    def process_files(self, files: List[str], force_reextract: bool = False,
                     max_workers: int = None) -> Dict[str, Any]:
        """
        Process files in parallel.
        
        Args:
            files: List of file paths to process
            force_reextract: If True, bypass cache for all files
            max_workers: Maximum number of workers (uses auto-determination if None)
            
        Returns:
            Dictionary with processing results and statistics
        """
        if not files:
            logger.warning("No files provided for parallel processing")
            return {'success_count': 0, 'failed_count': 0, 'total_time': 0}
        
        # Determine optimal worker count
        if max_workers is None:
            max_workers = self.max_workers or self.resource_manager.get_optimal_worker_count()
        
        logger.info(f"Starting parallel processing of {len(files)} files")
        logger.debug(f"   Force re-extract: {force_reextract}")
        logger.debug(f"   Max workers: {max_workers}")
        
        # Get progress bar
        progress_bar = get_progress_bar()
        progress_bar.start_analysis(len(files), "Parallel Analysis")
        
        start_time = time.time()
        results = {
            'success_count': 0,
            'failed_count': 0,
            'total_time': 0,
            'processed_files': [],
            'worker_count': max_workers
        }
        
        try:
            # Process files in parallel
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_file = {
                    executor.submit(self._process_single_file_worker, file_path, force_reextract): file_path
                    for file_path in files
                }
                
                # Collect results as they complete
                completed_count = 0
                for future in as_completed(future_to_file, timeout=self.timeout_seconds * len(files)):
                    file_path = future_to_file[future]
                    filename = os.path.basename(file_path)
                    completed_count += 1
                    
                    try:
                        success = future.result(timeout=30)  # 30s timeout per result
                        
                        if success:
                            results['success_count'] += 1
                            results['processed_files'].append({
                                'file_path': file_path,
                                'status': 'success',
                                'timestamp': datetime.now().isoformat()
                            })
                            logger.debug(f"Completed: {filename}")
                        else:
                            results['failed_count'] += 1
                            results['processed_files'].append({
                                'file_path': file_path,
                                'status': 'failed',
                                'timestamp': datetime.now().isoformat()
                            })
                            logger.debug(f"Failed: {filename}")
                        
                        # Update progress bar
                        progress_bar.update_analysis_progress(completed_count, filename)
                            
                    except Exception as e:
                        logger.error(f"Error processing {filename}: {e}")
                        results['failed_count'] += 1
                        results['processed_files'].append({
                            'file_path': file_path,
                            'status': 'error',
                            'error': str(e),
                            'timestamp': datetime.now().isoformat()
                        })
                
                # Cancel any remaining futures
                for future in future_to_file:
                    if not future.done():
                        future.cancel()
                        
        except Exception as e:
            logger.error(f"Error in parallel processing: {e}")
            # Count remaining files as failed
            remaining_files = [f for f in files if f not in [p['file_path'] for p in results['processed_files']]]
            results['failed_count'] += len(remaining_files)
        
        total_time = time.time() - start_time
        results['total_time'] = total_time
        
        # Complete progress bar
        progress_bar.complete_analysis(
            len(files), 
            results['success_count'], 
            results['failed_count'], 
            "Parallel Analysis"
        )
        
        # Calculate batch processing statistics
        success_rate = (results['success_count'] / len(files) * 100) if files else 0
        avg_duration = total_time / len(files) if files else 0
        throughput = len(files) / total_time if total_time > 0 else 0
        
        # Get resource usage statistics (approximate)
        import psutil
        process = psutil.Process()
        memory_usage_mb = process.memory_info().rss / (1024 * 1024)
        cpu_usage_percent = process.cpu_percent()
        
        logger.info(f"Parallel processing completed in {total_time:.2f}s")
        logger.info(f"Results: {results['success_count']} successful, {results['failed_count']} failed")
        logger.info(f"Success rate: {success_rate:.1f}%, Throughput: {throughput:.2f} files/s")
        
        # Log detailed batch processing statistics
        log_batch_processing_detailed(
            batch_id=f"parallel_{int(start_time)}",
            total_files=len(files),
            successful_files=results['success_count'],
            failed_files=results['failed_count'],
            total_duration=total_time,
            processing_mode='parallel',
            avg_memory_usage_mb=memory_usage_mb,
            avg_cpu_usage_percent=cpu_usage_percent,
            peak_memory_mb=memory_usage_mb,
            peak_cpu_percent=cpu_usage_percent,
            worker_count=max_workers,
            success_rate=success_rate,
            throughput_files_per_second=throughput
        )
        
        # Log performance
        log_performance("Parallel file processing", total_time,
                       total_files=len(files),
                       success_count=results['success_count'],
                       failed_count=results['failed_count'],
                       worker_count=max_workers)
        
        return results

    def _process_single_file_worker(self, file_path: str, force_reextract: bool = False) -> bool:
        """
        Worker function to process a single file.
        
        Args:
            file_path: Path to the file to process
            force_reextract: If True, bypass cache
            
        Returns:
            True if successful, False otherwise
        """
        import psutil
        import threading
        
        worker_id = f"worker_{threading.current_thread().ident}"
        start_time = time.time()
        
        try:
            # Set up signal handler for timeout
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(self.timeout_seconds)
            
            # Check if file exists
            if not os.path.exists(file_path):
                logger.warning(f"File not found: {file_path}")
                duration = time.time() - start_time
                log_worker_performance(worker_id, "file_check", file_path, duration, success=False, error="File not found")
                return False
            
            # Get initial resource usage
            process = psutil.Process()
            initial_memory = process.memory_info().rss / (1024 * 1024)  # MB
            initial_cpu = process.cpu_percent()
            
            # Import audio analyzer here to avoid memory issues
            from .audio_analyzer import AudioAnalyzer
            
            # Create analyzer instance
            analyzer = AudioAnalyzer()
            
            # Get analysis configuration from analysis manager
            analysis_config = self._get_analysis_config(file_path)
            
            # Extract features
            analysis_result = analyzer.extract_features(file_path, analysis_config)
            
            # Get final resource usage
            final_memory = process.memory_info().rss / (1024 * 1024)  # MB
            final_cpu = process.cpu_percent()
            memory_usage = final_memory - initial_memory
            cpu_usage = (initial_cpu + final_cpu) / 2  # Average
            
            duration = time.time() - start_time
            
            if analysis_result and analysis_result.get('success', False):
                # Save to database
                filename = os.path.basename(file_path)
                file_size_bytes = os.path.getsize(file_path)
                file_hash = self._calculate_file_hash(file_path)
                
                success = self.db_manager.save_analysis_result(
                    file_path=file_path,
                    filename=filename,
                    file_size_bytes=file_size_bytes,
                    file_hash=file_hash,
                    analysis_data=analysis_result.get('features', {}),
                    metadata=analysis_result.get('metadata', {})
                )
                
                # Log successful worker performance
                log_worker_performance(worker_id, "file_analysis", file_path, duration, 
                                    memory_usage_mb=memory_usage, cpu_usage_percent=cpu_usage, 
                                    success=success, file_size_mb=file_size_bytes/(1024*1024))
                
                return success
            else:
                # Mark as failed
                filename = os.path.basename(file_path)
                self.db_manager.mark_analysis_failed(file_path, filename, "Analysis failed")
                
                # Log failed worker performance
                log_worker_performance(worker_id, "file_analysis", file_path, duration, 
                                    memory_usage_mb=memory_usage, cpu_usage_percent=cpu_usage, 
                                    success=False, error="Analysis failed")
                
                return False
                
        except TimeoutException:
            duration = time.time() - start_time
            logger.error(f"⏰ Analysis timed out for {os.path.basename(file_path)}")
            filename = os.path.basename(file_path)
            self.db_manager.mark_analysis_failed(file_path, filename, "Analysis timed out")
            
            # Log timeout worker performance
            log_worker_performance(worker_id, "file_analysis", file_path, duration, 
                                success=False, error="Analysis timed out")
            
            return False
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Worker error for {os.path.basename(file_path)}: {e}")
            filename = os.path.basename(file_path)
            self.db_manager.mark_analysis_failed(file_path, filename, str(e))
            
            # Log error worker performance
            log_worker_performance(worker_id, "file_analysis", file_path, duration, 
                                success=False, error=str(e))
            
            return False

    def _get_analysis_config(self, file_path: str) -> Dict[str, Any]:
        """
        Get analysis configuration for a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Analysis configuration dictionary
        """
        try:
            # Import analysis manager to get deterministic analysis type
            from .analysis_manager import analysis_manager
            
            # Get deterministic analysis configuration
            analysis_config = analysis_manager.determine_analysis_type(file_path)
            
            logger.debug(f"Analysis config for {os.path.basename(file_path)}: {analysis_config['analysis_type']}")
            
            return analysis_config
            
        except Exception as e:
            logger.warning(f"Error getting analysis config for {file_path}: {e}")
            # Return basic analysis config as fallback
            return {
                'analysis_type': 'basic',
                'use_full_analysis': False,
                'features_config': {
                    'extract_rhythm': True,
                    'extract_spectral': True,
                    'extract_loudness': True,
                    'extract_key': True,
                    'extract_mfcc': True,
                    'extract_musicnn': False,
                    'extract_metadata': True
                }
            }

    def _calculate_file_hash(self, file_path: str) -> str:
        """
        Calculate a simple hash for file change detection.
        
        Args:
            file_path: Path to the file
            
        Returns:
            File hash string
        """
        try:
            import hashlib
            
            # Use file size and modification time as simple hash
            stat = os.stat(file_path)
            hash_data = f"{stat.st_size}_{stat.st_mtime}"
            
            return hashlib.md5(hash_data.encode()).hexdigest()
            
        except Exception as e:
            logger.warning(f"Could not calculate hash for {file_path}: {e}")
            return "unknown"

    @log_function_call
    def get_optimal_worker_count(self, max_workers: int = None) -> int:
        """
        Calculate optimal worker count for parallel processing.
        
        Args:
            max_workers: Maximum number of workers (uses auto-determination if None)
            
        Returns:
            Optimal number of workers
        """
        return self.resource_manager.get_optimal_worker_count(max_workers)

    def get_config(self) -> Dict[str, Any]:
        """Get current analyzer configuration."""
        return {
            'timeout_seconds': self.timeout_seconds,
            'memory_threshold_percent': self.memory_threshold_percent,
            'max_workers': self.max_workers
        }
    
    def update_config(self, new_config: Dict[str, Any]) -> bool:
        """
        Update analyzer configuration.
        
        Args:
            new_config: New configuration dictionary
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if 'timeout_seconds' in new_config:
                self.timeout_seconds = new_config['timeout_seconds']
            if 'memory_threshold_percent' in new_config:
                self.memory_threshold_percent = new_config['memory_threshold_percent']
            if 'max_workers' in new_config:
                self.max_workers = new_config['max_workers']
            
            logger.info(f" Updated parallel analyzer configuration: {new_config}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating parallel analyzer configuration: {e}")
            return False


# Global parallel analyzer instance
parallel_analyzer = ParallelAnalyzer() 