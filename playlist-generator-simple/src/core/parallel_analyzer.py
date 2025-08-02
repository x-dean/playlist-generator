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

# Set multiprocessing start method to avoid pickling issues
if mp.get_start_method(allow_none=True) != 'spawn':
    mp.set_start_method('spawn', force=True)

# Import local modules
from .database import DatabaseManager
from .logging_setup import get_logger, log_function_call, log_universal
from .resource_manager import ResourceManager

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


def _standalone_worker_process(file_path: str, force_reextract: bool = False, 
                             timeout_seconds: int = 300, db_path: str = None,
                             analysis_config: Dict[str, Any] = None) -> bool:
    """
    Standalone worker function that can be pickled for multiprocessing.
    
    Args:
        file_path: Path to the file to process
        force_reextract: If True, bypass cache
        timeout_seconds: Timeout for analysis
        db_path: Database path
        analysis_config: Analysis configuration dictionary (uses default if None)
        
    Returns:
        True if successful, False otherwise
    """
    import os
    import time
    import signal
    import psutil
    import threading
    import gc
    
    # Set up logging for worker process
    try:
        from .logging_setup import get_logger, log_universal
        logger = get_logger('playlista.parallel_worker')
    except ImportError:
        import logging
        logger = logging.getLogger('playlista.parallel_worker')
        # Fallback to basic logging if universal logging not available
    
    worker_id = f"worker_{threading.current_thread().ident}"
    start_time = time.time()
    
            try:
            # Force garbage collection before starting
            gc.collect()
            
            # Set up signal handler for timeout (only on Unix systems)
            try:
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(timeout_seconds)
            except (AttributeError, OSError):
                # Windows doesn't support SIGALRM, skip timeout handling
                pass
    """
    Standalone worker function that can be pickled for multiprocessing.
    
    Args:
        file_path: Path to the file to process
        force_reextract: If True, bypass cache
        timeout_seconds: Timeout for analysis
        db_path: Database path
        analysis_config: Analysis configuration dictionary (uses default if None)
        
    Returns:
        True if successful, False otherwise
    """
    import os
    import time
    import signal
    import psutil
    import threading
    
    # Set up logging for worker process
    try:
        from .logging_setup import get_logger, log_universal
        logger = get_logger('playlista.parallel_worker')
    except ImportError:
        import logging
        logger = logging.getLogger('playlista.parallel_worker')
        # Fallback to basic logging if universal logging not available
    
    worker_id = f"worker_{threading.current_thread().ident}"
    start_time = time.time()
    
    try:
        # Set up signal handler for timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_seconds)
        
        # Check if file exists
        if not os.path.exists(file_path):
            log_universal('WARNING', 'Parallel', f'File not found: {file_path}')
            return False
        
        # Get initial resource usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / (1024 * 1024)  # MB
        initial_cpu = process.cpu_percent()
        
        # Import audio analyzer here to avoid memory issues
        try:
            from .audio_analyzer import AudioAnalyzer
        except ImportError as e:
            log_universal('ERROR', 'Parallel', f'Failed to import AudioAnalyzer: {e}')
            return False
        
        # Create analyzer instance with configuration
        try:
            if analysis_config is None:
                # Use default configuration
                analyzer = AudioAnalyzer()
            else:
                # Apply the analysis configuration to the analyzer
                analyzer = AudioAnalyzer(config=analysis_config)
        except Exception as e:
            log_universal('ERROR', 'Parallel', f'Failed to create AudioAnalyzer: {e}')
            return False
        
        # Set memory limit for this process
        try:
            import resource
            # Set memory limit to 1GB per process
            resource.setrlimit(resource.RLIMIT_AS, (1024 * 1024 * 1024, -1))
        except (ImportError, OSError):
            # Windows doesn't support resource module, skip memory limit
            pass
        
        # Extract features using the correct method
        try:
            analysis_result = analyzer.analyze_audio_file(file_path, force_reextract)
        except Exception as e:
            log_universal('ERROR', 'Parallel', f'Analysis failed for {file_path}: {e}')
            return False
        
        # Get final resource usage
        final_memory = process.memory_info().rss / (1024 * 1024)  # MB
        final_cpu = process.cpu_percent()
        memory_usage = final_memory - initial_memory
        cpu_usage = (initial_cpu + final_cpu) / 2  # Average
        
        duration = time.time() - start_time
        
        # Force garbage collection after analysis
        gc.collect()
        
        if analysis_result:
            # Save to database using standalone database manager
            if db_path:
                from .database import DatabaseManager
                db_manager = DatabaseManager(db_path=db_path)
                
                filename = os.path.basename(file_path)
                file_size_bytes = os.path.getsize(file_path)
                
                # Calculate hash (consistent with file discovery)
                import hashlib
                stat = os.stat(file_path)
                filename = os.path.basename(file_path)
                content = f"{filename}:{stat.st_mtime}:{stat.st_size}"
                file_hash = hashlib.md5(content.encode()).hexdigest()
                
                # Prepare analysis data with status
                analysis_data = analysis_result.get('features', {})
                analysis_data['status'] = 'analyzed'
                analysis_data['analysis_type'] = 'full'
                
                # Extract metadata
                metadata = analysis_result.get('metadata', {})
                
                # Determine long audio category
                long_audio_category = None
                if 'long_audio_category' in analysis_data:
                    long_audio_category = analysis_data['long_audio_category']
                
                success = db_manager.save_analysis_result(
                    file_path=file_path,
                    filename=filename,
                    file_size_bytes=file_size_bytes,
                    file_hash=file_hash,
                    analysis_data=analysis_data,
                    metadata=metadata,
                    discovery_source='file_system'
                )
                
                log_universal('INFO', 'Parallel', f'{worker_id} completed: {filename} in {duration:.2f}s')
                return success
            else:
                log_universal('INFO', 'Parallel', f'{worker_id} completed: {os.path.basename(file_path)} in {duration:.2f}s')
                return True
        else:
            # Mark as failed
            if db_path:
                from .database import DatabaseManager
                db_manager = DatabaseManager(db_path=db_path)
                filename = os.path.basename(file_path)
                
                # Use analysis_cache table for failed analysis
                with db_manager._get_db_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT OR REPLACE INTO analysis_cache 
                        (file_path, filename, error_message, status, retry_count, last_retry_date)
                        VALUES (?, ?, ?, 'failed', 0, CURRENT_TIMESTAMP)
                    """, (file_path, filename, "Analysis failed"))
                    conn.commit()
            
            log_universal('DEBUG', 'Parallel', f"Worker {worker_id} failed: {os.path.basename(file_path)}")
            return False
            
    except TimeoutException:
        duration = time.time() - start_time
        log_universal('ERROR', 'Parallel', f"⏰ Analysis timed out for {os.path.basename(file_path)}")
        
        if db_path:
            from .database import DatabaseManager
            db_manager = DatabaseManager(db_path=db_path)
            filename = os.path.basename(file_path)
            
            # Use analysis_cache table for failed analysis
            with db_manager._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO analysis_cache 
                    (file_path, filename, error_message, status, retry_count, last_retry_date)
                    VALUES (?, ?, ?, 'failed', 0, CURRENT_TIMESTAMP)
                """, (file_path, filename, "Analysis timed out"))
                conn.commit()
        
        return False
    except Exception as e:
        duration = time.time() - start_time
        log_universal('ERROR', 'Parallel', f"Worker {worker_id} error processing {os.path.basename(file_path)}: {e}")
        
        if db_path:
            from .database import DatabaseManager
            db_manager = DatabaseManager(db_path=db_path)
            filename = os.path.basename(file_path)
            
            # Use analysis_cache table for failed analysis
            with db_manager._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO analysis_cache 
                    (file_path, filename, error_message, status, retry_count, last_retry_date)
                    VALUES (?, ?, ?, 'failed', 0, CURRENT_TIMESTAMP)
                """, (file_path, filename, f"Worker error: {str(e)}"))
                conn.commit()
        
        return False


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
        
        log_universal('INFO', 'Parallel', f"Initializing ParallelAnalyzer")
        log_universal('DEBUG', 'Parallel', f"Timeout: {self.timeout_seconds}s, Memory threshold: {self.memory_threshold_percent}%")
        log_universal('INFO', 'Parallel', f"ParallelAnalyzer initialized successfully")

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
            log_universal('WARNING', 'Parallel', "No files provided for parallel processing")
            return {'success_count': 0, 'failed_count': 0, 'total_time': 0}
        
        # Determine optimal worker count dynamically
        if max_workers is None:
            optimal_workers = self.resource_manager.get_optimal_worker_count()
            max_workers = optimal_workers
            log_universal('INFO', 'Parallel', f"Dynamic worker count: {max_workers} (based on available memory and CPU)")
        else:
            max_workers = min(max_workers, optimal_workers)
            log_universal('INFO', 'Parallel', f"Manual worker count: {max_workers} (capped by optimal: {optimal_workers})")
        
        # Calculate batch size based on available memory and workers
        batch_size = max(1, min(len(files), max_workers * 5))  # Process 5x workers per batch
        total_batches = (len(files) + batch_size - 1) // batch_size
        
        log_universal('INFO', 'Parallel', f"Starting parallel processing of {len(files)} files")
        log_universal('INFO', 'Parallel', f"  Workers: {max_workers} (5 files per worker per batch)")
        log_universal('INFO', 'Parallel', f"  Batch size: {batch_size} files")
        log_universal('INFO', 'Parallel', f"  Total batches: {total_batches}")
        log_universal('DEBUG', 'Parallel', f"  Force re-extract: {force_reextract}")
        
        start_time = time.time()
        results = {
            'success_count': 0,
            'failed_count': 0,
            'total_time': 0,
            'processed_files': [],
            'worker_count': max_workers,
            'batches_processed': 0
        }
        
        try:
            # Process files in batches
            for i in range(0, len(files), batch_size):
                batch_files = files[i:i + batch_size]
                batch_num = (i // batch_size) + 1
                total_batches = (len(files) + batch_size - 1) // batch_size
                
                log_universal('INFO', 'Parallel', f"Processing batch {batch_num}/{total_batches} ({len(batch_files)} files)")
                
                try:
                    with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp.get_context('spawn')) as executor:
                        # Submit batch tasks
                        future_to_file = {}
                        for file_path in batch_files:
                            try:
                                analysis_config = self._get_analysis_config(file_path)
                                future = executor.submit(_standalone_worker_process, file_path, force_reextract, 
                                                      self.timeout_seconds, self.db_manager.db_path,
                                                      analysis_config)
                                future_to_file[future] = file_path
                            except Exception as e:
                                log_universal('ERROR', 'Parallel', f"Failed to submit task for {file_path}: {e}")
                                results['failed_count'] += 1
                                results['processed_files'].append({
                                    'file_path': file_path,
                                    'status': 'error',
                                    'error': f"Task submission failed: {str(e)}",
                                    'timestamp': datetime.now().isoformat()
                                })
                        
                        # Collect batch results
                        completed_count = 0
                        for future in as_completed(future_to_file, timeout=self.timeout_seconds * len(batch_files)):
                            file_path = future_to_file[future]
                            filename = os.path.basename(file_path)
                            completed_count += 1
                            
                            try:
                                success = future.result(timeout=60)
                                
                                if success:
                                    results['success_count'] += 1
                                    results['processed_files'].append({
                                        'file_path': file_path,
                                        'status': 'success',
                                        'timestamp': datetime.now().isoformat()
                                    })
                                    log_universal('DEBUG', 'Parallel', f"Completed: {filename}")
                                else:
                                    results['failed_count'] += 1
                                    results['processed_files'].append({
                                        'file_path': file_path,
                                        'status': 'failed',
                                        'timestamp': datetime.now().isoformat()
                                    })
                                    log_universal('DEBUG', 'Parallel', f"Failed: {filename}")
                                
                                log_universal('INFO', 'Parallel', f"Batch {batch_num}: Completed {completed_count}/{len(batch_files)}: {filename}")
                                    
                            except Exception as e:
                                log_universal('ERROR', 'Parallel', f"Error processing {filename}: {e}")
                                results['failed_count'] += 1
                                results['processed_files'].append({
                                    'file_path': file_path,
                                    'status': 'error',
                                    'error': str(e),
                                    'timestamp': datetime.now().isoformat()
                                })
                        
                        # Cancel any remaining futures in this batch
                        for future in future_to_file:
                            if not future.done():
                                future.cancel()
                        
                        results['batches_processed'] += 1
                        
                        # Force garbage collection between batches
                        import gc
                        gc.collect()
                        
                except Exception as e:
                    log_universal('ERROR', 'Parallel', f"Error processing batch {batch_num}: {e}")
                    # Mark remaining files in batch as failed
                    for file_path in batch_files:
                        if file_path not in [p['file_path'] for p in results['processed_files']]:
                            results['failed_count'] += 1
                            results['processed_files'].append({
                                'file_path': file_path,
                                'status': 'error',
                                'error': f"Batch processing failed: {str(e)}",
                                'timestamp': datetime.now().isoformat()
                            })
            except Exception as e:
                log_universal('ERROR', 'Parallel', f"Error creating process pool: {e}")
                # Fall back to sequential processing with better error handling
                log_universal('INFO', 'Parallel', "Falling back to sequential processing due to multiprocessing error")
                for file_path in files:
                    try:
                        # Get analysis config for each file
                        analysis_config = self._get_analysis_config(file_path)
                        success = _standalone_worker_process(file_path, force_reextract, 
                                                          self.timeout_seconds, self.db_manager.db_path,
                                                          analysis_config)
                        if success:
                            results['success_count'] += 1
                        else:
                            results['failed_count'] += 1
                    except Exception as worker_error:
                        log_universal('ERROR', 'Parallel', f"Sequential processing failed for {file_path}: {worker_error}")
                        results['failed_count'] += 1
                        
        except Exception as e:
            log_universal('ERROR', 'Parallel', f"Error in parallel processing: {e}")
            # Count remaining files as failed
            remaining_files = [f for f in files if f not in [p['file_path'] for p in results['processed_files']]]
            results['failed_count'] += len(remaining_files)
        
        total_time = time.time() - start_time
        results['total_time'] = total_time
        
        # Calculate batch processing statistics
        success_rate = (results['success_count'] / len(files) * 100) if files else 0
        avg_duration = total_time / len(files) if files else 0
        throughput = len(files) / total_time if total_time > 0 else 0
        
        # Get resource usage statistics (approximate)
        import psutil
        process = psutil.Process()
        memory_usage_mb = process.memory_info().rss / (1024 * 1024)
        cpu_usage_percent = process.cpu_percent()
        
        log_universal('INFO', 'Parallel', f"Parallel processing completed in {total_time:.2f}s")
        log_universal('INFO', 'Parallel', f"Results: {results['success_count']} successful, {results['failed_count']} failed")
        log_universal('INFO', 'Parallel', f"Success rate: {success_rate:.1f}%, Throughput: {throughput:.2f} files/s")
        
        # Log detailed batch processing statistics
        log_universal('INFO', 'Parallel', f"Parallel file processing completed in {total_time:.2f}s")
        log_universal('INFO', 'Parallel', f"Results: {results['success_count']} successful, {results['failed_count']} failed")
        log_universal('INFO', 'Parallel', f"Success rate: {success_rate:.1f}%, Throughput: {throughput:.2f} files/s")
        
        # Log performance
        log_universal('INFO', 'Parallel', f"Parallel file processing completed in {total_time:.2f}s")
        log_universal('INFO', 'Parallel', f"Results: {results['success_count']} successful, {results['failed_count']} failed")
        log_universal('INFO', 'Parallel', f"Success rate: {success_rate:.1f}%, Throughput: {throughput:.2f} files/s")
        
        return results

    # DEPRECATED: This method has pickling issues. Use _standalone_worker_process instead.
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
                log_universal('WARNING', 'Parallel', f"File not found: {file_path}")
                duration = time.time() - start_time
                log_universal('INFO', 'Parallel', f"Worker {worker_id} file check failed: {os.path.basename(file_path)}")
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
            analysis_result = analyzer.analyze_audio_file(file_path, force_reextract)
            
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
                
                # Prepare analysis data with status
                analysis_data = analysis_result.get('features', {})
                analysis_data['status'] = 'analyzed'
                analysis_data['analysis_type'] = 'full'
                
                # Extract metadata
                metadata = analysis_result.get('metadata', {})
                
                # Determine long audio category
                long_audio_category = None
                if 'long_audio_category' in analysis_data:
                    long_audio_category = analysis_data['long_audio_category']
                
                success = self.db_manager.save_analysis_result(
                    file_path=file_path,
                    filename=filename,
                    file_size_bytes=file_size_bytes,
                    file_hash=file_hash,
                    analysis_data=analysis_data,
                    metadata=metadata,
                    discovery_source='file_system'
                )
                
                # Log successful worker performance
                log_universal('INFO', 'Parallel', f"Worker {worker_id} file analysis completed: {os.path.basename(file_path)}")
                
                return success
            else:
                # Mark as failed
                filename = os.path.basename(file_path)
                
                # Use analysis_cache table for failed analysis
                with self.db_manager._get_db_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT OR REPLACE INTO analysis_cache 
                        (file_path, filename, error_message, status, retry_count, last_retry_date)
                        VALUES (?, ?, ?, 'failed', 0, CURRENT_TIMESTAMP)
                    """, (file_path, filename, "Analysis failed"))
                    conn.commit()
                
                # Log failed worker performance
                log_universal('ERROR', 'Parallel', f"Worker {worker_id} file analysis failed: {os.path.basename(file_path)}")
                
                return False
                
        except TimeoutException:
            duration = time.time() - start_time
            log_universal('ERROR', 'Parallel', f"⏰ Analysis timed out for {os.path.basename(file_path)}")
            filename = os.path.basename(file_path)
            
            # Use analysis_cache table for failed analysis
            with self.db_manager._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO analysis_cache 
                    (file_path, filename, error_message, status, retry_count, last_retry_date)
                    VALUES (?, ?, ?, 'failed', 0, CURRENT_TIMESTAMP)
                """, (file_path, filename, "Analysis timed out"))
                conn.commit()
            
            # Log timeout worker performance
            log_universal('ERROR', 'Parallel', f"Worker {worker_id} file analysis timeout: {os.path.basename(file_path)}")
            
            return False
        except Exception as e:
            duration = time.time() - start_time
            log_universal('ERROR', 'Parallel', f"Worker error for {os.path.basename(file_path)}: {e}")
            filename = os.path.basename(file_path)
            
            # Use analysis_cache table for failed analysis
            with self.db_manager._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO analysis_cache 
                    (file_path, filename, error_message, status, retry_count, last_retry_date)
                    VALUES (?, ?, ?, 'failed', 0, CURRENT_TIMESTAMP)
                """, (file_path, filename, str(e)))
                conn.commit()
            
            # Log error worker performance
            log_universal('ERROR', 'Parallel', f"Worker {worker_id} file analysis error: {os.path.basename(file_path)}")
            
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
            # Get file size for analysis type determination
            file_size_bytes = os.path.getsize(file_path)
            file_size_mb = file_size_bytes / (1024 * 1024)
            
            # Simple analysis type determination based on file size
            if file_size_mb > 50:  # Large files
                analysis_type = 'basic'
                use_full_analysis = False
            else:  # Smaller files
                analysis_type = 'basic'
                use_full_analysis = False
            
            # Enable MusiCNN for parallel processing (smaller files)
            enable_musicnn = True
            
            analysis_config = {
                'analysis_type': analysis_type,
                'use_full_analysis': use_full_analysis,
                'EXTRACT_RHYTHM': True,
                'EXTRACT_SPECTRAL': True,
                'EXTRACT_LOUDNESS': True,
                'EXTRACT_KEY': True,
                'EXTRACT_MFCC': True,
                'EXTRACT_MUSICNN': enable_musicnn,
                'EXTRACT_METADATA': True,
                'EXTRACT_DANCEABILITY': True,
                'EXTRACT_ONSET_RATE': True,
                'EXTRACT_ZCR': True,
                'EXTRACT_SPECTRAL_CONTRAST': True,
                'EXTRACT_CHROMA': True
            }
            
            log_universal('DEBUG', 'Parallel', f"Analysis config for {os.path.basename(file_path)}: {analysis_config['analysis_type']}")
            log_universal('DEBUG', 'Parallel', f"MusiCNN enabled: {enable_musicnn}")
            
            return analysis_config
            
        except Exception as e:
            log_universal('WARNING', 'Parallel', f"Error getting analysis config for {file_path}: {e}")
            # Return basic analysis config as fallback
            return {
                'analysis_type': 'basic',
                'use_full_analysis': False,
                'EXTRACT_RHYTHM': True,
                'EXTRACT_SPECTRAL': True,
                'EXTRACT_LOUDNESS': True,
                'EXTRACT_KEY': True,
                'EXTRACT_MFCC': True,
                'EXTRACT_MUSICNN': True,  # Enabled in fallback for parallel
                'EXTRACT_METADATA': True,
                'EXTRACT_DANCEABILITY': True,
                'EXTRACT_ONSET_RATE': True,
                'EXTRACT_ZCR': True,
                'EXTRACT_SPECTRAL_CONTRAST': True,
                'EXTRACT_CHROMA': True
            }

    def _calculate_file_hash(self, file_path: str) -> str:
        """
        Calculate a hash for file change detection.
        
        Args:
            file_path: Path to the file
            
        Returns:
            File hash string
        """
        try:
            import hashlib
            
            # Use filename + modification time + size for hash (consistent with file discovery)
            stat = os.stat(file_path)
            filename = os.path.basename(file_path)
            content = f"{filename}:{stat.st_mtime}:{stat.st_size}"
            
            return hashlib.md5(content.encode()).hexdigest()
            
        except Exception as e:
            log_universal('WARNING', 'Parallel', f"Could not calculate hash for {file_path}: {e}")
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
            
            log_universal('INFO', 'Parallel', f"Updated parallel analyzer configuration: {new_config}")
            return True
            
        except Exception as e:
            log_universal('ERROR', 'Parallel', f"Error updating parallel analyzer configuration: {e}")
            return False


# Global parallel analyzer instance - created lazily to avoid circular imports
_parallel_analyzer_instance = None

def get_parallel_analyzer() -> 'ParallelAnalyzer':
    """Get the global parallel analyzer instance, creating it if necessary."""
    global _parallel_analyzer_instance
    if _parallel_analyzer_instance is None:
        _parallel_analyzer_instance = ParallelAnalyzer()
    return _parallel_analyzer_instance 