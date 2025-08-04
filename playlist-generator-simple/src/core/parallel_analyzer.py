"""
Parallel Analyzer for Playlist Generator Simple.
Processes smaller files in parallel for efficiency.
"""

import os
import time
import threading
import signal
from typing import List, Dict, Any, Optional, Tuple, Iterator
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import local modules
from .database import DatabaseManager
from .logging_setup import get_logger, log_function_call, log_universal
from .resource_manager import ResourceManager

# Import constants from audio_analyzer
try:
    from .audio_analyzer import TENSORFLOW_AVAILABLE
except ImportError:
    TENSORFLOW_AVAILABLE = False

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
                 max_workers: int = None,
                 config: Dict[str, Any] = None):
        """
        Initialize the parallel analyzer.
        
        Args:
            db_manager: Database manager instance (creates new one if None)
            resource_manager: Resource manager instance (creates new one if None)
            timeout_seconds: Timeout for analysis in seconds
            memory_threshold_percent: Memory threshold percentage
            max_workers: Maximum number of workers (auto-determined if None)
            config: Configuration dictionary (uses global config if None)
        """
        # Load configuration
        if config is None:
            from .config_loader import config_loader
            config = config_loader.get_audio_analysis_config()
        
        self.config = config
        self.db_manager = db_manager or DatabaseManager()
        self.resource_manager = resource_manager or ResourceManager()
        
        # Analysis settings
        self.timeout_seconds = timeout_seconds or DEFAULT_TIMEOUT_SECONDS
        self.memory_threshold_percent = memory_threshold_percent or DEFAULT_MEMORY_THRESHOLD_PERCENT
        self.max_workers = max_workers or DEFAULT_MAX_WORKERS
        
        log_universal('INFO', 'Parallel', f"Initializing ParallelAnalyzer")
        log_universal('DEBUG', 'Parallel', f"Timeout: {self.timeout_seconds}s, Memory threshold: {self.memory_threshold_percent}%")
        log_universal('INFO', 'Parallel', f"ParallelAnalyzer initialized successfully")

    def _process_files_threaded(self, files: List[str], force_reextract: bool = False,
                               max_workers: int = 2) -> Dict[str, Any]:
        """
        Process files using threaded approach for better memory management.
        
        Args:
            files: List of file paths to process
            force_reextract: If True, bypass cache
            max_workers: Maximum number of worker threads
            
        Returns:
            Dictionary with processing results and statistics
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from .audio_analyzer import AudioAnalyzer
        from .model_manager import get_model_manager
        
        # Initialize shared model manager
        model_manager = get_model_manager(self.config)
        
        log_universal('INFO', 'Parallel', f"Threaded analysis: {len(files)} files, {max_workers} workers")
        log_universal('DEBUG', 'Parallel', f"Using shared model manager: {model_manager.get_model_info()}")
        start_time = time.time()
        results = {
            'success_count': 0,
            'failed_count': 0,
            'processed_files': []
        }

        def _thread_initializer():
            """Initialize thread-local resources."""
            try:
                # Pre-initialize shared models in main thread
                if not model_manager.is_musicnn_available():
                    log_universal('DEBUG', 'Parallel', f"MusicNN models not available in thread {threading.current_thread().ident}")
                else:
                    log_universal('DEBUG', 'Parallel', f"MusicNN models available in thread {threading.current_thread().ident}")
                
                log_universal('DEBUG', 'Parallel', f"Thread {threading.current_thread().ident} initialized")
            except Exception as e:
                log_universal('ERROR', 'Parallel', f"Thread initialization failed: {e}")

        def _thread_worker(file_path: str) -> Tuple[str, bool]:
            """Worker function for threaded processing."""
            try:
                # Get analysis configuration for this file
                analysis_config = self._get_analysis_config(file_path)
                
                # Create analyzer instance for this thread (uses shared models)
                analyzer = AudioAnalyzer(config=analysis_config, processing_mode='parallel')
                
                # Analyze the file
                result = analyzer.analyze_audio_file(file_path, force_reextract)
                
                # Save to database if successful
                if result:
                    filename = os.path.basename(file_path)
                    file_size_bytes = os.path.getsize(file_path)
                    file_hash = self._calculate_file_hash(file_path)
                    
                    # Prepare analysis data
                    analysis_data = result.get('features', {})
                    analysis_data['status'] = 'analyzed'
                    analysis_data['analysis_type'] = 'full'
                    
                    # Extract metadata
                    metadata = result.get('metadata', {})
                    
                    # Save to database
                    success = self.db_manager.save_analysis_result(
                        file_path=file_path,
                        filename=filename,
                        file_size_bytes=file_size_bytes,
                        file_hash=file_hash,
                        analysis_data=analysis_data,
                        metadata=metadata,
                        discovery_source='file_system'
                    )
                    
                    return file_path, success
                else:
                    # Mark as failed in database
                    filename = os.path.basename(file_path)
                    with self.db_manager._get_db_connection() as conn:
                        cursor = conn.cursor()
                        cursor.execute("""
                            INSERT OR REPLACE INTO analysis_cache 
                            (file_path, filename, error_message, status, retry_count, last_retry_date)
                            VALUES (?, ?, ?, 'failed', 0, CURRENT_TIMESTAMP)
                        """, (file_path, filename, "Analysis failed"))
                        conn.commit()
                    
                    return file_path, False
                    
            except Exception as e:
                log_universal('ERROR', 'Threaded', f"Failed: {file_path}: {e}")
                
                # Mark as failed in database
                try:
                    filename = os.path.basename(file_path)
                    with self.db_manager._get_db_connection() as conn:
                        cursor = conn.cursor()
                        cursor.execute("""
                            INSERT OR REPLACE INTO analysis_cache 
                            (file_path, filename, error_message, status, retry_count, last_retry_date)
                            VALUES (?, ?, ?, 'failed', 0, CURRENT_TIMESTAMP)
                        """, (file_path, filename, f"Thread error: {str(e)}"))
                        conn.commit()
                except Exception as db_error:
                    log_universal('ERROR', 'Threaded', f"Database error: {db_error}")
                
                return file_path, False

        with ThreadPoolExecutor(max_workers=max_workers, initializer=_thread_initializer) as executor:
            futures = [executor.submit(_thread_worker, f) for f in files]
            for future in as_completed(futures):
                path, success = future.result()
                results['processed_files'].append({
                    'file_path': path,
                    'status': 'success' if success else 'failed',
                    'timestamp': datetime.now().isoformat()
                })
                if success:
                    results['success_count'] += 1
                else:
                    results['failed_count'] += 1

        results['total_time'] = time.time() - start_time
        success_rate = (results['success_count'] / len(files)) * 100
        throughput = len(files) / results['total_time']

        log_universal('INFO', 'Parallel', f"Threaded: {results['success_count']} succeeded, "
                                      f"{results['failed_count']} failed in {results['total_time']:.2f}s")
        log_universal('INFO', 'Parallel', f"Success rate: {success_rate:.1f}%, Throughput: {throughput:.2f} files/s")

        return results

    @log_function_call
    def process_files(self, files: List[str], force_reextract: bool = False,
                     max_workers: int = None) -> Dict[str, Any]:
        """
        Process files using threaded parallel processing.
        
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
        optimal_workers = self.resource_manager.get_optimal_worker_count()
        if max_workers is None:
            max_workers = optimal_workers
            log_universal('INFO', 'Parallel', f"Dynamic worker count: {max_workers} (based on available memory and CPU)")
        else:
            max_workers = min(max_workers, optimal_workers)
            log_universal('INFO', 'Parallel', f"Manual worker count: {max_workers} (capped by optimal: {optimal_workers})")
        
        log_universal('INFO', 'Parallel', f"Starting threaded processing of {len(files)} files")
        log_universal('INFO', 'Parallel', f"  Workers: {max_workers}")
        log_universal('DEBUG', 'Parallel', f"  Force re-extract: {force_reextract}")
        
        # Use threaded processing
        return self._process_files_threaded(files, force_reextract, max_workers)

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
            
            # Enable MusiCNN for parallel processing (all files, with optimized sampling for long tracks)
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
        # Load config for the global instance
        from .config_loader import config_loader
        config = config_loader.get_audio_analysis_config()
        _parallel_analyzer_instance = ParallelAnalyzer(config=config)
    return _parallel_analyzer_instance 