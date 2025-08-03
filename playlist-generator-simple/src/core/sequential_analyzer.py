"""
Sequential Analyzer for Playlist Generator Simple.
Processes large files sequentially to manage memory usage.
"""

import os
import time
import threading
import gc
import signal
import subprocess
import sys
import json
import tempfile
import multiprocessing as mp
from typing import List, Dict, Any, Optional, Tuple, Iterator
from datetime import datetime

# Import local modules
from .database import DatabaseManager
from .logging_setup import get_logger, log_function_call, log_universal
from .resource_manager import ResourceManager

logger = get_logger('playlista.sequential_analyzer')

# Constants
DEFAULT_TIMEOUT_SECONDS = 600  # 10 minutes for large files
DEFAULT_MEMORY_THRESHOLD_PERCENT = 85
DEFAULT_RSS_LIMIT_GB = 6.0


class TimeoutException(Exception):
    """Exception raised when analysis times out."""
    pass


def timeout_handler(signum, frame):
    """Signal handler for timeout."""
    raise TimeoutException("Analysis timed out")


def _worker_process_function(file_path: str, force_reextract: bool, timeout_seconds: int) -> Dict[str, Any]:
    """
    Worker function for multiprocessing - runs in isolated process.
    
    Args:
        file_path: Path to the file to process
        force_reextract: If True, bypass cache
        timeout_seconds: Timeout for analysis
        
    Returns:
        Dictionary with result information
    """
    try:
        # Set up timeout
        def timeout_handler_worker(signum, frame):
            raise TimeoutError("Worker process timed out")
        
        signal.signal(signal.SIGALRM, timeout_handler_worker)
        signal.alarm(timeout_seconds)
        
        # Import required modules in worker process
        from .audio_analyzer import AudioAnalyzer
        from .database import DatabaseManager
        
        # Initialize components
        db_manager = DatabaseManager()
        analyzer = AudioAnalyzer(processing_mode='sequential')
        
        # Process file
        result = analyzer.analyze_audio_file(file_path, force_reextract)
        
        if result:
            # Save to database
            filename = os.path.basename(file_path)
            file_size_bytes = os.path.getsize(file_path)
            file_hash = analyzer._calculate_file_hash(file_path)
            
            # Prepare analysis data
            analysis_data = result.get('features', {})
            analysis_data['status'] = 'analyzed'
            analysis_data['analysis_type'] = 'full'
            
            # Extract metadata
            metadata = result.get('metadata', {})
            
            # Save to database
            success = db_manager.save_analysis_result(
                file_path=file_path,
                filename=filename,
                file_size_bytes=file_size_bytes,
                file_hash=file_hash,
                analysis_data=analysis_data,
                metadata=metadata,
                discovery_source='file_system'
            )
            
            signal.alarm(0)  # Cancel timeout
            return {'success': success, 'error': None}
        else:
            signal.alarm(0)  # Cancel timeout
            return {'success': False, 'error': 'Analysis failed - no valid result returned'}
            
    except Exception as e:
        try:
            signal.alarm(0)  # Cancel timeout
        except:
            pass
        return {'success': False, 'error': str(e)}


class SequentialAnalyzer:
    """
    Analyzes large files sequentially to manage memory usage.
    
    Handles:
    - Large file processing with timeout
    - Memory monitoring during analysis
    - Process isolation for stability
    - Database integration
    - Error handling and recovery
    """
    
    def __init__(self, db_manager: DatabaseManager = None, 
                 resource_manager: ResourceManager = None,
                 timeout_seconds: int = None,
                 memory_threshold_percent: int = None,
                 rss_limit_gb: float = None):
        """
        Initialize the sequential analyzer.
        
        Args:
            db_manager: Database manager instance (creates new one if None)
            resource_manager: Resource manager instance (creates new one if None)
            timeout_seconds: Timeout for analysis in seconds
            memory_threshold_percent: Memory threshold percentage
            rss_limit_gb: RSS memory limit in GB
        """
        self.db_manager = db_manager or DatabaseManager()
        self.resource_manager = resource_manager or ResourceManager()
        
        # Analysis settings
        self.timeout_seconds = timeout_seconds or DEFAULT_TIMEOUT_SECONDS
        self.memory_threshold_percent = memory_threshold_percent or DEFAULT_MEMORY_THRESHOLD_PERCENT
        self.rss_limit_gb = rss_limit_gb or DEFAULT_RSS_LIMIT_GB
        
        log_universal('INFO', 'Sequential', 'Initializing SequentialAnalyzer')
        log_universal('INFO', 'Sequential', f'Timeout: {self.timeout_seconds}s, Memory threshold: {self.memory_threshold_percent}%')
        log_universal('INFO', 'Sequential', 'SequentialAnalyzer initialized successfully')

    @log_function_call
    def process_files(self, files: List[str], force_reextract: bool = False) -> Dict[str, Any]:
        """
        Process files sequentially with process isolation using multiprocessing.
        
        Args:
            files: List of file paths to process
            force_reextract: If True, bypass cache for all files
            
        Returns:
            Dictionary with processing results and statistics
        """
        if not files:
            log_universal('WARNING', 'Sequential', 'No files provided for sequential processing')
            return {'success_count': 0, 'failed_count': 0, 'total_time': 0}
        
        log_universal('INFO', 'Sequential', f'Starting sequential processing of {len(files)} files')
        log_universal('INFO', 'Sequential', f'Force re-extract: {force_reextract}')
        
        start_time = time.time()
        results = {
            'success_count': 0,
            'failed_count': 0,
            'total_time': 0,
            'processed_files': []
        }
        
        for i, file_path in enumerate(files, 1):
            try:
                filename = os.path.basename(file_path)
                log_universal('INFO', 'Sequential', f'Processing file {i}/{len(files)}: {filename}')
                
                # Process single file using multiprocessing for isolation
                success = self._process_single_file_multiprocessing(file_path, force_reextract)
                
                if success:
                    results['success_count'] += 1
                    results['processed_files'].append({
                        'file_path': file_path,
                        'status': 'success',
                        'timestamp': datetime.now().isoformat()
                    })
                else:
                    results['failed_count'] += 1
                    results['processed_files'].append({
                        'file_path': file_path,
                        'status': 'failed',
                        'timestamp': datetime.now().isoformat()
                    })
                
                # Memory cleanup between files
                self._cleanup_memory()
                
            except Exception as e:
                log_universal('ERROR', 'Sequential', f"Error processing {file_path}: {e}")
                results['failed_count'] += 1
                results['processed_files'].append({
                    'file_path': file_path,
                    'status': 'error',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
        
        total_time = time.time() - start_time
        results['total_time'] = total_time
        
        log_universal('INFO', 'Sequential', f"Sequential file processing completed in {total_time:.2f}s")
        log_universal('INFO', 'Sequential', f"Results: {results['success_count']} successful, {results['failed_count']} failed")
        
        return results

    def _process_single_file_isolated(self, file_path: str, force_reextract: bool = False) -> bool:
        """
        Process a single file in an isolated subprocess to prevent main app crashes.
        
        Args:
            file_path: Path to the file to process
            force_reextract: If True, bypass cache
            
        Returns:
            True if successful, False otherwise
        """
        filename = os.path.basename(file_path)
        log_universal('DEBUG', 'Sequential', f"Processing in isolated subprocess: {filename}")
        
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                log_universal('WARNING', 'Sequential', f"File not found: {file_path}")
                self.db_manager.mark_analysis_failed(file_path, filename, "File not found")
                return False
            
            # Create temporary file for subprocess communication
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
                temp_file_path = temp_file.name
            
            # Prepare subprocess arguments
            script_path = os.path.join(os.path.dirname(__file__), 'sequential_worker.py')
            
            # Create worker script if it doesn't exist
            if not os.path.exists(script_path):
                self._create_worker_script(script_path)
            
            # Run subprocess with timeout
            cmd = [
                sys.executable, script_path,
                '--file-path', file_path,
                '--force-reextract', str(force_reextract),
                '--output-file', temp_file_path,
                '--timeout', str(self.timeout_seconds)
            ]
            
            log_universal('DEBUG', 'Sequential', f"Starting subprocess: {' '.join(cmd)}")
            
            # Run subprocess with timeout
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            try:
                stdout, stderr = process.communicate(timeout=self.timeout_seconds)
                return_code = process.returncode
                
                # Check if subprocess completed successfully
                if return_code == 0:
                    # Read result from temporary file
                    if os.path.exists(temp_file_path):
                        with open(temp_file_path, 'r') as f:
                            result_data = json.load(f)
                        
                        if result_data.get('success', False):
                            log_universal('INFO', 'Sequential', f"Successfully processed: {filename}")
                            return True
                        else:
                            error_msg = result_data.get('error', 'Unknown error')
                            log_universal('ERROR', 'Sequential', f"Subprocess failed for {filename}: {error_msg}")
                            self.db_manager.mark_analysis_failed(file_path, filename, error_msg)
                            return False
                    else:
                        log_universal('ERROR', 'Sequential', f"No result file found for {filename}")
                        self.db_manager.mark_analysis_failed(file_path, filename, "No result file generated")
                        return False
                else:
                    # Subprocess failed
                    error_msg = stderr.strip() if stderr else f"Subprocess failed with return code {return_code}"
                    log_universal('ERROR', 'Sequential', f"Subprocess failed for {filename}: {error_msg}")
                    self.db_manager.mark_analysis_failed(file_path, filename, error_msg)
                    return False
                    
            except subprocess.TimeoutExpired:
                # Kill subprocess if it times out
                process.kill()
                process.communicate()
                log_universal('ERROR', 'Sequential', f"Subprocess timed out for {filename}")
                self.db_manager.mark_analysis_failed(file_path, filename, "Analysis timed out")
                return False
                
        except Exception as e:
            log_universal('ERROR', 'Sequential', f"Error in isolated processing for {filename}: {e}")
            self.db_manager.mark_analysis_failed(file_path, filename, str(e))
            return False
        finally:
            # Clean up temporary file
            try:
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
            except Exception:
                pass

    def _process_single_file_multiprocessing(self, file_path: str, force_reextract: bool = False) -> bool:
        """
        Process a single file using multiprocessing for better isolation.
        
        Args:
            file_path: Path to the file to process
            force_reextract: If True, bypass cache
            
        Returns:
            True if successful, False otherwise
        """
        filename = os.path.basename(file_path)
        log_universal('DEBUG', 'Sequential', f"Processing with multiprocessing: {filename}")
        
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                log_universal('WARNING', 'Sequential', f"File not found: {file_path}")
                self.db_manager.mark_analysis_failed(file_path, filename, "File not found")
                return False
            
            # Calculate file hash for potential failure tracking
            file_hash = self._calculate_file_hash(file_path)
            
            # Use multiprocessing to isolate the analysis
            with mp.Pool(processes=1) as pool:
                # Submit the work to the pool
                future = pool.apply_async(
                    _worker_process_function, 
                    args=(file_path, force_reextract, self.timeout_seconds)
                )
                
                try:
                    # Wait for result with timeout
                    result = future.get(timeout=self.timeout_seconds)
                    
                    if result['success']:
                        log_universal('INFO', 'Sequential', f"Successfully processed: {filename}")
                        return True
                    else:
                        error_msg = result.get('error', 'Unknown error')
                        log_universal('ERROR', 'Sequential', f"Multiprocessing failed for {filename}: {error_msg}")
                        self.db_manager.mark_analysis_failed(file_path, filename, error_msg, file_hash)
                        return False
                        
                except mp.TimeoutError:
                    # Process timed out
                    log_universal('ERROR', 'Sequential', f"Multiprocessing timed out for {filename}")
                    self.db_manager.mark_analysis_failed(file_path, filename, "Analysis timed out", file_hash)
                    return False
                    
        except Exception as e:
            log_universal('ERROR', 'Sequential', f"Error in multiprocessing for {filename}: {e}")
            # Calculate file hash for error case
            try:
                file_hash = self._calculate_file_hash(file_path)
            except:
                file_hash = None
            self.db_manager.mark_analysis_failed(file_path, filename, str(e), file_hash)
            return False

    def _process_single_file(self, file_path: str, force_reextract: bool = False) -> bool:
        """
        Process a single file with timeout and memory monitoring.
        This method now uses multiprocessing for better isolation.
        
        Args:
            file_path: Path to the file to process
            force_reextract: If True, bypass cache
            
        Returns:
            True if successful, False otherwise
        """
        return self._process_single_file_multiprocessing(file_path, force_reextract)

    def _extract_features_in_process(self, file_path: str, force_reextract: bool = False) -> bool:
        """
        Extract features sequentially in the same process.
        This method is kept for backward compatibility but now uses multiprocessing.
        
        Args:
            file_path: Path to the file
            force_reextract: If True, bypass cache
            
        Returns:
            True if successful, False otherwise
        """
        return self._process_single_file_multiprocessing(file_path, force_reextract)

    def _create_worker_script(self, script_path: str):
        """Create the worker script for isolated processing."""
        worker_script = '''#!/usr/bin/env python3
"""
Sequential worker script for isolated file processing.
"""

import os
import sys
import json
import argparse
import signal
import traceback
from pathlib import Path

# Add the correct path to find the modules
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def timeout_handler(signum, frame):
    raise TimeoutError("Worker timed out")

def process_file_isolated(file_path: str, force_reextract: bool = False) -> dict:
    """Process a single file in isolation."""
    try:
        # Import required modules with proper error handling
        try:
            from core.audio_analyzer import AudioAnalyzer
            from core.database import DatabaseManager
        except ImportError as e:
            return {'success': False, 'error': f'Import error: {e}'}
        
        # Initialize components
        db_manager = DatabaseManager()
        analyzer = AudioAnalyzer(processing_mode='sequential')
        
        # Process file
        result = analyzer.analyze_audio_file(file_path, force_reextract)
        
        if result:
            # Save to database
            filename = os.path.basename(file_path)
            file_size_bytes = os.path.getsize(file_path)
            file_hash = analyzer._calculate_file_hash(file_path)
            
            # Prepare analysis data
            analysis_data = result.get('features', {})
            analysis_data['status'] = 'analyzed'
            analysis_data['analysis_type'] = 'full'
            
            # Extract metadata
            metadata = result.get('metadata', {})
            
            # Save to database
            success = db_manager.save_analysis_result(
                file_path=file_path,
                filename=filename,
                file_size_bytes=file_size_bytes,
                file_hash=file_hash,
                analysis_data=analysis_data,
                metadata=metadata,
                discovery_source='file_system'
            )
            
            return {'success': success}
        else:
            return {'success': False, 'error': 'Analysis failed - no valid result returned'}
            
    except Exception as e:
        error_details = f"{str(e)}\\n{traceback.format_exc()}"
        return {'success': False, 'error': error_details}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file-path', required=True)
    parser.add_argument('--force-reextract', default='False')
    parser.add_argument('--output-file', required=True)
    parser.add_argument('--timeout', type=int, default=600)
    
    args = parser.parse_args()
    
    # Set up timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(args.timeout)
    
    try:
        # Process file
        result = process_file_isolated(
            args.file_path, 
            force_reextract=args.force_reextract.lower() == 'true'
        )
        
        # Write result to output file
        with open(args.output_file, 'w') as f:
            json.dump(result, f)
        
        sys.exit(0 if result['success'] else 1)
        
    except Exception as e:
        # Write error result
        error_result = {'success': False, 'error': str(e)}
        with open(args.output_file, 'w') as f:
            json.dump(error_result, f)
        sys.exit(1)
    finally:
        signal.alarm(0)

if __name__ == "__main__":
    main()
'''
        
        with open(script_path, 'w') as f:
            f.write(worker_script)
        
        # Make script executable
        os.chmod(script_path, 0o755)
        log_universal('INFO', 'Sequential', f'Created worker script: {script_path}')

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
            
            # Check if this is a long audio track (20+ minutes)
            from .audio_analyzer import AudioAnalyzer
            temp_analyzer = AudioAnalyzer()
            is_long_audio = temp_analyzer._is_long_audio_track(file_path)
            
            # Enable MusiCNN for all tracks in sequential processing (needed for categorization)
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
            
            log_universal('DEBUG', 'Sequential', f"Analysis config for {os.path.basename(file_path)}: {analysis_config['analysis_type']}")
            log_universal('DEBUG', 'Sequential', f"MusiCNN enabled: {enable_musicnn} (always enabled for sequential categorization)")
            
            return analysis_config
            
        except Exception as e:
            log_universal('WARNING', 'Sequential', f"Error getting analysis config for {file_path}: {e}")
            # Return basic analysis config as fallback
            return {
                'analysis_type': 'basic',
                'use_full_analysis': False,
                'EXTRACT_RHYTHM': True,
                'EXTRACT_SPECTRAL': True,
                'EXTRACT_LOUDNESS': True,
                'EXTRACT_KEY': True,
                'EXTRACT_MFCC': True,
                'EXTRACT_MUSICNN': False,  # Disabled in fallback
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
            log_universal('WARNING', 'Sequential', f"Could not calculate hash for {file_path}: {e}")
            return "unknown"

    def _cleanup_memory(self):
        """Force memory cleanup."""
        try:
            # Force garbage collection
            gc.collect()
            
            # Log memory after cleanup
            import psutil
            memory = psutil.virtual_memory()
            memory_used_gb = memory.used / (1024**3)
            log_universal('DEBUG', 'Sequential', f"Memory cleanup completed: {memory_used_gb:.2f}GB used")
            
        except Exception as e:
            log_universal('ERROR', 'Sequential', f"Error during memory cleanup: {e}")

    def get_config(self) -> Dict[str, Any]:
        """Get current analyzer configuration."""
        return {
            'timeout_seconds': self.timeout_seconds,
            'memory_threshold_percent': self.memory_threshold_percent,
            'rss_limit_gb': self.rss_limit_gb
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
            if 'rss_limit_gb' in new_config:
                self.rss_limit_gb = new_config['rss_limit_gb']
            
            log_universal('INFO', 'Sequential', f"Updated sequential analyzer configuration: {new_config}")
            return True
            
        except Exception as e:
            log_universal('ERROR', 'Sequential', f"Error updating sequential analyzer configuration: {e}")
            return False


# Global sequential analyzer instance - created lazily to avoid circular imports
_sequential_analyzer_instance = None

def get_sequential_analyzer() -> 'SequentialAnalyzer':
    """Get the global sequential analyzer instance, creating it if necessary."""
    global _sequential_analyzer_instance
    if _sequential_analyzer_instance is None:
        _sequential_analyzer_instance = SequentialAnalyzer()
    return _sequential_analyzer_instance 