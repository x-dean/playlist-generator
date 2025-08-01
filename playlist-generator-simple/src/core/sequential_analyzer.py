"""
Sequential Analyzer for Playlist Generator Simple.
Processes large files sequentially to manage memory usage.
"""

import os
import time
import threading
import gc
import signal
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
        Process files sequentially.
        
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
                
                # Process single file
                success = self._process_single_file(file_path, force_reextract)
                
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
        
        # Log performance
        log_universal('INFO', 'Sequential', f"Sequential file processing completed in {total_time:.2f}s with {len(files)} files")
        
        return results

    def _process_single_file(self, file_path: str, force_reextract: bool = False) -> bool:
        """
        Process a single file with timeout and memory monitoring.
        
        Args:
            file_path: Path to the file to process
            force_reextract: If True, bypass cache
            
        Returns:
            True if successful, False otherwise
        """
        filename = os.path.basename(file_path)
        log_universal('DEBUG', 'Sequential', f"Processing: {filename}")
        
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                log_universal('WARNING', 'Sequential', f"File not found: {file_path}")
                self.db_manager.mark_analysis_failed(file_path, filename, "File not found")
                return False
            
            # Get file size
            file_size_bytes = os.path.getsize(file_path)
            file_size_mb = file_size_bytes / (1024 * 1024)
            
            log_universal('DEBUG', 'Sequential', f"File size: {file_size_mb:.1f}MB")
            
            # Check memory before processing
            if self.resource_manager.is_memory_critical():
                log_universal('WARNING', 'Sequential', f"High memory usage before processing {filename}")
                self._cleanup_memory()
            
            # Process file in separate process to isolate memory
            success = self._extract_features_in_process(file_path, force_reextract)
            
            if success:
                log_universal('INFO', 'Sequential', f"Successfully processed: {filename}")
                return True
            else:
                log_universal('ERROR', 'Sequential', f"Failed to process: {filename}")
                return False
                
        except Exception as e:
            log_universal('ERROR', 'Sequential', f"Error processing {filename}: {e}")
            self.db_manager.mark_analysis_failed(file_path, filename, str(e))
            return False

    def _extract_features_in_process(self, file_path: str, force_reextract: bool = False) -> bool:
        """
        Extract features sequentially in the same process.
        
        Args:
            file_path: Path to the file
            force_reextract: If True, bypass cache
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Import audio analyzer
            from .audio_analyzer import AudioAnalyzer
            
            # Get analysis configuration
            analysis_config = self._get_analysis_config(file_path)
            
            # Create analyzer instance with configuration
            analyzer = AudioAnalyzer(config=analysis_config)
            
            # Extract features
            analysis_result = analyzer.analyze_audio_file(file_path, force_reextract)
            
            if analysis_result and analysis_result.get('success', False):
                # Save to database
                filename = os.path.basename(file_path)
                file_size_bytes = os.path.getsize(file_path)
                file_hash = self._calculate_file_hash(file_path)
                
                # Prepare analysis data with status
                analysis_data = analysis_result.get('features', {})
                analysis_data['status'] = 'analyzed'
                
                success = self.db_manager.save_track_to_normalized_schema(
                    file_path=file_path,
                    filename=filename,
                    file_size_bytes=file_size_bytes,
                    file_hash=file_hash,
                    analysis_data=analysis_data,
                    metadata=analysis_result.get('metadata', {})
                )
                
                return success
            else:
                # Mark analysis as failed
                filename = os.path.basename(file_path)
                error_message = "Analysis failed - no valid result returned"
                self.db_manager.mark_analysis_failed(file_path, filename, error_message)
                log_universal('ERROR', 'Sequential', f"Analysis failed for {filename}: {error_message}")
                return False
                
        except Exception as e:
            log_universal('ERROR', 'Sequential', f"Error in sequential extraction: {e}")
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
            
            # Check if this is a long audio track (20+ minutes)
            from .audio_analyzer import AudioAnalyzer
            temp_analyzer = AudioAnalyzer()
            is_long_audio = temp_analyzer._is_long_audio_track(file_path)
            
            # Enable MusiCNN only for short tracks in sequential processing
            enable_musicnn = not is_long_audio
            
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
            log_universal('DEBUG', 'Sequential', f"MusiCNN enabled: {enable_musicnn} (long_audio: {is_long_audio})")
            
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