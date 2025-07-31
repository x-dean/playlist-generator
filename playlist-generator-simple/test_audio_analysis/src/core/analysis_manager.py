"""
Analysis Manager for Playlist Generator Simple.
Coordinates analysis operations, database integration, and analyzer selection.
"""

import os
import time
import logging
import multiprocessing as mp
from typing import List, Dict, Any, Optional, Tuple, Iterator
from datetime import datetime
from pathlib import Path

# Import local modules
from .database import DatabaseManager
from .logging_setup import get_logger, log_function_call, log_performance, log_analysis_operation, log_resource_decision
from .file_discovery import FileDiscovery
from .progress_bar import get_progress_bar

logger = get_logger('playlista.analysis_manager')

# Constants
BIG_FILE_SIZE_MB = 50  # Files larger than this use sequential processing
DEFAULT_TIMEOUT_SECONDS = 300  # 5 minutes timeout for analysis
DEFAULT_MEMORY_THRESHOLD_PERCENT = 85


class AnalysisManager:
    """
    Coordinates analysis operations, database integration, and analyzer selection.
    
    Handles:
    - File selection for analysis
    - Analyzer type selection (sequential vs parallel)
    - Database integration for results
    - Analysis decision making (deterministic based on file size)
    - Progress tracking
    """
    
    def __init__(self, db_manager: DatabaseManager = None, config: Dict[str, Any] = None):
        """
        Initialize the analysis manager.
        
        Args:
            db_manager: Database manager instance (creates new one if None)
            config: Configuration dictionary (uses global config if None)
        """
        self.db_manager = db_manager or DatabaseManager()
        
        # Load configuration
        if config is None:
            from .config_loader import config_loader
            config = config_loader.get_analysis_config()
        
        self.config = config
        self.file_discovery = FileDiscovery()
        
        # Analysis settings
        self.big_file_size_mb = config.get('BIG_FILE_SIZE_MB', BIG_FILE_SIZE_MB)
        self.timeout_seconds = config.get('ANALYSIS_TIMEOUT_SECONDS', DEFAULT_TIMEOUT_SECONDS)
        self.memory_threshold_percent = config.get('MEMORY_THRESHOLD_PERCENT', DEFAULT_MEMORY_THRESHOLD_PERCENT)
        
        # Advanced analysis settings
        self.sequential_timeout_seconds = config.get('SEQUENTIAL_TIMEOUT_SECONDS', DEFAULT_TIMEOUT_SECONDS)
        self.parallel_timeout_seconds = config.get('PARALLEL_TIMEOUT_SECONDS', DEFAULT_TIMEOUT_SECONDS)
        self.audio_sample_rate = config.get('AUDIO_SAMPLE_RATE', 44100)
        self.audio_hop_size = config.get('AUDIO_HOP_SIZE', 512)
        self.audio_frame_size = config.get('AUDIO_FRAME_SIZE', 2048)
        self.force_reextract = config.get('FORCE_REEXTRACT', False)
        self.include_failed_files = config.get('INCLUDE_FAILED_FILES', False)
        self.max_workers = config.get('MAX_WORKERS', None)
        self.worker_timeout_seconds = config.get('WORKER_TIMEOUT_SECONDS', DEFAULT_TIMEOUT_SECONDS)
        self.analysis_cache_enabled = config.get('ANALYSIS_CACHE_ENABLED', True)
        self.analysis_cache_expiry_hours = config.get('ANALYSIS_CACHE_EXPIRY_HOURS', 24)
        self.analysis_retry_attempts = config.get('ANALYSIS_RETRY_ATTEMPTS', 3)
        self.analysis_retry_delay_seconds = config.get('ANALYSIS_RETRY_DELAY_SECONDS', 5)
        self.analysis_progress_reporting = config.get('ANALYSIS_PROGRESS_REPORTING', True)
        self.analysis_statistics_collection = config.get('ANALYSIS_STATISTICS_COLLECTION', True)
        self.analysis_cleanup_enabled = config.get('ANALYSIS_CLEANUP_ENABLED', True)
        self.extract_musicnn = config.get('EXTRACT_MUSICNN', True)
        self.smart_analysis_enabled = config.get('SMART_ANALYSIS_ENABLED', True)
        self.analysis_type_fallback = config.get('ANALYSIS_TYPE_FALLBACK', True)
        self.resource_monitoring_enabled = config.get('RESOURCE_MONITORING_ENABLED', True)
        
        # Smart analysis thresholds
        self.min_full_analysis_size_mb = config.get('MIN_FULL_ANALYSIS_SIZE_MB', 1)
        self.min_memory_for_full_analysis_gb = config.get('MIN_MEMORY_FOR_FULL_ANALYSIS_GB', 4.0)
        self.memory_buffer_gb = config.get('MEMORY_BUFFER_GB', 1.0)
        self.max_cpu_for_full_analysis_percent = config.get('MAX_CPU_FOR_FULL_ANALYSIS_PERCENT', 80)
        self.cpu_check_interval_seconds = config.get('CPU_CHECK_INTERVAL_SECONDS', 1)
        
        # Parallel processing thresholds
        self.parallel_max_file_size_mb = config.get('PARALLEL_MAX_FILE_SIZE_MB', 50)
        self.parallel_min_memory_gb = config.get('PARALLEL_MIN_MEMORY_GB', 2.0)
        self.parallel_max_cpu_percent = config.get('PARALLEL_MAX_CPU_PERCENT', 70)
        
        # Sequential processing thresholds
        self.sequential_max_file_size_mb = config.get('SEQUENTIAL_MAX_FILE_SIZE_MB', 200)
        self.sequential_min_memory_gb = config.get('SEQUENTIAL_MIN_MEMORY_GB', 3.0)
        self.sequential_max_cpu_percent = config.get('SEQUENTIAL_MAX_CPU_PERCENT', 85)
        
        logger.info(f"Initializing AnalysisManager")
        logger.debug(f"Analysis configuration: {config}")
        
        # Initialize analyzers
        self._init_analyzers()
        
        logger.info(f"AnalysisManager initialized successfully")

    def _init_analyzers(self):
        """Initialize sequential and parallel analyzers."""
        try:
            from .sequential_analyzer import SequentialAnalyzer
            from .parallel_analyzer import ParallelAnalyzer
            
            self.sequential_analyzer = SequentialAnalyzer(
                db_manager=self.db_manager,
                timeout_seconds=self.timeout_seconds,
                memory_threshold_percent=self.memory_threshold_percent
            )
            
            self.parallel_analyzer = ParallelAnalyzer(
                db_manager=self.db_manager,
                timeout_seconds=self.timeout_seconds,
                memory_threshold_percent=self.memory_threshold_percent
            )
            
            logger.debug("Analyzers initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing analyzers: {e}")
            raise

    @log_function_call
    def select_files_for_analysis(self, music_path: str = None, force_reextract: bool = False,
                                include_failed: bool = False) -> List[str]:
        """
        Select files for analysis based on various criteria.
        
        Args:
            music_path: Path to music directory (uses config default if None)
            force_reextract: If True, re-analyze all files
            include_failed: If True, include previously failed files
            
        Returns:
            List of file paths to analyze
        """
        if music_path is None:
            music_path = self.config.get('MUSIC_PATH', '/music')
        
        logger.info(f"Selecting files for analysis from: {music_path}")
        logger.debug(f"  Force re-extract: {force_reextract}")
        logger.debug(f"  Include failed: {include_failed}")
        
        start_time = time.time()
        
        try:
            # Discover audio files
            audio_files = self.file_discovery.discover_files()
            logger.info(f"Found {len(audio_files)} audio files")
            
            if not audio_files:
                logger.warning("️ No audio files found for analysis")
                return []
            
            # Filter files based on analysis status
            files_to_analyze = []
            skipped_count = 0
            failed_count = 0
            
            for file_path in audio_files:
                # Check if file should be analyzed
                should_analyze = self._should_analyze_file(file_path, force_reextract, include_failed)
                
                if should_analyze:
                    files_to_analyze.append(file_path)
                else:
                    skipped_count += 1
                    
                    # Count failed files for reporting
                    if self.db_manager.get_analysis_result(file_path) is None:
                        failed_entry = self.db_manager.get_failed_analysis_files()
                        if any(f['file_path'] == file_path for f in failed_entry):
                            failed_count += 1
            
            select_time = time.time() - start_time
            logger.info(f"File selection completed in {select_time:.2f}s")
            logger.info(f"Selected {len(files_to_analyze)} files for analysis")
            logger.info(f"⏭️ Skipped {skipped_count} files (already analyzed)")
            logger.info(f"Previously failed: {failed_count} files")
            
            # Log performance
            log_performance("File selection", select_time, 
                          total_files=len(audio_files),
                          selected_files=len(files_to_analyze),
                          skipped_files=skipped_count)
            
            return files_to_analyze
            
        except Exception as e:
            logger.error(f"Error selecting files for analysis: {e}")
            return []

    def _should_analyze_file(self, file_path: str, force_reextract: bool, include_failed: bool) -> bool:
        """
        Determine if a file should be analyzed.
        
        Args:
            file_path: Path to the file
            force_reextract: If True, re-analyze regardless of cache
            include_failed: If True, include previously failed files
            
        Returns:
            True if file should be analyzed
        """
        # Check if file exists
        if not os.path.exists(file_path):
            logger.debug(f"File not found: {file_path}")
            return False
        
        # Check if file is in excluded directory
        if self.file_discovery._is_in_excluded_directory(file_path):
            logger.debug(f"File in excluded directory: {file_path}")
            return False
        
        # If force re-extract, analyze all files
        if force_reextract:
            return True
        
        # Check if analysis result exists and is valid
        analysis_result = self.db_manager.get_analysis_result(file_path)
        if analysis_result:
            # Check if file has changed (using file size as simple indicator)
            current_size = os.path.getsize(file_path)
            if analysis_result['file_size_bytes'] == current_size:
                logger.debug(f"File already analyzed and unchanged: {file_path}")
                return False
        
        # Check if file previously failed
        if not include_failed:
            failed_files = self.db_manager.get_failed_analysis_files()
            if any(f['file_path'] == file_path for f in failed_files):
                logger.debug(f"File previously failed analysis: {file_path}")
                return False
        
        return True

    def determine_analysis_type(self, file_path: str) -> Dict[str, Any]:
        """
        Determine analysis type and features based on file size and smart analysis.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary with analysis configuration
        """
        try:
            # Get file size
            file_size_bytes = os.path.getsize(file_path)
            file_size_mb = file_size_bytes / (1024 * 1024)
            
            # Use smart analysis if enabled, otherwise use deterministic
            if self.smart_analysis_enabled:
                analysis_config = self._determine_smart_analysis_type(file_path, file_size_mb)
            else:
                analysis_config = self._determine_deterministic_analysis_type(file_path, file_size_mb)
            
            # Log the analysis decision with enhanced details
            log_analysis_operation(
                operation="determine_analysis_type",
                file_path=file_path,
                file_size_mb=file_size_mb,
                analysis_type=analysis_config['analysis_type'],
                success=True,
                reason=analysis_config.get('reason', 'Unknown')
            )
            
            return analysis_config
            
        except Exception as e:
            logger.warning(f"Error determining analysis type for {file_path}: {e}")
            
            # Log the error
            log_analysis_operation(
                operation="determine_analysis_type",
                file_path=file_path,
                file_size_mb=0,
                analysis_type='basic',
                success=False,
                error=str(e)
            )
            
            # Default to basic analysis on error
            return {
                'analysis_type': 'basic',
                'use_full_analysis': False,
                'file_size_mb': 0,
                'features_config': {
                    'extract_rhythm': True,
                    'extract_spectral': True,
                    'extract_loudness': True,
                    'extract_key': True,
                    'extract_mfcc': True,
                    'extract_musicnn': False,
                    'extract_metadata': True
                },
                'reason': 'Error determining analysis type - using basic'
            }

    def _determine_deterministic_analysis_type(self, file_path: str, file_size_mb: float) -> Dict[str, Any]:
        """Determine analysis type based on file size only (deterministic)."""
        max_full_analysis_size_mb = self.config.get('MAX_FULL_ANALYSIS_SIZE_MB', 100)
        use_full_analysis = file_size_mb <= max_full_analysis_size_mb
        
        features_config = {
            'extract_rhythm': True,
            'extract_spectral': True,
            'extract_loudness': True,
            'extract_key': True,
            'extract_mfcc': True,
            'extract_musicnn': use_full_analysis,
            'extract_metadata': True
        }
        
        return {
            'analysis_type': 'full' if use_full_analysis else 'basic',
            'use_full_analysis': use_full_analysis,
            'file_size_mb': file_size_mb,
            'features_config': features_config,
            'reason': f"File size {file_size_mb:.1f}MB {'≤' if use_full_analysis else '>'} {max_full_analysis_size_mb}MB threshold"
        }

    def _determine_smart_analysis_type(self, file_path: str, file_size_mb: float) -> Dict[str, Any]:
        """Determine analysis type using smart analysis with resource awareness."""
        import psutil
        
        # Get current system resources
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        memory_available_gb = memory.available / (1024**3)
        memory_used_percent = memory.percent
        
        # Check if file meets minimum size for full analysis
        if file_size_mb < self.min_full_analysis_size_mb:
            reason = "File too small for full analysis"
            log_resource_decision(file_path, file_size_mb, 'basic', reason, memory_available_gb, cpu_percent, forced=False)
            return self._create_basic_analysis_config(file_size_mb, reason)
        
        # Check memory constraints
        if memory_available_gb < self.min_memory_for_full_analysis_gb:
            reason = f"Insufficient memory: {memory_available_gb:.1f}GB < {self.min_memory_for_full_analysis_gb}GB"
            log_resource_decision(file_path, file_size_mb, 'basic', reason, memory_available_gb, cpu_percent, forced=True)
            return self._create_basic_analysis_config(file_size_mb, reason)
        
        # Check CPU constraints
        if cpu_percent > self.max_cpu_for_full_analysis_percent:
            reason = f"High CPU usage: {cpu_percent:.1f}% > {self.max_cpu_for_full_analysis_percent}%"
            log_resource_decision(file_path, file_size_mb, 'basic', reason, memory_available_gb, cpu_percent, forced=True)
            return self._create_basic_analysis_config(file_size_mb, reason)
        
        # Check file size constraints
        max_full_analysis_size_mb = self.config.get('MAX_FULL_ANALYSIS_SIZE_MB', 100)
        if file_size_mb > max_full_analysis_size_mb:
            reason = f"File too large: {file_size_mb:.1f}MB > {max_full_analysis_size_mb}MB"
            log_resource_decision(file_path, file_size_mb, 'basic', reason, memory_available_gb, cpu_percent, forced=False)
            return self._create_basic_analysis_config(file_size_mb, reason)
        
        # All checks passed - use full analysis
        reason = f"Smart analysis: File {file_size_mb:.1f}MB, Memory {memory_available_gb:.1f}GB, CPU {cpu_percent:.1f}%"
        log_resource_decision(file_path, file_size_mb, 'full', reason, memory_available_gb, cpu_percent, forced=False)
        
        features_config = {
            'extract_rhythm': True,
            'extract_spectral': True,
            'extract_loudness': True,
            'extract_key': True,
            'extract_mfcc': True,
            'extract_musicnn': True,
            'extract_metadata': True
        }
        
        return {
            'analysis_type': 'full',
            'use_full_analysis': True,
            'file_size_mb': file_size_mb,
            'features_config': features_config,
            'reason': reason
        }

    def _create_basic_analysis_config(self, file_size_mb: float, reason: str) -> Dict[str, Any]:
        """Create basic analysis configuration."""
        features_config = {
            'extract_rhythm': True,
            'extract_spectral': True,
            'extract_loudness': True,
            'extract_key': True,
            'extract_mfcc': True,
            'extract_musicnn': False,
            'extract_metadata': True
        }
        
        return {
            'analysis_type': 'basic',
            'use_full_analysis': False,
            'file_size_mb': file_size_mb,
            'features_config': features_config,
            'reason': reason
        }

    @log_function_call
    def analyze_files(self, files: List[str], force_reextract: bool = False,
                     max_workers: int = None) -> Dict[str, Any]:
        """
        Analyze a list of files using appropriate analyzer.
        
        Args:
            files: List of file paths to analyze
            force_reextract: If True, bypass cache for all files
            max_workers: Maximum number of parallel workers (auto-determined if None)
            
        Returns:
            Dictionary with analysis results and statistics
        """
        if not files:
            logger.warning("️ No files provided for analysis")
            return {'success_count': 0, 'failed_count': 0, 'total_time': 0}
        
        logger.info(f"Starting analysis of {len(files)} files")
        logger.debug(f"  Force re-extract: {force_reextract}")
        logger.debug(f"  Max workers: {max_workers}")
        
        # Note: Individual analyzers will create their own progress bars
        # No need for overall progress bar here to avoid conflicts
        
        start_time = time.time()
        
        # Categorize files by size
        big_files, small_files = self._categorize_files_by_size(files)
        
        logger.info(f"File categorization:")
        logger.info(f"  Large files (>={self.big_file_size_mb}MB): {len(big_files)}")
        logger.info(f"  Small files (<{self.big_file_size_mb}MB): {len(small_files)}")
        
        results = {
            'success_count': 0,
            'failed_count': 0,
            'total_time': 0,
            'big_files_processed': 0,
            'small_files_processed': 0
        }
        
        # Process big files sequentially
        if big_files:
            logger.info(f"Processing {len(big_files)} large files sequentially")
            big_results = self.sequential_analyzer.process_files(big_files, force_reextract)
            results['success_count'] += big_results['success_count']
            results['failed_count'] += big_results['failed_count']
            results['big_files_processed'] = len(big_files)
        
        # Process small files in parallel
        if small_files:
            logger.info(f"Processing {len(small_files)} small files in parallel")
            small_results = self.parallel_analyzer.process_files(
                small_files, force_reextract, max_workers
            )
            results['success_count'] += small_results['success_count']
            results['failed_count'] += small_results['failed_count']
            results['small_files_processed'] = len(small_files)
        
        total_time = time.time() - start_time
        results['total_time'] = total_time
        
        # Note: Individual analyzers handle their own progress bars
        
        logger.info(f"Analysis completed in {total_time:.2f}s")
        logger.info(f"Results: {results['success_count']} successful, {results['failed_count']} failed")
        
        # Log performance
        log_performance("File analysis", total_time,
                       total_files=len(files),
                       success_count=results['success_count'],
                       failed_count=results['failed_count'])
        
        return results

    def _categorize_files_by_size(self, files: List[str]) -> Tuple[List[str], List[str]]:
        """
        Categorize files by size for appropriate processing.
        
        Args:
            files: List of file paths
            
        Returns:
            Tuple of (big_files, small_files)
        """
        big_files = []
        small_files = []
        
        for file_path in files:
            try:
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                if size_mb >= self.big_file_size_mb:
                    big_files.append(file_path)
                else:
                    small_files.append(file_path)
            except Exception as e:
                logger.warning(f"Could not determine size for {file_path}: {e}")
                # Default to sequential for unknown sizes
                big_files.append(file_path)
        
        return big_files, small_files

    @log_function_call
    def get_analysis_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive analysis statistics.
        
        Returns:
            Dictionary with analysis statistics
        """
        logger.debug("Generating analysis statistics")
        
        try:
            # Get database statistics
            db_stats = self.db_manager.get_database_statistics()
            
            # Get analysis results
            all_results = self.db_manager.get_all_analysis_results()
            failed_files = self.db_manager.get_failed_analysis_files()
            
            # Calculate statistics
            total_analyzed = len(all_results)
            total_failed = len(failed_files)
            total_files = total_analyzed + total_failed
            
            # Calculate average file size
            total_size_bytes = sum(r['file_size_bytes'] for r in all_results)
            avg_size_mb = (total_size_bytes / total_analyzed) / (1024 * 1024) if total_analyzed > 0 else 0
            
            # Get recent activity (last 24 hours)
            recent_results = [r for r in all_results 
                            if (datetime.now() - datetime.fromisoformat(r['analysis_date'])).days < 1]
            
            stats = {
                'total_files_analyzed': total_analyzed,
                'total_files_failed': total_failed,
                'total_files': total_files,
                'success_rate': (total_analyzed / total_files * 100) if total_files > 0 else 0,
                'average_file_size_mb': avg_size_mb,
                'recent_analysis_count': len(recent_results),
                'database_stats': db_stats
            }
            
            logger.info(f"Analysis statistics generated")
            logger.info(f"Total analyzed: {total_analyzed}, Failed: {total_failed}")
            logger.info(f"Success rate: {stats['success_rate']:.1f}%")
            logger.info(f"Average file size: {avg_size_mb:.1f}MB")
            
            return stats
            
        except Exception as e:
            logger.error(f"Error generating analysis statistics: {e}")
            return {}

    @log_function_call
    def cleanup_failed_analysis(self, max_retries: int = 3) -> int:
        """
        Clean up failed analysis entries.
        
        Args:
            max_retries: Maximum retry count to keep
            
        Returns:
            Number of entries cleaned up
        """
        logger.info(f"Cleaning up failed analysis entries (max retries: {max_retries})")
        
        try:
            failed_files = self.db_manager.get_failed_analysis_files(max_retries)
            cleaned_count = 0
            
            for failed_file in failed_files:
                if failed_file['retry_count'] > max_retries:
                    if self.db_manager.delete_failed_analysis(failed_file['file_path']):
                        cleaned_count += 1
            
            logger.info(f"Cleaned up {cleaned_count} failed analysis entries")
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Error cleaning up failed analysis: {e}")
            return 0

    def get_config(self) -> Dict[str, Any]:
        """Get current analysis configuration."""
        return self.config.copy()
    
    def update_config(self, new_config: Dict[str, Any]) -> bool:
        """
        Update analysis configuration.
        
        Args:
            new_config: New configuration dictionary
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.config.update(new_config)
            
            # Update analyzer settings
            self.big_file_size_mb = self.config.get('BIG_FILE_SIZE_MB', BIG_FILE_SIZE_MB)
            self.timeout_seconds = self.config.get('ANALYSIS_TIMEOUT_SECONDS', DEFAULT_TIMEOUT_SECONDS)
            self.memory_threshold_percent = self.config.get('MEMORY_THRESHOLD_PERCENT', DEFAULT_MEMORY_THRESHOLD_PERCENT)
            
            logger.info(f"Updated analysis configuration: {new_config}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating analysis configuration: {e}")
            return False


# Global analysis manager instance
analysis_manager = AnalysisManager() 