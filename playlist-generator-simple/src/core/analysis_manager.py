"""
Analysis Manager for Playlist Generator Simple.
Coordinates analysis operations, database integration, and analyzer selection.
"""

import os
# Configure TensorFlow logging BEFORE any imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Hide INFO and WARNING, show only ERROR
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimization messages
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU to avoid GPU-related warnings
import time
import logging
import multiprocessing as mp
from typing import List, Dict, Any, Optional, Tuple, Iterator
from datetime import datetime
from pathlib import Path

# Import local modules
from .database import DatabaseManager
from .logging_setup import get_logger, log_function_call, log_universal
from .file_discovery import FileDiscovery


logger = get_logger('playlista.analysis_manager')

# Constants
BIG_FILE_SIZE_MB = 200  # Files larger than this use sequential processing
HALF_TRACK_SIZE_MB = 25  # Files larger than this use multi-segment loading
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
    - Analysis coordination
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
            config = config_loader.get_audio_analysis_config()
        
        self.config = config
        self.file_discovery = FileDiscovery()
        
        # Analysis settings
        self.big_file_size_mb = config.get('VERY_LARGE_FILE_THRESHOLD_MB', BIG_FILE_SIZE_MB)
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
        self.failed_files_max_retries = config.get('FAILED_FILES_MAX_RETRIES', 3)
        self.failed_files_retry_delay_hours = config.get('FAILED_FILES_RETRY_DELAY_HOURS', 24)
        self.max_workers = config.get('MAX_WORKERS', None)
        self.worker_timeout_seconds = config.get('WORKER_TIMEOUT_SECONDS', DEFAULT_TIMEOUT_SECONDS)
        self.analysis_cache_enabled = config.get('ANALYSIS_CACHE_ENABLED', True)
        self.analysis_cache_expiry_hours = config.get('ANALYSIS_CACHE_EXPIRY_HOURS', 24)
        self.analysis_retry_attempts = config.get('ANALYSIS_RETRY_ATTEMPTS', 3)
        self.analysis_retry_delay_seconds = config.get('ANALYSIS_RETRY_DELAY_SECONDS', 5)
        self.analysis_progress_reporting = config.get('ANALYSIS_PROGRESS_REPORTING', False)
        self.analysis_statistics_collection = config.get('ANALYSIS_STATISTICS_COLLECTION', True)
        self.analysis_cleanup_enabled = config.get('ANALYSIS_CLEANUP_ENABLED', True)
        self.extract_musicnn = config.get('EXTRACT_MUSICNN', True)
        self.smart_analysis_enabled = config.get('SMART_ANALYSIS_ENABLED', True)
        self.analysis_type_fallback = config.get('ANALYSIS_TYPE_FALLBACK', True)
        self.resource_monitoring_enabled = config.get('RESOURCE_MONITORING_ENABLED', True)
        
        # Smart analysis thresholds
        self.min_full_analysis_size_mb = config.get('MIN_FULL_ANALYSIS_SIZE_MB', 1)
        self.min_memory_for_full_analysis_gb = config.get('MIN_MEMORY_FOR_FULL_ANALYSIS_GB', 2.0)
        self.memory_buffer_gb = config.get('MEMORY_BUFFER_GB', 1.0)
        self.max_cpu_for_full_analysis_percent = config.get('MAX_CPU_FOR_FULL_ANALYSIS_PERCENT', 80)
        self.cpu_check_interval_seconds = config.get('CPU_CHECK_INTERVAL_SECONDS', 5)
        
        # MusicNN specific thresholds
        self.musicnn_max_file_size_mb = config.get('MUSICNN_MAX_FILE_SIZE_MB', 500)
        self.musicnn_min_memory_gb = config.get('MUSICNN_MIN_MEMORY_GB', 3.0)
        self.musicnn_max_cpu_percent = config.get('MUSICNN_MAX_CPU_PERCENT', 70)
        self.musicnn_enabled = config.get('MUSICNN_ENABLED', True)
        
        # Parallel processing thresholds
        self.parallel_max_file_size_mb = config.get('PARALLEL_MAX_FILE_SIZE_MB', 100)
        self.parallel_min_memory_gb = config.get('PARALLEL_MIN_MEMORY_GB', 4.0)
        self.parallel_max_cpu_percent = config.get('PARALLEL_MAX_CPU_PERCENT', 90)
        
        # Sequential processing thresholds
        self.sequential_max_file_size_mb = config.get('SEQUENTIAL_MAX_FILE_SIZE_MB', 2000)
        self.sequential_min_memory_gb = config.get('SEQUENTIAL_MIN_MEMORY_GB', 2.0)
        self.sequential_max_cpu_percent = config.get('SEQUENTIAL_MAX_CPU_PERCENT', 70)
        
        log_universal('INFO', 'Analysis', f"Initializing AnalysisManager")
        log_universal('DEBUG', 'Analysis', f"Analysis configuration: {config}")
        
        # Initialize analyzers
        self._init_analyzers()
        
        log_universal('INFO', 'Analysis', f"AnalysisManager initialized successfully")

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
                memory_threshold_percent=self.memory_threshold_percent,
                config=self.config
            )
            
            log_universal('DEBUG', 'Analysis', "Analyzers initialized successfully")
            
        except Exception as e:
            log_universal('ERROR', 'Analysis', f"Error initializing analyzers: {e}")
            raise

    @log_function_call
    def select_files_for_analysis(self, music_path: str = None, force_reextract: bool = False,
                                include_failed: bool = False) -> List[str]:
        """
        Select files for analysis based on configuration and current state.
        
        Args:
            music_path: Path to music directory
            force_reextract: If True, re-analyze regardless of cache
            include_failed: If True, include previously failed files
            
        Returns:
            List of file paths to analyze
        """
        start_time = time.time()
        
        try:
            # Get music path from config if not provided
            if music_path is None:
                music_path = self.config.get('MUSIC_PATH', '/music')
            
            # Discover audio files
            audio_files = self.file_discovery.discover_files()
            
            if not audio_files:
                log_universal('WARNING', 'Analysis', f"No audio files found in {music_path}")
                return []
            
            log_universal('INFO', 'Analysis', f"Discovered {len(audio_files)} audio files")
            
            # Cache failed files list to avoid duplicate database calls
            failed_files_cache = None
            if not include_failed:
                failed_files_cache = self.db_manager.get_failed_analysis_files(max_retries=self.failed_files_max_retries)
                failed_files_paths = {f['file_path'] for f in failed_files_cache}
                log_universal('DEBUG', 'Analysis', f'Retrieved {len(failed_files_cache)} failed files from database')
                if failed_files_cache:
                    for failed_file in failed_files_cache:
                        log_universal('DEBUG', 'Analysis', f'Failed file: {failed_file["filename"]} - {failed_file["error_message"]}')
            
            files_to_analyze = []
            skipped_count = 0
            failed_count = 0
            
            for file_path in audio_files:
                # Check if file should be analyzed
                should_analyze = self._should_analyze_file(file_path, force_reextract, include_failed, failed_files_cache)
                
                if should_analyze:
                    files_to_analyze.append(file_path)
                else:
                    skipped_count += 1
                    
                    # Count failed files for reporting
                    if self.db_manager.get_analysis_result(file_path) is None:
                        if failed_files_cache and file_path in failed_files_paths:
                            failed_count += 1
            
            select_time = time.time() - start_time
            log_universal('INFO', 'Analysis', f"File selection completed in {select_time:.2f}s")
            log_universal('INFO', 'Analysis', f"Selected {len(files_to_analyze)} files for analysis")
            log_universal('INFO', 'Analysis', f"Skipped {skipped_count} files (already analyzed)")
            log_universal('INFO', 'Analysis', f"Previously failed: {failed_count} files")
            
            return files_to_analyze
            
        except Exception as e:
            log_universal('ERROR', 'Analysis', f"Error selecting files for analysis: {e}")
            return []

    def _should_analyze_file(self, file_path: str, force_reextract: bool, include_failed: bool, failed_files_cache: Optional[List[Dict[str, Any]]]) -> bool:
        """
        Determine if a file should be analyzed.
        
        Args:
            file_path: Path to the file
            force_reextract: If True, re-analyze regardless of cache
            include_failed: If True, include previously failed files
            failed_files_cache: Optional cached list of failed files
            
        Returns:
            True if file should be analyzed
        """
        # Check if file exists
        if not os.path.exists(file_path):
            log_universal('DEBUG', 'Analysis', f"File not found: {file_path}")
            return False
        
        # Check if file is in excluded directory
        if self.file_discovery._is_in_excluded_directory(file_path):
            log_universal('DEBUG', 'Analysis', f"File in excluded directory: {file_path}")
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
                # Check if file has been fully analyzed
                analyzed = analysis_result.get('analyzed', False)
                if analyzed:
                    log_universal('DEBUG', 'Analysis', f"File already analyzed and unchanged: {file_path}")
                    return False
                else:
                    log_universal('DEBUG', 'Analysis', f"File discovered but not yet analyzed: {file_path}")
                    return True
        
        # Check if file previously failed
        if not include_failed:
            if failed_files_cache:
                for failed_file in failed_files_cache:
                    if failed_file['file_path'] == file_path:
                        retry_count = failed_file.get('retry_count', 0)
                        if retry_count >= self.failed_files_max_retries:
                            log_universal('DEBUG', 'Analysis', f"File previously failed analysis (max retries reached): {file_path}")
                            return False
                        else:
                            log_universal('DEBUG', 'Analysis', f"File previously failed analysis (retry {retry_count}/{self.failed_files_max_retries}): {file_path}")
                            # Increment retry count when including for retry
                            self.db_manager.increment_failed_analysis_retry(file_path)
                            return True  # Include for retry if under max retries
        else:
            # If include_failed is True, include failed files for retry
            if failed_files_cache:
                for failed_file in failed_files_cache:
                    if failed_file['file_path'] == file_path:
                        retry_count = failed_file.get('retry_count', 0)
                        if retry_count < self.failed_files_max_retries:
                            log_universal('DEBUG', 'Analysis', f"Including previously failed file for retry: {file_path}")
                            # Increment retry count when including for retry
                            self.db_manager.increment_failed_analysis_retry(file_path)
                            return True
        
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
            log_universal(
                'INFO', 'Analysis',
                f"Determined analysis type for {file_path}: {analysis_config['analysis_type']}"
            )
            
            return analysis_config
            
        except Exception as e:
            log_universal('WARNING', 'Analysis', f"Error determining analysis type for {file_path}: {e}")
            
            # Log the error
            log_universal(
                'WARNING', 'Analysis',
                f"Error determining analysis type for {file_path}: {e}"
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
                    'extract_musicnn': True,  # Enabled since we handle large files with multi-segment loading
                    'extract_metadata': True
                },
                'reason': 'Error determining analysis type - using basic'
            }

    def _determine_deterministic_analysis_type(self, file_path: str, file_size_mb: float) -> Dict[str, Any]:
        """Determine analysis type based on file size only (deterministic)."""
        max_full_analysis_size_mb = self.config.get('MAX_FULL_ANALYSIS_SIZE_MB', 25)  # Aligned with half-track threshold
        use_full_analysis = file_size_mb <= max_full_analysis_size_mb
        
        features_config = {
            'extract_rhythm': True,
            'extract_spectral': True,
            'extract_loudness': True,
            'extract_key': True,
            'extract_mfcc': True,
            'extract_musicnn': True,  # Always enable MusiCNN since we handle large files with multi-segment loading
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
            log_universal(
                'INFO', 'Analysis',
                f"Smart analysis: File {file_size_mb:.1f}MB, Memory {memory_available_gb:.1f}GB, CPU {cpu_percent:.1f}%, Reason: {reason}"
            )
            return self._create_basic_analysis_config(file_size_mb, reason)
        
        # Check memory constraints
        if memory_available_gb < self.min_memory_for_full_analysis_gb:
            reason = f"Insufficient memory: {memory_available_gb:.1f}GB < {self.min_memory_for_full_analysis_gb}GB"
            log_universal(
                'WARNING', 'Analysis',
                f"Smart analysis: File {file_size_mb:.1f}MB, Memory {memory_available_gb:.1f}GB, CPU {cpu_percent:.1f}%, Reason: {reason}"
            )
            return self._create_basic_analysis_config(file_size_mb, reason)
        
        # Check CPU constraints
        if cpu_percent > self.max_cpu_for_full_analysis_percent:
            reason = f"High CPU usage: {cpu_percent:.1f}% > {self.max_cpu_for_full_analysis_percent}%"
            log_universal(
                'WARNING', 'Analysis',
                f"Smart analysis: File {file_size_mb:.1f}MB, Memory {memory_available_gb:.1f}GB, CPU {cpu_percent:.1f}%, Reason: {reason}"
            )
            return self._create_basic_analysis_config(file_size_mb, reason)
        
        # Check file size constraints - aligned with multi-segment threshold
        max_full_analysis_size_mb = self.config.get('MAX_FULL_ANALYSIS_SIZE_MB', 25)  # Aligned with multi-segment threshold
        if file_size_mb > max_full_analysis_size_mb:
            reason = f"File too large: {file_size_mb:.1f}MB > {max_full_analysis_size_mb}MB"
            log_universal(
                'INFO', 'Analysis',
                f"Smart analysis: File {file_size_mb:.1f}MB, Memory {memory_available_gb:.1f}GB, CPU {cpu_percent:.1f}%, Reason: {reason}"
            )
            return self._create_basic_analysis_config(file_size_mb, reason)
        
        # Check MusicNN-specific constraints - always enable MusiCNN since we handle large files with multi-segment loading
        musicnn_enabled = self.musicnn_enabled  # Always enable if MusicNN is available
        
        # All checks passed - use full analysis
        reason = f"Smart analysis: File {file_size_mb:.1f}MB, Memory {memory_available_gb:.1f}GB, CPU {cpu_percent:.1f}%"
        log_universal(
            'INFO', 'Analysis',
            f"Smart analysis: File {file_size_mb:.1f}MB, Memory {memory_available_gb:.1f}GB, CPU {cpu_percent:.1f}%, Reason: {reason}"
        )
        
        features_config = {
            'extract_rhythm': True,
            'extract_spectral': True,
            'extract_loudness': True,
            'extract_key': True,
            'extract_mfcc': True,
            'extract_musicnn': musicnn_enabled,
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
            'extract_musicnn': True,  # Enabled since we handle large files with multi-segment loading
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
            log_universal('WARNING', 'Analysis', "No files provided for analysis")
            return {'success_count': 0, 'failed_count': 0, 'total_time': 0}
        
        log_universal('INFO', 'Analysis', f"Starting analysis of {len(files)} files")
        log_universal('DEBUG', 'Analysis', f"  Force re-extract: {force_reextract}")
        log_universal('DEBUG', 'Analysis', f"  Max workers: {max_workers}")
        
        # Note: Individual analyzers handle their own logging
        
        start_time = time.time()
        
        # Categorize files by size
        sequential_files, parallel_half_files, parallel_full_files = self._categorize_files_by_size(files)
        
        log_universal('INFO', 'Analysis', f"File categorization:")
        log_universal('INFO', 'Analysis', f"  Sequential files (>200MB): {len(sequential_files)} (Sequential + Aggressive sampling + Lightweight categorization)")
        log_universal('INFO', 'Analysis', f"  Parallel half files (25-200MB): {len(parallel_half_files)} (Parallel + Multi-segment + Full)")
        log_universal('INFO', 'Analysis', f"  Parallel full files (<25MB): {len(parallel_full_files)} (Parallel + Full)")
        
        results = {
            'success_count': 0,
            'failed_count': 0,
            'total_time': 0,
            'sequential_files_processed': 0,
            'parallel_half_files_processed': 0,
            'parallel_full_files_processed': 0
        }
        
        # Process sequential files
        if sequential_files:
            log_universal('INFO', 'Analysis', f"Processing {len(sequential_files)} sequential files (Multi-segment loading)")
            sequential_results = self.sequential_analyzer.process_files(sequential_files, force_reextract)
            results['success_count'] += sequential_results['success_count']
            results['failed_count'] += sequential_results['failed_count']
            results['sequential_files_processed'] = len(sequential_files)
        
        # Process parallel half files
        if parallel_half_files:
            log_universal('INFO', 'Analysis', f"Processing {len(parallel_half_files)} parallel half files (Multi-segment loading)")
            log_universal('INFO', 'Analysis', f"Direct parallel processing selected for files 25-200MB")
            
            # Use threaded processing (now the default)
            log_universal('INFO', 'Analysis', f"Using threaded processing for parallel half files")
            
            # Use direct parallel processing (proven approach)
            parallel_half_results = self.parallel_analyzer.process_files(
                parallel_half_files, force_reextract, max_workers
            )
            results['success_count'] += parallel_half_results['success_count']
            results['failed_count'] += parallel_half_results['failed_count']
            results['parallel_half_files_processed'] = len(parallel_half_files)
            
            log_universal('INFO', 'Analysis', f"Parallel processing completed: {parallel_half_results['success_count']} successful, {parallel_half_results['failed_count']} failed")
        
        # Process parallel full files
        if parallel_full_files:
            log_universal('INFO', 'Analysis', f"Processing {len(parallel_full_files)} parallel full files (Full processing)")
            log_universal('INFO', 'Analysis', f"Direct parallel processing selected for files < 25MB")
            
            # Use threaded processing (now the default)
            log_universal('INFO', 'Analysis', f"Using threaded processing for parallel full files")
            
            # Use direct parallel processing (proven approach)
            parallel_full_results = self.parallel_analyzer.process_files(
                parallel_full_files, force_reextract, max_workers
            )
            results['success_count'] += parallel_full_results['success_count']
            results['failed_count'] += parallel_full_results['failed_count']
            results['parallel_full_files_processed'] = len(parallel_full_files)
            
            log_universal('INFO', 'Analysis', f"Parallel processing completed: {parallel_full_results['success_count']} successful, {parallel_full_results['failed_count']} failed")
        
        total_time = time.time() - start_time
        results['total_time'] = total_time
        
        # Note: Individual analyzers handle their own logging
        
        log_universal('INFO', 'Analysis', f"File analysis completed in {total_time:.2f}s")
        log_universal('INFO', 'Analysis', f"Analysis completed in {total_time:.2f}s")
        log_universal('INFO', 'Analysis', f"Results: {results['success_count']} successful, {results['failed_count']} failed")
        
        # Log performance
        log_universal('INFO', 'Analysis', f"File analysis completed in {total_time:.2f}s")
        
        return results

    def _categorize_files_by_size(self, files: List[str]) -> Tuple[List[str], List[str], List[str]]:
        """
        Categorize files by size for appropriate processing.
        
        Args:
            files: List of file paths
            
        Returns:
            Tuple of (sequential_files, parallel_half_files, parallel_full_files)
        """
        sequential_files = []  # >200MB: Sequential + Aggressive sampling + Full processing
        parallel_half_files = []  # 25-200MB: Parallel + Multi-segment + Full processing
        parallel_full_files = []  # <25MB: Parallel + Full processing
        
        for file_path in files:
            try:
                # Get file size from database, not filesystem
                file_size_mb = self.db_manager.get_file_size_mb(file_path)
                
                if file_size_mb > 200:  # Sequential + Aggressive sampling + Full processing
                    sequential_files.append(file_path)
                elif file_size_mb > 25:  # Parallel + Multi-segment + Full processing
                    parallel_half_files.append(file_path)
                else:  # Parallel + Full processing
                    parallel_full_files.append(file_path)
                    
            except Exception as e:
                log_universal('WARNING', 'Analysis', f"Could not determine size for {file_path}: {e}")
                # Default to sequential for unknown sizes
                sequential_files.append(file_path)
        
        return sequential_files, parallel_half_files, parallel_full_files

    def _get_analysis_config(self, file_path: str) -> Dict[str, Any]:
        """
        Get analysis configuration for a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Analysis configuration dictionary
        """
        try:
            # Get file size from database for analysis type determination
            file_size_mb = self.db_manager.get_file_size_mb(file_path)
            
            # Simplified analysis type determination based on file size
            if file_size_mb > 25:  # Files over 25MB use multi-segment
                analysis_type = 'half_track'
                use_full_analysis = False
            else:  # Files up to 25MB use full analysis
                analysis_type = 'full'
                use_full_analysis = True
            
            # Enable MusiCNN for all files
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
            
            log_universal('DEBUG', 'Analysis', f"Analysis config for {os.path.basename(file_path)}: {analysis_config['analysis_type']}")
            log_universal('DEBUG', 'Analysis', f"MusiCNN enabled: {enable_musicnn}")
            
            return analysis_config
            
        except Exception as e:
            log_universal('WARNING', 'Analysis', f"Error getting analysis config for {file_path}: {e}")
            # Return basic analysis config as fallback
            return {
                'analysis_type': 'full',
                'use_full_analysis': True,
                'EXTRACT_RHYTHM': True,
                'EXTRACT_SPECTRAL': True,
                'EXTRACT_LOUDNESS': True,
                'EXTRACT_KEY': True,
                'EXTRACT_MFCC': True,
                'EXTRACT_MUSICNN': True,
                'EXTRACT_METADATA': True,
                'EXTRACT_DANCEABILITY': True,
                'EXTRACT_ONSET_RATE': True,
                'EXTRACT_ZCR': True,
                'EXTRACT_SPECTRAL_CONTRAST': True,
                'EXTRACT_CHROMA': True
            }

    @log_function_call
    def get_analysis_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive analysis statistics.
        
        Returns:
            Dictionary with analysis statistics
        """
        log_universal('DEBUG', 'Analysis', "Generating analysis statistics")
        
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
            
            log_universal('INFO', 'Analysis', f"Analysis statistics generated")
            log_universal('INFO', 'Analysis', f"Total analyzed: {total_analyzed}, Failed: {total_failed}")
            log_universal('INFO', 'Analysis', f"Success rate: {stats['success_rate']:.1f}%")
            log_universal('INFO', 'Analysis', f"Average file size: {avg_size_mb:.1f}MB")
            
            return stats
            
        except Exception as e:
            log_universal('ERROR', 'Analysis', f"Error generating analysis statistics: {e}")
            return {}

    @log_function_call
    def final_retry_failed_files(self) -> Dict[str, Any]:
        """
        Final retry step: re-analyze failed files and move permanently failed ones to failed directory.
        
        Returns:
            Dictionary with retry results
        """
        log_universal('INFO', 'Analysis', "Starting final retry of failed files")
        
        try:
            # Get all failed files
            failed_files = self.db_manager.get_failed_analysis_files()
            
            if not failed_files:
                log_universal('INFO', 'Analysis', "No failed files to retry")
                return {'retried': 0, 'successful': 0, 'moved_to_failed_dir': 0, 'total_time': 0}
            
            log_universal('INFO', 'Analysis', f"Found {len(failed_files)} failed files for final retry")
            
            start_time = time.time()
            retried_count = 0
            successful_count = 0
            moved_to_failed_dir_count = 0
            
            # Create failed directory if it doesn't exist
            failed_dir = self.config.get('FAILED_FILES_DIR', '/music/failed')
            os.makedirs(failed_dir, exist_ok=True)
            
            for failed_file in failed_files:
                file_path = failed_file['file_path']
                filename = failed_file['filename']
                retry_count = failed_file.get('retry_count', 0)
                
                log_universal('INFO', 'Analysis', f"Final retry for {filename} (attempt {retry_count + 1})")
                
                # Check if file still exists
                if not os.path.exists(file_path):
                    log_universal('WARNING', 'Analysis', f"File no longer exists: {file_path}")
                    continue
                
                retried_count += 1
                
                # Try to analyze the file one more time
                success = self._analyze_single_file_final_retry(file_path)
                
                if success:
                    # Remove from failed files database
                    self.db_manager.delete_failed_analysis(file_path)
                    successful_count += 1
                    log_universal('INFO', 'Analysis', f"Final retry successful: {filename}")
                else:
                    # Move to failed directory
                    moved = self._move_to_failed_directory(file_path, failed_dir)
                    if moved:
                        moved_to_failed_dir_count += 1
                        log_universal('INFO', 'Analysis', f"Moved to failed directory: {filename}")
                    else:
                        log_universal('ERROR', 'Analysis', f"Failed to move to failed directory: {filename}")
            
            total_time = time.time() - start_time
            
            results = {
                'retried': retried_count,
                'successful': successful_count,
                'moved_to_failed_dir': moved_to_failed_dir_count,
                'total_time': total_time
            }
            
            log_universal('INFO', 'Analysis', f"Final retry completed in {total_time:.2f}s")
            log_universal('INFO', 'Analysis', f"Results: {successful_count} successful, {moved_to_failed_dir_count} moved to failed directory")
            
            return results
            
        except Exception as e:
            log_universal('ERROR', 'Analysis', f"Error in final retry of failed files: {e}")
            return {'retried': 0, 'successful': 0, 'moved_to_failed_dir': 0, 'total_time': 0}

    def _analyze_single_file_final_retry(self, file_path: str) -> bool:
        """
        Analyze a single file during final retry with simplified analysis.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Use simplified analysis for final retry
            analysis_config = {
                'analysis_type': 'basic',
                'use_full_analysis': False,
                'features_config': {
                    'extract_rhythm': True,
                    'extract_loudness': True,
                    'extract_metadata': True,
                    'extract_spectral': False,
                    'extract_key': False,
                    'extract_mfcc': False,
                    'extract_musicnn': True  # Enabled since we handle large files with multi-segment loading
                }
            }
            
            # Use sequential analyzer for final retry
            success = self.sequential_analyzer._process_single_file(file_path, force_reextract=True)
            return success
            
        except Exception as e:
            log_universal('ERROR', 'Analysis', f"Error in final retry analysis for {file_path}: {e}")
            return False

    def _move_to_failed_directory(self, file_path: str, failed_dir: str) -> bool:
        """
        Move a file to the failed directory.
        
        Args:
            file_path: Path to the file
            failed_dir: Path to the failed directory
            
        Returns:
            True if successful, False otherwise
        """
        try:
            import shutil
            from pathlib import Path
            
            # Get filename
            filename = os.path.basename(file_path)
            
            # Create destination path
            dest_path = os.path.join(failed_dir, filename)
            
            # Handle filename conflicts
            counter = 1
            original_dest_path = dest_path
            while os.path.exists(dest_path):
                name, ext = os.path.splitext(filename)
                dest_path = os.path.join(failed_dir, f"{name}_{counter}{ext}")
                counter += 1
            
            # Move the file
            shutil.move(file_path, dest_path)
            
            # Remove from failed analysis database
            self.db_manager.delete_failed_analysis(file_path)
            
            log_universal('INFO', 'Analysis', f"Moved {filename} to failed directory: {dest_path}")
            return True
            
        except Exception as e:
            log_universal('ERROR', 'Analysis', f"Error moving file to failed directory: {e}")
            return False

    @log_function_call
    def cleanup_failed_analysis(self, max_retries: int = 3) -> int:
        """
        Clean up failed analysis entries.
        
        Args:
            max_retries: Maximum retry count to keep
            
        Returns:
            Number of entries cleaned up
        """
        log_universal('INFO', 'Analysis', f"Cleaning up failed analysis entries (max retries: {max_retries})")
        
        try:
            failed_files = self.db_manager.get_failed_analysis_files(max_retries)
            cleaned_count = 0
            
            for failed_file in failed_files:
                if failed_file['retry_count'] > max_retries:
                    if self.db_manager.delete_failed_analysis(failed_file['file_path']):
                        cleaned_count += 1
            
            log_universal('INFO', 'Analysis', f"Cleaned up {cleaned_count} failed analysis entries")
            return cleaned_count
            
        except Exception as e:
            log_universal('ERROR', 'Analysis', f"Error cleaning up failed analysis: {e}")
            return 0

    @log_function_call
    def retry_current_failed_files(self) -> Dict[str, Any]:
        """
        Retry files that failed in the current analysis run with simplified analysis.
        
        Returns:
            Dictionary with retry results
        """
        log_universal('INFO', 'Analysis', "Starting retry of files that failed in current analysis run")
        
        try:
            # Get failed files from current run (files that were processed but failed)
            failed_files = self.db_manager.get_failed_analysis_files(max_retries=0)  # Only get files that failed in current run
            
            if not failed_files:
                log_universal('INFO', 'Analysis', "No files failed in current analysis run")
                return {'retried': 0, 'successful': 0, 'still_failed': 0, 'total_time': 0}
            
            log_universal('INFO', 'Analysis', f"Found {len(failed_files)} files that failed in current run")
            
            start_time = time.time()
            retried_count = 0
            successful_count = 0
            still_failed_count = 0
            
            for failed_file in failed_files:
                file_path = failed_file['file_path']
                filename = failed_file['filename']
                
                log_universal('INFO', 'Analysis', f"Retrying {filename}")
                
                # Check if file still exists
                if not os.path.exists(file_path):
                    log_universal('WARNING', 'Analysis', f"File no longer exists: {file_path}")
                    continue
                
                retried_count += 1
                
                # Try to analyze the file with simplified analysis
                success = self._analyze_single_file_final_retry(file_path)
                
                if success:
                    # Remove from failed files database
                    self.db_manager.delete_failed_analysis(file_path)
                    successful_count += 1
                    log_universal('INFO', 'Analysis', f"Retry successful: {filename}")
                else:
                    still_failed_count += 1
                    log_universal('WARNING', 'Analysis', f"Retry failed: {filename}")
            
            total_time = time.time() - start_time
            
            results = {
                'retried': retried_count,
                'successful': successful_count,
                'still_failed': still_failed_count,
                'total_time': total_time
            }
            
            log_universal('INFO', 'Analysis', f"Current run retry completed in {total_time:.2f}s")
            log_universal('INFO', 'Analysis', f"Results: {successful_count} successful, {still_failed_count} still failed")
            
            return results
            
        except Exception as e:
            log_universal('ERROR', 'Analysis', f"Error in retry of current failed files: {e}")
            return {'retried': 0, 'successful': 0, 'still_failed': 0, 'total_time': 0}

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
            self.big_file_size_mb = self.config.get('VERY_LARGE_FILE_THRESHOLD_MB', BIG_FILE_SIZE_MB)
            self.timeout_seconds = self.config.get('ANALYSIS_TIMEOUT_SECONDS', DEFAULT_TIMEOUT_SECONDS)
            self.memory_threshold_percent = self.config.get('MEMORY_THRESHOLD_PERCENT', DEFAULT_MEMORY_THRESHOLD_PERCENT)
            
            log_universal('INFO', 'Analysis', f"Updated analysis configuration: {new_config}")
            return True
            
        except Exception as e:
            log_universal('ERROR', 'Analysis', f"Error updating analysis configuration: {e}")
            return False


# Global analysis manager instance - created lazily to avoid circular imports
_analysis_manager_instance = None

def get_analysis_manager() -> 'AnalysisManager':
    """Get the global analysis manager instance, creating it if necessary."""
    global _analysis_manager_instance
    if _analysis_manager_instance is None:
        _analysis_manager_instance = AnalysisManager()
    return _analysis_manager_instance 
