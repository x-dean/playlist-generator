"""
Clean Analysis Manager using only the Unified Analyzer approach.

This replaces the complex AnalysisManager with a simplified version that uses
only the OptimizedPipeline through the UnifiedAnalyzer.
"""

import os
import time
from typing import Dict, List, Any, Tuple
from pathlib import Path

from .logging_setup import get_logger, log_function_call, log_universal
from .database import DatabaseManager, get_db_manager
from .file_discovery import FileDiscovery
from .config_loader import config_loader
from .single_analyzer import get_single_analyzer


logger = get_logger('playlista.analysis_manager')


class AnalysisManager:
    """
    Simplified analysis manager using unified analyzer approach.
    
    Features:
    - Single unified analyzer that automatically selects optimal method
    - File discovery and filtering
    - Database integration
    - Progress tracking and statistics
    """
    
    def __init__(self, db_manager: DatabaseManager = None, config: Dict[str, Any] = None):
        """
        Initialize the analysis manager.
        
        Args:
            db_manager: Database manager instance (optional)
            config: Configuration dictionary (optional)
        """
        self.config = config or config_loader.get_config()
        self.db_manager = db_manager or get_db_manager()
        
        # Analysis settings
        self.timeout_seconds = self.config.get('ANALYSIS_TIMEOUT', 600)
        
        # Initialize file discovery
        self.file_discovery = FileDiscovery(
                db_manager=self.db_manager,
                config=self.config
            )
            
        # Initialize single analyzer (the one and only)
        self.single_analyzer = get_single_analyzer(self.config)
            
        log_universal('DEBUG', 'System', f"Analysis manager initialized with single analyzer")

    @log_function_call
    def select_files_for_analysis(self, music_path: str = None, force_reextract: bool = False,
                                include_failed: bool = False) -> List[str]:
        """
        Select files for analysis based on criteria.
        
        Args:
            music_path: Path to music directory (uses config default if None)
            force_reextract: Include already analyzed files
            include_failed: Include previously failed files
            
        Returns:
            List of file paths to analyze
        """
        try:
            if music_path is None:
                music_path = self.config.get('MUSIC_PATH', '/music')
            
            log_universal('DEBUG', 'FileDiscovery', f"Scanning {music_path} (force: {force_reextract}, include_failed: {include_failed})")
            
            # Discover all audio files
            all_files = self.file_discovery.discover_audio_files(music_path)
            
            if not all_files:
                    log_universal('WARNING', 'Analysis', f"No audio files found in {music_path}")
                    return []
                
            log_universal('INFO', 'Analysis', f"Found {len(all_files)} total audio files")
            
            # Filter files based on analysis criteria
            files_to_analyze = []
            
            for file_path in all_files:
                should_analyze = False
                
                if force_reextract:
                    # Force re-analysis of all files
                    should_analyze = True
                    log_universal('DEBUG', 'FileDiscovery', f"Force re-extract: {os.path.basename(file_path)}")
                    
                elif not self.single_analyzer._load_from_cache(file_path):
                    # File has never been analyzed
                    should_analyze = True
                    log_universal('DEBUG', 'FileDiscovery', f"Not analyzed: {os.path.basename(file_path)}")
                    
                elif include_failed and self.db_manager.is_file_failed(file_path):
                    # Previously failed file and we want to retry
                    should_analyze = True
                    log_universal('DEBUG', 'FileDiscovery', f"Previously failed: {os.path.basename(file_path)}")
                
                if should_analyze:
                    files_to_analyze.append(file_path)
            
            log_universal('INFO', 'Analysis', f"Selected {len(files_to_analyze)} files for analysis")
            
            return files_to_analyze
            
        except Exception as e:
            log_universal('ERROR', 'Analysis', f"File selection failed: {e}")
            return []

    @log_function_call
    def analyze_files(self, files: List[str], force_reextract: bool = False,
                     max_workers: int = None) -> Dict[str, Any]:
        """
        Analyze a list of files using the unified analyzer.
        
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
        
        log_universal('INFO', 'Analysis', f"Starting unified analysis of {len(files)} files")
        log_universal('DEBUG', 'Analysis', f"  Force re-extract: {force_reextract}")
        log_universal('DEBUG', 'Analysis', f"  Max workers: {max_workers}")
        
        # Use single analyzer for all files
        result = self.single_analyzer.analyze_batch(
            files=files,
            force_reanalysis=force_reextract,
            max_workers=max_workers
        )
        
        log_universal('INFO', 'Analysis', f"Completed: {result['success_count']} success, {result['failed_count']} failed in {result['total_time']:.1f}s")
        
        return result
    
    def get_analysis_statistics(self) -> Dict[str, Any]:
        """Get analysis statistics."""
        try:
            return self.single_analyzer.get_statistics()
        except Exception as e:
            log_universal('ERROR', 'Analysis', f"Failed to get statistics: {e}")
            return {}

    def retry_failed_files(self, limit: int = None) -> Dict[str, Any]:
        """Retry failed file analysis."""
        try:
            # For now, retry is just re-analyzing failed files
            failed_files = self.db_manager.get_failed_files(limit)
            if not failed_files:
                return {'success_count': 0, 'failed_count': 0, 'total_time': 0}
            
            file_paths = [f['file_path'] for f in failed_files]
            return self.single_analyzer.analyze_batch(file_paths, force_reanalysis=True)
        except Exception as e:
            log_universal('ERROR', 'Analysis', f"Failed to retry files: {e}")
            return {'success_count': 0, 'failed_count': 0, 'total_time': 0}
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """Get processing summary and statistics."""
        try:
            stats = self.get_analysis_statistics()
            
            total_files = stats.get('total_files', 0)
            analyzed_files = stats.get('analyzed_files', 0)
            failed_files = stats.get('failed_files', 0)
            pending_files = total_files - analyzed_files - failed_files
            
            summary = {
                'total_files': total_files,
                'analyzed_files': analyzed_files,
                'failed_files': failed_files,
                'pending_files': max(0, pending_files),
                'success_rate': (analyzed_files / total_files * 100) if total_files > 0 else 0,
                'pipeline_info': stats.get('pipeline_config', {})
            }
            
            return summary
            
        except Exception as e:
            log_universal('ERROR', 'Analysis', f"Failed to get processing summary: {e}")
            return {
                'total_files': 0,
                'analyzed_files': 0,
                'failed_files': 0,
                'pending_files': 0,
                'success_rate': 0,
                'pipeline_info': {}
            }


# Global instance for shared use
_analysis_manager = None


def get_analysis_manager(db_manager: DatabaseManager = None, config: Dict[str, Any] = None) -> AnalysisManager:
    """Get shared analysis manager instance."""
    global _analysis_manager
    
    if _analysis_manager is None:
        _analysis_manager = AnalysisManager(db_manager, config)
    
    return _analysis_manager
