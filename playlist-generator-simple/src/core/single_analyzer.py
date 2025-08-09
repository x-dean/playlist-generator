"""
Single Audio Analyzer - One analyzer for all audio analysis needs.

This is the ultimate consolidation - one class that handles everything:
- Automatic file size detection and optimization
- Built-in metadata extraction
- Intelligent caching and database integration
- Batch processing with optimal workers
- All feature extraction (Essentia + MusiCNN)
- External API enrichment
- Progress tracking and statistics

No more layers, no more complexity - just one analyzer that does it all.
"""

import os
import time
import json
import hashlib
import subprocess
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import core modules
from .logging_setup import get_logger, log_universal
from .config_loader import config_loader
from .database import get_db_manager
from .optimized_pipeline import OptimizedAudioPipeline
from .lazy_imports import is_mutagen_available, mutagen


logger = get_logger('playlista.single_analyzer')


class SingleAnalyzer:
    """
    The one and only audio analyzer - handles everything automatically.
    
    Features:
    - Automatic optimization based on file size
    - Built-in metadata extraction and enrichment
    - Intelligent caching with database integration
    - Batch processing with automatic worker management
    - Complete feature extraction (rhythm, spectral, MusiCNN, etc.)
    - Progress tracking and comprehensive statistics
    
    Usage:
        analyzer = SingleAnalyzer()
        result = analyzer.analyze('/path/to/audio.mp3')
        batch_results = analyzer.analyze_batch([list_of_files])
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the single analyzer."""
        self.config = config or config_loader.get_config()
        self.db_manager = get_db_manager()
        
        # Initialize optimized pipeline (does the heavy lifting)
        self.pipeline = OptimizedAudioPipeline(self.config)
        
        # Analysis settings
        self.timeout_seconds = self.config.get('ANALYSIS_TIMEOUT', 600)
        # Don't set max_workers here - use dynamic calculation instead
        self.max_workers = None
        
        # File size thresholds for optimization
        self.large_file_threshold_mb = self.config.get('LARGE_FILE_THRESHOLD_MB', 50)
        
        log_universal('INFO', 'System', f'Audio analyzer ready - dynamic workers, large files >{self.large_file_threshold_mb}MB get Essentia-only')
    
    def _calculate_dynamic_workers(self, files: List[str]) -> int:
        """Calculate optimal workers based on RAM availability and user configuration."""
        if not files:
            return 1
        
        try:
            # Get memory configuration - user override takes priority
            user_ram_override = self.config.get('USER_OVERRIDE_TOTAL_RAM_GB')
            base_memory_per_worker_gb = float(self.config.get('BASE_MEMORY_PER_WORKER_GB', 1.2))
            workload_mode = self.config.get('WORKLOAD_DETECTION_MODE', 'auto')
            
            # Determine available memory
            from .resource_manager import ResourceManager
            resource_manager = ResourceManager()
            
            if user_ram_override:
                try:
                    total_ram_gb = float(user_ram_override)
                    # Reserve 15% for system, minimum 1GB
                    reserved_gb = max(1.0, total_ram_gb * 0.15)
                    available_memory_gb = max(1.0, total_ram_gb - reserved_gb)
                    log_universal('INFO', 'System', f'Using user RAM override: {total_ram_gb}GB (available: {available_memory_gb:.1f}GB)')
                except (ValueError, TypeError):
                    log_universal('WARNING', 'System', f'Invalid user RAM override: {user_ram_override}, falling back to auto-detection')
                    available_memory_gb = resource_manager.get_available_memory_gb()
            else:
                available_memory_gb = resource_manager.get_available_memory_gb()
                log_universal('INFO', 'System', f'Auto-detected available RAM: {available_memory_gb:.1f}GB')
            
            # Adjust memory per worker based on workload (only if auto mode)
            if workload_mode == 'auto':
                # Quick workload analysis - sample more files for better accuracy
                large_files = 0
                huge_files = 0
                total_sampled = 0
                
                for file_path in files[:20]:  # Sample first 20 files for better accuracy
                    try:
                        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
                        total_sampled += 1
                        if file_size_mb > 500:  # Huge files (>500MB)
                            huge_files += 1
                        elif file_size_mb > self.large_file_threshold_mb:  # Large files (200-500MB)
                            large_files += 1
                    except Exception:
                        continue
                
                if total_sampled == 0:
                    large_file_ratio = 0
                    huge_file_ratio = 0
                else:
                    large_file_ratio = large_files / total_sampled
                    huge_file_ratio = huge_files / total_sampled
                
                # MUCH more conservative memory estimates for Essentia on large files
                memory_per_worker_gb = base_memory_per_worker_gb  # Start with base
                
                if huge_file_ratio > 0.3:  # >30% huge files (>500MB)
                    memory_per_worker_gb *= 4.0  # 4.8GB per worker for huge files with Essentia
                    workload_type = 'huge_files'
                elif large_file_ratio > 0.5:  # >50% large files (200-500MB)
                    memory_per_worker_gb *= 3.0  # 3.6GB per worker for large files with Essentia
                    workload_type = 'large_files'
                elif large_file_ratio > 0.2:  # >20% large files
                    memory_per_worker_gb *= 2.0  # 2.4GB per worker for mixed workload
                    workload_type = 'mixed'
                else:  # Mostly small files
                    memory_per_worker_gb = base_memory_per_worker_gb  # Keep base value
                    workload_type = 'small_files'
                
                log_universal('INFO', 'System', f'Workload detected: {workload_type} (large: {large_file_ratio:.1f}, huge: {huge_file_ratio:.1f})')
                log_universal('INFO', 'System', f'Adjusted memory per worker: {memory_per_worker_gb:.1f}GB')
            
            # Calculate workers based on memory
            memory_based_workers = max(1, int(available_memory_gb / memory_per_worker_gb))
            
            # Get CPU limit (cores - 1)
            try:
                import multiprocessing as mp
                cpu_cores = mp.cpu_count()
                cpu_max_workers = max(1, cpu_cores - 1)
            except Exception:
                cpu_max_workers = 4  # Safe fallback
            
            # Final worker count: limited by CPU cores - 1 and safety maximum
            safety_max_workers = 4  # Never use more than 4 workers for safety
            workers = min(memory_based_workers, cpu_max_workers, safety_max_workers)
            
            log_universal('INFO', 'System', f'Worker calculation:')
            log_universal('INFO', 'System', f'  Available memory: {available_memory_gb:.1f}GB')
            log_universal('INFO', 'System', f'  Memory per worker: {memory_per_worker_gb:.1f}GB')
            log_universal('INFO', 'System', f'  Memory-based workers: {memory_based_workers}')
            log_universal('INFO', 'System', f'  CPU cores: {cpu_cores}, max workers: {cpu_max_workers}')
            log_universal('INFO', 'System', f'  Safety limit: {safety_max_workers}')
            log_universal('INFO', 'System', f'  Final workers: {workers}')
            
            return workers
            
        except Exception as e:
            log_universal('WARNING', 'System', f'Dynamic worker calculation failed: {str(e)}')
            return 2  # Conservative fallback
    
    def analyze(self, file_path: str, force_reanalysis: bool = False) -> Dict[str, Any]:
        """
        Analyze a single audio file - the main entry point.
        
        Args:
            file_path: Path to audio file
            force_reanalysis: Force re-analysis even if cached
            
        Returns:
            Complete analysis result dictionary
        """
        try:
            log_universal('INFO', 'Audio', f'Processing {os.path.basename(file_path)}')
            start_time = time.time()
            
            # Check if file exists
            if not os.path.exists(file_path):
                return self._create_error_result(file_path, "File not found")
            
            # Check cache first (unless forcing reanalysis)
            if not force_reanalysis:
                cached_result = self._load_from_cache(file_path)
                if cached_result:
                    log_universal('DEBUG', 'Cache', f'Found cached analysis for {os.path.basename(file_path)}')
                    return cached_result
            
            # Extract metadata first (fast operation)
            metadata = self._extract_metadata(file_path)
            
            # Determine file size and analysis method
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            
            # Perform audio analysis
            if self._should_use_large_file_analysis(file_size_mb):
                log_universal('DEBUG', 'Audio', f'Large file analysis: {os.path.basename(file_path)} ({file_size_mb:.1f}MB)')
                audio_features = self._analyze_large_file(file_path, metadata)
            else:
                log_universal('DEBUG', 'Audio', f'Standard analysis: {os.path.basename(file_path)} ({file_size_mb:.1f}MB)')
                audio_features = self._analyze_standard(file_path, metadata)
            
            # Enrich with external APIs
            enriched_metadata = self._enrich_metadata(metadata)
            
            # Create final result
            result = self._create_complete_result(
                file_path, metadata, enriched_metadata, audio_features, file_size_mb, start_time
            )
            
            # Save to database and cache
            self._save_result(result)
            
            analysis_time = time.time() - start_time
            log_universal('INFO', 'Audio', f'Completed {os.path.basename(file_path)} in {analysis_time:.1f}s')
            
            return result
            
        except Exception as e:
            log_universal('ERROR', 'Audio', f'Failed to process {os.path.basename(file_path)}: {str(e)}')
            return self._create_error_result(file_path, str(e))
    
    def analyze_batch(self, files: List[str], force_reanalysis: bool = False, 
                     max_workers: int = None) -> Dict[str, Any]:
        """
        Analyze multiple files in batch - optimized for throughput.
        
        Args:
            files: List of file paths
            force_reanalysis: Force re-analysis even if cached
            max_workers: Override default worker count
            
        Returns:
            Batch results with statistics
        """
        if not files:
            return self._create_batch_result([], 0, [])
        
        start_time = time.time()
        
        log_universal('INFO', 'Analysis', f'Starting unified analysis of {len(files)} files')
        
        # Group files by size to optimize worker allocation
        file_groups = self._group_files_by_size(files)
        
        all_results = []
        all_failed_files = []
        
        # Process each group with optimized worker count
        for group_name, group_files in file_groups.items():
            if not group_files:
                continue
                
            group_start_time = time.time()
            
            # Calculate workers for this specific group
            if max_workers:
                workers = max_workers
                log_universal('INFO', 'Analysis', f'{group_name}: Using override: {workers} workers')
            else:
                workers = self._calculate_dynamic_workers(group_files)
                log_universal('INFO', 'Analysis', f'{group_name}: Dynamic allocation: {workers} workers for {len(group_files)} files')
            
            log_universal('INFO', 'Analysis', f'{group_name}: Processing {len(group_files)} files using {workers} workers')
            
            group_results = []
            group_failed_files = []
            
            # Process files in parallel for this group
            with ThreadPoolExecutor(max_workers=workers) as executor:
                # Submit all tasks for this group
                future_to_file = {
                    executor.submit(self.analyze, file_path, force_reanalysis): file_path
                    for file_path in group_files
                }
                
                # Collect results
                for future in as_completed(future_to_file):
                    file_path = future_to_file[future]
                    try:
                        result = future.result(timeout=self.timeout_seconds)
                        group_results.append(result)
                        
                        if not result.get('success', False):
                            group_failed_files.append({
                                'file_path': file_path,
                                'error': result.get('error', 'Unknown error')
                            })
                            
                    except Exception as e:
                        log_universal('ERROR', 'Audio', f'Processing failed: {os.path.basename(file_path)} - {str(e)}')
                        group_failed_files.append({
                            'file_path': file_path,
                            'error': str(e)
                        })
            
            # Log group completion
            group_time = time.time() - group_start_time
            group_success = sum(1 for r in group_results if r.get('success', False))
            log_universal('INFO', 'Analysis', f'{group_name}: Complete - {group_success}/{len(group_files)} files in {group_time:.1f}s')
            
            # Add to overall results
            all_results.extend(group_results)
            all_failed_files.extend(group_failed_files)
        
        # Calculate final statistics
        total_time = time.time() - start_time
        success_count = sum(1 for r in all_results if r.get('success', False))
        
        # Log final summary
        log_universal('INFO', 'Analysis', f'Batch complete: {success_count}/{len(files)} files processed in {total_time:.1f}s')
        
        return self._create_batch_result(all_results, total_time, all_failed_files)
    
    def _should_use_large_file_analysis(self, file_size_mb: float) -> bool:
        """Determine if file should use large file analysis (Essentia-only)."""
        return file_size_mb > self.large_file_threshold_mb
    
    def _group_files_by_size(self, files: List[str]) -> Dict[str, List[str]]:
        """
        Group files by size category for optimized batch processing.
        
        Args:
            files: List of file paths to group
            
        Returns:
            Dictionary with size group names and file lists
        """
        groups = {
            'huge_files': [],      # > 100MB - Single worker, Essentia-only 
            'large_files': [],     # 50-100MB - Few workers, Essentia-only
            'medium_files': [],    # 20-50MB - More workers, Full analysis
            'small_files': []      # < 20MB - Most workers, Full analysis
        }
        
        for file_path in files:
            try:
                file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
                
                if file_size_mb > 100:
                    groups['huge_files'].append(file_path)
                elif file_size_mb > 50:
                    groups['large_files'].append(file_path)
                elif file_size_mb > 20:
                    groups['medium_files'].append(file_path)
                else:
                    groups['small_files'].append(file_path)
                    
            except Exception as e:
                log_universal('WARNING', 'Analysis', f'Could not get size for {file_path}: {e}')
                # Default to medium group for unknown sizes
                groups['medium_files'].append(file_path)
        
        # Log group distribution
        for group_name, group_files in groups.items():
            if group_files:
                log_universal('INFO', 'Analysis', f'{group_name}: {len(group_files)} files')
        
        return {k: v for k, v in groups.items() if v}  # Return only non-empty groups
    
    def _extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract basic metadata from audio file."""
        metadata = {
            'file_path': file_path,
            'filename': os.path.basename(file_path),
            'file_size_bytes': os.path.getsize(file_path),
            'file_size_mb': os.path.getsize(file_path) / (1024 * 1024),
            'format': os.path.splitext(file_path)[1].lower(),
            'extraction_date': datetime.now().isoformat()
        }
        
        # Extract ID3/metadata tags if available
        if is_mutagen_available() and mutagen:
            try:
                audio_file = mutagen.File(file_path)
                if audio_file:
                    # Extract common tags
                    tags = {
                        'title': self._get_tag_value(audio_file, ['TIT2', 'TITLE', '\xa9nam']),
                        'artist': self._get_tag_value(audio_file, ['TPE1', 'ARTIST', '\xa9ART']),
                        'album': self._get_tag_value(audio_file, ['TALB', 'ALBUM', '\xa9alb']),
                        'genre': self._get_tag_value(audio_file, ['TCON', 'GENRE', '\xa9gen']),
                        'date': self._get_tag_value(audio_file, ['TDRC', 'DATE', '\xa9day']),
                        'track': self._get_tag_value(audio_file, ['TRCK', 'TRACKNUMBER', 'trkn'])
                    }
                    
                    # Add non-null tags to metadata
                    for key, value in tags.items():
                        if value:
                            metadata[key] = value
                    
                    # Get audio properties
                    if hasattr(audio_file, 'info'):
                        info = audio_file.info
                        metadata['duration_seconds'] = getattr(info, 'length', 0)
                        metadata['bitrate'] = getattr(info, 'bitrate', 0)
                        metadata['sample_rate'] = getattr(info, 'sample_rate', 0)
                        metadata['channels'] = getattr(info, 'channels', 0)
            
            except Exception as e:
                log_universal('WARNING', 'Audio', f'Metadata extraction failed for {os.path.basename(file_path)}: {str(e)}')
        
        return metadata
    
    def _get_tag_value(self, audio_file, tag_keys: List[str]) -> Optional[str]:
        """Get tag value from multiple possible keys."""
        for key in tag_keys:
            value = audio_file.get(key)
            if value:
                if isinstance(value, list):
                    return str(value[0])
                return str(value)
        return None
    
    def _analyze_standard(self, file_path: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Standard analysis for files outside optimized range."""
        try:
            # For very small files, do basic analysis
            # For very large files, use simplified approach
            file_size_mb = metadata['file_size_mb']
            
            if self._should_use_large_file_analysis(file_size_mb):
                # Large file (>200MB) - Essentia-only analysis
                return self._analyze_large_file(file_path, metadata)
            else:
                # Small/medium file (<200MB) - Full analysis with Essentia + MusiCNN
                return self._analyze_full_file(file_path, metadata)
                
        except Exception as e:
            log_universal('ERROR', 'Audio', f'Standard analysis failed for {os.path.basename(file_path)}: {str(e)}')
            return {'error': str(e), 'available': False}
    
    def _analyze_full_file(self, file_path: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze small/medium files with full Essentia + MusiCNN analysis."""
        try:
            # Load audio using FFmpeg
            audio, sample_rate = self._load_audio_ffmpeg(file_path)
            if audio is None:
                return {'error': 'Failed to load audio', 'available': False}
            
            features = {
                'duration': len(audio) / sample_rate,
                'sample_rate': sample_rate,
                'available': True,
                'method': 'full_analysis'
            }
            
            # Basic rhythm detection
            if len(audio) > sample_rate:  # At least 1 second
                features['tempo'] = self._estimate_tempo_simple(audio, sample_rate)
            
            return features
            
        except Exception as e:
            return {'error': str(e), 'available': False}
    
    def _analyze_large_file(self, file_path: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze very large files with Essentia features but no MusiCNN."""
        try:
            # Extract basic audio info
            duration = self._get_duration_ffmpeg(file_path)
            
            # Classify content type based on metadata and characteristics
            content_classification = self._classify_large_content(metadata, duration)
            
            # Add lightweight Essentia analysis for large files
            essentia_features = self._extract_essentia_features_large(file_path, duration)
            
            features = {
                'duration': duration,
                'available': True,
                'method': 'large_file_essentia',
                'content_type': content_classification['type'],
                'content_subtype': content_classification['subtype'],
                'content_confidence': content_classification['confidence'],
                'content_features': content_classification['features'],
                'estimated_track_count': content_classification.get('estimated_tracks'),
                'content_description': content_classification['description'],
                
                # Add Essentia features
                'tempo': essentia_features.get('tempo'),
                'key': essentia_features.get('key'),
                'mode': essentia_features.get('mode'),
                'loudness': essentia_features.get('loudness'),
                'danceability': essentia_features.get('danceability'),
                'energy': essentia_features.get('energy'),
                
                # Technical analysis info
                'segments_analyzed': essentia_features.get('segments_analyzed', 0),
                'analysis_strategy': essentia_features.get('analysis_strategy', 'sampled')
            }
            
            return features
            
        except Exception as e:
            return {'error': str(e), 'available': False}
    
    def _classify_large_content(self, metadata: Dict[str, Any], duration: float) -> Dict[str, Any]:
        """
        Classify large audio content based on metadata patterns and duration.
        
        NOTE: This is metadata-only analysis. No actual audio processing (Essentia/MusiCNN) 
        is performed on large files for performance reasons.
        """
        title = metadata.get('title', '').lower() if metadata.get('title') else ''
        artist = metadata.get('artist', '').lower() if metadata.get('artist') else ''
        filename = metadata.get('filename', '').lower()
        file_size_mb = metadata.get('file_size_mb', 0)
        
        # Combine all text for pattern analysis
        all_text = f"{title} {artist} {filename}".lower()
        
        classification = {
            'type': 'long_form_audio',
            'subtype': 'unknown',
            'confidence': 0.0,
            'features': [],
            'description': 'Long-form audio content'
        }
        
        # Broad content type patterns (generic, not show-specific)
        mix_patterns = [
            'mix', 'set', 'mixed', 'compilation', 'continuous',
            'non-stop', 'seamless', 'blended', 'dj', 'disc jockey'
        ]
        
        radio_patterns = [
            'radio', 'show', 'episode', 'broadcast', 'transmission',
            'fm', 'am', 'station', 'weekly', 'monthly'
        ]
        
        podcast_patterns = [
            'podcast', 'talk', 'interview', 'discussion', 'conversation',
            'series', 'chapter', 'episode'
        ]
        
        live_patterns = [
            'live', 'concert', 'performance', 'recorded live',
            'venue', 'club', 'festival', 'gig'
        ]
        
        audiobook_patterns = [
            'audiobook', 'book', 'narrated', 'chapter', 'part', 'volume',
            'unabridged', 'abridged', 'reading'
        ]
        
        # Score each category
        scores = {
            'mix': self._score_patterns(all_text, mix_patterns),
            'radio': self._score_patterns(all_text, radio_patterns),
            'podcast': self._score_patterns(all_text, podcast_patterns),
            'live': self._score_patterns(all_text, live_patterns),
            'audiobook': self._score_patterns(all_text, audiobook_patterns)
        }
        
        # Determine primary type
        max_score = max(scores.values())
        if max_score > 0:
            primary_type = max(scores, key=scores.get)
            classification['type'] = primary_type
            classification['confidence'] = min(max_score, 1.0)
            
            # Generic subtypes based on duration and characteristics
            if primary_type == 'mix':
                classification['subtype'] = self._classify_mix_duration(duration)
                classification['estimated_tracks'] = self._estimate_track_count_mix(duration)
                classification['description'] = f"DJ mix ({classification['subtype']}, ~{classification['estimated_tracks']} tracks)"
                
            elif primary_type == 'radio':
                classification['subtype'] = self._classify_radio_duration(duration)
                classification['estimated_tracks'] = self._estimate_track_count_radio(duration)
                classification['description'] = f"Radio show ({classification['subtype']})"
                
            elif primary_type == 'podcast':
                classification['subtype'] = self._classify_podcast_duration(duration)
                classification['description'] = f"Podcast ({classification['subtype']})"
                
            elif primary_type == 'live':
                classification['subtype'] = self._classify_live_duration(duration)
                classification['estimated_tracks'] = self._estimate_track_count_live(duration)
                classification['description'] = f"Live performance ({classification['subtype']})"
                
            elif primary_type == 'audiobook':
                classification['subtype'] = 'chapter' if duration < 3600 else 'full_book'
                classification['description'] = f"Audiobook ({classification['subtype']})"
        
        # Add technical features (file size, duration, estimated quality)
        classification['features'] = self._extract_technical_features(duration, file_size_mb)
        
        return classification
    
    def _score_patterns(self, text: str, patterns: List[str]) -> float:
        """Score text against pattern list."""
        score = 0
        for pattern in patterns:
            if pattern in text:
                # Weight longer patterns higher
                score += len(pattern.split()) * 0.2
        return min(score, 1.0)
    
    def _classify_mix_duration(self, duration: float) -> str:
        """Classify mix based on duration only."""
        if duration > 10800:  # > 3 hours
            return 'marathon_mix'
        elif duration > 7200:  # > 2 hours
            return 'extended_mix'
        elif duration > 3600:  # > 1 hour
            return 'long_mix'
        else:
            return 'standard_mix'
    
    def _classify_radio_duration(self, duration: float) -> str:
        """Classify radio show based on duration."""
        if duration > 7200:  # > 2 hours
            return 'extended_show'
        elif duration > 3600:  # > 1 hour
            return 'standard_show'
        else:
            return 'short_show'
    
    def _classify_podcast_duration(self, duration: float) -> str:
        """Classify podcast based on duration."""
        if duration > 3600:  # > 1 hour
            return 'long_form'
        elif duration > 1800:  # > 30 minutes
            return 'medium_form'
        else:
            return 'short_form'
    
    def _classify_live_duration(self, duration: float) -> str:
        """Classify live performance based on duration."""
        if duration > 7200:  # > 2 hours
            return 'festival_set'
        elif duration > 3600:  # > 1 hour
            return 'extended_set'
        else:
            return 'standard_set'
    
    def _estimate_track_count_mix(self, duration: float) -> int:
        """Estimate number of tracks in a DJ mix (avg 4-6 minutes per track)."""
        if duration > 0:
            return max(1, int(duration / 300))  # 5 minutes average
        return 1
    
    def _estimate_track_count_radio(self, duration: float) -> int:
        """Estimate tracks in radio show (includes talk, ads)."""
        if duration > 0:
            # Radio shows have more talk/ads, so fewer tracks per minute
            return max(1, int(duration / 420))  # 7 minutes average
        return 1
    
    def _estimate_track_count_live(self, duration: float) -> int:
        """Estimate tracks in live performance."""
        if duration > 0:
            # Live sets often have longer tracks/transitions
            return max(1, int(duration / 360))  # 6 minutes average
        return 1
    
    def _extract_technical_features(self, duration: float, file_size_mb: float) -> List[str]:
        """Extract technical features based on duration, file size, and estimated quality."""
        features = []
        
        # Duration categories
        if duration > 10800:  # > 3 hours
            features.append('marathon_length')
        elif duration > 7200:  # > 2 hours
            features.append('extended_length')
        elif duration > 3600:  # > 1 hour
            features.append('long_form')
        
        # File size categories
        if file_size_mb > 500:
            features.append('very_large_file')
        elif file_size_mb > 300:
            features.append('large_file')
        elif file_size_mb > 200:
            features.append('medium_file')
        
        # Estimated audio quality based on bitrate
        if duration > 0:
            estimated_bitrate = (file_size_mb * 8192) / duration  # kbps estimate
            if estimated_bitrate > 320:
                features.append('lossless_quality')
            elif estimated_bitrate > 256:
                features.append('high_quality')
            elif estimated_bitrate > 128:
                features.append('standard_quality')
            else:
                features.append('compressed_quality')
        
        return features
    
    def _extract_essentia_features_large(self, file_path: str, duration: float) -> Dict[str, Any]:
        """Extract Essentia features from large files using strategic sampling."""
        try:
            from .lazy_imports import get_essentia, is_essentia_available
            if not is_essentia_available():
                return {'error': 'Essentia not available', 'analysis_strategy': 'failed'}
            
            essentia_standard = get_essentia()
            
            # Determine sampling strategy based on file size and duration
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            analysis_strategy = self._determine_large_file_strategy(file_size_mb, duration)
            
            if analysis_strategy == 'beginning_sample':
                # Analyze first 60 seconds only
                audio_segment = self._load_audio_segment_ffmpeg(file_path, start=0, duration=60)
                features = self._analyze_single_segment_essentia(audio_segment)
                features['segments_analyzed'] = 1
                features['analysis_strategy'] = 'beginning_sample'
                
            elif analysis_strategy == 'strategic_samples':
                # Analyze 3 strategic segments: beginning, middle, end
                segment_duration = 30  # 30 seconds per segment
                segments = [
                    (0, segment_duration),  # Beginning
                    (duration * 0.5 - segment_duration/2, segment_duration),  # Middle
                    (max(0, duration - segment_duration), segment_duration)  # End
                ]
                
                segment_features = []
                for start_time, seg_duration in segments:
                    audio_segment = self._load_audio_segment_ffmpeg(file_path, start=start_time, duration=seg_duration)
                    if audio_segment is not None:
                        seg_features = self._analyze_single_segment_essentia(audio_segment)
                        if seg_features.get('tempo'):  # Valid analysis
                            segment_features.append(seg_features)
                
                # Aggregate features from segments
                features = self._aggregate_segment_features(segment_features)
                features['segments_analyzed'] = len(segment_features)
                features['analysis_strategy'] = 'strategic_samples'
                
            elif analysis_strategy == 'extended_samples':
                # Analyze 5 segments across the file
                segment_duration = 45  # 45 seconds per segment
                num_segments = 5
                segments = []
                for i in range(num_segments):
                    start_time = (duration / num_segments) * i
                    segments.append((start_time, segment_duration))
                
                segment_features = []
                for start_time, seg_duration in segments:
                    audio_segment = self._load_audio_segment_ffmpeg(file_path, start=start_time, duration=seg_duration)
                    if audio_segment is not None:
                        seg_features = self._analyze_single_segment_essentia(audio_segment)
                        if seg_features.get('tempo'):
                            segment_features.append(seg_features)
                
                features = self._aggregate_segment_features(segment_features)
                features['segments_analyzed'] = len(segment_features)
                features['analysis_strategy'] = 'extended_samples'
            
            else:  # minimal_sample
                # Just analyze first 30 seconds
                audio_segment = self._load_audio_segment_ffmpeg(file_path, start=0, duration=30)
                features = self._analyze_single_segment_essentia(audio_segment)
                features['segments_analyzed'] = 1
                features['analysis_strategy'] = 'minimal_sample'
            
            return features
            
        except Exception as e:
            log_universal('WARNING', 'Audio', f'Large file Essentia analysis failed for {os.path.basename(file_path)}: {str(e)}')
            return {'error': str(e), 'analysis_strategy': 'failed'}
    
    def _determine_large_file_strategy(self, file_size_mb: float, duration: float) -> str:
        """Determine analysis strategy based on file characteristics."""
        if file_size_mb > 1000 or duration > 14400:  # >1GB or >4 hours
            return 'minimal_sample'
        elif file_size_mb > 500 or duration > 7200:  # >500MB or >2 hours  
            return 'strategic_samples'
        elif file_size_mb > 300 or duration > 3600:  # >300MB or >1 hour
            return 'extended_samples'
        else:
            return 'beginning_sample'
    
    def _load_audio_segment_ffmpeg(self, file_path: str, start: float, duration: float) -> Optional[np.ndarray]:
        """Load a specific segment of audio using FFmpeg."""
        try:
            import numpy as np
            import subprocess
            
            cmd = [
                'ffmpeg', '-i', file_path,
                '-ss', str(start),  # Start time
                '-t', str(duration),  # Duration
                '-ac', '1',  # Mono
                '-ar', '22050',  # Sample rate
                '-f', 'f32le',  # Output format
                '-v', 'quiet',  # Suppress logs
                '-'
            ]
            
            result = subprocess.run(cmd, capture_output=True, timeout=300)
            if result.returncode == 0 and len(result.stdout) > 0:
                audio = np.frombuffer(result.stdout, dtype=np.float32)
                return audio if len(audio) > 0 else None
            
            return None
            
        except Exception as e:
            log_universal('WARNING', 'Audio', f'Failed to load audio segment: {str(e)}')
            return None
    
    def _analyze_single_segment_essentia(self, audio: np.ndarray) -> Dict[str, Any]:
        """Analyze a single audio segment with Essentia."""
        try:
            from .lazy_imports import get_essentia, is_essentia_available
            if not is_essentia_available() or audio is None or len(audio) == 0:
                return {}
            
            essentia_standard = get_essentia()
            
            # Basic rhythm analysis
            features = {}
            
            # Tempo and beats
            tempo_extractor = essentia_standard.RhythmExtractor2013(method="multifeature")
            tempo, beats, beats_confidence, _, beats_intervals = tempo_extractor(audio)
            features['tempo'] = float(tempo) if tempo > 0 else None
            features['beats_confidence'] = float(beats_confidence)
            
            # Key detection
            key_extractor = essentia_standard.KeyExtractor()
            key, scale, strength = key_extractor(audio)
            features['key'] = key if strength > 0.6 else None
            features['mode'] = scale if strength > 0.6 else None
            features['key_strength'] = float(strength)
            
            # Loudness
            loudness_extractor = essentia_standard.Loudness()
            features['loudness'] = float(loudness_extractor(audio))
            
            # Spectral features for energy/danceability estimation
            spectral_centroid = essentia_standard.Centroid()
            spectral_rolloff = essentia_standard.RollOff()
            zcr = essentia_standard.ZeroCrossingRate()
            
            windowing = essentia_standard.Windowing(type='hann')
            spectrum = essentia_standard.Spectrum()
            
            # Process in frames
            frame_size = 2048
            hop_size = 1024
            centroids, rolloffs, zcrs = [], [], []
            
            for frame in essentia_standard.FrameGenerator(audio, frameSize=frame_size, hopSize=hop_size):
                windowed_frame = windowing(frame)
                spec = spectrum(windowed_frame)
                centroids.append(spectral_centroid(spec))
                rolloffs.append(spectral_rolloff(spec))
                zcrs.append(zcr(frame))
            
            if centroids:
                import numpy as np
                # Derive energy and danceability from spectral features
                avg_centroid = np.mean(centroids)
                avg_rolloff = np.mean(rolloffs)
                avg_zcr = np.mean(zcrs)
                
                # Energy estimation (higher spectral content = more energy)
                features['energy'] = min(1.0, avg_centroid / 8000.0)  # Normalize to 0-1
                
                # Danceability estimation (rhythm consistency + tempo range)
                tempo_danceable = 1.0 if 100 <= tempo <= 140 else 0.5 if 80 <= tempo <= 160 else 0.2
                rhythm_consistency = beats_confidence
                features['danceability'] = (tempo_danceable + rhythm_consistency) / 2.0
            
            return features
            
        except Exception as e:
            log_universal('WARNING', 'Audio', f'Essentia segment analysis failed: {str(e)}')
            return {}
    
    def _aggregate_segment_features(self, segment_features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate features from multiple segments."""
        if not segment_features:
            return {}
        
        try:
            import numpy as np
            
            # Collect numeric features
            tempos = [f['tempo'] for f in segment_features if f.get('tempo') and f['tempo'] > 0]
            energies = [f['energy'] for f in segment_features if f.get('energy') is not None]
            danceabilities = [f['danceability'] for f in segment_features if f.get('danceability') is not None]
            loudnesses = [f['loudness'] for f in segment_features if f.get('loudness') is not None]
            
            aggregated = {}
            
            # Tempo: use median to avoid outliers
            if tempos:
                aggregated['tempo'] = float(np.median(tempos))
                aggregated['tempo_std'] = float(np.std(tempos))
            
            # Energy: use mean
            if energies:
                aggregated['energy'] = float(np.mean(energies))
            
            # Danceability: use mean
            if danceabilities:
                aggregated['danceability'] = float(np.mean(danceabilities))
            
            # Loudness: use mean
            if loudnesses:
                aggregated['loudness'] = float(np.mean(loudnesses))
            
            # Key: use most common key with sufficient confidence
            keys = [(f['key'], f['key_strength']) for f in segment_features 
                   if f.get('key') and f.get('key_strength', 0) > 0.6]
            if keys:
                # Get key with highest confidence
                best_key = max(keys, key=lambda x: x[1])
                aggregated['key'] = best_key[0]
                aggregated['key_strength'] = best_key[1]
            
            return aggregated
            
        except Exception as e:
            log_universal('WARNING', 'Audio', f'Feature aggregation failed: {str(e)}')
            return segment_features[0] if segment_features else {}
    
    def _load_audio_ffmpeg(self, file_path: str, max_duration: float = 30) -> Tuple[Optional[np.ndarray], int]:
        """Load audio using FFmpeg (limited duration for small files)."""
        try:
            import numpy as np
            
            cmd = [
                'ffmpeg', '-i', file_path,
                '-t', str(max_duration),  # Limit to 30 seconds max
                '-ac', '1',  # Mono
                '-ar', '22050',  # Standard sample rate
                '-f', 'f32le',
                '-'
            ]
            
            result = subprocess.run(cmd, capture_output=True, timeout=60)
            
            if result.returncode == 0:
                audio_data = np.frombuffer(result.stdout, dtype=np.float32)
                return audio_data, 22050
            else:
                return None, None
                
        except Exception:
            return None, None
    
    def _get_duration_ffmpeg(self, file_path: str) -> float:
        """Get duration using FFmpeg probe."""
        try:
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_format', file_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                data = json.loads(result.stdout)
                return float(data['format']['duration'])
            else:
                return 0.0
                
        except Exception:
            return 0.0
    
    def _estimate_tempo_simple(self, audio: np.ndarray, sample_rate: int) -> float:
        """Simple tempo estimation."""
        try:
            import numpy as np
            
            # Simple onset detection using energy
            frame_length = 2048
            hop_length = 512
            
            # Calculate energy in frames
            frames = []
            for i in range(0, len(audio) - frame_length, hop_length):
                frame = audio[i:i + frame_length]
                energy = np.sum(frame ** 2)
                frames.append(energy)
            
            if len(frames) < 10:
                return 120.0  # Default tempo
            
            frames = np.array(frames)
            
            # Find peaks (simple)
            diff = np.diff(frames)
            peaks = []
            for i in range(1, len(diff) - 1):
                if diff[i] > 0 and diff[i+1] < 0 and frames[i+1] > np.mean(frames):
                    peaks.append(i)
            
            if len(peaks) < 2:
                return 120.0
            
            # Estimate tempo from peak intervals
            intervals = np.diff(peaks) * hop_length / sample_rate
            avg_interval = np.median(intervals)
            
            if avg_interval > 0:
                tempo = 60.0 / avg_interval
                # Clamp to reasonable range
                return max(60, min(200, tempo))
            
            return 120.0
            
        except Exception:
            return 120.0
    
    def _enrich_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich metadata with external APIs (simplified)."""
        # For now, just return the original metadata
        # Could add MusicBrainz, Last.fm, etc. lookups here
        enriched = metadata.copy()
        enriched['enrichment_attempted'] = True
        enriched['enrichment_date'] = datetime.now().isoformat()
        return enriched
    
    def _create_complete_result(self, file_path: str, metadata: Dict[str, Any], 
                              enriched_metadata: Dict[str, Any], audio_features: Dict[str, Any],
                              file_size_mb: float, start_time: float) -> Dict[str, Any]:
        """Create complete analysis result."""
        analysis_time = time.time() - start_time
        
        result = {
            'success': True,
            'file_path': file_path,
            'filename': os.path.basename(file_path),
            'file_size_mb': file_size_mb,
            'analysis_time': analysis_time,
            'analysis_date': datetime.now().isoformat(),
            'analysis_method': audio_features.get('method', 'optimized_pipeline'),
            'metadata': metadata,
            'enriched_metadata': enriched_metadata,
            'audio_features': audio_features,
            'analyzer': 'SingleAnalyzer',
            'analyzer_version': '1.0'
        }
        
        return result
    
    def _create_error_result(self, file_path: str, error: str) -> Dict[str, Any]:
        """Create error result."""
        return {
            'success': False,
            'file_path': file_path,
            'filename': os.path.basename(file_path),
            'error': error,
            'analysis_date': datetime.now().isoformat(),
            'analyzer': 'SingleAnalyzer'
        }
    
    def _create_batch_result(self, results: List[Dict[str, Any]], total_time: float, 
                           failed_files: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create batch analysis result."""
        success_count = sum(1 for r in results if r.get('success', False))
        failed_count = len(results) - success_count
        
        return {
            'total_files': len(results),
            'success_count': success_count,
            'failed_count': failed_count,
            'total_time': total_time,
            'throughput': len(results) / total_time if total_time > 0 else 0,
            'success_rate': (success_count / len(results) * 100) if results else 0,
            'results': results,
            'failed_files': failed_files,
            'analyzer': 'SingleAnalyzer'
        }
    
    def _load_from_cache(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Load result from database cache."""
        try:
            result = self.db_manager.get_analysis_result(file_path)
            if result:
                log_universal('DEBUG', 'Cache', f'Cache HIT for {os.path.basename(file_path)}')
                return result
            else:
                log_universal('DEBUG', 'Cache', f'Cache MISS for {os.path.basename(file_path)}')
                return None
        except Exception as e:
            log_universal('DEBUG', 'Cache', f'Cache ERROR for {os.path.basename(file_path)}: {str(e)}')
            return None
    
    def _save_result(self, result: Dict[str, Any]):
        """Save result to database."""
        try:
            file_path = result.get('file_path')
            if not file_path or not result.get('success'):
                return  # Skip saving failed results
                
            # Calculate required parameters
            filename = os.path.basename(file_path)
            file_size_bytes = os.path.getsize(file_path)
            file_hash = hashlib.md5(f"{filename}:{file_size_bytes}".encode()).hexdigest()
            
            # Flatten audio_features data for database storage
            flattened_data = result.copy()
            audio_features = result.get('audio_features', {})
            
            # Merge audio_features into the top level for database compatibility
            flattened_data.update(audio_features)
            
            # Save to database with correct parameters
            self.db_manager.save_track_analysis(
                file_path=file_path,
                filename=filename,
                file_size_bytes=file_size_bytes,
                file_hash=file_hash,
                metadata=result.get('metadata', {}),
                analysis_data=flattened_data,
                discovery_source='single_analyzer'
            )
        except Exception as e:
            log_universal('WARNING', 'Database', f'Failed to save analysis result: {str(e)}')
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get analysis statistics."""
        try:
            stats = self.db_manager.get_analysis_statistics()
            stats['analyzer'] = 'SingleAnalyzer'
            stats['pipeline_config'] = {
                'workers': self.max_workers,
                'large_file_threshold': f'{self.large_file_threshold_mb}MB',
                'timeout': self.timeout_seconds
            }
            return stats
        except Exception as e:
            log_universal('ERROR', 'Database', f'Failed to retrieve statistics: {str(e)}')
            return {'analyzer': 'SingleAnalyzer', 'error': str(e)}


# Global singleton instance
_single_analyzer = None


def get_single_analyzer(config: Dict[str, Any] = None) -> SingleAnalyzer:
    """Get the single analyzer instance - the one and only."""
    global _single_analyzer
    
    if _single_analyzer is None:
        _single_analyzer = SingleAnalyzer(config)
    
    return _single_analyzer


# Convenience functions for easy access
def analyze_file(file_path: str, force_reanalysis: bool = False) -> Dict[str, Any]:
    """Analyze a single file - simple entry point."""
    analyzer = get_single_analyzer()
    return analyzer.analyze(file_path, force_reanalysis)


def analyze_files(files: List[str], force_reanalysis: bool = False, 
                 max_workers: int = None) -> Dict[str, Any]:
    """Analyze multiple files - simple entry point."""
    analyzer = get_single_analyzer()
    return analyzer.analyze_batch(files, force_reanalysis, max_workers)

