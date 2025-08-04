"""
Audio analysis module for Playlist Generator Simple.
Handles audio feature extraction, metadata parsing, and analysis caching.
"""

import os
import time
import hashlib
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime
import json

# Import configuration and logging
from .config_loader import config_loader
from .logging_setup import get_logger, log_function_call, log_universal
from .database import get_db_manager

logger = get_logger('playlista.audio_analyzer')

# Check for required libraries
try:
    import essentia.standard as es
    ESSENTIA_AVAILABLE = True
except ImportError:
    ESSENTIA_AVAILABLE = False
    log_universal('WARNING', 'Audio', 'Essentia not available - using librosa fallback')

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    log_universal('WARNING', 'Audio', 'Librosa not available')

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    log_universal('WARNING', 'Audio', 'TensorFlow not available - MusiCNN features disabled')

try:
    from mutagen import File
    MUTAGEN_AVAILABLE = True
except ImportError:
    MUTAGEN_AVAILABLE = False
    log_universal('WARNING', 'Audio', 'Mutagen not available - metadata extraction disabled')

# Constants
DEFAULT_SAMPLE_RATE = 44100
DEFAULT_HOP_SIZE = 512
DEFAULT_FRAME_SIZE = 2048
DEFAULT_TIMEOUT_SECONDS = 600


def safe_essentia_load(audio_path: str, sample_rate: int = 44100, config: Dict[str, Any] = None, processing_mode: str = 'parallel') -> Tuple[Optional[np.ndarray], Optional[int]]:
    """
    Safely load audio file using Essentia with fallback to librosa.
    Implements aggressive memory management for parallel processing.
    
    Args:
        audio_path: Path to audio file
        sample_rate: Target sample rate
        
    Returns:
        Tuple of (audio_data, sample_rate) or (None, None) on failure
    """
    try:
        if not os.path.exists(audio_path):
            log_universal('ERROR', 'Audio', f'File not found: {audio_path}')
            return None, None
        
        # Check available memory before loading
        try:
            import psutil
            available_memory_mb = psutil.virtual_memory().available / (1024 * 1024)
            min_memory_mb = config.get('MIN_MEMORY_FOR_FULL_ANALYSIS_GB', 2.0) * 1024 if config else 2048  # Convert GB to MB
            if available_memory_mb < min_memory_mb:
                log_universal('WARNING', 'Audio', f'Low memory available ({available_memory_mb:.1f}MB) - skipping {os.path.basename(audio_path)}')
                return None, None
        except Exception:
            pass  # Continue if memory check fails
        
        # Check file size and implement configurable limits
        file_size_mb = 0  # Initialize to avoid NameError
        try:
            file_size = os.path.getsize(audio_path)
            file_size_mb = file_size / (1024 * 1024)
            
            if file_size < 1024:  # Less than 1KB
                log_universal('WARNING', 'Audio', f'File too small ({file_size} bytes): {os.path.basename(audio_path)}')
                return None, None
            
            # Configurable limits based on processing mode - INCREASED for sequential processing
            if processing_mode == 'sequential':
                max_file_size_mb = config.get('SEQUENTIAL_MAX_FILE_SIZE_MB', 5000) if config else 5000  # Increased from 2000
                warning_threshold_mb = config.get('LARGE_FILE_WARNING_THRESHOLD_MB', 1000) if config else 1000  # Increased from 500
            else:  # parallel
                max_file_size_mb = config.get('PARALLEL_MAX_FILE_SIZE_MB', 100) if config else 100
                warning_threshold_mb = config.get('LARGE_FILE_WARNING_THRESHOLD_MB', 500) if config else 500
            
            # Skip large files to prevent RAM saturation (only for parallel processing)
            if processing_mode == 'parallel' and file_size_mb > max_file_size_mb:
                log_universal('WARNING', 'Audio', f'File too large ({file_size_mb:.1f}MB): {os.path.basename(audio_path)} - skipping to prevent RAM saturation')
                return None, None
                
            # Warn for large files but don't skip them in sequential mode
            if file_size_mb > warning_threshold_mb:
                log_universal('WARNING', 'Audio', f'Large file detected ({file_size_mb:.1f}MB): {os.path.basename(audio_path)} - may cause memory issues')
                
        except Exception as e:
            log_universal('WARNING', 'Audio', f'Cannot check file size for {os.path.basename(audio_path)}: {e}')
        
        # Force garbage collection before loading
        try:
            import gc
            gc.collect()
        except Exception:
            pass
        
        try:
            import essentia.standard as es
            log_universal('DEBUG', 'Audio', f'Loading {os.path.basename(audio_path)} with Essentia MonoLoader')
            
            # For large files, try to load only a sample
            if processing_mode == 'sequential':
                sample_threshold_mb = config.get('SEQUENTIAL_MAX_FILE_SIZE_MB', 5000) if config else 5000  # Increased from 2000
            else:  # parallel
                sample_threshold_mb = config.get('PARALLEL_MAX_FILE_SIZE_MB', 100) if config else 100
                
            if file_size_mb > sample_threshold_mb:  # Files larger than threshold
                log_universal('INFO', 'Audio', f'Large file detected ({file_size_mb:.1f}MB) - loading sample only')
                try:
                    # Try to load just the first 30 seconds
                    loader = es.MonoLoader(filename=audio_path, sampleRate=sample_rate)
                    audio = loader()
                    if audio is not None and len(audio) > 0:
                        # Take only first 30 seconds
                        max_samples = 30 * sample_rate
                        if len(audio) > max_samples:
                            audio = audio[:max_samples]
                            log_universal('INFO', 'Audio', f'Loaded sample: {len(audio)} samples (30s) from {os.path.basename(audio_path)}')
                        else:
                            log_universal('DEBUG', 'Audio', f'Successfully loaded {os.path.basename(audio_path)}: {len(audio)} samples at {sample_rate}Hz')
                        return audio, sample_rate
                except Exception as sample_e:
                    log_universal('WARNING', 'Audio', f'Sample loading failed for {os.path.basename(audio_path)}: {sample_e}')
                    # Fall through to normal loading
            else:
                # Normal loading for smaller files
                loader = es.MonoLoader(filename=audio_path, sampleRate=sample_rate)
                audio = loader()
                
                if audio is not None and len(audio) > 0:
                    log_universal('DEBUG', 'Audio', f'Successfully loaded {os.path.basename(audio_path)}: {len(audio)} samples at {sample_rate}Hz')
                    return audio, sample_rate
                else:
                    log_universal('WARNING', 'Audio', f'Essentia returned empty audio for {os.path.basename(audio_path)}')
                    return None, None
                    
        except Exception as e:
            log_universal('ERROR', 'Audio', f'Essentia audio loading failed for {os.path.basename(audio_path)}: {e}')
            log_universal('DEBUG', 'Audio', f'Full error details: {type(e).__name__}: {str(e)}')
            
            # Try librosa as fallback (with memory management)
            if LIBROSA_AVAILABLE:
                try:
                    log_universal('DEBUG', 'Audio', f'Trying librosa fallback for {os.path.basename(audio_path)}')
                    import librosa
                    
                    # For large files, use offset and duration to load only a portion
                    if processing_mode == 'sequential':
                        sample_threshold_mb = config.get('SEQUENTIAL_MAX_FILE_SIZE_MB', 5000) if config else 5000  # Increased from 2000
                    else:  # parallel
                        sample_threshold_mb = config.get('PARALLEL_MAX_FILE_SIZE_MB', 100) if config else 100
                    if file_size_mb > sample_threshold_mb:
                        log_universal('INFO', 'Audio', f'Loading 30-second sample with librosa for {os.path.basename(audio_path)}')
                        audio, sr = librosa.load(audio_path, sr=sample_rate, mono=True, duration=30)
                    else:
                        audio, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
                        
                    if audio is not None and len(audio) > 0:
                        log_universal('DEBUG', 'Audio', f'Librosa fallback successful: {len(audio)} samples at {sr}Hz')
                        return audio, sr
                    else:
                        log_universal('WARNING', 'Audio', f'Librosa fallback returned empty audio for {os.path.basename(audio_path)}')
                except Exception as librosa_e:
                    log_universal('ERROR', 'Audio', f'Librosa fallback also failed for {os.path.basename(audio_path)}: {librosa_e}')
            
            return None, None
    except Exception as e:
        log_universal('ERROR', 'Audio', f'Unexpected error in safe_essentia_load for {os.path.basename(audio_path)}: {e}')
        return None, None


class AudioAnalyzer:
    """
    Comprehensive audio analyzer with caching support.
    
    Features:
    - Audio feature extraction (rhythm, spectral, loudness, key, etc.)
    - Metadata extraction from file tags
    - External API enrichment
    - Intelligent caching to avoid re-analysis
    - Fallback mechanisms for missing libraries
    """
    
    def __init__(self, config: Dict[str, Any] = None, processing_mode: str = 'parallel'):
        """
        Initialize audio analyzer.
        
        Args:
            config: Configuration dictionary (uses global config if None)
        """
        if config is None:
            config = config_loader.get_audio_analysis_config()
        
        self.config = config
        self.processing_mode = processing_mode
        self.db_manager = get_db_manager()
        
        # Analysis settings
        self.sample_rate = config.get('SAMPLE_RATE', DEFAULT_SAMPLE_RATE)
        self.hop_size = config.get('HOP_SIZE', DEFAULT_HOP_SIZE)
        self.frame_size = config.get('FRAME_SIZE', DEFAULT_FRAME_SIZE)
        self.timeout_seconds = config.get('TIMEOUT_SECONDS', DEFAULT_TIMEOUT_SECONDS)
        
        # Feature extraction settings
        self.extract_rhythm = config.get('EXTRACT_RHYTHM', True)
        self.extract_spectral = config.get('EXTRACT_SPECTRAL', True)
        self.extract_loudness = config.get('EXTRACT_LOUDNESS', True)
        self.extract_key = config.get('EXTRACT_KEY', True)
        self.extract_mfcc = config.get('EXTRACT_MFCC', True)
        self.extract_musicnn = config.get('EXTRACT_MUSICNN', True)
        self.extract_chroma = config.get('EXTRACT_CHROMA', True)
        
        # Caching settings
        self.cache_enabled = config.get('CACHE_ENABLED', True)
        self.cache_expiry_hours = config.get('CACHE_EXPIRY_HOURS', 168)  # 1 week
        self.force_reanalysis = config.get('FORCE_REANALYSIS', False)
        
        log_universal('INFO', 'Audio', 'AudioAnalyzer initialized')
        log_universal('DEBUG', 'Audio', f'Feature extraction flags: rhythm={self.extract_rhythm}, spectral={self.extract_spectral}, loudness={self.extract_loudness}, key={self.extract_key}, mfcc={self.extract_mfcc}, musicnn={self.extract_musicnn}, chroma={self.extract_chroma}')
        log_universal('DEBUG', 'Audio', f'Processing mode: {self.processing_mode}')
        log_universal('DEBUG', 'Audio', f'TensorFlow available: {TENSORFLOW_AVAILABLE}')
    
    def _get_analysis_cache_key(self, file_path: str, file_hash: str) -> str:
        """Generate cache key for analysis results."""
        # Use file hash and path for cache key
        cache_string = f"analysis:{file_hash}:{file_path}"
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    def _get_metadata_cache_key(self, file_path: str, file_hash: str) -> str:
        """Generate cache key for metadata extraction."""
        # Use file hash and path for cache key
        cache_string = f"metadata:{file_hash}:{file_path}"
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    @log_function_call
    def analyze_audio_file(self, file_path: str, force_reanalysis: bool = None) -> Optional[Dict[str, Any]]:
        """
        Analyze audio file with comprehensive feature extraction and caching.
        
        Args:
            file_path: Path to audio file
            force_reanalysis: Force re-analysis even if cached (uses config if None)
            
        Returns:
            Analysis results dictionary or None on failure
        """
        if force_reanalysis is None:
            force_reanalysis = self.force_reanalysis
        
        log_universal('INFO', 'Audio', f'Starting analysis of: {os.path.basename(file_path)}')
        start_time = time.time()
        
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                log_universal('ERROR', 'Audio', f'File not found: {file_path}')
                return None
            
            # Calculate file hash for change detection
            try:
                file_hash = self._calculate_file_hash(file_path)
                file_size = os.path.getsize(file_path)
                log_universal('DEBUG', 'Audio', f'File size: {file_size} bytes, Hash: {file_hash[:8]}...')
            except Exception as e:
                log_universal('ERROR', 'Audio', f'Failed to calculate file hash: {e}')
                return None
            
            # Check cache first (unless forced)
            if self.cache_enabled and not force_reanalysis:
                cached_result = self._get_cached_analysis(file_path, file_hash)
                if cached_result:
                    analysis_time = time.time() - start_time
                    log_universal('INFO', 'Audio', f'Using cached analysis for {os.path.basename(file_path)} ({analysis_time:.2f}s)')
                    return cached_result
            
            # Extract metadata first
            log_universal('DEBUG', 'Audio', f'Extracting metadata for: {os.path.basename(file_path)}')
            metadata = self._extract_metadata(file_path)
            
            # Enrich metadata with external APIs FIRST (before audio analysis)
            if metadata:
                # Add filename to metadata for fallback extraction
                metadata['filename'] = os.path.basename(file_path)
                log_universal('DEBUG', 'Audio', f'Enriching metadata for: {os.path.basename(file_path)}')
                enriched_metadata = self._enrich_metadata_with_external_apis(metadata)
                metadata = enriched_metadata
            
            # Check if we should skip audio loading for large files
            if self._should_skip_audio_loading(file_path):
                log_universal('WARNING', 'Audio', f'Skipping audio analysis for large file: {os.path.basename(file_path)}')
                # Return basic analysis with enriched metadata
                return self._create_basic_analysis_for_large_file(file_path, file_size, file_hash, metadata)
            
            # Load audio data
            log_universal('DEBUG', 'Audio', f'Loading audio data for: {os.path.basename(file_path)}')
            audio, sample_rate = safe_essentia_load(file_path, self.sample_rate, self.config, self.processing_mode)
            if audio is None:
                log_universal('ERROR', 'Audio', f'Failed to load audio: {os.path.basename(file_path)}')
                return None
            
            log_universal('DEBUG', 'Audio', f'Successfully loaded audio: {len(audio)} samples at {sample_rate}Hz')
            
            # Check if this is a long audio track
            is_long_audio = self._is_long_audio_track(file_path)
            
            # Add long audio flag to metadata for feature extraction
            if metadata:
                metadata['is_long_audio'] = is_long_audio
            else:
                metadata = {'is_long_audio': is_long_audio}
            
            # Perform audio analysis with enriched metadata
            log_universal('DEBUG', 'Audio', f'Extracting audio features for: {os.path.basename(file_path)}')
            analysis_result = self._extract_audio_features(audio, sample_rate, metadata)
            if analysis_result is None:
                log_universal('ERROR', 'Audio', f'Feature extraction failed: {os.path.basename(file_path)}')
                return None
            
            # Set the enriched metadata in results
            analysis_result['metadata'] = metadata or {}
            
            # Determine audio type and add file information
            audio_type = self._get_audio_type(file_path, audio)
            is_long_audio = self._is_long_audio_track(file_path)
            
            # Determine long audio category if it's a long audio track
            long_audio_category = None
            if is_long_audio:
                log_universal('INFO', 'Audio', f'Long audio track detected: {os.path.basename(file_path)}')
                # For now, determine category without audio features (will be updated after analysis)
                long_audio_category = self._determine_long_audio_category(file_path, metadata, None)
                log_universal('INFO', 'Audio', f'Long audio category determined: {long_audio_category}')
            
            analysis_result.update({
                'file_path': file_path,
                'filename': os.path.basename(file_path),
                'file_size_bytes': file_size,
                'file_hash': file_hash,
                'analysis_date': datetime.now().isoformat(),
                'analysis_version': '1.0.0',
                'audio_type': audio_type,
                'is_long_audio': is_long_audio,
                'is_extremely_large': self._is_extremely_large_for_processing(audio),
                'long_audio_category': long_audio_category
            })
            
            # Update long audio category with actual audio features if available
            if is_long_audio and analysis_result:
                # Re-determine category with actual audio features
                updated_category = self._determine_long_audio_category(file_path, metadata, analysis_result)
                if updated_category != long_audio_category:
                    log_universal('INFO', 'Audio', f'Updated long audio category: {long_audio_category} -> {updated_category}')
                    long_audio_category = updated_category
                    analysis_result['long_audio_category'] = long_audio_category
            
            # Update metadata with long audio category
            if metadata and long_audio_category:
                metadata['long_audio_category'] = long_audio_category
                log_universal('DEBUG', 'Audio', f'Set long_audio_category in metadata: {long_audio_category}')
            
            # Also update the analysis result's metadata
            if analysis_result and 'metadata' in analysis_result and long_audio_category:
                analysis_result['metadata']['long_audio_category'] = long_audio_category
                log_universal('DEBUG', 'Audio', f'Set long_audio_category in analysis_result metadata: {long_audio_category}')
            
            # Cache results
            if self.cache_enabled:
                self._cache_analysis_result(file_path, file_hash, analysis_result)
            
            analysis_time = time.time() - start_time
            log_universal('INFO', 'Audio', f'Analysis completed for {os.path.basename(file_path)} ({analysis_time:.2f}s)')
            
            return analysis_result
            
        except Exception as e:
            log_universal('ERROR', 'Audio', f'Unexpected error during analysis of {os.path.basename(file_path)}: {e}')
            import traceback
            log_universal('ERROR', 'Audio', f'Traceback: {traceback.format_exc()}')
            return None
    
    def _get_cached_analysis(self, file_path: str, file_hash: str) -> Optional[Dict[str, Any]]:
        """Get cached analysis results if available and valid."""
        cache_key = self._get_analysis_cache_key(file_path, file_hash)
        cached_result = self.db_manager.get_cache(cache_key)
        
        if cached_result:
            # Verify the cached result is still valid
            if (cached_result.get('file_hash') == file_hash and 
                cached_result.get('file_path') == file_path):
                log_universal('DEBUG', 'Audio', f'Found valid cached analysis for: {os.path.basename(file_path)}')
                return cached_result
            else:
                log_universal('DEBUG', 'Audio', f'Cached analysis invalid for: {os.path.basename(file_path)}')
        
        return None
    
    def _cache_analysis_result(self, file_path: str, file_hash: str, analysis_result: Dict[str, Any]):
        """Cache analysis results."""
        cache_key = self._get_analysis_cache_key(file_path, file_hash)
        success = self.db_manager.save_cache(cache_key, analysis_result, expires_hours=self.cache_expiry_hours)
        
        if success:
            log_universal('DEBUG', 'Audio', f'Cached analysis result for: {os.path.basename(file_path)}')
        else:
            log_universal('WARNING', 'Audio', f'Failed to cache analysis result for: {os.path.basename(file_path)}')
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of file for change detection."""
        hash_sha256 = hashlib.sha256()
        
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        
        return hash_sha256.hexdigest()
    
    def _extract_metadata(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Extract metadata from audio file tags using enhanced tag mapper.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Metadata dictionary or None on failure
        """
        if not MUTAGEN_AVAILABLE:
            log_universal('WARNING', 'Audio', 'Mutagen not available - skipping metadata extraction')
            return None
        
        # Check cache first
        file_hash = self._calculate_file_hash(file_path)
        cache_key = self._get_metadata_cache_key(file_path, file_hash)
        cached_metadata = self.db_manager.get_cache(cache_key)
        
        if cached_metadata:
            log_universal('DEBUG', 'Audio', f'Using cached metadata for: {os.path.basename(file_path)}')
            return cached_metadata
        
        try:
            log_universal('DEBUG', 'Audio', f'Extracting metadata from: {os.path.basename(file_path)}')
            
            # Check file size before attempting metadata extraction
            try:
                file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
                metadata_warning_threshold = self.config.get('LARGE_FILE_WARNING_THRESHOLD_MB', 500)
                if file_size_mb > metadata_warning_threshold:  # Files larger than threshold
                    log_universal('WARNING', 'Audio', f'Large file detected ({file_size_mb:.1f}MB) - metadata extraction may fail')
            except Exception:
                pass
            
            # Force garbage collection before metadata extraction
            import gc
            gc.collect()
            
            # Check available memory before proceeding
            try:
                import psutil
                available_memory_mb = psutil.virtual_memory().available / (1024 * 1024)
                min_memory_mb = self.config.get('MIN_MEMORY_FOR_FULL_ANALYSIS_GB', 2.0) * 1024  # Convert GB to MB
                if available_memory_mb < min_memory_mb:
                    log_universal('WARNING', 'Audio', f'Low memory available ({available_memory_mb:.1f}MB) - skipping metadata extraction')
                    return None
            except Exception:
                pass  # Continue if memory check fails
            
            audio_file = File(file_path)
            if audio_file is None:
                log_universal('WARNING', 'Audio', f'Could not open file for metadata: {os.path.basename(file_path)}')
                return None
            
            metadata = {}
            
            # Extract tags using enhanced tag mapper with memory protection
            if hasattr(audio_file, 'tags') and audio_file.tags:
                try:
                    from .tag_mapping import get_tag_mapper
                    tag_mapper = get_tag_mapper()
                    
                    # Convert mutagen tags to dictionary format with memory protection
                    tags_dict = {}
                    tag_count = 0
                    max_tags = 100  # Limit number of tags to prevent memory issues
                    
                    for key, value in audio_file.tags.items():
                        if tag_count >= max_tags:
                            log_universal('WARNING', 'Audio', f'Too many tags ({tag_count}) - truncating to prevent memory issues')
                            break
                        
                        # Limit tag value size to prevent memory issues
                        if isinstance(value, str) and len(value) > 1000:
                            value = value[:1000] + "..."
                        elif isinstance(value, list):
                            # Limit list size and individual item size
                            value = [str(item)[:500] for item in value[:10]]
                        
                        tags_dict[key] = value
                        tag_count += 1
                    
                    # Map tags using the enhanced mapper
                    mapped_metadata = tag_mapper.map_tags(tags_dict)
                    metadata.update(mapped_metadata)
                    
                    # Add filename for enrichment
                    metadata['filename'] = os.path.basename(file_path)
                    
                    log_universal('DEBUG', 'Audio', f'Extracted {len(mapped_metadata)} metadata fields')
                    
                except MemoryError as me:
                    log_universal('ERROR', 'Audio', f'Memory error during tag mapping for {os.path.basename(file_path)}: {me}')
                    # Fall back to basic metadata
                    metadata['filename'] = os.path.basename(file_path)
                except Exception as e:
                    log_universal('ERROR', 'Audio', f'Tag mapping failed for {os.path.basename(file_path)}: {e}')
                    # Fall back to basic metadata
                    metadata['filename'] = os.path.basename(file_path)
            else:
                log_universal('DEBUG', 'Audio', 'No tags found in file')
                metadata['filename'] = os.path.basename(file_path)
            
            # Extract audio properties with memory protection
            try:
                if hasattr(audio_file, 'info'):
                    info = audio_file.info
                    metadata['duration'] = getattr(info, 'length', None)
                    metadata['bitrate'] = getattr(info, 'bitrate', None)
                    metadata['sample_rate'] = getattr(info, 'sample_rate', None)
                    metadata['channels'] = getattr(info, 'channels', None)
            except MemoryError as me:
                log_universal('ERROR', 'Audio', f'Memory error during audio property extraction for {os.path.basename(file_path)}: {me}')
            except Exception as e:
                log_universal('WARNING', 'Audio', f'Failed to extract audio properties for {os.path.basename(file_path)}: {e}')
            
            # Extract BPM from metadata if available
            try:
                bpm_from_metadata = self._extract_bpm_from_metadata(metadata)
                if bpm_from_metadata:
                    metadata['bpm_from_metadata'] = bpm_from_metadata
            except Exception as e:
                log_universal('WARNING', 'Audio', f'Failed to extract BPM from metadata for {os.path.basename(file_path)}: {e}')
            
            # Cache metadata
            try:
                self.db_manager.save_cache(cache_key, metadata, expires_hours=self.cache_expiry_hours)
            except Exception as e:
                log_universal('WARNING', 'Audio', f'Failed to cache metadata for {os.path.basename(file_path)}: {e}')
            
            log_universal('DEBUG', 'Audio', f'Extracted metadata: {len(metadata)} fields from {os.path.basename(file_path)}')
            return metadata
            
        except MemoryError as me:
            log_universal('ERROR', 'Audio', f'Memory error during metadata extraction for {os.path.basename(file_path)}: {me}')
            return None
        except Exception as e:
            log_universal('ERROR', 'Audio', f'Metadata extraction failed for {os.path.basename(file_path)}: {e}')
            return None
    
    def _get_tag_value(self, tags, possible_keys):
        """Extract tag value from multiple possible keys."""
        for key in possible_keys:
            try:
                if key in tags:
                    value = tags[key]
                    
                    # Handle different value types
                    if isinstance(value, list):
                        if len(value) > 0:
                            return str(value[0])
                    elif isinstance(value, str):
                        if value.strip():  # Check if string is not empty after stripping whitespace
                            return value.strip()
                    else:
                        # Convert other types to string
                        result = str(value)
                        if result.strip():
                            return result
            except Exception:
                continue
        return None
    
    def _extract_bpm_from_metadata(self, metadata: Dict[str, Any]) -> Optional[float]:
        """
        Extract BPM from metadata fields.
        
        Args:
            metadata: Metadata dictionary
            
        Returns:
            BPM value or None if not found
        """
        # Check various BPM-related fields
        bpm_fields = ['bpm', 'tempo', 'TBPM', 'BPM', 'Tempo']
        
        for field in bpm_fields:
            if field in metadata and metadata[field]:
                try:
                    bpm_value = metadata[field]
                    
                    # Handle different formats
                    if isinstance(bpm_value, str):
                        # Remove non-numeric characters and convert
                        import re
                        numeric_match = re.search(r'(\d+(?:\.\d+)?)', str(bpm_value))
                        if numeric_match:
                            bpm = float(numeric_match.group(1))
                        else:
                            continue
                    else:
                        bpm = float(bpm_value)
                    
                    # Validate BPM range
                    if 60 <= bpm <= 200:
                        log_universal('DEBUG', 'Audio', f'Found BPM in metadata: {bpm} from field {field}')
                        return bpm
                    else:
                        log_universal('DEBUG', 'Audio', f'BPM {bpm} from field {field} outside valid range (60-200)')
                        
                except (ValueError, TypeError):
                    continue
        
        # Check comments field for BPM
        if 'comment' in metadata and metadata['comment']:
            try:
                import re
                comment = str(metadata['comment'])
                # Look for BPM patterns in comments
                bpm_patterns = [
                    r'BPM[:\s]*(\d+(?:\.\d+)?)',
                    r'Tempo[:\s]*(\d+(?:\.\d+)?)',
                    r'(\d+(?:\.\d+)?)\s*BPM',
                    r'(\d+(?:\.\d+)?)\s*Tempo'
                ]
                
                for pattern in bpm_patterns:
                    match = re.search(pattern, comment, re.IGNORECASE)
                    if match:
                        bpm = float(match.group(1))
                        if 60 <= bpm <= 200:
                            log_universal('DEBUG', 'Audio', f'Found BPM in comment: {bpm}')
                            return bpm
            except Exception:
                pass
        
        return None
    
    def _enrich_metadata_with_external_apis(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enrich metadata using enhanced external APIs service.
        
        Args:
            metadata: Original metadata dictionary
            
        Returns:
            Enriched metadata dictionary
        """
        try:
            from .external_apis import get_enhanced_metadata_enrichment_service
            
            enrichment_service = get_enhanced_metadata_enrichment_service()
            if enrichment_service.is_available():
                log_universal('DEBUG', 'Audio', 'Enriching metadata with enhanced external APIs')
                return enrichment_service.enrich_metadata(metadata)
            else:
                log_universal('DEBUG', 'Audio', 'No external APIs available for enrichment')
                return metadata
                
        except Exception as e:
            log_universal('WARNING', 'Audio', f'External API enrichment failed: {e}')
            return metadata
    
    def _extract_audio_features(self, audio: np.ndarray, sample_rate: int, metadata: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """
        Extract comprehensive audio features.
        Uses original skip logic for extremely large files to prevent RAM saturation.
        Also uses simplified analysis for long audio tracks based on configuration.
        
        Args:
            audio: Audio data as numpy array
            sample_rate: Sample rate in Hz
            metadata: Optional metadata dictionary
            
        Returns:
            Features dictionary or None on failure
        """
        features = {}
        
        # Check if file is extremely large (original logic)
        is_extremely_large_for_processing = self._is_extremely_large_for_processing(audio)
        
        # Check if this is a long audio track that should use simplified analysis
        is_long_audio = metadata.get('is_long_audio', False) if metadata else False
        long_audio_simplified = self.config.get('LONG_AUDIO_SIMPLIFIED_FEATURES', True)
        long_audio_skip_detailed = self.config.get('LONG_AUDIO_SKIP_DETAILED_ANALYSIS', True)
        
        # For sequential processing of long audio tracks, use optimized analysis for categorization
        if self.processing_mode == 'sequential' and is_long_audio:
            log_universal('INFO', 'Audio', 'Sequential processing: using optimized analysis for long audio categorization')
            return self._extract_optimized_features_for_categorization(audio, sample_rate, metadata)
        # Use simplified analysis for long audio tracks if configured (only for parallel processing)
        elif is_long_audio and long_audio_simplified and long_audio_skip_detailed:
            log_universal('INFO', 'Audio', 'Using simplified analysis for long audio track')
            return self._extract_simplified_features(audio, sample_rate, metadata)
        
        try:
            # Extract rhythm features (always extract for sequential processing)
            if self.extract_rhythm:
                rhythm_features = self._extract_rhythm_features(audio, sample_rate)
                features.update(rhythm_features)
            
            # Extract spectral features (always extract for sequential processing)
            if self.extract_spectral:
                spectral_features = self._extract_spectral_features(audio, sample_rate)
                features.update(spectral_features)
            
            # Extract loudness features (always extract for sequential processing)
            if self.extract_loudness:
                loudness_features = self._extract_loudness_features(audio, sample_rate)
                features.update(loudness_features)
            
            # Extract key and mode (always extract - not memory intensive)
            if self.extract_key:
                key_features = self._extract_key_features(audio, sample_rate)
                features.update(key_features)
            
            # Extract MFCC features (always extract for sequential processing)
            if self.extract_mfcc:
                mfcc_features = self._extract_mfcc_features(audio, sample_rate)
                features.update(mfcc_features)
            
            # Extract MusiCNN features (always extract for sequential processing)
            if self.extract_musicnn and TENSORFLOW_AVAILABLE:
                log_universal('INFO', 'Audio', 'Starting MusiCNN feature extraction')
                musicnn_features = self._extract_musicnn_features(audio, sample_rate)
                log_universal('INFO', 'Audio', f'MusiCNN extraction completed: {len(musicnn_features)} features')
                features.update(musicnn_features)
            else:
                log_universal('INFO', 'Audio', f'MusiCNN extraction skipped: extract_musicnn={self.extract_musicnn}, TENSORFLOW_AVAILABLE={TENSORFLOW_AVAILABLE}')
            
            # Extract chroma features (always extract for sequential processing)
            if self.extract_chroma:
                chroma_features = self._extract_chroma_features(audio, sample_rate)
                features.update(chroma_features)
            
            # Add BPM from metadata if available
            if metadata and 'bpm_from_metadata' in metadata:
                features['external_bpm'] = metadata['bpm_from_metadata']
            
            # Extract Spotify-style features for playlist generation
            # These are derived from the extracted features above
            try:
                spotify_features = self._extract_spotify_style_features(
                    audio, sample_rate,
                    rhythm_features=features,
                    spectral_features=features,
                    loudness_features=features,
                    key_features=features,
                    mfcc_features=features,
                    musicnn_features=features,
                    chroma_features=features
                )
                features.update(spotify_features)
            except Exception as e:
                log_universal('WARNING', 'Audio', f'Failed to extract Spotify-style features: {e}')
                # Add default values
                features.update({
                    'danceability': 0.5,
                    'energy': 0.5,
                    'mode': 0.0,
                    'acousticness': 0.5,
                    'instrumentalness': 0.5,
                    'speechiness': 0.5,
                    'valence': 0.5,
                    'liveness': 0.5,
                    'popularity': 0.5
                })
            
            return features
            
        except Exception as e:
            log_universal('ERROR', 'Audio', f'Feature extraction failed: {e}')
            return None
    
    def _extract_rhythm_features(self, audio: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Extract rhythm-related features using lightweight approach for large files."""
        features = {}
        
        log_universal('DEBUG', 'Audio', f'Starting rhythm extraction: audio_length={len(audio)}, sample_rate={sample_rate}')
        
        try:
            if ESSENTIA_AVAILABLE:
                # For large files, use a much smaller sample to avoid RAM issues
                if len(audio) > 5000000:  # More than 5M samples (~1.9 minutes)
                    log_universal('INFO', 'Audio', 'Using lightweight rhythm analysis (large file)')
                    # Use only first 15 seconds for rhythm analysis - ultra memory efficient
                    sample_size = min(15 * sample_rate, len(audio))
                    audio_sample = audio[:sample_size]
                else:
                    audio_sample = audio
                
                # Try a simpler approach first - use TempoTapDegara which is more reliable
                try:
                    tempo_tap = es.TempoTapDegara()
                    ticks = tempo_tap(audio_sample)
                    if len(ticks) > 0:
                        # Calculate BPM from ticks
                        intervals = np.diff(ticks)
                        if len(intervals) > 0:
                            mean_interval = np.mean(intervals)
                            bpm = 60.0 / mean_interval if mean_interval > 0 else 120.0
                            features['bpm'] = float(bpm)
                            features['rhythm_confidence'] = 0.7
                            features['bpm_estimates'] = [bpm]
                            features['bpm_intervals'] = intervals.tolist()
                        else:
                            features['bpm'] = -1  # Out of range for failed extraction
                            features['rhythm_confidence'] = 0.0
                            features['bpm_estimates'] = []
                            features['bpm_intervals'] = []
                    else:
                        features['bpm'] = -1  # Out of range for failed extraction
                        features['rhythm_confidence'] = 0.0
                        features['bpm_estimates'] = []
                        features['bpm_intervals'] = []
                        
                except Exception as e:
                    log_universal('DEBUG', 'Audio', f'TempoTapDegara failed, trying RhythmExtractor2013: {e}')
                    
                    # Fallback to RhythmExtractor2013 with better error handling
                    rhythm_extractor = es.RhythmExtractor2013()
                    rhythm_result = rhythm_extractor(audio_sample)
                    
                    # Handle the result more carefully
                    try:
                        if hasattr(rhythm_result, '__len__') and len(rhythm_result) > 0:
                            # It's an array-like object
                            if hasattr(rhythm_result[0], '__len__'):
                                # First element is also an array, take its first element
                                features['bpm'] = float(rhythm_result[0][0]) if len(rhythm_result[0]) > 0 else -1
                            else:
                                features['bpm'] = float(rhythm_result[0])
                            
                            if len(rhythm_result) > 1:
                                if hasattr(rhythm_result[1], '__len__'):
                                    features['rhythm_confidence'] = float(rhythm_result[1][0]) if len(rhythm_result[1]) > 0 else 0.0
                                else:
                                    features['rhythm_confidence'] = float(rhythm_result[1])
                            else:
                                features['rhythm_confidence'] = 0.0
                        else:
                            # Single value
                            features['bpm'] = float(rhythm_result)
                            features['rhythm_confidence'] = 0.0
                            
                        features['bpm_estimates'] = [features['bpm']] if features['bpm'] > 0 else []
                        features['bpm_intervals'] = []
                        
                    except Exception as e2:
                        log_universal('DEBUG', 'Audio', f'RhythmExtractor2013 also failed: {e2}')
                        features['bpm'] = -1  # Out of range for failed extraction
                        features['rhythm_confidence'] = 0.0
                        features['bpm_estimates'] = []
                        features['bpm_intervals'] = []
                
            elif LIBROSA_AVAILABLE:
                # Use librosa as fallback
                tempo, beats = librosa.beat.beat_track(y=audio, sr=sample_rate)
                features['bpm'] = float(tempo)
                features['rhythm_confidence'] = 0.5  # Default confidence for librosa
                features['bpm_estimates'] = [tempo]
                features['bpm_intervals'] = []
            else:
                # No rhythm extraction available
                features['bpm'] = -1  # Out of range for failed extraction
                features['rhythm_confidence'] = 0.0
                features['bpm_estimates'] = []
                features['bpm_intervals'] = []
            
            log_universal('DEBUG', 'Audio', f'Extracted rhythm features: BPM={features.get("bpm")}')
            
        except Exception as e:
            log_universal('WARNING', 'Audio', f'Rhythm feature extraction failed: {e}')
            features['bpm'] = -1  # Out of range for failed extraction
            features['rhythm_confidence'] = 0.0
            features['bpm_estimates'] = []
            features['bpm_intervals'] = []
        
        return features
    
    def _extract_spectral_features(self, audio: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Extract spectral features using the old working approach."""
        features = {}
        
        try:
            if ESSENTIA_AVAILABLE:
                # Use SpectralCentroidTime like the old working version
                log_universal('DEBUG', 'Audio', "Initializing Essentia SpectralCentroidTime algorithm")
                centroid_algo = es.SpectralCentroidTime()
                log_universal('DEBUG', 'Audio', "Running spectral centroid analysis on audio")
                centroid_values = centroid_algo(audio)
                log_universal('DEBUG', 'Audio', "Spectral analysis completed successfully")
                
                centroid_mean = float(np.nanmean(centroid_values)) if isinstance(
                    centroid_values, (list, np.ndarray)) else float(centroid_values)
                log_universal('DEBUG', 'Audio', f"Calculated mean spectral centroid: {centroid_mean:.1f}")
                log_universal('INFO', 'Audio', f"Spectral features completed: centroid = {centroid_mean:.1f}Hz")
                
                features['spectral_centroid'] = centroid_mean
                
                # Add spectral flatness for better categorization
                try:
                    # Use FlatnessSFX which is available in Essentia
                    flatness_algo = es.FlatnessSFX()
                    flatness_values = flatness_algo(audio)
                    flatness_mean = float(np.nanmean(flatness_values)) if isinstance(
                        flatness_values, (list, np.ndarray)) else float(flatness_values)
                    features['spectral_flatness'] = flatness_mean
                except Exception as e:
                    log_universal('WARNING', 'Audio', f'Spectral flatness extraction failed: {e}')
                    features['spectral_flatness'] = 0.5  # Default value
                
                features['spectral_rolloff'] = 0.0  # Not available in old version
                features['spectral_bandwidth'] = 0.0  # Not available in old version
                features['spectral_contrast_mean'] = 0.0  # Not available in old version
                features['spectral_contrast_std'] = 0.0  # Not available in old version
                
            elif LIBROSA_AVAILABLE:
                # Use librosa as fallback
                spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)
                features['spectral_centroid'] = float(np.mean(spectral_centroids))
                features['spectral_rolloff'] = 0.0
                features['spectral_flatness'] = 0.0
                features['spectral_bandwidth'] = 0.0
                features['spectral_contrast_mean'] = 0.0
                features['spectral_contrast_std'] = 0.0
            
            log_universal('DEBUG', 'Audio', f'Extracted spectral features')
            
        except Exception as e:
            log_universal('WARNING', 'Audio', f'Spectral feature extraction failed: {e}')
            # Return default values like the old working version
            features.update({
                'spectral_centroid': 0.0,
                'spectral_rolloff': 0.0,
                'spectral_flatness': 0.0,
                'spectral_bandwidth': 0.0,
                'spectral_contrast_mean': 0.0,
                'spectral_contrast_std': 0.0
            })
        
        return features
    
    def _extract_loudness_features(self, audio: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Extract loudness features using the old working approach."""
        features = {}
        
        try:
            if ESSENTIA_AVAILABLE:
                # Use RMS like the old working version
                log_universal('DEBUG', 'Audio', "Initializing Essentia RMS algorithm")
                rms_algo = es.RMS()
                log_universal('DEBUG', 'Audio', "Running RMS analysis on audio")
                rms_values = rms_algo(audio)
                log_universal('DEBUG', 'Audio', "Loudness analysis completed successfully")
                
                rms_mean = float(np.nanmean(rms_values)) if isinstance(
                    rms_values, (list, np.ndarray)) else float(rms_values)
                log_universal('DEBUG', 'Audio', f"Calculated mean RMS: {rms_mean:.3f}")
                log_universal('INFO', 'Audio', f"Loudness extraction completed: RMS = {rms_mean:.3f}")
                
                features['loudness'] = rms_mean
                
                # Add dynamic complexity for better categorization
                try:
                    # Use RMS-based dynamic complexity calculation
                    rms_values = rms_algo(audio)
                    if isinstance(rms_values, (list, np.ndarray)):
                        rms_std = float(np.nanstd(rms_values))
                        rms_mean = float(np.nanmean(rms_values))
                        # Dynamic complexity as coefficient of variation
                        features['dynamic_complexity'] = rms_std / rms_mean if rms_mean > 0 else 0.5
                    else:
                        features['dynamic_complexity'] = 0.5
                except Exception as e:
                    log_universal('WARNING', 'Audio', f'Dynamic complexity extraction failed: {e}')
                    features['dynamic_complexity'] = 0.5  # Default value
                
                features['loudness_range'] = 0.0  # Not available in old version
                features['dynamic_range'] = 0.0  # Not available in old version
                
            elif LIBROSA_AVAILABLE:
                # Use librosa as fallback
                rms = librosa.feature.rms(y=audio)
                features['loudness'] = float(np.mean(rms))
                features['loudness_range'] = 0.0
                features['dynamic_complexity'] = 0.0
                features['dynamic_range'] = 0.0
            
            log_universal('DEBUG', 'Audio', f'Extracted loudness features')
            
        except Exception as e:
            log_universal('WARNING', 'Audio', f'Loudness feature extraction failed: {e}')
            # Return default values like the old working version
            features.update({
                'loudness': 0.0,
                'loudness_range': 0.0,
                'dynamic_complexity': 0.0,
                'dynamic_range': 0.0
            })
        
        return features
    
    def _extract_key_features(self, audio: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Extract key and mode features using the old working approach."""
        features = {}
        
        try:
            if ESSENTIA_AVAILABLE:
                # Validate audio parameters before key extraction
                if len(audio) < 1024:
                    log_universal('WARNING', 'Audio', f'Audio too short for key extraction: {len(audio)} samples')
                    features.update({
                        'key': 'C',
                        'scale': 'major',
                        'key_strength': 0.0
                    })
                    return features
                
                # Ensure audio is mono and has proper sample rate
                if len(audio.shape) > 1:
                    audio = np.mean(audio, axis=1)
                
                # Resample to 16kHz if needed (KeyExtractor works better with 16kHz)
                if sample_rate != 16000:
                    try:
                        import librosa
                        audio_16k = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
                        log_universal('DEBUG', 'Audio', f'Resampled audio from {sample_rate}Hz to 16000Hz for key extraction')
                    except Exception as e:
                        log_universal('WARNING', 'Audio', f'Failed to resample audio for key extraction: {e}')
                        audio_16k = audio
                else:
                    audio_16k = audio
                
                # Use KeyExtractor like the old working version
                log_universal('DEBUG', 'Audio', "Using Essentia KeyExtractor for key detection")
                key_algo = es.KeyExtractor()
                log_universal('DEBUG', 'Audio', "Running key analysis on audio")
                key, scale, strength = key_algo(audio_16k)
                log_universal('DEBUG', 'Audio', f"Extracted key: {key} {scale}, strength: {strength}")
                
                features['key'] = str(key)
                features['scale'] = str(scale)
                features['key_strength'] = float(strength)
                log_universal('INFO', 'Audio', f"Key extraction completed: {key} {scale} (strength: {strength:.3f})")
                
            elif LIBROSA_AVAILABLE:
                # Use librosa as fallback
                chroma = librosa.feature.chroma_cqt(y=audio, sr=sample_rate)
                key_detector = librosa.feature.key_mode(chroma)
                features['key'] = key_detector[0]
                features['scale'] = key_detector[1]
                features['key_strength'] = 0.5
                features['key_confidence'] = 0.5
            
            log_universal('DEBUG', 'Audio', f'Extracted key features: {features.get("key")} {features.get("scale")}')
            
        except Exception as e:
            log_universal('WARNING', 'Audio', f'Key feature extraction failed: {e}')
            # Return default values like the old working version
            features.update({
                'key': 'C',
                'scale': 'major',
                'key_strength': 0.0
            })
        
        return features
    
    def _extract_mfcc_features(self, audio: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Extract MFCC features."""
        features = {}
        
        try:
            if ESSENTIA_AVAILABLE:
                # Validate audio parameters before MFCC extraction
                if len(audio) < 512:
                    log_universal('WARNING', 'Audio', f'Audio too short for MFCC extraction: {len(audio)} samples')
                    features.update({
                        'mfcc_coefficients': [],
                        'mfcc_bands': [],
                        'mfcc_std': []
                    })
                    return features
                
                # Ensure audio is mono
                if len(audio.shape) > 1:
                    audio = np.mean(audio, axis=1)
                
                # Limit audio length to prevent spectrum size issues
                max_audio_length = 60 * sample_rate  # 60 seconds max
                if len(audio) > max_audio_length:
                    log_universal('WARNING', 'Audio', f'Audio too long for MFCC extraction, truncating to 60s')
                    audio = audio[:max_audio_length]
                
                # MFCC
                mfcc = es.MFCC()
                mfcc_result = mfcc(audio)
                mfcc_coefficients = mfcc_result[0]
                mfcc_bands = mfcc_result[1]
                
                features['mfcc_coefficients'] = mfcc_coefficients.tolist()
                features['mfcc_bands'] = mfcc_bands.tolist()
                features['mfcc_std'] = np.std(mfcc_coefficients, axis=0).tolist()
                
            elif LIBROSA_AVAILABLE:
                # Use librosa as fallback
                mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
                features['mfcc_coefficients'] = mfcc.tolist()
                features['mfcc_bands'] = []
                features['mfcc_std'] = np.std(mfcc, axis=1).tolist()
            
            log_universal('DEBUG', 'Audio', f'Extracted MFCC features')
            
        except Exception as e:
            log_universal('WARNING', 'Audio', f'MFCC feature extraction failed: {e}')
            features.update({
                'mfcc_coefficients': [],
                'mfcc_bands': [],
                'mfcc_std': []
            })
        
        return features
    
    def _extract_musicnn_features(self, audio: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Extract MusiCNN features using Essentia's TensorflowPredictMusiCNN."""
        features = {}
        
        log_universal('DEBUG', 'Audio', f'MusiCNN extraction started: audio shape={audio.shape}, sample_rate={sample_rate}')
        
        try:
            if not TENSORFLOW_AVAILABLE:
                log_universal('WARNING', 'Audio', 'TensorFlow not available - skipping MusiCNN features')
                features.update({'embedding': [], 'tags': {}})
                return features
            
            # Get model paths from configuration
            model_path = self.config.get('MUSICNN_MODEL_PATH', '/app/models/msd-musicnn-1.pb')
            json_path = self.config.get('MUSICNN_JSON_PATH', '/app/models/msd-musicnn-1.json')
            
            log_universal('DEBUG', 'Audio', f'MusiCNN model path: {model_path}')
            log_universal('DEBUG', 'Audio', f'MusiCNN JSON path: {json_path}')
            
            if not os.path.exists(model_path):
                log_universal('WARNING', 'Audio', f'MusiCNN model not found: {model_path}')
                features.update({'embedding': [], 'tags': {}})
                return features
            
            if not os.path.exists(json_path):
                log_universal('WARNING', 'Audio', f'MusiCNN JSON config not found: {json_path}')
                features.update({'embedding': [], 'tags': {}})
                return features
            
            log_universal('DEBUG', 'Audio', 'MusiCNN model files found')
            
            # Load tag names from JSON (only once)
            if not hasattr(self, '_musicnn_tag_names'):
                try:
                    import json
                    with open(json_path, 'r') as f:
                        metadata = json.load(f)
                    self._musicnn_tag_names = metadata.get('classes', [])
                    self._musicnn_metadata = metadata
                    log_universal('DEBUG', 'Audio', f'Loaded {len(self._musicnn_tag_names)} tag names from MusiCNN config')
                except Exception as e:
                    log_universal('WARNING', 'Audio', f'Failed to load MusiCNN JSON config: {e}')
                    features.update({'embedding': [], 'tags': {}})
                    return features
            
            # Load MusicNN models (only once)
            if not hasattr(self, '_musicnn_activations_model'):
                try:
                    import essentia.standard as es
                    
                    log_universal('DEBUG', 'Audio', 'Loading MusiCNN activations model...')
                    
                    # Initialize MusiCNN for activations (auto-tagging)
                    self._musicnn_activations_model = es.TensorflowPredictMusiCNN(graphFilename=model_path)
                    log_universal('INFO', 'Audio', 'Loaded MusiCNN activations model')
                    
                    # Get embeddings using different output layer
                    output_layer = 'model/dense_1/BiasAdd'
                    if 'schema' in self._musicnn_metadata and 'outputs' in self._musicnn_metadata['schema']:
                        for output in self._musicnn_metadata['schema']['outputs']:
                            if 'description' in output and output['description'] == 'embeddings':
                                output_layer = output['name']
                                break
                    
                    log_universal('DEBUG', 'Audio', f'Loading MusiCNN embeddings model with output layer: {output_layer}')
                    
                    self._musicnn_embeddings_model = es.TensorflowPredictMusiCNN(
                        graphFilename=model_path,
                        output=output_layer
                    )
                    log_universal('INFO', 'Audio', 'Loaded MusiCNN embeddings model')
                    
                except Exception as e:
                    log_universal('WARNING', 'Audio', f'Failed to load MusiCNN models: {e}')
                    features.update({'embedding': [], 'tags': {}})
                    return features
            
            # Use loaded models for inference
            try:
                log_universal('DEBUG', 'Audio', 'Starting MusiCNN inference...')
                
                # Resample audio to 16kHz for MusiCNN (like old setup)
                if sample_rate != 16000:
                    import librosa
                    audio_16k = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
                else:
                    audio_16k = audio
                
                log_universal('DEBUG', 'Audio', f'Resampled audio to 16kHz: {len(audio_16k)} samples')
                
                # Run activations inference using loaded model
                log_universal('DEBUG', 'Audio', 'Running MusiCNN activations inference...')
                activations = self._musicnn_activations_model(audio_16k)  # shape: [time, tags]
                
                # Handle different return types
                if isinstance(activations, list):
                    activations = np.array(activations)
                
                log_universal('DEBUG', 'Audio', f'MusiCNN activations shape: {activations.shape}')
                
                # Calculate tag probabilities (mean across time)
                tag_probs = activations.mean(axis=0)
                tags = dict(zip(self._musicnn_tag_names, [float(prob) for prob in tag_probs]))
                
                log_universal('DEBUG', 'Audio', f'MusiCNN tags extracted: {len(tags)} tags')
                
                # Run embeddings inference using loaded model
                log_universal('DEBUG', 'Audio', 'Running MusiCNN embeddings inference...')
                embeddings = self._musicnn_embeddings_model(audio_16k)
                
                # Handle embeddings
                if isinstance(embeddings, list):
                    embeddings = np.array(embeddings)
                
                log_universal('DEBUG', 'Audio', f'MusiCNN embeddings shape: {embeddings.shape}')
                
                # Calculate embedding statistics
                embedding_mean = embeddings.mean(axis=0).tolist()
                embedding_std = embeddings.std(axis=0).tolist()
                
                features.update({
                    'embedding': embedding_mean,
                    'embedding_std': embedding_std,
                    'tags': tags
                })
                
                log_universal('INFO', 'Audio', f'MusiCNN extraction completed successfully: {len(tags)} tags, {len(embedding_mean)} embedding dimensions')
                
            except Exception as e:
                log_universal('ERROR', 'Audio', f'MusiCNN inference failed: {e}')
                features.update({'embedding': [], 'tags': {}})
            
        except Exception as e:
            log_universal('ERROR', 'Audio', f'MusiCNN feature extraction failed: {e}')
            features.update({'embedding': [], 'tags': {}})
        
        return features
    
    def _extract_chroma_features(self, audio: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Extract chroma features using frame-by-frame processing."""
        features = {}
        
        try:
            if ESSENTIA_AVAILABLE:
                # Use frame-by-frame approach like the old working version
                frame_size = 2048
                hop_size = 1024
                
                # Limit audio length to prevent spectrum size issues
                max_audio_length = 60 * sample_rate  # 60 seconds max
                if len(audio) > max_audio_length:
                    log_universal('WARNING', 'Audio', f'Audio too long for chroma extraction, truncating to 60s')
                    audio = audio[:max_audio_length]
                
                # Initialize algorithms
                window = es.Windowing(type='blackmanharris62')
                spectrum = es.Spectrum()
                spectral_peaks = es.SpectralPeaks()
                hpcp = es.HPCP()
                
                # Process audio in frames
                chroma_values = []
                frame_count = 0
                
                for frame in es.FrameGenerator(audio, frameSize=frame_size, hopSize=hop_size):
                    frame_count += 1
                    
                    try:
                        windowed = window(frame)
                        spec = spectrum(windowed)
                        
                        # Validate spectrum size before processing
                        if len(spec) > 10000:  # Limit spectrum size to prevent TriangularBands error
                            log_universal('WARNING', 'Audio', f'Spectrum size too large ({len(spec)}), skipping frame')
                            continue
                        
                        frequencies, magnitudes = spectral_peaks(spec)
                        
                        # Check if we have valid spectral peaks
                        if len(frequencies) > 0 and len(magnitudes) > 0:
                            freq_array = np.array(frequencies)
                            mag_array = np.array(magnitudes)
                            
                            # Check for valid frequency and magnitude ranges
                            if len(freq_array) > 0 and len(mag_array) > 0 and np.any(mag_array > 0):
                                hpcp_value = hpcp(freq_array, mag_array)
                                if hpcp_value is not None and len(hpcp_value) == 12:
                                    chroma_values.append(hpcp_value)
                    except Exception as frame_error:
                        log_universal('DEBUG', 'Audio', f'Frame {frame_count} processing failed: {frame_error}')
                        continue
                
                # Calculate global average
                if chroma_values:
                    chroma_avg = np.mean(chroma_values, axis=0).tolist()
                    features['chroma_mean'] = chroma_avg
                    features['chroma_std'] = np.std(chroma_values, axis=0).tolist()
                else:
                    features['chroma_mean'] = [0.0] * 12
                    features['chroma_std'] = [0.0] * 12
                
            elif LIBROSA_AVAILABLE:
                # Use librosa as fallback
                chroma = librosa.feature.chroma_cqt(y=audio, sr=sample_rate)
                features['chroma_mean'] = np.mean(chroma, axis=1).tolist()
                features['chroma_std'] = np.std(chroma, axis=1).tolist()
            
            log_universal('DEBUG', 'Audio', f'Extracted chroma features')
            
        except Exception as e:
            log_universal('WARNING', 'Audio', f'Chroma feature extraction failed: {e}')
            features.update({
                'chroma_mean': [0.0] * 12,
                'chroma_std': [0.0] * 12
            })
        
        return features
    
    def _is_long_audio_track(self, file_path: str) -> bool:
        """
        Check if a file is a long audio track (20+ minutes).
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            True if it's a long audio track, False otherwise
        """
        try:
            # Get file size
            file_size_bytes = os.path.getsize(file_path)
            file_size_mb = file_size_bytes / (1024 * 1024)
            
            # Estimate duration based on file size
            # Assuming average bitrate of 320kbps for MP3
            # Duration = (file_size_bytes * 8) / (bitrate * 1000)
            estimated_duration_seconds = (file_size_bytes * 8) / (320 * 1000)
            estimated_duration_minutes = estimated_duration_seconds / 60
            
            # Much higher threshold for sequential processing: consider it long if estimated duration > 120 minutes (increased from 60)
            is_long = estimated_duration_minutes > 120
            
            log_universal('DEBUG', 'Audio', f'File {os.path.basename(file_path)}: {estimated_duration_minutes:.1f} minutes estimated, long_audio: {is_long}')
            
            return is_long
            
        except Exception as e:
            log_universal('WARNING', 'Audio', f'Could not determine if {file_path} is long audio: {e}')
            return False
    
    def _is_extremely_large_for_processing(self, audio: np.ndarray) -> bool:
        """
        Check if audio is extremely large and should skip certain features.
        Based on original logic: skip for files > 500M samples.
        
        Args:
            audio: Audio array
            
        Returns:
            True if extremely large, False otherwise
        """
        try:
            # Much higher threshold for sequential processing: skip for files > 5B samples (was 1B)
            is_extremely_large = len(audio) > 5000000000
            
            if is_extremely_large:
                log_universal('WARNING', 'Audio', f'Extremely large file detected: {len(audio)} samples ({len(audio)/44100/60:.1f} minutes)')
                log_universal('WARNING', 'Audio', f'Skipping memory-intensive features for this file')
            
            return is_extremely_large
            
        except Exception as e:
            log_universal('WARNING', 'Audio', f'Could not determine if file is extremely large: {e}')
            return False

    def _should_skip_audio_loading(self, file_path: str) -> bool:
        """
        Check if audio file should be skipped from loading into RAM.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            True if should be skipped, False otherwise
        """
        try:
            file_size = os.path.getsize(file_path)
            file_size_mb = file_size / (1024 * 1024)
            
            # Get configuration values - much higher thresholds for sequential processing
            streaming_threshold_mb = self.config.get('STREAMING_LARGE_FILE_THRESHOLD_MB', 2000)  # Increased from 500MB to 2GB
            skip_large_files = self.config.get('SKIP_LARGE_FILES', False)  # Changed default to False for sequential processing
            
            # Convert string config to boolean if needed
            if isinstance(skip_large_files, str):
                skip_large_files = skip_large_files.lower() in ('true', '1', 'yes', 'on')
            
            # Skip files larger than streaming threshold to prevent RAM saturation
            if skip_large_files and file_size_mb > streaming_threshold_mb:
                log_universal('WARNING', 'Audio', f'Using lightweight analysis for large file ({file_size_mb:.1f}MB): {os.path.basename(file_path)}')
                return True
                
            return False
            
        except Exception as e:
            log_universal('WARNING', 'Audio', f'Cannot check file size for {os.path.basename(file_path)}: {e}')
            return False
    
    def _get_audio_type(self, file_path: str, audio: np.ndarray = None) -> str:
        """
        Determine the type of audio file (normal, long_mix, radio, etc.).
        Based on duration and characteristics.
        
        Args:
            file_path: Path to audio file
            audio: Optional audio array for duration calculation
            
        Returns:
            Audio type string
        """
        try:
            # Calculate duration
            if audio is not None:
                duration_minutes = len(audio) / self.sample_rate / 60
            else:
                # Estimate from file size
                file_size_bytes = os.path.getsize(file_path)
                estimated_duration_seconds = (file_size_bytes * 8) / (320 * 1000)
                duration_minutes = estimated_duration_seconds / 60
            
            # Determine type based on duration - expanded for sequential processing
            if duration_minutes > 120:  # Over 2 hours
                return 'radio'
            elif duration_minutes > 60:  # 1-2 hours
                return 'long_mix'
            elif duration_minutes > 30:  # 30-60 minutes
                return 'mix'
            else:
                return 'normal'
                
        except Exception as e:
            log_universal('WARNING', 'Audio', f'Could not determine audio type: {e}')
            return 'normal'

    def _determine_long_audio_category(self, audio_path: str, metadata: Dict[str, Any] = None, 
                                     audio_features: Dict[str, Any] = None) -> str:
        """
        Determine the category of a long audio track using metadata and audio features.

        Args:
            audio_path: Path to the audio file
            metadata: Optional metadata dictionary
            audio_features: Optional audio features dictionary

        Returns:
            Category string (long_mix, podcast, radio, compilation, or unknown)
        """
        try:
            # Get configured categories
            categories_str = self.config.get('LONG_AUDIO_CATEGORIES', 'long_mix,podcast,radio,compilation')
            if isinstance(categories_str, str):
                categories = categories_str.split(',')
            else:
                categories = categories_str

            # Priority 1: Use audio features for categorization (most accurate)
            if audio_features:
                category = self._categorize_by_audio_features(audio_features)
                if category:
                    log_universal('INFO', 'Audio', f"Category determined by audio features: {category}")
                    return category

            # Priority 2: Check metadata for explicit indicators
            if metadata:
                title = str(metadata.get('title', '')).lower()
                artist = str(metadata.get('artist', '')).lower()
                album = str(metadata.get('album', '')).lower()

                # Strong metadata indicators (explicit keywords)
                # Enhanced metadata detection with more keywords
                # Check for radio shows first (they often have "episode" but are not podcasts)
                if any(word in title for word in ['radio', 'broadcast', 'live', 'fm', 'am', 'station', 'trance', 'state of trance']):
                    log_universal('INFO', 'Audio', f"Category determined by metadata (title): radio")
                    return 'radio'
                if any(word in artist for word in ['radio', 'station', 'broadcast', 'fm', 'am']):
                    log_universal('INFO', 'Audio', f"Category determined by metadata (artist): radio")
                    return 'radio'
                
                # Then check for actual podcasts (speech-based content)
                if any(word in title for word in ['podcast', 'talk', 'interview', 'conversation', 'discussion']):
                    log_universal('INFO', 'Audio', f"Category determined by metadata (title): podcast")
                    return 'podcast'
                if any(word in artist for word in ['podcast', 'show', 'network']):
                    log_universal('INFO', 'Audio', f"Category determined by metadata (artist): podcast")
                    return 'podcast'
                if any(word in artist for word in ['podcast', 'radio', 'station', 'show', 'network']):
                    log_universal('INFO', 'Audio', f"Category determined by metadata (artist): podcast")
                    return 'podcast'

                if any(word in title for word in ['radio', 'broadcast', 'live', 'fm', 'am', 'station']):
                    log_universal('INFO', 'Audio', f"Category determined by metadata (title): radio")
                    return 'radio'
                if any(word in artist for word in ['radio', 'station', 'broadcast', 'fm', 'am']):
                    log_universal('INFO', 'Audio', f"Category determined by metadata (artist): radio")
                    return 'radio'

                if any(word in title for word in ['mix', 'dj', 'set', 'session', 'live set', 'concert', 'performance']):
                    log_universal('INFO', 'Audio', f"Category determined by metadata (title): long_mix")
                    return 'long_mix'
                if any(word in artist for word in ['dj', 'mix', 'session', 'live', 'concert']):
                    log_universal('INFO', 'Audio', f"Category determined by metadata (artist): long_mix")
                    return 'long_mix'

                if any(word in title for word in ['compilation', 'collection', 'various', 'best of', 'greatest hits', 'anthology']):
                    log_universal('INFO', 'Audio', f"Category determined by metadata (title): compilation")
                    return 'compilation'
                if any(word in album for word in ['compilation', 'collection', 'various', 'best of', 'greatest hits', 'anthology']):
                    log_universal('INFO', 'Audio', f"Category determined by metadata (album): compilation")
                    return 'compilation'

                # New categories for better classification
                if any(word in title for word in ['classical', 'orchestra', 'symphony', 'concerto', 'sonata']):
                    log_universal('INFO', 'Audio', f"Category determined by metadata (title): classical")
                    return 'classical'
                if any(word in artist for word in ['orchestra', 'symphony', 'philharmonic', 'quartet']):
                    log_universal('INFO', 'Audio', f"Category determined by metadata (artist): classical")
                    return 'classical'

                if any(word in title for word in ['jazz', 'swing', 'bebop', 'fusion']):
                    log_universal('INFO', 'Audio', f"Category determined by metadata (title): jazz")
                    return 'jazz'
                if any(word in artist for word in ['jazz', 'swing', 'bebop', 'fusion']):
                    log_universal('INFO', 'Audio', f"Category determined by metadata (artist): jazz")
                    return 'jazz'

            # Priority 3: Check filename for patterns
            filename = os.path.basename(audio_path).lower()

            if any(word in filename for word in ['podcast', 'episode', 'show', 'talk', 'interview']):
                return 'podcast'
            if any(word in filename for word in ['radio', 'broadcast', 'live', 'fm', 'am']):
                return 'radio'
            if any(word in filename for word in ['mix', 'dj', 'set', 'concert', 'performance']):
                return 'long_mix'
            if any(word in filename for word in ['compilation', 'collection', 'various', 'best', 'hits']):
                return 'compilation'
            if any(word in filename for word in ['classical', 'orchestra', 'symphony']):
                return 'classical'
            if any(word in filename for word in ['jazz', 'swing', 'bebop']):
                return 'jazz'

            # Default fallback
            return 'long_mix'

        except Exception as e:
            log_universal('WARNING', 'Audio', f"Error determining long audio category: {e}")
            return 'long_mix'

    def _categorize_by_audio_features(self, features: Dict[str, Any]) -> Optional[str]:
        """
        Categorize long audio track based on comprehensive audio features including MusiCNN tags.
        Enhanced for big files with full feature extraction.

        Args:
            features: Dictionary of audio features

        Returns:
            Category string or None if cannot determine
        """
        log_universal('DEBUG', 'Audio', f'Starting audio feature categorization with {len(features)} features')
        log_universal('DEBUG', 'Audio', f'Available features: {list(features.keys())}')
        
        try:
            # Extract basic features with None handling
            bpm = features.get('bpm')
            confidence = features.get('rhythm_confidence', features.get('confidence', 0.0))
            spectral_centroid = features.get('spectral_centroid')
            spectral_flatness = features.get('spectral_flatness')
            loudness = features.get('loudness')
            dynamic_complexity = features.get('dynamic_complexity')
            
            # Extract advanced features
            mfcc_coefficients = features.get('mfcc_coefficients', [])
            chroma_mean = features.get('chroma_mean', [])
            tags = features.get('tags', {})
            
            # Use external BPM if available and main BPM is None
            if bpm is None and 'external_bpm' in features:
                bpm = features.get('external_bpm')
                log_universal('INFO', 'Audio', f'Using metadata BPM as fallback: {bpm:.1f}')
            elif bpm == -1 and 'external_bpm' in features:
                # Rhythm extraction failed, but we have metadata BPM
                bpm = features.get('external_bpm')
                log_universal('INFO', 'Audio', f'Rhythm extraction failed, using metadata BPM: {bpm:.1f}')
            elif bpm == -1:
                log_universal('WARNING', 'Audio', 'Rhythm extraction failed and no metadata BPM available')
            
            # Set default values for None features
            if bpm is None or bpm == -1:
                bpm = -1
            if spectral_centroid is None:
                spectral_centroid = 0
            if spectral_flatness is None:
                spectral_flatness = 0.5
            if loudness is None:
                loudness = 0.5
            if dynamic_complexity is None:
                dynamic_complexity = 0.5

            # Calculate derived features
            mfcc_variance = np.var(mfcc_coefficients) if mfcc_coefficients else 0
            chroma_variance = np.var(chroma_mean) if chroma_mean else 0
            
            # Extract MusiCNN tag scores (using actual MusiCNN tag names)
            # Map common MusiCNN tags to categorization features
            electronic_score = 0
            dance_score = 0
            rock_score = 0
            speechiness = 0
            acousticness = 0
            
            if isinstance(tags, dict):
                # Map actual MusiCNN tags to categorization features
                for tag, score in tags.items():
                    tag_lower = tag.lower()
                    
                    # Electronic/Dance/Trance music (actual MusiCNN tags)
                    if tag_lower in ['electronic', 'dance', 'electronica', 'electro', 'house', 'party']:
                        electronic_score = max(electronic_score, score)
                    # Pop/Dance music
                    elif tag_lower in ['pop', 'dance', 'funk', 'catchy']:
                        dance_score = max(dance_score, score)
                    # Rock/Metal music
                    elif tag_lower in ['rock', 'metal', 'punk', 'alternative', 'alternative rock', 'classic rock', 'hard rock', 'heavy metal', 'progressive rock']:
                        rock_score = max(rock_score, score)
                    # Vocal content
                    elif tag_lower in ['female vocalists', 'male vocalists', 'female vocalist']:
                        speechiness = max(speechiness, score)
                    # Acoustic/Folk music
                    elif tag_lower in ['acoustic', 'folk', 'country', 'jazz', 'blues', 'mellow', 'easy listening']:
                        acousticness = max(acousticness, score)
                    # Chill/Ambient (could be electronic)
                    elif tag_lower in ['chill', 'chillout', 'ambient']:
                        electronic_score = max(electronic_score, score * 0.7)
                    # Instrumental (could be electronic)
                    elif tag_lower == 'instrumental':
                        electronic_score = max(electronic_score, score * 0.6)
                    # Experimental (could be electronic)
                    elif tag_lower == 'experimental':
                        electronic_score = max(electronic_score, score * 0.5)
                    # Non-electronic genres (ignore for electronic detection)
                    elif tag_lower in ['rnb', 'soul', 'hip-hop', 'indie', 'indie rock', 'indie pop', 'oldies', '60s', '70s', '80s', '90s', '00s']:
                        pass
                    # Special handling for potentially misleading tags in electronic music
                    elif tag_lower in ['mellow', 'beautiful', 'sexy', 'sad', 'happy']:
                        # These are mood tags that could apply to any genre
                        # Don't count them strongly for any category
                        pass
            
            # Debug logging for MusiCNN tags
            log_universal('DEBUG', 'Audio', f'MusiCNN tags available: {list(tags.keys()) if isinstance(tags, dict) else "No tags"}')
            log_universal('DEBUG', 'Audio', f'Mapped MusiCNN scores: electronic={electronic_score:.2f}, dance={dance_score:.2f}, rock={rock_score:.2f}, speechiness={speechiness:.2f}, acousticness={acousticness:.2f}')

            log_universal('DEBUG', 'Audio', f'Enhanced categorization with: BPM={bpm}, Confidence={confidence:.2f}, '
                          f'Spectral_Centroid={spectral_centroid:.0f}, Spectral_Flatness={spectral_flatness:.2f}, '
                          f'Loudness={loudness:.2f}, Dynamic_Complexity={dynamic_complexity:.2f}, '
                          f'MFCC_Variance={mfcc_variance:.2f}, Chroma_Variance={chroma_variance:.2f}, '
                          f'Speechiness={speechiness:.2f}, Electronic={electronic_score:.2f}')

            # Enhanced categorization logic using comprehensive features including MusiCNN tags

            # 1. Podcast detection (speech-like characteristics)
            # - High speechiness from MusiCNN
            # - Low BPM and confidence
            # - Low spectral centroid (speech frequencies)
            # - High spectral flatness (noise-like)
            # - Low MFCC variance (consistent speech patterns)
            if (speechiness > 0.6 or 
                (bpm > 0 and bpm < 100 and confidence < 0.7 and
                 spectral_centroid < 3000 and spectral_flatness > 0.25 and
                 mfcc_variance < 0.1)):
                log_universal('INFO', 'Audio', f'Detected as podcast based on speech-like characteristics (speechiness={speechiness:.2f})')
                return 'podcast'

            # 1.5. Radio Show detection (mixed electronic content with consistent BPM)
            # - Medium electronic score from MusiCNN OR consistent BPM characteristics
            # - Consistent BPM (120-140 typical for trance)
            # - High confidence (consistent rhythm)
            # - High spectral centroid (rich harmonics)
            # - Low spectral flatness (tonal content)
            if ((electronic_score > 0.2 or  # MusiCNN electronic score
                 (bpm > 0 and 110 <= bpm <= 150 and confidence > 0.5 and  # OR consistent BPM
                  spectral_centroid > 2000 and spectral_flatness < 0.5))):  # AND spectral characteristics
                log_universal('INFO', 'Audio', f'Detected as radio show based on characteristics (electronic={electronic_score:.2f}, BPM={bpm:.1f}, confidence={confidence:.2f})')
                return 'radio'

            # 2. Electronic/Dance Mix detection (high energy, consistent electronic music)
            # - High electronic score from MusiCNN
            # - High BPM (dance/electronic music)
            # - High confidence (consistent rhythm)
            # - High spectral centroid (rich harmonics)
            # - Low spectral flatness (tonal content)
            # - High loudness (energy)
            # - Low chroma variance (consistent key)
            if ((electronic_score > 0.3 or dance_score > 0.4 or  # Lowered thresholds for MusiCNN scores
                (bpm > 0 and bpm > 120 and confidence > 0.6 and
                 spectral_centroid > 2500 and spectral_flatness < 0.4 and
                 loudness > 0.4 and chroma_variance < 0.2))):
                log_universal('INFO', 'Audio', f'Detected as electronic mix based on electronic/dance characteristics (electronic={electronic_score:.2f}, dance={dance_score:.2f})')
                return 'electronic_mix'

            # 3. Rock/Alternative Mix detection
            # - High rock score from MusiCNN
            # - Medium-high BPM
            # - High dynamic complexity
            # - Medium spectral characteristics
            if (rock_score > 0.7 or
                (bpm > 0 and 100 <= bpm <= 140 and dynamic_complexity > 0.6 and
                 spectral_centroid > 2000 and spectral_flatness < 0.5)):
                log_universal('INFO', 'Audio', 'Detected as rock mix based on rock/alternative characteristics')
                return 'rock_mix'

            # 4. Acoustic/Classical Mix detection
            # - High acousticness from MusiCNN
            # - Low electronic score
            # - Medium BPM
            # - Lower spectral centroid (acoustic frequencies)
            if (acousticness > 0.7 and electronic_score < 0.3 or
                (bpm > 0 and 60 <= bpm <= 120 and spectral_centroid < 2500 and
                 spectral_flatness < 0.4 and acousticness > 0.5)):
                log_universal('INFO', 'Audio', 'Detected as acoustic mix based on acoustic/classical characteristics')
                return 'acoustic_mix'

            # 5. Radio detection (mixed content, variable characteristics)
            # - Medium BPM range (mixed music)
            # - Variable confidence (mixed content)
            # - Medium spectral characteristics
            # - High MFCC variance (variable content)
            # - High chroma variance (key changes)
            if (bpm > 0 and 70 <= bpm <= 150 and 0.3 <= confidence <= 0.8 and
                spectral_centroid > 1500 and spectral_flatness < 0.5 and
                mfcc_variance > 0.15 and chroma_variance > 0.1):
                log_universal('INFO', 'Audio', f'Detected as radio based on mixed content characteristics (BPM={bpm:.1f}, confidence={confidence:.2f})')
                return 'radio'

            # 6. Compilation detection (variable characteristics)
            # - Variable BPM (different songs)
            # - Lower confidence (inconsistent rhythm)
            # - Higher spectral flatness (variable content)
            # - Higher dynamic complexity (varied sections)
            # - High MFCC and chroma variance (different songs)
            if (bpm > 0 and bpm < 120 and confidence < 0.7 and
                spectral_flatness > 0.3 and dynamic_complexity > 0.5 and
                mfcc_variance > 0.2 and chroma_variance > 0.15):
                log_universal('INFO', 'Audio', 'Detected as compilation based on variable characteristics')
                return 'compilation'

            # 7. Enhanced fallback categorization using comprehensive feature analysis
            if bpm > 0:
                # Calculate feature scores for better decision making
                electronic_indicators = 0
                dance_indicators = 0
                rock_indicators = 0
                acoustic_indicators = 0
                speech_indicators = 0
                
                # Score based on BPM
                if 120 <= bpm <= 140:  # Typical electronic/dance range
                    electronic_indicators += 2
                    dance_indicators += 2
                elif 140 < bpm <= 160:  # High energy dance
                    dance_indicators += 3
                elif 80 <= bpm <= 120:  # Rock/pop range
                    rock_indicators += 1
                elif bpm < 80:  # Slow/ambient
                    acoustic_indicators += 1
                
                # Score based on confidence
                if confidence > 0.7:  # Strong rhythm
                    dance_indicators += 1
                    electronic_indicators += 1
                elif confidence < 0.4:  # Weak rhythm
                    speech_indicators += 1
                
                # Score based on spectral characteristics
                if spectral_centroid > 2500:  # Rich harmonics (electronic)
                    electronic_indicators += 1
                elif spectral_centroid < 1500:  # Low frequencies (speech/acoustic)
                    speech_indicators += 1
                    acoustic_indicators += 1
                
                if spectral_flatness < 0.3:  # Tonal content (music)
                    electronic_indicators += 1
                    dance_indicators += 1
                elif spectral_flatness > 0.5:  # Noise-like (speech/ambient)
                    speech_indicators += 1
                
                # Score based on loudness
                if loudness > 0.6:  # High energy
                    dance_indicators += 1
                elif loudness < 0.3:  # Low energy
                    acoustic_indicators += 1
                
                # Score based on MFCC variance
                if mfcc_variance > 0.2:  # Variable content
                    speech_indicators += 1
                elif mfcc_variance < 0.1:  # Consistent content
                    electronic_indicators += 1
                
                # Score based on chroma variance
                if chroma_variance > 0.15:  # Key changes
                    speech_indicators += 1
                elif chroma_variance < 0.1:  # Consistent key
                    electronic_indicators += 1
                
                # Add MusiCNN scores (weighted)
                electronic_indicators += electronic_score * 2
                dance_indicators += dance_score * 2
                rock_indicators += rock_score * 2
                speech_indicators += speechiness * 2
                acoustic_indicators += acousticness * 2
                
                # Determine category based on highest score
                scores = {
                    'electronic': electronic_indicators,
                    'dance': dance_indicators,
                    'rock': rock_indicators,
                    'acoustic': acoustic_indicators,
                    'speech': speech_indicators
                }
                
                best_category = max(scores, key=scores.get)
                best_score = scores[best_category]
                
                log_universal('INFO', 'Audio', f'Enhanced fallback analysis - Scores: {scores}')
                
                if best_score >= 3:  # Strong indication
                    if best_category in ['electronic', 'dance']:
                        log_universal('INFO', 'Audio', f'Fallback: Detected as electronic mix based on {best_category} characteristics (score={best_score:.1f})')
                        return 'electronic_mix'
                    elif best_category == 'speech':
                        log_universal('INFO', 'Audio', f'Fallback: Detected as podcast based on speech characteristics (score={best_score:.1f})')
                        return 'podcast'
                    elif best_category == 'rock':
                        log_universal('INFO', 'Audio', f'Fallback: Detected as rock mix based on rock characteristics (score={best_score:.1f})')
                        return 'rock_mix'
                    elif best_category == 'acoustic':
                        log_universal('INFO', 'Audio', f'Fallback: Detected as acoustic mix based on acoustic characteristics (score={best_score:.1f})')
                        return 'acoustic_mix'
                else:
                    # Weak indicators - use BPM-based fallback
                    if bpm > 120 and confidence > 0.5:
                        log_universal('INFO', 'Audio', f'Fallback: Detected as electronic mix based on high BPM ({bpm:.1f}) and confidence ({confidence:.2f})')
                        return 'electronic_mix'
                    elif bpm < 90 and confidence < 0.6:
                        log_universal('INFO', 'Audio', f'Fallback: Detected as podcast based on low BPM ({bpm:.1f}) and confidence ({confidence:.2f})')
                        return 'podcast'
                    else:
                        log_universal('INFO', 'Audio', f'Fallback: Detected as radio based on medium characteristics (BPM={bpm:.1f}, confidence={confidence:.2f})')
                        return 'radio'
            
            # 8. Handle failed rhythm extraction (BPM = -1) with enhanced analysis
            if bpm == -1:
                log_universal('INFO', 'Audio', 'Rhythm extraction failed, using comprehensive features for categorization')
                
                # Calculate feature scores without BPM
                electronic_indicators = electronic_score * 3  # Weight MusiCNN more heavily
                dance_indicators = dance_score * 3
                rock_indicators = rock_score * 3
                acoustic_indicators = acousticness * 3
                speech_indicators = speechiness * 3
                
                # Score based on spectral characteristics
                if spectral_centroid > 2500:  # Rich harmonics (electronic)
                    electronic_indicators += 2
                elif spectral_centroid < 1500:  # Low frequencies (speech/acoustic)
                    speech_indicators += 2
                    acoustic_indicators += 1
                
                if spectral_flatness < 0.3:  # Tonal content (music)
                    electronic_indicators += 1
                    dance_indicators += 1
                elif spectral_flatness > 0.5:  # Noise-like (speech/ambient)
                    speech_indicators += 2
                
                # Score based on loudness
                if loudness > 0.6:  # High energy
                    dance_indicators += 1
                elif loudness < 0.3:  # Low energy
                    acoustic_indicators += 1
                
                # Score based on MFCC and chroma variance
                if mfcc_variance > 0.2:  # Variable content
                    speech_indicators += 1
                elif mfcc_variance < 0.1:  # Consistent content
                    electronic_indicators += 1
                
                if chroma_variance > 0.15:  # Key changes
                    speech_indicators += 1
                elif chroma_variance < 0.1:  # Consistent key
                    electronic_indicators += 1
                
                # Determine category based on highest score
                scores = {
                    'electronic': electronic_indicators,
                    'dance': dance_indicators,
                    'rock': rock_indicators,
                    'acoustic': acoustic_indicators,
                    'speech': speech_indicators
                }
                
                best_category = max(scores, key=scores.get)
                best_score = scores[best_category]
                
                log_universal('INFO', 'Audio', f'No-BPM analysis - Scores: {scores}')
                
                if best_score >= 2:  # Strong indication
                    if best_category in ['electronic', 'dance']:
                        log_universal('INFO', 'Audio', f'No-BPM fallback: Detected as electronic mix based on {best_category} characteristics (score={best_score:.1f})')
                        return 'electronic_mix'
                    elif best_category == 'speech':
                        log_universal('INFO', 'Audio', f'No-BPM fallback: Detected as podcast based on speech characteristics (score={best_score:.1f})')
                        return 'podcast'
                    elif best_category in ['rock', 'acoustic']:
                        log_universal('INFO', 'Audio', f'No-BPM fallback: Detected as {best_category} mix based on {best_category} characteristics (score={best_score:.1f})')
                        return f'{best_category}_mix'
                else:
                    # Weak indicators - use spectral characteristics
                    if spectral_centroid > 2500 and spectral_flatness < 0.3:
                        log_universal('INFO', 'Audio', 'No-BPM fallback: Detected as electronic mix based on spectral characteristics')
                        return 'electronic_mix'
                    elif spectral_centroid < 2000:
                        log_universal('INFO', 'Audio', 'No-BPM fallback: Detected as podcast based on low spectral centroid')
                        return 'podcast'
                    else:
                        log_universal('INFO', 'Audio', 'No-BPM fallback: Detected as radio based on spectral characteristics')
                        return 'radio'
            
            # 9. Final fallback based on comprehensive characteristics
            if speechiness > 0.4:
                log_universal('INFO', 'Audio', 'Final fallback: Detected as podcast based on speechiness')
                return 'podcast'
            elif electronic_score > 0.5 or dance_score > 0.6:
                log_universal('INFO', 'Audio', 'Final fallback: Detected as electronic mix based on electronic/dance tags')
                return 'electronic_mix'
            elif spectral_centroid < 2000:
                log_universal('INFO', 'Audio', 'Final fallback: Detected as podcast based on low spectral centroid')
                return 'podcast'
            elif spectral_flatness > 0.4 and mfcc_variance > 0.2:
                log_universal('INFO', 'Audio', 'Final fallback: Detected as compilation based on high spectral flatness and MFCC variance')
                return 'compilation'
            else:
                log_universal('INFO', 'Audio', 'Final fallback: Detected as electronic mix based on tonal characteristics')
                return 'electronic_mix'

        except Exception as e:
            log_universal('WARNING', 'Audio', f"Error in audio feature categorization: {e}")
            return 'electronic_mix'  # Safe fallback instead of None

    def _extract_optimized_features_for_categorization(self, audio: np.ndarray, sample_rate: int,
                                                      metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Extract optimized features for long audio categorization.
        Focuses on efficient sampling and categorization-relevant features.
        
        Args:
            audio: Audio array
            sample_rate: Sample rate
            metadata: Optional metadata
            
        Returns:
            Dictionary with optimized features for categorization
        """
        log_universal('INFO', 'Audio', f'Starting optimized categorization analysis: audio_length={len(audio)}, sample_rate={sample_rate}')
        features = {}
        
        try:
            # Sample 4 different 30-second segments for better representation
            sample_duration = 30  # seconds
            sample_size = sample_duration * sample_rate
            total_duration = len(audio) / sample_rate
            
            # Calculate 4 sample positions: 10%, 30%, 60%, 80% of track
            positions = [0.1, 0.3, 0.6, 0.8]
            audio_samples = []
            
            for i, pos in enumerate(positions):
                start_sample = int(pos * len(audio))
                end_sample = min(start_sample + sample_size, len(audio))
                sample = audio[start_sample:end_sample]
                audio_samples.append(sample)
                log_universal('DEBUG', 'Audio', f'Sample {i+1}: {pos*100:.0f}% of track ({start_sample/sample_rate/60:.1f}min)')
            
            # Use the first sample for initial analysis, then combine results
            audio_sample = audio_samples[0]
            
            log_universal('INFO', 'Audio', f'Using 4x{sample_duration}s samples from {len(audio)/sample_rate/60:.1f}min track for categorization')
            
            # Extract rhythm features from all 4 samples for better BPM detection
            if self.extract_rhythm:
                log_universal('INFO', 'Audio', 'Starting rhythm analysis for categorization (4 samples)')
                
                all_rhythm_features = []
                all_bpms = []
                
                for i, sample in enumerate(audio_samples):
                    log_universal('DEBUG', 'Audio', f'Analyzing rhythm sample {i+1}/4')
                    sample_rhythm = self._extract_rhythm_features(sample, sample_rate)
                    if sample_rhythm and 'bpm' in sample_rhythm:
                        all_rhythm_features.append(sample_rhythm)
                        all_bpms.append(sample_rhythm['bpm'])
                
                # Combine rhythm features from all samples
                if all_rhythm_features:
                    # Use median BPM (more robust than average)
                    if all_bpms:
                        median_bpm = np.median(all_bpms)
                        features['bpm'] = float(median_bpm)
                        features['bpm_estimates'] = all_bpms
                        log_universal('INFO', 'Audio', f'Combined BPM from {len(all_bpms)} samples: median={median_bpm:.1f}, range={min(all_bpms):.1f}-{max(all_bpms):.1f}')
                    
                    # Use highest confidence from all samples
                    confidences = [f.get('rhythm_confidence', 0) for f in all_rhythm_features]
                    if confidences:
                        max_confidence = max(confidences)
                        features['rhythm_confidence'] = max_confidence
                        log_universal('INFO', 'Audio', f'Best rhythm confidence: {max_confidence:.2f}')
                    
                    log_universal('INFO', 'Audio', f'Rhythm features combined from {len(all_rhythm_features)} samples')
                else:
                    log_universal('WARNING', 'Audio', 'Rhythm features extraction returned None for all samples')
            else:
                log_universal('INFO', 'Audio', f'Rhythm extraction disabled: extract_rhythm={self.extract_rhythm}')
            
            # Extract basic spectral features (centroid, flatness)
            if self.extract_spectral:
                spectral_features = self._extract_spectral_features(audio_sample, sample_rate)
                features.update(spectral_features)
            
            # Extract loudness features (RMS)
            if self.extract_loudness:
                loudness_features = self._extract_loudness_features(audio_sample, sample_rate)
                features.update(loudness_features)
            
            # Extract key features (for genre classification)
            if self.extract_key:
                key_features = self._extract_key_features(audio_sample, sample_rate)
                features.update(key_features)
            
            # Extract lightweight MFCC (for genre classification)
            if self.extract_mfcc:
                mfcc_features = self._extract_mfcc_features(audio_sample, sample_rate)
                features.update(mfcc_features)
            
            # Extract MusiCNN on all 4 samples for better genre classification
            if self.extract_musicnn and TENSORFLOW_AVAILABLE:
                log_universal('INFO', 'Audio', 'Starting MusiCNN analysis for categorization (4 samples)')
                
                # Analyze all 4 samples and combine results
                all_musicnn_features = []
                all_tags = []
                
                for i, sample in enumerate(audio_samples):
                    log_universal('DEBUG', 'Audio', f'Analyzing MusiCNN sample {i+1}/4')
                    
                    # Use shorter samples for MusiCNN (3 seconds each) to avoid memory issues
                    musicnn_sample_duration = 3  # seconds - MusiCNN expects ~3 seconds
                    musicnn_sample_size = musicnn_sample_duration * sample_rate
                    
                    # Take a 3-second segment from the middle of the 30-second sample
                    if len(sample) > musicnn_sample_size:
                        start_idx = (len(sample) - musicnn_sample_size) // 2
                        musicnn_sample = sample[start_idx:start_idx + musicnn_sample_size]
                        log_universal('DEBUG', 'Audio', f'Using 3s segment from 30s sample {i+1} (position {start_idx/sample_rate:.1f}s)')
                    else:
                        musicnn_sample = sample
                        log_universal('DEBUG', 'Audio', f'Using full sample {i+1} (shorter than 3s)')
                    
                    sample_features = self._extract_musicnn_features(musicnn_sample, sample_rate)
                    if sample_features:
                        all_musicnn_features.append(sample_features)
                        if 'tags' in sample_features:
                            all_tags.append(sample_features['tags'])
                
                # Combine results from all samples
                if all_musicnn_features:
                    # Average embeddings from all samples
                    embeddings = [f.get('embedding', []) for f in all_musicnn_features if 'embedding' in f]
                    if embeddings:
                        avg_embedding = np.mean(embeddings, axis=0).tolist()
                        features['embedding'] = avg_embedding
                        log_universal('INFO', 'Audio', f'Combined MusiCNN embedding from {len(embeddings)} samples')
                    
                    # Combine tags from all samples (average scores)
                    if all_tags:
                        combined_tags = {}
                        tag_count = {}
                        
                        for tags in all_tags:
                            if isinstance(tags, dict):
                                for tag, score in tags.items():
                                    if tag in combined_tags:
                                        combined_tags[tag] += score
                                        tag_count[tag] += 1
                                    else:
                                        combined_tags[tag] = score
                                        tag_count[tag] = 1
                        
                        # Average the scores
                        for tag in combined_tags:
                            combined_tags[tag] /= tag_count[tag]
                        
                        features['tags'] = combined_tags
                        log_universal('INFO', 'Audio', f'Combined MusiCNN tags from {len(all_tags)} samples')
                        
                        # Log top tags from combined analysis
                        top_tags = sorted(combined_tags.items(), key=lambda x: x[1], reverse=True)[:5]
                        top_tags_str = ', '.join([f'{tag}: {score:.2f}' for tag, score in top_tags])
                        log_universal('INFO', 'Audio', f'Combined top MusiCNN tags: {top_tags_str}')
                    
                    log_universal('INFO', 'Audio', f'MusiCNN features combined from {len(all_musicnn_features)} samples')
                else:
                    log_universal('WARNING', 'Audio', 'MusiCNN features extraction returned None for all samples')
            else:
                log_universal('INFO', 'Audio', f'MusiCNN extraction disabled: extract_musicnn={self.extract_musicnn}, TENSORFLOW_AVAILABLE={TENSORFLOW_AVAILABLE}')
            
            # Extract chroma features (for genre classification)
            if self.extract_chroma:
                chroma_features = self._extract_chroma_features(audio_sample, sample_rate)
                features.update(chroma_features)
            
            # Add categorization-specific flags
            features['is_long_audio'] = True
            features['analysis_type'] = 'categorization_optimized'
            features['sample_duration_seconds'] = sample_duration
            
            # Add BPM from metadata if available
            if metadata and 'bpm_from_metadata' in metadata:
                features['external_bpm'] = metadata['bpm_from_metadata']
            
            log_universal('INFO', 'Audio', f'Optimized categorization analysis completed for long audio track')
            return features
            
        except Exception as e:
            log_universal('ERROR', 'Audio', f'Optimized categorization analysis failed: {e}')
            return None

    def _extract_simplified_features(self, audio: np.ndarray, sample_rate: int,
                                   metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Extract simplified features for long audio tracks.
        
        Args:
            audio: Audio data as numpy array
            sample_rate: Sample rate in Hz
            metadata: Optional metadata dictionary
            
        Returns:
            Simplified features dictionary
        """
        features = {}
        
        try:
            # Basic rhythm features (essential for categorization)
            if self.extract_rhythm:
                rhythm_features = self._extract_rhythm_features(audio, sample_rate)
                features.update(rhythm_features)
            
            # Basic loudness features (essential for categorization)
            if self.extract_loudness:
                loudness_features = self._extract_loudness_features(audio, sample_rate)
                features.update(loudness_features)
            
            # Basic spectral features (essential for categorization)
            if self.extract_spectral:
                spectral_features = self._extract_spectral_features(audio, sample_rate)
                features.update(spectral_features)
            
            # Skip memory-intensive features for long audio tracks
            log_universal('INFO', 'Audio', 'Skipping detailed analysis (MFCC, MusiCNN, chroma) for long audio track')
            
            # Add BPM from metadata if available
            if metadata and 'bpm_from_metadata' in metadata:
                features['external_bpm'] = metadata['bpm_from_metadata']
            
            return features
            
        except Exception as e:
            log_universal('ERROR', 'Audio', f'Simplified feature extraction failed: {e}')
            return {}

    def _create_basic_analysis_for_large_file(self, file_path: str, file_size: int, file_hash: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Create basic analysis result for large files with lightweight feature extraction.
        
        Args:
            file_path: Path to the audio file
            file_size: File size in bytes
            file_hash: File hash
            metadata: Optional metadata dictionary
            
        Returns:
            Basic analysis result dictionary
        """
        try:
            # Determine if it's a long audio track based on file size
            is_long_audio = self._is_long_audio_track(file_path)
            
            # Extract lightweight features for categorization
            audio_features = self._extract_lightweight_features_for_large_file(file_path, metadata)
            
            # Determine long audio category if it's a long audio track
            long_audio_category = None
            if is_long_audio:
                log_universal('INFO', 'Audio', f'Large long audio track detected: {os.path.basename(file_path)}')
                long_audio_category = self._determine_long_audio_category(file_path, metadata, audio_features)
                log_universal('INFO', 'Audio', f'Long audio category determined: {long_audio_category}')
            
            # Create basic analysis result
            analysis_result = {
                'file_path': file_path,
                'filename': os.path.basename(file_path),
                'file_size_bytes': file_size,
                'file_hash': file_hash,
                'analysis_date': datetime.now().isoformat(),
                'analysis_version': '1.0.0',
                'audio_type': 'large_file',
                'is_long_audio': is_long_audio,
                'is_extremely_large': True,
                'long_audio_category': long_audio_category,
                'metadata': metadata or {},
                'status': 'analyzed_large_file',
                'warning': 'File too large for full analysis - lightweight features extracted'
            }
            
            # Add basic features that can be estimated from file size
            estimated_duration_seconds = (file_size * 8) / (320 * 1000)  # Assuming 320kbps
            analysis_result.update({
                'estimated_duration_seconds': estimated_duration_seconds,
                'estimated_duration_minutes': estimated_duration_seconds / 60,
                'file_size_mb': file_size / (1024 * 1024)
            })
            
            # Add lightweight audio features if available
            if audio_features:
                analysis_result['audio_features'] = audio_features
                log_universal('INFO', 'Audio', f'Extracted lightweight features for large file: {list(audio_features.keys())}')
            
            # Update metadata with long audio category
            if metadata and long_audio_category:
                metadata['long_audio_category'] = long_audio_category
                analysis_result['metadata']['long_audio_category'] = long_audio_category
                log_universal('DEBUG', 'Audio', f'Set long_audio_category in metadata: {long_audio_category}')
            
            log_universal('INFO', 'Audio', f'Created analysis for large file: {os.path.basename(file_path)}')
            return analysis_result
            
        except Exception as e:
            log_universal('ERROR', 'Audio', f'Error creating analysis for large file {os.path.basename(file_path)}: {e}')
            return None

    def _extract_lightweight_features_for_large_file(self, file_path: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Extract lightweight audio features for large files using streaming approach.
        Only extracts features needed for long audio categorization.
        
        Args:
            file_path: Path to the audio file
            metadata: Optional metadata dictionary for fallback BPM
            
        Returns:
            Dictionary of lightweight audio features
        """
        try:
            log_universal('INFO', 'Audio', f'Extracting lightweight features for large file: {os.path.basename(file_path)}')
            
            features = {}
            
            # Try to load a small sample for analysis (first 15 seconds)
            sample_duration = 15  # seconds
            sample_size = sample_duration * self.sample_rate
            
            try:
                # Load only a sample of the audio for lightweight analysis
                audio_sample, sample_rate = self._load_audio_sample(file_path, sample_duration)
                
                if audio_sample is not None and len(audio_sample) > 0:
                    log_universal('INFO', 'Audio', f'Successfully loaded {sample_duration}s sample for lightweight analysis')
                    
                    # Extract essential features for categorization
                    if self.extract_rhythm:
                        rhythm_features = self._extract_rhythm_features(audio_sample, sample_rate)
                        features.update(rhythm_features)
                        log_universal('DEBUG', 'Audio', f'Extracted rhythm features: {list(rhythm_features.keys())}')
                    
                    if self.extract_spectral:
                        spectral_features = self._extract_spectral_features(audio_sample, sample_rate)
                        features.update(spectral_features)
                        log_universal('DEBUG', 'Audio', f'Extracted spectral features: {list(spectral_features.keys())}')
                    
                    if self.extract_loudness:
                        loudness_features = self._extract_loudness_features(audio_sample, sample_rate)
                        features.update(loudness_features)
                        log_universal('DEBUG', 'Audio', f'Extracted loudness features: {list(loudness_features.keys())}')
                    
                    log_universal('INFO', 'Audio', f'Successfully extracted {len(features)} lightweight features')
                else:
                    log_universal('WARNING', 'Audio', f'Failed to load audio sample for lightweight analysis')
                    
            except Exception as e:
                log_universal('WARNING', 'Audio', f'Error extracting lightweight features: {e}')
            
            # Add BPM from metadata if available (fallback when audio extraction fails)
            if metadata and 'bpm_from_metadata' in metadata:
                features['external_bpm'] = metadata['bpm_from_metadata']
                log_universal('DEBUG', 'Audio', f'Added BPM from metadata: {metadata["bpm_from_metadata"]}')
            
            return features
            
        except Exception as e:
            log_universal('ERROR', 'Audio', f'Error in lightweight feature extraction: {e}')
            return {}

    def _load_audio_sample(self, file_path: str, duration_seconds: int) -> Tuple[Optional[np.ndarray], Optional[int]]:
        """
        Load a sample of audio from the beginning of the file.
        
        Args:
            file_path: Path to the audio file
            duration_seconds: Duration of sample to load in seconds
            
        Returns:
            Tuple of (audio_sample, sample_rate) or (None, None) on failure
        """
        try:
            # Calculate sample size
            sample_size = duration_seconds * self.sample_rate
            
            # Try using Essentia with sample loading
            try:
                import essentia.standard as es
                log_universal('DEBUG', 'Audio', f'Loading {duration_seconds}s sample with Essentia')
                
                # Use MonoLoader with sample size limit
                loader = es.MonoLoader(filename=file_path, sampleRate=self.sample_rate)
                audio = loader()
                
                # Take only the first N samples
                if audio is not None and len(audio) > 0:
                    sample_end = min(sample_size, len(audio))
                    audio_sample = audio[:sample_end]
                    
                    log_universal('DEBUG', 'Audio', f'Loaded sample: {len(audio_sample)} samples ({len(audio_sample)/self.sample_rate:.1f}s)')
                    return audio_sample, self.sample_rate
                    
            except Exception as e:
                log_universal('WARNING', 'Audio', f'Essentia sample loading failed: {e}')
            
            # Fallback to librosa if available
            if LIBROSA_AVAILABLE:
                try:
                    import librosa
                    log_universal('DEBUG', 'Audio', f'Loading {duration_seconds}s sample with librosa')
                    
                    # Load with duration parameter
                    audio_sample, sr = librosa.load(file_path, sr=self.sample_rate, duration=duration_seconds, mono=True)
                    
                    if audio_sample is not None and len(audio_sample) > 0:
                        log_universal('DEBUG', 'Audio', f'Loaded sample with librosa: {len(audio_sample)} samples ({len(audio_sample)/sr:.1f}s)')
                        return audio_sample, sr
                        
                except Exception as e:
                    log_universal('WARNING', 'Audio', f'Librosa sample loading failed: {e}')
            
            log_universal('WARNING', 'Audio', f'All sample loading methods failed for {os.path.basename(file_path)}')
            return None, None
            
        except Exception as e:
            log_universal('ERROR', 'Audio', f'Error in audio sample loading: {e}')
            return None, None

    def _compute_mel_spectrogram(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Compute mel-spectrogram for MusiCNN input.
        
        Args:
            audio: Audio signal
            sample_rate: Sample rate of the audio
            
        Returns:
            Mel-spectrogram as numpy array with shape [time, 96] for MusiCNN input
        """
        try:
            if LIBROSA_AVAILABLE:
                import librosa
                
                # MusiCNN specifications:
                # - Window size: 512 with 50% hop (hop_length = 256)
                # - Mel bands: 96 bands from 0 to 11025 Hz
                # - Sampling rate: 22050 Hz
                # - Fixed time dimension: 187 frames (3 seconds at 22050 Hz)
                n_fft = 512
                hop_length = 256  # 50% of window size
                n_mels = 96
                fmin = 0
                fmax = 11025  # Half of 22050 Hz (Nyquist frequency)
                target_frames = 187  # Fixed time dimension for MusiCNN
                
                # Compute mel-spectrogram
                mel_spec = librosa.feature.melspectrogram(
                    y=audio,
                    sr=sample_rate,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    n_mels=n_mels,
                    fmin=fmin,
                    fmax=fmax
                )
                
                # Convert to log power scale (required for MusiCNN)
                mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                
                # Transpose to get [time, 96] shape
                mel_spec_transposed = mel_spec_db.T
                
                # Pad or truncate to exactly 187 frames
                if mel_spec_transposed.shape[0] > target_frames:
                    # Truncate to 187 frames
                    mel_spec_final = mel_spec_transposed[:target_frames, :]
                elif mel_spec_transposed.shape[0] < target_frames:
                    # Pad with zeros to 187 frames
                    padding = np.zeros((target_frames - mel_spec_transposed.shape[0], n_mels))
                    mel_spec_final = np.vstack([mel_spec_transposed, padding])
                else:
                    mel_spec_final = mel_spec_transposed
                
                log_universal('DEBUG', 'Audio', f'Mel-spectrogram shape: {mel_spec_final.shape}')
                return mel_spec_final
            else:
                log_universal('WARNING', 'Audio', 'Librosa not available for mel-spectrogram computation')
                return np.zeros((187, 96))  # Fixed size for MusiCNN
                
        except Exception as e:
            log_universal('WARNING', 'Audio', f'Error computing mel-spectrogram: {e}')
            return np.zeros((187, 96))  # Fixed size for MusiCNN

    def _extract_spotify_style_features(self, audio: np.ndarray, sample_rate: int, 
                                      rhythm_features: Dict[str, Any], 
                                      spectral_features: Dict[str, Any],
                                      loudness_features: Dict[str, Any],
                                      key_features: Dict[str, Any],
                                      mfcc_features: Dict[str, Any],
                                      musicnn_features: Dict[str, Any],
                                      chroma_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract Spotify-style features for playlist generation.
        These are derived from existing features and MusiCNN tags.
        """
        try:
            # Extract base features
            bpm = rhythm_features.get('bpm', 0)
            rhythm_confidence = rhythm_features.get('rhythm_confidence', 0)
            spectral_centroid = spectral_features.get('spectral_centroid', 0)
            spectral_flatness = spectral_features.get('spectral_flatness', 0)
            loudness = loudness_features.get('loudness', 0)
            dynamic_complexity = loudness_features.get('dynamic_complexity', 0)
            key = key_features.get('key', '')
            scale = key_features.get('scale', '')
            key_strength = key_features.get('key_strength', 0)
            
            # Extract MusiCNN tags
            tags = musicnn_features.get('tags', {})
            if isinstance(tags, str):
                try:
                    tags = json.loads(tags)
                except:
                    tags = {}
            
            # Extract MFCC and chroma features
            mfcc_coefficients = mfcc_features.get('mfcc_coefficients', [])
            chroma_mean = chroma_features.get('chroma_mean', [])
            
            # Calculate Spotify-style features
            
            # 1. DANCEABILITY (0.0 to 1.0)
            # Based on: BPM, rhythm confidence, spectral characteristics, MusiCNN dance tags
            danceability = 0.0
            
            # BPM contribution (optimal range: 120-140 BPM)
            if bpm > 0:
                if 120 <= bpm <= 140:
                    danceability += 0.4  # Optimal dance range
                elif 100 <= bpm <= 160:
                    danceability += 0.3  # Good dance range
                elif 80 <= bpm <= 180:
                    danceability += 0.2  # Acceptable range
                else:
                    danceability += 0.1  # Outside dance range
            
            # Rhythm confidence contribution
            if rhythm_confidence > 0.7:
                danceability += 0.3
            elif rhythm_confidence > 0.5:
                danceability += 0.2
            elif rhythm_confidence > 0.3:
                danceability += 0.1
            
            # Spectral characteristics (tonal content is more danceable)
            if spectral_flatness < 0.3:
                danceability += 0.2
            elif spectral_flatness < 0.5:
                danceability += 0.1
            
            # MusiCNN dance tags
            if isinstance(tags, dict):
                dance_tags = ['dance', 'pop', 'funk', 'catchy', 'party', 'electronic', 'house']
                for tag, score in tags.items():
                    if tag.lower() in dance_tags:
                        danceability += min(score * 0.3, 0.3)  # Cap at 0.3
            
            # Cap danceability at 1.0
            danceability = min(danceability, 1.0)
            
            # 2. ENERGY (0.0 to 1.0)
            # Based on: loudness, dynamic complexity, spectral centroid, BPM
            energy = 0.0
            
            # Loudness contribution
            if loudness > 0.6:
                energy += 0.4
            elif loudness > 0.4:
                energy += 0.3
            elif loudness > 0.2:
                energy += 0.2
            else:
                energy += 0.1
            
            # Dynamic complexity (higher = more energy)
            if dynamic_complexity > 0.7:
                energy += 0.3
            elif dynamic_complexity > 0.5:
                energy += 0.2
            elif dynamic_complexity > 0.3:
                energy += 0.1
            
            # Spectral centroid (higher = more energy)
            if spectral_centroid > 3000:
                energy += 0.2
            elif spectral_centroid > 2000:
                energy += 0.1
            
            # BPM contribution
            if bpm > 140:
                energy += 0.2
            elif bpm > 120:
                energy += 0.1
            
            # Cap energy at 1.0
            energy = min(energy, 1.0)
            
            # 3. MODE (0 = minor, 1 = major)
            # Based on: key scale
            mode = 0.0
            if scale.lower() == 'major':
                mode = 1.0
            elif scale.lower() == 'minor':
                mode = 0.0
            else:
                # Default to minor for unknown scales
                mode = 0.0
            
            # 4. ACOUSTICNESS (0.0 to 1.0)
            # Based on: MusiCNN acoustic tags, spectral characteristics
            acousticness = 0.0
            
            # MusiCNN acoustic tags
            if isinstance(tags, dict):
                acoustic_tags = ['acoustic', 'folk', 'country', 'jazz', 'blues', 'mellow', 'easy listening']
                for tag, score in tags.items():
                    if tag.lower() in acoustic_tags:
                        acousticness += min(score * 0.4, 0.4)  # Cap at 0.4
            
            # Spectral characteristics (lower centroid = more acoustic)
            if spectral_centroid < 2000:
                acousticness += 0.3
            elif spectral_centroid < 3000:
                acousticness += 0.2
            elif spectral_centroid < 4000:
                acousticness += 0.1
            
            # Lower loudness = more acoustic
            if loudness < 0.3:
                acousticness += 0.2
            elif loudness < 0.5:
                acousticness += 0.1
            
            # Cap acousticness at 1.0
            acousticness = min(acousticness, 1.0)
            
            # 5. INSTRUMENTALNESS (0.0 to 1.0)
            # Based on: MusiCNN instrumental tags, speechiness inverse
            instrumentalness = 0.0
            
            # MusiCNN instrumental tags
            if isinstance(tags, dict):
                if 'instrumental' in tags:
                    instrumentalness += min(tags['instrumental'] * 0.5, 0.5)
                
                # Electronic music is often instrumental
                electronic_tags = ['electronic', 'electronica', 'electro', 'house', 'techno']
                for tag, score in tags.items():
                    if tag.lower() in electronic_tags:
                        instrumentalness += min(score * 0.3, 0.3)
            
            # Inverse relationship with speechiness
            speechiness = 0.0
            if isinstance(tags, dict):
                speech_tags = ['female vocalists', 'male vocalists', 'female vocalist', 'male vocalist']
                for tag, score in tags.items():
                    if tag.lower() in speech_tags:
                        speechiness += min(score * 0.4, 0.4)
            
            # If high speechiness, reduce instrumentalness
            if speechiness > 0.5:
                instrumentalness = max(0.0, instrumentalness - speechiness * 0.5)
            
            # Cap instrumentalness at 1.0
            instrumentalness = min(instrumentalness, 1.0)
            
            # 6. SPEECHINESS (0.0 to 1.0)
            # Already calculated above, but ensure it's capped
            speechiness = min(speechiness, 1.0)
            
            # 7. VALENCE (0.0 to 1.0) - Positivity/happiness
            # Based on: key (major = happier), BPM, MusiCNN mood tags
            valence = 0.5  # Default neutral
            
            # Major key = happier
            if scale.lower() == 'major':
                valence += 0.2
            elif scale.lower() == 'minor':
                valence -= 0.1
            
            # BPM contribution (faster = happier)
            if bpm > 120:
                valence += 0.1
            elif bpm < 80:
                valence -= 0.1
            
            # MusiCNN mood tags
            if isinstance(tags, dict):
                positive_tags = ['happy', 'upbeat', 'catchy', 'fun', 'energetic']
                negative_tags = ['sad', 'melancholy', 'dark', 'moody']
                
                for tag, score in tags.items():
                    if tag.lower() in positive_tags:
                        valence += min(score * 0.2, 0.2)
                    elif tag.lower() in negative_tags:
                        valence -= min(score * 0.2, 0.2)
            
            # Cap valence at 0.0 to 1.0
            valence = max(0.0, min(valence, 1.0))
            
            # 8. LIVENESS (0.0 to 1.0) - Presence of audience/performance
            # Based on: dynamic complexity, spectral characteristics
            liveness = 0.0
            
            # Higher dynamic complexity = more live
            if dynamic_complexity > 0.7:
                liveness += 0.4
            elif dynamic_complexity > 0.5:
                liveness += 0.3
            elif dynamic_complexity > 0.3:
                liveness += 0.2
            
            # Spectral characteristics (live music has more variation)
            if spectral_flatness > 0.4:
                liveness += 0.2
            elif spectral_flatness > 0.2:
                liveness += 0.1
            
            # MFCC variance (more variation = more live)
            if mfcc_coefficients:
                mfcc_variance = np.var(mfcc_coefficients)
                if mfcc_variance > 0.2:
                    liveness += 0.2
                elif mfcc_variance > 0.1:
                    liveness += 0.1
            
            # Cap liveness at 1.0
            liveness = min(liveness, 1.0)
            
            # 9. POPULARITY (0.0 to 1.0) - Placeholder for now
            # This would typically come from external APIs (Spotify, LastFM, etc.)
            popularity = 0.5  # Default neutral
            
            # Could be enhanced with:
            # - External API data
            # - Play count from user library
            # - Social media mentions
            # - Chart positions
            
            log_universal('DEBUG', 'Audio', f'Spotify-style features calculated: '
                          f'danceability={danceability:.2f}, energy={energy:.2f}, mode={mode:.2f}, '
                          f'acousticness={acousticness:.2f}, instrumentalness={instrumentalness:.2f}, '
                          f'speechiness={speechiness:.2f}, valence={valence:.2f}, '
                          f'liveness={liveness:.2f}, popularity={popularity:.2f}')
            
            return {
                'danceability': danceability,
                'energy': energy,
                'mode': mode,
                'acousticness': acousticness,
                'instrumentalness': instrumentalness,
                'speechiness': speechiness,
                'valence': valence,
                'liveness': liveness,
                'popularity': popularity
            }
            
        except Exception as e:
            log_universal('WARNING', 'Audio', f'Failed to extract Spotify-style features: {e}')
            # Return default values
            return {
                'danceability': 0.5,
                'energy': 0.5,
                'mode': 0.0,
                'acousticness': 0.5,
                'instrumentalness': 0.5,
                'speechiness': 0.5,
                'valence': 0.5,
                'liveness': 0.5,
                'popularity': 0.5
            }


def get_audio_analyzer(config: Dict[str, Any] = None) -> 'AudioAnalyzer':
    """Get a configured audio analyzer instance."""
    return AudioAnalyzer(config)
