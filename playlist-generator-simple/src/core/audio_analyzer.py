"""
Audio Analyzer for Playlist Generator Simple.
Extracts audio features including MusicNN embeddings and auto-tags.
"""

import os
# Configure TensorFlow logging BEFORE any imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Hide INFO and WARNING, show only ERROR
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimization messages
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU to avoid GPU-related warnings
import time
import json
import hashlib
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime
from collections import Counter

# Suppress TensorFlow warnings
try:
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all TF warnings
    tf.autograph.set_verbosity(0)
    
    # Suppress all TensorFlow warnings
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
    warnings.filterwarnings('ignore', category=FutureWarning, module='tensorflow')
    
    # Disable TensorFlow GPU warnings
    os.environ['TF_GPU_ALLOCATOR'] = 'cpu'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
except ImportError:
    pass
# Import local modules
from .logging_setup import get_logger, log_function_call, log_universal
from .database import DatabaseManager, get_db_manager
from .config_loader import config_loader
from .lazy_imports import (
    get_tensorflow, get_essentia, 
    is_tensorflow_available, is_essentia_available, 
    is_librosa_available, is_mutagen_available,
    librosa, mutagen
)

logger = get_logger('playlista.audio_analyzer')

# Check for required libraries using lazy imports
ESSENTIA_AVAILABLE = is_essentia_available()
if not ESSENTIA_AVAILABLE:
    log_universal('WARNING', 'Audio', 'Essentia not available - using librosa fallback')

LIBROSA_AVAILABLE = is_librosa_available()
if not LIBROSA_AVAILABLE:
    log_universal('WARNING', 'Audio', 'Librosa not available')

TENSORFLOW_AVAILABLE = is_tensorflow_available()
if not TENSORFLOW_AVAILABLE:
    log_universal('WARNING', 'Audio', 'TensorFlow not available - MusiCNN features disabled')

MUTAGEN_AVAILABLE = is_mutagen_available()
if not MUTAGEN_AVAILABLE:
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
        
        # Check available memory before loading - CONSERVATIVE THRESHOLDS
        try:
            import psutil
            available_memory_mb = psutil.virtual_memory().available / (1024 * 1024)
            # Increased minimum memory requirement from 1.5GB to 2.5GB for safety
            min_memory_mb = config.get('MIN_MEMORY_FOR_FULL_ANALYSIS_GB', 2.5) * 1024 if config else 2560  # Convert GB to MB
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
            
            # Configurable limits based on processing mode - REALISTIC THRESHOLDS
            if processing_mode == 'sequential':
                max_file_size_mb = config.get('SEQUENTIAL_MAX_FILE_SIZE_MB', 500) if config else 500  # Realistic for large lossless files
                warning_threshold_mb = config.get('LARGE_FILE_WARNING_THRESHOLD_MB', 100) if config else 100  # Realistic for high-quality tracks
            else:  # parallel
                max_file_size_mb = config.get('PARALLEL_MAX_FILE_SIZE_MB', 200) if config else 200  # Realistic for parallel processing
                warning_threshold_mb = config.get('PARALLEL_LARGE_FILE_WARNING_THRESHOLD_MB', 50) if config else 50  # Realistic for parallel processing
            
            if file_size_mb > max_file_size_mb:
                log_universal('WARNING', 'Audio', f'File too large ({file_size_mb:.1f}MB > {max_file_size_mb}MB): {os.path.basename(audio_path)}')
                return None, None
            
            if file_size_mb > warning_threshold_mb:
                log_universal('INFO', 'Audio', f'Large file detected ({file_size_mb:.1f}MB): {os.path.basename(audio_path)}')
                
        except Exception as e:
            log_universal('WARNING', 'Audio', f'Could not check file size: {e}')
        
        # Load audio using Essentia if available
        if ESSENTIA_AVAILABLE:
            try:
                import essentia.standard as es
                
                # Use Essentia loader with optimized settings
                loader = es.MonoLoader(filename=audio_path, sampleRate=sample_rate)
                audio = loader()
                
                log_universal('DEBUG', 'Audio', f'Loaded with Essentia: {os.path.basename(audio_path)} ({len(audio)} samples, {sample_rate}Hz')
                return audio, sample_rate
                
            except Exception as e:
                log_universal('WARNING', 'Audio', f'Essentia loading failed for {os.path.basename(audio_path)}: {e}')
                # Fall through to librosa
        
        # Fallback to librosa
        if LIBROSA_AVAILABLE:
            try:
                # Use librosa with optimized settings
                audio, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
                
                log_universal('DEBUG', 'Audio', f'Loaded with librosa: {os.path.basename(audio_path)} ({len(audio)} samples, {sr}Hz')
                return audio, sr
                
            except Exception as e:
                log_universal('ERROR', 'Audio', f'Librosa loading failed for {os.path.basename(audio_path)}: {e}')
                return None, None
        
        log_universal('ERROR', 'Audio', f'No audio loading library available for {os.path.basename(audio_path)}')
        return None, None
        
    except Exception as e:
        log_universal('ERROR', 'Audio', f'Unexpected error loading {os.path.basename(audio_path)}: {e}')
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
            try:
                config = config_loader.get_audio_analysis_config()
            except NameError:
                # Handle case where config_loader is not available (e.g., in multiprocessing)
                from .config_loader import ConfigLoader
                local_config_loader = ConfigLoader()
                config = local_config_loader.get_audio_analysis_config()
        
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
        self.extract_danceability = config.get('EXTRACT_DANCEABILITY', True)
        self.extract_onset_rate = config.get('EXTRACT_ONSET_RATE', True)
        self.extract_zcr = config.get('EXTRACT_ZCR', True)
        self.extract_spectral_contrast = config.get('EXTRACT_SPECTRAL_CONTRAST', True)
        
        # Caching settings
        self.cache_enabled = config.get('CACHE_ENABLED', True)
        self.cache_expiry_hours = config.get('CACHE_EXPIRY_HOURS', 168)  # 1 week
        self.force_reanalysis = config.get('FORCE_REANALYSIS', False)
        
        log_universal('INFO', 'Audio', 'AudioAnalyzer initialized')
        log_universal('DEBUG', 'Audio', f'Feature extraction flags: rhythm={self.extract_rhythm}, spectral={self.extract_spectral}, loudness={self.extract_loudness}, key={self.extract_key}, mfcc={self.extract_mfcc}, musicnn={self.extract_musicnn}, chroma={self.extract_chroma}, danceability={self.extract_danceability}, onset_rate={self.extract_onset_rate}, zcr={self.extract_zcr}, spectral_contrast={self.extract_spectral_contrast}')
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
        Analyze an audio file using simplified 5-step process.
        
        Steps:
        1. Extract artist/track using mutagen with tag mapping (save to DB)
        2. Fallback: filename pattern extraction if mutagen fails
        3. Enrich metadata with external APIs (MusicBrainz, Last.fm, etc.)
        4. Extract values using Essentia
        5. Extract values using MusicNN
        6. Ensure everything written to DB
        
        Args:
            file_path: Path to the audio file
            force_reanalysis: If True, bypass cache and re-analyze
            
        Returns:
            Analysis result dictionary or None on failure
        """
        if not os.path.exists(file_path):
            log_universal('ERROR', 'Audio', f'File not found: {file_path}')
            return None
        
        try:
            # Calculate file hash for caching
            file_hash = self._calculate_file_hash(file_path)
            
            # Check cache unless force reanalysis
            if force_reanalysis is None:
                force_reanalysis = self.config.get('FORCE_REANALYSIS', False)
            
            if not force_reanalysis:
                cached_result = self._get_cached_analysis(file_path, file_hash)
                if cached_result:
                    log_universal('INFO', 'Audio', f'Using cached analysis for {os.path.basename(file_path)}')
                    return cached_result
            
            # STEP 1: Extract artist/track using mutagen with tag mapping
            log_universal('INFO', 'Audio', f'Step 1: Extracting metadata with mutagen for {os.path.basename(file_path)}')
            metadata = self._extract_metadata_with_mutagen(file_path)
            
            # Save metadata to database
            if metadata:
                self.db_manager.save_metadata(file_path, metadata)
                log_universal('INFO', 'Audio', f'Metadata saved to database: {metadata.get("artist", "Unknown")} - {metadata.get("title", "Unknown")}')
            
            # STEP 2: Fallback - filename pattern extraction if mutagen fails
            if not metadata or not metadata.get('artist') or not metadata.get('title'):
                log_universal('INFO', 'Audio', f'Step 2: Fallback to filename pattern extraction for {os.path.basename(file_path)}')
                metadata = self._extract_from_filename_pattern(file_path)
                if metadata:
                    self.db_manager.save_metadata(file_path, metadata)
                    log_universal('INFO', 'Audio', f'Filename pattern metadata saved: {metadata.get("artist", "Unknown")} - {metadata.get("title", "Unknown")}')
            
            # STEP 3: Enrich metadata with external APIs
            log_universal('INFO', 'Audio', f'Step 3: Enriching metadata with external APIs for {os.path.basename(file_path)}')
            enriched_metadata = self._enrich_metadata_with_external_apis(metadata)
            if enriched_metadata and enriched_metadata != metadata:
                self.db_manager.save_metadata(file_path, enriched_metadata)
                log_universal('INFO', 'Audio', f'Enriched metadata saved: {enriched_metadata.get("artist", "Unknown")} - {enriched_metadata.get("title", "Unknown")}')
                metadata = enriched_metadata
            
            # Load audio data
            log_universal('INFO', 'Audio', f'Loading audio data for {os.path.basename(file_path)}')
            audio, sample_rate = safe_essentia_load(file_path, self.sample_rate, self.config, self.processing_mode)
            
            if audio is None:
                log_universal('ERROR', 'Audio', f'Failed to load audio: {os.path.basename(file_path)}')
                return None
            
            # STEP 4: Extract values using Essentia
            log_universal('INFO', 'Audio', f'Step 4: Extracting Essentia features for {os.path.basename(file_path)}')
            essentia_features = self._extract_essentia_features(audio, sample_rate)
            if essentia_features:
                self.db_manager.save_essentia_features(file_path, essentia_features)
                log_universal('INFO', 'Audio', f'Essentia features saved to database')
            
            # STEP 5: Extract values using MusicNN
            log_universal('INFO', 'Audio', f'Step 5: Extracting MusicNN features for {os.path.basename(file_path)}')
            musicnn_features = self._extract_musicnn_features(audio, sample_rate)
            if musicnn_features:
                self.db_manager.save_musicnn_features(file_path, musicnn_features)
                log_universal('INFO', 'Audio', f'MusicNN features saved to database')
            
            # STEP 6: Ensure everything written to DB
            log_universal('INFO', 'Audio', f'Step 6: Committing all analysis results to database for {os.path.basename(file_path)}')
            all_features = {
                'essentia': essentia_features,
                'musicnn': musicnn_features,
                'metadata': metadata
            }
            self.db_manager.commit_analysis_results(file_path, all_features)
            
            # Prepare result
            result = {
                'success': True,
                'features': all_features,
                'metadata': metadata,
                'file_path': file_path,
                'sample_rate': sample_rate,
                'audio_length': len(audio),
                'analysis_mode': 'simplified_6_step'
            }
            
            # Cache the result
            self._cache_analysis_result(file_path, file_hash, result)
            
            log_universal('INFO', 'Audio', f'Analysis completed: {os.path.basename(file_path)} ({len(audio)} samples, {sample_rate}Hz)')
            return result
            
        except Exception as e:
            log_universal('ERROR', 'Audio', f'Analysis failed for {os.path.basename(file_path)}: {e}')
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
    
    def _extract_metadata_with_mutagen(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Extract metadata using mutagen with tag mapping.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Metadata dictionary or None on failure
        """
        try:
            from mutagen import File
            audio_file = File(file_path)
            if audio_file is None:
                log_universal('WARNING', 'Audio', f'Could not open file for metadata: {os.path.basename(file_path)}')
                return None
            
            metadata = {}
            
            # Extract tags using enhanced tag mapper
            if hasattr(audio_file, 'tags') and audio_file.tags:
                try:
                    from .tag_mapping import get_tag_mapper
                    tag_mapper = get_tag_mapper()
                    
                    # Convert mutagen tags to dictionary format
                    tags_dict = {}
                    for key, value in audio_file.tags.items():
                        # Limit tag value size to prevent memory issues
                        if isinstance(value, str) and len(value) > 1000:
                            value = value[:1000] + "..."
                        elif isinstance(value, list):
                            # Limit list size and individual item size
                            value = [str(item)[:500] for item in value[:10]]
                        
                        tags_dict[key] = value
                    
                    # Map tags using the enhanced mapper
                    mapped_metadata = tag_mapper.map_tags(tags_dict)
                    metadata.update(mapped_metadata)
                    
                except Exception as e:
                    log_universal('WARNING', 'Audio', f'Tag mapping failed: {e}')
                    # Fallback to basic extraction
                    metadata = self._extract_basic_metadata(audio_file)
            
            # Extract basic metadata if no tags
            if not metadata:
                metadata = self._extract_basic_metadata(audio_file)
            
            return metadata
            
        except Exception as e:
            log_universal('WARNING', 'Audio', f'Mutagen metadata extraction failed: {e}')
            return None

    def _extract_from_filename_pattern(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Extract artist and title from filename patterns.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Metadata dictionary with artist and title
        """
        try:
            filename = os.path.basename(file_path)
            name_without_ext = os.path.splitext(filename)[0]
            
            metadata = {}
            
            # Method 1: Try "Artist - Title" format (most common)
            if ' - ' in name_without_ext:
                parts = name_without_ext.split(' - ', 1)
                if len(parts) == 2:
                    metadata['artist'] = parts[0].strip()
                    metadata['title'] = parts[1].strip()
                    log_universal('INFO', 'Audio', f'Extracted from filename pattern: {metadata["artist"]} - {metadata["title"]}')
                    return metadata
            
            # Method 2: Try "Artist_-_Title" format (underscore separator)
            if '_-_' in name_without_ext:
                parts = name_without_ext.split('_-_', 1)
                if len(parts) == 2:
                    metadata['artist'] = parts[0].strip()
                    metadata['title'] = parts[1].strip()
                    log_universal('INFO', 'Audio', f'Extracted from filename pattern: {metadata["artist"]} - {metadata["title"]}')
                    return metadata
            
            # Method 3: Try "Artist__Title" format (double underscore)
            if '__' in name_without_ext:
                parts = name_without_ext.split('__', 1)
                if len(parts) == 2:
                    metadata['artist'] = parts[0].strip()
                    metadata['title'] = parts[1].strip()
                    log_universal('INFO', 'Audio', f'Extracted from filename pattern: {metadata["artist"]} - {metadata["title"]}')
                    return metadata
            
            # Method 4: Try directory name as artist if no artist found
            dir_name = os.path.basename(os.path.dirname(file_path))
            if dir_name and dir_name not in ['', '.', '..']:
                metadata['artist'] = dir_name
                metadata['title'] = name_without_ext
                log_universal('INFO', 'Audio', f'Using directory name as artist: {dir_name}')
                return metadata
            
            # Method 5: Use filename as title if no pattern found
            metadata['artist'] = 'Unknown Artist'
            metadata['title'] = name_without_ext
            log_universal('INFO', 'Audio', f'Using filename as title: {name_without_ext}')
            return metadata
            
        except Exception as e:
            log_universal('WARNING', 'Audio', f'Filename pattern extraction failed: {e}')
            return None

    def _extract_essentia_features(self, audio: np.ndarray, sample_rate: int) -> Optional[Dict[str, Any]]:
        """
        Extract features using Essentia.
        
        Args:
            audio: Audio data as numpy array
            sample_rate: Sample rate in Hz
            
        Returns:
            Essentia features dictionary or None on failure
        """
        try:
            features = {}
            
            # Extract rhythm features
            if self.extract_rhythm:
                rhythm_features = self._extract_rhythm_features(audio, sample_rate)
                features.update(rhythm_features)
            
            # Extract spectral features
            if self.extract_spectral:
                spectral_features = self._extract_spectral_features(audio, sample_rate)
                features.update(spectral_features)
            
            # Extract loudness features
            if self.extract_loudness:
                loudness_features = self._extract_loudness_features(audio, sample_rate)
                features.update(loudness_features)
            
            # Extract key and mode
            if self.extract_key:
                key_features = self._extract_key_features(audio, sample_rate)
                features.update(key_features)
            
            # Extract MFCC features
            if self.extract_mfcc:
                mfcc_features = self._extract_mfcc_features(audio, sample_rate)
                features.update(mfcc_features)
            
            # Extract chroma features
            if hasattr(self, 'extract_chroma'):
                chroma_features = self._extract_chroma_features(audio, sample_rate)
                features.update(chroma_features)
            
            return features
            
        except Exception as e:
            log_universal('WARNING', 'Audio', f'Essentia feature extraction failed: {e}')
            return None

    def _extract_basic_metadata(self, audio_file) -> Dict[str, Any]:
        """
        Extract basic metadata from audio file.
        
        Args:
            audio_file: Mutagen audio file object
            
        Returns:
            Basic metadata dictionary
        """
        metadata = {}
        
        try:
            # Extract basic info
            if hasattr(audio_file, 'info'):
                info = audio_file.info
                if hasattr(info, 'length'):
                    metadata['duration'] = info.length
                if hasattr(info, 'bitrate'):
                    metadata['bitrate'] = info.bitrate
                if hasattr(info, 'sample_rate'):
                    metadata['sample_rate'] = info.sample_rate
            
            # Extract tags if available
            if hasattr(audio_file, 'tags') and audio_file.tags:
                for tag_type in audio_file.tags:
                    if tag_type in audio_file.tags:
                        tags = audio_file.tags[tag_type]
                        for key, value in tags.items():
                            if isinstance(value, list):
                                value = value[0] if value else ''
                            metadata[key] = str(value)
            
        except Exception as e:
            log_universal('WARNING', 'Audio', f'Basic metadata extraction failed: {e}')
        
        return metadata
    
    def _extract_rhythm_features(self, audio: np.ndarray, sample_rate: int, file_size_mb: float = None) -> Dict[str, Any]:
        """Extract rhythm-related features using lightweight approach for large files."""
        features = {}
        
        log_universal('DEBUG', 'Audio', f'Starting rhythm extraction: audio_length={len(audio)}, sample_rate={sample_rate}')
        
        try:
            if ESSENTIA_AVAILABLE:
                import essentia.standard as es
                # Use 3-tier system for rhythm analysis: align with file size thresholds
                # Files < 100MB: Full processing, Files 100-200MB: Half-track, Files > 200MB: Half-track
                half_track_threshold_mb = self.config.get('HALF_TRACK_THRESHOLD_MB', 100)
                
                if file_size_mb is not None:
                    # Use actual file size for consistent decisions
                    if file_size_mb > half_track_threshold_mb:
                        # Use half-track loading for large files (middle 50%)
                        start_sample = len(audio) // 4
                        end_sample = 3 * len(audio) // 4
                        audio_sample = audio[start_sample:end_sample]
                        log_universal('INFO', 'Audio', f'Large file detected ({file_size_mb:.1f}MB) - using half-track loading for rhythm analysis')
                    else:
                        # Use full audio for small files
                        audio_sample = audio
                        log_universal('DEBUG', 'Audio', f'Small file ({file_size_mb:.1f}MB) - using full audio for rhythm analysis')
                else:
                    # Fallback to estimated size if actual size not provided
                    estimated_size_mb = (len(audio) * 4) / (1024 * 1024)  # Rough estimate
                    if estimated_size_mb > half_track_threshold_mb:
                        # Use half-track loading for large files (middle 50%)
                        start_sample = len(audio) // 4
                        end_sample = 3 * len(audio) // 4
                        audio_sample = audio[start_sample:end_sample]
                        log_universal('INFO', 'Audio', f'Large file detected ({estimated_size_mb:.1f}MB estimated) - using half-track loading for rhythm analysis')
                    else:
                        # Use full audio for small files
                        audio_sample = audio
                        log_universal('DEBUG', 'Audio', f'Small file ({estimated_size_mb:.1f}MB estimated) - using full audio for rhythm analysis')
                
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
                import essentia.standard as es
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
                import essentia.standard as es
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
                import essentia.standard as es
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
    
    def _extract_mfcc_features(self, audio: np.ndarray, sample_rate: int, file_size_mb: float = None) -> Dict[str, Any]:
        """Extract MFCC features."""
        features = {}
        
        try:
            if ESSENTIA_AVAILABLE:
                import essentia.standard as es
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
                
                # Use 3-tier system for MFCC extraction instead of hardcoded 60s limit
                # Files < 100MB: Full processing, Files 100-200MB: Half-track, Files > 200MB: Half-track
                half_track_threshold_mb = self.config.get('HALF_TRACK_THRESHOLD_MB', 100)
                
                if file_size_mb is not None:
                    # Use actual file size for consistent decisions
                    if file_size_mb > half_track_threshold_mb:
                        # Use half-track loading for large files (middle 50%)
                        start_sample = len(audio) // 4
                        end_sample = 3 * len(audio) // 4
                        audio = audio[start_sample:end_sample]
                        log_universal('INFO', 'Audio', f'Large file detected ({file_size_mb:.1f}MB) - using half-track loading for MFCC extraction')
                    else:
                        # Use full audio for small files
                        log_universal('DEBUG', 'Audio', f'Small file ({file_size_mb:.1f}MB) - using full audio for MFCC extraction')
                else:
                    # Fallback to estimated size if actual size not provided
                    estimated_size_mb = (len(audio) * 4) / (1024 * 1024)  # Rough estimate
                    if estimated_size_mb > half_track_threshold_mb:
                        # Use half-track loading for large files (middle 50%)
                        start_sample = len(audio) // 4
                        end_sample = 3 * len(audio) // 4
                        audio = audio[start_sample:end_sample]
                        log_universal('INFO', 'Audio', f'Large file detected ({estimated_size_mb:.1f}MB estimated) - using half-track loading for MFCC extraction')
                    else:
                        # Use full audio for small files
                        log_universal('DEBUG', 'Audio', f'Small file ({estimated_size_mb:.1f}MB estimated) - using full audio for MFCC extraction')
                
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
        """Extract MusiCNN features using shared model instances with enhanced error handling."""
        features = {}
        
        log_universal('INFO', 'Audio', f'MusiCNN extraction started: audio shape={audio.shape}, sample_rate={sample_rate}')
        
        try:
            if not TENSORFLOW_AVAILABLE:
                log_universal('WARNING', 'Audio', 'TensorFlow not available - skipping MusiCNN features')
                features.update({'embedding': [], 'tags': {}, 'musicnn_skipped': 1})
                return features
            
            # Get shared model instances from model manager
            from .model_manager import get_model_manager
            model_manager = get_model_manager(self.config)
            
            # Check if MusicNN is available
            if not model_manager.is_musicnn_available():
                log_universal('WARNING', 'Audio', 'MusicNN models not available - skipping MusiCNN features')
                features.update({'embedding': [], 'tags': {}, 'musicnn_skipped': 1})
                return features
            
            # Get shared model instances
            activations_model, embeddings_model, tag_names, metadata = model_manager.get_musicnn_models()
            
            if not activations_model or not embeddings_model or not tag_names:
                log_universal('WARNING', 'Audio', 'MusicNN models not properly loaded - skipping MusiCNN features')
                features.update({'embedding': [], 'tags': {}, 'musicnn_skipped': 1})
                return features
            
            log_universal('INFO', 'Audio', 'Using shared MusicNN models')
            
            # Check if we need to use half-track loading for large files
            audio_size_bytes = len(audio) * 4  # Estimate size (4 bytes per float32 sample)
            file_size_mb = audio_size_bytes / (1024 * 1024)
            half_track_threshold_mb = self.config.get('MUSICNN_HALF_TRACK_THRESHOLD_MB', 50)
            print(f"MUSICNN_DEBUG: threshold={half_track_threshold_mb}MB, file_size={file_size_mb:.1f}MB")  # Direct print for debugging
            log_universal('DEBUG', 'Audio', f'MusicNN half-track threshold: {half_track_threshold_mb}MB, file size: {file_size_mb:.1f}MB')
            
            # Use half-track for files larger than threshold, but only if MusicNN is suitable
            use_half_track = file_size_mb > half_track_threshold_mb and model_manager.is_file_suitable_for_musicnn(audio_size_bytes)
            print(f"MUSICNN_DEBUG: decision: file_size({file_size_mb:.1f}) > threshold({half_track_threshold_mb}) = {file_size_mb > half_track_threshold_mb}, suitable = {model_manager.is_file_suitable_for_musicnn(audio_size_bytes)}, use_half_track = {use_half_track}")  # Direct print for debugging
            log_universal('DEBUG', 'Audio', f'MusicNN decision: file_size_mb({file_size_mb:.1f}) > threshold({half_track_threshold_mb}) = {file_size_mb > half_track_threshold_mb}, suitable = {model_manager.is_file_suitable_for_musicnn(audio_size_bytes)}, use_half_track = {use_half_track}')
            
            if use_half_track:
                log_universal('INFO', 'Audio', f'Large audio detected ({file_size_mb:.1f}MB, {len(audio)} samples) - using half-track loading for MusiCNN')
                # Use middle 50% of the audio for MusiCNN analysis
                start_sample = len(audio) // 4
                end_sample = 3 * len(audio) // 4
                audio = audio[start_sample:end_sample]
                log_universal('INFO', 'Audio', f'Half-track loaded: {len(audio)} samples for MusiCNN analysis')
            else:
                log_universal('DEBUG', 'Audio', f'Using full track for MusiCNN analysis ({file_size_mb:.1f}MB, {len(audio)} samples)')
            
            # Use shared models for inference
            try:
                log_universal('INFO', 'Audio', 'Starting MusiCNN inference...')
                
                # Suppress TensorFlow warnings during inference
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    
                    # Enhanced audio preprocessing for MusiCNN
                    audio_16k = self._prepare_audio_for_musicnn(audio, sample_rate)
                    if audio_16k is None:
                        log_universal('ERROR', 'Audio', 'Failed to prepare audio for MusiCNN')
                        features.update({'embedding': [], 'tags': {}, 'musicnn_skipped': 1})
                        return features
                    
                    # Run activations inference using shared model
                    log_universal('INFO', 'Audio', 'Running MusiCNN activations inference...')
                    activations = activations_model(audio_16k)  # shape: [time, tags]
                    
                    # Enhanced return type handling
                    activations = self._validate_and_convert_musicnn_output(activations, 'activations')
                    if activations is None:
                        features.update({'embedding': [], 'tags': {}, 'musicnn_skipped': 1})
                        return features
                    
                    # Calculate tag probabilities with enhanced validation
                    tags = self._calculate_musicnn_tags(activations, tag_names)
                    if tags is None:
                        features.update({'embedding': [], 'tags': {}, 'musicnn_skipped': 1})
                        return features
                    
                    log_universal('INFO', 'Audio', f'MusiCNN tags extracted: {len(tags)} tags')
                    log_universal('INFO', 'Audio', f'Top 5 predicted tags: {sorted(tags.items(), key=lambda x: x[1], reverse=True)[:5]}')
                    
                    # Run embeddings inference using shared model
                    log_universal('INFO', 'Audio', 'Running MusiCNN embeddings inference...')
                    embeddings = embeddings_model(audio_16k)
                    
                    # Enhanced embeddings handling
                    embeddings = self._validate_and_convert_musicnn_output(embeddings, 'embeddings')
                    if embeddings is None:
                        features.update({'embedding': [], 'tags': {}, 'musicnn_skipped': 1})
                        return features
                    
                    # Calculate embedding statistics with enhanced validation
                    embedding_stats = self._calculate_musicnn_embeddings(embeddings)
                    if embedding_stats is None:
                        features.update({'embedding': [], 'tags': {}, 'musicnn_skipped': 1})
                        return features
                    
                    features.update({
                        'embedding': embedding_stats['mean'],
                        'embedding_std': embedding_stats['std'],
                        'embedding_min': embedding_stats['min'],
                        'embedding_max': embedding_stats['max'],
                        'tags': tags,
                        'musicnn_skipped': 0
                    })
                    
                    log_universal('INFO', 'Audio', f'MusiCNN extraction completed successfully: {len(tags)} tags, {len(embedding_stats["mean"])} embedding dimensions')
                    
            except Exception as e:
                log_universal('ERROR', 'Audio', f'MusiCNN inference failed: {e}')
                features.update({'embedding': [], 'tags': {}, 'musicnn_skipped': 1})
            
        except Exception as e:
            log_universal('ERROR', 'Audio', f'MusiCNN feature extraction failed: {e}')
            features.update({'embedding': [], 'tags': {}, 'musicnn_skipped': 1})
        
        return features
    
    def _prepare_audio_for_musicnn(self, audio: np.ndarray, sample_rate: int) -> Optional[np.ndarray]:
        """Prepare audio for MusiCNN with enhanced preprocessing."""
        try:
            # Resample to 16kHz if needed
            if sample_rate != 16000:
                if ESSENTIA_AVAILABLE:
                    import essentia.standard as es
                    # Use Essentia resampler for better quality
                    resampler = es.Resample(inputSampleRate=sample_rate, outputSampleRate=16000)
                    audio_16k = resampler(audio)
                    log_universal('DEBUG', 'Audio', f'Resampled with Essentia: {len(audio_16k)} samples at 16kHz')
                else:
                    # Fallback to librosa
                    import librosa
                    audio_16k = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
                    log_universal('DEBUG', 'Audio', f'Resampled with librosa: {len(audio_16k)} samples at 16kHz')
            else:
                audio_16k = audio
                log_universal('DEBUG', 'Audio', f'Using original audio: {len(audio_16k)} samples at 16kHz')
            
            # Validate audio quality
            if len(audio_16k) < 16000:  # Less than 1 second
                log_universal('WARNING', 'Audio', 'Audio too short for MusiCNN analysis')
                return None
            
            # Normalize audio if needed
            if np.max(np.abs(audio_16k)) > 1.0:
                audio_16k = audio_16k / np.max(np.abs(audio_16k))
                log_universal('DEBUG', 'Audio', 'Normalized audio amplitude')
            
            return audio_16k
            
        except Exception as e:
            log_universal('ERROR', 'Audio', f'Audio preparation failed: {e}')
            return None
    
    def _validate_and_convert_musicnn_output(self, output, output_type: str) -> Optional[np.ndarray]:
        """Validate and convert MusiCNN output with enhanced error handling."""
        try:
            if isinstance(output, list):
                log_universal('DEBUG', 'Audio', f'MusiCNN {output_type} returned as list with {len(output)} elements')
                output = np.array(output)
            elif hasattr(output, 'shape'):
                log_universal('DEBUG', 'Audio', f'MusiCNN {output_type} shape: {output.shape}')
            else:
                log_universal('WARNING', 'Audio', f'Unexpected {output_type} type: {type(output)}')
                return None
            
            # Validate shape
            if len(output.shape) != 2:
                log_universal('WARNING', 'Audio', f'Unexpected {output_type} shape: {output.shape}, expected 2D array')
                return None
            
            # Check for NaN or infinite values
            if np.any(np.isnan(output)) or np.any(np.isinf(output)):
                log_universal('WARNING', 'Audio', f'{output_type} contains NaN or infinite values')
                return None
            
            return output
            
        except Exception as e:
            log_universal('ERROR', 'Audio', f'{output_type} validation failed: {e}')
            return None
    
    def _calculate_musicnn_tags(self, activations: np.ndarray, tag_names: list) -> Optional[Dict[str, float]]:
        """Calculate MusiCNN tags with enhanced validation."""
        try:
            # Calculate mean across time dimension
            tag_probs = activations.mean(axis=0)
            
            # Validate probabilities
            if np.any(np.isnan(tag_probs)) or np.any(np.isinf(tag_probs)):
                log_universal('WARNING', 'Audio', 'Tag probabilities contain invalid values')
                return None
            
            # Convert to dictionary with validation
            tags = {}
            for i, (tag_name, prob) in enumerate(zip(tag_names, tag_probs)):
                if isinstance(tag_name, str) and not np.isnan(prob) and not np.isinf(prob):
                    tags[tag_name] = float(prob)
                else:
                    log_universal('DEBUG', 'Audio', f'Skipping invalid tag {i}: name={tag_name}, prob={prob}')
            
            return tags
            
        except Exception as e:
            log_universal('ERROR', 'Audio', f'Tag calculation failed: {e}')
            return None
    
    def _calculate_musicnn_embeddings(self, embeddings: np.ndarray) -> Optional[Dict[str, list]]:
        """Calculate MusiCNN embedding statistics with enhanced validation."""
        try:
            # Calculate statistics across time dimension
            embedding_mean = embeddings.mean(axis=0)
            embedding_std = embeddings.std(axis=0)
            embedding_min = embeddings.min(axis=0)
            embedding_max = embeddings.max(axis=0)
            
            # Validate statistics
            for stat_name, stat_values in [('mean', embedding_mean), ('std', embedding_std), 
                                         ('min', embedding_min), ('max', embedding_max)]:
                if np.any(np.isnan(stat_values)) or np.any(np.isinf(stat_values)):
                    log_universal('WARNING', 'Audio', f'Embedding {stat_name} contains invalid values')
                    return None
            
            return {
                'mean': embedding_mean.tolist(),
                'std': embedding_std.tolist(),
                'min': embedding_min.tolist(),
                'max': embedding_max.tolist()
            }
            
        except Exception as e:
            log_universal('ERROR', 'Audio', f'Embedding calculation failed: {e}')
            return None
    
    def _extract_chroma_features(self, audio: np.ndarray, sample_rate: int, file_size_mb: float = None) -> Dict[str, Any]:
        """Extract chroma features using frame-by-frame processing."""
        features = {}

        try:
            if ESSENTIA_AVAILABLE:
                import essentia.standard as es
                essentia.log.setLevel(essentia.log.Error)

                # Validate audio parameters before chroma extraction
                if len(audio) < self.frame_size:
                    log_universal('WARNING', 'Audio', f'Audio too short for chroma extraction: {len(audio)} samples')
                    features['chroma_mean'] = [0.0] * 12
                    features['chroma_std']  = [0.0] * 12
                    return features

                # Ensure audio is mono
                if len(audio.shape) > 1:
                    audio = np.mean(audio, axis=1)

                # 3-tier system based on file size
                half_track_threshold_mb = self.config.get('HALF_TRACK_THRESHOLD_MB', 100)

                if file_size_mb is not None:
                    if file_size_mb > half_track_threshold_mb:
                        start_sample = len(audio) // 4
                        end_sample = 3 * len(audio) // 4
                        audio = audio[start_sample:end_sample]
                        log_universal('INFO', 'Audio', f'Large file detected ({file_size_mb:.1f}MB) - using half-track loading for chroma extraction')
                    else:
                        log_universal('DEBUG', 'Audio', f'Small file ({file_size_mb:.1f}MB) - using full audio for chroma extraction')
                else:
                    estimated_size_mb = (len(audio) * 4) / (1024 * 1024)
                    if estimated_size_mb > half_track_threshold_mb:
                        start_sample = len(audio) // 4
                        end_sample = 3 * len(audio) // 4
                        audio = audio[start_sample:end_sample]
                        log_universal('INFO', 'Audio', f'Large file detected ({estimated_size_mb:.1f}MB estimated) - using half-track loading for chroma extraction')
                    else:
                        log_universal('DEBUG', 'Audio', f'Small file ({estimated_size_mb:.1f}MB estimated) - using full audio for chroma extraction')

                # Initialize Chromagram with configured frameSize/hopSize
                chroma_algo = es.Chromagram(
                    frameSize=self.frame_size,
                    hopSize=self.hop_size,
                    sampleRate=sample_rate
                )

                chroma_frames = []
                for frame in es.FrameGenerator(audio, frameSize=self.frame_size, hopSize=self.hop_size, startFromZero=True):
                    chroma_frames.append(chroma_algo(frame))

                if chroma_frames:
                    chroma_matrix = np.stack(chroma_frames, axis=1)
                    features['chroma_mean'] = np.mean(chroma_matrix, axis=1).tolist()
                    features['chroma_std']  = np.std(chroma_matrix, axis=1).tolist()
                else:
                    features['chroma_mean'] = [0.0] * 12
                    features['chroma_std']  = [0.0] * 12

            elif LIBROSA_AVAILABLE:
                chroma = librosa.feature.chroma_cqt(y=audio, sr=sample_rate, hop_length=self.hop_size)
                features['chroma_mean'] = np.mean(chroma, axis=1).tolist()
                features['chroma_std']  = np.std(chroma, axis=1).tolist()

            log_universal('DEBUG', 'Audio', f'Extracted chroma features')

        except Exception as e:
            log_universal('WARNING', 'Audio', f'Chroma feature extraction failed: {e}')
            features['chroma_mean'] = [0.0] * 12
            features['chroma_std']  = [0.0] * 12

        return features

    
    def _is_long_audio_track(self, file_path: str) -> bool:
        """
        Check if a file is a long audio track (45+ minutes).
        
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
            
            # Realistic threshold for long audio: 45 minutes (typical album side, long DJ mix)
            long_threshold = self.config.get('LONG_AUDIO_DURATION_THRESHOLD_MINUTES', 45)
            is_long = estimated_duration_minutes > long_threshold
            
            log_universal('DEBUG', 'Audio', f'File {os.path.basename(file_path)}: {estimated_duration_minutes:.1f} minutes estimated, long_audio: {is_long}')
            
            return is_long
            
        except Exception as e:
            log_universal('WARNING', 'Audio', f'Could not determine if {file_path} is long audio: {e}')
            return False
    
    def _is_extremely_large_for_processing(self, audio: np.ndarray) -> bool:
        """
        Check if audio is extremely large and should skip certain features.
        Based on realistic duration limits for actual music files.
        
        Args:
            audio: Audio array
            
        Returns:
            True if extremely large, False otherwise
        """
        try:
            # Calculate duration in minutes
            duration_minutes = len(audio) / (self.sample_rate * 60)
            
            # Realistic threshold for extremely large files: 12+ hours
            # This covers very long radio shows, podcasts, or live recordings
            extremely_large_threshold_minutes = self.config.get('EXTREMELY_LARGE_DURATION_THRESHOLD_MINUTES', 720)  # 12 hours
            is_extremely_large = duration_minutes > extremely_large_threshold_minutes
            
            if is_extremely_large:
                log_universal('WARNING', 'Audio', f'Extremely large file detected: {duration_minutes:.1f} minutes')
                log_universal('WARNING', 'Audio', f'Skipping memory-intensive features for this file')
            
            return is_extremely_large
            
        except Exception as e:
            log_universal('WARNING', 'Audio', f'Could not determine if file is extremely large: {e}')
            return False

    def _should_skip_audio_loading(self, file_path: str) -> bool:
        """
        Check if audio file should be skipped from loading into RAM.
        Based on duration rather than file size for better accuracy.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            True if should be skipped, False otherwise
        """
        try:
            file_size = os.path.getsize(file_path)
            file_size_mb = file_size / (1024 * 1024)
            
            # Estimate duration for more accurate decision making
            estimated_duration_seconds = (file_size * 8) / (320 * 1000)
            estimated_duration_minutes = estimated_duration_seconds / 60
            
            # Get configuration values - use duration-based thresholds
            streaming_threshold_minutes = self.config.get('STREAMING_DURATION_THRESHOLD_MINUTES', 180)  # 3 hours
            skip_large_files = self.config.get('SKIP_LARGE_FILES', False)
            
            # Convert string config to boolean if needed
            if isinstance(skip_large_files, str):
                skip_large_files = skip_large_files.lower() in ('true', '1', 'yes', 'on')
            
            # Skip files longer than streaming threshold to prevent RAM saturation
            if skip_large_files and estimated_duration_minutes > streaming_threshold_minutes:
                log_universal('WARNING', 'Audio', f'Using lightweight analysis for long file ({estimated_duration_minutes:.1f}min): {os.path.basename(file_path)}')
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
            # Get duration in minutes
            if audio is not None:
                duration_minutes = len(audio) / (self.sample_rate * 60)
            else:
                # Estimate from file size
                file_size_bytes = os.path.getsize(file_path)
                estimated_duration_seconds = (file_size_bytes * 8) / (320 * 1000)
                duration_minutes = estimated_duration_seconds / 60
            
            # Get configured thresholds
            radio_threshold = self.config.get('RADIO_DURATION_MINUTES', 90)
            long_mix_threshold = self.config.get('LONG_MIX_DURATION_MINUTES', 45)
            mix_threshold = self.config.get('MIX_DURATION_MINUTES', 20)
            
            # Determine type based on duration - realistic thresholds for actual music
            if duration_minutes > radio_threshold:  # Over 90 minutes
                return 'radio'
            elif duration_minutes > long_mix_threshold:  # 45-90 minutes
                return 'long_mix'
            elif duration_minutes > mix_threshold:  # 20-45 minutes
                return 'mix'
            else:
                return 'normal'
                
        except Exception as e:
            log_universal('WARNING', 'Audio', f'Could not determine audio type: {e}')
            return 'normal'

    def _determine_long_audio_category(self, audio_path: str, metadata: Dict[str, Any] = None, 
                                     audio_features: Dict[str, Any] = None) -> str:
        """
        Determine the category of a long audio track using duration first, then content.
        
        Args:
            audio_path: Path to the audio file
            metadata: Optional metadata dictionary
            audio_features: Optional audio features dictionary

        Returns:
            Category string (long_mix, podcast, radio, compilation, or unknown)
        """
        try:
            # Priority 1: Use duration for initial categorization
            file_size_bytes = os.path.getsize(audio_path)
            estimated_duration_seconds = (file_size_bytes * 8) / (320 * 1000)
            estimated_duration_minutes = estimated_duration_seconds / 60
            
            # Get configured thresholds
            radio_threshold = self.config.get('RADIO_DURATION_MINUTES', 90)
            long_mix_threshold = self.config.get('LONG_MIX_DURATION_MINUTES', 45)
            
            # Initial categorization based on duration
            if estimated_duration_minutes > radio_threshold:
                initial_category = 'radio'
            elif estimated_duration_minutes > long_mix_threshold:
                initial_category = 'long_mix'
            else:
                initial_category = 'mix'
            
            log_universal('INFO', 'Audio', f'Initial categorization by duration ({estimated_duration_minutes:.1f}min): {initial_category}')
            
            # Priority 2: Refine with audio features (most accurate)
            if audio_features:
                refined_category = self._categorize_by_audio_features(audio_features)
                if refined_category and refined_category != 'unknown':
                    log_universal('INFO', 'Audio', f'Refined by audio features: {initial_category} -> {refined_category}')
                    return refined_category
            
            # Priority 3: Refine with metadata keywords
            if metadata:
                title = str(metadata.get('title', '')).lower()
                artist = str(metadata.get('artist', '')).lower()
                album = str(metadata.get('album', '')).lower()

                # Strong metadata indicators override duration-based categorization
                if any(word in title for word in ['podcast', 'talk', 'interview', 'conversation', 'discussion']):
                    log_universal('INFO', 'Audio', f'Refined by metadata (title): {initial_category} -> podcast')
                    return 'podcast'
                if any(word in artist for word in ['podcast', 'show', 'network']):
                    log_universal('INFO', 'Audio', f'Refined by metadata (artist): {initial_category} -> podcast')
                    return 'podcast'
                
                if any(word in title for word in ['compilation', 'collection', 'various', 'best of', 'greatest hits', 'anthology']):
                    log_universal('INFO', 'Audio', f'Refined by metadata (title): {initial_category} -> compilation')
                    return 'compilation'
                if any(word in album for word in ['compilation', 'collection', 'various', 'best of', 'greatest hits', 'anthology']):
                    log_universal('INFO', 'Audio', f'Refined by metadata (album): {initial_category} -> compilation')
                    return 'compilation'
                
                if any(word in title for word in ['classical', 'orchestra', 'symphony', 'concerto', 'sonata']):
                    log_universal('INFO', 'Audio', f'Refined by metadata (title): {initial_category} -> classical')
                    return 'classical'
                if any(word in artist for word in ['orchestra', 'symphony', 'philharmonic', 'quartet']):
                    log_universal('INFO', 'Audio', f'Refined by metadata (artist): {initial_category} -> classical')
                    return 'classical'
                
                if any(word in title for word in ['jazz', 'swing', 'bebop', 'fusion']):
                    log_universal('INFO', 'Audio', f'Refined by metadata (title): {initial_category} -> jazz')
                    return 'jazz'
                if any(word in artist for word in ['jazz', 'swing', 'bebop', 'fusion']):
                    log_universal('INFO', 'Audio', f'Refined by metadata (artist): {initial_category} -> jazz')
                    return 'jazz'

            # Priority 4: Check filename for patterns
            filename = os.path.basename(audio_path).lower()
            if any(word in filename for word in ['podcast', 'episode', 'show', 'talk', 'interview']):
                log_universal('INFO', 'Audio', f'Refined by filename: {initial_category} -> podcast')
                return 'podcast'
            if any(word in filename for word in ['compilation', 'collection', 'various']):
                log_universal('INFO', 'Audio', f'Refined by filename: {initial_category} -> compilation')
                return 'compilation'

            # Return duration-based categorization if no content-based refinement
            log_universal('INFO', 'Audio', f'Using duration-based categorization: {initial_category}')
            return initial_category
                
        except Exception as e:
            log_universal('WARNING', 'Audio', f'Could not determine long audio category: {e}')
            return 'unknown'

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
            # Sample configurable segments for better representation
            sample_duration = self.config.get('CATEGORIZATION_SAMPLE_DURATION_SECONDS', 30)  # seconds
            sample_count = self.config.get('CATEGORIZATION_SAMPLE_COUNT', 4)
            sample_size = sample_duration * sample_rate
            total_duration = len(audio) / sample_rate
            
            # Calculate sample positions: 10%, 30%, 60%, 80% of track (configurable)
            positions = [0.1, 0.3, 0.6, 0.8][:sample_count]  # Use first N positions
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
            
            # Extract basic spectral features from all 4 samples for better representation
            if self.extract_spectral:
                log_universal('INFO', 'Audio', 'Starting spectral analysis for categorization (4 samples)')
                
                all_spectral_features = []
                for i, sample in enumerate(audio_samples):
                    log_universal('DEBUG', 'Audio', f'Analyzing spectral sample {i+1}/4')
                    sample_spectral = self._extract_spectral_features(sample, sample_rate)
                    if sample_spectral:
                        all_spectral_features.append(sample_spectral)
                
                # Combine spectral features from all samples
                if all_spectral_features:
                    # Average spectral features for better representation
                    combined_spectral = {}
                    for key in all_spectral_features[0].keys():
                        values = [f.get(key, 0) for f in all_spectral_features if f.get(key) is not None]
                        if values:
                            combined_spectral[key] = float(np.mean(values))
                    
                    features.update(combined_spectral)
                    log_universal('INFO', 'Audio', f'Spectral features combined from {len(all_spectral_features)} samples')
                else:
                    log_universal('WARNING', 'Audio', 'Spectral features extraction returned None for all samples')
            
            # Extract loudness features from all 4 samples
            if self.extract_loudness:
                log_universal('INFO', 'Audio', 'Starting loudness analysis for categorization (4 samples)')
                
                all_loudness_features = []
                for i, sample in enumerate(audio_samples):
                    log_universal('DEBUG', 'Audio', f'Analyzing loudness sample {i+1}/4')
                    sample_loudness = self._extract_loudness_features(sample, sample_rate)
                    if sample_loudness:
                        all_loudness_features.append(sample_loudness)
                
                # Combine loudness features from all samples
                if all_loudness_features:
                    # Average loudness features for better representation
                    combined_loudness = {}
                    for key in all_loudness_features[0].keys():
                        values = [f.get(key, 0) for f in all_loudness_features if f.get(key) is not None]
                        if values:
                            combined_loudness[key] = float(np.mean(values))
                    
                    features.update(combined_loudness)
                    log_universal('INFO', 'Audio', f'Loudness features combined from {len(all_loudness_features)} samples')
                else:
                    log_universal('WARNING', 'Audio', 'Loudness features extraction returned None for all samples')
            
            # Extract key features from all 4 samples for better key detection
            if self.extract_key:
                log_universal('INFO', 'Audio', 'Starting key analysis for categorization (4 samples)')
                
                all_key_features = []
                all_keys = []
                all_modes = []
                
                for i, sample in enumerate(audio_samples):
                    log_universal('DEBUG', 'Audio', f'Analyzing key sample {i+1}/4')
                    sample_key = self._extract_key_features(sample, sample_rate)
                    if sample_key and 'key' in sample_key:
                        all_key_features.append(sample_key)
                        all_keys.append(sample_key['key'])
                        if 'mode' in sample_key:
                            all_modes.append(sample_key['mode'])
                
                # Combine key features from all samples
                if all_key_features:
                    # Use most common key (most robust)
                    if all_keys:
                        from collections import Counter
                        key_counter = Counter(all_keys)
                        most_common_key = key_counter.most_common(1)[0][0]
                        features['key'] = most_common_key
                        features['key_estimates'] = all_keys
                        log_universal('INFO', 'Audio', f'Combined key from {len(all_keys)} samples: most_common={most_common_key}, all={all_keys}')
                    
                    # Use most common mode
                    if all_modes:
                        mode_counter = Counter(all_modes)
                        most_common_mode = mode_counter.most_common(1)[0][0]
                        features['mode'] = most_common_mode
                        log_universal('INFO', 'Audio', f'Combined mode from {len(all_modes)} samples: most_common={most_common_mode}')
                    
                    log_universal('INFO', 'Audio', f'Key features combined from {len(all_key_features)} samples')
                else:
                    log_universal('WARNING', 'Audio', 'Key features extraction returned None for all samples')
            
            # Extract lightweight MFCC from all 4 samples for better genre classification
            if self.extract_mfcc:
                log_universal('INFO', 'Audio', 'Starting MFCC analysis for categorization (4 samples)')
                
                all_mfcc_features = []
                for i, sample in enumerate(audio_samples):
                    log_universal('DEBUG', 'Audio', f'Analyzing MFCC sample {i+1}/4')
                    sample_mfcc = self._extract_mfcc_features(sample, sample_rate)
                    if sample_mfcc:
                        all_mfcc_features.append(sample_mfcc)
                
                # Combine MFCC features from all samples
                if all_mfcc_features:
                    # Average MFCC features for better representation
                    combined_mfcc = {}
                    for key in all_mfcc_features[0].keys():
                        values = [f.get(key, 0) for f in all_mfcc_features if f.get(key) is not None]
                        if values:
                            if isinstance(values[0], (list, np.ndarray)):
                                # For array-like features, average the arrays
                                combined_mfcc[key] = np.mean(values, axis=0).tolist()
                            else:
                                # For scalar features, average the values
                                combined_mfcc[key] = float(np.mean(values))
                    
                    features.update(combined_mfcc)
                    log_universal('INFO', 'Audio', f'MFCC features combined from {len(all_mfcc_features)} samples')
                else:
                    log_universal('WARNING', 'Audio', 'MFCC features extraction returned None for all samples')
            
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
                features.update({'embedding': [], 'tags': {}, 'musicnn_skipped': 1})
            
            # Extract chroma features from all 4 samples for better genre classification
            if self.extract_chroma:
                log_universal('INFO', 'Audio', 'Starting chroma analysis for categorization (4 samples)')
                
                all_chroma_features = []
                for i, sample in enumerate(audio_samples):
                    log_universal('DEBUG', 'Audio', f'Analyzing chroma sample {i+1}/4')
                    sample_chroma = self._extract_chroma_features(sample, sample_rate)
                    if sample_chroma:
                        all_chroma_features.append(sample_chroma)
                
                # Combine chroma features from all samples
                if all_chroma_features:
                    # Average chroma features for better representation
                    combined_chroma = {}
                    for key in all_chroma_features[0].keys():
                        values = [f.get(key, 0) for f in all_chroma_features if f.get(key) is not None]
                        if values:
                            if isinstance(values[0], (list, np.ndarray)):
                                # For array-like features, average the arrays
                                combined_chroma[key] = np.mean(values, axis=0).tolist()
                            else:
                                # For scalar features, average the values
                                combined_chroma[key] = float(np.mean(values))
                    
                    features.update(combined_chroma)
                    log_universal('INFO', 'Audio', f'Chroma features combined from {len(all_chroma_features)} samples')
                else:
                    log_universal('WARNING', 'Audio', 'Chroma features extraction returned None for all samples')
            
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
            features.update({'embedding': [], 'tags': {}, 'musicnn_skipped': 1})
            
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
                except (json.JSONDecodeError, ValueError):
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

    def _extract_missing_artist_title(self, metadata: Dict[str, Any], file_path: str):
        """
        Comprehensive fallback for missing artist/title extraction.
        
        Args:
            metadata: Metadata dictionary to update
            file_path: Path to audio file
        """
        try:
            # Check if we already have artist and title
            artist = metadata.get('artist', '').strip()
            title = metadata.get('title', '').strip()
            
            if artist and title:
                log_universal('DEBUG', 'Audio', f'Artist and title already present: {artist} - {title}')
                return
            
            filename = os.path.basename(file_path)
            name_without_ext = os.path.splitext(filename)[0]
            
            log_universal('DEBUG', 'Audio', f'Attempting to extract artist/title from filename: {name_without_ext}')
            
            # Method 1: Try "Artist - Title" format (most common)
            if ' - ' in name_without_ext:
                parts = name_without_ext.split(' - ', 1)
                if len(parts) == 2:
                    extracted_artist = parts[0].strip()
                    extracted_title = parts[1].strip()
                    
                    if extracted_artist and extracted_title:
                        if not artist:
                            metadata['artist'] = extracted_artist
                            log_universal('INFO', 'Audio', f'Extracted artist from filename: {extracted_artist}')
                        if not title:
                            metadata['title'] = extracted_title
                            log_universal('INFO', 'Audio', f'Extracted title from filename: {extracted_title}')
                        return
            
            # Method 2: Try "Artist_-_Title" format (underscore separator)
            if '_-_' in name_without_ext:
                parts = name_without_ext.split('_-_', 1)
                if len(parts) == 2:
                    extracted_artist = parts[0].strip()
                    extracted_title = parts[1].strip()
                    
                    if extracted_artist and extracted_title:
                        if not artist:
                            metadata['artist'] = extracted_artist
                            log_universal('INFO', 'Audio', f'Extracted artist from filename: {extracted_artist}')
                        if not title:
                            metadata['title'] = extracted_title
                            log_universal('INFO', 'Audio', f'Extracted title from filename: {extracted_title}')
                        return
            
            # Method 3: Try "Artist__Title" format (double underscore)
            if '__' in name_without_ext:
                parts = name_without_ext.split('__', 1)
                if len(parts) == 2:
                    extracted_artist = parts[0].strip()
                    extracted_title = parts[1].strip()
                    
                    if extracted_artist and extracted_title:
                        if not artist:
                            metadata['artist'] = extracted_artist
                            log_universal('INFO', 'Audio', f'Extracted artist from filename: {extracted_artist}')
                        if not title:
                            metadata['title'] = extracted_title
                            log_universal('INFO', 'Audio', f'Extracted title from filename: {extracted_title}')
                        return
            
            # Method 4: Try directory name as artist if no artist found
            if not artist:
                dir_name = os.path.basename(os.path.dirname(file_path))
                if dir_name and dir_name not in ['', '.', '..']:
                    metadata['artist'] = dir_name
                    log_universal('INFO', 'Audio', f'Using directory name as artist: {dir_name}')
            
            # Method 5: Use filename as title if no title found
            if not title:
                # Clean up filename for title
                clean_title = name_without_ext
                # Remove common prefixes/suffixes
                for prefix in ['track', 'song', 'audio', 'music']:
                    if clean_title.lower().startswith(prefix):
                        clean_title = clean_title[len(prefix):].strip(' -_')
                
                if clean_title:
                    metadata['title'] = clean_title
                    log_universal('INFO', 'Audio', f'Using cleaned filename as title: {clean_title}')
            
            # Final check and logging
            final_artist = metadata.get('artist', '').strip()
            final_title = metadata.get('title', '').strip()
            
            if final_artist and final_title:
                log_universal('INFO', 'Audio', f'Final metadata: {final_artist} - {final_title}')
            else:
                log_universal('WARNING', 'Audio', f'Could not extract artist/title from: {filename}')
                if not final_artist:
                    metadata['artist'] = 'Unknown Artist'
                if not final_title:
                    metadata['title'] = name_without_ext or 'Unknown Title'
                
        except Exception as e:
            log_universal('ERROR', 'Audio', f'Error extracting artist/title from filename: {e}')
            # Set defaults
            if not metadata.get('artist'):
                metadata['artist'] = 'Unknown Artist'
            if not metadata.get('title'):
                metadata['title'] = os.path.basename(file_path) or 'Unknown Title'

    def _enrich_metadata_with_external_apis(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enrich metadata using external APIs (MusicBrainz, Last.fm, etc.).
        
        Args:
            metadata: Original metadata dictionary
            
        Returns:
            Enriched metadata dictionary
        """
        try:
            from .external_apis import get_enhanced_metadata_enrichment_service
            
            enrichment_service = get_enhanced_metadata_enrichment_service()
            if enrichment_service.is_available():
                log_universal('DEBUG', 'Audio', 'Enriching metadata with external APIs')
                enriched_metadata = enrichment_service.enrich_metadata(metadata)
                
                # Log what was enriched
                original_artist = metadata.get('artist', 'Unknown')
                original_title = metadata.get('title', 'Unknown')
                enriched_artist = enriched_metadata.get('artist', original_artist)
                enriched_title = enriched_metadata.get('title', original_title)
                
                if enriched_artist != original_artist or enriched_title != original_title:
                    log_universal('INFO', 'Audio', f'External API enrichment successful: {original_artist} - {original_title} → {enriched_artist} - {enriched_title}')
                else:
                    log_universal('INFO', 'Audio', f'External API enrichment completed (no changes): {original_artist} - {original_title}')
                
                return enriched_metadata
            else:
                log_universal('DEBUG', 'Audio', 'No external APIs available for enrichment')
                return metadata
                
        except Exception as e:
            log_universal('WARNING', 'Audio', f'External API enrichment failed: {e}')
            return metadata


def get_audio_analyzer(config: Dict[str, Any] = None) -> 'AudioAnalyzer':
    """Get a configured audio analyzer instance."""
    return AudioAnalyzer(config)

def load_half_track(audio_path: str, sample_rate: int = 44100, config: Dict[str, Any] = None, processing_mode: str = 'parallel') -> Tuple[Optional[np.ndarray], Optional[int]]:
    """
    Load only half of the audio track for memory efficiency.
    This provides sufficient data for accurate feature extraction while reducing memory usage.
    
    Args:
        audio_path: Path to audio file
        sample_rate: Target sample rate
        config: Configuration dictionary
        processing_mode: Processing mode ('parallel' or 'sequential')
        
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
            min_memory_mb = config.get('MIN_MEMORY_FOR_HALF_TRACK_GB', 1.0) * 1024 if config else 1024  # Reduced requirement
            if available_memory_mb < min_memory_mb:
                log_universal('WARNING', 'Audio', f'Low memory available ({available_memory_mb:.1f}MB) - skipping {os.path.basename(audio_path)}')
                return None, None
        except Exception:
            pass  # Continue if memory check fails
        
        # Check file size
        file_size_mb = 0
        try:
            file_size = os.path.getsize(audio_path)
            file_size_mb = file_size / (1024 * 1024)
            
            if file_size < 1024:  # Less than 1KB
                log_universal('WARNING', 'Audio', f'File too small ({file_size} bytes): {os.path.basename(audio_path)}')
                return None, None
                
        except Exception as e:
            log_universal('WARNING', 'Audio', f'Could not check file size: {e}')
        
        # Load audio using Essentia if available
        if ESSENTIA_AVAILABLE:
            try:
                import essentia.standard as es
                
                # Load full audio first to get duration
                loader = es.MonoLoader(filename=audio_path, sampleRate=sample_rate)
                full_audio = loader()
                
                if full_audio is None or len(full_audio) == 0:
                    log_universal('WARNING', 'Audio', f'Empty audio file: {os.path.basename(audio_path)}')
                    return None, None
                
                # Calculate half duration
                total_samples = len(full_audio)
                half_samples = total_samples // 2
                
                # Take the middle half of the track (most representative)
                start_sample = total_samples // 4  # Start at 25%
                end_sample = start_sample + half_samples  # End at 75%
                
                # Ensure we don't go out of bounds
                end_sample = min(end_sample, total_samples)
                start_sample = max(0, end_sample - half_samples)
                
                # Extract half track
                half_audio = full_audio[start_sample:end_sample]
                
                log_universal('DEBUG', 'Audio', f'Loaded half track: {os.path.basename(audio_path)} ({len(half_audio)}/{total_samples} samples, {sample_rate}Hz)')
                log_universal('DEBUG', 'Audio', f'  Half track duration: {len(half_audio)/sample_rate:.1f}s (from {start_sample/sample_rate:.1f}s to {end_sample/sample_rate:.1f}s)')
                
                return half_audio, sample_rate
                
            except Exception as e:
                log_universal('WARNING', 'Audio', f'Essentia half-track loading failed for {os.path.basename(audio_path)}: {e}')
                # Fall through to librosa
        
        # Fallback to librosa
        if LIBROSA_AVAILABLE:
            try:
                import librosa
                
                # Get audio duration first
                duration = librosa.get_duration(path=audio_path)
                half_duration = duration / 2
                start_time = duration / 4  # Start at 25%
                
                # Load half track with librosa
                half_audio, sr = librosa.load(
                    audio_path, 
                    sr=sample_rate, 
                    mono=True,
                    offset=start_time,
                    duration=half_duration
                )
                
                log_universal('DEBUG', 'Audio', f'Loaded half track with librosa: {os.path.basename(audio_path)} ({len(half_audio)} samples, {sr}Hz)')
                log_universal('DEBUG', 'Audio', f'  Half track duration: {len(half_audio)/sr:.1f}s (from {start_time:.1f}s to {start_time + half_duration:.1f}s)')
                
                return half_audio, sr
                
            except Exception as e:
                log_universal('ERROR', 'Audio', f'Librosa half-track loading failed for {os.path.basename(audio_path)}: {e}')
                return None, None
        
        log_universal('ERROR', 'Audio', f'No audio loading library available for {os.path.basename(audio_path)}')
        return None, None
        
    except Exception as e:
        log_universal('ERROR', 'Audio', f'Unexpected error loading half track {os.path.basename(audio_path)}: {e}')
        return None, None
