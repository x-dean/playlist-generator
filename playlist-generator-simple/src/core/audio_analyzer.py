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


def safe_essentia_load(audio_path: str, sample_rate: int = 44100) -> Tuple[Optional[np.ndarray], Optional[int]]:
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
            if available_memory_mb < 500:  # Less than 500MB available
                log_universal('WARNING', 'Audio', f'Low memory available ({available_memory_mb:.1f}MB) - skipping {os.path.basename(audio_path)}')
                return None, None
        except Exception:
            pass  # Continue if memory check fails
        
        # Check file size and implement aggressive limits for parallel processing
        file_size_mb = 0  # Initialize to avoid NameError
        try:
            file_size = os.path.getsize(audio_path)
            file_size_mb = file_size / (1024 * 1024)
            
            if file_size < 1024:  # Less than 1KB
                log_universal('WARNING', 'Audio', f'File too small ({file_size} bytes): {os.path.basename(audio_path)}')
                return None, None
            
            # Much more aggressive limits for parallel processing
            max_file_size_mb = 500  # 500MB limit
            warning_threshold_mb = 100  # 100MB warning threshold
            
            # Skip large files to prevent RAM saturation
            if file_size_mb > max_file_size_mb:
                log_universal('WARNING', 'Audio', f'File too large ({file_size_mb:.1f}MB): {os.path.basename(audio_path)} - skipping to prevent RAM saturation')
                return None, None
                
            # Warn for large files
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
            if file_size_mb > 50:  # Files larger than 50MB
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
                    if file_size_mb > 50:
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
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize audio analyzer.
        
        Args:
            config: Configuration dictionary (uses global config if None)
        """
        if config is None:
            config = config_loader.get_audio_analysis_config()
        
        self.config = config
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
        
        # Check if file exists
        if not os.path.exists(file_path):
            log_universal('ERROR', 'Audio', f'File not found: {file_path}')
            return None
        
        # Calculate file hash for change detection
        try:
            file_hash = self._calculate_file_hash(file_path)
            file_size = os.path.getsize(file_path)
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
        metadata = self._extract_metadata(file_path)
        
        # Enrich metadata with external APIs FIRST (before audio analysis)
        if metadata:
            # Add filename to metadata for fallback extraction
            metadata['filename'] = os.path.basename(file_path)
            enriched_metadata = self._enrich_metadata_with_external_apis(metadata)
            metadata = enriched_metadata
        
        # Check if we should skip audio loading for large files
        if self._should_skip_audio_loading(file_path):
            log_universal('WARNING', 'Audio', f'Skipping audio analysis for large file: {os.path.basename(file_path)}')
            # Return basic analysis with enriched metadata
            return self._create_basic_analysis_for_large_file(file_path, file_size, file_hash, metadata)
        
        # Load audio data
        audio, sample_rate = safe_essentia_load(file_path, self.sample_rate)
        if audio is None:
            log_universal('ERROR', 'Audio', f'Failed to load audio: {os.path.basename(file_path)}')
            return None
        
        # Check if this is a long audio track
        is_long_audio = self._is_long_audio_track(file_path)
        
        # Add long audio flag to metadata for feature extraction
        if metadata:
            metadata['is_long_audio'] = is_long_audio
        else:
            metadata = {'is_long_audio': is_long_audio}
        
        # Perform audio analysis with enriched metadata
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
                if file_size_mb > 500:  # Files larger than 500MB
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
                if available_memory_mb < 100:  # Less than 100MB available
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
        
        # Use simplified analysis for long audio tracks if configured
        if is_long_audio and long_audio_simplified and long_audio_skip_detailed:
            log_universal('INFO', 'Audio', 'Using simplified analysis for long audio track')
            return self._extract_simplified_features(audio, sample_rate, metadata)
        
        try:
            # Extract rhythm features (skip for extremely large files)
            if self.extract_rhythm:
                if is_extremely_large_for_processing:
                    log_universal('WARNING', 'Audio', 'Skipping rhythm extraction for extremely large file')
                    features['bpm'] = -1.0  # Special marker for failed BPM extraction
                    features['rhythm_confidence'] = 0.0
                else:
                    rhythm_features = self._extract_rhythm_features(audio, sample_rate)
                    features.update(rhythm_features)
            
            # Extract spectral features (skip for extremely large files)
            if self.extract_spectral:
                if is_extremely_large_for_processing:
                    log_universal('WARNING', 'Audio', 'Skipping spectral extraction for extremely large file')
                    features['spectral_centroid'] = 0.0
                    features['spectral_rolloff'] = 0.0
                    features['spectral_bandwidth'] = 0.0
                else:
                    spectral_features = self._extract_spectral_features(audio, sample_rate)
                    features.update(spectral_features)
            
            # Extract loudness features (skip for extremely large files)
            if self.extract_loudness:
                if is_extremely_large_for_processing:
                    log_universal('WARNING', 'Audio', 'Skipping loudness extraction for extremely large file')
                    features['loudness'] = 0.0
                    features['dynamic_complexity'] = 0.0
                else:
                    loudness_features = self._extract_loudness_features(audio, sample_rate)
                    features.update(loudness_features)
            
            # Extract key and mode (always extract - not memory intensive)
            if self.extract_key:
                key_features = self._extract_key_features(audio, sample_rate)
                features.update(key_features)
            
            # Extract MFCC features (skip for extremely large files)
            if self.extract_mfcc:
                if is_extremely_large_for_processing:
                    log_universal('WARNING', 'Audio', 'Skipping MFCC extraction for extremely large file to avoid memory issues')
                    features['mfcc_coefficients'] = []
                    features['mfcc_bands'] = []
                    features['mfcc_std'] = []
                else:
                    mfcc_features = self._extract_mfcc_features(audio, sample_rate)
                    features.update(mfcc_features)
            
            # Extract MusiCNN features (skip for extremely large files)
            if self.extract_musicnn and TENSORFLOW_AVAILABLE:
                if is_extremely_large_for_processing:
                    log_universal('WARNING', 'Audio', 'Skipping MusiCNN extraction for extremely large file')
                    features['embedding'] = []
                    features['tags'] = {}
                else:
                    musicnn_features = self._extract_musicnn_features(audio, sample_rate)
                    features.update(musicnn_features)
            
            # Extract chroma features (skip for extremely large files)
            if self.extract_chroma:
                if is_extremely_large_for_processing:
                    log_universal('WARNING', 'Audio', 'Skipping chroma, spectral contrast, flatness, and rolloff for extremely large file')
                    features['chroma_mean'] = []
                    features['chroma_std'] = []
                else:
                    chroma_features = self._extract_chroma_features(audio, sample_rate)
                    features.update(chroma_features)
            
            # Add BPM from metadata if available
            if metadata and 'bpm_from_metadata' in metadata:
                features['external_bpm'] = metadata['bpm_from_metadata']
            
            return features
            
        except Exception as e:
            log_universal('ERROR', 'Audio', f'Feature extraction failed: {e}')
            return None
    
    def _extract_rhythm_features(self, audio: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Extract rhythm-related features using lightweight approach for large files."""
        features = {}
        
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
                # Use KeyExtractor like the old working version
                log_universal('DEBUG', 'Audio', "Using Essentia KeyExtractor for key detection")
                key_algo = es.KeyExtractor()
                log_universal('DEBUG', 'Audio', "Running key analysis on audio")
                key, scale, strength = key_algo(audio)
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
        """Extract MusiCNN features."""
        features = {}
        
        try:
            if TENSORFLOW_AVAILABLE:
                # Get model paths from configuration
                model_path = self.config.get('MUSICNN_MODEL_PATH', '/app/models/msd-musicnn-1.pb')
                json_path = self.config.get('MUSICNN_JSON_PATH', '/app/models/msd-musicnn-1.json')
                
                # Load model if not already loaded
                if not hasattr(self, '_musicnn_model'):
                    log_universal('DEBUG', 'Audio', f'Checking MusiCNN model at: {model_path}')
                    
                    # Check if model directory exists
                    model_dir = os.path.dirname(model_path)
                    if os.path.exists(model_dir):
                        log_universal('DEBUG', 'Audio', f'Model directory exists: {model_dir}')
                        try:
                            files = os.listdir(model_dir)
                            log_universal('DEBUG', 'Audio', f'Files in model directory: {files}')
                        except Exception as e:
                            log_universal('WARNING', 'Audio', f'Cannot list model directory: {e}')
                    else:
                        log_universal('WARNING', 'Audio', f'Model directory does not exist: {model_dir}')
                    
                    if os.path.exists(model_path):
                        log_universal('DEBUG', 'Audio', f'Model file exists: {model_path}')
                        try:
                            file_size = os.path.getsize(model_path)
                            log_universal('DEBUG', 'Audio', f'Model file size: {file_size} bytes')
                            
                            if model_path.endswith('.pb'):
                                # Load protobuf graph definition
                                with tf.io.gfile.GFile(model_path, 'rb') as f:
                                    graph_def = tf.compat.v1.GraphDef()
                                    graph_def.ParseFromString(f.read())
                                
                                # Create a new graph and import the graph definition
                                graph = tf.Graph()
                                with graph.as_default():
                                    tf.compat.v1.import_graph_def(graph_def, name='')
                                
                                # Create a session to run the model
                                self._musicnn_session = tf.compat.v1.Session(graph=graph)
                                self._musicnn_graph = graph
                                
                                # Find input and output tensors by examining the graph
                                input_tensors = []
                                output_tensors = []
                                
                                for op in graph.get_operations():
                                    if op.type == 'Placeholder':
                                        input_tensors.append(op.name)
                                    elif 'dense' in op.name.lower() or 'output' in op.name.lower() or 'biasadd' in op.name.lower():
                                        output_tensors.append(op.name)
                                
                                log_universal('DEBUG', 'Audio', f'Found input tensors: {input_tensors}')
                                log_universal('DEBUG', 'Audio', f'Found output tensors: {output_tensors}')
                                
                                # Use the first input tensor found
                                if input_tensors:
                                    self._musicnn_input = graph.get_tensor_by_name(f'{input_tensors[0]}:0')
                                else:
                                    log_universal('WARNING', 'Audio', 'No input tensors found in MusiCNN model')
                                    self._musicnn_input = None
                                
                                # Try to find both embeddings and tags outputs
                                self._musicnn_outputs = []
                                for output_name in output_tensors:
                                    try:
                                        tensor = graph.get_tensor_by_name(f'{output_name}:0')
                                        self._musicnn_outputs.append(tensor)
                                        log_universal('DEBUG', 'Audio', f'Added output tensor: {output_name}:0 with shape {tensor.shape}')
                                    except Exception as e:
                                        log_universal('WARNING', 'Audio', f'Failed to get tensor {output_name}:0: {e}')
                                
                                if not self._musicnn_outputs:
                                    log_universal('WARNING', 'Audio', 'No output tensors found in MusiCNN model')
                                    self._musicnn_outputs = None
                                
                                log_universal('DEBUG', 'Audio', 'Loaded MusiCNN protobuf model')
                            elif model_path.endswith('.h5'):
                                self._musicnn_model = tf.keras.models.load_model(model_path)
                                log_universal('DEBUG', 'Audio', 'Loaded MusiCNN Keras model')
                            else:
                                log_universal('WARNING', 'Audio', f'Unsupported MusiCNN model format: {model_path}')
                                self._musicnn_model = None
                        except Exception as e:
                            log_universal('WARNING', 'Audio', f'Failed to load MusiCNN model: {e}')
                            log_universal('DEBUG', 'Audio', f'Model path: {model_path}')
                            log_universal('DEBUG', 'Audio', f'Model exists: {os.path.exists(model_path)}')
                            self._musicnn_model = None
                    else:
                        log_universal('WARNING', 'Audio', f'MusiCNN model not found: {model_path}')
                        
                        # Check if it's a placeholder file
                        try:
                            with open(model_path, 'r') as f:
                                content = f.read()
                                if 'placeholder' in content.lower():
                                    log_universal('INFO', 'Audio', 'Found MusiCNN placeholder file. MusiCNN features will be disabled.')
                                    log_universal('INFO', 'Audio', 'To enable MusiCNN:')
                                    log_universal('INFO', 'Audio', '1. Download the model from https://github.com/jordipons/musicnn')
                                    log_universal('INFO', 'Audio', '2. Replace the placeholder file with the actual model')
                                    log_universal('INFO', 'Audio', '3. Or set EXTRACT_MUSICNN=false to disable MusiCNN features')
                                else:
                                    log_universal('INFO', 'Audio', 'To enable MusiCNN features, download the model files:')
                                    log_universal('INFO', 'Audio', '1. Run: python download_musicnn_models.py')
                                    log_universal('INFO', 'Audio', '2. Or set EXTRACT_MUSICNN=false to disable MusiCNN features')
                        except:
                            log_universal('INFO', 'Audio', 'To enable MusiCNN features, download the model files:')
                            log_universal('INFO', 'Audio', '1. Run: python download_musicnn_models.py')
                            log_universal('INFO', 'Audio', '2. Or set EXTRACT_MUSICNN=false to disable MusiCNN features')
                        
                        self._musicnn_model = None
                
                # Load JSON configuration if available
                musicnn_config = {}
                if os.path.exists(json_path):
                    try:
                        import json
                        with open(json_path, 'r') as f:
                            musicnn_config = json.load(f)
                        log_universal('DEBUG', 'Audio', 'Loaded MusiCNN JSON configuration')
                    except Exception as e:
                        log_universal('WARNING', 'Audio', f'Failed to load MusiCNN JSON config: {e}')
                
                # Extract features if model is available
                if hasattr(self, '_musicnn_session') or self._musicnn_model is not None:
                    # Resample audio to 22050 Hz (MusiCNN specification)
                    if sample_rate != 22050:
                        import librosa
                        audio_22050 = librosa.resample(audio, orig_sr=sample_rate, target_sr=22050)
                    else:
                        audio_22050 = audio
                    
                    # Prepare input for MusiCNN (expects [batch, time, 96] log-mel spectrogram)
                    mel_spec = self._compute_mel_spectrogram(audio_22050, 22050)
                    
                    # Run inference
                    if hasattr(self, '_musicnn_session') and self._musicnn_input is not None and self._musicnn_outputs is not None:
                        # Protobuf model with session - run all outputs
                        predictions = self._musicnn_session.run(
                            self._musicnn_outputs,
                            feed_dict={self._musicnn_input: mel_spec[np.newaxis, ...]}
                        )
                        log_universal('DEBUG', 'Audio', f'MusiCNN predictions shape: {[p.shape for p in predictions] if isinstance(predictions, (list, tuple)) else predictions.shape}')
                    elif hasattr(self._musicnn_model, 'predict'):
                        # Keras model
                        predictions = self._musicnn_model.predict(mel_spec[np.newaxis, ...], verbose=0)
                    else:
                        # SavedModel
                        predictions = self._musicnn_model(mel_spec[np.newaxis, ...])
                    
                    # Extract embeddings and tags
                    if isinstance(predictions, (list, tuple)):
                        log_universal('DEBUG', 'Audio', f'MusiCNN returned {len(predictions)} outputs')
                        # Try to identify embeddings vs tags based on shape
                        embeddings = None
                        tags = None
                        
                        for i, pred in enumerate(predictions):
                            shape = pred.shape if hasattr(pred, 'shape') else 'unknown'
                            log_universal('DEBUG', 'Audio', f'Output {i}: shape {shape}')
                            
                            # Large output (200+ dimensions) is likely embeddings
                            if hasattr(pred, 'flatten') and pred.flatten().shape[0] > 200:
                                embeddings = pred
                                log_universal('DEBUG', 'Audio', f'Identified embeddings in output {i}')
                            # Smaller output (50-100 dimensions) is likely tags
                            elif hasattr(pred, 'flatten') and 50 <= pred.flatten().shape[0] <= 100:
                                tags = pred
                                log_universal('DEBUG', 'Audio', f'Identified tags in output {i}')
                        
                        # Fallback: use first output as embeddings if none identified
                        if embeddings is None and len(predictions) > 0:
                            embeddings = predictions[0]
                            log_universal('DEBUG', 'Audio', 'Using first output as embeddings (fallback)')
                    else:
                        embeddings = predictions
                        tags = None
                        log_universal('DEBUG', 'Audio', f'Single output with shape: {embeddings.shape if hasattr(embeddings, "shape") else "unknown"}')
                    
                    # Convert embeddings to standard 200-dimensional format
                    if hasattr(embeddings, 'flatten'):
                        embedding_flat = embeddings.flatten()
                        # Take first 200 dimensions or average if larger
                        if len(embedding_flat) > 200:
                            # For large embeddings, take first 200 or average chunks
                            if len(embedding_flat) <= 10000:
                                # Take first 200 dimensions
                                features['embedding'] = embedding_flat[:200].tolist()
                            else:
                                # For very large embeddings, average chunks
                                chunk_size = len(embedding_flat) // 200
                                averaged = []
                                for i in range(200):
                                    start = i * chunk_size
                                    end = start + chunk_size
                                    chunk = embedding_flat[start:end]
                                    averaged.append(float(np.mean(chunk)))
                                features['embedding'] = averaged
                        else:
                            # Pad with zeros if smaller than 200
                            embedding_list = embedding_flat.tolist()
                            features['embedding'] = embedding_list + [0.0] * (200 - len(embedding_list))
                    else:
                        features['embedding'] = embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings
                    
                    if tags is not None:
                        # Map tag indices to tag names if available in config
                        tag_names = musicnn_config.get('tag_names', [])
                        if tag_names and hasattr(tags, 'flatten'):
                            tag_probs = tags.flatten()
                            # Create dictionary mapping tag names to probabilities
                            features['tags'] = {tag_names[i]: float(tag_probs[i]) for i in range(min(len(tag_names), len(tag_probs)))}
                        else:
                            # Handle case where tags is a raw array/list
                            if hasattr(tags, 'flatten'):
                                tag_probs = tags.flatten()
                                if tag_names:
                                    # Map to tag names if available
                                    features['tags'] = {tag_names[i]: float(tag_probs[i]) for i in range(min(len(tag_names), len(tag_probs)))}
                                else:
                                    # Return as list if no tag names available
                                    features['tags'] = tag_probs.tolist()
                            else:
                                # Fallback to list format
                                features['tags'] = tags.tolist() if hasattr(tags, 'tolist') else tags
                    else:
                        features['tags'] = {}
                    
                    # Log MusiCNN results
                    log_universal('INFO', 'Audio', f'MusiCNN analysis completed: embedding size {len(features["embedding"])}')
                    
                    # Log top tags with confidence scores
                    if features['tags']:
                        # Handle both dict and list formats
                        if isinstance(features['tags'], dict):
                            sorted_tags = sorted(features['tags'].items(), key=lambda x: x[1], reverse=True)
                            top_tags = sorted_tags[:5]  # Top 5 tags
                            tag_summary = ', '.join([f"{tag}: {conf:.2f}" for tag, conf in top_tags])
                            log_universal('INFO', 'Audio', f'Top MusiCNN tags: {tag_summary}')
                        elif isinstance(features['tags'], list):
                            # Convert raw tag probabilities to named tags
                            tag_names = musicnn_config.get('tag_names', [])
                            if tag_names and len(features['tags']) == len(tag_names):
                                # Create dictionary with tag names
                                named_tags = {}
                                for i, prob in enumerate(features['tags']):
                                    if i < len(tag_names):
                                        named_tags[tag_names[i]] = float(prob)
                                features['tags'] = named_tags
                                
                                # Log top tags
                                sorted_tags = sorted(named_tags.items(), key=lambda x: x[1], reverse=True)
                                top_tags = sorted_tags[:5]
                                tag_summary = ', '.join([f"{tag}: {conf:.2f}" for tag, conf in top_tags])
                                log_universal('INFO', 'Audio', f'Top MusiCNN tags: {tag_summary}')
                            else:
                                log_universal('INFO', 'Audio', f'MusiCNN tags returned as list with {len(features["tags"])} values')
                                log_universal('DEBUG', 'Audio', f'First 5 tag values: {features["tags"][:5]}')
                        else:
                            log_universal('WARNING', 'Audio', f'Unexpected tags format: {type(features["tags"])}')
                    else:
                        log_universal('WARNING', 'Audio', 'No MusiCNN tags extracted')
                else:
                    features['embedding'] = []
                    features['tags'] = {}
                    log_universal('WARNING', 'Audio', 'MusiCNN model not available - skipping features')
            else:
                features['embedding'] = []
                features['tags'] = {}
                log_universal('DEBUG', 'Audio', 'TensorFlow not available - skipping MusiCNN features')
                
        except Exception as e:
            log_universal('WARNING', 'Audio', f'MusiCNN feature extraction failed: {e}')
            features.update({
                'embedding': [],
                'tags': {}
            })
        
        return features
    
    def _extract_chroma_features(self, audio: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Extract chroma features using frame-by-frame processing."""
        features = {}
        
        try:
            if ESSENTIA_AVAILABLE:
                # Use frame-by-frame approach like the old working version
                frame_size = 2048
                hop_size = 1024
                
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
            
            # Consider it long if estimated duration > 60 minutes (increased from 20)
            is_long = estimated_duration_minutes > 60
            
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
            # Increased threshold: skip for files > 1B samples (was 500M)
            is_extremely_large = len(audio) > 1000000000
            
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
            
            # Get configuration values
            streaming_threshold_mb = self.config.get('STREAMING_LARGE_FILE_THRESHOLD_MB', 500)  # Increased from 100MB
            skip_large_files = self.config.get('SKIP_LARGE_FILES', True)
            
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
            
            # Determine type based on duration
            if duration_minutes > 60:  # Over 1 hour
                return 'radio'
            elif duration_minutes > 30:  # 30-60 minutes
                return 'long_mix'
            elif duration_minutes > 20:  # 20-30 minutes
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
                if any(word in title for word in ['podcast', 'episode', 'show', 'talk']):
                    log_universal('INFO', 'Audio', f"Category determined by metadata (title): podcast")
                    return 'podcast'
                if any(word in artist for word in ['podcast', 'radio', 'station']):
                    log_universal('INFO', 'Audio', f"Category determined by metadata (artist): podcast")
                    return 'podcast'

                if any(word in title for word in ['radio', 'broadcast', 'live']):
                    log_universal('INFO', 'Audio', f"Category determined by metadata (title): radio")
                    return 'radio'
                if any(word in artist for word in ['radio', 'station', 'broadcast']):
                    log_universal('INFO', 'Audio', f"Category determined by metadata (artist): radio")
                    return 'radio'

                if any(word in title for word in ['mix', 'dj', 'set', 'session']):
                    log_universal('INFO', 'Audio', f"Category determined by metadata (title): long_mix")
                    return 'long_mix'
                if any(word in artist for word in ['dj', 'mix', 'session']):
                    log_universal('INFO', 'Audio', f"Category determined by metadata (artist): long_mix")
                    return 'long_mix'

                if any(word in title for word in ['compilation', 'collection', 'various']):
                    log_universal('INFO', 'Audio', f"Category determined by metadata (title): compilation")
                    return 'compilation'
                if any(word in album for word in ['compilation', 'collection', 'various']):
                    log_universal('INFO', 'Audio', f"Category determined by metadata (album): compilation")
                    return 'compilation'

            # Priority 3: Check filename for patterns
            filename = os.path.basename(audio_path).lower()

            if any(word in filename for word in ['podcast', 'episode', 'show']):
                return 'podcast'
            if any(word in filename for word in ['radio', 'broadcast', 'live']):
                return 'radio'
            if any(word in filename for word in ['mix', 'dj', 'set']):
                return 'long_mix'
            if any(word in filename for word in ['compilation', 'collection', 'various']):
                return 'compilation'

            # Default fallback
            return 'long_mix'

        except Exception as e:
            log_universal('WARNING', 'Audio', f"Error determining long audio category: {e}")
            return 'long_mix'

    def _categorize_by_audio_features(self, features: Dict[str, Any]) -> Optional[str]:
        """
        Categorize long audio track based on audio features with improved accuracy.

        Args:
            features: Dictionary of audio features

        Returns:
            Category string or None if cannot determine
        """
        try:
            # Extract key features with None handling
            bpm = features.get('bpm')
            confidence = features.get('confidence')
            spectral_centroid = features.get('spectral_centroid')
            spectral_flatness = features.get('spectral_flatness')
            loudness = features.get('loudness')
            dynamic_complexity = features.get('dynamic_complexity')
            
            # Use external BPM if available and main BPM is None
            if bpm is None and 'external_bpm' in features:
                bpm = features.get('external_bpm')
            
            # Set default values for None features
            if bpm is None:
                bpm = -1
            if confidence is None:
                confidence = 0.0
            if spectral_centroid is None:
                spectral_centroid = 0
            if spectral_flatness is None:
                spectral_flatness = 0.5
            if loudness is None:
                loudness = 0.5
            if dynamic_complexity is None:
                dynamic_complexity = 0.5

            log_universal('DEBUG', 'Audio', f'Categorizing with features: BPM={bpm}, Confidence={confidence:.2f}, '
                          f'Spectral_Centroid={spectral_centroid:.0f}, Spectral_Flatness={spectral_flatness:.2f}, '
                          f'Loudness={loudness:.2f}, Dynamic_Complexity={dynamic_complexity:.2f}')

            # Enhanced categorization logic with better thresholds and fallbacks

            # 1. Podcast detection (speech-like characteristics)
            # - Low BPM (speech tempo)
            # - Low confidence (speech is less rhythmic)
            # - Low spectral centroid (speech frequencies)
            # - High spectral flatness (noise-like)
            if (bpm > 0 and bpm < 100 and confidence < 0.7 and
                spectral_centroid < 3000 and spectral_flatness > 0.25):
                log_universal('INFO', 'Audio', 'Detected as podcast based on speech-like characteristics')
                return 'podcast'

            # 2. Radio detection (mixed content, variable characteristics)
            # - Medium BPM range (mixed music)
            # - Variable confidence (mixed content)
            # - Medium spectral characteristics
            if (bpm > 0 and 70 <= bpm <= 150 and 0.3 <= confidence <= 0.8 and
                spectral_centroid > 1500 and spectral_flatness < 0.5):
                log_universal('INFO', 'Audio', 'Detected as radio based on mixed content characteristics')
                return 'radio'

            # 3. Long mix detection (consistent music, high energy)
            # - High BPM (dance/electronic music)
            # - High confidence (consistent rhythm)
            # - High spectral centroid (rich harmonics)
            # - Low spectral flatness (tonal content)
            # - High loudness (energy)
            if (bpm > 0 and bpm > 110 and confidence > 0.6 and
                spectral_centroid > 2500 and spectral_flatness < 0.4 and
                loudness > 0.4):
                log_universal('INFO', 'Audio', 'Detected as long mix based on consistent music characteristics')
                return 'long_mix'

            # 4. Compilation detection (variable characteristics)
            # - Variable BPM (different songs)
            # - Lower confidence (inconsistent rhythm)
            # - Higher spectral flatness (variable content)
            # - Higher dynamic complexity (varied sections)
            # But exclude high-BPM electronic music which can have high flatness
            if (bpm > 0 and bpm < 120 and confidence < 0.7 and
                spectral_flatness > 0.3 and dynamic_complexity > 0.5):
                log_universal('INFO', 'Audio', 'Detected as compilation based on variable characteristics')
                return 'compilation'

            # 5. Fallback categorization based on strongest indicators
            if bpm > 0:
                if bpm > 120 and confidence > 0.5:
                    log_universal('INFO', 'Audio', 'Fallback: Detected as long mix based on high BPM and confidence')
                    return 'long_mix'
                elif bpm < 90 and confidence < 0.6:
                    log_universal('INFO', 'Audio', 'Fallback: Detected as podcast based on low BPM and confidence')
                    return 'podcast'
                elif spectral_flatness > 0.4:
                    log_universal('INFO', 'Audio', 'Fallback: Detected as compilation based on high spectral flatness')
                    return 'compilation'
                else:
                    log_universal('INFO', 'Audio', 'Fallback: Detected as radio based on medium characteristics')
                    return 'radio'
            
            # 6. Handle failed rhythm extraction (BPM = -1)
            if bpm == -1:
                log_universal('INFO', 'Audio', 'Rhythm extraction failed, using spectral features for categorization')
                # Use spectral features when BPM extraction fails
                if spectral_centroid > 2500 and spectral_flatness < 0.3:
                    log_universal('INFO', 'Audio', 'Fallback: Detected as long mix based on spectral characteristics')
                    return 'long_mix'
                elif spectral_flatness > 0.4:
                    log_universal('INFO', 'Audio', 'Fallback: Detected as compilation based on spectral flatness')
                    return 'compilation'
                elif spectral_centroid < 2000:
                    log_universal('INFO', 'Audio', 'Fallback: Detected as podcast based on low spectral centroid')
                    return 'podcast'
                else:
                    log_universal('INFO', 'Audio', 'Fallback: Detected as radio based on spectral characteristics')
                    return 'radio'
            
            # 6. Final fallback based on spectral characteristics
            if spectral_centroid < 2000:
                log_universal('INFO', 'Audio', 'Final fallback: Detected as podcast based on low spectral centroid')
                return 'podcast'
            elif spectral_flatness > 0.4 and spectral_centroid < 2500:
                log_universal('INFO', 'Audio', 'Final fallback: Detected as compilation based on high spectral flatness and low centroid')
                return 'compilation'
            else:
                log_universal('INFO', 'Audio', 'Final fallback: Detected as long mix based on tonal characteristics')
                return 'long_mix'

        except Exception as e:
            log_universal('WARNING', 'Audio', f"Error in audio feature categorization: {e}")
            return 'long_mix'  # Safe fallback instead of None

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


def get_audio_analyzer(config: Dict[str, Any] = None) -> 'AudioAnalyzer':
    """Get a configured audio analyzer instance."""
    return AudioAnalyzer(config)
