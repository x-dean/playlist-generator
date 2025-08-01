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
    
    Args:
        audio_path: Path to audio file
        sample_rate: Target sample rate
        
    Returns:
        Tuple of (audio_data, sample_rate) or (None, None) on failure
    """
    if not os.path.exists(audio_path):
        log_universal('ERROR', 'Audio', f'File not found: {audio_path}')
        return None, None
    
    # Check file size
    try:
        file_size = os.path.getsize(audio_path)
        if file_size < 1024:  # Less than 1KB
            log_universal('WARNING', 'Audio', f'File too small ({file_size} bytes): {os.path.basename(audio_path)}')
            return None, None
    except Exception as e:
        log_universal('WARNING', 'Audio', f'Cannot check file size for {os.path.basename(audio_path)}: {e}')
    
    try:
        import essentia.standard as es
        log_universal('DEBUG', 'Audio', f'Loading {os.path.basename(audio_path)} with Essentia MonoLoader')
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
        
        # Try librosa as fallback
        if LIBROSA_AVAILABLE:
            try:
                log_universal('DEBUG', 'Audio', f'Trying librosa fallback for {os.path.basename(audio_path)}')
                import librosa
                audio, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
                if audio is not None and len(audio) > 0:
                    log_universal('DEBUG', 'Audio', f'Librosa fallback successful: {len(audio)} samples at {sr}Hz')
                    return audio, sr
                else:
                    log_universal('WARNING', 'Audio', f'Librosa fallback returned empty audio for {os.path.basename(audio_path)}')
            except Exception as librosa_e:
                log_universal('ERROR', 'Audio', f'Librosa fallback also failed for {os.path.basename(audio_path)}: {librosa_e}')
        
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
        
        # Load audio data
        audio, sample_rate = safe_essentia_load(file_path, self.sample_rate)
        if audio is None:
            log_universal('ERROR', 'Audio', f'Failed to load audio: {os.path.basename(file_path)}')
            return None
        
        # Extract metadata first
        metadata = self._extract_metadata(file_path)
        
        # Perform audio analysis
        analysis_result = self._extract_audio_features(audio, sample_rate, metadata)
        if analysis_result is None:
            log_universal('ERROR', 'Audio', f'Feature extraction failed: {os.path.basename(file_path)}')
            return None
        
        # Enrich metadata with external APIs
        if metadata:
            enriched_metadata = self._enrich_metadata_with_external_apis(metadata)
            analysis_result['metadata'] = enriched_metadata
        else:
            analysis_result['metadata'] = metadata or {}
        
        # Add file information
        analysis_result.update({
            'file_path': file_path,
            'filename': os.path.basename(file_path),
            'file_size_bytes': file_size,
            'file_hash': file_hash,
            'analysis_date': datetime.now(),
            'analysis_version': '1.0.0'
        })
        
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
        Extract metadata from audio file tags.
        
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
            
            audio_file = File(file_path)
            if audio_file is None:
                log_universal('WARNING', 'Audio', f'Could not open file for metadata: {os.path.basename(file_path)}')
                return None
            
            metadata = {}
            
            # Extract common tags
            if hasattr(audio_file, 'tags') and audio_file.tags:
                tags = audio_file.tags
                
                # Basic metadata
                metadata['title'] = self._get_tag_value(tags, ['title', 'TIT2', 'TITLE'])
                metadata['artist'] = self._get_tag_value(tags, ['artist', 'TPE1', 'ARTIST'])
                metadata['album'] = self._get_tag_value(tags, ['album', 'TALB', 'ALBUM'])
                metadata['genre'] = self._get_tag_value(tags, ['genre', 'TCON', 'GENRE'])
                metadata['year'] = self._get_tag_value(tags, ['year', 'TYER', 'YEAR', 'date'])
                metadata['track_number'] = self._get_tag_value(tags, ['tracknumber', 'TRCK', 'TRACK'])
                metadata['disc_number'] = self._get_tag_value(tags, ['discnumber', 'TPOS', 'DISC'])
                
                # Extended metadata
                metadata['composer'] = self._get_tag_value(tags, ['composer', 'TCOM', 'COMPOSER'])
                metadata['lyricist'] = self._get_tag_value(tags, ['lyricist', 'TEXT', 'LYRICIST'])
                metadata['band'] = self._get_tag_value(tags, ['band', 'TPE2', 'BAND'])
                metadata['conductor'] = self._get_tag_value(tags, ['conductor', 'TPE3', 'CONDUCTOR'])
                metadata['remixer'] = self._get_tag_value(tags, ['remixer', 'TPE4', 'REMIXER'])
                metadata['subtitle'] = self._get_tag_value(tags, ['subtitle', 'TIT3', 'SUBTITLE'])
                metadata['grouping'] = self._get_tag_value(tags, ['grouping', 'TIT1', 'GROUPING'])
                metadata['publisher'] = self._get_tag_value(tags, ['publisher', 'TPUB', 'PUBLISHER'])
                metadata['copyright'] = self._get_tag_value(tags, ['copyright', 'TCOP', 'COPYRIGHT'])
                metadata['encoded_by'] = self._get_tag_value(tags, ['encodedby', 'TENC', 'ENCODEDBY'])
                metadata['language'] = self._get_tag_value(tags, ['language', 'TLAN', 'LANGUAGE'])
                metadata['mood'] = self._get_tag_value(tags, ['mood', 'TMOO', 'MOOD'])
                metadata['style'] = self._get_tag_value(tags, ['style', 'TSTY', 'STYLE'])
                metadata['quality'] = self._get_tag_value(tags, ['quality', 'TQUA', 'QUALITY'])
                
                # Original metadata
                metadata['original_artist'] = self._get_tag_value(tags, ['originalartist', 'TOPE', 'ORIGINALARTIST'])
                metadata['original_album'] = self._get_tag_value(tags, ['originalalbum', 'TOAL', 'ORIGINALALBUM'])
                metadata['original_year'] = self._get_tag_value(tags, ['originalyear', 'TOYE', 'ORIGINALYEAR'])
                metadata['original_filename'] = self._get_tag_value(tags, ['originalfilename', 'TOFN', 'ORIGINALFILENAME'])
                
                # Content grouping
                metadata['content_group'] = self._get_tag_value(tags, ['contentgroup', 'TIT1', 'CONTENTGROUP'])
                
                # Technical metadata
                metadata['encoder'] = self._get_tag_value(tags, ['encoder', 'TENC', 'ENCODER'])
                metadata['file_type'] = self._get_tag_value(tags, ['filetype', 'TFLT', 'FILETYPE'])
                metadata['playlist_delay'] = self._get_tag_value(tags, ['playlistdelay', 'TDLY', 'PLAYLISTDELAY'])
                metadata['recording_time'] = self._get_tag_value(tags, ['recordingtime', 'TDRC', 'RECORDINGTIME'])
                metadata['tempo'] = self._get_tag_value(tags, ['tempo', 'TBPM', 'TEMPO'])
                metadata['length'] = self._get_tag_value(tags, ['length', 'TLEN', 'LENGTH'])
                
                # ReplayGain metadata
                metadata['replaygain_track_gain'] = self._get_tag_value(tags, ['replaygain_track_gain', 'RGAD', 'REPLAYGAIN_TRACK_GAIN'])
                metadata['replaygain_album_gain'] = self._get_tag_value(tags, ['replaygain_album_gain', 'RGAD', 'REPLAYGAIN_ALBUM_GAIN'])
                metadata['replaygain_track_peak'] = self._get_tag_value(tags, ['replaygain_track_peak', 'RGAD', 'REPLAYGAIN_TRACK_PEAK'])
                metadata['replaygain_album_peak'] = self._get_tag_value(tags, ['replaygain_album_peak', 'RGAD', 'REPLAYGAIN_ALBUM_PEAK'])
                
                # MusicBrainz IDs
                metadata['musicbrainz_track_id'] = self._get_tag_value(tags, ['musicbrainz_trackid', 'TXXX', 'MUSICBRAINZ_TRACKID'])
                metadata['musicbrainz_artist_id'] = self._get_tag_value(tags, ['musicbrainz_artistid', 'TXXX', 'MUSICBRAINZ_ARTISTID'])
                metadata['musicbrainz_album_id'] = self._get_tag_value(tags, ['musicbrainz_albumid', 'TXXX', 'MUSICBRAINZ_ALBUMID'])
                metadata['musicbrainz_album_artist_id'] = self._get_tag_value(tags, ['musicbrainz_albumartistid', 'TXXX', 'MUSICBRAINZ_ALBUMARTISTID'])
                metadata['musicbrainz_release_group_id'] = self._get_tag_value(tags, ['musicbrainz_releasegroupid', 'TXXX', 'MUSICBRAINZ_RELEASEGROUPID'])
                metadata['musicbrainz_recording_id'] = self._get_tag_value(tags, ['musicbrainz_recordingid', 'TXXX', 'MUSICBRAINZ_RECORDINGID'])
                metadata['musicbrainz_work_id'] = self._get_tag_value(tags, ['musicbrainz_workid', 'TXXX', 'MUSICBRAINZ_WORKID'])
                
                # Custom tags
                metadata['custom_tags'] = {}
                for key, value in tags.items():
                    if key.startswith('TXXX') or key.startswith('custom'):
                        metadata['custom_tags'][key] = value
            
            # Extract audio properties
            if hasattr(audio_file, 'info'):
                info = audio_file.info
                metadata['duration'] = getattr(info, 'length', None)
                metadata['bitrate'] = getattr(info, 'bitrate', None)
                metadata['sample_rate'] = getattr(info, 'sample_rate', None)
                metadata['channels'] = getattr(info, 'channels', None)
            
            # Extract BPM from metadata if available
            bpm_from_metadata = self._extract_bpm_from_metadata(metadata)
            if bpm_from_metadata:
                metadata['bpm_from_metadata'] = bpm_from_metadata
            
            # Cache metadata
            self.db_manager.save_cache(cache_key, metadata, expires_hours=self.cache_expiry_hours)
            
            log_universal('DEBUG', 'Audio', f'Extracted metadata: {len(metadata)} fields from {os.path.basename(file_path)}')
            return metadata
            
        except Exception as e:
            log_universal('ERROR', 'Audio', f'Metadata extraction failed for {os.path.basename(file_path)}: {e}')
            return None
    
    def _get_tag_value(self, tags, possible_keys):
        """Extract tag value from multiple possible keys."""
        for key in possible_keys:
            try:
                if key in tags:
                    value = tags[key]
                    if hasattr(value, '__len__') and len(value) > 0:
                        return str(value[0]) if isinstance(value, list) else str(value)
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
        Enrich metadata using external APIs.
        
        Args:
            metadata: Original metadata dictionary
            
        Returns:
            Enriched metadata dictionary
        """
        try:
            from .external_apis import get_metadata_enrichment_service
            
            enrichment_service = get_metadata_enrichment_service()
            if enrichment_service.is_available():
                log_universal('DEBUG', 'Audio', 'Enriching metadata with external APIs')
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
        
        Args:
            audio: Audio data as numpy array
            sample_rate: Sample rate in Hz
            metadata: Optional metadata dictionary
            
        Returns:
            Features dictionary or None on failure
        """
        features = {}
        
        try:
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
            
            # Extract MusiCNN features
            if self.extract_musicnn and TENSORFLOW_AVAILABLE:
                musicnn_features = self._extract_musicnn_features(audio, sample_rate)
                features.update(musicnn_features)
            
            # Extract chroma features
            if self.extract_chroma:
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
        """Extract rhythm-related features."""
        features = {}
        
        try:
            if ESSENTIA_AVAILABLE:
                # Use Essentia for rhythm analysis
                rhythm_extractor = es.RhythmExtractor2013()
                rhythm_result = rhythm_extractor(audio)
                
                features['bpm'] = float(rhythm_result[0])
                features['rhythm_confidence'] = float(rhythm_result[1])
                features['bpm_estimates'] = rhythm_result[2].tolist() if len(rhythm_result) > 2 else []
                features['bpm_intervals'] = rhythm_result[3].tolist() if len(rhythm_result) > 3 else []
                
            elif LIBROSA_AVAILABLE:
                # Use librosa as fallback
                tempo, beats = librosa.beat.beat_track(y=audio, sr=sample_rate)
                features['bpm'] = float(tempo)
                features['rhythm_confidence'] = 0.5  # Default confidence for librosa
                features['bpm_estimates'] = [tempo]
                features['bpm_intervals'] = []
            
            log_universal('DEBUG', 'Audio', f'Extracted rhythm features: BPM={features.get("bpm")}')
            
        except Exception as e:
            log_universal('WARNING', 'Audio', f'Rhythm feature extraction failed: {e}')
            features['bpm'] = None
            features['rhythm_confidence'] = None
        
        return features
    
    def _extract_spectral_features(self, audio: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Extract spectral features."""
        features = {}
        
        try:
            if ESSENTIA_AVAILABLE:
                # Spectral centroid
                centroid = es.SpectralCentroid()
                spectral_centroid = centroid(audio)
                features['spectral_centroid'] = float(np.mean(spectral_centroid))
                
                # Spectral rolloff
                rolloff = es.SpectralRolloff()
                spectral_rolloff = rolloff(audio)
                features['spectral_rolloff'] = float(np.mean(spectral_rolloff))
                
                # Spectral flatness
                flatness = es.SpectralFlatness()
                spectral_flatness = flatness(audio)
                features['spectral_flatness'] = float(np.mean(spectral_flatness))
                
                # Spectral bandwidth
                bandwidth = es.SpectralBandwidth()
                spectral_bandwidth = bandwidth(audio)
                features['spectral_bandwidth'] = float(np.mean(spectral_bandwidth))
                
                # Spectral contrast
                contrast = es.SpectralContrast()
                spectral_contrast = contrast(audio)
                features['spectral_contrast_mean'] = float(np.mean(spectral_contrast))
                features['spectral_contrast_std'] = float(np.std(spectral_contrast))
                
            elif LIBROSA_AVAILABLE:
                # Use librosa as fallback
                spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)
                features['spectral_centroid'] = float(np.mean(spectral_centroids))
                
                spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)
                features['spectral_rolloff'] = float(np.mean(spectral_rolloff))
                
                # Simplified features for librosa
                features['spectral_flatness'] = 0.0
                features['spectral_bandwidth'] = 0.0
                features['spectral_contrast_mean'] = 0.0
                features['spectral_contrast_std'] = 0.0
            
            log_universal('DEBUG', 'Audio', f'Extracted spectral features')
            
        except Exception as e:
            log_universal('WARNING', 'Audio', f'Spectral feature extraction failed: {e}')
            features.update({
                'spectral_centroid': None,
                'spectral_rolloff': None,
                'spectral_flatness': None,
                'spectral_bandwidth': None,
                'spectral_contrast_mean': None,
                'spectral_contrast_std': None
            })
        
        return features
    
    def _extract_loudness_features(self, audio: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Extract loudness features."""
        features = {}
        
        try:
            if ESSENTIA_AVAILABLE:
                # Loudness
                loudness = es.Loudness()
                loudness_value = loudness(audio)
                features['loudness'] = float(loudness_value)
                
                # Loudness range
                loudness_range = es.LoudnessRange()
                loudness_range_value = loudness_range(audio)
                features['loudness_range'] = float(loudness_range_value)
                
                # Dynamic complexity
                dynamic_complexity = es.DynamicComplexity()
                dynamic_complexity_value = dynamic_complexity(audio)
                features['dynamic_complexity'] = float(dynamic_complexity_value)
                
                # Dynamic range
                dynamic_range = es.DynamicRange()
                dynamic_range_value = dynamic_range(audio)
                features['dynamic_range'] = float(dynamic_range_value)
                
            elif LIBROSA_AVAILABLE:
                # Use librosa as fallback
                rms = librosa.feature.rms(y=audio)
                features['loudness'] = float(np.mean(rms))
                features['loudness_range'] = float(np.std(rms))
                features['dynamic_complexity'] = 0.0
                features['dynamic_range'] = 0.0
            
            log_universal('DEBUG', 'Audio', f'Extracted loudness features')
            
        except Exception as e:
            log_universal('WARNING', 'Audio', f'Loudness feature extraction failed: {e}')
            features.update({
                'loudness': None,
                'loudness_range': None,
                'dynamic_complexity': None,
                'dynamic_range': None
            })
        
        return features
    
    def _extract_key_features(self, audio: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Extract key and mode features."""
        features = {}
        
        try:
            if ESSENTIA_AVAILABLE:
                # Key detection
                key_detector = es.Key()
                key_result = key_detector(audio)
                features['key'] = key_result[0]
                features['mode'] = key_result[1]
                features['key_strength'] = float(key_result[2])
                features['key_confidence'] = float(key_result[3])
                
            elif LIBROSA_AVAILABLE:
                # Use librosa as fallback
                chroma = librosa.feature.chroma_cqt(y=audio, sr=sample_rate)
                key_detector = librosa.feature.key_mode(chroma)
                features['key'] = key_detector[0]
                features['mode'] = key_detector[1]
                features['key_strength'] = 0.5
                features['key_confidence'] = 0.5
            
            log_universal('DEBUG', 'Audio', f'Extracted key features: {features.get("key")} {features.get("mode")}')
            
        except Exception as e:
            log_universal('WARNING', 'Audio', f'Key feature extraction failed: {e}')
            features.update({
                'key': None,
                'mode': None,
                'key_strength': None,
                'key_confidence': None
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
                # This would require the MusiCNN model
                # For now, we'll skip this as it requires additional setup
                features['embedding'] = []
                features['tags'] = {}
                log_universal('DEBUG', 'Audio', 'MusiCNN features not implemented (requires model setup)')
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
        """Extract chroma features."""
        features = {}
        
        try:
            if ESSENTIA_AVAILABLE:
                # Chroma
                chroma = es.Chromagram()
                chroma_result = chroma(audio)
                features['chroma_mean'] = np.mean(chroma_result, axis=1).tolist()
                features['chroma_std'] = np.std(chroma_result, axis=1).tolist()
                
            elif LIBROSA_AVAILABLE:
                # Use librosa as fallback
                chroma = librosa.feature.chroma_cqt(y=audio, sr=sample_rate)
                features['chroma_mean'] = np.mean(chroma, axis=1).tolist()
                features['chroma_std'] = np.std(chroma, axis=1).tolist()
            
            log_universal('DEBUG', 'Audio', f'Extracted chroma features')
            
        except Exception as e:
            log_universal('WARNING', 'Audio', f'Chroma feature extraction failed: {e}')
            features.update({
                'chroma_mean': [],
                'chroma_std': []
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
            
            # Consider it long if estimated duration > 20 minutes
            is_long = estimated_duration_minutes > 20
            
            log_universal('DEBUG', 'Audio', f'File {os.path.basename(file_path)}: {estimated_duration_minutes:.1f} minutes estimated, long_audio: {is_long}')
            
            return is_long
            
        except Exception as e:
            log_universal('WARNING', 'Audio', f'Could not determine if {file_path} is long audio: {e}')
            return False


def get_audio_analyzer(config: Dict[str, Any] = None) -> 'AudioAnalyzer':
    """Get a configured audio analyzer instance."""
    return AudioAnalyzer(config)
