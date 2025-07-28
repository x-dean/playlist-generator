#!/usr/bin/env python3
"""
Core audio analyzer for extracting features from audio files.
"""

import os
import logging
import time
import numpy as np
import essentia.standard as es
import essentia
import librosa
from typing import Dict, Any, Optional, Tuple, List
from functools import wraps
import signal

from .database_manager import AudioDatabaseManager
from .metadata_enricher import MetadataEnricher
from .feature_validator import FeatureValidator
from utils.path_converter import PathConverter

logger = logging.getLogger(__name__)


def timeout(seconds=60, error_message="Processing timed out"):
    """Decorator to add timeout to functions."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            def _handle_timeout(signum, frame):
                raise TimeoutError(error_message)
            
            # Set the signal handler and a 5-second alarm
            old_handler = signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
            return result
        return wrapper
    return decorator


def safe_essentia_call(func, *args, **kwargs):
    """Safely call Essentia functions with error handling."""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger.warning(f"Essentia call failed: {e}")
        return None


class AudioAnalyzer:
    """Core audio analyzer for extracting features from audio files."""
    
    VERSION = "5.0.0"
    
    def __init__(self, cache_file: str = None, library: str = None, music: str = None):
        """Initialize the audio analyzer.
        
        Args:
            cache_file (str, optional): Path to cache database. Defaults to None.
            library (str, optional): Host library path. Defaults to None.
            music (str, optional): Container music path. Defaults to None.
        """
        self.cache_file = cache_file or os.getenv('CACHE_FILE', '/app/cache/audio_analysis.db')
        self.library = library or os.getenv('HOST_LIBRARY_PATH', '/root/music/library')
        self.music = music or os.getenv('MUSIC_PATH', '/music')
        
        # Ensure cache directory exists
        cache_dir = os.path.dirname(self.cache_file)
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
            logger.info(f"Created cache directory: {cache_dir}")
        
        # Initialize components
        self.db_manager = AudioDatabaseManager(self.cache_file)
        self.metadata_enricher = MetadataEnricher()
        self.feature_validator = FeatureValidator()
        self.path_converter = PathConverter(self.library, self.music)
        
        # Check TensorFlow support
        self._check_tensorflow_support()
    
    def _check_tensorflow_support(self):
        """Check if TensorFlow support is available for MusiCNN."""
        try:
            from essentia.standard import TensorflowPredictMusiCNN
            self.tensorflow_available = True
            logger.info("TensorFlow support: AVAILABLE")
        except ImportError:
            self.tensorflow_available = False
            logger.warning("TensorFlow support: NOT AVAILABLE - MusiCNN embeddings will not work")
    
    def _get_file_info(self, filepath: str) -> Dict[str, Any]:
        """Get file information including metadata.
        
        Args:
            filepath (str): Path to the audio file.
            
        Returns:
            Dict[str, Any]: File information dictionary.
        """
        try:
            # Get basic file info
            stat = os.stat(filepath)
            file_size = stat.st_size
            
            # Extract metadata using mutagen
            from mutagen import File as MutagenFile
            audio = MutagenFile(filepath)
            
            metadata = {}
            if audio:
                # Extract basic metadata
                if hasattr(audio, 'tags') and audio.tags:
                    tags = audio.tags
                    metadata.update({
                        'artist': str(tags.get('artist', ['Unknown Artist'])[0]) if 'artist' in tags else 'Unknown Artist',
                        'title': str(tags.get('title', ['Unknown Title'])[0]) if 'title' in tags else 'Unknown Title',
                        'album': str(tags.get('album', ['Unknown Album'])[0]) if 'album' in tags else 'Unknown Album',
                        'year': str(tags.get('date', [''])[0]) if 'date' in tags else '',
                        'genre': str(tags.get('genre', [''])[0]) if 'genre' in tags else '',
                        'tracknumber': str(tags.get('tracknumber', [''])[0]) if 'tracknumber' in tags else '',
                        'discnumber': str(tags.get('discnumber', [''])[0]) if 'discnumber' in tags else '',
                    })
            
            # Normalize filepath to library path
            normalized_path = self.path_converter.normalize_to_library_path(filepath)
            
            return {
                'filepath': normalized_path,
                'file_size': file_size,
                'metadata': metadata,
                'file_hash': self._get_file_hash(filepath)
            }
            
        except Exception as e:
            logger.error(f"Error getting file info for {filepath}: {e}")
            return {
                'filepath': filepath,
                'file_size': 0,
                'metadata': {'artist': 'Unknown Artist', 'title': 'Unknown Title'},
                'file_hash': ''
            }
    
    def _get_file_hash(self, filepath: str) -> str:
        """Get file hash for caching.
        
        Args:
            filepath (str): Path to the file.
            
        Returns:
            str: File hash.
        """
        try:
            import hashlib
            hash_md5 = hashlib.md5()
            with open(filepath, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            logger.warning(f"Could not generate hash for {filepath}: {e}")
            return ""
    
    def _safe_audio_load(self, audio_path: str) -> Optional[np.ndarray]:
        """Safely load audio file.
        
        Args:
            audio_path (str): Path to the audio file.
            
        Returns:
            Optional[np.ndarray]: Audio data or None if failed.
        """
        try:
            # Try loading with librosa first
            audio, sr = librosa.load(audio_path, sr=None, mono=True)
            return audio
        except Exception as e:
            logger.warning(f"Librosa load failed for {audio_path}: {e}")
            try:
                # Fallback to Essentia
                loader = es.MonoLoader(filename=audio_path)
                audio = loader()
                return audio
            except Exception as e2:
                logger.error(f"Essentia load also failed for {audio_path}: {e2}")
                return None
    
    @timeout(300)  # 5 minute timeout
    def _extract_rhythm_features(self, audio: np.ndarray, audio_path: str = None) -> Dict[str, Any]:
        """Extract rhythm-related features.
        
        Args:
            audio (np.ndarray): Audio data.
            audio_path (str, optional): Path to audio file. Defaults to None.
            
        Returns:
            Dict[str, Any]: Rhythm features.
        """
        features = {}
        
        try:
            # BPM detection
            rhythm_extractor = es.RhythmExtractor2013()
            bpm, beats, beats_confidence, _, _ = rhythm_extractor(audio)
            features['bpm'] = float(bpm)
            features['beats_confidence'] = float(beats_confidence)
            
            # Onset rate
            onset_detector = es.OnsetDetection()
            onset_rate = len(onset_detector(audio)) / (len(audio) / 44100)  # Approximate
            features['onset_rate'] = float(onset_rate)
            
        except Exception as e:
            logger.warning(f"Rhythm feature extraction failed: {e}")
            features['bpm'] = 120.0  # Default
            features['onset_rate'] = 0.0
        
        return features
    
    @timeout(300)
    def _extract_spectral_features(self, audio: np.ndarray) -> Dict[str, Any]:
        """Extract spectral features.
        
        Args:
            audio (np.ndarray): Audio data.
            
        Returns:
            Dict[str, Any]: Spectral features.
        """
        features = {}
        
        try:
            # Spectral centroid
            spectral_centroid = es.SpectralCentroid()
            centroid = spectral_centroid(audio)
            features['spectral_centroid'] = float(np.mean(centroid))
            
            # Spectral rolloff
            spectral_rolloff = es.SpectralRolloff()
            rolloff = spectral_rolloff(audio)
            features['spectral_rolloff'] = float(np.mean(rolloff))
            
            # Spectral flatness
            spectral_flatness = es.SpectralFlatness()
            flatness = spectral_flatness(audio)
            features['spectral_flatness'] = float(np.mean(flatness))
            
            # Zero crossing rate
            zcr = es.ZeroCrossingRate()
            zero_crossing_rate = zcr(audio)
            features['zcr'] = float(np.mean(zero_crossing_rate))
            
        except Exception as e:
            logger.warning(f"Spectral feature extraction failed: {e}")
        
        return features
    
    @timeout(300)
    def _extract_mfcc(self, audio: np.ndarray, num_coeffs: int = 13) -> Dict[str, Any]:
        """Extract MFCC features.
        
        Args:
            audio (np.ndarray): Audio data.
            num_coeffs (int): Number of MFCC coefficients. Defaults to 13.
            
        Returns:
            Dict[str, Any]: MFCC features.
        """
        features = {}
        
        try:
            mfcc = es.MFCC(numberCoefficients=num_coeffs)
            mfcc_coeffs, mfcc_bands = mfcc(audio)
            features['mfcc'] = mfcc_coeffs.mean(axis=0).tolist()
            
        except Exception as e:
            logger.warning(f"MFCC extraction failed: {e}")
        
        return features
    
    @timeout(300)
    def _extract_chroma(self, audio: np.ndarray) -> Dict[str, Any]:
        """Extract chroma features.
        
        Args:
            audio (np.ndarray): Audio data.
            
        Returns:
            Dict[str, Any]: Chroma features.
        """
        features = {}
        
        try:
            chroma = es.Chromagram()
            chroma_features = chroma(audio)
            features['chroma'] = chroma_features.mean(axis=0).tolist()
            
        except Exception as e:
            logger.warning(f"Chroma extraction failed: {e}")
        
        return features
    
    @timeout(300)
    def _extract_key(self, audio: np.ndarray) -> Dict[str, Any]:
        """Extract key and mode information.
        
        Args:
            audio (np.ndarray): Audio data.
            
        Returns:
            Dict[str, Any]: Key features.
        """
        features = {}
        
        try:
            key_extractor = es.Key()
            key, scale, key_strength = key_extractor(audio)
            features['key'] = key
            features['mode'] = scale
            features['key_confidence'] = float(key_strength)
            
        except Exception as e:
            logger.warning(f"Key extraction failed: {e}")
            features['key'] = 'C'
            features['mode'] = 'major'
            features['key_confidence'] = 0.0
        
        return features
    
    @timeout(300)
    def _extract_danceability(self, audio: np.ndarray) -> Dict[str, Any]:
        """Extract danceability and related features.
        
        Args:
            audio (np.ndarray): Audio data.
            
        Returns:
            Dict[str, Any]: Danceability features.
        """
        features = {}
        
        try:
            # Simple danceability estimation based on rhythm strength
            rhythm_extractor = es.RhythmExtractor2013()
            bpm, _, _, _, _ = rhythm_extractor(audio)
            
            # Estimate danceability based on BPM and rhythm strength
            if 90 <= bpm <= 150:
                danceability = 0.8
            elif 70 <= bpm <= 180:
                danceability = 0.6
            else:
                danceability = 0.4
            
            features['danceability'] = danceability
            features['energy'] = 0.7  # Placeholder
            features['valence'] = 0.5  # Placeholder
            features['acousticness'] = 0.3  # Placeholder
            features['instrumentalness'] = 0.5  # Placeholder
            features['liveness'] = 0.3  # Placeholder
            features['speechiness'] = 0.1  # Placeholder
            
        except Exception as e:
            logger.warning(f"Danceability extraction failed: {e}")
            features['danceability'] = 0.5
            features['energy'] = 0.5
            features['valence'] = 0.5
            features['acousticness'] = 0.5
            features['instrumentalness'] = 0.5
            features['liveness'] = 0.5
            features['speechiness'] = 0.1
        
        return features
    
    def extract_features(self, audio_path: str, force_reextract: bool = False) -> Optional[Tuple[Dict[str, Any], str, bool]]:
        """Extract features from an audio file.
        
        Args:
            audio_path (str): Path to the audio file.
            force_reextract (bool): Force re-extraction even if cached.
            
        Returns:
            Optional[Tuple[Dict[str, Any], str, bool]]: (features, filepath, success) or None.
        """
        try:
            # Get file info
            file_info = self._get_file_info(audio_path)
            
            # Check cache unless force re-extraction
            if not force_reextract:
                cached_features = self.db_manager.get_features(file_info['filepath'])
                if cached_features and not cached_features['failed']:
                    logger.debug(f"Using cached features for {audio_path}")
                    return cached_features['features'], file_info['filepath'], True
            
            # Load audio
            audio = self._safe_audio_load(audio_path)
            if audio is None:
                logger.error(f"Failed to load audio: {audio_path}")
                self.db_manager.mark_as_failed(file_info['filepath'])
                return None, file_info['filepath'], False
            
            # Extract features
            features = {}
            
            # Extract all feature types
            features.update(self._extract_rhythm_features(audio, audio_path))
            features.update(self._extract_spectral_features(audio))
            features.update(self._extract_mfcc(audio))
            features.update(self._extract_chroma(audio))
            features.update(self._extract_key(audio))
            features.update(self._extract_danceability(audio))
            
            # Enrich metadata
            if file_info['metadata'].get('artist') and file_info['metadata'].get('title'):
                enriched_metadata = self.metadata_enricher.enrich_metadata(
                    file_info['metadata']['artist'],
                    file_info['metadata']['title']
                )
                file_info['metadata'].update(enriched_metadata)
            
            # Validate features
            if not self.feature_validator.is_valid_feature_set(features):
                logger.warning(f"Feature validation failed for {audio_path}")
                self.db_manager.mark_as_failed(file_info['filepath'])
                return None, file_info['filepath'], False
            
            # Save to database
            success = self.db_manager.save_features(file_info, features)
            if not success:
                logger.error(f"Failed to save features for {audio_path}")
                return None, file_info['filepath'], False
            
            logger.info(f"Successfully extracted features for {audio_path}")
            return features, file_info['filepath'], True
            
        except Exception as e:
            logger.error(f"Feature extraction failed for {audio_path}: {e}")
            if 'file_info' in locals():
                self.db_manager.mark_as_failed(file_info['filepath'])
            return None, audio_path, False
    
    def get_all_features(self, include_failed: bool = False) -> List[Dict[str, Any]]:
        """Get all features from the database.
        
        Args:
            include_failed (bool): Whether to include failed files.
            
        Returns:
            List[Dict[str, Any]]: List of feature dictionaries.
        """
        return self.db_manager.get_all_features(include_failed)
    
    def cleanup_database(self) -> List[str]:
        """Clean up database by removing missing files.
        
        Returns:
            List[str]: List of removed file paths.
        """
        return self.db_manager.cleanup_database()
    
    def get_failed_files(self) -> List[str]:
        """Get list of failed files.
        
        Returns:
            List[str]: List of failed file paths.
        """
        return self.db_manager.get_failed_files()
    
    def get_files_needing_analysis(self) -> List[Tuple[str, Any]]:
        """Get list of files that need analysis.
        
        This method compares the current file system state with the database
        to determine which files need to be analyzed.
        
        Returns:
            List[Tuple[str, Any]]: List of (filepath, metadata) tuples that need analysis.
        """
        try:
            # Get file discovery changes
            new_files, modified_files, _ = self.db_manager.get_file_discovery_changes()
            
            # Combine new and modified files
            files_needing_analysis = []
            
            # Add new files with metadata
            for filepath in new_files:
                try:
                    file_info = self._get_file_info(filepath)
                    files_needing_analysis.append((filepath, file_info))
                except Exception as e:
                    logger.warning(f"Could not get file info for {filepath}: {e}")
                    files_needing_analysis.append((filepath, {}))
            
            # Add modified files with metadata
            for filepath in modified_files:
                try:
                    file_info = self._get_file_info(filepath)
                    files_needing_analysis.append((filepath, file_info))
                except Exception as e:
                    logger.warning(f"Could not get file info for {filepath}: {e}")
                    files_needing_analysis.append((filepath, {}))
            
            # Also include failed files that should be retried
            failed_files = self.db_manager.get_failed_files()
            for filepath in failed_files:
                try:
                    file_info = self._get_file_info(filepath)
                    files_needing_analysis.append((filepath, file_info))
                except Exception as e:
                    logger.warning(f"Could not get file info for failed file {filepath}: {e}")
                    files_needing_analysis.append((filepath, {}))
            
            return files_needing_analysis
            
        except Exception as e:
            logger.error(f"Failed to get files needing analysis: {e}")
            return []
    
    def get_file_sizes_from_db(self, file_paths: List[str]) -> Dict[str, int]:
        """Get file sizes from the database.
        
        Args:
            file_paths (List[str]): List of file paths.
            
        Returns:
            Dict[str, int]: Dictionary mapping file paths to their sizes in bytes.
        """
        return self.db_manager.get_file_sizes_from_db(file_paths)
    
    def _mark_failed(self, file_info: Dict[str, Any]) -> bool:
        """Mark a file as failed in the database.
        
        Args:
            file_info (Dict[str, Any]): File information dictionary.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            filepath = file_info.get('filepath', '')
            if filepath:
                return self.db_manager.mark_as_failed(filepath)
            return False
        except Exception as e:
            logger.error(f"Failed to mark file as failed: {e}")
            return False 