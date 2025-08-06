"""
Service implementations for Playlist Generator.
Audio analysis and metadata enrichment services.
"""

import os
# Configure TensorFlow logging BEFORE any imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Hide INFO and WARNING, show only ERROR
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimization messages
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU to avoid GPU-related warnings
import time
import hashlib
import numpy as np
from typing import List, Optional, Dict, Any
from pathlib import Path

from ..domain.interfaces import IAudioAnalyzer, IMetadataEnrichmentService
from ..domain.entities import Track, AnalysisResult, TrackMetadata
from ..domain.exceptions import AnalysisFailedException, MetadataEnrichmentException


class EssentiaAudioAnalyzer(IAudioAnalyzer):
    """Essentia-based audio analyzer implementation."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.sample_rate = self.config.get('sample_rate', 44100)
        self.hop_size = self.config.get('hop_size', 512)
        self.frame_size = self.config.get('frame_size', 2048)
        
        # Check for required libraries
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check if required audio libraries are available."""
        try:
            import essentia.standard as es
            self.essentia_available = True
        except ImportError:
            self.essentia_available = False
        
        try:
            import librosa
            self.librosa_available = True
        except ImportError:
            self.librosa_available = False
        
        # Don't raise exception during initialization, just set availability
        # This allows the service to be created even without dependencies
    
    def analyze_track(self, track: Track) -> AnalysisResult:
        """Analyze a track and return analysis result."""
        if not self.is_available():
            raise AnalysisFailedException("No audio analysis libraries available")
        
        try:
            # Load audio
            audio, sample_rate = self._load_audio(track.path)
            if audio is None:
                raise AnalysisFailedException(f"Failed to load audio file: {track.path}")
            
            # Extract features
            features = self._extract_features(audio, sample_rate)
            
            # Calculate confidence based on feature quality
            confidence = self._calculate_confidence(features)
            
            return AnalysisResult(
                features=features,
                confidence=confidence
            )
            
        except Exception as e:
            raise AnalysisFailedException(f"Analysis failed for {track.path}: {str(e)}")
    
    def is_available(self) -> bool:
        """Check if analyzer is available."""
        return self.essentia_available or self.librosa_available
    
    def get_supported_formats(self) -> List[str]:
        """Get supported audio formats."""
        return ['.mp3', '.flac', '.wav', '.m4a', '.ogg']
    
    def _load_audio(self, file_path: str) -> tuple:
        """Load audio file using available libraries."""
        if self.essentia_available:
            return self._load_with_essentia(file_path)
        elif self.librosa_available:
            return self._load_with_librosa(file_path)
        else:
            return None, None
    
    def _load_with_essentia(self, file_path: str) -> tuple:
        """Load audio using multi-segment approach for consistency."""
        try:
            from ..core.audio_analyzer import extract_multiple_segments
            audio, sample_rate = extract_multiple_segments(
                file_path,
                self.sample_rate,
                {'OPTIMIZED_SEGMENT_DURATION_SECONDS': 60},
                'service'
            )
            return audio, sample_rate
        except Exception as e:
            return None, None
    
    def _load_with_librosa(self, file_path: str) -> tuple:
        """Load audio using multi-segment approach for consistency."""
        try:
            from ..core.audio_analyzer import extract_multiple_segments
            audio, sample_rate = extract_multiple_segments(
                file_path,
                self.sample_rate,
                {'OPTIMIZED_SEGMENT_DURATION_SECONDS': 60},
                'service'
            )
            return audio, sample_rate
        except Exception as e:
            return None, None
    
    def _extract_features(self, audio: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Extract audio features."""
        features = {}
        
        if self.essentia_available:
            features.update(self._extract_essentia_features(audio, sample_rate))
        elif self.librosa_available:
            features.update(self._extract_librosa_features(audio, sample_rate))
        
        return features
    
    def _extract_essentia_features(self, audio: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Extract features using Essentia."""
        import essentia.standard as es
        
        # Convert audio for Essentia compatibility
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        features = {}
        
        # Rhythm features
        try:
            rhythm_extractor = es.RhythmExtractor2013()
            bpm, confidence, ticks, estimates, bpm_intervals = rhythm_extractor(audio)
            features['bpm'] = float(bpm)
            features['rhythm_confidence'] = float(confidence)
        except:
            features['bpm'] = 0.0
            features['rhythm_confidence'] = 0.0
        
        # Spectral features
        try:
            spectral_centroid = es.Centroid()
            features['spectral_centroid'] = float(spectral_centroid(audio))
        except:
            features['spectral_centroid'] = 0.0
        
        # Loudness features
        try:
            loudness = es.Loudness()
            features['loudness'] = float(loudness(audio))
        except:
            features['loudness'] = 0.0
        
        return features
    
    def _extract_librosa_features(self, audio: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Extract features using Librosa."""
        import librosa
        
        features = {}
        
        # Rhythm features
        try:
            tempo, _ = librosa.beat.beat_track(y=audio, sr=sample_rate)
            features['bpm'] = float(tempo)
            features['rhythm_confidence'] = 0.8  # Default confidence
        except:
            features['bpm'] = 0.0
            features['rhythm_confidence'] = 0.0
        
        # Spectral features
        try:
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)
            features['spectral_centroid'] = float(np.mean(spectral_centroids))
        except:
            features['spectral_centroid'] = 0.0
        
        # Loudness features
        try:
            rms = librosa.feature.rms(y=audio)
            features['loudness'] = float(np.mean(rms))
        except:
            features['loudness'] = 0.0
        
        return features
    
    def _calculate_confidence(self, features: Dict[str, Any]) -> float:
        """Calculate analysis confidence based on feature quality."""
        confidence = 0.0
        
        # Check if we have basic features
        if features.get('bpm', 0) > 0:
            confidence += 0.3
        
        if features.get('spectral_centroid', 0) > 0:
            confidence += 0.3
        
        if features.get('loudness', 0) > 0:
            confidence += 0.2
        
        # Add rhythm confidence if available
        rhythm_confidence = features.get('rhythm_confidence', 0)
        confidence += rhythm_confidence * 0.2
        
        return min(confidence, 1.0)


class MusicBrainzEnrichmentService(IMetadataEnrichmentService):
    """MusicBrainz metadata enrichment service."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.user_agent = self.config.get('musicbrainz_user_agent', 'playlista/1.0')
        self.enabled = self.config.get('musicbrainz_enabled', True)
        
        # Check for required libraries
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check if MusicBrainz library is available."""
        try:
            import musicbrainzngs
            self.musicbrainz_available = True
            musicbrainzngs.set_useragent("playlista", "1.0", "https://github.com/playlista")
        except ImportError:
            self.musicbrainz_available = False
    
    def enrich_metadata(self, track: Track) -> Track:
        """Enrich track metadata with MusicBrainz data."""
        if not self.is_available():
            return track
        
        try:
            # Search for track in MusicBrainz
            mb_data = self._search_track(track.metadata.title, track.metadata.artist)
            
            if mb_data:
                # Update metadata with enriched data
                track.metadata.album = mb_data.get('album', track.metadata.album)
                track.metadata.year = mb_data.get('year', track.metadata.year)
                track.metadata.genre = mb_data.get('genre', track.metadata.genre)
                
                # Add tags
                for tag in mb_data.get('tags', []):
                    track.metadata.add_tag(tag)
            
            return track
            
        except Exception as e:
            # Return original track if enrichment fails
            return track
    
    def is_available(self) -> bool:
        """Check if enrichment service is available."""
        return self.enabled and self.musicbrainz_available
    
    def _search_track(self, title: str, artist: str) -> Optional[Dict[str, Any]]:
        """Search for track in MusicBrainz."""
        import musicbrainzngs
        
        try:
            # Search for recording
            result = musicbrainzngs.search_recordings(
                query=f"{title} AND artist:{artist}",
                limit=1
            )
            
            if result.get('recording_list'):
                recording = result['recording_list'][0]
                
                # Extract basic info
                mb_data = {
                    'title': recording.get('title', title),
                    'artist': recording.get('artist-credit-phrase', artist),
                    'album': None,
                    'year': None,
                    'genre': None,
                    'tags': []
                }
                
                # Get release info if available
                if recording.get('release_list'):
                    release = recording['release_list'][0]
                    mb_data['album'] = release.get('title')
                    
                    # Get release date
                    if release.get('date'):
                        try:
                            mb_data['year'] = int(release['date'][:4])
                        except:
                            pass
                
                # Get tags
                if recording.get('tag_list'):
                    mb_data['tags'] = [tag['name'] for tag in recording['tag_list']]
                
                return mb_data
            
        except Exception as e:
            return None
        
        return None 