import logging
import numpy as np
import os
from typing import Dict, Any, Optional, List
import requests
import json

logger = logging.getLogger(__name__)


class AdvancedAudioModels:
    """Advanced audio analysis models for enhanced playlist generation."""
    
    def __init__(self, cache_dir: str = '/app/cache'):
        self.cache_dir = cache_dir
        self.models_dir = os.path.join(cache_dir, 'models')
        os.makedirs(self.models_dir, exist_ok=True)
        
    def extract_emotional_features(self, audio_path: str) -> Dict[str, Any]:
        """Extract emotional and aesthetic features from audio."""
        features = {}
        
        try:
            # Extract valence and arousal using spectral features
            features.update(self._extract_valence_arousal(audio_path))
            
            # Extract mood classification
            features.update(self._extract_mood_classification(audio_path))
            
            # Extract energy level
            features.update(self._extract_energy_level(audio_path))
            
            # Extract complexity score
            features.update(self._extract_complexity_score(audio_path))
            
            logger.info(f"Extracted emotional features for {os.path.basename(audio_path)}")
            return features
            
        except Exception as e:
            logger.warning(f"Emotional feature extraction failed: {str(e)}")
            return self._get_default_emotional_features()
    
    def _extract_valence_arousal(self, audio_path: str) -> Dict[str, float]:
        """Extract valence (positive/negative) and arousal (calm/excited) from audio."""
        try:
            import essentia.standard as es
            
            # Load audio
            audio = es.MonoLoader(filename=audio_path)()
            
            # Extract spectral features for emotion analysis
            spectral_centroid = es.SpectralCentroid()
            spectral_rolloff = es.SpectralRolloff()
            spectral_flatness = es.SpectralFlatness()
            
            # Calculate frame-by-frame features
            frame_size = 2048
            hop_size = 1024
            window = es.Windowing(type='hann')
            spectrum = es.Spectrum()
            
            centroids = []
            rolloffs = []
            flatnesses = []
            
            for frame in es.FrameGenerator(audio, frameSize=frame_size, hopSize=hop_size):
                spec = spectrum(window(frame))
                centroids.append(spectral_centroid(spec))
                rolloffs.append(spectral_rolloff(spec))
                flatnesses.append(spectral_flatness(spec))
            
            # Calculate statistics
            centroid_mean = np.mean(centroids)
            rolloff_mean = np.mean(rolloffs)
            flatness_mean = np.mean(flatnesses)
            
            # Map to valence and arousal (simplified mapping)
            # Higher centroid and rolloff = more positive valence
            valence = min(1.0, max(0.0, (centroid_mean / 5000 + rolloff_mean / 8000) / 2))
            
            # Higher flatness = lower arousal (more noise = calmer)
            arousal = min(1.0, max(0.0, 1.0 - flatness_mean))
            
            return {
                'valence': float(valence),
                'arousal': float(arousal)
            }
            
        except Exception as e:
            logger.warning(f"Valence/arousal extraction failed: {str(e)}")
            return {'valence': 0.5, 'arousal': 0.5}
    
    def _extract_mood_classification(self, audio_path: str) -> Dict[str, Any]:
        """Classify mood based on audio features."""
        try:
            # Get basic features first
            from .feature_extractor import AudioAnalyzer
            analyzer = AudioAnalyzer()
            features, _, _ = analyzer.extract_features(audio_path)
            
            if not features:
                return self._get_default_mood_features()
            
            # Classify mood based on features
            bpm = features.get('bpm', 120)
            danceability = features.get('danceability', 0.5)
            centroid = features.get('centroid', 2000)
            loudness = features.get('loudness', -20)
            
            # Mood classification logic
            mood_scores = {
                'happy': 0.0,
                'sad': 0.0,
                'energetic': 0.0,
                'calm': 0.0,
                'aggressive': 0.0,
                'melancholic': 0.0
            }
            
            # Happy: high BPM, high danceability, bright timbre
            if bpm > 120 and danceability > 0.6 and centroid > 3000:
                mood_scores['happy'] += 0.4
            if bpm > 140 and danceability > 0.7:
                mood_scores['happy'] += 0.3
            
            # Sad: low BPM, low danceability, dark timbre
            if bpm < 80 and danceability < 0.4 and centroid < 2000:
                mood_scores['sad'] += 0.4
            if bpm < 70 and loudness < -25:
                mood_scores['sad'] += 0.3
            
            # Energetic: high BPM, high loudness
            if bpm > 130 and loudness > -15:
                mood_scores['energetic'] += 0.5
            if bpm > 150:
                mood_scores['energetic'] += 0.3
            
            # Calm: low BPM, low loudness
            if bpm < 90 and loudness < -20:
                mood_scores['calm'] += 0.5
            if bpm < 70 and danceability < 0.3:
                mood_scores['calm'] += 0.3
            
            # Aggressive: high loudness, high centroid
            if loudness > -10 and centroid > 4000:
                mood_scores['aggressive'] += 0.4
            if bpm > 140 and loudness > -15:
                mood_scores['aggressive'] += 0.3
            
            # Melancholic: low BPM, low centroid, moderate loudness
            if bpm < 85 and centroid < 2500 and -25 < loudness < -15:
                mood_scores['melancholic'] += 0.4
            
            # Normalize scores
            total_score = sum(mood_scores.values())
            if total_score > 0:
                mood_scores = {k: v / total_score for k, v in mood_scores.items()}
            
            # Get primary mood
            primary_mood = max(mood_scores.items(), key=lambda x: x[1])
            
            return {
                'mood_scores': mood_scores,
                'primary_mood': primary_mood[0],
                'mood_confidence': primary_mood[1]
            }
            
        except Exception as e:
            logger.warning(f"Mood classification failed: {str(e)}")
            return self._get_default_mood_features()
    
    def _extract_energy_level(self, audio_path: str) -> Dict[str, float]:
        """Extract energy level from audio features."""
        try:
            from .feature_extractor import AudioAnalyzer
            analyzer = AudioAnalyzer()
            features, _, _ = analyzer.extract_features(audio_path)
            
            if not features:
                return {'energy_level': 0.5}
            
            bpm = features.get('bpm', 120)
            danceability = features.get('danceability', 0.5)
            loudness = features.get('loudness', -20)
            centroid = features.get('centroid', 2000)
            
            # Calculate energy level (0-1)
            energy_factors = []
            
            # BPM factor (0-1)
            bpm_factor = min(1.0, max(0.0, (bpm - 60) / 120))
            energy_factors.append(bpm_factor)
            
            # Danceability factor
            energy_factors.append(danceability)
            
            # Loudness factor (normalize from -60 to 0 dB)
            loudness_factor = min(1.0, max(0.0, (loudness + 60) / 60))
            energy_factors.append(loudness_factor)
            
            # Spectral centroid factor
            centroid_factor = min(1.0, max(0.0, centroid / 8000))
            energy_factors.append(centroid_factor)
            
            # Average energy level
            energy_level = np.mean(energy_factors)
            
            return {'energy_level': float(energy_level)}
            
        except Exception as e:
            logger.warning(f"Energy level extraction failed: {str(e)}")
            return {'energy_level': 0.5}
    
    def _extract_complexity_score(self, audio_path: str) -> Dict[str, float]:
        """Calculate musical complexity score."""
        try:
            from .feature_extractor import AudioAnalyzer
            analyzer = AudioAnalyzer()
            features, _, _ = analyzer.extract_features(audio_path)
            
            if not features:
                return {'complexity_score': 0.5}
            
            # Factors that contribute to complexity
            complexity_factors = []
            
            # Onset rate (more onsets = more complex)
            onset_rate = features.get('onset_rate', 0)
            onset_factor = min(1.0, onset_rate / 10)  # Normalize
            complexity_factors.append(onset_factor)
            
            # Zero crossing rate (more crossings = more complex)
            zcr = features.get('zcr', 0)
            zcr_factor = min(1.0, zcr / 0.5)  # Normalize
            complexity_factors.append(zcr_factor)
            
            # Spectral contrast (higher contrast = more complex)
            spectral_contrast = features.get('spectral_contrast', 0)
            contrast_factor = min(1.0, spectral_contrast / 10)  # Normalize
            complexity_factors.append(contrast_factor)
            
            # BPM variation (higher BPM = potentially more complex)
            bpm = features.get('bpm', 120)
            bpm_factor = min(1.0, max(0.0, (bpm - 60) / 120))
            complexity_factors.append(bpm_factor)
            
            # Calculate average complexity
            complexity_score = np.mean(complexity_factors)
            
            return {'complexity_score': float(complexity_score)}
            
        except Exception as e:
            logger.warning(f"Complexity score extraction failed: {str(e)}")
            return {'complexity_score': 0.5}
    
    def extract_spotify_features(self, artist: str, title: str) -> Dict[str, Any]:
        """Extract features from Spotify API (requires API key)."""
        spotify_api_key = os.getenv('SPOTIFY_API_KEY')
        if not spotify_api_key:
            logger.debug("SPOTIFY_API_KEY not set, skipping Spotify features")
            return {}
        
        try:
            # Search for track
            search_url = "https://api.spotify.com/v1/search"
            headers = {
                'Authorization': f'Bearer {spotify_api_key}',
                'Content-Type': 'application/json'
            }
            params = {
                'q': f'artist:{artist} track:{title}',
                'type': 'track',
                'limit': 1
            }
            
            response = requests.get(search_url, headers=headers, params=params)
            if response.status_code != 200:
                logger.warning(f"Spotify API error: {response.status_code}")
                return {}
            
            data = response.json()
            if not data.get('tracks', {}).get('items'):
                logger.debug(f"No Spotify track found for {artist} - {title}")
                return {}
            
            track = data['tracks']['items'][0]
            track_id = track['id']
            
            # Get audio features
            features_url = f"https://api.spotify.com/v1/audio-features/{track_id}"
            features_response = requests.get(features_url, headers=headers)
            
            if features_response.status_code == 200:
                features = features_response.json()
                return {
                    'spotify_acousticness': features.get('acousticness'),
                    'spotify_danceability': features.get('danceability'),
                    'spotify_energy': features.get('energy'),
                    'spotify_instrumentalness': features.get('instrumentalness'),
                    'spotify_liveness': features.get('liveness'),
                    'spotify_loudness': features.get('loudness'),
                    'spotify_speechiness': features.get('speechiness'),
                    'spotify_tempo': features.get('tempo'),
                    'spotify_valence': features.get('valence'),
                    'spotify_popularity': track.get('popularity')
                }
            
        except Exception as e:
            logger.warning(f"Spotify feature extraction failed: {str(e)}")
        
        return {}
    
    def _get_default_emotional_features(self) -> Dict[str, Any]:
        """Return default emotional features when extraction fails."""
        return {
            'valence': 0.5,
            'arousal': 0.5,
            'mood_scores': {
                'happy': 0.2,
                'sad': 0.2,
                'energetic': 0.2,
                'calm': 0.2,
                'aggressive': 0.1,
                'melancholic': 0.1
            },
            'primary_mood': 'happy',
            'mood_confidence': 0.2,
            'energy_level': 0.5,
            'complexity_score': 0.5
        }
    
    def _get_default_mood_features(self) -> Dict[str, Any]:
        """Return default mood features when classification fails."""
        return {
            'mood_scores': {
                'happy': 0.2,
                'sad': 0.2,
                'energetic': 0.2,
                'calm': 0.2,
                'aggressive': 0.1,
                'melancholic': 0.1
            },
            'primary_mood': 'happy',
            'mood_confidence': 0.2
        } 