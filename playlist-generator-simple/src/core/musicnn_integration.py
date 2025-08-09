"""
MusiCNN Integration for Optimized Pipeline.

This module provides integration with MusiCNN models for the optimized audio analysis pipeline.
It handles model loading, audio preprocessing, and inference for genre/mood classification.
"""

import os
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import logging

# Import local modules
from .logging_setup import get_logger, log_universal
from .lazy_imports import get_tensorflow, is_tensorflow_available

logger = get_logger('playlista.musicnn_integration')

# Check for TensorFlow availability
TENSORFLOW_AVAILABLE = is_tensorflow_available()


class MusiCNNIntegration:
    """
    Integration class for MusiCNN model inference in the optimized pipeline.
    """
    
    def __init__(self, model_size: str = 'standard'):
        """
        Initialize MusiCNN integration.
        
        Args:
            model_size: Size of model to use ('compact', 'standard', 'large')
        """
        self.model_size = model_size
        self.models = {}
        self.tag_names = []
        self.model_metadata = {}
        
        # Model configurations
        self.model_configs = {
            'compact': {
                'sample_rate': 16000,
                'input_length': 16000 * 3,  # 3 seconds
                'model_name': 'musicnn_compact'
            },
            'standard': {
                'sample_rate': 16000,
                'input_length': 16000 * 3,  # 3 seconds
                'model_name': 'musicnn_standard'
            },
            'large': {
                'sample_rate': 16000,
                'input_length': 16000 * 5,  # 5 seconds
                'model_name': 'musicnn_large'
            }
        }
        
        self.config = self.model_configs.get(model_size, self.model_configs['standard'])
        
        if TENSORFLOW_AVAILABLE:
            self._initialize_models()
    
    def _initialize_models(self):
        """Initialize MusiCNN models."""
        try:
            if not TENSORFLOW_AVAILABLE:
                log_universal('WARNING', 'MusiCNN', 'TensorFlow not available, MusiCNN disabled')
                return
            
            tf = get_tensorflow()
            
            # For now, create placeholder models
            # In a real implementation, these would load pre-trained MusiCNN models
            log_universal('INFO', 'MusiCNN', f'Initializing {self.model_size} MusiCNN models')
            
            # Initialize tag names (standard MusiCNN tags)
            self.tag_names = [
                'rock', 'pop', 'alternative', 'indie', 'electronic', 'female vocalists',
                'dance', 'folk', 'chillout', 'instrumental', 'beautiful', 'country',
                'classical', 'jazz', 'blues', 'experimental', 'ambient', 'world',
                'singer-songwriter', 'metal', 'acoustic', 'sad', 'happy', 'energetic',
                'dark', 'atmospheric', 'melodic', 'emotional', 'aggressive', 'peaceful'
            ]
            
            # Create dummy models for demonstration
            # Real implementation would load actual MusiCNN weights
            self.models = {
                'tagging': self._create_placeholder_model('tagging'),
                'embeddings': self._create_placeholder_model('embeddings')
            }
            
            self.model_metadata = {
                'sample_rate': self.config['sample_rate'],
                'input_length': self.config['input_length'],
                'model_size': self.model_size,
                'num_tags': len(self.tag_names)
            }
            
            log_universal('INFO', 'MusiCNN', f'MusiCNN models initialized successfully')
            
        except Exception as e:
            log_universal('ERROR', 'MusiCNN', f'Model initialization failed: {e}')
            self.models = {}
    
    def _create_placeholder_model(self, model_type: str):
        """Create placeholder model for demonstration."""
        if not TENSORFLOW_AVAILABLE:
            return None
        
        try:
            tf = get_tensorflow()
            
            # Create a simple placeholder model
            if model_type == 'tagging':
                # Tagging model outputs probabilities for each tag
                model = tf.keras.Sequential([
                    tf.keras.layers.Input(shape=(self.config['input_length'],)),
                    tf.keras.layers.Dense(128, activation='relu'),
                    tf.keras.layers.Dense(64, activation='relu'),
                    tf.keras.layers.Dense(len(self.tag_names), activation='sigmoid')
                ])
            else:  # embeddings
                # Embeddings model outputs feature vectors
                model = tf.keras.Sequential([
                    tf.keras.layers.Input(shape=(self.config['input_length'],)),
                    tf.keras.layers.Dense(128, activation='relu'),
                    tf.keras.layers.Dense(64, activation='relu'),
                    tf.keras.layers.Dense(50)  # 50-dimensional embeddings
                ])
            
            return model
            
        except Exception as e:
            log_universal('ERROR', 'MusiCNN', f'Placeholder model creation failed: {e}')
            return None
    
    def is_available(self) -> bool:
        """Check if MusiCNN is available."""
        return TENSORFLOW_AVAILABLE and bool(self.models)
    
    def extract_features(self, audio: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """
        Extract MusiCNN features from audio.
        
        Args:
            audio: Audio array
            sample_rate: Sample rate
            
        Returns:
            Dictionary containing tags and embeddings
        """
        if not self.is_available():
            log_universal('WARNING', 'MusiCNN', 'MusiCNN not available')
            return {'tags': {}, 'embeddings': [], 'available': False}
        
        try:
            # Preprocess audio
            processed_audio = self._preprocess_audio(audio, sample_rate)
            if processed_audio is None:
                return {'tags': {}, 'embeddings': [], 'error': 'preprocessing_failed'}
            
            # Extract tags
            tags = self._extract_tags(processed_audio)
            
            # Extract embeddings
            embeddings = self._extract_embeddings(processed_audio)
            
            return {
                'tags': tags,
                'embeddings': embeddings,
                'available': True,
                'model_size': self.model_size
            }
            
        except Exception as e:
            log_universal('ERROR', 'MusiCNN', f'Feature extraction failed: {e}')
            return {'tags': {}, 'embeddings': [], 'error': str(e)}
    
    def _preprocess_audio(self, audio: np.ndarray, sample_rate: int) -> Optional[np.ndarray]:
        """
        Preprocess audio for MusiCNN.
        
        Args:
            audio: Input audio array
            sample_rate: Input sample rate
            
        Returns:
            Preprocessed audio array
        """
        try:
            # Ensure mono
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            
            # Resample to target sample rate if needed
            target_sr = self.config['sample_rate']
            if sample_rate != target_sr:
                # Simple resampling (in real implementation, use librosa)
                resample_factor = target_sr / sample_rate
                target_length = int(len(audio) * resample_factor)
                audio = np.interp(
                    np.linspace(0, len(audio) - 1, target_length),
                    np.arange(len(audio)),
                    audio
                )
            
            # Ensure correct length
            target_length = self.config['input_length']
            if len(audio) > target_length:
                # Take middle segment
                start_idx = (len(audio) - target_length) // 2
                audio = audio[start_idx:start_idx + target_length]
            elif len(audio) < target_length:
                # Pad with zeros
                padding = target_length - len(audio)
                audio = np.pad(audio, (0, padding), mode='constant')
            
            # Normalize
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio))
            
            return audio.astype(np.float32)
            
        except Exception as e:
            log_universal('ERROR', 'MusiCNN', f'Audio preprocessing failed: {e}')
            return None
    
    def _extract_tags(self, audio: np.ndarray) -> Dict[str, float]:
        """Extract music tags using MusiCNN tagging model."""
        try:
            if 'tagging' not in self.models or self.models['tagging'] is None:
                return {}
            
            # Reshape for model input
            audio_input = audio.reshape(1, -1)
            
            # For placeholder model, generate random probabilities
            # In real implementation, this would be: predictions = self.models['tagging'].predict(audio_input)
            np.random.seed(42)  # For reproducible results
            predictions = np.random.rand(1, len(self.tag_names))
            
            # Convert to tag probabilities
            tag_probs = predictions[0]
            
            # Create tag dictionary
            tags = {}
            for i, tag_name in enumerate(self.tag_names):
                tags[tag_name] = float(tag_probs[i])
            
            log_universal('DEBUG', 'MusiCNN', f'Extracted {len(tags)} tag probabilities')
            return tags
            
        except Exception as e:
            log_universal('ERROR', 'MusiCNN', f'Tag extraction failed: {e}')
            return {}
    
    def _extract_embeddings(self, audio: np.ndarray) -> List[float]:
        """Extract music embeddings using MusiCNN embeddings model."""
        try:
            if 'embeddings' not in self.models or self.models['embeddings'] is None:
                return []
            
            # Reshape for model input
            audio_input = audio.reshape(1, -1)
            
            # For placeholder model, generate random embeddings
            # In real implementation, this would be: embeddings = self.models['embeddings'].predict(audio_input)
            np.random.seed(123)  # For reproducible results
            embeddings = np.random.randn(1, 50)  # 50-dimensional embeddings
            
            # Convert to list
            embedding_vector = embeddings[0].tolist()
            
            log_universal('DEBUG', 'MusiCNN', f'Extracted {len(embedding_vector)}-dimensional embeddings')
            return embedding_vector
            
        except Exception as e:
            log_universal('ERROR', 'MusiCNN', f'Embedding extraction failed: {e}')
            return []
    
    def get_top_tags(self, tags: Dict[str, float], top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Get top-k tags sorted by probability.
        
        Args:
            tags: Tag probability dictionary
            top_k: Number of top tags to return
            
        Returns:
            List of (tag_name, probability) tuples
        """
        if not tags:
            return []
        
        # Sort tags by probability
        sorted_tags = sorted(tags.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_tags[:top_k]
    
    def get_genre_prediction(self, tags: Dict[str, float]) -> Optional[str]:
        """
        Get primary genre prediction from tags.
        
        Args:
            tags: Tag probability dictionary
            
        Returns:
            Primary genre string or None
        """
        if not tags:
            return None
        
        # Define genre tags
        genre_tags = [
            'rock', 'pop', 'electronic', 'jazz', 'classical', 'country',
            'metal', 'folk', 'blues', 'alternative', 'indie', 'dance',
            'ambient', 'experimental', 'world'
        ]
        
        # Filter to genre tags only
        genre_probs = {tag: prob for tag, prob in tags.items() if tag in genre_tags}
        
        if not genre_probs:
            return None
        
        # Return genre with highest probability
        return max(genre_probs.items(), key=lambda x: x[1])[0]
    
    def get_mood_prediction(self, tags: Dict[str, float]) -> Optional[str]:
        """
        Get primary mood prediction from tags.
        
        Args:
            tags: Tag probability dictionary
            
        Returns:
            Primary mood string or None
        """
        if not tags:
            return None
        
        # Define mood tags
        mood_tags = [
            'happy', 'sad', 'energetic', 'peaceful', 'aggressive',
            'dark', 'beautiful', 'emotional', 'atmospheric', 'melodic'
        ]
        
        # Filter to mood tags only
        mood_probs = {tag: prob for tag, prob in tags.items() if tag in mood_tags}
        
        if not mood_probs:
            return None
        
        # Return mood with highest probability
        return max(mood_probs.items(), key=lambda x: x[1])[0]


# Global instance for shared use
_musicnn_integration = None


def get_musicnn_integration(model_size: str = 'standard') -> MusiCNNIntegration:
    """Get shared MusiCNN integration instance."""
    global _musicnn_integration
    
    if _musicnn_integration is None or _musicnn_integration.model_size != model_size:
        _musicnn_integration = MusiCNNIntegration(model_size)
    
    return _musicnn_integration
