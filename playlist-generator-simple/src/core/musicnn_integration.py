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
        """Extract music tags using Essentia TensorFlow MusiCNN model."""
        try:
            # Use Essentia's TensorFlowPredictor2D for real MusiCNN inference
            from .lazy_imports import get_essentia
            es = get_essentia()
            
            if es is None:
                log_universal('WARNING', 'MusiCNN', 'Essentia not available, using fallback tags')
                return self._generate_fallback_tags()
            
            # Try multiple model approaches in order of preference
            
            # Try to load MusiCNN models using config paths
            from .config_loader import ConfigLoader
            try:
                config_loader = ConfigLoader()
                config = config_loader.get_audio_analysis_config()
                
                musicnn_model_path = config.get('MUSICNN_MODEL_PATH', '/app/models/musicnn/msd-musicnn-1.pb')
                
                model_paths = [
                    musicnn_model_path,
                    "/app/models/musicnn/msd-musicnn-1.pb",  # Your actual model path
                    "/app/models/musicnn_msd.pb"  # Legacy fallback
                ]
            except Exception as e:
                log_universal('WARNING', 'MusiCNN', f'Config loading failed: {e}, using default paths')
                model_paths = [
                    "/app/models/musicnn/msd-musicnn-1.pb",  # Your actual model path
                    "/app/models/musicnn_msd.pb"  # Legacy fallback
                ]
            
            model_found = False
            for model_path in model_paths:
                try:
                    import os
                    if os.path.exists(model_path):
                        log_universal('DEBUG', 'MusiCNN', f'Found model at {model_path}')
                        
                        # Load MusiCNN tag names from JSON if available
                        json_path = model_path.replace('.pb', '.json')
                        if os.path.exists(json_path):
                            log_universal('DEBUG', 'MusiCNN', f'Loading tag names from {json_path}')
                            self._load_musicnn_tag_names(json_path)
                        
                        # Use TensorFlowPredictor2D for MusiCNN models
                        model = es.TensorFlowPredictor2D(
                            graphFilename=model_path,
                            inputs=["model/Placeholder"],
                            outputs=["model/Sigmoid"]
                        )
                        
                        # Prepare audio for model
                        audio_processed = self._prepare_audio_for_essentia(audio)
                        if audio_processed is None:
                            continue
                        
                        # Get predictions
                        predictions = model(audio_processed)
                        tags = self._convert_musicnn_predictions_to_tags(predictions)
                        
                        log_universal('DEBUG', 'MusiCNN', f'Extracted {len(tags)} tags using MusiCNN at {model_path}')
                        return tags
                        
                except Exception as e:
                    log_universal('DEBUG', 'MusiCNN', f'Model at {model_path} failed: {e}')
                    continue
            
            # If no TensorFlow models work, use descriptive analysis
            log_universal('INFO', 'MusiCNN', 'No TensorFlow models available, using descriptive analysis')
            return self._extract_tags_alternative(audio)
                
        except Exception as e:
            log_universal('ERROR', 'MusiCNN', f'Tag extraction failed: {e}')
            return self._generate_fallback_tags()
    
    def _prepare_audio_for_essentia(self, audio: np.ndarray) -> np.ndarray:
        """Prepare audio for MusiCNN TensorFlow models (optimized for 16kHz input)."""
        try:
            # Ensure mono (audio should already be mono from FFmpeg)
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            
            # Audio is already at 16kHz from FFmpeg, no resampling needed
            target_sr = 16000
            
            # Normalize audio
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio))
            
            # Ensure exactly 3 seconds (48000 samples at 16kHz)
            target_length = target_sr * 3  # 48000 samples
            if len(audio) > target_length:
                # Take middle segment for best representation
                start_idx = (len(audio) - target_length) // 2
                audio = audio[start_idx:start_idx + target_length]
            elif len(audio) < target_length:
                # Pad with zeros
                audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
            
            return audio.astype(np.float32)
            
        except Exception as e:
            log_universal('ERROR', 'MusiCNN', f'Audio preparation failed: {e}')
            return None
    
    def _extract_tags_alternative(self, audio: np.ndarray) -> Dict[str, float]:
        """Alternative tag extraction using basic Essentia descriptors."""
        try:
            from .lazy_imports import get_essentia
            es = get_essentia()
            
            if es is None:
                return self._generate_fallback_tags()
            
            # Use basic Essentia descriptors to estimate tags
            tags = {}
            
            # Extract basic features
            windowing = es.Windowing(type='hann')
            spectrum = es.Spectrum()
            spectral_centroid = es.Centroid(range=8000)
            
            # Tempo for dance/electronic detection
            rhythm_extractor = es.RhythmExtractor2013()
            bpm, _, beats_confidence, _, _ = rhythm_extractor(audio)
            
            # Zero crossing rate for rock/metal detection
            zcr = es.ZeroCrossingRate()
            
            # Energy for energetic tags
            energy_extractor = es.Energy()
            
            # Process audio in frames
            centroids, zcr_values, energies = [], [], []
            for frame in es.FrameGenerator(audio, frameSize=2048, hopSize=512):
                windowed_frame = windowing(frame)
                spectrum_frame = spectrum(windowed_frame)
                
                centroids.append(spectral_centroid(spectrum_frame))
                zcr_values.append(zcr(frame))
                energies.append(energy_extractor(frame))
            
            # Convert features to tag probabilities
            avg_centroid = np.mean(centroids) if centroids else 0
            avg_zcr = np.mean(zcr_values) if zcr_values else 0
            avg_energy = np.mean(energies) if energies else 0
            
            # Map features to common music tags (normalized 0-1)
            tags['electronic'] = min(1.0, max(0.0, (avg_centroid - 1000) / 3000))  # Higher centroid = electronic
            tags['rock'] = min(1.0, max(0.0, avg_zcr * 10))  # Higher ZCR = rock
            tags['dance'] = min(1.0, max(0.0, (bpm - 100) / 80)) if bpm > 100 else 0.1  # Fast BPM = dance
            tags['energetic'] = min(1.0, max(0.0, avg_energy * 100))  # Higher energy = energetic
            tags['pop'] = 0.5  # Default moderate probability
            tags['classical'] = max(0.0, 1.0 - tags['electronic'] - tags['rock'])  # Opposite of electronic/rock
            tags['acoustic'] = 1.0 - tags['electronic']  # Opposite of electronic
            tags['instrumental'] = 0.3  # Default low probability
            tags['happy'] = min(1.0, max(0.1, (bpm - 80) / 100)) if bpm > 80 else 0.2  # Faster = happier
            tags['sad'] = 1.0 - tags['happy']  # Opposite of happy
            
            log_universal('DEBUG', 'MusiCNN', f'Generated {len(tags)} tags using Essentia descriptors')
            return tags
            
        except Exception as e:
            log_universal('ERROR', 'MusiCNN', f'Alternative tag extraction failed: {e}')
            return self._generate_fallback_tags()
    
    def _convert_musicnn_predictions_to_tags(self, predictions: np.ndarray) -> Dict[str, float]:
        """Convert MusiCNN predictions to tag dictionary."""
        try:
            # MusiCNN outputs tag probabilities
            if len(predictions.shape) > 1:
                predictions = predictions[0]  # Take first batch item
            
            # Create tag dictionary using loaded tag names or defaults
            tags = {}
            tag_names = self.tag_names if hasattr(self, 'tag_names') and self.tag_names else self._get_default_tag_names()
            
            for i, tag_name in enumerate(tag_names[:min(len(tag_names), len(predictions))]):
                tags[tag_name] = float(predictions[i])
            
            # Ensure all expected tag names have values
            for tag_name in tag_names:
                if tag_name not in tags:
                    tags[tag_name] = 0.1  # Default low probability
            
            return tags
            
        except Exception as e:
            log_universal('ERROR', 'MusiCNN', f'MusiCNN prediction conversion failed: {e}')
            return self._generate_fallback_tags()
    
    def _load_musicnn_tag_names(self, json_path: str):
        """Load tag names from MusiCNN JSON metadata file."""
        try:
            import json
            with open(json_path, 'r') as f:
                metadata = json.load(f)
            
            # Extract tag names from JSON (format may vary)
            if 'classes' in metadata:
                self.tag_names = metadata['classes']
            elif 'tags' in metadata:
                self.tag_names = metadata['tags']
            elif 'labels' in metadata:
                self.tag_names = metadata['labels']
            else:
                log_universal('WARNING', 'MusiCNN', f'No tag names found in {json_path}, using defaults')
                self.tag_names = self._get_default_tag_names()
            
            log_universal('INFO', 'MusiCNN', f'Loaded {len(self.tag_names)} tag names from {json_path}')
            
        except Exception as e:
            log_universal('ERROR', 'MusiCNN', f'Failed to load tag names from {json_path}: {e}')
            self.tag_names = self._get_default_tag_names()
    
    def _get_default_tag_names(self) -> List[str]:
        """Get default MusiCNN tag names if JSON loading fails."""
        return [
            'rock', 'pop', 'alternative', 'indie', 'electronic', 'female vocalists',
            'dance', 'folk', 'chillout', 'instrumental', 'beautiful', 'ambient',
            'jazz', 'chill', 'experimental', 'electronica', 'male vocalists',
            'hip hop', 'classical', 'soul', 'funk', 'reggae', 'country',
            'rnb', 'psychedelic', 'new age', 'world', 'blues', 'house',
            'metal', 'latin'
        ]
    
    def _generate_fallback_tags(self) -> Dict[str, float]:
        """Generate basic fallback tags when models aren't available."""
        # Return basic tag set with low probabilities
        fallback_tags = {}
        for tag_name in self.tag_names:
            fallback_tags[tag_name] = 0.1  # Very low default probability
        
        # Add some basic reasonable defaults
        fallback_tags.update({
            'pop': 0.3,
            'instrumental': 0.2,
            'electronic': 0.2,
            'acoustic': 0.2
        })
        
        log_universal('DEBUG', 'MusiCNN', f'Using fallback tags: {len(fallback_tags)} tags')
        return fallback_tags
    
    def _extract_embeddings(self, audio: np.ndarray) -> List[float]:
        """Extract music embeddings using Essentia TensorFlow models."""
        try:
            from .lazy_imports import get_essentia
            es = get_essentia()
            
            if es is None:
                log_universal('WARNING', 'MusiCNN', 'Essentia not available, generating basic embeddings')
                return self._generate_basic_embeddings(audio)
            
            # Try to use MusiCNN for embeddings
            try:
                # Use same MusiCNN model but get intermediate layer for embeddings
                from .config_loader import ConfigLoader
                config_loader = ConfigLoader()
                config = config_loader.get_audio_analysis_config()
                model_path = config.get('MUSICNN_MODEL_PATH', '/app/models/musicnn/msd-musicnn-1.pb')
                
                if not os.path.exists(model_path):
                    model_path = "/app/models/musicnn/msd-musicnn-1.pb"
                
                if os.path.exists(model_path):
                    model = es.TensorFlowPredictor2D(
                        graphFilename=model_path,
                        inputs=["model/Placeholder"],
                        outputs=["model/dense/BiasAdd"]  # Get dense layer for embeddings
                    )
                
                # Prepare audio
                audio_processed = self._prepare_audio_for_essentia(audio)
                if audio_processed is None:
                    return self._generate_basic_embeddings(audio)
                
                # Get embeddings
                embeddings = model(audio_processed)
                
                # Convert to list and ensure 50 dimensions
                if len(embeddings.shape) > 1:
                    embeddings = embeddings[0]
                
                    # Resize to 50 dimensions if needed
                    if len(embeddings) > 50:
                        embeddings = embeddings[:50]
                    elif len(embeddings) < 50:
                        embeddings = np.pad(embeddings, (0, 50 - len(embeddings)), mode='constant')
                    
                    embedding_vector = embeddings.astype(float).tolist()
                    
                    log_universal('DEBUG', 'MusiCNN', f'Extracted {len(embedding_vector)}-dimensional embeddings using MusiCNN')
                    return embedding_vector
                else:
                    log_universal('WARNING', 'MusiCNN', f'MusiCNN model not found at {model_path}, using basic embeddings')
                    return self._generate_basic_embeddings(audio)
                
            except Exception as e:
                log_universal('WARNING', 'MusiCNN', f'MusiCNN embeddings failed: {e}, using basic embeddings')
                return self._generate_basic_embeddings(audio)
                
        except Exception as e:
            log_universal('ERROR', 'MusiCNN', f'Embedding extraction failed: {e}')
            return self._generate_basic_embeddings(audio)
    
    def _generate_basic_embeddings(self, audio: np.ndarray) -> List[float]:
        """Generate basic audio embeddings using statistical features."""
        try:
            from .lazy_imports import get_essentia
            es = get_essentia()
            
            embeddings = []
            
            if es is not None:
                # Extract basic audio features for embeddings
                windowing = es.Windowing(type='hann')
                spectrum = es.Spectrum()
                mfcc = es.MFCC(numberCoefficients=13)
                spectral_centroid = es.Centroid(range=8000)
                
                # Collect features across frames
                features_per_frame = []
                for frame in es.FrameGenerator(audio, frameSize=2048, hopSize=512):
                    windowed_frame = windowing(frame)
                    spectrum_frame = spectrum(windowed_frame)
                    
                    # Get MFCC
                    mfcc_bands, mfcc_coeffs = mfcc(spectrum_frame)
                    
                    # Get spectral centroid
                    centroid = spectral_centroid(spectrum_frame)
                    
                    # Combine features
                    frame_features = list(mfcc_coeffs) + [centroid]
                    features_per_frame.append(frame_features)
                
                # Aggregate features (mean and std)
                if features_per_frame:
                    features_array = np.array(features_per_frame)
                    mean_features = np.mean(features_array, axis=0)
                    std_features = np.std(features_array, axis=0)
                    embeddings = list(mean_features) + list(std_features)
            
            # Ensure 50 dimensions
            if len(embeddings) > 50:
                embeddings = embeddings[:50]
            elif len(embeddings) < 50:
                # Pad with statistical features of the audio
                additional_features = []
                if len(audio) > 0:
                    additional_features = [
                        float(np.mean(audio)), float(np.std(audio)), 
                        float(np.min(audio)), float(np.max(audio)),
                        float(np.median(audio))
                    ]
                
                # Pad to 50 dimensions
                padding_needed = 50 - len(embeddings) - len(additional_features)
                embeddings.extend(additional_features)
                embeddings.extend([0.0] * max(0, padding_needed))
                embeddings = embeddings[:50]  # Ensure exactly 50
            
            log_universal('DEBUG', 'MusiCNN', f'Generated {len(embeddings)}-dimensional basic embeddings')
            return embeddings
            
        except Exception as e:
            log_universal('ERROR', 'MusiCNN', f'Basic embedding generation failed: {e}')
            # Return zero vector as last resort
            return [0.0] * 50
    
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
