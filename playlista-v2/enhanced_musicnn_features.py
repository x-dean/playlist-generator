#!/usr/bin/env python3
"""
Enhanced feature extraction with MusiCNN models
This will work when Essentia with TensorFlow support is available
"""

import asyncio
import os
import sys
import time
import numpy as np
from pathlib import Path

sys.path.append('/app')

from app.analysis.features import FeatureExtractor
from app.analysis.models import model_manager
from app.core.logging import get_logger

logger = get_logger("enhanced_musicnn")

class EnhancedMusiCNNExtractor:
    """Enhanced feature extractor with MusiCNN model support"""
    
    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        self.musicnn_tags = [
            "rock", "pop", "alternative", "indie", "electronic", "dance", "jazz", "blues",
            "country", "folk", "classical", "metal", "punk", "hip hop", "rap", "reggae",
            "funk", "soul", "rnb", "ambient", "experimental", "instrumental", "vocal",
            "acoustic", "hard rock", "progressive rock", "psychedelic", "new age",
            "world", "latin", "oldies", "60s", "70s", "80s", "90s", "00s", "10s",
            "love", "party", "sad", "happy", "energetic", "calm", "aggressive", "mellow",
            "upbeat", "chill", "romantic", "dark", "atmospheric", "melodic"
        ]
    
    async def analyze_with_musicnn(self, audio_path: str) -> dict:
        """Analyze audio with enhanced MusiCNN features"""
        
        print(f"\n🎵 Enhanced Analysis: {os.path.basename(audio_path)}")
        
        # Check if Essentia with TensorFlow is available
        essentia_available = self._check_essentia_tensorflow()
        
        # Check if MusiCNN model files are available
        musicnn_model_available = self._check_musicnn_models()
        
        print(f"   🔧 Essentia with TensorFlow: {'✅' if essentia_available else '❌'}")
        print(f"   🤖 MusiCNN Model Available: {'✅' if musicnn_model_available else '❌'}")
        
        # Extract standard features first
        start_time = time.time()
        features = await self.feature_extractor.extract_comprehensive_features(audio_path)
        standard_time = time.time() - start_time
        
        print(f"   📊 Standard features extracted in {standard_time:.1f}s")
        
        # Try enhanced analysis if Essentia+TF is available
        if essentia_available and musicnn_model_available:
            enhanced_features = await self._extract_musicnn_features(audio_path)
            features.update(enhanced_features)
            print(f"   🎯 Enhanced MusiCNN features extracted")
        else:
            # Simulate what would be available
            print(f"   💡 Simulating enhanced features (Essentia+TF not fully available)")
            enhanced_features = self._simulate_musicnn_features()
            features['simulated_musicnn'] = enhanced_features
        
        return features
    
    def _check_essentia_tensorflow(self) -> bool:
        """Check if Essentia with TensorFlow support is available"""
        try:
            import essentia.standard as es
            from essentia.standard import TensorflowPredictMusiCNN
            return True
        except ImportError:
            return False
    
    def _check_musicnn_models(self) -> bool:
        """Check if MusiCNN model files are available"""
        model_paths = [
            "/models/msd-musicnn-1.pb",
            "/app/musicnn_models/msd-musicnn-1.pb"
        ]
        
        for path in model_paths:
            if os.path.exists(path):
                print(f"   📁 Found MusiCNN model at: {path}")
                return True
        
        return False
    
    async def _extract_musicnn_features(self, audio_path: str) -> dict:
        """Extract features using actual MusiCNN model"""
        
        try:
            import essentia.standard as es
            from essentia.standard import TensorflowPredictMusiCNN
            
            # Load audio
            loader = es.MonoLoader(filename=audio_path, sampleRate=16000)
            audio = loader()
            
            # Find MusiCNN model
            model_paths = [
                "/models/msd-musicnn-1.pb",
                "/app/musicnn_models/msd-musicnn-1.pb"
            ]
            
            model_path = None
            for path in model_paths:
                if os.path.exists(path):
                    model_path = path
                    break
            
            if not model_path:
                raise FileNotFoundError("MusiCNN model not found")
            
            # Load MusiCNN model
            musicnn = TensorflowPredictMusiCNN(
                graphFilename=model_path,
                input="model/Placeholder",
                output="model/Sigmoid"
            )
            
            # Get predictions
            predictions = musicnn(audio)
            
            # Process predictions
            enhanced_features = {
                'musicnn_predictions': predictions.tolist(),
                'musicnn_confidence_mean': float(np.mean(predictions)),
                'musicnn_confidence_max': float(np.max(predictions)),
                'musicnn_active_tags': self._get_active_tags(predictions),
                'musicnn_top_genres': self._get_top_genres(predictions),
                'musicnn_mood_scores': self._extract_mood_scores(predictions)
            }
            
            return enhanced_features
            
        except Exception as e:
            logger.warning(f"MusiCNN feature extraction failed: {e}")
            return {}
    
    def _simulate_musicnn_features(self) -> dict:
        """Simulate MusiCNN features for demonstration"""
        # Generate realistic-looking predictions
        np.random.seed(42)
        predictions = np.random.beta(0.5, 5, 50)  # Beta distribution for realistic tag probabilities
        
        return {
            'simulated_predictions': predictions.tolist(),
            'simulated_confidence_mean': float(np.mean(predictions)),
            'simulated_confidence_max': float(np.max(predictions)),
            'simulated_active_tags': self._get_active_tags(predictions),
            'simulated_top_genres': self._get_top_genres(predictions),
            'simulated_mood_scores': self._extract_mood_scores(predictions),
            'note': 'These are simulated features - real MusiCNN requires Essentia+TensorFlow'
        }
    
    def _get_active_tags(self, predictions: np.ndarray, threshold: float = 0.1) -> list:
        """Get tags above threshold"""
        active_tags = []
        for i, score in enumerate(predictions):
            if score > threshold and i < len(self.musicnn_tags):
                active_tags.append({
                    'tag': self.musicnn_tags[i],
                    'confidence': float(score)
                })
        return sorted(active_tags, key=lambda x: x['confidence'], reverse=True)
    
    def _get_top_genres(self, predictions: np.ndarray, top_k: int = 5) -> list:
        """Get top K genre predictions"""
        # Focus on genre-related tags (first 20 are typically genres)
        genre_predictions = predictions[:20]
        top_indices = np.argsort(genre_predictions)[-top_k:][::-1]
        
        top_genres = []
        for idx in top_indices:
            if idx < len(self.musicnn_tags):
                top_genres.append({
                    'genre': self.musicnn_tags[idx],
                    'confidence': float(genre_predictions[idx])
                })
        
        return top_genres
    
    def _extract_mood_scores(self, predictions: np.ndarray) -> dict:
        """Extract mood-related scores from predictions"""
        # Map specific tags to mood dimensions
        mood_mapping = {
            'happy': [37, 43],      # happy, upbeat indices
            'sad': [38, 48],        # sad, dark indices  
            'energetic': [40, 46],  # energetic, upbeat indices
            'calm': [41, 47],       # calm, chill indices
            'aggressive': [42],     # aggressive index
            'romantic': [48]        # romantic index
        }
        
        mood_scores = {}
        for mood, indices in mood_mapping.items():
            valid_indices = [i for i in indices if i < len(predictions)]
            if valid_indices:
                mood_scores[mood] = float(np.mean([predictions[i] for i in valid_indices]))
            else:
                mood_scores[mood] = 0.0
        
        return mood_scores

async def test_enhanced_analysis():
    """Test enhanced analysis with MusiCNN"""
    
    print("🎵 ENHANCED MUSICNN ANALYSIS TEST")
    print("=" * 50)
    
    # Initialize
    await model_manager.load_models()
    extractor = EnhancedMusiCNNExtractor()
    
    # Find audio files
    music_dir = "/music"
    audio_files = []
    
    for root, dirs, files in os.walk(music_dir):
        for file in files:
            if file.lower().endswith(('.mp3', '.wav', '.flac', '.m4a')):
                audio_files.append(os.path.join(root, file))
    
    # Test on first 2 files
    test_files = audio_files[:2]
    
    for i, file_path in enumerate(test_files, 1):
        print(f"\n[{i}] Testing: {os.path.basename(file_path)}")
        
        try:
            features = await extractor.analyze_with_musicnn(file_path)
            
            # Display enhanced results
            if 'musicnn_predictions' in features:
                print("   🎯 Real MusiCNN predictions available!")
                if 'musicnn_top_genres' in features:
                    print("   🎭 Top genres:")
                    for genre in features['musicnn_top_genres'][:3]:
                        print(f"      - {genre['genre']}: {genre['confidence']:.3f}")
                
                if 'musicnn_mood_scores' in features:
                    print("   😊 Mood scores:")
                    mood_scores = features['musicnn_mood_scores']
                    for mood, score in mood_scores.items():
                        if score > 0.1:
                            print(f"      - {mood}: {score:.3f}")
            
            elif 'simulated_musicnn' in features:
                sim = features['simulated_musicnn']
                print("   💡 Simulated MusiCNN features:")
                print(f"      - Mean confidence: {sim['simulated_confidence_mean']:.3f}")
                print(f"      - Max confidence: {sim['simulated_confidence_max']:.3f}")
                print(f"      - Active tags: {len(sim['simulated_active_tags'])}")
                
                print("   🎭 Simulated top genres:")
                for genre in sim['simulated_top_genres'][:3]:
                    print(f"      - {genre['genre']}: {genre['confidence']:.3f}")
        
        except Exception as e:
            print(f"   ❌ Analysis failed: {e}")
    
    print("\n🎯 Enhanced analysis test complete!")

if __name__ == "__main__":
    asyncio.run(test_enhanced_analysis())
