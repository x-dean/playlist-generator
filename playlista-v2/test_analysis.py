#!/usr/bin/env python3
"""
Test script to demonstrate music analysis functionality
"""

import asyncio
import json
import sys
import os
sys.path.append('/app')

from app.analysis.features import FeatureExtractor
from app.analysis.models import model_manager
from app.core.logging import get_logger

logger = get_logger("test_analysis")

async def test_music_analysis():
    """Test music analysis on available files"""
    
    print("🎵 Testing Playlista v2 Music Analysis")
    print("=" * 50)
    
    # Initialize feature extractor
    feature_extractor = FeatureExtractor()
    
    # Initialize models
    await model_manager.load_models()
    print("✅ ML Models loaded successfully")
    
    # Check for audio files
    music_dir = "/music"
    audio_files = []
    
    if os.path.exists(music_dir):
        for file in os.listdir(music_dir):
            if file.endswith(('.mp3', '.wav', '.flac', '.m4a')):
                audio_files.append(os.path.join(music_dir, file))
    
    print(f"📁 Found {len(audio_files)} audio files")
    
    if not audio_files:
        print("⚠️  No audio files found for analysis")
        return
    
    # Analyze the first audio file
    test_file = audio_files[0]
    print(f"🎧 Analyzing: {os.path.basename(test_file)}")
    
    try:
        # Extract comprehensive features
        features = await feature_extractor.extract_comprehensive_features(test_file)
        
        print("\n🔍 Analysis Results:")
        print("-" * 30)
        
        # Display basic features
        if 'basic_features' in features:
            basic = features['basic_features']
            print(f"Duration: {basic.get('duration', 'N/A'):.2f} seconds")
            print(f"Sample Rate: {basic.get('sample_rate', 'N/A')} Hz")
            print(f"Channels: {basic.get('channels', 'N/A')}")
        
        # Display spectral features
        if 'spectral_features' in features:
            spectral = features['spectral_features']
            print(f"Spectral Centroid: {spectral.get('spectral_centroid_mean', 'N/A'):.2f}")
            print(f"Spectral Rolloff: {spectral.get('spectral_rolloff_mean', 'N/A'):.2f}")
            print(f"Zero Crossing Rate: {spectral.get('zcr_mean', 'N/A'):.4f}")
        
        # Display rhythm features
        if 'rhythm_features' in features:
            rhythm = features['rhythm_features']
            print(f"Tempo: {rhythm.get('tempo', 'N/A'):.1f} BPM")
        
        # Display harmonic features
        if 'harmonic_features' in features:
            harmonic = features['harmonic_features']
            print(f"Chroma Mean: {harmonic.get('chroma_mean', 'N/A')}")
        
        # Test ML predictions
        print("\n🤖 ML Predictions:")
        print("-" * 20)
        
        # Generate dummy spectrogram for ML prediction
        import numpy as np
        dummy_features = np.random.random((128, 1292))
        
        # Genre prediction
        genre_probs = await model_manager.predict_genre(dummy_features)
        top_genre = max(genre_probs.items(), key=lambda x: x[1])
        print(f"Top Genre: {top_genre[0]} ({top_genre[1]:.2%})")
        
        # Mood prediction
        mood_scores = await model_manager.predict_mood(dummy_features)
        print(f"Valence: {mood_scores['valence']:.2f}")
        print(f"Energy: {mood_scores['energy']:.2f}")
        print(f"Danceability: {mood_scores['danceability']:.2f}")
        
        # Audio embeddings
        embeddings = await model_manager.extract_embeddings(dummy_features)
        print(f"Embeddings: {len(embeddings)} dimensions")
        
        print("\n✅ Music analysis completed successfully!")
        
    except Exception as e:
        print(f"❌ Analysis failed: {str(e)}")
        logger.error(f"Analysis error: {e}")

if __name__ == "__main__":
    asyncio.run(test_music_analysis())
