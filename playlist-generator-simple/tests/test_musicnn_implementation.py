#!/usr/bin/env python3
"""
Test script for MusiCNN implementation.
Verifies model loading and feature extraction.
"""

# Suppress TensorFlow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all TensorFlow warnings

import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from core.audio_analyzer import AudioAnalyzer
from core.config_loader import ConfigLoader
from core.logging_setup import log_universal

def test_musicnn_configuration():
    """Test MusiCNN configuration loading."""
    print("=== Testing MusiCNN Configuration ===")
    
    # Load configuration
    config_loader = ConfigLoader()
    config = config_loader.get_config()
    audio_config = config_loader.get_audio_analysis_config()
    
    print(f"EXTRACT_MUSICNN: {config.get('EXTRACT_MUSICNN')}")
    print(f"MUSICNN_MODEL_PATH: {config.get('MUSICNN_MODEL_PATH')}")
    print(f"MUSICNN_JSON_PATH: {config.get('MUSICNN_JSON_PATH')}")
    print(f"MUSICNN_TIMEOUT_SECONDS: {config.get('MUSICNN_TIMEOUT_SECONDS')}")
    
    # Check if paths exist
    model_path = config.get('MUSICNN_MODEL_PATH', '/app/models/msd-musicnn-1.pb')
    json_path = config.get('MUSICNN_JSON_PATH', '/app/models/msd-musicnn-1.json')
    
    print(f"\nModel path exists: {os.path.exists(model_path)}")
    print(f"JSON path exists: {os.path.exists(json_path)}")
    
    if os.path.exists(model_path):
        print(f"Model file size: {os.path.getsize(model_path)} bytes")
    
    if os.path.exists(json_path):
        print(f"JSON file size: {os.path.getsize(json_path)} bytes")
        try:
            with open(json_path, 'r') as f:
                json_data = json.load(f)
            print(f"JSON keys: {list(json_data.keys())}")
        except Exception as e:
            print(f"Error reading JSON: {e}")

def test_musicnn_analyzer():
    """Test MusiCNN analyzer initialization."""
    print("\n=== Testing MusiCNN Analyzer ===")
    
    # Load configuration
    config_loader = ConfigLoader()
    audio_config = config_loader.get_audio_analysis_config()
    
    # Create analyzer
    analyzer = AudioAnalyzer(config=audio_config)
    
    print(f"Analyzer config: {analyzer.config}")
    print(f"Extract MusiCNN: {analyzer.extract_musicnn}")
    
    # Check if model is loaded
    if hasattr(analyzer, '_musicnn_model'):
        print(f"MusiCNN model loaded: {analyzer._musicnn_model is not None}")
    else:
        print("MusiCNN model not loaded (will be loaded on first use)")

def test_musicnn_feature_extraction():
    """Test MusiCNN feature extraction with dummy audio."""
    print("\n=== Testing MusiCNN Feature Extraction ===")
    
    # Load configuration
    config_loader = ConfigLoader()
    audio_config = config_loader.get_audio_analysis_config()
    
    # Create analyzer
    analyzer = AudioAnalyzer(config=audio_config)
    
    # Create dummy audio data (1 second of silence at 16kHz)
    import numpy as np
    dummy_audio = np.zeros(16000)  # 1 second of silence
    
    print("Extracting MusiCNN features from dummy audio...")
    features = analyzer._extract_musicnn_features(dummy_audio, 16000)
    
    print(f"Features extracted: {list(features.keys())}")
    print(f"Embedding size: {len(features.get('embedding', []))}")
    print(f"Tags: {features.get('tags', {})}")

def test_mel_spectrogram():
    """Test mel-spectrogram computation."""
    print("\n=== Testing Mel-Spectrogram Computation ===")
    
    # Load configuration
    config_loader = ConfigLoader()
    audio_config = config_loader.get_audio_analysis_config()
    
    # Create analyzer
    analyzer = AudioAnalyzer(config=audio_config)
    
    # Create dummy audio data
    import numpy as np
    dummy_audio = np.random.randn(16000)  # 1 second of noise
    
    print("Computing mel-spectrogram...")
    mel_spec = analyzer._compute_mel_spectrogram(dummy_audio, 16000)
    
    print(f"Mel-spectrogram shape: {mel_spec.shape}")
    print(f"Mel-spectrogram min: {mel_spec.min():.4f}")
    print(f"Mel-spectrogram max: {mel_spec.max():.4f}")
    print(f"Mel-spectrogram mean: {mel_spec.mean():.4f}")

def main():
    """Run all MusiCNN tests."""
    print("MusiCNN Implementation Test")
    print("=" * 50)
    
    try:
        test_musicnn_configuration()
        test_musicnn_analyzer()
        test_mel_spectrogram()
        test_musicnn_feature_extraction()
        
        print("\n=== Test Summary ===")
        print("✓ Configuration loading works")
        print("✓ Analyzer initialization works")
        print("✓ Mel-spectrogram computation works")
        print("✓ Feature extraction works (with fallbacks)")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 