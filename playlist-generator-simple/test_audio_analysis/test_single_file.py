#!/usr/bin/env python3
"""
Test script to analyze a single file and see the corrected feature values.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from core.audio_analyzer import AudioAnalyzer
import logging

# Set up logging to see the output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_single_file():
    """Test feature extraction on a single file."""
    
    # Initialize analyzer
    analyzer = AudioAnalyzer()
    
    # Test file path (using a smaller file)
    test_file = "/music/22Gz - Crime Rate.mp3"
    
    print(f"Testing feature extraction on: {test_file}")
    print("=" * 60)
    
    try:
        # Extract features
        result = analyzer.extract_features(test_file)
        
        if result and result.get('success'):
            features = result.get('features', {})
            
            print("\n FEATURE EXTRACTION RESULTS:")
            print("=" * 60)
            
            # Check for the specific features we fixed
            if 'bpm' in features:
                print(f"BPM: {features['bpm']}")
            
            if 'confidence' in features:
                print(f"Rhythm Confidence: {features['confidence']}")
            
            if 'spectral_centroid' in features:
                print(f"Spectral Centroid: {features['spectral_centroid']:.1f} Hz")
            
            if 'spectral_rolloff' in features:
                print(f"Spectral Rolloff: {features['spectral_rolloff']:.1f} Hz")
            
            if 'spectral_flatness' in features:
                print(f"Spectral Flatness: {features['spectral_flatness']:.3f}")
            
            if 'loudness' in features:
                print(f"Loudness (RMS): {features['loudness']:.3f}")
            
            if 'dynamic_complexity' in features:
                print(f"Dynamic Complexity: {features['dynamic_complexity']:.3f}")
            
            print("\n" + "=" * 60)
            print("Expected ranges:")
            print("- BPM: 30-300")
            print("- Rhythm Confidence: 0.0-1.0")
            print("- Spectral Centroid: 1000-8000 Hz")
            print("- Spectral Rolloff: 1000-8000 Hz")
            print("- Spectral Flatness: 0.0-1.0")
            print("- Loudness (RMS): 0.0-1.0")
            print("- Dynamic Complexity: 0.0-10.0")
            
        else:
            print(" Feature extraction failed")
            if result:
                print(f"Error: {result}")
    
    except Exception as e:
        print(f" Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_single_file() 