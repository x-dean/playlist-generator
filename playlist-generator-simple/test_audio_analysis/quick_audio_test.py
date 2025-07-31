#!/usr/bin/env python3
"""
Quick test to verify audio loading improvements.
"""

import sys
import os
sys.path.append('/app/src')

from core.audio_analyzer import AudioAnalyzer
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def test_audio_loading():
    """Test the improved audio loading functionality."""
    print("=" * 60)
    print("QUICK AUDIO LOADING TEST")
    print("=" * 60)
    
    # Initialize analyzer
    print("Initializing AudioAnalyzer...")
    analyzer = AudioAnalyzer()
    print("✅ AudioAnalyzer initialized successfully")
    
    # Test with a small file first
    test_file = "/music/Alex Warren - Ordinary.mp3"
    print(f"\nTesting audio loading with: {os.path.basename(test_file)}")
    
    try:
        # Test the audio loading directly
        audio = analyzer._safe_audio_load(test_file)
        
        if audio is not None:
            print(f"✅ Audio loaded successfully!")
            print(f"   Samples: {len(audio)}")
            print(f"   Duration: {len(audio) / 44100:.2f} seconds")
            print(f"   Data type: {audio.dtype}")
            print(f"   Min/Max values: {audio.min():.3f} / {audio.max():.3f}")
        else:
            print("❌ Audio loading failed")
            return False
            
    except Exception as e:
        print(f"❌ Error during audio loading: {e}")
        return False
    
    # Test feature extraction
    print(f"\nTesting feature extraction...")
    try:
        features = analyzer.extract_features(test_file)
        
        if features is not None:
            print("✅ Feature extraction successful!")
            print(f"   Features extracted: {len(features)}")
            if 'metadata' in features:
                print(f"   Metadata: {list(features['metadata'].keys())}")
            if 'rhythm' in features:
                print(f"   Rhythm features: {list(features['rhythm'].keys())}")
        else:
            print("❌ Feature extraction failed")
            return False
            
    except Exception as e:
        print(f"❌ Error during feature extraction: {e}")
        return False
    
    print("\n✅ All tests passed!")
    return True

if __name__ == "__main__":
    success = test_audio_loading()
    sys.exit(0 if success else 1) 