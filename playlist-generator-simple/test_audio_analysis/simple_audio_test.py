#!/usr/bin/env python3
"""
Simple test to verify audio loading libraries work.
"""

import sys
import os
import numpy as np

def test_audio_libraries():
    """Test audio loading libraries directly."""
    print("=" * 60)
    print("SIMPLE AUDIO LIBRARY TEST")
    print("=" * 60)
    
    # Test imports
    print("Testing library imports...")
    
    try:
        import librosa
        print("✅ Librosa imported successfully")
        LIBROSA_AVAILABLE = True
    except ImportError as e:
        print(f"❌ Librosa import failed: {e}")
        LIBROSA_AVAILABLE = False
    
    try:
        import essentia.standard as es
        print("✅ Essentia imported successfully")
        ESSENTIA_AVAILABLE = True
    except ImportError as e:
        print(f"❌ Essentia import failed: {e}")
        ESSENTIA_AVAILABLE = False
    
    try:
        import soundfile as sf
        print("✅ SoundFile imported successfully")
        SOUNDFILE_AVAILABLE = True
    except ImportError as e:
        print(f"❌ SoundFile import failed: {e}")
        SOUNDFILE_AVAILABLE = False
    
    try:
        import wave
        print("✅ Wave module imported successfully")
        WAVE_AVAILABLE = True
    except ImportError as e:
        print(f"❌ Wave import failed: {e}")
        WAVE_AVAILABLE = False
    
    # Test audio loading
    test_file = "/music/Alex Warren - Ordinary.mp3"
    print(f"\nTesting audio loading with: {os.path.basename(test_file)}")
    
    if not os.path.exists(test_file):
        print(f"❌ Test file not found: {test_file}")
        return False
    
    # Try librosa
    if LIBROSA_AVAILABLE:
        try:
            print("Trying librosa.load...")
            audio, sr = librosa.load(test_file, sr=44100, mono=True, duration=10.0)
            print(f"✅ Librosa loaded: {len(audio)} samples, {sr}Hz")
            print(f"   Duration: {len(audio) / sr:.2f} seconds")
            print(f"   Min/Max: {audio.min():.3f} / {audio.max():.3f}")
        except Exception as e:
            print(f"❌ Librosa loading failed: {e}")
    
    # Try essentia
    if ESSENTIA_AVAILABLE:
        try:
            print("Trying essentia MonoLoader...")
            loader = es.MonoLoader(
                filename=test_file,
                sampleRate=44100,
                downmix='mix',
                resampleQuality=1
            )
            audio = loader()
            print(f"✅ Essentia loaded: {len(audio)} samples, 44100Hz")
            print(f"   Duration: {len(audio) / 44100:.2f} seconds")
            print(f"   Min/Max: {audio.min():.3f} / {audio.max():.3f}")
        except Exception as e:
            print(f"❌ Essentia loading failed: {e}")
    
    # Try soundfile
    if SOUNDFILE_AVAILABLE:
        try:
            print("Trying soundfile.read...")
            audio, sr = sf.read(test_file)
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            print(f"✅ SoundFile loaded: {len(audio)} samples, {sr}Hz")
            print(f"   Duration: {len(audio) / sr:.2f} seconds")
            print(f"   Min/Max: {audio.min():.3f} / {audio.max():.3f}")
        except Exception as e:
            print(f"❌ SoundFile loading failed: {e}")
    
    print("\n✅ Audio library test completed!")
    return True

if __name__ == "__main__":
    success = test_audio_libraries()
    sys.exit(0 if success else 1) 