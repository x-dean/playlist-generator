#!/usr/bin/env python3
"""
Debug script to test audio loading specifically.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import logging
import time

# Set up basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_direct_librosa():
    """Test direct librosa loading."""
    print("Testing direct librosa loading...")
    
    test_file = "/music/22Gz - Crime Rate.mp3"
    
    try:
        import librosa
        print("✅ Librosa imported successfully")
        
        start_time = time.time()
        print(f"Loading audio with librosa: {test_file}")
        
        audio, sr = librosa.load(test_file, sr=44100, mono=True)
        
        load_time = time.time() - start_time
        print(f"✅ Librosa loading successful in {load_time:.2f}s")
        print(f"   Audio length: {len(audio)} samples")
        print(f"   Sample rate: {sr} Hz")
        print(f"   Duration: {len(audio) / sr:.2f} seconds")
        
        return audio
        
    except Exception as e:
        print(f"❌ Librosa loading failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_direct_essentia():
    """Test direct essentia loading."""
    print("\nTesting direct essentia loading...")
    
    test_file = "/music/22Gz - Crime Rate.mp3"
    
    try:
        import essentia.standard as es
        print("✅ Essentia imported successfully")
        
        start_time = time.time()
        print(f"Loading audio with essentia: {test_file}")
        
        loader = es.MonoLoader(
            filename=test_file,
            sampleRate=44100,
            downmix='mix',
            resampleQuality=1
        )
        audio = loader()
        
        load_time = time.time() - start_time
        print(f"✅ Essentia loading successful in {load_time:.2f}s")
        print(f"   Audio length: {len(audio)} samples")
        print(f"   Duration: {len(audio) / 44100:.2f} seconds")
        
        return audio
        
    except Exception as e:
        print(f"❌ Essentia loading failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_analyzer_loading():
    """Test analyzer's audio loading method."""
    print("\nTesting analyzer audio loading...")
    
    try:
        from core.audio_analyzer import AudioAnalyzer
        analyzer = AudioAnalyzer()
        print("✅ Analyzer initialized")
        
        test_file = "/music/22Gz - Crime Rate.mp3"
        
        start_time = time.time()
        print(f"Loading audio with analyzer: {test_file}")
        
        audio = analyzer._safe_audio_load(test_file)
        
        load_time = time.time() - start_time
        if audio is not None:
            print(f"✅ Analyzer loading successful in {load_time:.2f}s")
            print(f"   Audio length: {len(audio)} samples")
            print(f"   Duration: {len(audio) / 44100:.2f} seconds")
            return True
        else:
            print(f"❌ Analyzer loading failed in {load_time:.2f}s")
            return False
            
    except Exception as e:
        print(f"❌ Analyzer loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("AUDIO LOADING DEBUG TEST")
    print("=" * 60)
    
    # Test direct librosa
    librosa_audio = test_direct_librosa()
    
    # Test direct essentia
    essentia_audio = test_direct_essentia()
    
    # Test analyzer loading
    analyzer_success = test_analyzer_loading()
    
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print(f"Librosa loading: {'✅' if librosa_audio is not None else '❌'}")
    print(f"Essentia loading: {'✅' if essentia_audio is not None else '❌'}")
    print(f"Analyzer loading: {'✅' if analyzer_success else '❌'}")
    
    if librosa_audio is not None and essentia_audio is not None:
        print("\n✅ Both libraries can load audio directly")
        print("The issue is likely in the analyzer's loading logic")
    elif librosa_audio is not None:
        print("\n✅ Only librosa works - essentia may have issues")
    elif essentia_audio is not None:
        print("\n✅ Only essentia works - librosa may have issues")
    else:
        print("\n❌ Both libraries failed - there may be a file access issue") 