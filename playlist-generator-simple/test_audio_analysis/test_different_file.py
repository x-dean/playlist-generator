#!/usr/bin/env python3
"""
Test with a different audio file.
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

def test_smaller_file():
    """Test with a smaller file."""
    print("Testing with a smaller file...")
    
    # Try a smaller file
    test_file = "/music/Alex Warren - Ordinary.mp3"  # 3.0 MB
    
    if not os.path.exists(test_file):
        print(f"❌ Test file not found: {test_file}")
        return False
    
    try:
        import librosa
        print("✅ Librosa imported successfully")
        
        start_time = time.time()
        print(f"Loading smaller file: {test_file}")
        
        # Try loading only first 3 seconds
        audio, sr = librosa.load(test_file, sr=44100, mono=True, duration=3.0)
        
        load_time = time.time() - start_time
        print(f"✅ Librosa load successful in {load_time:.2f}s")
        print(f"   Audio length: {len(audio)} samples")
        print(f"   Sample rate: {sr} Hz")
        print(f"   Duration: {len(audio) / sr:.2f} seconds")
        
        return True
        
    except Exception as e:
        print(f"❌ Librosa load failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_essentia_smaller():
    """Test essentia with smaller file."""
    print("\nTesting essentia with smaller file...")
    
    test_file = "/music/Alex Warren - Ordinary.mp3"
    
    if not os.path.exists(test_file):
        print(f"❌ Test file not found: {test_file}")
        return False
    
    try:
        import essentia.standard as es
        print("✅ Essentia imported successfully")
        
        start_time = time.time()
        print(f"Loading smaller file: {test_file}")
        
        loader = es.MonoLoader(
            filename=test_file,
            sampleRate=44100,
            downmix='mix',
            resampleQuality=1
        )
        audio = loader()
        
        # Truncate to first 3 seconds
        samples_3_sec = 44100 * 3
        if len(audio) > samples_3_sec:
            audio = audio[:samples_3_sec]
        
        load_time = time.time() - start_time
        print(f"✅ Essentia load successful in {load_time:.2f}s")
        print(f"   Audio length: {len(audio)} samples")
        print(f"   Duration: {len(audio) / 44100:.2f} seconds")
        
        return True
        
    except Exception as e:
        print(f"❌ Essentia load failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_analyzer_smaller():
    """Test analyzer with smaller file."""
    print("\nTesting analyzer with smaller file...")
    
    try:
        from core.audio_analyzer import AudioAnalyzer
        
        analyzer = AudioAnalyzer()
        print("✅ Analyzer initialized")
        
        test_file = "/music/Alex Warren - Ordinary.mp3"
        
        if not os.path.exists(test_file):
            print(f"❌ Test file not found: {test_file}")
            return False
        
        start_time = time.time()
        print(f"Loading smaller file with analyzer: {test_file}")
        
        audio = analyzer._safe_audio_load(test_file)
        
        load_time = time.time() - start_time
        if audio is not None:
            print(f"✅ Analyzer load successful in {load_time:.2f}s")
            print(f"   Audio length: {len(audio)} samples")
            print(f"   Duration: {len(audio) / 44100:.2f} seconds")
            return True
        else:
            print(f"❌ Analyzer load failed in {load_time:.2f}s")
            return False
            
    except Exception as e:
        print(f"❌ Analyzer load failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("TESTING WITH SMALLER FILE")
    print("=" * 60)
    
    # Test with smaller file
    librosa_ok = test_smaller_file()
    essentia_ok = test_essentia_smaller()
    analyzer_ok = test_analyzer_smaller()
    
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print(f"Librosa (smaller): {'✅' if librosa_ok else '❌'}")
    print(f"Essentia (smaller): {'✅' if essentia_ok else '❌'}")
    print(f"Analyzer (smaller): {'✅' if analyzer_ok else '❌'}")
    
    if librosa_ok or essentia_ok or analyzer_ok:
        print("\n✅ At least one method works with smaller file")
        print("The issue may be with the specific large file")
    else:
        print("\n❌ All methods failed even with smaller file")
        print("There may be a deeper issue with audio loading") 