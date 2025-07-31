#!/usr/bin/env python3
"""
Minimal test to debug audio loading issues.
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

def test_file_list():
    """List available files in music directory."""
    print("Listing files in music directory...")
    
    music_dir = "/music"
    if os.path.exists(music_dir):
        files = os.listdir(music_dir)
        print(f"Found {len(files)} files in {music_dir}:")
        for i, file in enumerate(files[:10]):  # Show first 10 files
            file_path = os.path.join(music_dir, file)
            if os.path.isfile(file_path):
                size = os.path.getsize(file_path)
                size_mb = size / (1024 * 1024)
                print(f"  {i+1}. {file} ({size_mb:.1f} MB)")
    else:
        print(f" Music directory not found: {music_dir}")

def test_librosa_with_duration():
    """Test librosa with duration limit."""
    print("\nTesting librosa with duration limit...")
    
    test_file = "/music/22Gz - Crime Rate.mp3"
    
    try:
        import librosa
        print(" Librosa imported successfully")
        
        start_time = time.time()
        print(f"Loading first 5 seconds of audio: {test_file}")
        
        # Try loading only first 5 seconds
        audio, sr = librosa.load(test_file, sr=44100, mono=True, duration=5.0)
        
        load_time = time.time() - start_time
        print(f" Librosa load successful in {load_time:.2f}s")
        print(f"   Audio length: {len(audio)} samples")
        print(f"   Sample rate: {sr} Hz")
        print(f"   Duration: {len(audio) / sr:.2f} seconds")
        
        return True
        
    except Exception as e:
        print(f" Librosa load failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_essentia_with_duration():
    """Test essentia with duration limit."""
    print("\nTesting essentia with duration limit...")
    
    test_file = "/music/22Gz - Crime Rate.mp3"
    
    try:
        import essentia.standard as es
        print(" Essentia imported successfully")
        
        start_time = time.time()
        print(f"Loading first 5 seconds of audio: {test_file}")
        
        # Try loading only first 5 seconds
        loader = es.MonoLoader(
            filename=test_file,
            sampleRate=44100,
            downmix='mix',
            resampleQuality=1
        )
        audio = loader()
        
        # Truncate to first 5 seconds
        samples_5_sec = 44100 * 5
        if len(audio) > samples_5_sec:
            audio = audio[:samples_5_sec]
        
        load_time = time.time() - start_time
        print(f" Essentia load successful in {load_time:.2f}s")
        print(f"   Audio length: {len(audio)} samples")
        print(f"   Duration: {len(audio) / 44100:.2f} seconds")
        
        return True
        
    except Exception as e:
        print(f" Essentia load failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_analyzer_simple():
    """Test analyzer with simple configuration."""
    print("\nTesting analyzer simple loading...")
    
    try:
        from core.audio_analyzer import AudioAnalyzer
        
        # Create analyzer with minimal config
        analyzer = AudioAnalyzer()
        print(" Analyzer initialized")
        
        test_file = "/music/22Gz - Crime Rate.mp3"
        
        start_time = time.time()
        print(f"Loading audio with analyzer: {test_file}")
        
        # Try to load audio directly
        audio = analyzer._safe_audio_load(test_file)
        
        load_time = time.time() - start_time
        if audio is not None:
            print(f" Analyzer load successful in {load_time:.2f}s")
            print(f"   Audio length: {len(audio)} samples")
            print(f"   Duration: {len(audio) / 44100:.2f} seconds")
            return True
        else:
            print(f" Analyzer load failed in {load_time:.2f}s")
            return False
            
    except Exception as e:
        print(f" Analyzer load failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("MINIMAL AUDIO TEST")
    print("=" * 60)
    
    # List available files
    test_file_list()
    
    # Test librosa with duration limit
    librosa_ok = test_librosa_with_duration()
    
    # Test essentia with duration limit
    essentia_ok = test_essentia_with_duration()
    
    # Test analyzer
    analyzer_ok = test_analyzer_simple()
    
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print(f"Librosa (5s): {'' if librosa_ok else ''}")
    print(f"Essentia (5s): {'' if essentia_ok else ''}")
    print(f"Analyzer: {'' if analyzer_ok else ''}")
    
    if librosa_ok or essentia_ok:
        print("\n At least one library can load audio")
        print("The issue may be with the specific file or loading parameters")
    else:
        print("\n All loading methods failed")
        print("There may be an issue with the audio file or libraries") 