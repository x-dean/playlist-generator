#!/usr/bin/env python3
"""
Quick test with timeouts to avoid hanging.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import logging
import time
import signal
import threading

# Set up basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out")

def run_with_timeout(func, timeout_seconds=30):
    """Run a function with a timeout."""
    def target():
        try:
            return func()
        except Exception as e:
            return e
    
    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()
    thread.join(timeout_seconds)
    
    if thread.is_alive():
        return TimeoutError(f"Function timed out after {timeout_seconds} seconds")
    else:
        return thread.result if hasattr(thread, 'result') else None

def test_file_info():
    """Test basic file information."""
    print("Testing file information...")
    
    test_file = "/music/22Gz - Crime Rate.mp3"
    
    if os.path.exists(test_file):
        stat = os.stat(test_file)
        size_mb = stat.st_size / (1024 * 1024)
        print(f" File exists: {test_file}")
        print(f"   Size: {size_mb:.1f} MB")
        print(f"   Permissions: {oct(stat.st_mode)[-3:]}")
        return True
    else:
        print(f" File not found: {test_file}")
        return False

def test_librosa_import():
    """Test librosa import only."""
    print("\nTesting librosa import...")
    
    try:
        import librosa
        print(" Librosa imported successfully")
        print(f"   Version: {librosa.__version__}")
        return True
    except Exception as e:
        print(f" Librosa import failed: {e}")
        return False

def test_essentia_import():
    """Test essentia import only."""
    print("\nTesting essentia import...")
    
    try:
        import essentia.standard as es
        print(" Essentia imported successfully")
        return True
    except Exception as e:
        print(f" Essentia import failed: {e}")
        return False

def test_analyzer_import():
    """Test analyzer import only."""
    print("\nTesting analyzer import...")
    
    try:
        from core.audio_analyzer import AudioAnalyzer
        print(" AudioAnalyzer imported successfully")
        return True
    except Exception as e:
        print(f" AudioAnalyzer import failed: {e}")
        return False

def test_simple_librosa_load():
    """Test simple librosa load with timeout."""
    print("\nTesting simple librosa load (with timeout)...")
    
    def load_func():
        import librosa
        test_file = "/music/22Gz - Crime Rate.mp3"
        audio, sr = librosa.load(test_file, sr=44100, mono=True, duration=10.0)  # Only load first 10 seconds
        return audio, sr
    
    result = run_with_timeout(load_func, timeout_seconds=15)
    
    if isinstance(result, Exception):
        print(f" Librosa load failed: {result}")
        return False
    elif result is None:
        print(" Librosa load timed out")
        return False
    else:
        audio, sr = result
        print(f" Librosa load successful")
        print(f"   Audio length: {len(audio)} samples")
        print(f"   Sample rate: {sr} Hz")
        print(f"   Duration: {len(audio) / sr:.2f} seconds")
        return True

if __name__ == "__main__":
    print("=" * 60)
    print("QUICK AUDIO TEST")
    print("=" * 60)
    
    # Test file info
    file_ok = test_file_info()
    
    # Test imports
    librosa_ok = test_librosa_import()
    essentia_ok = test_essentia_import()
    analyzer_ok = test_analyzer_import()
    
    # Test simple load
    load_ok = test_simple_librosa_load()
    
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print(f"File exists: {'' if file_ok else ''}")
    print(f"Librosa import: {'' if librosa_ok else ''}")
    print(f"Essentia import: {'' if essentia_ok else ''}")
    print(f"Analyzer import: {'' if analyzer_ok else ''}")
    print(f"Simple load: {'' if load_ok else ''}")
    
    if file_ok and librosa_ok and load_ok:
        print("\n Basic audio functionality works")
        print("The issue may be in the analyzer's complex loading logic")
    elif file_ok and librosa_ok:
        print("\n️ File and librosa work, but loading fails")
        print("There may be an issue with the specific file or loading parameters")
    else:
        print("\n Basic functionality has issues")
        print("Need to fix file access or library imports") 