#!/usr/bin/env python3
"""
Test script to verify the duration calculation fix.
"""

import os
import tempfile
import numpy as np
from src.core.streaming_audio_loader import StreamingAudioLoader

def create_test_audio_file(duration_seconds=10.0, sample_rate=44100):
    """Create a test audio file for testing."""
    # Generate a simple sine wave
    t = np.linspace(0, duration_seconds, int(duration_seconds * sample_rate), False)
    audio = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        # Save as WAV file using scipy if available
        try:
            from scipy.io import wavfile
            wavfile.write(f.name, sample_rate, audio.astype(np.float32))
            return f.name
        except ImportError:
            # Fallback: create a simple binary file (not a real WAV)
            f.write(audio.astype(np.float32).tobytes())
            return f.name

def test_duration_calculation():
    """Test that duration calculation works correctly."""
    print("🧪 Testing duration calculation fix...")
    
    try:
        # Create test audio file
        test_file = create_test_audio_file(duration_seconds=5.0)
        print(f"✅ Created test file: {test_file}")
        
        # Initialize streaming loader
        loader = StreamingAudioLoader()
        print("✅ StreamingAudioLoader initialized")
        
        # Test duration calculation
        duration = loader._get_audio_duration(test_file)
        print(f"✅ Duration calculation: {duration:.2f}s")
        
        if duration is not None and abs(duration - 5.0) < 1.0:
            print("✅ Duration calculation is working correctly!")
            return True
        else:
            print(f"❌ Duration calculation failed: expected ~5.0s, got {duration}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False
    finally:
        # Clean up test file
        if 'test_file' in locals():
            try:
                os.unlink(test_file)
                print("✅ Cleaned up test file")
            except:
                pass

def test_import():
    """Test that the streaming loader can be imported."""
    print("🧪 Testing import...")
    try:
        from src.core.streaming_audio_loader import StreamingAudioLoader
        loader = StreamingAudioLoader()
        print("✅ Import and initialization successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Testing Duration Calculation Fix")
    print("=" * 40)
    
    # Test import
    import_success = test_import()
    
    # Test duration calculation
    duration_success = test_duration_calculation()
    
    print("\n📊 Test Results:")
    print(f"   Import: {'✅ PASSED' if import_success else '❌ FAILED'}")
    print(f"   Duration: {'✅ PASSED' if duration_success else '❌ FAILED'}")
    
    if import_success and duration_success:
        print("\n🎉 All tests passed! The duration calculation fix is working.")
    else:
        print("\n❌ Some tests failed. Please check the errors above.") 