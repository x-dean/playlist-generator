#!/usr/bin/env python3
"""
Memory monitoring test for streaming audio loader.
"""

import os
import time
import psutil
import tempfile
import numpy as np
from src.core.streaming_audio_loader import StreamingAudioLoader

def create_large_test_audio_file(duration_seconds=300.0, sample_rate=44100):
    """Create a large test audio file for memory testing."""
    print(f"🎵 Creating large test audio file ({duration_seconds}s)...")
    
    # Generate a simple sine wave
    t = np.linspace(0, duration_seconds, int(duration_seconds * sample_rate), False)
    audio = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        # Save as WAV file using scipy if available
        try:
            from scipy.io import wavfile
            wavfile.write(f.name, sample_rate, audio.astype(np.float32))
            print(f"✅ Created test file: {f.name}")
            print(f"   Size: {os.path.getsize(f.name) / (1024*1024):.1f}MB")
            print(f"   Duration: {duration_seconds}s")
            return f.name
        except ImportError:
            # Fallback: create a simple binary file (not a real WAV)
            f.write(audio.astype(np.float32).tobytes())
            print(f"✅ Created test file: {f.name}")
            print(f"   Size: {os.path.getsize(f.name) / (1024*1024):.1f}MB")
            return f.name

def monitor_memory_usage():
    """Get current memory usage information."""
    try:
        memory = psutil.virtual_memory()
        return {
            'total_gb': memory.total / (1024 ** 3),
            'available_gb': memory.available / (1024 ** 3),
            'used_gb': memory.used / (1024 ** 3),
            'percent_used': memory.percent
        }
    except Exception as e:
        print(f"⚠️ Could not get memory info: {e}")
        return {'available_gb': 1.0, 'percent_used': 0.0}

def test_memory_efficient_streaming():
    """Test memory-efficient streaming."""
    print("🧪 Testing Memory-Efficient Streaming")
    print("=" * 50)
    
    try:
        # Create large test file
        test_file = create_large_test_audio_file(duration_seconds=60.0)  # 1 minute file
        
        # Initialize streaming loader with conservative settings
        loader = StreamingAudioLoader(
            memory_limit_percent=25,  # Very conservative
            chunk_duration_seconds=10  # Small chunks
        )
        
        # Monitor initial memory
        initial_memory = monitor_memory_usage()
        print(f"\n📊 Initial memory usage:")
        print(f"   Total RAM: {initial_memory['total_gb']:.1f}GB")
        print(f"   Available RAM: {initial_memory['available_gb']:.1f}GB")
        print(f"   Used RAM: {initial_memory['used_gb']:.1f}GB ({initial_memory['percent_used']:.1f}%)")
        
        # Process chunks
        chunk_count = 0
        start_time = time.time()
        
        print(f"\n🔄 Processing chunks...")
        for chunk, start_time_chunk, end_time_chunk in loader.load_audio_chunks(test_file):
            chunk_count += 1
            
            # Monitor memory every 3 chunks
            if chunk_count % 3 == 0:
                current_memory = monitor_memory_usage()
                elapsed_time = time.time() - start_time
                print(f"   Chunk {chunk_count}: {start_time_chunk:.1f}s - {end_time_chunk:.1f}s")
                print(f"   Memory: {current_memory['percent_used']:.1f}% ({current_memory['used_gb']:.1f}GB)")
                print(f"   Elapsed: {elapsed_time:.1f}s")
                
                # Check for memory issues
                if current_memory['percent_used'] > 95:
                    print(f"❌ CRITICAL: Memory usage too high! {current_memory['percent_used']:.1f}%")
                    return False
                elif current_memory['percent_used'] > 85:
                    print(f"⚠️ WARNING: High memory usage! {current_memory['percent_used']:.1f}%")
        
        # Final memory check
        final_memory = monitor_memory_usage()
        total_time = time.time() - start_time
        
        print(f"\n📊 Final Results:")
        print(f"   Chunks processed: {chunk_count}")
        print(f"   Total time: {total_time:.1f}s")
        print(f"   Final memory: {final_memory['percent_used']:.1f}% ({final_memory['used_gb']:.1f}GB)")
        print(f"   Memory change: {final_memory['used_gb'] - initial_memory['used_gb']:+.1f}GB")
        
        # Success criteria
        memory_increase = final_memory['used_gb'] - initial_memory['used_gb']
        success = (
            chunk_count > 0 and
            final_memory['percent_used'] < 95 and
            memory_increase < 2.0  # Should not increase by more than 2GB
        )
        
        if success:
            print(f"\n✅ Memory-efficient streaming test PASSED!")
            return True
        else:
            print(f"\n❌ Memory-efficient streaming test FAILED!")
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

def test_memory_pressure():
    """Test behavior under memory pressure."""
    print("\n🧪 Testing Memory Pressure Handling")
    print("=" * 50)
    
    try:
        # Create streaming loader
        loader = StreamingAudioLoader(memory_limit_percent=10)  # Very low limit
        
        # Test memory calculation
        chunk_duration = loader._calculate_optimal_chunk_duration(100.0, 300.0)
        print(f"✅ Conservative chunk duration: {chunk_duration:.1f}s")
        
        # Test memory monitoring
        memory_info = loader._get_current_memory_usage()
        print(f"✅ Memory monitoring: {memory_info['percent_used']:.1f}% used")
        
        return True
        
    except Exception as e:
        print(f"❌ Memory pressure test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Memory Monitoring Test Suite")
    print("=" * 50)
    
    # Test 1: Memory pressure handling
    pressure_success = test_memory_pressure()
    
    # Test 2: Memory-efficient streaming
    streaming_success = test_memory_efficient_streaming()
    
    print(f"\n📊 Test Results:")
    print(f"   Memory Pressure Test: {'✅ PASSED' if pressure_success else '❌ FAILED'}")
    print(f"   Streaming Test: {'✅ PASSED' if streaming_success else '❌ FAILED'}")
    
    if pressure_success and streaming_success:
        print(f"\n🎉 All memory tests passed! Streaming improvements are working.")
    else:
        print(f"\n❌ Some memory tests failed. Please check the issues above.") 