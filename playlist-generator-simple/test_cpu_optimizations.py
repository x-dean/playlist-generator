#!/usr/bin/env python3
"""
Test script for CPU optimizations for small models like MusicNN.
"""

import os
import time
import tempfile
import numpy as np
from src.core.cpu_optimized_analyzer import get_cpu_optimized_analyzer

def create_test_audio_files(num_files=4, duration_seconds=30.0, sample_rate=44100):
    """Create test audio files for testing."""
    test_files = []
    
    print(f"🎵 Creating {num_files} test audio files...")
    
    for i in range(num_files):
        # Generate a simple sine wave with different frequencies
        t = np.linspace(0, duration_seconds, int(duration_seconds * sample_rate), False)
        frequency = 440 + (i * 100)  # Different frequency for each file
        audio = np.sin(2 * np.pi * frequency * t)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            # Save as WAV file using scipy if available
            try:
                from scipy.io import wavfile
                wavfile.write(f.name, sample_rate, audio.astype(np.float32))
                test_files.append(f.name)
                print(f"✅ Created test file {i+1}: {f.name} ({frequency}Hz)")
            except ImportError:
                # Fallback: create a simple binary file
                f.write(audio.astype(np.float32).tobytes())
                test_files.append(f.name)
                print(f"✅ Created test file {i+1}: {f.name} ({frequency}Hz)")
    
    return test_files

def test_sequential_vs_parallel():
    """Test sequential vs parallel melspectrogram extraction."""
    print("🧪 Testing Sequential vs Parallel Processing")
    print("=" * 50)
    
    # Create test files
    test_files = create_test_audio_files(num_files=4)
    
    try:
        # Test sequential processing
        print("\n📊 Testing Sequential Processing...")
        analyzer = get_cpu_optimized_analyzer(num_workers=1)
        
        start_time = time.time()
        sequential_results = analyzer.extract_melspectrograms_batch(test_files)
        sequential_time = time.time() - start_time
        
        print(f"✅ Sequential processing: {len(sequential_results)} melspectrograms in {sequential_time:.2f}s")
        
        # Test parallel processing
        print("\n📊 Testing Parallel Processing...")
        analyzer_parallel = get_cpu_optimized_analyzer(num_workers=3)
        
        start_time = time.time()
        parallel_results = analyzer_parallel.extract_melspectrograms_batch(test_files)
        parallel_time = time.time() - start_time
        
        print(f"✅ Parallel processing: {len(parallel_results)} melspectrograms in {parallel_time:.2f}s")
        
        # Calculate speedup
        if sequential_time > 0:
            speedup = sequential_time / parallel_time
            print(f"\n📈 Speedup: {speedup:.2f}x")
            print(f"   Sequential: {sequential_time:.2f}s")
            print(f"   Parallel: {parallel_time:.2f}s")
            print(f"   Time saved: {sequential_time - parallel_time:.2f}s")
        
        return len(sequential_results) > 0 and len(parallel_results) > 0
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    finally:
        # Clean up test files
        for file in test_files:
            try:
                os.unlink(file)
            except:
                pass

def test_musicnn_optimization():
    """Test MusicNN-specific optimizations."""
    print("\n🧪 Testing MusicNN Optimization")
    print("=" * 50)
    
    # Create test files
    test_files = create_test_audio_files(num_files=2)
    
    try:
        analyzer = get_cpu_optimized_analyzer(num_workers=2)
        
        print("🎵 Processing with MusicNN optimizations...")
        start_time = time.time()
        
        results = analyzer.optimize_for_musicnn(test_files)
        
        processing_time = time.time() - start_time
        
        print(f"✅ MusicNN optimization completed in {processing_time:.2f}s")
        
        # Analyze results
        successful = sum(1 for r in results if r['success'])
        print(f"📊 Results: {successful}/{len(results)} successful")
        
        for i, result in enumerate(results):
            if result['success']:
                print(f"   File {i+1}: {result['shape']} shape, {result['duration']:.1f}s duration")
                print(f"   Model ready: {result['model_ready']}")
                print(f"   Model type: {result['model_type']}")
                print(f"   Sample rate: {result['sample_rate']}Hz")
            else:
                print(f"   File {i+1}: Failed - {result.get('error', 'Unknown error')}")
        
        return successful > 0
        
    except Exception as e:
        print(f"❌ MusicNN test failed: {e}")
        return False
    finally:
        # Clean up test files
        for file in test_files:
            try:
                os.unlink(file)
            except:
                pass

def test_batch_processing():
    """Test batch processing capabilities."""
    print("\n🧪 Testing Batch Processing")
    print("=" * 50)
    
    # Create more test files
    test_files = create_test_audio_files(num_files=8)
    
    try:
        analyzer = get_cpu_optimized_analyzer(num_workers=2)
        
        print("🔄 Testing batch processing...")
        start_time = time.time()
        
        results = analyzer.process_audio_batch(test_files, batch_size=3)
        
        processing_time = time.time() - start_time
        
        print(f"✅ Batch processing completed in {processing_time:.2f}s")
        
        # Analyze results
        successful = sum(1 for r in results if r['success'])
        print(f"📊 Results: {successful}/{len(results)} successful")
        
        # Show batch statistics
        batch_sizes = []
        for i in range(0, len(results), 3):
            batch = results[i:i+3]
            batch_successful = sum(1 for r in batch if r['success'])
            batch_sizes.append(len(batch))
            print(f"   Batch {i//3 + 1}: {batch_successful}/{len(batch)} successful")
        
        return successful > 0
        
    except Exception as e:
        print(f"❌ Batch processing test failed: {e}")
        return False
    finally:
        # Clean up test files
        for file in test_files:
            try:
                os.unlink(file)
            except:
                pass

def test_processing_stats():
    """Test processing statistics."""
    print("\n🧪 Testing Processing Statistics")
    print("=" * 50)
    
    try:
        analyzer = get_cpu_optimized_analyzer(num_workers=2)
        
        stats = analyzer.get_processing_stats()
        
        print("📊 Processing Statistics:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"❌ Statistics test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 CPU Optimization Test Suite")
    print("=" * 50)
    
    # Test 1: Sequential vs Parallel
    sequential_parallel_success = test_sequential_vs_parallel()
    
    # Test 2: MusicNN Optimization
    musicnn_success = test_musicnn_optimization()
    
    # Test 3: Batch Processing
    batch_success = test_batch_processing()
    
    # Test 4: Processing Statistics
    stats_success = test_processing_stats()
    
    print(f"\n📊 Test Results:")
    print(f"   Sequential vs Parallel: {'✅ PASSED' if sequential_parallel_success else '❌ FAILED'}")
    print(f"   MusicNN Optimization: {'✅ PASSED' if musicnn_success else '❌ FAILED'}")
    print(f"   Batch Processing: {'✅ PASSED' if batch_success else '❌ FAILED'}")
    print(f"   Processing Statistics: {'✅ PASSED' if stats_success else '❌ FAILED'}")
    
    if all([sequential_parallel_success, musicnn_success, batch_success, stats_success]):
        print(f"\n🎉 All CPU optimization tests passed!")
        print(f"✅ Multi-process melspectrogram extraction working")
        print(f"✅ MusicNN-specific optimizations working")
        print(f"✅ Batch processing working")
        print(f"✅ Processing statistics working")
    else:
        print(f"\n❌ Some CPU optimization tests failed. Please check the issues above.") 