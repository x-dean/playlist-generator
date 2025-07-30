#!/usr/bin/env python3
"""
Simple test runner for the analysis system integration tests.
Can be run directly without Docker for development and debugging.
"""

import os
import sys
import unittest
import tempfile
import shutil
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def run_tests_without_docker():
    """Run tests without Docker for development purposes."""
    print("🚀 Running Analysis System Integration Tests (Development Mode)")
    print("=" * 60)
    
    # Check if we can import the required modules
    try:
        from core.analysis_manager import AnalysisManager
        from core.resource_manager import ResourceManager
        from core.audio_analyzer import AudioAnalyzer
        from core.parallel_analyzer import ParallelAnalyzer
        from core.sequential_analyzer import SequentialAnalyzer
        from core.database import DatabaseManager
        print("✅ All core modules imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import core modules: {e}")
        print("Please ensure all dependencies are installed:")
        print("pip install psutil numpy")
        return False
    
    # Import test module
    try:
        from test_analysis_integration import TestAnalysisSystemIntegration, TestDockerCompatibility
        print("✅ Test modules imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import test modules: {e}")
        return False
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTest(unittest.makeSuite(TestAnalysisSystemIntegration))
    suite.addTest(unittest.makeSuite(TestDockerCompatibility))
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 60)
    print("📊 Test Results Summary")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\n❌ Failures:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")
    
    if result.errors:
        print("\n❌ Errors:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")
    
    if result.wasSuccessful():
        print("\n✅ All tests passed!")
        print("\n🎉 Analysis System Integration Test Results:")
        print("✅ Analysis Manager: Deterministic decisions based on file size")
        print("✅ Resource Manager: Forced guidance based on resource constraints")
        print("✅ Audio Analyzer: On/off feature extraction with configuration")
        print("✅ Parallel Analyzer: Simplified worker behavior")
        print("✅ Sequential Analyzer: Large file processing")
        print("✅ Docker Compatibility: Paths and dependencies")
        print("\n📊 Key Test Scenarios Verified:")
        print("  • File size-based analysis decisions (deterministic)")
        print("  • Resource constraint handling (forced basic analysis)")
        print("  • Feature extraction with on/off control")
        print("  • Worker simplification (just do the job)")
        print("  • Docker environment compatibility")
    else:
        print("\n❌ Some tests failed!")
        print("\n🔍 Development Mode Notes:")
        print("  • Some tests may fail in development mode due to missing audio libraries")
        print("  • This is expected - the tests verify the architecture and logic")
        print("  • For full testing with audio processing, use Docker mode")
    
    return result.wasSuccessful()

def check_dependencies():
    """Check if required dependencies are available."""
    print("🔍 Checking Dependencies...")
    
    dependencies = {
        'numpy': 'NumPy for numerical operations',
        'psutil': 'psutil for system resource monitoring',
        'unittest': 'unittest for testing framework',
        'tempfile': 'tempfile for temporary file creation',
        'shutil': 'shutil for file operations'
    }
    
    missing_deps = []
    
    for dep, description in dependencies.items():
        try:
            __import__(dep)
            print(f"✅ {dep}: {description}")
        except ImportError:
            print(f"❌ {dep}: {description} - MISSING")
            missing_deps.append(dep)
    
    if missing_deps:
        print(f"\n⚠️ Missing dependencies: {', '.join(missing_deps)}")
        print("Install missing dependencies with:")
        print("pip install " + " ".join(missing_deps))
        return False
    
    print("✅ All required dependencies are available")
    return True

def check_optional_dependencies():
    """Check optional audio processing dependencies."""
    print("\n🔍 Checking Optional Dependencies...")
    
    optional_deps = {
        'essentia': 'Essentia for advanced audio features',
        'librosa': 'Librosa for audio processing',
        'tensorflow': 'TensorFlow for MusiCNN features',
        'mutagen': 'Mutagen for metadata extraction'
    }
    
    available_deps = []
    
    for dep, description in optional_deps.items():
        try:
            __import__(dep)
            print(f"✅ {dep}: {description}")
            available_deps.append(dep)
        except ImportError:
            print(f"⚠️ {dep}: {description} - NOT AVAILABLE")
    
    if available_deps:
        print(f"\n✅ Available optional dependencies: {', '.join(available_deps)}")
    else:
        print("\n⚠️ No optional audio processing dependencies available")
        print("This is normal for development mode - tests will use mocks")
    
    return True

def main():
    """Main function to run tests."""
    print("🎯 Analysis System Integration Test Runner")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Cannot run tests - missing required dependencies")
        return False
    
    check_optional_dependencies()
    
    print("\n" + "=" * 60)
    print("🚀 Starting Tests...")
    print("=" * 60)
    
    # Run tests
    success = run_tests_without_docker()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 Test run completed successfully!")
    else:
        print("⚠️ Test run completed with some failures")
        print("This is normal in development mode without full audio libraries")
    
    return success

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1) 