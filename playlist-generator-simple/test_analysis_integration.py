"""
Integration tests for the reorganized analysis system.
Tests different scenarios for resource management, analysis decisions, and feature extraction.
"""

import os
import sys
import time
import tempfile
import shutil
import unittest
from unittest.mock import Mock, patch, MagicMock
import json
import numpy as np
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import the core modules
from core.analysis_manager import AnalysisManager
from core.resource_manager import ResourceManager
from core.audio_analyzer import AudioAnalyzer
from core.parallel_analyzer import ParallelAnalyzer
from core.sequential_analyzer import SequentialAnalyzer
from core.database import DatabaseManager
from core.config_loader import ConfigLoader


class TestAnalysisSystemIntegration(unittest.TestCase):
    """Test the integrated analysis system behavior."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.music_dir = os.path.join(self.test_dir, 'music')
        os.makedirs(self.music_dir, exist_ok=True)
        
        # Create test configuration
        self.config = {
            'MUSIC_PATH': self.music_dir,
            'BIG_FILE_SIZE_MB': 50,
            'MAX_FULL_ANALYSIS_SIZE_MB': 100,
            'MEMORY_LIMIT_GB': 6.0,
            'CPU_THRESHOLD_PERCENT': 90,
            'DISK_THRESHOLD_PERCENT': 85,
            'ANALYSIS_TIMEOUT_SECONDS': 300,
            'MEMORY_THRESHOLD_PERCENT': 85
        }
        
        # Initialize managers with test config
        self.analysis_manager = AnalysisManager(config=self.config)
        self.resource_manager = ResourceManager(config=self.config)
        self.db_manager = DatabaseManager()
        
        # Create test audio files
        self._create_test_audio_files()
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def _create_test_audio_files(self):
        """Create test audio files of different sizes."""
        # Small file (basic analysis)
        small_file = os.path.join(self.music_dir, 'small_song.mp3')
        self._create_mock_audio_file(small_file, size_mb=10)
        
        # Medium file (full analysis)
        medium_file = os.path.join(self.music_dir, 'medium_song.mp3')
        self._create_mock_audio_file(medium_file, size_mb=75)
        
        # Large file (basic analysis)
        large_file = os.path.join(self.music_dir, 'large_song.mp3')
        self._create_mock_audio_file(large_file, size_mb=150)
        
        self.test_files = [small_file, medium_file, large_file]
    
    def _create_mock_audio_file(self, file_path: str, size_mb: int):
        """Create a mock audio file with specified size."""
        size_bytes = size_mb * 1024 * 1024
        with open(file_path, 'wb') as f:
            f.write(b'0' * size_bytes)
    
    def test_analysis_manager_deterministic_decisions(self):
        """Test that analysis manager makes deterministic decisions based on file size."""
        print("\n=== Testing Analysis Manager Deterministic Decisions ===")
        
        for file_path in self.test_files:
            filename = os.path.basename(file_path)
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            
            # Get analysis configuration
            analysis_config = self.analysis_manager.determine_analysis_type(file_path)
            
            print(f"File: {filename} ({file_size_mb:.1f}MB)")
            print(f"  Analysis Type: {analysis_config['analysis_type']}")
            print(f"  Use Full Analysis: {analysis_config['use_full_analysis']}")
            print(f"  Reason: {analysis_config['reason']}")
            
            # Verify deterministic behavior
            if file_size_mb <= 100:
                self.assertEqual(analysis_config['analysis_type'], 'full')
                self.assertTrue(analysis_config['use_full_analysis'])
                self.assertTrue(analysis_config['features_config']['extract_musicnn'])
            else:
                self.assertEqual(analysis_config['analysis_type'], 'basic')
                self.assertFalse(analysis_config['use_full_analysis'])
                self.assertFalse(analysis_config['features_config']['extract_musicnn'])
    
    def test_resource_manager_forced_guidance(self):
        """Test resource manager's forced guidance based on resource constraints."""
        print("\n=== Testing Resource Manager Forced Guidance ===")
        
        # Test normal conditions
        guidance = self.resource_manager.get_forced_analysis_guidance()
        print(f"Normal conditions: {guidance}")
        self.assertFalse(guidance['force_basic_analysis'])
        
        # Mock high memory usage
        with patch('psutil.virtual_memory') as mock_memory:
            mock_memory.return_value.percent = 95
            mock_memory.return_value.used = 5.5 * (1024**3)  # 5.5GB
            
            # Trigger resource check
            self.resource_manager._update_forced_analysis_state(5.5, 95, 50, 70)
            
            guidance = self.resource_manager.get_forced_analysis_guidance()
            print(f"High memory conditions: {guidance}")
            self.assertTrue(guidance['force_basic_analysis'])
            self.assertIn('High memory usage', guidance['reason'])
    
    def test_audio_analyzer_feature_extraction(self):
        """Test audio analyzer's feature extraction with different configurations."""
        print("\n=== Testing Audio Analyzer Feature Extraction ===")
        
        # Test with basic analysis config
        basic_config = {
            'analysis_type': 'basic',
            'use_full_analysis': False,
            'features_config': {
                'extract_rhythm': True,
                'extract_spectral': True,
                'extract_loudness': True,
                'extract_key': True,
                'extract_mfcc': True,
                'extract_musicnn': False,
                'extract_metadata': True
            }
        }
        
        # Test with full analysis config
        full_config = {
            'analysis_type': 'full',
            'use_full_analysis': True,
            'features_config': {
                'extract_rhythm': True,
                'extract_spectral': True,
                'extract_loudness': True,
                'extract_key': True,
                'extract_mfcc': True,
                'extract_musicnn': True,
                'extract_metadata': True
            }
        }
        
        test_file = self.test_files[0]  # Use small file
        
        # Mock the audio analyzer to avoid actual processing
        with patch.object(AudioAnalyzer, '_safe_audio_load') as mock_load:
            mock_load.return_value = np.random.random(44100)  # 1 second of audio
            
            with patch.object(AudioAnalyzer, '_extract_rhythm_features') as mock_rhythm:
                mock_rhythm.return_value = {'bpm': 120, 'confidence': 0.8}
                
                with patch.object(AudioAnalyzer, '_extract_spectral_features') as mock_spectral:
                    mock_spectral.return_value = {'spectral_centroid': 2000}
                    
                    with patch.object(AudioAnalyzer, '_extract_loudness') as mock_loudness:
                        mock_loudness.return_value = {'loudness': -20}
                        
                        with patch.object(AudioAnalyzer, '_extract_key') as mock_key:
                            mock_key.return_value = {'key': 'C', 'scale': 'major'}
                            
                            with patch.object(AudioAnalyzer, '_extract_mfcc') as mock_mfcc:
                                mock_mfcc.return_value = {'mfcc': [1, 2, 3, 4, 5]}
                                
                                with patch.object(AudioAnalyzer, '_extract_musicnn_features') as mock_musicnn:
                                    mock_musicnn.return_value = {'musicnn_features': [0.1, 0.2, 0.3]}
                                    
                                    # Test basic analysis
                                    analyzer = AudioAnalyzer()
                                    result_basic = analyzer.extract_features(test_file, basic_config)
                                    
                                    print(f"Basic analysis result: {result_basic['analysis_type'] if result_basic else 'None'}")
                                    if result_basic:
                                        print(f"  Features extracted: {list(result_basic['features'].keys())}")
                                        self.assertNotIn('musicnn_features', result_basic['features'])
                                    
                                    # Test full analysis
                                    result_full = analyzer.extract_features(test_file, full_config)
                                    
                                    print(f"Full analysis result: {result_full['analysis_type'] if result_full else 'None'}")
                                    if result_full:
                                        print(f"  Features extracted: {list(result_full['features'].keys())}")
                                        self.assertIn('musicnn_features', result_full['features'])
    
    def test_parallel_analyzer_worker_behavior(self):
        """Test parallel analyzer worker behavior with different configurations."""
        print("\n=== Testing Parallel Analyzer Worker Behavior ===")
        
        # Mock the worker to avoid actual processing
        with patch.object(ParallelAnalyzer, '_process_single_file_worker') as mock_worker:
            mock_worker.return_value = True
            
            analyzer = ParallelAnalyzer()
            
            # Test with small files (should use parallel processing)
            small_files = [self.test_files[0]]  # Small file
            
            result = analyzer.process_files(small_files, force_reextract=False)
            print(f"Parallel processing result: {result}")
            
            self.assertEqual(result['success_count'], 1)
            self.assertEqual(result['failed_count'], 0)
    
    def test_sequential_analyzer_worker_behavior(self):
        """Test sequential analyzer worker behavior with large files."""
        print("\n=== Testing Sequential Analyzer Worker Behavior ===")
        
        # Mock the worker to avoid actual processing
        with patch.object(SequentialAnalyzer, '_extract_features_in_process') as mock_extract:
            mock_extract.return_value = True
            
            analyzer = SequentialAnalyzer()
            
            # Test with large files (should use sequential processing)
            large_files = [self.test_files[2]]  # Large file
            
            result = analyzer.process_files(large_files, force_reextract=False)
            print(f"Sequential processing result: {result}")
            
            self.assertEqual(result['success_count'], 1)
            self.assertEqual(result['failed_count'], 0)
    
    def test_integrated_analysis_flow(self):
        """Test the complete integrated analysis flow."""
        print("\n=== Testing Integrated Analysis Flow ===")
        
        # Mock the audio analyzer to avoid actual processing
        with patch.object(AudioAnalyzer, '_safe_audio_load') as mock_load:
            mock_load.return_value = np.random.random(44100)
            
            with patch.object(AudioAnalyzer, '_extract_rhythm_features') as mock_rhythm:
                mock_rhythm.return_value = {'bpm': 120}
                
                with patch.object(AudioAnalyzer, '_extract_spectral_features') as mock_spectral:
                    mock_spectral.return_value = {'spectral_centroid': 2000}
                    
                    with patch.object(AudioAnalyzer, '_extract_loudness') as mock_loudness:
                        mock_loudness.return_value = {'loudness': -20}
                        
                        with patch.object(AudioAnalyzer, '_extract_key') as mock_key:
                            mock_key.return_value = {'key': 'C'}
                            
                            with patch.object(AudioAnalyzer, '_extract_mfcc') as mock_mfcc:
                                mock_mfcc.return_value = {'mfcc': [1, 2, 3]}
                                
                                # Test complete flow
                                test_file = self.test_files[0]
                                
                                # 1. Analysis manager determines type
                                analysis_config = self.analysis_manager.determine_analysis_type(test_file)
                                print(f"Analysis config: {analysis_config['analysis_type']}")
                                
                                # 2. Resource manager provides guidance
                                guidance = self.resource_manager.get_forced_analysis_guidance()
                                print(f"Resource guidance: {guidance['force_basic_analysis']}")
                                
                                # 3. Audio analyzer extracts features
                                analyzer = AudioAnalyzer()
                                result = analyzer.extract_features(test_file, analysis_config)
                                
                                print(f"Extraction result: {result['analysis_type'] if result else 'None'}")
                                if result:
                                    print(f"  Features: {list(result['features'].keys())}")
                                
                                self.assertIsNotNone(result)
                                self.assertEqual(result['analysis_type'], analysis_config['analysis_type'])
    
    def test_resource_constraint_scenarios(self):
        """Test behavior under different resource constraint scenarios."""
        print("\n=== Testing Resource Constraint Scenarios ===")
        
        scenarios = [
            {
                'name': 'Normal Resources',
                'memory_percent': 60,
                'cpu_percent': 30,
                'disk_percent': 50,
                'expected_force': False
            },
            {
                'name': 'High Memory',
                'memory_percent': 90,
                'cpu_percent': 40,
                'disk_percent': 60,
                'expected_force': True
            },
            {
                'name': 'High CPU',
                'memory_percent': 70,
                'cpu_percent': 85,
                'disk_percent': 50,
                'expected_force': True
            },
            {
                'name': 'High Disk',
                'memory_percent': 65,
                'cpu_percent': 35,
                'disk_percent': 95,
                'expected_force': True
            }
        ]
        
        for scenario in scenarios:
            print(f"\nTesting scenario: {scenario['name']}")
            
            # Mock resource conditions
            with patch('psutil.virtual_memory') as mock_memory:
                mock_memory.return_value.percent = scenario['memory_percent']
                mock_memory.return_value.used = (scenario['memory_percent'] / 100) * 6 * (1024**3)
                
                with patch('psutil.cpu_percent') as mock_cpu:
                    mock_cpu.return_value = scenario['cpu_percent']
                    
                    # Update forced state
                    self.resource_manager._update_forced_analysis_state(
                        scenario['memory_percent'] / 100 * 6,
                        scenario['memory_percent'],
                        scenario['cpu_percent'],
                        scenario['disk_percent']
                    )
                    
                    # Check guidance
                    guidance = self.resource_manager.get_forced_analysis_guidance()
                    print(f"  Force basic: {guidance['force_basic_analysis']}")
                    print(f"  Reason: {guidance['reason']}")
                    
                    self.assertEqual(guidance['force_basic_analysis'], scenario['expected_force'])
    
    def test_file_size_scenarios(self):
        """Test analysis decisions for different file sizes."""
        print("\n=== Testing File Size Scenarios ===")
        
        file_scenarios = [
            {'size_mb': 5, 'expected_type': 'full', 'expected_musicnn': True},
            {'size_mb': 50, 'expected_type': 'full', 'expected_musicnn': True},
            {'size_mb': 100, 'expected_type': 'full', 'expected_musicnn': True},
            {'size_mb': 150, 'expected_type': 'basic', 'expected_musicnn': False},
            {'size_mb': 500, 'expected_type': 'basic', 'expected_musicnn': False}
        ]
        
        for scenario in file_scenarios:
            print(f"\nTesting file size: {scenario['size_mb']}MB")
            
            # Create test file with specific size
            test_file = os.path.join(self.test_dir, f'test_{scenario["size_mb"]}mb.mp3')
            self._create_mock_audio_file(test_file, scenario['size_mb'])
            
            # Get analysis configuration
            analysis_config = self.analysis_manager.determine_analysis_type(test_file)
            
            print(f"  Analysis type: {analysis_config['analysis_type']}")
            print(f"  Use full analysis: {analysis_config['use_full_analysis']}")
            print(f"  MusiCNN enabled: {analysis_config['features_config']['extract_musicnn']}")
            
            # Verify expectations
            self.assertEqual(analysis_config['analysis_type'], scenario['expected_type'])
            self.assertEqual(analysis_config['features_config']['extract_musicnn'], scenario['expected_musicnn'])
    
    def test_worker_simplification(self):
        """Test that workers are simplified and just do the job."""
        print("\n=== Testing Worker Simplification ===")
        
        # Test parallel analyzer worker
        parallel_analyzer = ParallelAnalyzer()
        
        # Mock the analysis manager to return known config
        test_config = {
            'analysis_type': 'basic',
            'use_full_analysis': False,
            'features_config': {
                'extract_rhythm': True,
                'extract_spectral': True,
                'extract_loudness': True,
                'extract_key': True,
                'extract_mfcc': True,
                'extract_musicnn': False,
                'extract_metadata': True
            }
        }
        
        with patch.object(parallel_analyzer, '_get_analysis_config') as mock_config:
            mock_config.return_value = test_config
            
            # Mock the audio analyzer
            with patch.object(AudioAnalyzer, 'extract_features') as mock_extract:
                mock_extract.return_value = {
                    'success': True,
                    'features': {'bpm': 120, 'loudness': -20},
                    'metadata': {'title': 'Test Song'}
                }
                
                # Mock database save
                with patch.object(DatabaseManager, 'save_analysis_result') as mock_save:
                    mock_save.return_value = True
                    
                    # Test worker behavior
                    result = parallel_analyzer._process_single_file_worker(
                        self.test_files[0], force_reextract=False
                    )
                    
                    print(f"Worker result: {result}")
                    print(f"Config used: {mock_config.call_args}")
                    print(f"Extract called: {mock_extract.call_args}")
                    
                    self.assertTrue(result)
                    self.assertEqual(mock_config.call_count, 1)
                    self.assertEqual(mock_extract.call_count, 1)


class TestDockerCompatibility(unittest.TestCase):
    """Test that the system works correctly in Docker environment."""
    
    def test_docker_paths(self):
        """Test that Docker paths are handled correctly."""
        print("\n=== Testing Docker Path Compatibility ===")
        
        # Test with Docker-style paths
        docker_config = {
            'MUSIC_PATH': '/music',
            'CACHE_FILE': '/app/cache/audio_analysis.db',
            'LIBRARY': '/root/music/library'
        }
        
        # Test analysis manager with Docker paths
        analysis_manager = AnalysisManager(config=docker_config)
        print(f"Analysis manager initialized with Docker paths")
        
        # Test resource manager with Docker paths
        resource_manager = ResourceManager(config=docker_config)
        print(f"Resource manager initialized with Docker paths")
        
        # Test audio analyzer with Docker paths
        audio_analyzer = AudioAnalyzer(
            cache_file='/app/cache/audio_analysis.db',
            library='/root/music/library',
            music='/music'
        )
        print(f"Audio analyzer initialized with Docker paths")
        
        self.assertIsNotNone(analysis_manager)
        self.assertIsNotNone(resource_manager)
        self.assertIsNotNone(audio_analyzer)
    
    def test_docker_requirements(self):
        """Test that all required dependencies are available."""
        print("\n=== Testing Docker Requirements ===")
        
        # Test essential imports
        try:
            import numpy as np
            print("✅ NumPy available")
        except ImportError:
            print("❌ NumPy not available")
            self.fail("NumPy is required")
        
        try:
            import psutil
            print("✅ psutil available")
        except ImportError:
            print("❌ psutil not available")
            self.fail("psutil is required")
        
        try:
            import multiprocessing as mp
            print("✅ multiprocessing available")
        except ImportError:
            print("❌ multiprocessing not available")
            self.fail("multiprocessing is required")
        
        # Test optional audio libraries
        try:
            import essentia.standard as es
            print("✅ Essentia available")
        except ImportError:
            print("⚠️ Essentia not available (optional)")
        
        try:
            import librosa
            print("✅ Librosa available")
        except ImportError:
            print("⚠️ Librosa not available (optional)")
        
        try:
            import tensorflow as tf
            print("✅ TensorFlow available")
        except ImportError:
            print("⚠️ TensorFlow not available (optional)")
        
        try:
            from mutagen import File as MutagenFile
            print("✅ Mutagen available")
        except ImportError:
            print("⚠️ Mutagen not available (optional)")


def run_tests():
    """Run all tests with detailed output."""
    print("🚀 Starting Analysis System Integration Tests")
    print("=" * 60)
    
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
    else:
        print("\n❌ Some tests failed!")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1) 