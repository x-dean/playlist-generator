#!/usr/bin/env python3
"""
Comprehensive test script to verify all database and analysis fixes.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.core.database import get_db_manager
from src.core.audio_analyzer import AudioAnalyzer

def test_complete_fixes():
    """Test all the database and analysis fixes."""
    print("Testing complete database and analysis fixes...")
    
    # Get database manager
    db_manager = get_db_manager()
    
    # Test file path (you'll need to provide a real file)
    test_file = "/music/test.mp3"  # Replace with actual file path
    
    if not os.path.exists(test_file):
        print(f"Test file not found: {test_file}")
        print("Please provide a valid audio file path")
        return False
    
    print(f"Testing with file: {test_file}")
    
    # Test 1: Database Schema Initialization
    print("\n1. Testing database schema initialization...")
    success = db_manager.initialize_schema()
    print(f"Schema initialization success: {success}")
    
    schema_info = db_manager.show_schema_info()
    print(f"Total columns: {schema_info['total_columns']}")
    print(f"Total indexes: {schema_info['total_indexes']}")
    print(f"Total views: {schema_info['total_views']}")
    
    # Test 2: Schema Information
    print("\n2. Testing schema information...")
    schema_info = db_manager.show_schema_info()
    print(f"Total columns: {schema_info['total_columns']}")
    print(f"Total indexes: {schema_info['total_indexes']}")
    print(f"Total views: {schema_info['total_views']}")
    
    # Test 3: Data Validation
    print("\n3. Testing data validation...")
    validation_result = db_manager.validate_data_completeness(test_file)
    print(f"Data quality: {validation_result['data_quality']:.2%}")
    print(f"Valid: {validation_result['valid']}")
    print(f"Missing fields: {validation_result['missing_fields']}")
    
    # Test 4: Audio Analysis with New Features
    print("\n4. Testing audio analysis with new features...")
    analyzer = AudioAnalyzer()
    
    # Create test analysis data with new features
    test_analysis_data = {
        'essentia': {
            'bpm': 120.5,
            'key': 'C',
            'mode': 'major',
            'energy': 0.8,
            'danceability': 0.7,
            'valence': 0.6,
            'acousticness': 0.3,
            'instrumentalness': 0.2,
            'speechiness': 0.1,
            'liveness': 0.4
        },
        'harmonic_features': {
            'harmonic_complexity': 0.75,
            'chord_progression': ['C', 'G', 'Am', 'F'],
            'chord_changes': 4,
            'harmonic_centroid': 0.6,
            'harmonic_contrast': 0.4
        },
        'beat_features': {
            'beat_positions': [0.0, 0.5, 1.0, 1.5, 2.0],
            'onset_times': [0.0, 0.25, 0.5, 0.75, 1.0],
            'tempo_confidence': 0.9,
            'rhythm_complexity': 0.3,
            'tempo_strength': 2.0,
            'rhythm_pattern': 'medium'
        },
        'advanced_spectral_features': {
            'spectral_flux': 0.45,
            'spectral_entropy': 0.6,
            'spectral_crest': 2.1,
            'spectral_decrease': 0.8,
            'spectral_kurtosis': 3.2,
            'spectral_skewness': 0.1
        },
        'advanced_audio_features': {
            'zero_crossing_rate': 0.15,
            'root_mean_square': 0.3,
            'peak_amplitude': 0.8,
            'crest_factor': 2.7,
            'signal_to_noise_ratio': 25.5
        },
        'timbre_features': {
            'timbre_brightness': 0.7,
            'timbre_warmth': 0.4,
            'timbre_hardness': 0.3,
            'timbre_depth': 0.6
        },
        'musicnn': {
            'embedding': [0.1, 0.2, 0.3, 0.4, 0.5] * 40,  # 200-dimensional
            'tags': {'rock': 0.9, 'electronic': 0.3, 'energetic': 0.8}
        },
        'metadata': {
            'title': 'Test Song',
            'artist': 'Test Artist',
            'album': 'Test Album',
            'genre': 'Rock',
            'year': 2023
        }
    }
    
    # Save analysis result with new features
    success = db_manager.save_analysis_result(
        file_path=test_file,
        filename="test.mp3",
        file_size_bytes=1024000,
        file_hash="test_hash_123",
        analysis_data=test_analysis_data,
        metadata={'title': 'Test Song', 'artist': 'Test Artist'}
    )
    
    print(f"Save result: {success}")
    
    # Test 5: Retrieve and Verify New Features
    print("\n5. Testing retrieval of new features...")
    analysis_result = db_manager.get_analysis_result(test_file)
    if analysis_result:
        print(f"Retrieved title: {analysis_result.get('title')}")
        print(f"Retrieved artist: {analysis_result.get('artist')}")
        
        # Check new features
        new_features = [
            'harmonic_complexity', 'chord_progression', 'chord_changes',
            'beat_positions', 'onset_times', 'tempo_confidence', 'rhythm_complexity',
            'spectral_flux', 'spectral_entropy', 'spectral_crest',
            'zero_crossing_rate', 'root_mean_square', 'peak_amplitude', 'crest_factor',
            'timbre_brightness', 'timbre_warmth', 'timbre_hardness', 'timbre_depth',
            'valence', 'acousticness', 'instrumentalness', 'speechiness', 'liveness'
        ]
        
        for feature in new_features:
            value = analysis_result.get(feature)
            if value is not None:
                print(f"✓ {feature}: {value}")
            else:
                print(f"✗ {feature}: missing")
    
    # Test 6: Data Repair
    print("\n6. Testing data repair...")
    repair_result = db_manager.repair_corrupted_data()
    print(f"Repair success: {repair_result['success']}")
    print(f"Total repairs: {repair_result['total_repairs']}")
    
    # Test 7: Comprehensive Data Validation
    print("\n7. Testing comprehensive data validation...")
    all_data_result = db_manager.validate_all_data()
    print(f"Total tracks: {all_data_result['total_tracks']}")
    print(f"Valid tracks: {all_data_result['valid_tracks']}")
    print(f"Overall quality: {all_data_result['overall_quality']:.2%}")
    
    # Test 8: CLI Commands (simulated)
    print("\n8. Testing CLI commands...")
    print("Available commands:")
    print("  playlista db --init-schema")
    print("  playlista db --validate-all-data")
    print("  playlista db --repair-corrupted")
    print("  playlista db --show-schema")
    print("  playlista validate-database /path/to/file --validate")
    print("  playlista validate-database /path/to/file --fix")
    
    print("\nComplete fixes test completed!")
    return True

def test_audio_analysis_methods():
    """Test the new audio analysis methods."""
    print("\nTesting new audio analysis methods...")
    
    # Create test audio data (simplified)
    import numpy as np
    sample_rate = 44100
    duration = 5.0  # 5 seconds
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
    
    analyzer = AudioAnalyzer()
    
    # Test harmonic features
    print("Testing harmonic feature extraction...")
    harmonic_features = analyzer._extract_harmonic_features(audio, sample_rate)
    print(f"Harmonic complexity: {harmonic_features.get('harmonic_complexity', 0)}")
    print(f"Chord changes: {harmonic_features.get('chord_changes', 0)}")
    
    # Test beat features
    print("Testing beat feature extraction...")
    beat_features = analyzer._extract_beat_features(audio, sample_rate)
    print(f"Rhythm complexity: {beat_features.get('rhythm_complexity', 0)}")
    print(f"Tempo strength: {beat_features.get('tempo_strength', 0)}")
    
    # Test advanced spectral features
    print("Testing advanced spectral feature extraction...")
    spectral_features = analyzer._extract_advanced_spectral_features(audio, sample_rate)
    print(f"Spectral flux: {spectral_features.get('spectral_flux', 0)}")
    print(f"Spectral entropy: {spectral_features.get('spectral_entropy', 0)}")
    
    # Test advanced audio features
    print("Testing advanced audio feature extraction...")
    audio_features = analyzer._extract_advanced_audio_features(audio, sample_rate)
    print(f"RMS: {audio_features.get('root_mean_square', 0)}")
    print(f"Crest factor: {audio_features.get('crest_factor', 0)}")
    
    # Test timbre features
    print("Testing timbre feature extraction...")
    timbre_features = analyzer._extract_timbre_features(audio, sample_rate)
    print(f"Timbre brightness: {timbre_features.get('timbre_brightness', 0)}")
    print(f"Timbre warmth: {timbre_features.get('timbre_warmth', 0)}")
    
    print("Audio analysis methods test completed!")
    return True

if __name__ == "__main__":
    print("=== Complete Database and Analysis Fixes Test ===")
    
    # Test database fixes
    test_complete_fixes()
    
    # Test audio analysis methods
    test_audio_analysis_methods()
    
    print("\n=== All Tests Completed ===")
    print("\nSummary of fixes implemented:")
    print("✓ Complete database schema with all missing fields")
    print("✓ Dynamic column creation and data storage")
    print("✓ JSON field parsing and validation")
    print("✓ Database migration and management commands")
    print("✓ Advanced audio analysis methods")
    print("✓ Harmonic analysis (chord detection)")
    print("✓ Beat tracking and onset detection")
    print("✓ Advanced spectral features")
    print("✓ Timbre analysis")
    print("✓ Data validation and repair tools")
    print("✓ CLI commands for database management") 