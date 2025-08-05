#!/usr/bin/env python3
"""
Test script to verify database fixes.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.core.database import get_db_manager
from src.core.audio_analyzer import AudioAnalyzer

def test_database_fixes():
    """Test the database fixes."""
    print("Testing database fixes...")
    
    # Get database manager
    db_manager = get_db_manager()
    
    # Test file path (you'll need to provide a real file)
    test_file = "/music/test.mp3"  # Replace with actual file path
    
    if not os.path.exists(test_file):
        print(f"Test file not found: {test_file}")
        print("Please provide a valid audio file path")
        return False
    
    print(f"Testing with file: {test_file}")
    
    # Test 1: Validate data completeness
    print("\n1. Testing data completeness validation...")
    validation_result = db_manager.validate_data_completeness(test_file)
    
    print(f"Overall data quality: {validation_result['data_quality']:.2%}")
    print(f"Valid: {validation_result['valid']}")
    
    if validation_result['missing_fields']:
        print("Missing fields by category:")
        for category, fields in validation_result['missing_fields'].items():
            print(f"  {category}: {', '.join(fields)}")
    
    # Test 2: Get analysis result with parsed JSON
    print("\n2. Testing analysis result retrieval with JSON parsing...")
    analysis_result = db_manager.get_analysis_result(test_file)
    
    if analysis_result:
        print(f"Title: {analysis_result.get('title', 'Unknown')}")
        print(f"Artist: {analysis_result.get('artist', 'Unknown')}")
        print(f"Duration: {analysis_result.get('duration', 'Unknown')}")
        
        # Check JSON fields
        json_fields = ['bpm_estimates', 'mfcc_coefficients', 'embedding', 'tags']
        for field in json_fields:
            if field in analysis_result and analysis_result[field]:
                if isinstance(analysis_result[field], (list, dict)):
                    print(f"{field}: {type(analysis_result[field]).__name__} with {len(analysis_result[field])} items")
                else:
                    print(f"{field}: {type(analysis_result[field]).__name__}")
    else:
        print("No analysis result found")
    
    # Test 3: Test dynamic column creation
    print("\n3. Testing dynamic column creation...")
    test_features = {
        'test_feature': 123.45,
        'nested': {
            'nested_feature': 'test_value',
            'nested_array': [1, 2, 3]
        }
    }
    
    with db_manager._get_db_connection() as conn:
        cursor = conn.cursor()
        columns = db_manager._ensure_dynamic_columns(cursor, test_features)
        print(f"Available columns: {len(columns)}")
    
    # Test 4: Test data saving with dynamic approach
    print("\n4. Testing dynamic data saving...")
    test_analysis_data = {
        'essentia': {
            'bpm': 120.5,
            'key': 'C',
            'mode': 'major',
            'energy': 0.8,
            'danceability': 0.7
        },
        'musicnn': {
            'embedding': [0.1, 0.2, 0.3],
            'tags': {'rock': 0.9, 'electronic': 0.3}
        },
        'metadata': {
            'title': 'Test Song',
            'artist': 'Test Artist',
            'album': 'Test Album'
        }
    }
    
    success = db_manager.save_analysis_result(
        file_path=test_file,
        filename="test.mp3",
        file_size_bytes=1024000,
        file_hash="test_hash_123",
        analysis_data=test_analysis_data,
        metadata={'title': 'Test Song', 'artist': 'Test Artist'}
    )
    
    print(f"Save result: {success}")
    
    # Test 5: Verify the saved data
    print("\n5. Verifying saved data...")
    saved_result = db_manager.get_analysis_result(test_file)
    if saved_result:
        print(f"Retrieved title: {saved_result.get('title')}")
        print(f"Retrieved artist: {saved_result.get('artist')}")
        print(f"Retrieved bpm: {saved_result.get('bpm')}")
        print(f"Retrieved energy: {saved_result.get('energy')}")
        
        # Check if JSON fields are properly parsed
        if 'embedding' in saved_result and saved_result['embedding']:
            print(f"Embedding type: {type(saved_result['embedding'])}")
            if isinstance(saved_result['embedding'], list):
                print(f"Embedding length: {len(saved_result['embedding'])}")
    
    print("\nDatabase fixes test completed!")
    return True

if __name__ == "__main__":
    test_database_fixes() 