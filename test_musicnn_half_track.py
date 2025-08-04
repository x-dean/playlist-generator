#!/usr/bin/env python3
"""
Test script to verify MusicNN half-track logic
"""

def test_musicnn_half_track_logic():
    """Test the MusicNN half-track decision logic"""
    
    # Configuration values
    MUSICNN_MAX_FILE_SIZE_MB = 200
    MUSICNN_HALF_TRACK_THRESHOLD_MB = 50
    
    def is_file_suitable_for_musicnn(file_size_mb):
        """Simulate the model manager's suitability check"""
        return file_size_mb <= MUSICNN_MAX_FILE_SIZE_MB
    
    def should_use_half_track(file_size_mb):
        """Simulate the fixed audio analyzer logic"""
        return file_size_mb > MUSICNN_HALF_TRACK_THRESHOLD_MB and is_file_suitable_for_musicnn(file_size_mb)
    
    # Test cases
    test_cases = [
        (10, "Small file"),
        (25, "Medium file"), 
        (75, "Large file"),
        (150, "Very large file"),
        (250, "Too large file")
    ]
    
    print("MusicNN Half-Track Logic Test")
    print("=" * 50)
    print(f"Config: MUSICNN_MAX_FILE_SIZE_MB={MUSICNN_MAX_FILE_SIZE_MB}")
    print(f"Config: MUSICNN_HALF_TRACK_THRESHOLD_MB={MUSICNN_HALF_TRACK_THRESHOLD_MB}")
    print()
    
    for file_size_mb, description in test_cases:
        suitable = is_file_suitable_for_musicnn(file_size_mb)
        use_half_track = should_use_half_track(file_size_mb)
        
        status = "✅" if suitable else "❌"
        half_track_status = "Half-track" if use_half_track else "Full track"
        
        print(f"{status} {description} ({file_size_mb}MB):")
        print(f"  - Suitable for MusicNN: {suitable}")
        print(f"  - Processing method: {half_track_status}")
        print()

if __name__ == "__main__":
    test_musicnn_half_track_logic() 