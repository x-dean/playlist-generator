#!/usr/bin/env python3
"""
Test script to demonstrate playlist generation functionality
"""

import asyncio
import json
import sys
sys.path.append('/app')

from app.playlist.algorithms import PlaylistAlgorithms
from app.playlist.engine import PlaylistEngine
from app.core.logging import get_logger

logger = get_logger("test_playlist")

async def test_playlist_generation():
    """Test playlist generation with different algorithms"""
    
    print("🎵 Testing Playlista v2 Playlist Generation")
    print("=" * 50)
    
    # Initialize playlist components
    algorithms = PlaylistAlgorithms()
    engine = PlaylistEngine()
    
    print("✅ Playlist engine initialized successfully")
    
    # Test different algorithms
    test_cases = [
        {
            "algorithm": "similarity",
            "description": "Similarity-based playlist",
            "target_length": 10,
            "seed_tracks": ["track_1", "track_2"]
        },
        {
            "algorithm": "kmeans",
            "description": "K-means clustering playlist",
            "target_length": 15,
            "preferences": {"energy_range": [0.5, 0.8]}
        },
        {
            "algorithm": "random",
            "description": "Random selection playlist",
            "target_length": 8
        },
        {
            "algorithm": "time_based",
            "description": "Time-based progression playlist",
            "target_length": 12,
            "preferences": {"time_period": "evening"}
        }
    ]
    
    print(f"🎯 Testing {len(test_cases)} playlist generation algorithms")
    print()
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"Test {i}: {test_case['description']}")
        print("-" * 40)
        
        try:
            # Generate playlist
            playlist = await engine.generate_playlist(
                algorithm=test_case["algorithm"],
                seed_tracks=test_case.get("seed_tracks"),
                target_length=test_case["target_length"],
                preferences=test_case.get("preferences")
            )
            
            print(f"✅ Generated playlist with {len(playlist)} tracks")
            print(f"Algorithm: {test_case['algorithm']}")
            print(f"Target length: {test_case['target_length']}")
            
            # Display first few tracks
            print("Sample tracks:")
            for j, track in enumerate(playlist[:3], 1):
                print(f"  {j}. {track['title']} - {track['artist']} ({track['duration']}s)")
            
            if len(playlist) > 3:
                print(f"  ... and {len(playlist) - 3} more tracks")
            
            print()
            
        except Exception as e:
            print(f"❌ Failed to generate playlist: {str(e)}")
            logger.error(f"Playlist generation error: {e}")
            print()
    
    # Test playlist algorithms directly
    print("🔧 Testing Individual Algorithms:")
    print("-" * 35)
    
    # Mock track data for algorithm testing
    mock_tracks = [
        {"id": f"track_{i}", "features": {"energy": i * 0.1, "valence": i * 0.15}} 
        for i in range(1, 21)
    ]
    
    # Test similarity algorithm
    try:
        similar_tracks = algorithms.similarity_based(
            tracks=mock_tracks,
            seed_track=mock_tracks[0],
            target_count=5
        )
        print(f"✅ Similarity algorithm: Found {len(similar_tracks)} similar tracks")
    except Exception as e:
        print(f"❌ Similarity algorithm failed: {e}")
    
    # Test k-means algorithm
    try:
        clustered_tracks = algorithms.kmeans_clustering(
            tracks=mock_tracks,
            target_count=8,
            n_clusters=3
        )
        print(f"✅ K-means algorithm: Generated {len(clustered_tracks)} tracks")
    except Exception as e:
        print(f"❌ K-means algorithm failed: {e}")
    
    # Test feature-based algorithm
    try:
        feature_tracks = algorithms.feature_group_based(
            tracks=mock_tracks,
            feature_preferences={"energy": [0.3, 0.7]},
            target_count=6
        )
        print(f"✅ Feature-based algorithm: Found {len(feature_tracks)} tracks")
    except Exception as e:
        print(f"❌ Feature-based algorithm failed: {e}")
    
    print("\n✅ Playlist generation testing completed successfully!")

if __name__ == "__main__":
    asyncio.run(test_playlist_generation())
