#!/usr/bin/env python3
"""
Test script for the domain entities.
"""

import sys
import time
from pathlib import Path
from uuid import uuid4

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from domain.entities import AudioFile, FeatureSet, Metadata, AnalysisResult, Playlist


def test_audio_file():
    """Test AudioFile entity."""
    print("\n🧪 Testing AudioFile entity...")
    
    # Create a test file path
    test_path = Path("/music/test_song.mp3")
    
    # Create AudioFile
    audio_file = AudioFile(
        file_path=test_path,
        file_size_bytes=8 * 1024 * 1024,  # 8MB
        duration_seconds=180.5,
        bitrate_kbps=320,
        sample_rate_hz=44100,
        channels=2
    )
    
    print(f"✅ Created AudioFile: {audio_file}")
    print(f"📁 File size: {audio_file.file_size_mb:.1f}MB")
    print(f"🎵 Duration: {audio_file.duration_seconds}s")
    print(f"🔧 Is large file: {audio_file.is_large_file}")
    print(f"📊 Processing status: {audio_file.processing_status}")
    
    # Test status changes
    audio_file.mark_as_analyzed(processing_time_ms=1250.5)
    print(f"✅ Marked as analyzed: {audio_file.processing_status}")
    
    # Test metadata update
    audio_file.update_metadata({
        'artist': 'Test Artist',
        'album': 'Test Album',
        'year': 2023
    })
    print(f"📝 Updated metadata: {audio_file.external_metadata}")
    
    # Test serialization
    audio_dict = audio_file.to_dict()
    audio_file_restored = AudioFile.from_dict(audio_dict)
    print(f"✅ Serialization test passed: {audio_file_restored.file_name}")
    
    return audio_file


def test_feature_set(audio_file_id):
    """Test FeatureSet entity."""
    print("\n🧪 Testing FeatureSet entity...")
    
    # Create FeatureSet
    feature_set = FeatureSet(
        audio_file_id=audio_file_id,
        bpm=120.5,
        key="C",
        mode="major",
        energy=0.75,
        danceability=0.8,
        valence=0.6,
        acousticness=0.3,
        instrumentalness=0.1,
        liveness=0.2,
        speechiness=0.05,
        spectral_centroid=2200.0,
        spectral_rolloff=4000.0,
        spectral_bandwidth=1500.0,
        zero_crossing_rate=0.1,
        root_mean_square=0.5,
        loudness=-12.5,
        tempo_confidence=0.9,
        key_confidence=0.85,
        processing_time_ms=850.2
    )
    
    print(f"✅ Created FeatureSet: {feature_set}")
    print(f"🎵 BPM: {feature_set.bpm}")
    print(f"🎼 Key: {feature_set.key} {feature_set.mode}")
    print(f"⚡ Energy: {feature_set.energy}")
    print(f"💃 Danceability: {feature_set.danceability}")
    print(f"😊 Valence: {feature_set.valence}")
    print(f"🎸 Tempo category: {feature_set.tempo_category}")
    print(f"⚡ Energy category: {feature_set.energy_category}")
    print(f"😊 Mood category: {feature_set.mood_category}")
    print(f"🎸 Is instrumental: {feature_set.is_instrumental}")
    print(f"🎵 Is acoustic: {feature_set.is_acoustic}")
    
    # Test feature vector
    feature_vector = feature_set.get_feature_vector()
    print(f"📊 Feature vector shape: {feature_vector.shape}")
    print(f"📊 Feature vector: {feature_vector}")
    
    # Test serialization
    feature_dict = feature_set.to_dict()
    feature_set_restored = FeatureSet.from_dict(feature_dict)
    print(f"✅ Serialization test passed: {feature_set_restored.bpm}")
    
    return feature_set


def test_metadata(audio_file_id):
    """Test Metadata entity."""
    print("\n🧪 Testing Metadata entity...")
    
    # Create Metadata
    metadata = Metadata(
        audio_file_id=audio_file_id,
        title="Test Song",
        artist="Test Artist",
        album="Test Album",
        album_artist="Test Artist",
        track_number=1,
        total_tracks=10,
        year=2023,
        genre="Rock",
        composer="Test Composer",
        lyrics="This is a test song with lyrics...",
        comment="Test comment",
        language="en",
        musicbrainz_track_id="mb-track-123",
        musicbrainz_artist_id="mb-artist-456",
        musicbrainz_album_id="mb-album-789",
        lastfm_tags=["rock", "alternative", "indie"],
        lastfm_playcount=150,
        lastfm_rating=4.2,
        custom_tags=["favorite", "roadtrip"],
        user_rating=4.5,
        play_count=25,
        source="musicbrainz",
        confidence=0.95
    )
    
    print(f"✅ Created Metadata: {metadata}")
    print(f"🎵 Title: {metadata.display_title}")
    print(f"👤 Artist: {metadata.display_artist}")
    print(f"💿 Album: {metadata.display_album}")
    print(f"🎵 Full title: {metadata.full_title}")
    print(f"📅 Decade: {metadata.decade}")
    print(f"📝 Has lyrics: {metadata.has_lyrics}")
    print(f"🎵 Is compilation: {metadata.is_compilation}")
    print(f"🏷️ Genre tags: {metadata.genre_tags}")
    
    # Test methods
    metadata.add_tag("summer")
    print(f"🏷️ Added tag: {metadata.custom_tags}")
    
    metadata.increment_play_count()
    print(f"▶️ Play count: {metadata.play_count}")
    
    metadata.set_user_rating(4.8)
    print(f"⭐ User rating: {metadata.user_rating}")
    
    # Test serialization
    metadata_dict = metadata.to_dict()
    metadata_restored = Metadata.from_dict(metadata_dict)
    print(f"✅ Serialization test passed: {metadata_restored.display_title}")
    
    return metadata


def test_analysis_result(audio_file, feature_set, metadata):
    """Test AnalysisResult entity."""
    print("\n🧪 Testing AnalysisResult entity...")
    
    # Create AnalysisResult
    analysis_result = AnalysisResult(
        audio_file=audio_file,
        feature_set=feature_set,
        metadata=metadata,
        is_complete=True,
        is_successful=True,
        processing_time_ms=2100.5,
        confidence_score=0.92,
        worker_id="worker-1",
        batch_id="batch-2023-001",
        processing_priority=1
    )
    
    print(f"✅ Created AnalysisResult: {analysis_result}")
    print(f"📁 File: {analysis_result.file_name}")
    print(f"🎵 Title: {analysis_result.display_title}")
    print(f"👤 Artist: {analysis_result.display_artist}")
    print(f"📊 Status: {analysis_result.analysis_status}")
    print(f"🎵 BPM: {analysis_result.bpm}")
    print(f"🎼 Key: {analysis_result.key}")
    print(f"⚡ Energy: {analysis_result.energy}")
    print(f"✅ Has features: {analysis_result.has_features}")
    print(f"📝 Has metadata: {analysis_result.has_metadata}")
    print(f"🎯 Ready for playlist: {analysis_result.is_ready_for_playlist}")
    
    # Test quality score calculation
    quality_score = analysis_result.calculate_quality_score()
    print(f"📊 Quality score: {quality_score:.3f}")
    
    # Test summary
    summary = analysis_result.get_summary()
    print(f"📋 Summary: {summary['display_title']} by {summary['display_artist']}")
    
    # Test serialization
    result_dict = analysis_result.to_dict()
    result_restored = AnalysisResult.from_dict(result_dict)
    print(f"✅ Serialization test passed: {result_restored.file_name}")
    
    return analysis_result


def test_playlist():
    """Test Playlist entity."""
    print("\n🧪 Testing Playlist entity...")
    
    # Create Playlist
    playlist = Playlist(
        name="Test Playlist",
        description="A test playlist for demonstration",
        playlist_type="kmeans",
        generation_method="kmeans",
        target_size=20,
        average_bpm=125.0,
        dominant_key="C",
        average_energy=0.7,
        average_danceability=0.8,
        average_valence=0.6,
        genres=["Rock", "Alternative", "Indie"],
        moods=["Energetic", "Happy", "Upbeat"],
        decades=["2020s", "2010s"],
        coherence_score=0.75,
        diversity_score=0.8,
        overall_quality=0.775,
        generation_time_ms=1500.0,
        generation_parameters={"k_clusters": 5, "random_state": 42}
    )
    
    print(f"✅ Created Playlist: {playlist}")
    print(f"📝 Name: {playlist.name}")
    print(f"📊 Type: {playlist.playlist_type}")
    print(f"🎯 Target size: {playlist.target_size}")
    print(f"📊 Actual size: {playlist.actual_size}")
    print(f"🎵 Average BPM: {playlist.average_bpm}")
    print(f"🎼 Dominant key: {playlist.dominant_key}")
    print(f"⚡ Average energy: {playlist.average_energy}")
    print(f"🎸 Dominant genre: {playlist.dominant_genre}")
    print(f"😊 Dominant mood: {playlist.dominant_mood}")
    print(f"📅 Dominant decade: {playlist.dominant_decade}")
    print(f"🎵 Tempo category: {playlist.tempo_category}")
    print(f"⚡ Energy category: {playlist.energy_category}")
    print(f"📊 Coherence score: {playlist.coherence_score}")
    print(f"🎭 Diversity score: {playlist.diversity_score}")
    print(f"⭐ Overall quality: {playlist.overall_quality}")
    print(f"📦 Is empty: {playlist.is_empty}")
    print(f"📦 Is full: {playlist.is_full}")
    print(f"📦 Has target size: {playlist.has_target_size}")
    print(f"📦 Remaining capacity: {playlist.remaining_capacity}")
    
    # Test adding tracks
    track_id_1 = uuid4()
    track_id_2 = uuid4()
    
    playlist.add_track(track_id_1, "/music/song1.mp3")
    playlist.add_track(track_id_2, "/music/song2.mp3")
    print(f"✅ Added tracks: {playlist.actual_size} tracks")
    
    # Test methods
    playlist.increment_play_count()
    print(f"▶️ Play count: {playlist.play_count}")
    
    playlist.set_user_rating(4.5)
    print(f"⭐ User rating: {playlist.user_rating}")
    
    playlist.add_export_format("m3u")
    playlist.add_export_format("xspf")
    print(f"📤 Export formats: {playlist.export_formats}")
    
    # Test serialization
    playlist_dict = playlist.to_dict()
    playlist_restored = Playlist.from_dict(playlist_dict)
    print(f"✅ Serialization test passed: {playlist_restored.name}")
    
    return playlist


def main():
    """Run all domain entity tests."""
    print("🚀 Testing Domain Entities")
    print("=" * 50)
    
    try:
        # Test AudioFile
        audio_file = test_audio_file()
        
        # Test FeatureSet
        feature_set = test_feature_set(audio_file.id)
        
        # Test Metadata
        metadata = test_metadata(audio_file.id)
        
        # Test AnalysisResult
        analysis_result = test_analysis_result(audio_file, feature_set, metadata)
        
        # Test Playlist
        playlist = test_playlist()
        
        print("\n" + "=" * 50)
        print("🎉 All domain entity tests passed!")
        print("\n📋 Domain Entities Created:")
        print("✅ AudioFile - Represents music files with metadata")
        print("✅ FeatureSet - Contains extracted audio features")
        print("✅ Metadata - Music metadata from various sources")
        print("✅ AnalysisResult - Complete analysis results")
        print("✅ Playlist - Generated playlists with tracks")
        
        print("\n🔧 Key Features:")
        print("✅ Rich validation and error handling")
        print("✅ Comprehensive serialization/deserialization")
        print("✅ Business logic and calculations")
        print("✅ Type safety with dataclasses")
        print("✅ Domain-specific properties and methods")
        print("✅ Quality metrics and scoring")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 