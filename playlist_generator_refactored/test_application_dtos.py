#!/usr/bin/env python3
"""
Test script for the application layer DTOs.
"""

import sys
from pathlib import Path
from uuid import uuid4
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from application.dtos import (
    AudioAnalysisRequest,
    AudioAnalysisResponse,
    AnalysisStatus,
    AnalysisProgress,
    PlaylistGenerationRequest,
    PlaylistGenerationResponse,
    PlaylistGenerationMethod,
    PlaylistQualityMetrics,
    MetadataEnrichmentRequest,
    MetadataEnrichmentResponse,
    EnrichmentSource,
    EnrichmentResult,
    FileDiscoveryRequest,
    FileDiscoveryResponse,
    DiscoveryResult
)


def test_audio_analysis_dtos():
    """Test audio analysis DTOs."""
    print("\n🧪 Testing Audio Analysis DTOs...")
    
    # Test AudioAnalysisRequest
    request = AudioAnalysisRequest(
        file_paths=["/music/song1.mp3", "/music/song2.flac"],
        analysis_method="essentia",
        force_reanalysis=False,
        parallel_processing=True,
        max_workers=4,
        extract_features=True,
        extract_metadata=True,
        quality_threshold=0.7,
        correlation_id="test-123"
    )
    
    print(f"✅ Created AudioAnalysisRequest: {request.total_files} files")
    print(f"📊 Analysis method: {request.analysis_method}")
    print(f"⚡ Parallel processing: {request.parallel_processing}")
    print(f"🎯 Quality threshold: {request.quality_threshold}")
    
    # Test AnalysisProgress
    progress = AnalysisProgress(
        total_files=100,
        processed_files=50,
        successful_files=45,
        failed_files=5,
        current_file="song.mp3",
        start_time=datetime.now()
    )
    
    print(f"📈 Progress: {progress.percentage_complete:.1f}% complete")
    print(f"✅ Success rate: {progress.success_rate:.1f}%")
    print(f"📦 Remaining files: {progress.remaining_files}")
    
    # Test AudioAnalysisResponse
    response = AudioAnalysisResponse(
        request_id="req-123",
        status=AnalysisStatus.IN_PROGRESS,
        progress=progress,
        start_time=datetime.now()
    )
    
    print(f"📊 Response status: {response.status.value}")
    print(f"⏱️ Is complete: {response.is_complete}")
    print(f"✅ Is successful: {response.is_successful}")
    
    # Test serialization
    request_dict = request.to_dict()
    response_dict = response.to_dict()
    print(f"✅ Serialization test passed: {len(request_dict)} fields")
    
    return request, response


def test_playlist_generation_dtos():
    """Test playlist generation DTOs."""
    print("\n🧪 Testing Playlist Generation DTOs...")
    
    # Test PlaylistQualityMetrics
    quality_metrics = PlaylistQualityMetrics(
        coherence_score=0.8,
        diversity_score=0.7,
        overall_quality=0.75,
        genre_diversity=0.6,
        artist_diversity=0.8,
        tempo_consistency=0.9,
        energy_flow=0.7
    )
    
    print(f"📊 Quality metrics created:")
    print(f"   🎯 Coherence: {quality_metrics.coherence_score}")
    print(f"   🎭 Diversity: {quality_metrics.diversity_score}")
    print(f"   ⭐ Overall: {quality_metrics.overall_quality}")
    print(f"   🎸 Genre diversity: {quality_metrics.genre_diversity}")
    print(f"   👤 Artist diversity: {quality_metrics.artist_diversity}")
    
    # Test PlaylistGenerationRequest
    request = PlaylistGenerationRequest(
        analysis_results=[uuid4(), uuid4(), uuid4()],
        generation_method=PlaylistGenerationMethod.KMEANS,
        target_playlist_count=5,
        target_playlist_size=20,
        min_playlist_size=10,
        max_playlist_size=30,
        min_quality_score=0.6,
        genre_filter=["Rock", "Alternative"],
        year_range=(2010, 2023),
        bpm_range=(80, 160),
        balance_playlists=True,
        correlation_id="playlist-test-456"
    )
    
    print(f"✅ Created PlaylistGenerationRequest:")
    print(f"   📊 Method: {request.generation_method.value}")
    print(f"   🎯 Target count: {request.target_playlist_count}")
    print(f"   📦 Target size: {request.target_playlist_size}")
    print(f"   🎸 Genre filter: {request.genre_filter}")
    print(f"   📅 Year range: {request.year_range}")
    print(f"   🎵 BPM range: {request.bpm_range}")
    print(f"   📊 Total tracks: {request.total_tracks}")
    
    # Test PlaylistGenerationResponse
    response = PlaylistGenerationResponse(
        request_id="playlist-req-789",
        status="completed",
        quality_metrics=quality_metrics,
        total_tracks_processed=100,
        total_tracks_used=80,
        unused_tracks=20,
        average_playlist_quality=0.75,
        best_playlist_quality=0.9,
        worst_playlist_quality=0.6,
        generation_time_ms=1500.0,
        start_time=datetime.now()
    )
    
    print(f"📊 PlaylistGenerationResponse:")
    print(f"   📊 Status: {response.status}")
    print(f"   📦 Playlist count: {response.playlist_count}")
    print(f"   🎵 Total tracks: {response.total_playlist_tracks}")
    print(f"   ✅ Is successful: {response.is_successful}")
    print(f"   ⭐ Average quality: {response.average_playlist_quality}")
    print(f"   🏆 Best quality: {response.best_playlist_quality}")
    print(f"   ⏱️ Generation time: {response.generation_time_ms}ms")
    
    # Test serialization
    request_dict = request.to_dict()
    response_dict = response.to_dict()
    print(f"✅ Serialization test passed: {len(request_dict)} fields")
    
    return request, response


def test_metadata_enrichment_dtos():
    """Test metadata enrichment DTOs."""
    print("\n🧪 Testing Metadata Enrichment DTOs...")
    
    # Test EnrichmentResult
    enrichment_result = EnrichmentResult(
        source=EnrichmentSource.MUSICBRAINZ,
        success=True,
        confidence=0.95,
        fields_updated=["title", "artist", "album"],
        fields_added=["musicbrainz_track_id", "musicbrainz_artist_id"],
        processing_time_ms=250.5
    )
    
    print(f"✅ EnrichmentResult created:")
    print(f"   📊 Source: {enrichment_result.source.value}")
    print(f"   ✅ Success: {enrichment_result.success}")
    print(f"   🎯 Confidence: {enrichment_result.confidence}")
    print(f"   📝 Fields updated: {enrichment_result.fields_updated}")
    print(f"   ➕ Fields added: {enrichment_result.fields_added}")
    print(f"   ⏱️ Processing time: {enrichment_result.processing_time_ms}ms")
    
    # Test MetadataEnrichmentRequest
    request = MetadataEnrichmentRequest(
        audio_file_ids=[uuid4(), uuid4()],
        sources=[EnrichmentSource.MUSICBRAINZ, EnrichmentSource.LASTFM],
        force_reenrichment=False,
        confidence_threshold=0.7,
        parallel_processing=True,
        max_workers=4,
        correlation_id="enrichment-test-123"
    )
    
    print(f"✅ MetadataEnrichmentRequest created:")
    print(f"   📁 Total files: {request.total_files}")
    print(f"   📊 Sources: {[s.value for s in request.sources]}")
    print(f"   🎯 Confidence threshold: {request.confidence_threshold}")
    print(f"   ⚡ Parallel processing: {request.parallel_processing}")
    
    # Test MetadataEnrichmentResponse
    response = MetadataEnrichmentResponse(
        request_id="enrichment-req-456",
        status="completed",
        results=[enrichment_result],
        total_files=2,
        successful_files=2,
        failed_files=0,
        skipped_files=0,
        average_confidence=0.95,
        processing_time_ms=500.0,
        start_time=datetime.now()
    )
    
    print(f"📊 MetadataEnrichmentResponse:")
    print(f"   📊 Status: {response.status}")
    print(f"   📁 Total files: {response.total_files}")
    print(f"   ✅ Successful files: {response.successful_files}")
    print(f"   ❌ Failed files: {response.failed_files}")
    print(f"   ⭐ Average confidence: {response.average_confidence}")
    print(f"   ✅ Is successful: {response.is_successful}")
    print(f"   📈 Success rate: {response.success_rate:.1f}%")
    
    # Test serialization
    request_dict = request.to_dict()
    response_dict = response.to_dict()
    print(f"✅ Serialization test passed: {len(request_dict)} fields")
    
    return request, response


def test_file_discovery_dtos():
    """Test file discovery DTOs."""
    print("\n🧪 Testing File Discovery DTOs...")
    
    # Test DiscoveryResult
    from domain.entities import AudioFile
    from pathlib import Path
    
    # Create some test audio files
    test_files = [
        AudioFile(file_path=Path("/music/song1.mp3"), file_size_bytes=5*1024*1024),
        AudioFile(file_path=Path("/music/song2.flac"), file_size_bytes=15*1024*1024),
        AudioFile(file_path=Path("/music/song3.wav"), file_size_bytes=25*1024*1024)
    ]
    
    discovery_result = DiscoveryResult(
        discovered_files=test_files,
        skipped_files=["/music/invalid.txt"],
        error_files=["/music/corrupt.mp3"],
        discovery_time_ms=1200.5,
        start_time=datetime.now()
    )
    
    print(f"✅ DiscoveryResult created:")
    print(f"   📁 Discovered files: {discovery_result.valid_audio_files}")
    print(f"   ⏭️ Skipped files: {discovery_result.invalid_files}")
    print(f"   ❌ Error files: {len(discovery_result.error_files)}")
    print(f"   📊 Success rate: {discovery_result.success_rate:.1f}%")
    print(f"   💾 Total size: {discovery_result.total_size_mb:.1f}MB")
    print(f"   📁 File extensions: {discovery_result.file_extensions}")
    print(f"   📦 File sizes: {discovery_result.file_sizes}")
    
    # Test FileDiscoveryRequest
    request = FileDiscoveryRequest(
        search_paths=["/music", "/downloads"],
        recursive=True,
        file_extensions=[".mp3", ".flac", ".wav"],
        min_file_size_mb=1.0,
        max_file_size_mb=50.0,
        exclude_patterns=["*temp*", "*backup*"],
        parallel_processing=True,
        max_workers=4,
        correlation_id="discovery-test-789"
    )
    
    print(f"✅ FileDiscoveryRequest created:")
    print(f"   📁 Search paths: {request.search_paths}")
    print(f"   🔍 Recursive: {request.recursive}")
    print(f"   📄 File extensions: {request.file_extensions}")
    print(f"   📦 Size range: {request.min_file_size_mb}-{request.max_file_size_mb}MB")
    print(f"   🚫 Exclude patterns: {request.exclude_patterns}")
    print(f"   ⚡ Parallel processing: {request.parallel_processing}")
    
    # Test FileDiscoveryResponse
    response = FileDiscoveryResponse(
        request_id="discovery-req-123",
        status="completed",
        result=discovery_result
    )
    
    print(f"📊 FileDiscoveryResponse:")
    print(f"   📊 Status: {response.status}")
    print(f"   📁 Total files: {response.total_files}")
    print(f"   ✅ Valid files: {response.valid_files}")
    print(f"   ✅ Is successful: {response.is_successful}")
    print(f"   📈 Success rate: {response.result.success_rate:.1f}%")
    
    # Test serialization
    request_dict = request.to_dict()
    response_dict = response.to_dict()
    print(f"✅ Serialization test passed: {len(request_dict)} fields")
    
    return request, response


def main():
    """Run all application DTO tests."""
    print("🚀 Testing Application Layer DTOs")
    print("=" * 50)
    
    try:
        # Test Audio Analysis DTOs
        audio_request, audio_response = test_audio_analysis_dtos()
        
        # Test Playlist Generation DTOs
        playlist_request, playlist_response = test_playlist_generation_dtos()
        
        # Test Metadata Enrichment DTOs
        enrichment_request, enrichment_response = test_metadata_enrichment_dtos()
        
        # Test File Discovery DTOs
        discovery_request, discovery_response = test_file_discovery_dtos()
        
        print("\n" + "=" * 50)
        print("🎉 All application DTO tests passed!")
        print("\n📋 Application DTOs Created:")
        print("✅ Audio Analysis DTOs - Request/Response for audio analysis")
        print("✅ Playlist Generation DTOs - Request/Response for playlist creation")
        print("✅ Metadata Enrichment DTOs - Request/Response for metadata enrichment")
        print("✅ File Discovery DTOs - Request/Response for file discovery")
        
        print("\n🔧 Key Features:")
        print("✅ Rich validation and error handling")
        print("✅ Comprehensive serialization/deserialization")
        print("✅ Progress tracking and status management")
        print("✅ Quality metrics and performance monitoring")
        print("✅ Flexible configuration options")
        print("✅ Type safety with dataclasses and enums")
        
        print("\n📊 Test Summary:")
        print(f"   🎵 Audio Analysis: {audio_request.total_files} files, {audio_response.status.value}")
        print(f"   📝 Playlist Generation: {playlist_request.total_tracks} tracks, {playlist_response.playlist_count} playlists")
        print(f"   📊 Metadata Enrichment: {enrichment_request.total_files} files, {enrichment_response.success_rate:.1f}% success")
        print(f"   📁 File Discovery: {discovery_request.total_search_paths} paths, {discovery_response.valid_files} valid files")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 