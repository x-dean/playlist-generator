"""
REST API interface for the Playlista application.

This module provides a FastAPI-based REST API for programmatic
access to the music analysis and playlist generation system.
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
from uuid import UUID

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import json

from application.services import (
    FileDiscoveryService,
    AudioAnalysisService,
    MetadataEnrichmentService,
    PlaylistGenerationService
)
from application.dtos import (
    FileDiscoveryRequest,
    AudioAnalysisRequest,
    MetadataEnrichmentRequest,
    PlaylistGenerationRequest,
    PlaylistGenerationMethod
)
from infrastructure.file_system.playlist_exporter import PlaylistExporter
from shared.config import get_config
from shared.exceptions import PlaylistaException
from infrastructure.logging import setup_logging


# Pydantic models for API requests/responses
class FileDiscoveryRequestModel(BaseModel):
    search_paths: List[str] = Field(..., description="Paths to search for audio files")
    recursive: bool = Field(False, description="Search recursively")
    file_extensions: Optional[List[str]] = Field(None, description="File extensions to include")
    max_files: Optional[int] = Field(None, description="Maximum number of files to discover")


class FileDiscoveryResponseModel(BaseModel):
    request_id: str
    status: str
    audio_files_count: int
    audio_files: List[Dict[str, Any]]
    processing_time_ms: float


class AudioAnalysisRequestModel(BaseModel):
    file_paths: List[str] = Field(..., description="Paths to audio files to analyze")
    parallel_processing: bool = Field(True, description="Use parallel processing")
    force_reanalysis: bool = Field(False, description="Force re-analysis of existing files")
    max_workers: Optional[int] = Field(None, description="Maximum number of worker processes")


class AudioAnalysisResponseModel(BaseModel):
    request_id: str
    status: str
    results_count: int
    results: List[Dict[str, Any]]
    processing_time_ms: float
    quality_metrics: Dict[str, Any]


class MetadataEnrichmentRequestModel(BaseModel):
    audio_file_ids: List[str] = Field(..., description="Audio file IDs to enrich")
    include_tags: bool = Field(True, description="Include tags from Last.fm")
    include_artist_info: bool = Field(True, description="Include artist information")


class MetadataEnrichmentResponseModel(BaseModel):
    request_id: str
    status: str
    enriched_count: int
    enriched_metadata: List[Dict[str, Any]]
    processing_time_ms: float


class PlaylistGenerationRequestModel(BaseModel):
    audio_file_ids: List[str] = Field(..., description="Audio file IDs to use for playlist generation")
    method: str = Field("kmeans", description="Playlist generation method")
    playlist_size: int = Field(20, description="Number of tracks in playlist")
    quality_threshold: Optional[float] = Field(None, description="Minimum quality threshold")


class PlaylistGenerationResponseModel(BaseModel):
    request_id: str
    status: str
    playlist: Optional[Dict[str, Any]]
    quality_metrics: Optional[Dict[str, Any]]
    processing_time_ms: float
    method_used: str


class PlaylistExportRequestModel(BaseModel):
    playlist_id: str = Field(..., description="Playlist ID to export")
    format: str = Field("m3u", description="Export format (m3u, pls, xspf, json, all)")


class PlaylistExportResponseModel(BaseModel):
    request_id: str
    status: str
    export_paths: Dict[str, str]
    processing_time_ms: float


class HealthResponseModel(BaseModel):
    status: str
    timestamp: str
    version: str
    services: Dict[str, str]


class ErrorResponseModel(BaseModel):
    error: str
    message: str
    timestamp: str
    request_id: Optional[str] = None


# Custom JSON encoder for UUIDs
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, UUID):
            return str(obj)
        return super().default(obj)


class RESTAPI:
    """REST API interface for the Playlista application."""
    
    def __init__(self):
        """Initialize the REST API."""
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
        
        # Initialize services
        self.discovery_service = FileDiscoveryService()
        self.analysis_service = AudioAnalysisService()
        self.enrichment_service = MetadataEnrichmentService()
        self.playlist_service = PlaylistGenerationService()
        self.exporter = PlaylistExporter()
        
        # Setup logging
        setup_logging()
        
        # Create FastAPI app
        self.app = FastAPI(
            title="Playlista API",
            description="Music Analysis and Playlist Generation API",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.app.get("/", response_model=Dict[str, str])
        async def root():
            """Root endpoint."""
            return {
                "message": "Playlista API",
                "version": "1.0.0",
                "docs": "/docs"
            }
        
        @self.app.get("/health", response_model=HealthResponseModel)
        async def health_check():
            """Health check endpoint."""
            return HealthResponseModel(
                status="healthy",
                timestamp=datetime.now().isoformat(),
                version="1.0.0",
                services={
                    "discovery": "ok",
                    "analysis": "ok",
                    "enrichment": "ok",
                    "playlist": "ok",
                    "export": "ok"
                }
            )
        
        @self.app.post("/discover", response_model=FileDiscoveryResponseModel)
        async def discover_files(request: FileDiscoveryRequestModel):
            """Discover audio files in specified paths."""
            try:
                # Convert request
                dto_request = FileDiscoveryRequest(
                    search_paths=request.search_paths,
                    recursive=request.recursive,
                    file_extensions=request.file_extensions,
                    max_files=request.max_files
                )
                
                # Execute service
                response = self.discovery_service.discover_files(dto_request)
                
                # Convert response
                audio_files_data = []
                for audio_file in response.audio_files:
                    audio_files_data.append({
                        "id": str(audio_file.id),
                        "file_name": audio_file.file_name,
                        "file_path": str(audio_file.file_path),
                        "file_size_bytes": audio_file.file_size_bytes,
                        "duration_seconds": audio_file.duration_seconds,
                        "bitrate_kbps": audio_file.bitrate_kbps,
                        "sample_rate_hz": audio_file.sample_rate_hz,
                        "channels": audio_file.channels
                    })
                
                return FileDiscoveryResponseModel(
                    request_id=response.request_id,
                    status=response.status,
                    audio_files_count=len(response.audio_files),
                    audio_files=audio_files_data,
                    processing_time_ms=response.processing_time_ms
                )
                
            except PlaylistaException as e:
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                self.logger.error(f"Discovery error: {e}")
                raise HTTPException(status_code=500, detail="Internal server error")
        
        @self.app.post("/analyze", response_model=AudioAnalysisResponseModel)
        async def analyze_audio_files(request: AudioAnalysisRequestModel):
            """Analyze audio files for features."""
            try:
                # Convert request
                dto_request = AudioAnalysisRequest(
                    file_paths=request.file_paths,
                    parallel_processing=request.parallel_processing,
                    force_reanalysis=request.force_reanalysis,
                    max_workers=request.max_workers
                )
                
                # Execute service
                response = self.analysis_service.analyze_audio_files(dto_request)
                
                # Convert response
                results_data = []
                for result in response.results:
                    results_data.append({
                        "audio_file": {
                            "id": str(result.audio_file.id),
                            "file_name": result.audio_file.file_name,
                            "file_path": str(result.audio_file.file_path)
                        },
                        "feature_set": {
                            "bpm": result.feature_set.bpm,
                            "energy": result.feature_set.energy,
                            "key": result.feature_set.key,
                            "mode": result.feature_set.mode,
                            "danceability": result.feature_set.danceability,
                            "valence": result.feature_set.valence,
                            "acousticness": result.feature_set.acousticness,
                            "instrumentalness": result.feature_set.instrumentalness
                        },
                        "quality_score": result.quality_score,
                        "processing_time_ms": result.processing_time_ms
                    })
                
                return AudioAnalysisResponseModel(
                    request_id=response.request_id,
                    status=response.status,
                    results_count=len(response.results),
                    results=results_data,
                    processing_time_ms=response.processing_time_ms,
                    quality_metrics=response.summary
                )
                
            except PlaylistaException as e:
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                self.logger.error(f"Analysis error: {e}")
                raise HTTPException(status_code=500, detail="Internal server error")
        
        @self.app.post("/enrich", response_model=MetadataEnrichmentResponseModel)
        async def enrich_metadata(request: MetadataEnrichmentRequestModel):
            """Enrich metadata for audio files."""
            try:
                # Convert UUID strings to UUID objects
                audio_file_ids = [UUID(id_str) for id_str in request.audio_file_ids]
                
                # Convert request
                dto_request = MetadataEnrichmentRequest(
                    audio_file_ids=audio_file_ids
                )
                
                # Execute service
                response = self.enrichment_service.enrich_metadata(dto_request)
                
                # Convert response
                metadata_data = []
                for metadata in response.enriched_metadata:
                    metadata_data.append({
                        "audio_file_id": str(metadata.audio_file_id),
                        "title": metadata.title,
                        "artist": metadata.artist,
                        "album": metadata.album,
                        "year": metadata.year,
                        "genre": metadata.genre,
                        "tags": metadata.tags,
                        "play_count": metadata.play_count,
                        "rating": metadata.rating
                    })
                
                return MetadataEnrichmentResponseModel(
                    request_id=response.request_id,
                    status=response.status,
                    enriched_count=len(response.enriched_metadata),
                    enriched_metadata=metadata_data,
                    processing_time_ms=response.processing_time_ms
                )
                
            except PlaylistaException as e:
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                self.logger.error(f"Enrichment error: {e}")
                raise HTTPException(status_code=500, detail="Internal server error")
        
        @self.app.post("/playlist", response_model=PlaylistGenerationResponseModel)
        async def generate_playlist(request: PlaylistGenerationRequestModel):
            """Generate a playlist using specified method."""
            try:
                # Convert UUID strings to UUID objects
                audio_file_ids = [UUID(id_str) for id_str in request.audio_file_ids]
                
                # Convert method string to enum
                method_map = {
                    "kmeans": PlaylistGenerationMethod.KMEANS,
                    "similarity": PlaylistGenerationMethod.SIMILARITY,
                    "feature": PlaylistGenerationMethod.FEATURE_BASED,
                    "random": PlaylistGenerationMethod.RANDOM,
                    "time": PlaylistGenerationMethod.TIME_BASED
                }
                
                method = method_map.get(request.method, PlaylistGenerationMethod.KMEANS)
                
                # For now, we'll create mock audio files
                # In a real app, you'd fetch these from the database
                from domain.entities import AudioFile
                from uuid import uuid4
                
                audio_files = []
                for i, file_id in enumerate(audio_file_ids):
                    audio_file = AudioFile(
                        id=file_id,
                        file_path=f"/music/song_{i}.mp3",
                        file_size_bytes=1024 * 1024 * 5,
                        duration_seconds=180.0,
                        bitrate_kbps=320,
                        sample_rate_hz=44100,
                        channels=2
                    )
                    audio_files.append(audio_file)
                
                # Convert request
                dto_request = PlaylistGenerationRequest(
                    audio_files=audio_files,
                    method=method,
                    playlist_size=request.playlist_size
                )
                
                # Execute service
                response = self.playlist_service.generate_playlist(dto_request)
                
                # Convert response
                playlist_data = None
                if response.playlist:
                    playlist_data = {
                        "id": str(response.playlist.id),
                        "name": response.playlist.name,
                        "description": response.playlist.description,
                        "track_count": len(response.playlist.track_ids),
                        "track_ids": [str(track_id) for track_id in response.playlist.track_ids],
                        "track_paths": response.playlist.track_paths
                    }
                
                quality_metrics_data = None
                if response.quality_metrics:
                    quality_metrics_data = {
                        "diversity_score": response.quality_metrics.diversity_score,
                        "coherence_score": response.quality_metrics.coherence_score,
                        "balance_score": response.quality_metrics.balance_score,
                        "overall_score": response.quality_metrics.overall_score
                    }
                
                return PlaylistGenerationResponseModel(
                    request_id=response.request_id,
                    status=response.status,
                    playlist=playlist_data,
                    quality_metrics=quality_metrics_data,
                    processing_time_ms=response.processing_time_ms,
                    method_used=request.method
                )
                
            except PlaylistaException as e:
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                self.logger.error(f"Playlist generation error: {e}")
                raise HTTPException(status_code=500, detail="Internal server error")
        
        @self.app.post("/export", response_model=PlaylistExportResponseModel)
        async def export_playlist(request: PlaylistExportRequestModel):
            """Export a playlist in specified format."""
            try:
                # In a real app, you'd load the playlist from the database
                # For now, we'll create a sample playlist
                from domain.entities import Playlist
                from uuid import uuid4
                
                sample_playlist = Playlist(
                    id=UUID(request.playlist_id),
                    name="Sample Playlist",
                    description="A sample playlist for export testing",
                    track_ids=[uuid4(), uuid4(), uuid4()],
                    track_paths=["/music/song1.mp3", "/music/song2.mp3", "/music/song3.mp3"]
                )
                
                start_time = datetime.now()
                
                if request.format == "all":
                    exports = self.exporter.export_all_formats(sample_playlist)
                    export_paths = {format_name: str(path) for format_name, path in exports.items()}
                else:
                    if request.format == "m3u":
                        path = self.exporter.export_m3u(sample_playlist)
                    elif request.format == "pls":
                        path = self.exporter.export_pls(sample_playlist)
                    elif request.format == "xspf":
                        path = self.exporter.export_xspf(sample_playlist)
                    elif request.format == "json":
                        path = self.exporter.export_json(sample_playlist)
                    else:
                        raise HTTPException(status_code=400, detail=f"Unsupported format: {request.format}")
                    
                    export_paths = {request.format: str(path)}
                
                processing_time = (datetime.now() - start_time).total_seconds() * 1000
                
                return PlaylistExportResponseModel(
                    request_id=str(uuid4()),
                    status="completed",
                    export_paths=export_paths,
                    processing_time_ms=processing_time
                )
                
            except PlaylistaException as e:
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                self.logger.error(f"Export error: {e}")
                raise HTTPException(status_code=500, detail="Internal server error")
        
        @self.app.get("/methods")
        async def get_playlist_methods():
            """Get available playlist generation methods."""
            return {
                "methods": [
                    {"id": "kmeans", "name": "K-means Clustering", "description": "Group similar tracks using K-means"},
                    {"id": "similarity", "name": "Similarity-based", "description": "Select tracks based on similarity"},
                    {"id": "feature", "name": "Feature-based", "description": "Use audio features for selection"},
                    {"id": "random", "name": "Random Selection", "description": "Random track selection"},
                    {"id": "time", "name": "Time-based", "description": "Select tracks based on time of day"}
                ]
            }
        
        @self.app.get("/formats")
        async def get_export_formats():
            """Get available export formats."""
            return {
                "formats": [
                    {"id": "m3u", "name": "M3U Playlist", "description": "Standard playlist format"},
                    {"id": "pls", "name": "PLS Playlist", "description": "Winamp playlist format"},
                    {"id": "xspf", "name": "XSPF Playlist", "description": "XML Shareable Playlist Format"},
                    {"id": "json", "name": "JSON Format", "description": "Structured JSON format"},
                    {"id": "all", "name": "All Formats", "description": "Export to all supported formats"}
                ]
            }
    
    def get_app(self) -> FastAPI:
        """Get the FastAPI application instance."""
        return self.app


# Create API instance
api = RESTAPI()
app = api.get_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 