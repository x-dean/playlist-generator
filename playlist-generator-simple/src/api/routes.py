"""
FastAPI routes for Playlist Generator API.
REST endpoints with proper error handling and validation.
"""

import time
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.responses import Response

from .models import *
from ..application.commands import AnalyzeTrackCommand, ImportTracksCommand, GeneratePlaylistCommand
from ..application.queries import GetAnalysisStatsQuery, SearchTracksQuery
from ..application.use_cases import AnalyzeTrackUseCase, ImportTracksUseCase, GeneratePlaylistUseCase, GetAnalysisStatsUseCase
from ..infrastructure.container import get_container
from ..infrastructure.logging import get_logger
from ..infrastructure.monitoring import get_metrics_collector, get_performance_monitor
from ..domain.exceptions import DomainException, TrackNotFoundException, AnalysisFailedException


router = APIRouter(prefix="/api/v1", tags=["playlist-generator"])


def get_container_dependency():
    """Dependency to get container instance."""
    return get_container()


def get_logger_dependency():
    """Dependency to get logger instance."""
    return get_logger()


def get_metrics_dependency():
    """Dependency to get metrics collector."""
    return get_metrics_collector()


def get_monitor_dependency():
    """Dependency to get performance monitor."""
    return get_performance_monitor()


@router.post("/tracks/analyze", response_model=AnalysisResultResponse)
async def analyze_track(
    request: AnalyzeTrackRequest,
    container = Depends(get_container_dependency),
    logger = Depends(get_logger_dependency),
    metrics = Depends(get_metrics_dependency),
    monitor = Depends(get_monitor_dependency)
):
    """Analyze a track and return analysis results."""
    try:
        with monitor.time_operation("analyze_track"):
            # Get use case from container
            use_case = container.resolve(AnalyzeTrackUseCase)
            
            # Create command
            command = AnalyzeTrackCommand(
                file_path=request.file_path,
                force_reanalysis=request.force_reanalysis
            )
            
            # Execute use case
            result = use_case.execute(command)
            
            # Log success
            logger.log_use_case_execution("analyze_track", "success", monitor.get_timing("analyze_track"))
            
            # Record metrics
            metrics.record_track_analysis(
                status="success",
                format=result.features.get("format", "unknown"),
                duration=monitor.get_timing("analyze_track"),
                confidence=result.confidence
            )
            
            return AnalysisResultResponse(
                features=result.features,
                confidence=result.confidence,
                analysis_date=result.analysis_date,
                processing_time=result.processing_time
            )
            
    except TrackNotFoundException as e:
        logger.error(f"Track not found: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except AnalysisFailedException as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except Exception as e:
        logger.exception(f"Unexpected error in analyze_track: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.post("/tracks/import", response_model=ImportResultResponse)
async def import_tracks(
    request: ImportTracksRequest,
    container = Depends(get_container_dependency),
    logger = Depends(get_logger_dependency),
    monitor = Depends(get_monitor_dependency)
):
    """Import tracks from directory."""
    try:
        with monitor.time_operation("import_tracks"):
            # Get use case from container
            use_case = container.resolve(ImportTracksUseCase)
            
            # Create command
            command = ImportTracksCommand(
                directory_path=request.directory_path,
                recursive=request.recursive,
                supported_formats=request.supported_formats
            )
            
            # Execute use case
            result = use_case.execute(command)
            
            # Log success
            logger.log_use_case_execution("import_tracks", "success", monitor.get_timing("import_tracks"))
            
            return ImportResultResponse(
                imported=result["imported"],
                skipped=result["skipped"],
                errors=result["errors"],
                total_files=result["total_files"],
                duration=monitor.get_timing("import_tracks")
            )
            
    except Exception as e:
        logger.exception(f"Error in import_tracks: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/playlists/generate", response_model=PlaylistResponse)
async def generate_playlist(
    request: GeneratePlaylistRequest,
    container = Depends(get_container_dependency),
    logger = Depends(get_logger_dependency),
    monitor = Depends(get_monitor_dependency)
):
    """Generate a playlist using specified method."""
    try:
        with monitor.time_operation("generate_playlist"):
            # Get use case from container
            use_case = container.resolve(GeneratePlaylistUseCase)
            
            # Create command
            command = GeneratePlaylistCommand(
                method=request.method.value,
                size=request.size,
                name=request.name,
                parameters=request.parameters
            )
            
            # Execute use case
            playlist = use_case.execute(command)
            
            # Log success
            logger.log_use_case_execution("generate_playlist", "success", monitor.get_timing("generate_playlist"))
            
            return PlaylistResponse.from_orm(playlist)
            
    except Exception as e:
        logger.exception(f"Error in generate_playlist: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/tracks", response_model=List[TrackResponse])
async def get_tracks(
    limit: Optional[int] = 50,
    offset: Optional[int] = 0,
    container = Depends(get_container_dependency),
    logger = Depends(get_logger_dependency)
):
    """Get all tracks with pagination."""
    try:
        # Get repository from container
        track_repo = container.resolve("ITrackRepository")
        
        # Get tracks
        tracks = track_repo.find_all()
        
        # Apply pagination
        start = offset
        end = start + limit
        paginated_tracks = tracks[start:end]
        
        return [TrackResponse.from_orm(track) for track in paginated_tracks]
        
    except Exception as e:
        logger.exception(f"Error in get_tracks: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.get("/tracks/{track_id}", response_model=TrackResponse)
async def get_track(
    track_id: str,
    container = Depends(get_container_dependency),
    logger = Depends(get_logger_dependency)
):
    """Get a specific track by ID."""
    try:
        # Get repository from container
        track_repo = container.resolve("ITrackRepository")
        
        # Find track
        track = track_repo.find_by_id(track_id)
        if not track:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Track not found"
            )
        
        return TrackResponse.from_orm(track)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error in get_track: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.get("/playlists", response_model=List[PlaylistResponse])
async def get_playlists(
    container = Depends(get_container_dependency),
    logger = Depends(get_logger_dependency)
):
    """Get all playlists."""
    try:
        # Get repository from container
        playlist_repo = container.resolve("IPlaylistRepository")
        
        # Get playlists
        playlists = playlist_repo.get_all_playlists()
        
        return [PlaylistResponse.from_orm(playlist) for playlist in playlists]
        
    except Exception as e:
        logger.exception(f"Error in get_playlists: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.get("/playlists/{playlist_id}", response_model=PlaylistResponse)
async def get_playlist(
    playlist_id: str,
    container = Depends(get_container_dependency),
    logger = Depends(get_logger_dependency)
):
    """Get a specific playlist by ID."""
    try:
        # Get repository from container
        playlist_repo = container.resolve("IPlaylistRepository")
        
        # Find playlist
        playlist = playlist_repo.get_playlist(playlist_id)
        if not playlist:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Playlist not found"
            )
        
        return PlaylistResponse.from_orm(playlist)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error in get_playlist: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.get("/stats/analysis", response_model=AnalysisStatsResponse)
async def get_analysis_stats(
    container = Depends(get_container_dependency),
    logger = Depends(get_logger_dependency)
):
    """Get analysis statistics."""
    try:
        # Get use case from container
        use_case = container.resolve(GetAnalysisStatsUseCase)
        
        # Create query
        query = GetAnalysisStatsQuery()
        
        # Execute use case
        stats = use_case.execute(query)
        
        return AnalysisStatsResponse(**stats)
        
    except Exception as e:
        logger.exception(f"Error in get_analysis_stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.get("/health", response_model=HealthResponse)
async def health_check(
    monitor = Depends(get_monitor_dependency)
):
    """Health check endpoint."""
    try:
        system_stats = monitor.get_system_stats()
        
        return HealthResponse(
            status="healthy",
            version="1.0.0",
            timestamp=time.time(),
            uptime=time.time(),  # Simplified uptime
            system_stats=system_stats
        )
        
    except Exception as e:
        return HealthResponse(
            status="unhealthy",
            version="1.0.0",
            timestamp=time.time(),
            uptime=0,
            system_stats={}
        )


@router.get("/metrics")
async def get_metrics(
    metrics = Depends(get_metrics_dependency)
):
    """Get Prometheus metrics."""
    try:
        metrics_data = metrics.get_metrics()
        return Response(
            content=metrics_data,
            media_type="text/plain; version=0.0.4; charset=utf-8"
        )
        
    except Exception as e:
        return Response(
            content="# Error getting metrics\n",
            media_type="text/plain; version=0.0.4; charset=utf-8"
        ) 