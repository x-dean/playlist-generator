"""
FastAPI routes for Playlist Generator API.
REST endpoints with proper error handling and validation.
"""

import time
import os
# Configure TensorFlow logging BEFORE any imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Hide INFO and WARNING, show only ERROR
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimization messages
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU to avoid GPU-related warnings
import shutil
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.responses import Response
from datetime import datetime

from .models import (
    AnalysisStatus, PlaylistMethod, DatabaseOperation,
    AnalyzeTrackRequest, ImportTracksRequest, GeneratePlaylistRequest,
    SearchTracksRequest, DatabaseManagementRequest,
    TrackResponse, PlaylistResponse, AnalysisStatsResponse,
    ImportResultResponse, SearchResultResponse, DatabaseManagementResponse,
    DatabaseInfoResponse, ErrorResponse, HealthResponse, MetricsResponse,
    StatusResponse, AnalysisResultResponse
)
from .performance_routes import router as performance_router
from ..application.commands import AnalyzeTrackCommand, ImportTracksCommand, GeneratePlaylistCommand
from ..application.queries import GetAnalysisStatsQuery, SearchTracksQuery
from ..application.use_cases import AnalyzeTrackUseCase, ImportTracksUseCase, GeneratePlaylistUseCase, GetAnalysisStatsUseCase
from ..infrastructure.container import get_container
from ..infrastructure.monitoring import get_metrics_collector, get_performance_monitor
from ..domain.exceptions import DomainException, TrackNotFoundException, AnalysisFailedException
from ..core.database import DatabaseManager
from ..core.async_audio_processor import get_async_audio_processor
from ..core.professional_logging import get_logger, LogCategory, LogLevel


router = APIRouter(prefix="/api/v1", tags=["playlist-generator"])

# Include performance monitoring routes
router.include_router(performance_router)


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


def get_database_manager():
    """Dependency to get database manager."""
    return DatabaseManager()


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
            # Get async audio processor
            processor = get_async_audio_processor()
            
            # Execute async analysis
            result = await processor.analyze_track_async(
                file_path=request.file_path,
                force_reanalysis=request.force_reanalysis,
                timeout=300  # 5 minute timeout
            )
            
            if not result:
                raise AnalysisFailedException(f"Failed to analyze track: {request.file_path}")
            
            # Log success with professional logging
            logger.info(
                LogCategory.API, 
                "track_analysis", 
                f"Successfully analyzed track: {request.file_path}",
                operation="analyze_track",
                duration_ms=monitor.get_timing("analyze_track") * 1000,
                metadata={
                    "file_path": request.file_path,
                    "format": result.get("format", "unknown"),
                    "confidence": result.get("confidence", 0.0),
                    "force_reanalysis": request.force_reanalysis
                }
            )
            
            # Record metrics
            metrics.record_track_analysis(
                status="success",
                format=result.get("format", "unknown"),
                duration=monitor.get_timing("analyze_track"),
                confidence=result.get("confidence", 0.0)
            )
            
            return AnalysisResultResponse(
                features=result,
                confidence=result.get("confidence", 0.0),
                analysis_date=result.get("analysis_date", datetime.now().isoformat()),
                processing_time=monitor.get_timing("analyze_track")
            )
            
    except TrackNotFoundException as e:
        logger.error(
            LogCategory.API, 
            "track_analysis", 
            f"Track not found: {request.file_path}",
            error_code="TrackNotFound",
            metadata={"file_path": request.file_path, "error": str(e)}
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except AnalysisFailedException as e:
        logger.error(
            LogCategory.API, 
            "track_analysis", 
            f"Analysis failed for track: {request.file_path}",
            error_code="AnalysisFailed",
            metadata={"file_path": request.file_path, "error": str(e)}
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except Exception as e:
        logger.log_exception(
            LogCategory.API, 
            "track_analysis", 
            f"Unexpected error in track analysis: {request.file_path}",
            exception=e,
            metadata={"file_path": request.file_path}
        )
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


@router.post("/database/manage", response_model=DatabaseManagementResponse)
async def manage_database(
    request: DatabaseManagementRequest,
    db_manager = Depends(get_database_manager),
    logger = Depends(get_logger_dependency),
    monitor = Depends(get_monitor_dependency)
):
    """Perform database management operations."""
    try:
        with monitor.time_operation("database_management"):
            operation = request.operation
            db_path = request.db_path
            details = {}
            
            if operation == DatabaseOperation.INIT:
                db_manager.initialize_schema()
                message = "Database schema initialized successfully"
                
            elif operation == DatabaseOperation.BACKUP:
                backup_path = db_manager.create_backup()
                details = {"backup_path": backup_path}
                message = f"Database backup created: {backup_path}"
                
            elif operation == DatabaseOperation.RESTORE:
                if not request.backup_path:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Backup path is required for restore operation"
                    )
                db_manager.restore_from_backup(request.backup_path)
                message = f"Database restored from: {request.backup_path}"
                
            elif operation == DatabaseOperation.INTEGRITY_CHECK:
                integrity_result = db_manager.check_integrity()
                details = {"integrity_result": integrity_result}
                message = "Database integrity check completed"
                
            elif operation == DatabaseOperation.VACUUM:
                db_manager.vacuum_database()
                message = "Database vacuum completed"
                
            elif operation == DatabaseOperation.CLEANUP:
                cleaned_count = db_manager.cleanup_old_data(request.days_to_keep)
                details = {"cleaned_entries": cleaned_count}
                message = f"Database cleanup completed: {cleaned_count} entries removed"
                
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Unsupported operation: {operation}"
                )
            
            # Log success
            logger.log_use_case_execution("database_management", "success", monitor.get_timing("database_management"))
            
            return DatabaseManagementResponse(
                operation=operation,
                status="success",
                message=message,
                details=details
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error in database management: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database operation failed: {str(e)}"
        )


@router.get("/database/info", response_model=DatabaseInfoResponse)
async def get_database_info(
    db_manager = Depends(get_database_manager),
    logger = Depends(get_logger_dependency)
):
    """Get database information and statistics."""
    try:
        db_path = db_manager.db_path
        
        # Get database size
        db_size_bytes = db_manager.get_database_size()
        db_size_mb = db_size_bytes / (1024 * 1024)
        
        # Get table counts
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            
            # Count tables
            cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
            table_count = cursor.fetchone()[0]
            
            # Count tracks
            cursor.execute("SELECT COUNT(*) FROM tracks")
            track_count = cursor.fetchone()[0]
            
            # Count playlists
            cursor.execute("SELECT COUNT(*) FROM playlists")
            playlist_count = cursor.fetchone()[0]
            
            # Check integrity
            cursor.execute("PRAGMA integrity_check")
            integrity_result = cursor.fetchone()
            integrity_status = "ok" if integrity_result[0] == "ok" else "issues"
            
            # Get file timestamps
            if os.path.exists(db_path):
                stat = os.stat(db_path)
                created_at = datetime.fromtimestamp(stat.st_ctime)
                last_modified = datetime.fromtimestamp(stat.st_mtime)
            else:
                created_at = None
                last_modified = None
        
        return DatabaseInfoResponse(
            db_path=db_path,
            db_size_bytes=db_size_bytes,
            db_size_mb=round(db_size_mb, 2),
            table_count=table_count,
            track_count=track_count,
            playlist_count=playlist_count,
            integrity_status=integrity_status,
            created_at=created_at,
            last_modified=last_modified
        )
        
    except Exception as e:
        logger.exception(f"Error getting database info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get database information"
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