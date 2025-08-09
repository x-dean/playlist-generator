"""
Music library management API endpoints
"""

import os
import uuid
from typing import List, Optional, Dict, Any
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_

from ..database import get_db_session, Track, AnalysisJob
from ..core.logging import get_logger, log_performance, LogContext, log_operation_start, log_operation_success
from ..core.config import get_settings
from ..utils.websocket_manager import WebSocketManager

router = APIRouter()
logger = get_logger("api.library")
settings = get_settings()
websocket_manager = WebSocketManager()


@router.get("/stats")
async def get_library_stats(db: AsyncSession = Depends(get_db_session)) -> Dict[str, Any]:
    """Get library statistics"""
    with LogContext(operation="get_library_stats"):
        log_operation_start(logger, "library stats")
        
        try:
            from sqlalchemy import text
            
            # Count total tracks
            result = await db.execute(text("SELECT COUNT(*) FROM tracks"))
            total_tracks = result.scalar() or 0
            
            # Count analyzed tracks (with audio_features)
            result = await db.execute(text("SELECT COUNT(*) FROM tracks WHERE audio_features IS NOT NULL"))
            analyzed_tracks = result.scalar() or 0
            
            # Count playlists
            result = await db.execute(text("SELECT COUNT(*) FROM playlists"))
            total_playlists = result.scalar() or 0
            
            # Sum total duration
            result = await db.execute(text("SELECT SUM(duration) FROM tracks WHERE duration IS NOT NULL"))
            total_duration = result.scalar() or 0
            
            stats = {
                "total_tracks": total_tracks,
                "analyzed_tracks": analyzed_tracks,
                "total_playlists": total_playlists,
                "total_duration": float(total_duration)
            }
            
            logger.info(f"Library stats retrieved: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get library stats: {e}")
            return {
                "total_tracks": 0,
                "analyzed_tracks": 0,
                "total_playlists": 0,
                "total_duration": 0.0
            }


class TrackResponse:
    """Track response model"""
    def __init__(self, track: Track):
        self.id = str(track.id)
        self.filename = track.filename
        self.title = track.title
        self.artist = track.artist
        self.album = track.album
        self.year = track.year
        self.genre = track.genre
        self.duration = track.duration or 0
        self.file_size = track.file_size_bytes or 0
        self.bpm = track.bpm
        self.key = track.key
        self.energy = track.energy
        self.danceability = track.danceability
        self.valence = track.valence
        self.status = track.status
        
        # Audio features for frontend compatibility
        self.audio_features = None
        if track.status == "analyzed":
            self.audio_features = {
                "tempo": track.bpm,
                "key": track.key,
                "energy": track.energy,
                "danceability": track.danceability,
                "valence": track.valence,
                "acousticness": track.acousticness,
                "instrumentalness": track.instrumentalness,
                "speechiness": track.speechiness,
                "liveness": track.liveness
            }
        
        self.analyzed_at = track.analyzed_at.isoformat() if track.analyzed_at else None
        self.created_at = track.created_at.isoformat() if track.created_at else None


@router.get("/tracks")
@log_performance("get_tracks")
async def get_tracks(
    page: int = Query(default=1, ge=1),
    per_page: int = Query(default=20, ge=1, le=100),
    status: Optional[str] = Query(default=None),
    genre: Optional[str] = Query(default=None),
    artist: Optional[str] = Query(default=None),
    search: Optional[str] = Query(default=None),
    sort_by: str = Query(default="created_at"),
    sort_order: str = Query(default="desc"),
    db: AsyncSession = Depends(get_db_session)
) -> Dict[str, Any]:
    """Get tracks with filtering and pagination"""
    
    with LogContext(
        operation="get_tracks",
        page=page,
        per_page=per_page,
        has_filters=bool(status or genre or artist or search)
    ):
        # Calculate offset
        offset = (page - 1) * per_page
        
        # Build base query
        query = select(Track)
        count_query = select(func.count(Track.id))
        
        # Apply filters
        filters = []
        if status:
            filters.append(Track.status == status)
        if genre:
            filters.append(Track.genre.ilike(f"%{genre}%"))
        if artist:
            filters.append(Track.artist.ilike(f"%{artist}%"))
        if search:
            filters.append(
                or_(
                    Track.title.ilike(f"%{search}%"),
                    Track.artist.ilike(f"%{search}%"),
                    Track.album.ilike(f"%{search}%")
                )
            )
        
        if filters:
            query = query.where(and_(*filters))
            count_query = count_query.where(and_(*filters))
        
        # Get total count
        total_result = await db.execute(count_query)
        total = total_result.scalar() or 0
        
        # Apply sorting
        sort_column = getattr(Track, sort_by, Track.created_at)
        if sort_order.lower() == "desc":
            query = query.order_by(sort_column.desc())
        else:
            query = query.order_by(sort_column.asc())
        
        # Apply pagination
        query = query.offset(offset).limit(per_page)
        
        result = await db.execute(query)
        tracks = result.scalars().all()
        
        # Calculate pagination info
        total_pages = (total + per_page - 1) // per_page
        
        logger.info(
            "Retrieved tracks",
            count=len(tracks),
            total=total,
            page=page,
            per_page=per_page
        )
        
        return {
            "tracks": [TrackResponse(track).__dict__ for track in tracks],
            "total": total,
            "page": page,
            "per_page": per_page,
            "total_pages": total_pages
        }


@router.get("/tracks/count")
async def get_tracks_count(
    status: Optional[str] = Query(default=None),
    db: AsyncSession = Depends(get_db_session)
) -> Dict[str, int]:
    """Get track counts by status"""
    
    # Base count query
    query = select(func.count(Track.id))
    if status:
        query = query.where(Track.status == status)
    
    result = await db.execute(query)
    total = result.scalar()
    
    # Status breakdown
    status_query = select(Track.status, func.count(Track.id)).group_by(Track.status)
    status_result = await db.execute(status_query)
    status_counts = dict(status_result.all())
    
    return {
        "total": total,
        "by_status": status_counts
    }


@router.get("/tracks/{track_id}")
async def get_track(
    track_id: str,
    db: AsyncSession = Depends(get_db_session)
) -> Dict[str, Any]:
    """Get detailed track information"""
    
    try:
        track_uuid = uuid.UUID(track_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid track ID format")
    
    query = select(Track).where(Track.id == track_uuid)
    result = await db.execute(query)
    track = result.scalar_one_or_none()
    
    if not track:
        raise HTTPException(status_code=404, detail="Track not found")
    
    # Include all audio features for detailed view
    track_data = TrackResponse(track).__dict__
    track_data.update({
        "file_path": track.file_path,
        "file_size_bytes": track.file_size_bytes,
        "sample_rate": track.sample_rate,
        "bitrate": track.bitrate,
        "channels": track.channels,
        "acousticness": track.acousticness,
        "instrumentalness": track.instrumentalness,
        "speechiness": track.speechiness,
        "liveness": track.liveness,
        "spectral_centroid": track.spectral_centroid,
        "mood_happy": track.mood_happy,
        "mood_sad": track.mood_sad,
        "mood_angry": track.mood_angry,
        "mood_relaxed": track.mood_relaxed,
        "mfcc_features": track.mfcc_features,
        "chroma_features": track.chroma_features,
        "ml_embeddings": track.ml_embeddings
    })
    
    return track_data


@router.post("/tracks/scan")
async def scan_library(
    path: str = Query(..., description="Directory path to scan"),
    db: AsyncSession = Depends(get_db_session)
) -> Dict[str, Any]:
    """Scan directory for music files and add to library"""
    
    if not os.path.exists(path):
        raise HTTPException(status_code=400, detail="Path does not exist")
    
    if not os.path.isdir(path):
        raise HTTPException(status_code=400, detail="Path is not a directory")
    
    # Supported audio formats
    audio_extensions = {'.mp3', '.flac', '.wav', '.m4a', '.aac', '.ogg', '.wma'}
    
    discovered_files = []
    added_count = 0
    
    # Scan directory recursively
    for root, dirs, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            file_ext = Path(file).suffix.lower()
            
            if file_ext in audio_extensions:
                # Check if file already exists
                existing_query = select(Track).where(Track.file_path == file_path)
                existing_result = await db.execute(existing_query)
                existing_track = existing_result.scalar_one_or_none()
                
                if not existing_track:
                    # Calculate file hash (simple version for now)
                    file_stats = os.stat(file_path)
                    file_hash = f"{file}{file_stats.st_size}{file_stats.st_mtime}"
                    
                    # Create new track record
                    new_track = Track(
                        file_path=file_path,
                        filename=file,
                        file_hash=file_hash,
                        file_size_bytes=file_stats.st_size,
                        status="discovered"
                    )
                    
                    db.add(new_track)
                    discovered_files.append(file_path)
                    added_count += 1
    
    await db.commit()
    
    logger.info(f"Library scan completed: {added_count} new files discovered")
    
    return {
        "status": "completed",
        "path": path,
        "files_discovered": added_count,
        "files": discovered_files[:10]  # Return first 10 for preview
    }


@router.post("/tracks/{track_id}/analyze")
async def analyze_track(
    track_id: str,
    priority: int = Query(default=0, description="Analysis priority"),
    db: AsyncSession = Depends(get_db_session)
) -> Dict[str, str]:
    """Queue track for analysis"""
    
    try:
        track_uuid = uuid.UUID(track_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid track ID format")
    
    # Check if track exists
    track_query = select(Track).where(Track.id == track_uuid)
    track_result = await db.execute(track_query)
    track = track_result.scalar_one_or_none()
    
    if not track:
        raise HTTPException(status_code=404, detail="Track not found")
    
    # Check if already in analysis queue
    job_query = select(AnalysisJob).where(
        and_(
            AnalysisJob.track_id == track_uuid,
            AnalysisJob.status.in_(["queued", "processing"])
        )
    )
    job_result = await db.execute(job_query)
    existing_job = job_result.scalar_one_or_none()
    
    if existing_job:
        return {
            "status": "already_queued",
            "job_id": str(existing_job.id)
        }
    
    # Create analysis job
    analysis_job = AnalysisJob(
        track_id=track_uuid,
        priority=priority,
        status="queued"
    )
    
    db.add(analysis_job)
    await db.commit()
    
    logger.info(f"Track queued for analysis: {track_id}")
    
    return {
        "status": "queued",
        "job_id": str(analysis_job.id)
    }


@router.post("/tracks/analyze-batch")
async def analyze_batch(
    track_ids: List[str],
    priority: int = Query(default=0),
    db: AsyncSession = Depends(get_db_session)
) -> Dict[str, Any]:
    """Queue multiple tracks for analysis"""
    
    if len(track_ids) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 tracks per batch")
    
    results = []
    
    for track_id in track_ids:
        try:
            track_uuid = uuid.UUID(track_id)
            
            # Check if track exists
            track_query = select(Track).where(Track.id == track_uuid)
            track_result = await db.execute(track_query)
            track = track_result.scalar_one_or_none()
            
            if not track:
                results.append({"track_id": track_id, "status": "not_found"})
                continue
            
            # Check if already queued
            job_query = select(AnalysisJob).where(
                and_(
                    AnalysisJob.track_id == track_uuid,
                    AnalysisJob.status.in_(["queued", "processing"])
                )
            )
            job_result = await db.execute(job_query)
            existing_job = job_result.scalar_one_or_none()
            
            if existing_job:
                results.append({"track_id": track_id, "status": "already_queued"})
                continue
            
            # Create analysis job
            analysis_job = AnalysisJob(
                track_id=track_uuid,
                priority=priority,
                status="queued"
            )
            
            db.add(analysis_job)
            results.append({"track_id": track_id, "status": "queued"})
            
        except ValueError:
            results.append({"track_id": track_id, "status": "invalid_id"})
    
    await db.commit()
    
    queued_count = sum(1 for r in results if r["status"] == "queued")
    logger.info(f"Batch analysis queued: {queued_count} tracks")
    
    return {
        "total_requested": len(track_ids),
        "queued": queued_count,
        "results": results
    }


@router.get("/analysis/jobs")
async def get_analysis_jobs(
    status: Optional[str] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    db: AsyncSession = Depends(get_db_session)
) -> List[Dict[str, Any]]:
    """Get analysis job status"""
    
    query = select(AnalysisJob)
    
    if status:
        query = query.where(AnalysisJob.status == status)
    
    query = query.order_by(AnalysisJob.priority.desc(), AnalysisJob.created_at.asc()).limit(limit)
    
    result = await db.execute(query)
    jobs = result.scalars().all()
    
    return [
        {
            "id": str(job.id),
            "track_id": str(job.track_id),
            "status": job.status,
            "priority": job.priority,
            "progress_percent": job.progress_percent,
            "current_step": job.current_step,
            "created_at": job.created_at.isoformat() if job.created_at else None,
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "processing_time": job.processing_time,
            "error_message": job.error_message
        }
        for job in jobs
    ]


@router.get("/stats")
async def get_library_stats(
    db: AsyncSession = Depends(get_db_session)
) -> Dict[str, Any]:
    """Get library statistics"""
    
    # Basic counts
    total_tracks_query = select(func.count(Track.id))
    total_tracks_result = await db.execute(total_tracks_query)
    total_tracks = total_tracks_result.scalar()
    
    # Status breakdown
    status_query = select(Track.status, func.count(Track.id)).group_by(Track.status)
    status_result = await db.execute(status_query)
    status_counts = dict(status_result.all())
    
    # Genre breakdown (top 10)
    genre_query = (
        select(Track.genre, func.count(Track.id))
        .where(Track.genre.isnot(None))
        .group_by(Track.genre)
        .order_by(func.count(Track.id).desc())
        .limit(10)
    )
    genre_result = await db.execute(genre_query)
    genre_counts = dict(genre_result.all())
    
    # Analysis stats
    analyzed_query = select(func.count(Track.id)).where(Track.status == "analyzed")
    analyzed_result = await db.execute(analyzed_query)
    analyzed_count = analyzed_result.scalar()
    
    # Average audio features (for analyzed tracks)
    features_query = select(
        func.avg(Track.bpm),
        func.avg(Track.energy),
        func.avg(Track.danceability),
        func.avg(Track.valence),
        func.avg(Track.duration)
    ).where(Track.status == "analyzed")
    features_result = await db.execute(features_query)
    avg_features = features_result.first()
    
    return {
        "total_tracks": total_tracks,
        "analyzed_tracks": analyzed_count,
        "analysis_percentage": round((analyzed_count / total_tracks * 100), 2) if total_tracks > 0 else 0,
        "status_breakdown": status_counts,
        "top_genres": genre_counts,
        "average_features": {
            "bpm": round(avg_features[0], 2) if avg_features[0] else None,
            "energy": round(avg_features[1], 3) if avg_features[1] else None,
            "danceability": round(avg_features[2], 3) if avg_features[2] else None,
            "valence": round(avg_features[3], 3) if avg_features[3] else None,
            "duration": round(avg_features[4], 2) if avg_features[4] else None
        }
    }
