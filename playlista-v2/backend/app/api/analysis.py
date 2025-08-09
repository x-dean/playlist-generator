"""
Analysis API endpoints
"""

import uuid
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_

from ..database import get_db_session, Track, AnalysisJob
from ..core.logging import get_logger, LogContext
from ..analysis.engine import analysis_engine

router = APIRouter()
logger = get_logger("api.analysis")


@router.post("/start")
async def start_analysis_processing(
    background_tasks: BackgroundTasks,
    quick: bool = False,
    limit: Optional[int] = None
) -> Dict[str, str]:
    """Start analysis processing with optional parameters"""
    
    with LogContext(operation="start_analysis_processing"):
        logger.info("Starting analysis processing", quick=quick, limit=limit)
        
        # Start processing in background
        background_tasks.add_task(analysis_engine.start_processing)
        
        return {
            "status": "started",
            "message": f"Analysis processing started ({'quick test' if quick else 'full analysis'})",
            "quick": quick,
            "limit": limit
        }


@router.post("/stop")
async def stop_analysis_processing() -> Dict[str, str]:
    """Stop the analysis processing engine"""
    
    with LogContext(operation="stop_analysis_processing"):
        logger.info("Stopping analysis processing engine")
        
        await analysis_engine.stop_processing()
        
        return {
            "status": "stopped",
            "message": "Analysis processing engine stopped"
        }


@router.get("/jobs")
async def get_analysis_jobs(
    status: Optional[str] = None,
    limit: int = 20,
    offset: int = 0,
    db: AsyncSession = Depends(get_db_session)
) -> List[Dict[str, Any]]:
    """Get analysis jobs list with optional filtering"""
    
    query = select(AnalysisJob)
    
    if status:
        query = query.where(AnalysisJob.status == status)
    
    query = query.order_by(AnalysisJob.created_at.desc()).limit(limit).offset(offset)
    
    result = await db.execute(query)
    jobs = result.scalars().all()
    
    return [
        {
            "id": str(job.id),
            "track_id": str(job.track_id),
            "status": job.status,
            "progress": job.progress_percent or 0,
            "tracks_processed": 1 if job.status == "completed" else 0,
            "total_tracks": 1,
            "created_at": job.created_at.isoformat() if job.created_at else None,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None
        }
        for job in jobs
    ]


@router.get("/status")
async def get_analysis_status(
    db: AsyncSession = Depends(get_db_session)
) -> Dict[str, Any]:
    """Get overall analysis status and statistics"""
    
    # Job counts by status
    job_counts_query = (
        select(AnalysisJob.status, func.count(AnalysisJob.id))
        .group_by(AnalysisJob.status)
    )
    job_counts_result = await db.execute(job_counts_query)
    job_counts = dict(job_counts_result.all())
    
    # Track counts by status
    track_counts_query = (
        select(Track.status, func.count(Track.id))
        .group_by(Track.status)
    )
    track_counts_result = await db.execute(track_counts_query)
    track_counts = dict(track_counts_result.all())
    
    # Recent processing times
    recent_jobs_query = (
        select(AnalysisJob.processing_time)
        .where(
            and_(
                AnalysisJob.status == "completed",
                AnalysisJob.processing_time.isnot(None)
            )
        )
        .order_by(AnalysisJob.completed_at.desc())
        .limit(100)
    )
    recent_jobs_result = await db.execute(recent_jobs_query)
    processing_times = [row[0] for row in recent_jobs_result.all()]
    
    # Calculate statistics
    avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
    
    total_tracks = sum(track_counts.values())
    analyzed_tracks = track_counts.get("analyzed", 0)
    analysis_progress = (analyzed_tracks / total_tracks * 100) if total_tracks > 0 else 0
    
    return {
        "analysis_progress_percent": round(analysis_progress, 2),
        "total_tracks": total_tracks,
        "track_status_breakdown": track_counts,
        "job_status_breakdown": job_counts,
        "processing_statistics": {
            "average_processing_time_seconds": round(avg_processing_time, 2),
            "recent_jobs_count": len(processing_times)
        },
        "engine_status": "running" if analysis_engine._processing else "stopped"
    }


@router.get("/jobs/{job_id}")
async def get_analysis_job(
    job_id: str,
    db: AsyncSession = Depends(get_db_session)
) -> Dict[str, Any]:
    """Get detailed information about a specific analysis job"""
    
    try:
        job_uuid = uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid job ID format")
    
    query = select(AnalysisJob).where(AnalysisJob.id == job_uuid)
    result = await db.execute(query)
    job = result.scalar_one_or_none()
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return {
        "id": str(job.id),
        "track_id": str(job.track_id),
        "status": job.status,
        "priority": job.priority,
        "progress_percent": job.progress_percent,
        "current_step": job.current_step,
        "worker_id": job.worker_id,
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        "processing_time": job.processing_time,
        "error_message": job.error_message,
        "retry_count": job.retry_count,
        "max_retries": job.max_retries
    }


@router.get("/features/{track_id}")
async def get_track_features(
    track_id: str,
    db: AsyncSession = Depends(get_db_session)
) -> Dict[str, Any]:
    """Get extracted features for a specific track"""
    
    try:
        track_uuid = uuid.UUID(track_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid track ID format")
    
    query = select(Track).where(Track.id == track_uuid)
    result = await db.execute(query)
    track = result.scalar_one_or_none()
    
    if not track:
        raise HTTPException(status_code=404, detail="Track not found")
    
    if track.status != "analyzed":
        raise HTTPException(status_code=400, detail="Track not yet analyzed")
    
    return {
        "track_id": str(track.id),
        "analysis_version": track.analysis_version,
        "analyzed_at": track.analyzed_at.isoformat() if track.analyzed_at else None,
        
        # Basic audio features
        "basic_features": {
            "duration": track.duration,
            "bpm": track.bpm,
            "key": track.key,
            "mode": track.mode,
            "loudness": track.loudness,
            "energy": track.energy,
            "danceability": track.danceability,
            "valence": track.valence,
            "acousticness": track.acousticness,
            "instrumentalness": track.instrumentalness,
            "speechiness": track.speechiness,
            "liveness": track.liveness
        },
        
        # Spectral features
        "spectral_features": {
            "spectral_centroid": track.spectral_centroid,
            "spectral_bandwidth": track.spectral_bandwidth,
            "spectral_rolloff": track.spectral_rolloff,
            "spectral_flatness": track.spectral_flatness,
            "zero_crossing_rate": track.zero_crossing_rate
        },
        
        # Mood features
        "mood_features": {
            "happy": track.mood_happy,
            "sad": track.mood_sad,
            "angry": track.mood_angry,
            "relaxed": track.mood_relaxed
        },
        
        # Complex features
        "mfcc_features": track.mfcc_features,
        "chroma_features": track.chroma_features,
        "ml_embeddings": track.ml_embeddings
    }


@router.post("/jobs/{job_id}/retry")
async def retry_analysis_job(
    job_id: str,
    db: AsyncSession = Depends(get_db_session)
) -> Dict[str, str]:
    """Retry a failed analysis job"""
    
    try:
        job_uuid = uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid job ID format")
    
    query = select(AnalysisJob).where(AnalysisJob.id == job_uuid)
    result = await db.execute(query)
    job = result.scalar_one_or_none()
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.status not in ["failed"]:
        raise HTTPException(status_code=400, detail="Job is not in a retryable state")
    
    if job.retry_count >= job.max_retries:
        raise HTTPException(status_code=400, detail="Job has exceeded maximum retry attempts")
    
    # Reset job for retry
    job.status = "queued"
    job.error_message = None
    job.progress_percent = 0
    job.current_step = None
    job.worker_id = None
    job.started_at = None
    job.completed_at = None
    job.processing_time = None
    
    await db.commit()
    
    logger.info(
        "Analysis job queued for retry",
        job_id=str(job.id),
        track_id=str(job.track_id),
        retry_count=job.retry_count
    )
    
    return {
        "status": "queued",
        "message": "Job queued for retry"
    }
