"""
Playlist generation and management API endpoints
"""

import time
import uuid
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete, and_

from ..database import get_db_session, Track, Playlist, PlaylistItem
from ..core.logging import get_logger, LogContext, log_operation_success
from ..playlist import PlaylistAlgorithms, PlaylistEngine

router = APIRouter()
logger = get_logger("api.playlists")
algorithms = PlaylistAlgorithms()
engine = PlaylistEngine()


@router.post("/generate")
async def generate_playlist(
    method: str = Query(..., description="Generation method"),
    size: int = Query(default=25, ge=5, le=200, description="Playlist size"),
    seed_track_id: Optional[str] = Query(default=None, description="Seed track ID for similarity"),
    energy_progression: Optional[str] = Query(default="ascending", description="Energy progression type"),
    start_mood: Optional[str] = Query(default="happy", description="Starting mood for journey"),
    end_mood: Optional[str] = Query(default="calm", description="Ending mood for journey"),
    name: Optional[str] = Query(default=None, description="Playlist name"),
    db: AsyncSession = Depends(get_db_session)
) -> Dict[str, Any]:
    """Generate a new playlist using specified algorithm"""
    
    start_time = time.time()
    
    with LogContext(
        operation="generate_playlist",
        method=method,
        size=size,
        has_seed=bool(seed_track_id)
    ):
        # Get analyzed tracks
        tracks_query = select(Track).where(Track.status == "analyzed")
        tracks_result = await db.execute(tracks_query)
        tracks = list(tracks_result.scalars().all())
        
        if not tracks:
            raise HTTPException(status_code=400, detail="No analyzed tracks available")
        
        logger.info(
            "Starting playlist generation",
            method=method,
            available_tracks=len(tracks),
            target_size=size
        )
        
        # Convert tracks to dict format for algorithms
        track_dicts = [_track_to_dict(track) for track in tracks]
        
        # Generate playlist based on method
        if method == "similarity":
            if not seed_track_id:
                raise HTTPException(status_code=400, detail="Seed track ID required for similarity method")
            
            # Get seed track
            seed_track = await _get_track_by_id(db, seed_track_id)
            if not seed_track:
                raise HTTPException(status_code=404, detail="Seed track not found")
            
            seed_dict = _track_to_dict(seed_track)
            generated_tracks = algorithms.generate_similar_tracks_playlist(
                track_dicts, seed_dict, size
            )
            
        elif method == "kmeans":
            num_clusters = min(5, size // 3)
            tracks_per_cluster = size // num_clusters
            generated_tracks = algorithms.generate_kmeans_playlist(
                track_dicts, num_clusters, tracks_per_cluster
            )
            
        elif method == "energy_flow":
            generated_tracks = algorithms.generate_energy_flow_playlist(
                track_dicts, size, energy_progression
            )
            
        elif method == "mood_journey":
            generated_tracks = algorithms.generate_mood_journey_playlist(
                track_dicts, start_mood, end_mood, size
            )
            
        elif method == "harmonic_mixing":
            generated_tracks = algorithms.generate_harmonic_mixing_playlist(
                track_dicts, size
            )
            
        else:
            raise HTTPException(status_code=400, detail=f"Unknown generation method: {method}")
        
        # Create playlist in database
        generation_time = time.time() - start_time
        
        playlist_name = name or f"{method.title()} Playlist {int(time.time())}"
        playlist = await _create_playlist(
            db, playlist_name, method, generated_tracks, generation_time
        )
        
        log_operation_success(
            logger,
            "playlist generation",
            generation_time * 1000,
            method=method,
            generated_size=len(generated_tracks),
            playlist_id=str(playlist.id)
        )
        
        return {
            "id": str(playlist.id),
            "name": playlist.name,
            "method": playlist.generation_method,
            "track_count": len(generated_tracks),
            "generation_time_seconds": round(generation_time, 2),
            "tracks": [
                {
                    "id": track["id"],
                    "title": track["title"],
                    "artist": track["artist"],
                    "album": track["album"],
                    "duration": track["duration"],
                    "bpm": track["bpm"],
                    "energy": track["energy"],
                    "key": track["key"]
                }
                for track in generated_tracks
            ]
        }


@router.get("/")
async def get_playlists(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    db: AsyncSession = Depends(get_db_session)
) -> List[Dict[str, Any]]:
    """Get all playlists"""
    
    query = (
        select(Playlist)
        .order_by(Playlist.created_at.desc())
        .offset(offset)
        .limit(limit)
    )
    
    result = await db.execute(query)
    playlists = list(result.scalars().all())
    
    return [
        {
            "id": str(playlist.id),
            "name": playlist.name,
            "description": playlist.description,
            "generation_method": playlist.generation_method,
            "track_count": playlist.track_count,
            "total_duration": playlist.total_duration,
            "avg_bpm": playlist.avg_bpm,
            "avg_energy": playlist.avg_energy,
            "created_at": playlist.created_at.isoformat() if playlist.created_at else None,
            "generation_time": playlist.generation_time
        }
        for playlist in playlists
    ]


@router.get("/{playlist_id}")
async def get_playlist(
    playlist_id: str,
    db: AsyncSession = Depends(get_db_session)
) -> Dict[str, Any]:
    """Get detailed playlist information including tracks"""
    
    try:
        playlist_uuid = uuid.UUID(playlist_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid playlist ID format")
    
    # Get playlist
    playlist_query = select(Playlist).where(Playlist.id == playlist_uuid)
    playlist_result = await db.execute(playlist_query)
    playlist = playlist_result.scalar_one_or_none()
    
    if not playlist:
        raise HTTPException(status_code=404, detail="Playlist not found")
    
    # Get playlist tracks with track information
    tracks_query = (
        select(PlaylistItem, Track)
        .join(Track, PlaylistItem.track_id == Track.id)
        .where(PlaylistItem.playlist_id == playlist_uuid)
        .order_by(PlaylistItem.position)
    )
    
    tracks_result = await db.execute(tracks_query)
    track_data = tracks_result.all()
    
    tracks = [
        {
            "position": playlist_track.position,
            "id": str(track.id),
            "title": track.title or "Unknown",
            "artist": track.artist or "Unknown",
            "album": track.album,
            "duration": track.duration,
            "bpm": track.bpm,
            "key": track.key,
            "energy": track.energy,
            "danceability": track.danceability,
            "valence": track.valence,
            "transition_score": playlist_track.transition_score,
            "harmonic_compatibility": playlist_track.harmonic_compatibility
        }
        for playlist_track, track in track_data
    ]
    
    return {
        "id": str(playlist.id),
        "name": playlist.name,
        "description": playlist.description,
        "generation_method": playlist.generation_method,
        "generation_params": playlist.generation_params,
        "track_count": playlist.track_count,
        "total_duration": playlist.total_duration,
        "avg_bpm": playlist.avg_bpm,
        "avg_energy": playlist.avg_energy,
        "created_at": playlist.created_at.isoformat() if playlist.created_at else None,
        "generation_time": playlist.generation_time,
        "tracks": tracks
    }


@router.delete("/{playlist_id}")
async def delete_playlist(
    playlist_id: str,
    db: AsyncSession = Depends(get_db_session)
) -> Dict[str, str]:
    """Delete a playlist"""
    
    try:
        playlist_uuid = uuid.UUID(playlist_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid playlist ID format")
    
    # Check if playlist exists
    playlist_query = select(Playlist).where(Playlist.id == playlist_uuid)
    playlist_result = await db.execute(playlist_query)
    playlist = playlist_result.scalar_one_or_none()
    
    if not playlist:
        raise HTTPException(status_code=404, detail="Playlist not found")
    
    # Delete playlist (cascade will handle playlist_tracks)
    await db.execute(delete(Playlist).where(Playlist.id == playlist_uuid))
    await db.commit()
    
    logger.info(
        "Playlist deleted",
        playlist_id=str(playlist_id),
        name=playlist.name
    )
    
    return {"status": "deleted", "message": "Playlist deleted successfully"}


@router.get("/methods/available")
async def get_available_methods() -> Dict[str, List[Dict[str, Any]]]:
    """Get available playlist generation methods"""
    
    methods = [
        {
            "name": "similarity",
            "display_name": "Similar Tracks",
            "description": "Generate playlist based on similarity to a seed track",
            "requires_seed": True,
            "parameters": ["seed_track_id"]
        },
        {
            "name": "kmeans",
            "display_name": "K-Means Clustering",
            "description": "Generate diverse playlist using clustering",
            "requires_seed": False,
            "parameters": []
        },
        {
            "name": "energy_flow",
            "display_name": "Energy Flow",
            "description": "Generate playlist with specific energy progression",
            "requires_seed": False,
            "parameters": ["energy_progression"]
        },
        {
            "name": "mood_journey",
            "display_name": "Mood Journey",
            "description": "Generate playlist that transitions between moods",
            "requires_seed": False,
            "parameters": ["start_mood", "end_mood"]
        },
        {
            "name": "harmonic_mixing",
            "display_name": "Harmonic Mixing",
            "description": "Generate DJ-style playlist with harmonic compatibility",
            "requires_seed": False,
            "parameters": []
        }
    ]
    
    return {"methods": methods}


def _track_to_dict(track: Track) -> Dict[str, Any]:
    """Convert Track model to dictionary for algorithms"""
    return {
        "id": str(track.id),
        "title": track.title or "Unknown",
        "artist": track.artist or "Unknown",
        "album": track.album,
        "duration": track.duration,
        "bpm": track.bpm,
        "key": track.key,
        "mode": track.mode,
        "energy": track.energy,
        "danceability": track.danceability,
        "valence": track.valence,
        "acousticness": track.acousticness,
        "instrumentalness": track.instrumentalness,
        "speechiness": track.speechiness,
        "liveness": track.liveness,
        "loudness": track.loudness,
        "spectral_centroid": track.spectral_centroid,
    }


async def _get_track_by_id(db: AsyncSession, track_id: str) -> Optional[Track]:
    """Get track by ID"""
    try:
        track_uuid = uuid.UUID(track_id)
        query = select(Track).where(Track.id == track_uuid)
        result = await db.execute(query)
        return result.scalar_one_or_none()
    except ValueError:
        return None


async def _create_playlist(
    db: AsyncSession,
    name: str,
    method: str,
    tracks: List[Dict[str, Any]],
    generation_time: float
) -> Playlist:
    """Create a new playlist in the database"""
    
    # Calculate playlist statistics
    total_duration = sum(track.get("duration", 0) for track in tracks if track.get("duration"))
    avg_bpm = sum(track.get("bpm", 0) for track in tracks if track.get("bpm")) / len(tracks) if tracks else 0
    avg_energy = sum(track.get("energy", 0) for track in tracks if track.get("energy")) / len(tracks) if tracks else 0
    
    # Create playlist
    playlist = Playlist(
        name=name,
        generation_method=method,
        generation_params={
            "method": method,
            "size": len(tracks),
            "generation_time": generation_time
        },
        track_count=len(tracks),
        total_duration=total_duration,
        avg_bpm=avg_bpm,
        avg_energy=avg_energy,
        generation_time=generation_time
    )
    
    db.add(playlist)
    await db.flush()  # Get the playlist ID
    
    # Add tracks to playlist
    for position, track in enumerate(tracks):
        playlist_item = PlaylistItem(
            playlist_id=playlist.id,
            track_id=uuid.UUID(track["id"]),
            position=position,
            transition_score=0.8,  # Placeholder
            harmonic_compatibility=0.7  # Placeholder
        )
        db.add(playlist_track)
    
    await db.commit()
    return playlist
