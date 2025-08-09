"""
Enhanced Analysis Engine for Playlista v2
Coordinates feature extraction, ML inference, and database storage
"""

import asyncio
import hashlib
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update

from ..database import get_db_session, Track, AnalysisJob, db_manager
from ..core.logging import get_logger, LogContext, log_operation_start, log_operation_success, log_operation_error
from ..core.config import get_settings
from ..utils.websocket_manager import WebSocketManager
from .features import feature_extractor
from .models import model_manager

logger = get_logger("analysis.engine")
settings = get_settings()


class AnalysisEngine:
    """
    Main analysis engine that orchestrates the complete analysis pipeline
    """
    
    def __init__(self):
        self.websocket_manager = WebSocketManager()
        self._processing = False
        self._worker_id = f"worker_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}"
        
        logger.info(
            "AnalysisEngine initialized",
            worker_id=self._worker_id,
            batch_size=settings.analysis_batch_size,
            timeout=settings.analysis_timeout
        )
    
    async def start_processing(self) -> None:
        """Start the analysis processing loop"""
        if self._processing:
            logger.warning("Analysis processing already running")
            return
        
        self._processing = True
        logger.info("Starting analysis processing loop", worker_id=self._worker_id)
        
        try:
            # First, discover any new music files
            await self._discover_music_files()
            
            while self._processing:
                await self._process_analysis_queue()
                await asyncio.sleep(5)  # Check queue every 5 seconds
                
        except Exception as e:
            logger.error(
                "Analysis processing loop failed",
                worker_id=self._worker_id,
                error_type=type(e).__name__,
                error_message=str(e)
            )
        finally:
            self._processing = False
            logger.info("Analysis processing loop stopped", worker_id=self._worker_id)
    
    async def stop_processing(self) -> None:
        """Stop the analysis processing loop"""
        self._processing = False
        logger.info("Stopping analysis processing", worker_id=self._worker_id)
    
    async def _process_analysis_queue(self) -> None:
        """Process pending analysis jobs"""
        async with get_db_session() as db:
            # Get next batch of jobs
            jobs = await self._get_pending_jobs(db, limit=settings.analysis_batch_size)
            
            if not jobs:
                return
            
            logger.info(
                "Processing analysis batch",
                job_count=len(jobs),
                worker_id=self._worker_id
            )
            
            # Process jobs in parallel (limited concurrency)
            semaphore = asyncio.Semaphore(settings.analysis_workers)
            tasks = [
                self._process_single_job(job, semaphore)
                for job in jobs
            ]
            
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _get_pending_jobs(self, db: AsyncSession, limit: int) -> List[AnalysisJob]:
        """Get pending analysis jobs ordered by priority"""
        query = (
            select(AnalysisJob)
            .where(AnalysisJob.status == "queued")
            .order_by(AnalysisJob.priority.desc(), AnalysisJob.created_at.asc())
            .limit(limit)
        )
        
        result = await db.execute(query)
        return list(result.scalars().all())
    
    async def _process_single_job(self, job: AnalysisJob, semaphore: asyncio.Semaphore) -> None:
        """Process a single analysis job"""
        async with semaphore:
            start_time = time.time()
            
            with LogContext(
                operation="process_analysis_job",
                job_id=str(job.id),
                track_id=str(job.track_id),
                worker_id=self._worker_id
            ):
                try:
                    # Update job status to processing
                    await self._update_job_status(
                        job.id, 
                        "processing", 
                        worker_id=self._worker_id,
                        started_at=start_time
                    )
                    
                    # Get track information
                    track = await self._get_track(job.track_id)
                    if not track:
                        raise ValueError(f"Track not found: {job.track_id}")
                    
                    # Analyze the track
                    analysis_result = await self._analyze_track(track, job.id)
                    
                    # Save results to database
                    await self._save_analysis_result(track, analysis_result)
                    
                    # Update job as completed
                    processing_time = time.time() - start_time
                    await self._update_job_status(
                        job.id,
                        "completed",
                        progress_percent=100,
                        processing_time=processing_time
                    )
                    
                    # Send WebSocket notification
                    await self.websocket_manager.send_analysis_complete(
                        str(track.id),
                        analysis_result
                    )
                    
                    log_operation_success(
                        logger,
                        "track analysis",
                        processing_time * 1000,
                        track_id=str(track.id),
                        file_path=track.filename
                    )
                    
                except Exception as e:
                    processing_time = time.time() - start_time
                    
                    # Update job as failed
                    await self._update_job_status(
                        job.id,
                        "failed",
                        error_message=str(e),
                        processing_time=processing_time,
                        retry_count=job.retry_count + 1
                    )
                    
                    log_operation_error(
                        logger,
                        "track analysis",
                        e,
                        processing_time * 1000,
                        track_id=str(job.track_id),
                        job_id=str(job.id)
                    )
    
    async def _analyze_track(self, track: Track, job_id: str) -> Dict[str, Any]:
        """Perform complete analysis of a track"""
        file_path = track.file_path
        
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        # Send progress update
        await self._update_job_progress(job_id, 10, "Loading audio file")
        await self.websocket_manager.send_analysis_progress(
            str(track.id), 10, "Loading audio file"
        )
        
        # Extract comprehensive features
        await self._update_job_progress(job_id, 30, "Extracting audio features")
        await self.websocket_manager.send_analysis_progress(
            str(track.id), 30, "Extracting audio features"
        )
        
        features = await feature_extractor.extract_comprehensive_features(file_path)
        
        # ML predictions
        await self._update_job_progress(job_id, 60, "Running ML analysis")
        await self.websocket_manager.send_analysis_progress(
            str(track.id), 60, "Running ML analysis"
        )
        
        # Use MusiCNN for advanced ML analysis
        from .musicnn_inference import musicnn_inference
        musicnn_results = musicnn_inference.predict(file_path)
        
        # Create dummy mel spectrogram for other ML models
        import numpy as np
        dummy_features = np.random.random((128, 1292))
        
        # Get genre predictions (enhanced with MusiCNN)
        genre_predictions = musicnn_results.get("genre_features", {})
        if not genre_predictions:
            genre_predictions = await model_manager.predict_genre(dummy_features)
        
        # Get mood predictions (enhanced with MusiCNN)
        mood_predictions = musicnn_results.get("mood_features", {})
        if not mood_predictions:
            mood_predictions = await model_manager.predict_mood(dummy_features)
        
        # Get embeddings
        embeddings = await model_manager.extract_embeddings(dummy_features)
        
        await self._update_job_progress(job_id, 90, "Finalizing results")
        await self.websocket_manager.send_analysis_progress(
            str(track.id), 90, "Finalizing results"
        )
        
        # Combine all results
        analysis_result = {
            "basic_features": features,
            "genre_predictions": genre_predictions,
            "mood_predictions": mood_predictions,
            "ml_embeddings": embeddings.tolist(),
            "musicnn_results": musicnn_results,
            "analysis_version": "2.0",
            "processing_info": {
                "worker_id": self._worker_id,
                "feature_count": len(features),
                "top_genre": max(genre_predictions.items(), key=lambda x: x[1])[0] if genre_predictions else "unknown",
                "confidence": max(genre_predictions.values()) if genre_predictions else 0.0,
                "musicnn_method": musicnn_results.get("inference_method", "unknown"),
                "musicnn_model": musicnn_results.get("model_version", "unknown")
            }
        }
        
        return analysis_result
    
    async def _save_analysis_result(self, track: Track, analysis_result: Dict[str, Any]) -> None:
        """Save analysis results to the track record"""
        async with get_db_session() as db:
            # Extract key features for direct database storage
            basic_features = analysis_result.get("basic_features", {})
            genre_predictions = analysis_result.get("genre_predictions", {})
            mood_predictions = analysis_result.get("mood_predictions", {})
            
            # Get top genre
            top_genre = max(genre_predictions.items(), key=lambda x: x[1])[0] if genre_predictions else None
            
            # Update track with analysis results
            update_data = {
                "status": "analyzed",
                "analyzed_at": time.time(),
                
                # Basic features
                "duration": basic_features.get("duration"),
                "bpm": basic_features.get("bpm"),
                "key": basic_features.get("key"),
                "mode": "major",  # Simplified
                "loudness": basic_features.get("loudness"),
                "energy": mood_predictions.get("energy"),
                "danceability": mood_predictions.get("danceability"),
                "valence": mood_predictions.get("valence"),
                "acousticness": basic_features.get("harmonicity", 0.5),
                "instrumentalness": 1.0 - basic_features.get("percussiveness", 0.5),
                "speechiness": 0.1,  # Default low value
                "liveness": 0.1,    # Default low value
                
                # Spectral features
                "spectral_centroid": basic_features.get("spectral_centroid"),
                "spectral_bandwidth": basic_features.get("spectral_bandwidth"),
                "spectral_rolloff": basic_features.get("spectral_rolloff"),
                "spectral_flatness": basic_features.get("spectral_flatness"),
                "zero_crossing_rate": basic_features.get("zcr_mean"),
                
                # Mood features
                "mood_happy": mood_predictions.get("valence", 0) * mood_predictions.get("energy", 0),
                "mood_sad": (1 - mood_predictions.get("valence", 0.5)) * (1 - mood_predictions.get("energy", 0.5)),
                "mood_angry": mood_predictions.get("arousal", 0) * (1 - mood_predictions.get("valence", 0.5)),
                "mood_relaxed": (1 - mood_predictions.get("arousal", 0.5)) * mood_predictions.get("valence", 0.5),
                
                # Genre
                "genre": top_genre,
                
                # Complex features as JSON
                "mfcc_features": basic_features.get("mfcc_features"),
                "chroma_features": basic_features.get("chroma_features"),
                "ml_embeddings": analysis_result.get("ml_embeddings"),
                
                # Analysis metadata
                "analysis_version": "2.0"
            }
            
            # Filter out None values
            update_data = {k: v for k, v in update_data.items() if v is not None}
            
            # Update the track
            stmt = (
                update(Track)
                .where(Track.id == track.id)
                .values(**update_data)
            )
            
            await db.execute(stmt)
            await db.commit()
            
            logger.info(
                "Analysis results saved",
                track_id=str(track.id),
                genre=top_genre,
                bpm=basic_features.get("bpm"),
                energy=mood_predictions.get("energy")
            )
    
    async def _get_track(self, track_id: str) -> Optional[Track]:
        """Get track by ID"""
        async with get_db_session() as db:
            query = select(Track).where(Track.id == track_id)
            result = await db.execute(query)
            return result.scalar_one_or_none()
    
    async def _update_job_status(
        self, 
        job_id: str, 
        status: str, 
        **kwargs
    ) -> None:
        """Update analysis job status"""
        async with get_db_session() as db:
            update_data = {"status": status, **kwargs}
            
            stmt = (
                update(AnalysisJob)
                .where(AnalysisJob.id == job_id)
                .values(**update_data)
            )
            
            await db.execute(stmt)
            await db.commit()
    
    async def _update_job_progress(
        self, 
        job_id: str, 
        progress_percent: int, 
        current_step: str
    ) -> None:
        """Update job progress"""
        await self._update_job_status(
            job_id,
            "processing",
            progress_percent=progress_percent,
            current_step=current_step
        )
    
    async def _discover_music_files(self) -> None:
        """Discover new music files in the library directory"""
        import mutagen
        
        music_dir = Path(settings.music_library_path)
        if not music_dir.exists():
            logger.warning(f"Music library directory does not exist: {music_dir}")
            return
        
        logger.info(f"Discovering music files in: {music_dir}")
        
        # Supported audio file extensions
        audio_extensions = {'.mp3', '.flac', '.wav', '.ogg', '.m4a', '.aac', '.wma'}
        
        # Get database session using the async generator pattern
        try:
            async for db in get_db_session():
                discovered_count = 0
                
                # Recursively scan for audio files
                for file_path in music_dir.rglob('*'):
                    if file_path.suffix.lower() in audio_extensions:
                        try:
                            # Check if file already exists in database
                            existing_query = select(Track).where(Track.file_path == str(file_path))
                            result = await db.execute(existing_query)
                            existing_track = result.scalar_one_or_none()
                            
                            if existing_track:
                                continue  # Skip files already in database
                            
                            # Calculate file hash for deduplication
                            file_hash = hashlib.md5(str(file_path).encode()).hexdigest()
                            
                            # Extract basic metadata with mutagen
                            metadata = {}
                            try:
                                audio_file = mutagen.File(str(file_path))
                                if audio_file:
                                    metadata = {
                                        'title': audio_file.get('TIT2', [str(file_path.stem)])[0] if 'TIT2' in audio_file else str(file_path.stem),
                                        'artist': audio_file.get('TPE1', ['Unknown'])[0] if 'TPE1' in audio_file else 'Unknown',
                                        'album': audio_file.get('TALB', ['Unknown'])[0] if 'TALB' in audio_file else 'Unknown',
                                        'duration': getattr(audio_file, 'info', {}).get('length', 0),
                                        'bitrate': getattr(audio_file, 'info', {}).get('bitrate', 0),
                                        'sample_rate': getattr(audio_file, 'info', {}).get('sample_rate', 0),
                                    }
                            except Exception as e:
                                logger.warning(f"Could not extract metadata from {file_path}: {e}")
                                metadata = {
                                    'title': str(file_path.stem),
                                    'artist': 'Unknown',
                                    'album': 'Unknown',
                                    'duration': 0,
                                    'bitrate': 0,
                                    'sample_rate': 0,
                                }
                            
                            # Create new track record
                            new_track = Track(
                                file_path=str(file_path),
                                file_hash=file_hash,
                                filename=file_path.name,
                                file_size_bytes=file_path.stat().st_size,
                                title=metadata.get('title'),
                                artist=metadata.get('artist'),
                                album=metadata.get('album'),
                                duration=metadata.get('duration'),
                                bitrate=metadata.get('bitrate'),
                                sample_rate=metadata.get('sample_rate'),
                                format=file_path.suffix.lower()[1:],  # Remove the dot
                                status="discovered"
                            )
                            
                            db.add(new_track)
                            await db.flush()  # Get the track ID
                            
                            # Create analysis job for this track
                            analysis_job = AnalysisJob(
                                track_id=new_track.id,
                                status="pending",
                                priority=1
                            )
                            
                            db.add(analysis_job)
                            discovered_count += 1
                            
                            logger.debug(f"Discovered new track: {file_path.name}")
                            
                        except Exception as e:
                            logger.error(f"Error processing file {file_path}: {e}")
                            continue
                
                await db.commit()
                
                if discovered_count > 0:
                    logger.info(f"Discovered {discovered_count} new music files")
                    await self.websocket_manager.send_discovery_update(
                        discovered_count, f"Discovered {discovered_count} new tracks"
                    )
                else:
                    logger.info("No new music files discovered")
                
                # Break after processing with one database session
                break
        
        except Exception as e:
            logger.error(f"Error during file discovery: {e}")
            raise


# Global analysis engine instance
analysis_engine = AnalysisEngine()
