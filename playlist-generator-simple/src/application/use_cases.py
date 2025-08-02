"""
Use cases for Playlist Generator.
Application services that coordinate business logic.
"""

import uuid
import time
from typing import List, Optional, Dict, Any
from pathlib import Path

from ..domain.entities import Track, TrackMetadata, AnalysisResult, Playlist
from ..domain.interfaces import ITrackRepository, IAnalysisRepository, IPlaylistRepository, IAudioAnalyzer, IMetadataEnrichmentService
from ..domain.exceptions import TrackNotFoundException, AnalysisFailedException, PlaylistGenerationException, InvalidTrackException
from .commands import AnalyzeTrackCommand, GeneratePlaylistCommand, ImportTracksCommand
from .queries import GetAnalysisStatsQuery, GetPlaylistQuery, GetTracksQuery


class AnalyzeTrackUseCase:
    """Use case for analyzing a track."""
    
    def __init__(self, 
                 track_repo: ITrackRepository,
                 analysis_repo: IAnalysisRepository,
                 audio_analyzer: IAudioAnalyzer,
                 metadata_enrichment: IMetadataEnrichmentService = None):
        self.track_repo = track_repo
        self.analysis_repo = analysis_repo
        self.audio_analyzer = audio_analyzer
        self.metadata_enrichment = metadata_enrichment
    
    def execute(self, command: AnalyzeTrackCommand) -> AnalysisResult:
        """Execute track analysis."""
        file_path = command.file_path
        
        # Validate file exists and is audio
        if not Path(file_path).exists():
            raise TrackNotFoundException(f"File not found: {file_path}")
        
        # Check if track already exists
        existing_track = self.track_repo.find_by_path(file_path)
        
        if existing_track and not command.force_reanalysis:
            # Return existing analysis if available
            if existing_track.is_analyzed():
                return existing_track.analysis_result
            track = existing_track
        else:
            # Create new track
            track = self._create_track_from_file(file_path)
            self.track_repo.save(track)
        
        # Enrich metadata if service available
        if self.metadata_enrichment and self.metadata_enrichment.is_available():
            track = self.metadata_enrichment.enrich_metadata(track)
            self.track_repo.save(track)
        
        # Analyze track
        start_time = time.time()
        try:
            analysis_result = self.audio_analyzer.analyze_track(track)
            analysis_result.processing_time = time.time() - start_time
            
            # Save analysis result
            self.analysis_repo.save_analysis(track.id, analysis_result)
            track.analyze(analysis_result)
            self.track_repo.save(track)
            
            return analysis_result
            
        except Exception as e:
            raise AnalysisFailedException(f"Analysis failed for {file_path}: {str(e)}") from e
    
    def _create_track_from_file(self, file_path: str) -> Track:
        """Create track entity from file."""
        path = Path(file_path)
        
        # Basic metadata from filename
        metadata = TrackMetadata(
            title=path.stem,
            artist="Unknown",
            album="Unknown"
        )
        
        track_id = str(uuid.uuid4())
        return Track(track_id, file_path, metadata)


class GeneratePlaylistUseCase:
    """Use case for generating playlists."""
    
    def __init__(self,
                 track_repo: ITrackRepository,
                 playlist_repo: IPlaylistRepository,
                 algorithm_factory):
        self.track_repo = track_repo
        self.playlist_repo = playlist_repo
        self.algorithm_factory = algorithm_factory
    
    def execute(self, command: GeneratePlaylistCommand) -> Playlist:
        """Execute playlist generation."""
        # Get available tracks
        tracks = self.track_repo.find_all()
        
        if not tracks:
            raise PlaylistGenerationException("No tracks available for playlist generation")
        
        # Get algorithm
        algorithm = self.algorithm_factory.create_algorithm(command.method)
        
        # Generate playlist
        playlist = algorithm.generate_playlist(tracks, command.size)
        playlist.name = command.name
        
        # Save playlist
        self.playlist_repo.save_playlist(playlist)
        
        return playlist


class GetAnalysisStatsUseCase:
    """Use case for getting analysis statistics."""
    
    def __init__(self, analysis_repo: IAnalysisRepository):
        self.analysis_repo = analysis_repo
    
    def execute(self, query: GetAnalysisStatsQuery) -> Dict[str, Any]:
        """Execute statistics query."""
        return self.analysis_repo.get_analysis_statistics()


class ImportTracksUseCase:
    """Use case for importing tracks from directory."""
    
    def __init__(self, track_repo: ITrackRepository):
        self.track_repo = track_repo
    
    def execute(self, command: ImportTracksCommand) -> Dict[str, Any]:
        """Execute track import."""
        directory = Path(command.directory_path)
        
        if not directory.exists():
            raise ValueError(f"Directory not found: {command.directory_path}")
        
        imported_count = 0
        skipped_count = 0
        error_count = 0
        
        # Find audio files
        if command.recursive:
            audio_files = []
            for ext in command.supported_formats:
                audio_files.extend(directory.rglob(f"*{ext}"))
        else:
            audio_files = []
            for ext in command.supported_formats:
                audio_files.extend(directory.glob(f"*{ext}"))
        
        for file_path in audio_files:
            try:
                # Check if track already exists
                existing_track = self.track_repo.find_by_path(str(file_path))
                if existing_track:
                    skipped_count += 1
                    continue
                
                # Create and save track
                track = self._create_track_from_file(str(file_path))
                self.track_repo.save(track)
                imported_count += 1
                
            except Exception as e:
                error_count += 1
                continue
        
        return {
            "imported": imported_count,
            "skipped": skipped_count,
            "errors": error_count,
            "total_files": len(audio_files)
        }
    
    def _create_track_from_file(self, file_path: str) -> Track:
        """Create track entity from file."""
        path = Path(file_path)
        
        # Basic metadata from filename
        metadata = TrackMetadata(
            title=path.stem,
            artist="Unknown",
            album="Unknown"
        )
        
        track_id = str(uuid.uuid4())
        return Track(track_id, file_path, metadata) 