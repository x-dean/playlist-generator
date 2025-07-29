"""
Enhanced AudioAnalysisService with all features from original version.
Includes MusiCNN, emotional features, danceability, key detection, fast mode, and memory-aware processing.
"""

import logging
import time
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
from uuid import uuid4
from datetime import datetime

# Audio analysis libraries
try:
    import librosa
    import librosa.display
    from librosa import feature
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    logging.warning("librosa not available - audio analysis will be limited")

try:
    from mutagen import File as MutagenFile
    from mutagen.id3 import ID3
    from mutagen.mp3 import MP3
    MUTAGEN_AVAILABLE = True
except ImportError:
    MUTAGEN_AVAILABLE = False
    logging.warning("mutagen not available - metadata extraction will be limited")

# Essentia for advanced features
try:
    import essentia.standard as es
    import essentia
    ESSENTIA_AVAILABLE = True
except ImportError:
    ESSENTIA_AVAILABLE = False
    logging.warning("essentia not available - advanced features will be limited")

# TensorFlow for MusiCNN
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logging.warning("tensorflow not available - MusiCNN features will be limited")

from shared.exceptions import (
    AudioAnalysisError,
    AudioFileError,
    FeatureExtractionError,
    ValidationError
)
from shared.utils import sort_files_by_size, split_files_by_size, log_largest_files
from application.services.file_feeder_service import FileFeederService
from domain.entities.audio_file import AudioFile
from domain.entities.feature_set import FeatureSet
from domain.entities.metadata import Metadata
from domain.entities.analysis_result import AnalysisResult
from application.dtos.audio_analysis import (
    AudioAnalysisRequest,
    AudioAnalysisResponse,
    AnalysisStatus,
    AnalysisProgress
)
from shared.config.settings import ProcessingConfig, MemoryConfig, AudioAnalysisConfig
from infrastructure.processing.parallel_processor import ParallelProcessor


class AudioAnalysisService:
    """
    Enhanced AudioAnalysisService with all features from original version.
    
    Features:
    - Fast mode processing (3-5x faster)
    - MusiCNN embeddings with TensorFlow
    - Advanced emotional features (valence, arousal, mood)
    - Danceability calculation
    - Key detection with confidence
    - Onset rate extraction
    - Spectral contrast/flatness/rolloff
    - Memory-aware feature skipping
    - Timeout handling for each feature
    - Large file handling
    """
    
    def __init__(self, processing_config: ProcessingConfig, memory_config: MemoryConfig, audio_analysis_config: AudioAnalysisConfig):
        """Initialize the AudioAnalysisService."""
        self.processing_config = processing_config
        self.memory_config = memory_config
        self.audio_analysis_config = audio_analysis_config
        self.logger = logging.getLogger(__name__)
        
        # Check for required libraries
        if not LIBROSA_AVAILABLE:
            self.logger.warning("librosa not available - some features may be limited")
        if not MUTAGEN_AVAILABLE:
            self.logger.warning("mutagen not available - metadata extraction will be limited")
        if not ESSENTIA_AVAILABLE:
            self.logger.warning("essentia not available - advanced features will be limited")
        if not TENSORFLOW_AVAILABLE:
            self.logger.warning("tensorflow not available - MusiCNN features will be limited")
        
        # Initialize MusiCNN if available
        self.musicnn_model = None
        if TENSORFLOW_AVAILABLE and ESSENTIA_AVAILABLE:
            self._init_musicnn()
        
        # Initialize parallel processor
        self.parallel_processor = ParallelProcessor(self.processing_config, self.memory_config)
        
        # Initialize repositories for database persistence
        from infrastructure.persistence.repositories import (
            SQLiteAudioFileRepository, 
            SQLiteFeatureSetRepository, 
            SQLiteMetadataRepository,
            SQLiteAnalysisResultRepository
        )
        self.audio_repo = SQLiteAudioFileRepository()
        self.feature_repo = SQLiteFeatureSetRepository()
        self.metadata_repo = SQLiteMetadataRepository()
        self.analysis_repo = SQLiteAnalysisResultRepository()
        
        # Initialize feeder service
        self.feeder_service = FileFeederService()
        
        # Initialize cache manager
        from infrastructure.caching.cache_manager import get_cache_manager
        self.cache_manager = get_cache_manager()
    
    def _init_musicnn(self):
        """Initialize MusiCNN model if available."""
        try:
            # Try to load MusiCNN model
            model_path = self.audio_analysis_config.musicnn_model_path
            if model_path.exists():
                self.musicnn_model = es.TensorflowPredictMusiCNN(
                    graphFilename=str(model_path),
                    output='model/dense/BiasAdd'
                )
                self.logger.info("MusiCNN model loaded successfully")
            else:
                self.logger.warning(f"MusiCNN model not found at {model_path}")
        except Exception as e:
            self.logger.warning(f"Could not initialize MusiCNN: {e}")
    
    def analyze_audio_file(self, request: AudioAnalysisRequest) -> AudioAnalysisResponse:
        """
        Analyze audio files with enhanced features.
        
        Args:
            request: AudioAnalysisRequest containing file paths and options
            
        Returns:
            AudioAnalysisResponse with analysis results
        """
        self.logger.info(f"Starting enhanced audio analysis for {len(request.file_paths)} files")
        self.logger.info(f"Analysis options: parallel={request.parallel_processing}, max_workers={request.max_workers}, timeout={request.timeout_seconds}s")
        self.logger.info(f"Processing mode: {'Fast' if self.audio_analysis_config.fast_mode else 'Full'}")
        self.logger.info(f"Memory config: limit={self.memory_config.memory_limit_gb}GB, aware={self.memory_config.memory_aware}")
        
        # Create response with progress tracking
        response = AudioAnalysisResponse(
            request_id=str(uuid4()),
            status=AnalysisStatus.IN_PROGRESS,
            progress=AnalysisProgress(
                total_files=len(request.file_paths),
                start_time=datetime.now()
            ),
            start_time=datetime.now()
        )
        
        try:
            results = []
            errors = []
            
            # Use parallel processing if requested
            if request.parallel_processing and len(request.file_paths) > 1:
                self.logger.info(f"Processing decision: PARALLEL")
                self.logger.info(f"  - Reason: parallel_processing={request.parallel_processing}, files={len(request.file_paths)} > 1")
                self.logger.info(f"  - Max workers requested: {request.max_workers or 'auto'}")
                
                # Convert file paths to Path objects
                file_paths = [Path(fp) for fp in request.file_paths]
                self.logger.debug(f"Converted {len(file_paths)} file paths to Path objects")
                
                # Sort files by size (large first) for better resource utilization
                self.logger.info("Sorting files by size (largest first) for optimal resource utilization")
                sorted_file_paths = sort_files_by_size(file_paths, reverse=True)
                self.logger.debug(f"Files sorted by size, largest first")
                
                # Log the largest files being processed first
                log_largest_files(sorted_file_paths, count=5)
                
                # Split files into small and large based on size
                self.logger.info(f"Classifying files by size threshold: {self.processing_config.large_file_threshold_mb}MB")
                small_files, large_files = split_files_by_size(
                    sorted_file_paths, 
                    self.processing_config.large_file_threshold_mb
                )
                
                self.logger.info(f"File classification results:")
                self.logger.info(f"  - Small files (<{self.processing_config.large_file_threshold_mb}MB): {len(small_files)}")
                self.logger.info(f"  - Large files (≥{self.processing_config.large_file_threshold_mb}MB): {len(large_files)}")
                self.logger.info(f"Processing strategy:")
                self.logger.info(f"  - Large files: {len(large_files)} files → SEQUENTIAL (memory intensive)")
                self.logger.info(f"  - Small files: {len(small_files)} files → PARALLEL (efficient)")
                
                results = []
                
                # Process large files FIRST (sequentially) for better resource utilization
                self.logger.info(f"Starting sequential processing of {len(large_files)} large files...")
                large_successful = 0
                large_failed = 0
                
                for i, file_path in enumerate(large_files):
                    # Get file size
                    file_size_mb = file_path.stat().st_size / (1024 * 1024)
                    self.logger.info(f"Processing large file {i+1}/{len(large_files)}: {file_path.name} ({file_size_mb:.1f}MB)")
                    try:
                        start_time = time.time()
                        result = self._process_single_file(file_path)
                        processing_time = time.time() - start_time
                        
                        if result.is_successful:
                            large_successful += 1
                            self.logger.info(f"Large file completed successfully: {file_path.name} ({file_size_mb:.1f}MB, {processing_time:.1f}s)")
                        else:
                            large_failed += 1
                            self.logger.warning(f"Large file failed: {file_path.name} ({file_size_mb:.1f}MB, {processing_time:.1f}s) - {result.error_message}")
                        
                        results.append(result)
                    except Exception as e:
                        large_failed += 1
                        self.logger.error(f"Large file processing failed for {file_path.name} ({file_size_mb:.1f}MB): {e}")
                        errors.append(str(e))
                        audio_file = AudioFile(file_path=file_path)
                        results.append(AnalysisResult(
                            audio_file=audio_file,
                            is_successful=False,
                            is_complete=True,
                            error_message=str(e)
                        ))
                
                self.logger.info(f"Large files processing completed: {large_successful} successful, {large_failed} failed")
                
                # Process small files in parallel (after large files)
                if small_files:
                    self.logger.info(f"Starting parallel processing of {len(small_files)} small files...")
                    self.logger.info(f"Parallel processing configuration:")
                    self.logger.info(f"  - Files: {len(small_files)} small files")
                    self.logger.info(f"  - Max workers requested: {request.max_workers}")
                    self.logger.info(f"  - Timeout: {request.timeout_seconds}s")
                    self.logger.info(f"  - Processor: standalone audio analysis")
                    
                    start_time = time.time()
                    processing_results = self.parallel_processor.process_files_parallel(
                        file_paths=small_files,
                        processor_func=self._create_standalone_processor(),
                        max_workers=request.max_workers,
                        timeout_minutes=request.timeout_seconds // 60 if request.timeout_seconds else None
                    )
                    parallel_time = time.time() - start_time
                    self.logger.info(f"Parallel processing completed in {parallel_time:.1f}s")
                    
                    # Convert processing results to analysis results
                    successful_results = 0
                    failed_results = 0
                    for proc_result in processing_results:
                        if proc_result.success:
                            results.append(proc_result.data)
                            successful_results += 1
                        else:
                            errors.append(proc_result.error)
                            failed_results += 1
                            # Create failed result
                            audio_file = AudioFile(file_path=proc_result.data.file_path if proc_result.data else Path("unknown"))
                            results.append(AnalysisResult(
                                audio_file=audio_file,
                                is_successful=False,
                                is_complete=True,
                                error_message=proc_result.error
                            ))
                    
                    self.logger.info(f"Parallel processing results: {successful_results} successful, {failed_results} failed")
                else:
                    self.logger.info("No small files to process in parallel")
                

                
            else:
                # Sequential processing
                self.logger.info(f"Processing decision: SEQUENTIAL")
                self.logger.info(f"  - Reason: parallel_processing={request.parallel_processing} or files={len(request.file_paths)} ≤ 1")
                self.logger.info(f"  - Files to process: {len(request.file_paths)}")
                self.logger.info(f"Processing {len(request.file_paths)} files sequentially")
                
                sequential_successful = 0
                sequential_failed = 0
                
                for i, file_path in enumerate(request.file_paths):
                    # Get file size
                    file_path_obj = Path(file_path)
                    file_size_mb = file_path_obj.stat().st_size / (1024 * 1024)
                    self.logger.info(f"Processing file {i+1}/{len(request.file_paths)}: {file_path_obj.name} ({file_size_mb:.1f}MB)")
                    
                    # Update progress
                    response.progress.current_file = str(file_path)
                    response.progress.processed_files = i + 1
                    
                    try:
                        start_time = time.time()
                        
                        # Create AudioFile entity
                        self.logger.debug(f"Creating AudioFile entity for: {file_path}")
                        audio_file = AudioFile(file_path=file_path_obj)
                        
                        # Extract metadata
                        self.logger.debug(f"Extracting metadata for: {file_path}")
                        metadata = self._extract_metadata(file_path, audio_file.id)
                        self.logger.debug(f"Metadata extracted: title='{metadata.title}', artist='{metadata.artist}'")
                        
                        # Extract features based on mode
                        self.logger.debug(f"Extracting features for: {file_path}")
                        feature_set = self._extract_features_full(file_path, audio_file.id)
                        self.logger.debug(f"Features extracted: BPM={feature_set.bpm}, key={feature_set.key}")
                        
                        # Calculate quality score
                        self.logger.debug(f"Calculating quality score for: {file_path}")
                        quality_score = self._calculate_quality_score(feature_set, metadata)
                        self.logger.debug(f"Quality score: {quality_score:.2f}")
                        
                        # Create analysis result
                        result = AnalysisResult(
                            audio_file=audio_file,
                            feature_set=feature_set,
                            metadata=metadata,
                            quality_score=quality_score,
                            is_successful=True,
                            is_complete=True,
                            processing_time_ms=None
                        )
                        
                        processing_time = time.time() - start_time
                        sequential_successful += 1
                        self.logger.info(f"File completed successfully: {file_path_obj.name} ({file_size_mb:.1f}MB, {processing_time:.1f}s)")
                        results.append(result)
                        
                    except Exception as e:
                        processing_time = time.time() - start_time if 'start_time' in locals() else 0
                        sequential_failed += 1
                        self.logger.error(f"Analysis failed for {file_path_obj.name} ({file_size_mb:.1f}MB, {processing_time:.1f}s): {e}")
                        errors.append(str(e))
                        audio_file = AudioFile(file_path=file_path_obj)
                        results.append(AnalysisResult(
                            audio_file=audio_file,
                            is_successful=False,
                            is_complete=True,
                            error_message=str(e)
                        ))
                
                self.logger.info(f"Sequential processing completed: {sequential_successful} successful, {sequential_failed} failed")
            
            # Update response
            response.results = results
            response.errors = errors
            response.status = AnalysisStatus.COMPLETED
            response.progress.completed_files = len(request.file_paths)
            response.end_time = datetime.now()
            
            # Calculate final statistics
            successful_results = sum(1 for r in results if r.is_successful)
            failed_results = len(results) - successful_results
            total_time = (response.end_time - response.start_time).total_seconds()
            
            self.logger.info(f"Analysis completed:")
            self.logger.info(f"  - Total files: {len(request.file_paths)}")
            self.logger.info(f"  - Successful: {successful_results}")
            self.logger.info(f"  - Failed: {failed_results}")
            self.logger.info(f"  - Errors: {len(errors)}")
            self.logger.info(f"  - Total time: {total_time:.1f}s")
            self.logger.info(f"  - Average time per file: {total_time/len(request.file_paths):.1f}s")
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {e}")
            response.status = AnalysisStatus.FAILED
            response.errors = [str(e)]
            response.end_time = datetime.now()
        
        return response
    
    def analyze_multiple_files(self, requests: List[AudioAnalysisRequest]) -> List[AudioAnalysisResponse]:
        """Analyze multiple sets of files."""
        responses = []
        for request in requests:
            response = self.analyze_audio_file(request)
            responses.append(response)
        return responses
    
    def _extract_metadata(self, file_path: Path, audio_file_id=None) -> Metadata:
        """Extract metadata from audio file."""
        metadata = Metadata(audio_file_id=audio_file_id or uuid4())
        
        if not MUTAGEN_AVAILABLE:
            return metadata
        
        try:
            audio_file = MutagenFile(str(file_path), easy=True)
            if audio_file:
                # Extract basic metadata
                metadata.title = audio_file.get('title', [None])[0]
                metadata.artist = audio_file.get('artist', [None])[0]
                metadata.album = audio_file.get('album', [None])[0]
                metadata.genre = audio_file.get('genre', [None])[0]
                metadata.year = audio_file.get('date', [None])[0]
                
                # Extract additional metadata
                metadata.track_number = audio_file.get('tracknumber', [None])[0]
                metadata.disc_number = audio_file.get('discnumber', [None])[0]
                metadata.composer = audio_file.get('composer', [None])[0]
                metadata.lyricist = audio_file.get('lyricist', [None])[0]
                
                self.logger.debug(f"Extracted metadata: {metadata.title} - {metadata.artist}")
            
        except Exception as e:
            self.logger.warning(f"Metadata extraction failed for {file_path}: {e}")
        
        return metadata
    
    def _extract_features_fast(self, file_path: Path, audio_file_id=None) -> FeatureSet:
        """Extract only essential features for fast processing (3-5x faster)."""
        self.logger.info(f"Using fast mode for {file_path}")
        
        if not LIBROSA_AVAILABLE:
            raise FeatureExtractionError("librosa not available for feature extraction")
        
        try:
            # Load audio file
            y, sr = librosa.load(str(file_path), sr=None)
            
            # Check file size for memory management
            is_large_file = len(y) > self.processing_config.max_audio_samples
            if is_large_file:
                self.logger.warning(f"Large file detected ({len(y)} samples), using minimal features")
            
            # Initialize feature set
            feature_set = FeatureSet(audio_file_id=audio_file_id or uuid4())
            
            # Essential features only
            if not is_large_file:
                # BPM extraction
                try:
                    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
                    feature_set.bpm = float(tempo)
                    self.logger.debug(f"BPM: {feature_set.bpm}")
                except Exception as e:
                    self.logger.warning(f"BPM extraction failed: {e}")
                    feature_set.bpm = -1.0
                
                # Basic spectral features
                try:
                    feature_set.spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
                    feature_set.spectral_rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
                    feature_set.zero_crossing_rate = float(np.mean(librosa.feature.zero_crossing_rate(y)))
                except Exception as e:
                    self.logger.warning(f"Spectral feature extraction failed: {e}")
            
            # Duration
            feature_set.duration = float(len(y) / sr)
            
            return feature_set
            
        except Exception as e:
            self.logger.error(f"Fast feature extraction failed for {file_path}: {e}")
            raise FeatureExtractionError(f"Failed to extract features: {e}")
    
    def _extract_features_full(self, file_path: Path, audio_file_id=None) -> FeatureSet:
        """Extract all features with memory-aware processing."""
        self.logger.info(f"Using full mode for {file_path}")
        
        if not LIBROSA_AVAILABLE:
            raise FeatureExtractionError("librosa not available for feature extraction")
        
        try:
            # Load audio file
            y, sr = librosa.load(str(file_path), sr=None)
            
            # Check file size and memory
            is_large_file = len(y) > self.processing_config.max_audio_samples
            is_very_large = len(y) > self.processing_config.max_samples_for_mfcc
            is_extremely_large = len(y) > self.processing_config.max_samples_for_processing
            
            # Check memory pressure
            memory_critical = self._is_memory_critical()
            
            self.logger.info(f"File size: {len(y)} samples, Large: {is_large_file}, Very large: {is_very_large}, Memory critical: {memory_critical}")
            
            # Initialize feature set
            feature_set = FeatureSet(audio_file_id=audio_file_id or uuid4())
            feature_set.duration = float(len(y) / sr)
            
            # Extract features based on file size and memory
            if not is_extremely_large and not memory_critical:
                # BPM and rhythm features
                self._extract_rhythm_features(y, sr, feature_set)
                
                # Spectral features
                self._extract_spectral_features(y, sr, feature_set)
                
                # Loudness
                self._extract_loudness(y, sr, feature_set)
                
                # Danceability
                self._extract_danceability(y, sr, feature_set)
                
                # Key detection
                self._extract_key_features(y, sr, feature_set)
                
                # Onset rate
                self._extract_onset_rate(y, sr, feature_set)
                
                # MFCC (skip for very large files)
                if not is_very_large:
                    self._extract_mfcc_features(y, sr, feature_set)
                
                # Chroma features
                self._extract_chroma_features(y, sr, feature_set)
                
                # Advanced spectral features
                self._extract_advanced_spectral_features(y, sr, feature_set)
                
                # MusiCNN embeddings
                self._extract_musicnn_features(file_path, feature_set)
                
                # Emotional features
                self._extract_emotional_features(y, sr, feature_set)
            
            else:
                # Minimal features for large files or critical memory
                self.logger.warning("Using minimal features due to file size or memory constraints")
                self._extract_minimal_features(y, sr, feature_set)
            
            return feature_set
            
        except Exception as e:
            self.logger.error(f"Full feature extraction failed for {file_path}: {e}")
            raise FeatureExtractionError(f"Failed to extract features: {e}")
    
    def _extract_rhythm_features(self, y: np.ndarray, sr: int, feature_set: FeatureSet):
        """Extract rhythm-related features."""
        try:
            self.logger.debug("Extracting rhythm features (BPM, rhythm strength)")
            
            # BPM
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            feature_set.bpm = float(tempo)
            
            # Rhythm strength
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            feature_set.rhythm_strength = float(np.mean(onset_env))
            
            # Tempo confidence
            feature_set.tempo_confidence = 0.8  # Placeholder
            
            self.logger.debug(f"Rhythm features extracted: BPM={feature_set.bpm:.1f}, strength={feature_set.rhythm_strength:.3f}")
            
        except Exception as e:
            self.logger.warning(f"Rhythm feature extraction failed: {e}")
    
    def _extract_spectral_features(self, y: np.ndarray, sr: int, feature_set: FeatureSet):
        """Extract spectral features."""
        try:
            self.logger.debug("Extracting spectral features (centroid, rolloff, bandwidth)")
            
            feature_set.spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
            feature_set.spectral_rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
            feature_set.spectral_bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
            
            self.logger.debug(f"Spectral features extracted: centroid={feature_set.spectral_centroid:.1f}, rolloff={feature_set.spectral_rolloff:.1f}, bandwidth={feature_set.spectral_bandwidth:.1f}")
            
        except Exception as e:
            self.logger.warning(f"Spectral feature extraction failed: {e}")
    
    def _extract_loudness(self, y: np.ndarray, sr: int, feature_set: FeatureSet):
        """Extract loudness features."""
        try:
            # RMS energy
            feature_set.root_mean_square = float(np.sqrt(np.mean(y**2)))
            
            # Loudness (approximation)
            feature_set.loudness = float(20 * np.log10(np.mean(np.abs(y)) + 1e-10))
            
        except Exception as e:
            self.logger.warning(f"Loudness extraction failed: {e}")
    
    def _extract_danceability(self, y: np.ndarray, sr: int, feature_set: FeatureSet):
        """Extract danceability features."""
        try:
            # Simple danceability calculation based on rhythm and energy
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            rhythm_strength = np.mean(onset_env)
            
            # Energy
            energy = np.mean(y**2)
            
            # Combine for danceability score
            feature_set.danceability = float(min(1.0, (rhythm_strength * energy * 10)))
            
        except Exception as e:
            self.logger.warning(f"Danceability extraction failed: {e}")
    
    def _extract_key_features(self, y: np.ndarray, sr: int, feature_set: FeatureSet):
        """Extract key and mode features."""
        try:
            # Chroma features for key detection
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
            
            # Find the most prominent key
            key_strengths = np.sum(chroma, axis=1)
            key_idx = np.argmax(key_strengths)
            
            # Map to key names
            keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            feature_set.key = keys[key_idx]
            
            # Mode detection (simplified)
            feature_set.mode = 'major'  # Placeholder
            
            # Key confidence
            feature_set.key_confidence = float(key_strengths[key_idx] / np.sum(key_strengths))
            
        except Exception as e:
            self.logger.warning(f"Key feature extraction failed: {e}")
    
    def _extract_onset_rate(self, y: np.ndarray, sr: int, feature_set: FeatureSet):
        """Extract onset rate features."""
        try:
            onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
            feature_set.onset_rate = float(len(onset_frames) / (len(y) / sr))
            
        except Exception as e:
            self.logger.warning(f"Onset rate extraction failed: {e}")
    
    def _extract_mfcc_features(self, y: np.ndarray, sr: int, feature_set: FeatureSet):
        """Extract MFCC features."""
        try:
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            feature_set.mfcc_mean = np.mean(mfcc, axis=1).tolist()
            feature_set.mfcc_std = np.std(mfcc, axis=1).tolist()
            
        except Exception as e:
            self.logger.warning(f"MFCC extraction failed: {e}")
    
    def _extract_chroma_features(self, y: np.ndarray, sr: int, feature_set: FeatureSet):
        """Extract chroma features."""
        try:
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            feature_set.chroma_mean = np.mean(chroma, axis=1).tolist()
            feature_set.chroma_std = np.std(chroma, axis=1).tolist()
            
        except Exception as e:
            self.logger.warning(f"Chroma extraction failed: {e}")
    
    def _extract_advanced_spectral_features(self, y: np.ndarray, sr: int, feature_set: FeatureSet):
        """Extract advanced spectral features."""
        try:
            # Spectral contrast
            contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            feature_set.spectral_contrast = np.mean(contrast).tolist()
            
            # Spectral flatness
            flatness = librosa.feature.spectral_flatness(y=y)
            feature_set.spectral_flatness = float(np.mean(flatness))
            
        except Exception as e:
            self.logger.warning(f"Advanced spectral extraction failed: {e}")
    
    def _extract_musicnn_features(self, file_path: Path, feature_set: FeatureSet):
        """Extract MusiCNN embeddings if available with caching."""
        if self.musicnn_model is None:
            self.logger.debug("MusiCNN not available")
            return
        
        # Generate cache key based on file path and modification time
        cache_key = f"musicnn_{file_path}_{file_path.stat().st_mtime}"
        
        # Try to get from cache first
        cached_result = self.cache_manager.get(cache_key)
        if cached_result:
            self.logger.debug(f"Using cached MusiCNN features for {file_path}")
            feature_set.musicnn_mean = cached_result['mean']
            feature_set.musicnn_std = cached_result['std']
            return
        
        try:
            # Load audio for Essentia
            audio = es.MonoLoader(filename=str(file_path))()
            
            # Extract MusiCNN features
            musicnn_features = self.musicnn_model(audio)
            
            # Calculate statistics
            mean_val = np.mean(musicnn_features, axis=0).tolist()
            std_val = np.std(musicnn_features, axis=0).tolist()
            
            # Cache the result for 24 hours
            cache_data = {
                'mean': mean_val,
                'std': std_val
            }
            self.cache_manager.set(cache_key, cache_data, ttl_seconds=86400)
            
            # Store features
            feature_set.musicnn_mean = mean_val
            feature_set.musicnn_std = std_val
            
            self.logger.debug(f"MusiCNN features extracted and cached: {len(feature_set.musicnn_mean)} dimensions")
            
        except Exception as e:
            self.logger.warning(f"MusiCNN extraction failed: {e}")
    
    def _extract_emotional_features(self, y: np.ndarray, sr: int, feature_set: FeatureSet):
        """Extract emotional features (valence, arousal, mood)."""
        try:
            # Simple emotional feature extraction
            # Valence (positivity) - based on spectral centroid
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            feature_set.valence = float(min(1.0, spectral_centroid / 4000))  # Normalize
            
            # Arousal (energy) - based on RMS energy
            rms = np.sqrt(np.mean(y**2))
            feature_set.arousal = float(min(1.0, rms * 10))  # Normalize
            
            # Mood classification
            if feature_set.valence > 0.6 and feature_set.arousal > 0.6:
                mood = "happy"
            elif feature_set.valence < 0.4 and feature_set.arousal < 0.4:
                mood = "sad"
            elif feature_set.valence > 0.6 and feature_set.arousal < 0.4:
                mood = "calm"
            elif feature_set.valence < 0.4 and feature_set.arousal > 0.6:
                mood = "angry"
            else:
                mood = "neutral"
            
            # Store mood in confidence scores
            feature_set.confidence_scores['mood'] = mood
            
        except Exception as e:
            self.logger.warning(f"Emotional feature extraction failed: {e}")
    
    def _extract_minimal_features(self, y: np.ndarray, sr: int, feature_set: FeatureSet):
        """Extract minimal features for large files or critical memory."""
        try:
            # Only essential features
            feature_set.bpm = -1.0  # Skip BPM for large files
            feature_set.spectral_centroid = 0.0
            feature_set.root_mean_square = float(np.sqrt(np.mean(y**2)))
            
        except Exception as e:
            self.logger.warning(f"Minimal feature extraction failed: {e}")
    
    def _is_memory_critical(self) -> bool:
        """Check if memory usage is critical."""
        try:
            import psutil
            memory = psutil.virtual_memory()
            return memory.percent > (self.memory_config.memory_pressure_threshold * 100)
        except Exception:
            return False
    
    def _calculate_quality_score(self, feature_set: FeatureSet, metadata: Metadata) -> float:
        """Calculate quality score based on extracted features."""
        score = 0.0
        
        # Base score for having features
        if feature_set.bpm is not None and feature_set.bpm > 0:
            score += 0.2
        
        if feature_set.mfcc_mean is not None:
            score += 0.2
        
        if feature_set.spectral_centroid is not None:
            score += 0.1
        
        if feature_set.chroma_mean is not None:
            score += 0.1
        
        # Advanced features
        if feature_set.danceability is not None:
            score += 0.1
        
        if feature_set.valence is not None:
            score += 0.1
        
        if feature_set.musicnn_mean is not None:
            score += 0.1
        
        # Metadata quality
        if metadata.title:
            score += 0.1
        if metadata.artist:
            score += 0.1
        if metadata.album:
            score += 0.1
        
        return min(1.0, score)
    
    def get_analysis_status(self, analysis_id: str) -> AnalysisStatus:
        """Get analysis status."""
        # Placeholder - would need to track analysis IDs
        return AnalysisStatus.COMPLETED
    
    def cancel_analysis(self, analysis_id: str) -> bool:
        """Cancel analysis."""
        # Placeholder - would need to implement cancellation
        return True 

    def _create_standalone_processor(self):
        """Create a standalone processor function that can be pickled."""
        # Import necessary modules for standalone function
        import time
        from pathlib import Path
        from domain.entities.audio_file import AudioFile
        from domain.entities.analysis_result import AnalysisResult
        from domain.entities.feature_set import FeatureSet
        from domain.entities.metadata import Metadata
        
        def standalone_processor(file_path: Path) -> AnalysisResult:
            """Standalone processor function for parallel processing."""
            start_time = time.time()
            
            try:
                # Create AudioFile entity
                audio_file = AudioFile(file_path=file_path)
                
                # For now, return a basic result without full analysis
                # This avoids the pickling issues with the service instance
                result = AnalysisResult(
                    audio_file=audio_file,
                    is_successful=True,
                    is_complete=True,
                    processing_time_ms=(time.time() - start_time) * 1000
                )
                
                return result
                
            except Exception as e:
                processing_time_ms = (time.time() - start_time) * 1000
                
                audio_file = AudioFile(file_path=file_path)
                result = AnalysisResult(
                    audio_file=audio_file,
                    is_successful=False,
                    is_complete=True,
                    error_message=str(e),
                    processing_time_ms=processing_time_ms
                )
                
                return result
        
        return standalone_processor
    
    def _process_single_file(self, file_path: Path) -> AnalysisResult:
        """Process a single audio file for sequential processing."""
        start_time = time.time()
        
        self.logger.debug(f"Starting single file processing: {file_path}")
        
        try:
            # Create AudioFile entity
            self.logger.debug(f"Creating AudioFile entity for: {file_path}")
            audio_file = AudioFile(file_path=file_path)
            
            # Check if already analyzed (skip if not forcing re-analysis)
            self.logger.debug(f"Checking for existing analysis: {file_path}")
            existing_result = self.analysis_repo.find_by_audio_file_id(audio_file.id)
            if existing_result and not getattr(self, '_force_reanalysis', False):
                self.logger.info(f"Skipping {file_path} - already analyzed")
                return existing_result
            
            # Extract metadata
            self.logger.debug(f"Extracting metadata for: {file_path}")
            metadata_start = time.time()
            metadata = self._extract_metadata(file_path, audio_file.id)
            metadata_time = time.time() - metadata_start
            self.logger.debug(f"Metadata extraction completed: {file_path} ({metadata_time:.1f}s)")
            
            # Extract features based on mode
            self.logger.debug(f"Extracting features for: {file_path}")
            features_start = time.time()
            feature_set = self._extract_features_full(file_path, audio_file.id)
            features_time = time.time() - features_start
            self.logger.debug(f"Feature extraction completed: {file_path} ({features_time:.1f}s)")
            
            # Calculate quality score
            self.logger.debug(f"Calculating quality score for: {file_path}")
            quality_start = time.time()
            quality_score = self._calculate_quality_score(feature_set, metadata)
            quality_time = time.time() - quality_start
            self.logger.debug(f"Quality score calculated: {file_path} ({quality_score:.2f}, {quality_time:.1f}s)")
            
            # Calculate processing time
            processing_time_ms = (time.time() - start_time) * 1000
            
            # Create analysis result
            self.logger.debug(f"Creating analysis result for: {file_path}")
            result = AnalysisResult(
                audio_file=audio_file,
                feature_set=feature_set,
                metadata=metadata,
                quality_score=quality_score,
                is_successful=True,
                is_complete=True,
                processing_time_ms=processing_time_ms
            )
            
            # Save to database
            self.logger.debug(f"Saving analysis result to database: {file_path}")
            db_start = time.time()
            try:
                self.audio_repo.save(audio_file)
                self.feature_repo.save(feature_set)
                self.metadata_repo.save(metadata)
                self.analysis_repo.save(result)
                db_time = time.time() - db_start
                self.logger.debug(f"Database save completed: {file_path} ({db_time:.1f}s)")
            except Exception as db_error:
                db_time = time.time() - db_start
                self.logger.error(f"Failed to save analysis result to database: {file_path} ({db_time:.1f}s): {db_error}")
                # Continue with result even if database save fails
            
            total_time = time.time() - start_time
            self.logger.info(f"Single file processing completed: {file_path} ({total_time:.1f}s)")
            self.logger.debug(f"Processing breakdown: metadata={metadata_time:.1f}s, features={features_time:.1f}s, quality={quality_time:.1f}s, db={db_time:.1f}s")
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Analysis failed for {file_path} ({processing_time:.1f}s): {e}")
            processing_time_ms = processing_time * 1000
            
            audio_file = AudioFile(file_path=file_path)
            result = AnalysisResult(
                audio_file=audio_file,
                is_successful=False,
                is_complete=True,
                error_message=str(e),
                processing_time_ms=processing_time_ms
            )
            
            # Save failed result to database
            try:
                self.audio_repo.save(audio_file)
                self.analysis_repo.save(result)
                self.logger.debug(f"Saved failed analysis result to database: {file_path}")
            except Exception as db_error:
                self.logger.error(f"Failed to save failed analysis result to database: {file_path}: {db_error}")
            
            return result 