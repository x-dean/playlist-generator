"""
Optimized Audio Analyzer for improved performance.
Implements batch processing with shared model instances.
"""

import os
import time
import threading
import json
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
import numpy as np

from .logging_setup import get_logger, log_universal
from .database import get_db_manager
from .model_manager import get_model_manager
from .audio_analyzer import safe_essentia_load
from .config_loader import config_loader
from .model_manager import ensure_models_loaded_once

logger = get_logger('playlista.optimized_analyzer')


class OptimizedAnalyzer:
    """
    High-performance audio analyzer with shared resources and batch processing.
    
    Key optimizations:
    - Shared model instances across all threads
    - Batch processing to reduce overhead
    - Efficient resource management
    - Pre-loaded models to avoid reinitialization
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the optimized analyzer."""
        self.config = config or config_loader.get_audio_analysis_config()
        self.db_manager = get_db_manager()
        
        # Initialize shared model manager once
        self.model_manager = get_model_manager(self.config)
        
        # Pre-load models to avoid per-thread loading
        self._preload_models()
        
        # Analysis settings
        self.sample_rate = self.config.get('SAMPLE_RATE', 44100)
        self.batch_size = self.config.get('BATCH_SIZE', 8)
        self.max_workers = self.config.get('MAX_WORKERS', 4)
        
        log_universal('INFO', 'OptimizedAnalyzer', 'Initialized with shared models')
    
    def _preload_models(self):
        """Pre-load all models to avoid per-thread initialization."""
        try:
            # Use global model loading to ensure they're loaded only once
            self.model_manager = ensure_models_loaded_once()
            if self.model_manager.is_musicnn_available():
                log_universal('INFO', 'OptimizedAnalyzer', 'MusicNN models pre-loaded globally')
            else:
                log_universal('WARNING', 'OptimizedAnalyzer', 'MusicNN models not available')
        except Exception as e:
            log_universal('ERROR', 'OptimizedAnalyzer', f'Model pre-loading failed: {e}')
    
    def analyze_files_batch(self, files: List[str], force_reanalysis: bool = False) -> Dict[str, Any]:
        """
        Analyze files using optimized batch processing.
        
        Args:
            files: List of file paths to analyze
            force_reanalysis: Force re-analysis even if cached
            
        Returns:
            Analysis results and statistics
        """
        if not files:
            return {'success_count': 0, 'failed_count': 0, 'total_time': 0}
        
        start_time = time.time()
        results = {
            'success_count': 0,
            'failed_count': 0,
            'processed_files': [],
            'total_time': 0
        }
        
        # Process in batches for better memory management
        batches = [files[i:i + self.batch_size] for i in range(0, len(files), self.batch_size)]
        
        log_universal('INFO', 'OptimizedAnalyzer', 
                     f'Processing {len(files)} files in {len(batches)} batches')
        
        for batch_idx, batch in enumerate(batches):
            log_universal('INFO', 'OptimizedAnalyzer', 
                         f'Processing batch {batch_idx + 1}/{len(batches)} ({len(batch)} files)')
            
            batch_results = self._process_batch(batch, force_reanalysis)
            
            results['success_count'] += batch_results['success_count']
            results['failed_count'] += batch_results['failed_count']
            results['processed_files'].extend(batch_results['processed_files'])
        
        results['total_time'] = time.time() - start_time
        
        # Calculate performance metrics
        success_rate = (results['success_count'] / len(files)) * 100 if files else 0
        throughput = len(files) / results['total_time'] if results['total_time'] > 0 else 0
        
        log_universal('INFO', 'OptimizedAnalyzer', 
                     f'Completed: {results["success_count"]} success, '
                     f'{results["failed_count"]} failed in {results["total_time"]:.2f}s')
        log_universal('INFO', 'OptimizedAnalyzer', 
                     f'Performance: {success_rate:.1f}% success rate, {throughput:.2f} files/s')
        
        return results
    
    def _process_batch(self, batch: List[str], force_reanalysis: bool) -> Dict[str, Any]:
        """Process a batch of files with optimized threading."""
        batch_results = {
            'success_count': 0,
            'failed_count': 0,
            'processed_files': []
        }
        
        # Ensure models are loaded globally before threading
        try:
            ensure_models_loaded_once()
            log_universal('DEBUG', 'OptimizedAnalyzer', 'Models globally ensured for batch processing')
        except Exception as e:
            log_universal('WARNING', 'OptimizedAnalyzer', f'Global model pre-loading failed: {e}')
        
        # Use ThreadPoolExecutor for I/O bound operations
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(batch))) as executor:
            # Submit all files in batch
            futures = {
                executor.submit(self._analyze_single_file, file_path, force_reanalysis): file_path
                for file_path in batch
            }
            
            # Collect results as they complete
            for future in as_completed(futures):
                file_path = futures[future]
                try:
                    success = future.result()
                    if success:
                        batch_results['success_count'] += 1
                    else:
                        batch_results['failed_count'] += 1
                    
                    batch_results['processed_files'].append({
                        'file_path': file_path,
                        'success': success,
                        'timestamp': time.time()
                    })
                    
                except Exception as e:
                    log_universal('ERROR', 'OptimizedAnalyzer', f'Future failed for {file_path}: {e}')
                    batch_results['failed_count'] += 1
        
        # Force aggressive memory cleanup after batch
        self.model_manager.force_memory_cleanup()
        
        return batch_results
    
    def _analyze_single_file(self, file_path: str, force_reanalysis: bool) -> bool:
        """
        Analyze a single file using shared resources.
        
        This method is optimized to use pre-loaded models and minimal overhead.
        """
        try:
            # Quick cache check first
            if not force_reanalysis and self._is_already_analyzed(file_path):
                log_universal('DEBUG', 'OptimizedAnalyzer', f'Already analyzed: {os.path.basename(file_path)}')
                return True
            
            # Load audio efficiently
            audio, sample_rate = safe_essentia_load(file_path, self.sample_rate)
            if audio is None:
                log_universal('WARNING', 'OptimizedAnalyzer', f'Failed to load audio: {file_path}')
                return False
            
            # Extract features using shared models
            features = self._extract_features_optimized(audio, sample_rate, file_path)
            
            # Clean up audio data immediately
            del audio
            
            # Save to database efficiently
            success = self._save_analysis_result(file_path, features)
            
            # Clean up features data
            del features
            
            if success:
                log_universal('DEBUG', 'OptimizedAnalyzer', f'Analyzed: {os.path.basename(file_path)}')
            else:
                log_universal('WARNING', 'OptimizedAnalyzer', f'Save failed: {os.path.basename(file_path)}')
            
            return success
            
        except Exception as e:
            log_universal('ERROR', 'OptimizedAnalyzer', f'Analysis failed for {file_path}: {e}')
            return False
        finally:
            # Force garbage collection after each file
            import gc
            gc.collect()
    
    def _is_already_analyzed(self, file_path: str) -> bool:
        """Check if file is already analyzed."""
        try:
            with self.db_manager._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT status FROM tracks WHERE file_path = ? AND status = 'analyzed'",
                    (file_path,)
                )
                return cursor.fetchone() is not None
        except Exception:
            return False
    
    def _extract_features_optimized(self, audio: np.ndarray, sample_rate: int, file_path: str) -> Dict[str, Any]:
        """
        Extract features using pre-loaded shared models.
        
        This avoids model reinitialization overhead.
        """
        features = {}
        
        try:
            # Basic audio features (fast)
            features.update(self._extract_basic_features(audio, sample_rate))
            
            # MusicNN features using shared models (if enabled and available)
            extract_musicnn = self.config.get('EXTRACT_MUSICNN', True)
            if extract_musicnn and self.model_manager.is_musicnn_available():
                musicnn_features = self._extract_musicnn_optimized(audio, sample_rate)
                features.update(musicnn_features)
            elif not extract_musicnn:
                log_universal('DEBUG', 'OptimizedAnalyzer', 'MusicNN extraction disabled by config')
            else:
                log_universal('DEBUG', 'OptimizedAnalyzer', 'MusicNN not available')
            
            # Additional features can be added here
            
        except Exception as e:
            log_universal('ERROR', 'OptimizedAnalyzer', f'Feature extraction failed: {e}')
        
        return features
    
    def _extract_basic_features(self, audio: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Extract basic audio features quickly."""
        try:
            # Duration
            duration = len(audio) / sample_rate
            
            # RMS energy
            rms = np.sqrt(np.mean(audio**2))
            
            # Zero crossing rate
            zcr = np.mean(np.abs(np.diff(np.sign(audio)))) / 2
            
            return {
                'duration': duration,
                'rms_energy': float(rms),
                'zero_crossing_rate': float(zcr),
                'sample_rate': sample_rate,
                'audio_length': len(audio)
            }
            
        except Exception as e:
            log_universal('ERROR', 'OptimizedAnalyzer', f'Basic features failed: {e}')
            return {}
    
    def _extract_musicnn_optimized(self, audio: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Extract MusicNN features using shared model instances."""
        try:
            # Get pre-loaded models (no initialization overhead)
            activations_model, embeddings_model, tag_names, metadata = self.model_manager.get_musicnn_models()
            
            # Prepare audio for MusicNN (16kHz, mono)
            if sample_rate != 16000:
                # Proper resampling using librosa for better quality
                try:
                    import librosa
                    audio_16k = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
                except ImportError:
                    # Fallback to simple downsampling
                    downsample_factor = sample_rate // 16000
                    if downsample_factor > 0:
                        audio_16k = audio[::downsample_factor]
                    else:
                        audio_16k = audio
            else:
                audio_16k = audio
            
            # Ensure audio is 1D vector (not matrix)
            if len(audio_16k.shape) > 1:
                audio_16k = audio_16k.flatten()
            
            # Ensure proper length (MusicNN expects specific segments)
            target_length = 16000 * 3  # 3 seconds at 16kHz
            if len(audio_16k) > target_length:
                # Take middle section
                start = (len(audio_16k) - target_length) // 2
                audio_16k = audio_16k[start:start + target_length]
            elif len(audio_16k) < target_length:
                # Pad with zeros
                audio_16k = np.pad(audio_16k, (0, target_length - len(audio_16k)))
            
            # Ensure audio is float32 and normalized
            audio_16k = audio_16k.astype(np.float32)
            if np.max(np.abs(audio_16k)) > 0:
                audio_16k = audio_16k / np.max(np.abs(audio_16k))
            
            # Run inference (using shared models) - ensure proper vector format
            activations = activations_model(audio_16k)
            embeddings = embeddings_model(audio_16k)
            
            # Process results - handle numpy arrays properly
            if hasattr(activations, 'numpy'):
                activations = activations.numpy()
            if hasattr(embeddings, 'numpy'):
                embeddings = embeddings.numpy()
                
            # Flatten if needed
            if len(activations.shape) > 1:
                activations = activations.flatten()
            if len(embeddings.shape) > 1:
                embeddings = embeddings.flatten()
            
            # Get top tags (ensure we have enough tags)
            num_tags = min(10, len(tag_names), len(activations))
            top_indices = np.argsort(activations)[-num_tags:][::-1]
            tags = {tag_names[i]: float(activations[i]) for i in top_indices}
            
            return {
                'musicnn_tags': tags,
                'musicnn_embeddings': embeddings.tolist(),
                'musicnn_activations': activations.tolist()
            }
            
        except Exception as e:
            log_universal('ERROR', 'OptimizedAnalyzer', f'MusicNN extraction failed: {e}')
            return {}
    
    def _save_analysis_result(self, file_path: str, features: Dict[str, Any]) -> bool:
        """Save analysis result to database efficiently."""
        try:
            # Update track status and basic info
            with self.db_manager._get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Update tracks table
                cursor.execute("""
                    UPDATE tracks 
                    SET status = 'analyzed', 
                        analysis_status = 'completed',
                        duration = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE file_path = ?
                """, (
                    features.get('duration', 0),  # Duration in seconds
                    file_path
                ))
                
                # Save MusicNN features if available
                if 'musicnn_tags' in features:
                    # Get the track_id for this file_path
                    cursor.execute("SELECT id FROM tracks WHERE file_path = ?", (file_path,))
                    track_result = cursor.fetchone()
                    if track_result:
                        track_id = track_result[0]
                        # Save to musicnn_features table using proper schema
                        cursor.execute("""
                            INSERT OR REPLACE INTO musicnn_features 
                            (track_id, embedding, tags, confidence, model_version, created_at, updated_at)
                            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                        """, (
                            track_id,
                            json.dumps(features.get('musicnn_embeddings', [])),
                            json.dumps(features['musicnn_tags']),
                            1.0,  # Default confidence
                            'musicnn-msd-1'  # Model version
                        ))
                
                conn.commit()
                return True
                
        except Exception as e:
            log_universal('ERROR', 'OptimizedAnalyzer', f'Save failed for {file_path}: {e}')
            return False


# Convenience function to use the optimized analyzer
def analyze_files_optimized(files: List[str], force_reanalysis: bool = False) -> Dict[str, Any]:
    """
    Analyze files using the optimized analyzer.
    
    Args:
        files: List of file paths to analyze
        force_reanalysis: Force re-analysis even if cached
        
    Returns:
        Analysis results and performance statistics
    """
    analyzer = OptimizedAnalyzer()
    return analyzer.analyze_files_batch(files, force_reanalysis)
