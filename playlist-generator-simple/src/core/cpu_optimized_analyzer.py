#!/usr/bin/env python3
"""
CPU-Optimized Audio Analyzer for Small Models (MusicNN, etc.)

Focuses on optimizing melspectrogram extraction and feature preprocessing
since the model inference is not the bottleneck for small models.
"""

import os
import time
import logging
import multiprocessing as mp
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from pathlib import Path

# Import audio processing libraries
try:
    import essentia.standard as es
    ESSENTIA_AVAILABLE = True
except ImportError:
    ESSENTIA_AVAILABLE = False

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

from .logging_setup import get_logger

logger = get_logger('playlista.cpu_optimizer')

# Constants
DEFAULT_SAMPLE_RATE = 44100
DEFAULT_N_MELS = 96
DEFAULT_N_FFT = 2048
DEFAULT_HOP_LENGTH = 512
DEFAULT_WINDOW_LENGTH = 1024
DEFAULT_NUM_WORKERS = max(1, mp.cpu_count() - 1)  # Leave one core free


class CPUOptimizedAnalyzer:
    """
    CPU-optimized analyzer for small models like MusicNN.
    
    Focuses on:
    - Multi-process melspectrogram extraction
    - Batch processing of audio features
    - Optimized preprocessing for small models
    - Memory-efficient processing
    """
    
    def __init__(self, 
                 num_workers: int = DEFAULT_NUM_WORKERS,
                 sample_rate: int = DEFAULT_SAMPLE_RATE,
                 n_mels: int = DEFAULT_N_MELS,
                 n_fft: int = DEFAULT_N_FFT,
                 hop_length: int = DEFAULT_HOP_LENGTH,
                 window_length: int = DEFAULT_WINDOW_LENGTH):
        """
        Initialize CPU-optimized analyzer.
        
        Args:
            num_workers: Number of worker processes for parallel processing
            sample_rate: Audio sample rate
            n_mels: Number of mel frequency bins
            n_fft: FFT window size
            hop_length: Hop length for STFT
            window_length: Window length for STFT
        """
        self.num_workers = num_workers
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window_length = window_length
        
        # Initialize processing pool
        self.pool = None
        self._init_processing_pool()
        
        logger.info(f" CPU-Optimized Analyzer initialized:")
        logger.info(f"   Workers: {self.num_workers}")
        logger.info(f"   Sample rate: {self.sample_rate}Hz")
        logger.info(f"   Mel bins: {self.n_mels}")
        logger.info(f"   FFT size: {self.n_fft}")
        logger.info(f"   Hop length: {self.hop_length}")
    
    def _init_processing_pool(self):
        """Initialize multiprocessing pool."""
        try:
            if self.num_workers > 1:
                self.pool = mp.Pool(processes=self.num_workers)
                logger.info(f" Initialized processing pool with {self.num_workers} workers")
            else:
                logger.info("ℹ️ Using single-threaded processing")
        except Exception as e:
            logger.warning(f"️ Could not initialize processing pool: {e}")
            self.pool = None
    
    def extract_melspectrograms_batch(self, audio_files: List[str]) -> List[np.ndarray]:
        """
        Extract melspectrograms from multiple audio files in parallel.
        
        Args:
            audio_files: List of audio file paths
            
        Returns:
            List of melspectrogram arrays
        """
        logger.info(f" Extracting melspectrograms for {len(audio_files)} files")
        start_time = time.time()
        
        try:
            if self.pool and len(audio_files) > 1:
                # Parallel processing
                results = self.pool.map(self._extract_melspectrogram_worker, audio_files)
                logger.info(f" Parallel extraction completed in {time.time() - start_time:.2f}s")
            else:
                # Sequential processing
                results = [self._extract_melspectrogram_worker(file) for file in audio_files]
                logger.info(f" Sequential extraction completed in {time.time() - start_time:.2f}s")
            
            # Filter out None results (failed extractions)
            valid_results = [r for r in results if r is not None]
            logger.info(f" Successfully extracted {len(valid_results)}/{len(audio_files)} melspectrograms")
            
            return valid_results
            
        except Exception as e:
            logger.error(f" Error in batch melspectrogram extraction: {e}")
            return []
    
    def _extract_melspectrogram_worker(self, audio_file: str) -> Optional[np.ndarray]:
        """
        Worker function for melspectrogram extraction.
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            Melspectrogram array or None if failed
        """
        try:
            if ESSENTIA_AVAILABLE:
                return self._extract_melspectrogram_essentia(audio_file)
            elif LIBROSA_AVAILABLE:
                return self._extract_melspectrogram_librosa(audio_file)
            else:
                logger.error(" No audio library available for melspectrogram extraction")
                return None
                
        except Exception as e:
            logger.warning(f"️ Failed to extract melspectrogram from {audio_file}: {e}")
            return None
    
    def _extract_melspectrogram_essentia(self, audio_file: str) -> Optional[np.ndarray]:
        """Extract melspectrogram using Essentia."""
        try:
            # Load audio
            loader = es.AudioLoader(filename=audio_file)
            audio, sample_rate, _, _ = loader()
            
            # Resample if needed
            if sample_rate != self.sample_rate:
                resampler = es.Resample(inputSampleRate=sample_rate, outputSampleRate=self.sample_rate)
                audio = resampler(audio)
            
            # Extract melspectrogram
            windowing = es.Windowing(type='blackmanharris62', size=self.window_length)
            spectrum = es.Spectrum(size=self.n_fft)
            mel_bands = es.MelBands(
                numberBands=self.n_mels,
                sampleRate=self.sample_rate,
                lowFrequencyBound=0,
                highFrequencyBound=self.sample_rate / 2
            )
            
            # Process audio in frames
            mel_spectrogram = []
            frame_size = self.n_fft
            hop_size = self.hop_length
            
            for i in range(0, len(audio) - frame_size + 1, hop_size):
                frame = audio[i:i + frame_size]
                windowed = windowing(frame)
                spec = spectrum(windowed)
                mel = mel_bands(spec)
                mel_spectrogram.append(mel)
            
            if mel_spectrogram:
                return np.array(mel_spectrogram)
            else:
                return None
                
        except Exception as e:
            logger.warning(f"️ Essentia melspectrogram extraction failed: {e}")
            return None
    
    def _extract_melspectrogram_librosa(self, audio_file: str) -> Optional[np.ndarray]:
        """Extract melspectrogram using Librosa."""
        try:
            # Load audio
            audio, sr = librosa.load(audio_file, sr=self.sample_rate, mono=True)
            
            # Extract melspectrogram
            mel_spectrogram = librosa.feature.melspectrogram(
                y=audio,
                sr=self.sample_rate,
                n_mels=self.n_mels,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window='blackmanharris'
            )
            
            # Convert to dB scale
            mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
            
            # Transpose to get time as first dimension
            return mel_spectrogram_db.T
            
        except Exception as e:
            logger.warning(f"️ Librosa melspectrogram extraction failed: {e}")
            return None
    
    def process_audio_batch(self, audio_files: List[str], 
                          batch_size: int = 4) -> List[Dict[str, Any]]:
        """
        Process multiple audio files in batches.
        
        Args:
            audio_files: List of audio file paths
            batch_size: Number of files to process in each batch
            
        Returns:
            List of processing results
        """
        logger.info(f" Processing {len(audio_files)} audio files in batches of {batch_size}")
        
        results = []
        total_batches = (len(audio_files) + batch_size - 1) // batch_size
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(audio_files))
            batch_files = audio_files[start_idx:end_idx]
            
            logger.info(f" Processing batch {batch_idx + 1}/{total_batches} ({len(batch_files)} files)")
            
            # Extract melspectrograms for this batch
            melspectrograms = self.extract_melspectrograms_batch(batch_files)
            
            # Process each file in the batch
            for i, (audio_file, melspectrogram) in enumerate(zip(batch_files, melspectrograms)):
                if melspectrogram is not None:
                    result = {
                        'file_path': audio_file,
                        'melspectrogram': melspectrogram,
                        'duration': melspectrogram.shape[0] * self.hop_length / self.sample_rate,
                        'shape': melspectrogram.shape,
                        'success': True
                    }
                else:
                    result = {
                        'file_path': audio_file,
                        'melspectrogram': None,
                        'duration': 0.0,
                        'shape': None,
                        'success': False,
                        'error': 'Failed to extract melspectrogram'
                    }
                
                results.append(result)
        
        # Log summary
        successful = sum(1 for r in results if r['success'])
        logger.info(f" Batch processing completed: {successful}/{len(results)} successful")
        
        return results
    
    def optimize_for_musicnn(self, audio_files: List[str]) -> List[Dict[str, Any]]:
        """
        Optimized processing specifically for MusicNN model.
        
        Args:
            audio_files: List of audio file paths
            
        Returns:
            List of MusicNN-ready features
        """
        logger.info(f" Optimizing for MusicNN: {len(audio_files)} files")
        
        # MusicNN-specific parameters
        musicnn_sample_rate = 16000  # MusicNN expects 16kHz
        musicnn_n_mels = 96
        musicnn_n_fft = 2048
        musicnn_hop_length = 512
        
        # Create MusicNN-optimized analyzer
        musicnn_analyzer = CPUOptimizedAnalyzer(
            num_workers=self.num_workers,
            sample_rate=musicnn_sample_rate,
            n_mels=musicnn_n_mels,
            n_fft=musicnn_n_fft,
            hop_length=musicnn_hop_length
        )
        
        # Process with MusicNN-optimized parameters
        results = musicnn_analyzer.process_audio_batch(audio_files, batch_size=4)
        
        # Add MusicNN-specific metadata
        for result in results:
            if result['success']:
                result['model_ready'] = True
                result['model_type'] = 'musicnn'
                result['sample_rate'] = musicnn_sample_rate
            else:
                result['model_ready'] = False
        
        return results
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            'num_workers': self.num_workers,
            'sample_rate': self.sample_rate,
            'n_mels': self.n_mels,
            'n_fft': self.n_fft,
            'hop_length': self.hop_length,
            'window_length': self.window_length,
            'pool_active': self.pool is not None
        }
    
    def cleanup(self):
        """Clean up resources."""
        if self.pool:
            self.pool.close()
            self.pool.join()
            logger.info(" Processing pool closed")


# Global analyzer instance
_cpu_optimized_analyzer: Optional[CPUOptimizedAnalyzer] = None


def get_cpu_optimized_analyzer(num_workers: int = DEFAULT_NUM_WORKERS) -> CPUOptimizedAnalyzer:
    """
    Get the global CPU-optimized analyzer instance.
    
    Args:
        num_workers: Number of worker processes
        
    Returns:
        CPU-optimized analyzer instance
    """
    global _cpu_optimized_analyzer
    if _cpu_optimized_analyzer is None:
        _cpu_optimized_analyzer = CPUOptimizedAnalyzer(num_workers=num_workers)
    return _cpu_optimized_analyzer 