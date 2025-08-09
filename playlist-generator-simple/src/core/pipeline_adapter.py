"""
Pipeline Adapter for Optimized Audio Analysis.

This module provides an adapter to integrate the optimized pipeline with the existing
audio analyzer system, allowing seamless use of the new optimized approach while
maintaining compatibility with the current architecture.
"""

import os
from typing import Dict, Any, Optional
from pathlib import Path

# Import local modules
from .logging_setup import get_logger, log_universal
from .config_loader import config_loader
from .optimized_pipeline import OptimizedAudioPipeline


logger = get_logger('playlista.pipeline_adapter')


class PipelineAdapter:
    """
    Adapter to integrate optimized pipeline with existing audio analyzer.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the pipeline adapter."""
        self.config = config or config_loader.get_audio_analysis_config()
        
        # Check if optimized pipeline is enabled
        self.optimized_enabled = self.config.get('OPTIMIZED_PIPELINE_ENABLED', True)
        
        # File size thresholds for using optimized pipeline
        self.min_file_size_mb = self.config.get('OPTIMIZED_PIPELINE_MIN_SIZE_MB', 5)
        self.max_file_size_mb = self.config.get('OPTIMIZED_PIPELINE_MAX_SIZE_MB', 200)
        
        # Initialize optimized pipeline if enabled
        self.optimized_pipeline = None
        if self.optimized_enabled:
            try:
                self.optimized_pipeline = OptimizedAudioPipeline(config)
                log_universal('INFO', 'PipelineAdapter', 'Optimized pipeline initialized successfully')
            except Exception as e:
                log_universal('ERROR', 'PipelineAdapter', f'Optimized pipeline initialization failed: {e}')
                self.optimized_enabled = False
    
    def should_use_optimized_pipeline(self, file_path: str, file_size_mb: float = None) -> bool:
        """
        Determine if optimized pipeline should be used for this file.
        
        Args:
            file_path: Path to audio file
            file_size_mb: File size in MB (optional, will be calculated if not provided)
            
        Returns:
            True if optimized pipeline should be used
        """
        if not self.optimized_enabled or not self.optimized_pipeline:
            return False
        
        try:
            # Get file size if not provided
            if file_size_mb is None:
                file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            
            # Check size thresholds
            use_optimized = (self.min_file_size_mb <= file_size_mb <= self.max_file_size_mb)
            
            log_universal('DEBUG', 'PipelineAdapter', 
                         f'File {file_path} ({file_size_mb:.1f}MB): '
                         f'use_optimized={use_optimized}')
            
            return use_optimized
            
        except Exception as e:
            log_universal('WARNING', 'PipelineAdapter', 
                         f'Error determining pipeline for {file_path}: {e}')
            return False
    
    def analyze_with_optimized_pipeline(self, file_path: str, 
                                      metadata: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """
        Analyze audio using the optimized pipeline.
        
        Args:
            file_path: Path to audio file
            metadata: Optional metadata dictionary
            
        Returns:
            Analysis results or None if failed
        """
        if not self.optimized_pipeline:
            log_universal('WARNING', 'PipelineAdapter', 'Optimized pipeline not available')
            return None
        
        try:
            log_universal('INFO', 'PipelineAdapter', f'Using optimized pipeline for {file_path}')
            
            # Run optimized analysis
            result = self.optimized_pipeline.analyze_track(file_path, metadata)
            
            # Convert to format compatible with existing analyzer
            converted_result = self._convert_optimized_result(result)
            
            log_universal('INFO', 'PipelineAdapter', 
                         f'Optimized analysis completed for {file_path}')
            
            return converted_result
            
        except Exception as e:
            log_universal('ERROR', 'PipelineAdapter', 
                         f'Optimized pipeline analysis failed for {file_path}: {e}')
            return None
    
    def _convert_optimized_result(self, optimized_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert optimized pipeline result to format compatible with existing analyzer.
        
        Args:
            optimized_result: Result from optimized pipeline
            
        Returns:
            Converted result dictionary
        """
        try:
            # Create result structure compatible with existing analyzer
            converted = {
                'analysis_method': 'optimized_pipeline',
                'analysis_success': True,
                'pipeline_info': optimized_result.get('pipeline_info', {})
            }
            
            # Map Essentia features
            essentia_features = self._extract_essentia_from_optimized(optimized_result)
            converted.update(essentia_features)
            
            # Map MusiCNN features
            musicnn_features = self._extract_musicnn_from_optimized(optimized_result)
            converted.update(musicnn_features)
            
            # Map segment information
            segment_info = optimized_result.get('segment_info', {})
            if segment_info:
                converted['segment_info'] = segment_info
            
            return converted
            
        except Exception as e:
            log_universal('ERROR', 'PipelineAdapter', f'Result conversion failed: {e}')
            return {
                'analysis_method': 'optimized_pipeline',
                'analysis_success': False,
                'error': str(e)
            }
    
    def _extract_essentia_from_optimized(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract Essentia features from optimized result."""
        essentia_features = {}
        
        # Map aggregated Essentia features
        feature_mappings = {
            'tempo_mean': 'tempo',
            'tempo_confidence_mean': 'tempo_confidence',
            'key_strength_mean': 'key_strength',
            'loudness_mean': 'loudness',
            'spectral_centroid_mean_mean': 'spectral_centroid',
            'spectral_centroid_std_mean': 'spectral_centroid_std'
        }
        
        for optimized_key, legacy_key in feature_mappings.items():
            if optimized_key in result:
                essentia_features[legacy_key] = result[optimized_key]
        
        # Map categorical features with confidence
        if 'key' in result:
            essentia_features['key'] = result['key']
        if 'scale' in result:
            essentia_features['scale'] = result['scale']
        if 'key_confidence' in result:
            essentia_features['key_confidence'] = result['key_confidence']
        
        # Calculate derived features
        if 'tempo_mean' in result:
            essentia_features['bpm'] = result['tempo_mean']
        
        return essentia_features
    
    def _extract_musicnn_from_optimized(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract MusiCNN features from optimized result."""
        musicnn_features = {}
        
        # Map MusiCNN tags
        if 'musicnn_tags' in result:
            musicnn_features['musicnn_tags'] = result['musicnn_tags']
            
            # Create tag list format for compatibility
            tags = result['musicnn_tags']
            if tags:
                sorted_tags = sorted(tags.items(), key=lambda x: x[1], reverse=True)
                musicnn_features['tags'] = {tag: prob for tag, prob in sorted_tags}
        
        # Map MusiCNN embeddings
        if 'musicnn_embeddings' in result:
            musicnn_features['embedding'] = result['musicnn_embeddings']
            musicnn_features['musicnn_embeddings'] = result['musicnn_embeddings']
        
        # Map derived predictions
        if 'musicnn_genre' in result:
            musicnn_features['predicted_genre'] = result['musicnn_genre']
        
        if 'musicnn_mood' in result:
            musicnn_features['predicted_mood'] = result['musicnn_mood']
        
        if 'musicnn_top_tags' in result:
            musicnn_features['top_tags'] = result['musicnn_top_tags']
        
        # Add availability flags
        musicnn_features['musicnn_available'] = result.get('musicnn_available', False)
        
        if 'musicnn_model_size' in result:
            musicnn_features['musicnn_model_size'] = result['musicnn_model_size']
        
        return musicnn_features
    
    def get_pipeline_statistics(self) -> Dict[str, Any]:
        """Get statistics about pipeline usage."""
        stats = {
            'optimized_enabled': self.optimized_enabled,
            'min_file_size_mb': self.min_file_size_mb,
            'max_file_size_mb': self.max_file_size_mb
        }
        
        if self.optimized_pipeline:
            pipeline_config = self.optimized_pipeline.config
            stats.update({
                'resource_mode': self.optimized_pipeline.resource_mode,
                'sample_rate': self.optimized_pipeline.optimized_sample_rate,
                'segment_length': self.optimized_pipeline.segment_length,
                'max_segments': self.optimized_pipeline.max_segments,
                'cache_enabled': self.optimized_pipeline.cache_enabled
            })
        
        return stats


# Global instance for shared use
_pipeline_adapter = None


def get_pipeline_adapter(config: Dict[str, Any] = None) -> PipelineAdapter:
    """Get shared pipeline adapter instance."""
    global _pipeline_adapter
    
    if _pipeline_adapter is None:
        _pipeline_adapter = PipelineAdapter(config)
    
    return _pipeline_adapter
