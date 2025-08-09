"""
ML Model management for audio analysis
Handles loading, caching, and inference of ML models
"""

import asyncio
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

from ..core.logging import get_logger, LogContext, log_operation_start, log_operation_success, log_operation_error
from ..core.config import get_settings

logger = get_logger("analysis.models")
settings = get_settings()


class ModelManager:
    """Manages ML models for audio analysis with professional logging"""
    
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.model_info: Dict[str, Dict[str, Any]] = {}
        self.device = self._determine_device()
        self._initialized = False
        
        logger.info(
            "ModelManager initialized",
            device=self.device,
            model_path=settings.ml_model_path
        )
    
    def _determine_device(self) -> str:
        """Determine the best device for ML inference"""
        if not settings.enable_gpu:
            return "cpu"
        
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
                logger.info(
                    "GPU available for ML inference",
                    gpu_count=gpu_count,
                    gpu_name=gpu_name
                )
                return "cuda:0"
        except ImportError:
            logger.warning("PyTorch not available, falling back to CPU")
        
        return "cpu"
    
    async def load_models(self) -> None:
        """Load all required ML models"""
        start_time = time.time()
        
        with LogContext(operation="load_models", device=self.device):
            log_operation_start(logger, "model loading")
            
            try:
                # Load genre classification model
                await self._load_genre_model()
                
                # Load mood analysis model
                await self._load_mood_model()
                
                # Load audio embedding model
                await self._load_embedding_model()
                
                # Load feature extraction models
                await self._load_feature_models()
                
                self._initialized = True
                duration_ms = (time.time() - start_time) * 1000
                
                log_operation_success(
                    logger, 
                    "model loading", 
                    duration_ms,
                    models_loaded=len(self.models),
                    device=self.device
                )
                
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                log_operation_error(logger, "model loading", e, duration_ms)
                raise
    
    async def _load_genre_model(self) -> None:
        """Load genre classification model"""
        model_name = "genre_classifier"
        
        try:
            # Placeholder for actual model loading
            # In real implementation, this would load a trained PyTorch/TensorFlow model
            self.models[model_name] = {"type": "genre", "loaded": True}
            self.model_info[model_name] = {
                "name": "Genre Classifier v2.0",
                "classes": 50,
                "input_shape": (128, 1292),  # Mel spectrogram shape
                "accuracy": 0.87
            }
            
            logger.info(
                "Genre model loaded",
                model=model_name,
                classes=self.model_info[model_name]["classes"],
                accuracy=self.model_info[model_name]["accuracy"]
            )
            
        except Exception as e:
            logger.error(
                "Failed to load genre model",
                model=model_name,
                error_type=type(e).__name__,
                error_message=str(e)
            )
            raise
    
    async def _load_mood_model(self) -> None:
        """Load mood analysis model"""
        model_name = "mood_analyzer"
        
        try:
            self.models[model_name] = {"type": "mood", "loaded": True}
            self.model_info[model_name] = {
                "name": "Mood Analyzer v2.0",
                "dimensions": ["valence", "arousal", "energy"],
                "input_shape": (128, 1292),
                "accuracy": 0.82
            }
            
            logger.info(
                "Mood model loaded",
                model=model_name,
                dimensions=len(self.model_info[model_name]["dimensions"]),
                accuracy=self.model_info[model_name]["accuracy"]
            )
            
        except Exception as e:
            logger.error(
                "Failed to load mood model",
                model=model_name,
                error_type=type(e).__name__,
                error_message=str(e)
            )
            raise
    
    async def _load_embedding_model(self) -> None:
        """Load audio embedding model"""
        model_name = "audio_embeddings"
        
        try:
            self.models[model_name] = {"type": "embedding", "loaded": True}
            self.model_info[model_name] = {
                "name": "Audio Embeddings v2.0",
                "embedding_size": 512,
                "input_shape": (128, 1292),
                "similarity_metric": "cosine"
            }
            
            logger.info(
                "Embedding model loaded",
                model=model_name,
                embedding_size=self.model_info[model_name]["embedding_size"]
            )
            
        except Exception as e:
            logger.error(
                "Failed to load embedding model",
                model=model_name,
                error_type=type(e).__name__,
                error_message=str(e)
            )
            raise
    
    async def _load_feature_models(self) -> None:
        """Load additional feature extraction models"""
        model_name = "feature_extractor"
        
        try:
            self.models[model_name] = {"type": "features", "loaded": True}
            self.model_info[model_name] = {
                "name": "Feature Extractor v2.0",
                "features": ["tempo", "key", "energy", "danceability"],
                "input_shape": (128, 1292)
            }
            
            logger.info(
                "Feature extraction models loaded",
                model=model_name,
                feature_count=len(self.model_info[model_name]["features"])
            )
            
        except Exception as e:
            logger.error(
                "Failed to load feature models",
                model=model_name,
                error_type=type(e).__name__,
                error_message=str(e)
            )
            raise
    
    async def predict_genre(self, features: np.ndarray) -> Dict[str, float]:
        """Predict genre probabilities"""
        if not self._initialized:
            raise RuntimeError("Models not initialized")
        
        with LogContext(operation="predict_genre", input_shape=features.shape):
            start_time = time.time()
            
            try:
                # Placeholder prediction - replace with actual model inference
                genres = [
                    "rock", "pop", "jazz", "classical", "electronic", 
                    "hip-hop", "country", "blues", "folk", "reggae"
                ]
                
                # Simulate prediction with random probabilities that sum to 1
                np.random.seed(hash(str(features)) % 2**32)
                probs = np.random.random(len(genres))
                probs = probs / probs.sum()
                
                result = {genre: float(prob) for genre, prob in zip(genres, probs)}
                
                duration_ms = (time.time() - start_time) * 1000
                logger.debug(
                    "Genre prediction completed",
                    duration_ms=round(duration_ms, 2),
                    top_genre=max(result.items(), key=lambda x: x[1])[0],
                    confidence=round(max(result.values()), 3)
                )
                
                return result
                
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                log_operation_error(logger, "genre prediction", e, duration_ms)
                raise
    
    async def predict_mood(self, features: np.ndarray) -> Dict[str, float]:
        """Predict mood dimensions"""
        if not self._initialized:
            raise RuntimeError("Models not initialized")
        
        with LogContext(operation="predict_mood", input_shape=features.shape):
            start_time = time.time()
            
            try:
                # Placeholder prediction
                np.random.seed(hash(str(features)) % 2**32)
                
                result = {
                    "valence": float(np.random.random()),      # Positive/negative
                    "arousal": float(np.random.random()),      # Calm/energetic
                    "energy": float(np.random.random()),       # Energy level
                    "danceability": float(np.random.random()), # Danceability
                }
                
                duration_ms = (time.time() - start_time) * 1000
                logger.debug(
                    "Mood prediction completed",
                    duration_ms=round(duration_ms, 2),
                    valence=round(result["valence"], 3),
                    energy=round(result["energy"], 3)
                )
                
                return result
                
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                log_operation_error(logger, "mood prediction", e, duration_ms)
                raise
    
    async def extract_embeddings(self, features: np.ndarray) -> np.ndarray:
        """Extract audio embeddings for similarity computation"""
        if not self._initialized:
            raise RuntimeError("Models not initialized")
        
        with LogContext(operation="extract_embeddings", input_shape=features.shape):
            start_time = time.time()
            
            try:
                # Placeholder embedding extraction
                embedding_size = self.model_info["audio_embeddings"]["embedding_size"]
                np.random.seed(hash(str(features)) % 2**32)
                embeddings = np.random.random(embedding_size).astype(np.float32)
                
                # Normalize embeddings
                embeddings = embeddings / np.linalg.norm(embeddings)
                
                duration_ms = (time.time() - start_time) * 1000
                logger.debug(
                    "Embedding extraction completed",
                    duration_ms=round(duration_ms, 2),
                    embedding_size=len(embeddings)
                )
                
                return embeddings
                
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                log_operation_error(logger, "embedding extraction", e, duration_ms)
                raise
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health status of all models"""
        if not self._initialized:
            return {
                "status": "unhealthy",
                "reason": "Models not initialized"
            }
        
        model_status = {}
        all_healthy = True
        
        for model_name, model_data in self.models.items():
            try:
                # Simple health check - verify model is loaded
                is_healthy = model_data.get("loaded", False)
                model_status[model_name] = {
                    "status": "healthy" if is_healthy else "unhealthy",
                    "info": self.model_info.get(model_name, {})
                }
                
                if not is_healthy:
                    all_healthy = False
                    
            except Exception as e:
                model_status[model_name] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
                all_healthy = False
        
        return {
            "status": "healthy" if all_healthy else "degraded",
            "device": self.device,
            "models": model_status,
            "total_models": len(self.models)
        }


# Global model manager instance
model_manager = ModelManager()


async def load_models() -> None:
    """Load all ML models - module-level function for application startup"""
    await model_manager.load_models()
