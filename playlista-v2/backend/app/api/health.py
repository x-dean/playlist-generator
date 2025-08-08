"""
Health check endpoints
"""

import asyncio
import time
from typing import Dict, Any

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from ..database.connection import get_db_session, get_redis
from ..core.logging import get_logger

router = APIRouter()
logger = get_logger("api.health")


@router.get("/")
async def health_check() -> Dict[str, Any]:
    """Basic health check"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "2.0.0"
    }


@router.get("/detailed")
async def detailed_health_check(
    db: AsyncSession = Depends(get_db_session)
) -> Dict[str, Any]:
    """Detailed health check including database connectivity"""
    
    health_data = {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "2.0.0",
        "checks": {}
    }
    
    # Check PostgreSQL
    try:
        start_time = time.time()
        await db.execute("SELECT 1")
        pg_time = (time.time() - start_time) * 1000
        health_data["checks"]["postgresql"] = {
            "status": "healthy",
            "response_time_ms": round(pg_time, 2)
        }
    except Exception as e:
        health_data["checks"]["postgresql"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        health_data["status"] = "degraded"
    
    # Check Redis
    try:
        start_time = time.time()
        redis = await get_redis()
        await redis.ping()
        redis_time = (time.time() - start_time) * 1000
        health_data["checks"]["redis"] = {
            "status": "healthy",
            "response_time_ms": round(redis_time, 2)
        }
    except Exception as e:
        health_data["checks"]["redis"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        health_data["status"] = "degraded"
    
    # Check ML models
    try:
        from ..analysis.models import model_manager
        models_status = await model_manager.health_check()
        health_data["checks"]["ml_models"] = models_status
    except Exception as e:
        health_data["checks"]["ml_models"] = {
            "status": "unhealthy",
            "error": str(e)
        }
    
    return health_data


@router.get("/ready")
async def readiness_check(
    db: AsyncSession = Depends(get_db_session)
) -> Dict[str, str]:
    """Kubernetes readiness probe"""
    try:
        # Quick database check
        await db.execute("SELECT 1")
        
        # Quick Redis check
        redis = await get_redis()
        await redis.ping()
        
        return {"status": "ready"}
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        raise HTTPException(status_code=503, detail="Service not ready")


@router.get("/live")
async def liveness_check() -> Dict[str, str]:
    """Kubernetes liveness probe"""
    return {"status": "alive"}
