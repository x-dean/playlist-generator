"""
Playlista v2 - FastAPI Main Application
High-performance music analysis and playlist generation system
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from .api import analysis, library, playlists, health
from .core.config import get_settings
from .core.logging import setup_logging
from .database.connection import init_db, close_db
from .utils.websocket_manager import WebSocketManager

# Initialize settings
settings = get_settings()

# Setup logging
setup_logging(settings.log_level)
logger = logging.getLogger("playlista.main")

# WebSocket manager for real-time updates
websocket_manager = WebSocketManager()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager for startup/shutdown tasks"""
    
    # Startup
    logger.info("Starting Playlista v2...")
    
    # Initialize database connections
    await init_db()
    logger.info("Database connections initialized")
    
    # Initialize ML models
    from .analysis.models import load_models
    await load_models()
    logger.info("ML models loaded")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Playlista v2...")
    
    # Close database connections
    await close_db()
    logger.info("Database connections closed")


# Create FastAPI application
app = FastAPI(
    title="Playlista v2 API",
    description="High-performance music analysis and playlist generation system",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan
)

# Add middleware with development-friendly CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Include API routers
app.include_router(health.router, prefix="/api/health", tags=["Health"])
app.include_router(library.router, prefix="/api/library", tags=["Library"])
app.include_router(analysis.router, prefix="/api/analysis", tags=["Analysis"])
app.include_router(playlists.router, prefix="/api/playlists", tags=["Playlists"])


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await websocket_manager.connect(websocket)
    try:
        while True:
            # Keep connection alive and handle incoming messages
            data = await websocket.receive_text()
            logger.debug(f"Received WebSocket message: {data}")
            
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)
        logger.info("WebSocket client disconnected")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "Playlista v2 API",
        "version": "2.0.0",
        "status": "running",
        "docs": "/api/docs"
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        access_log=True
    )
