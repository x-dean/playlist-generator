"""
FastAPI application entry point for Playlist Generator.
Main application with OpenAPI documentation and middleware.
"""

# Suppress TensorFlow warnings globally
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all TensorFlow warnings

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

# Use lazy imports for heavy libraries to improve startup time
from .core.lazy_imports import get_tensorflow

from .api.routes import router
from .infrastructure.container import configure_container
from .infrastructure.logging import setup_logging, get_logger
from .infrastructure.monitoring import get_metrics_collector


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger = get_logger()
    logger.info("Starting Playlist Generator API")
    
    # Configure container
    container = configure_container()
    app.state.container = container
    
    # Setup logging
    setup_logging()
    
    # Update metrics
    metrics = get_metrics_collector()
    metrics.update_system_metrics()
    
    logger.info("Playlist Generator API started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Playlist Generator API")


# Create FastAPI application
app = FastAPI(
    title="Playlist Generator API",
    description="REST API for audio track analysis and playlist generation",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger = get_logger()
    logger.exception(f"Unhandled exception: {exc}")
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred",
            "timestamp": "2024-01-01T00:00:00Z"
        }
    )


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Playlist Generator API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health"
    }


@app.get("/openapi.json")
async def get_openapi():
    """Get OpenAPI specification."""
    return app.openapi()


if __name__ == "__main__":
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 