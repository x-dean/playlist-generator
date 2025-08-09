"""
SQLAlchemy models for Playlista v2
Optimized for performance with proper indexing
"""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

Base = declarative_base()


class Track(Base):
    """Enhanced track model with comprehensive audio features"""
    
    __tablename__ = "tracks"
    
    # Primary identification
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    file_path = Column(String(1000), unique=True, nullable=False, index=True)
    file_hash = Column(String(64), nullable=False, index=True)
    filename = Column(String(500), nullable=False)
    file_size_bytes = Column(Integer, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    analyzed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Status tracking
    status = Column(String(20), default="discovered", index=True)  # discovered, analyzing, analyzed, failed
    analysis_version = Column(String(10), default="2.0")
    error_message = Column(Text, nullable=True)
    retry_count = Column(Integer, default=0)
    
    # Basic metadata
    title = Column(String(500), nullable=True, index=True)
    artist = Column(String(500), nullable=True, index=True)
    album = Column(String(500), nullable=True)
    album_artist = Column(String(500), nullable=True)
    track_number = Column(Integer, nullable=True)
    disc_number = Column(Integer, nullable=True)
    year = Column(Integer, nullable=True, index=True)
    genre = Column(String(100), nullable=True, index=True)
    duration = Column(Float, nullable=True)
    
    # Audio properties
    sample_rate = Column(Integer, nullable=True)
    bitrate = Column(Integer, nullable=True)
    channels = Column(Integer, nullable=True)
    format = Column(String(20), nullable=True)
    
    # Essential audio features (indexed for fast querying)
    bpm = Column(Float, nullable=True, index=True)
    key = Column(String(10), nullable=True, index=True)
    mode = Column(String(10), nullable=True)  # major, minor
    loudness = Column(Float, nullable=True)
    energy = Column(Float, nullable=True, index=True)
    danceability = Column(Float, nullable=True, index=True)
    valence = Column(Float, nullable=True, index=True)  # mood
    acousticness = Column(Float, nullable=True)
    instrumentalness = Column(Float, nullable=True)
    speechiness = Column(Float, nullable=True)
    liveness = Column(Float, nullable=True)
    
    # Advanced audio features
    spectral_centroid = Column(Float, nullable=True)
    spectral_bandwidth = Column(Float, nullable=True)
    spectral_rolloff = Column(Float, nullable=True)
    spectral_flatness = Column(Float, nullable=True)
    zero_crossing_rate = Column(Float, nullable=True)
    
    # ML-derived features
    mood_angry = Column(Float, nullable=True)
    mood_happy = Column(Float, nullable=True)
    mood_relaxed = Column(Float, nullable=True)
    mood_sad = Column(Float, nullable=True)
    
    # Complex feature data (JSON)
    mfcc_features = Column(JSON, nullable=True)
    chroma_features = Column(JSON, nullable=True)
    spectral_contrast = Column(JSON, nullable=True)
    tonnetz_features = Column(JSON, nullable=True)
    ml_embeddings = Column(JSON, nullable=True)  # Neural network embeddings
    
    # External metadata
    musicbrainz_id = Column(String(36), nullable=True, index=True)
    spotify_id = Column(String(22), nullable=True, index=True)
    lastfm_data = Column(JSON, nullable=True)
    external_tags = Column(JSON, nullable=True)
    
    # Relationships
    playlist_items = relationship("PlaylistItem", back_populates="track", cascade="all, delete-orphan")
    
    # Composite indexes for common queries
    __table_args__ = (
        Index("idx_artist_album", "artist", "album"),
        Index("idx_genre_year", "genre", "year"),
        Index("idx_bpm_energy", "bpm", "energy"),
        Index("idx_key_mode", "key", "mode"),
        Index("idx_mood_valence", "valence", "energy", "danceability"),
        Index("idx_status_analyzed", "status", "analyzed_at"),
    )


class Playlist(Base):
    """Playlist model with generation metadata"""
    
    __tablename__ = "playlists"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(200), nullable=False, index=True)
    description = Column(Text, nullable=True)
    
    # Generation metadata
    generation_method = Column(String(50), nullable=False, default="manual")
    generation_params = Column(JSON, nullable=True)
    generation_time = Column(Float, nullable=True)  # Time taken to generate
    
    # Statistics
    track_count = Column(Integer, default=0)
    total_duration = Column(Float, nullable=True)
    avg_bpm = Column(Float, nullable=True)
    avg_energy = Column(Float, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    tracks = relationship("PlaylistItem", back_populates="playlist", cascade="all, delete-orphan")


class PlaylistItem(Base):
    """Junction table for playlist tracks with ordering"""
    
    __tablename__ = "playlist_tracks"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    playlist_id = Column(UUID(as_uuid=True), ForeignKey("playlists.id"), nullable=False)
    track_id = Column(UUID(as_uuid=True), ForeignKey("tracks.id"), nullable=False)
    position = Column(Integer, nullable=False)
    
    # Transition metadata
    transition_score = Column(Float, nullable=True)  # How well this track follows the previous
    harmonic_compatibility = Column(Float, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    playlist = relationship("Playlist", back_populates="tracks")
    track = relationship("Track", back_populates="playlist_items")
    
    # Ensure unique position per playlist
    __table_args__ = (
        UniqueConstraint("playlist_id", "position", name="uix_playlist_position"),
        Index("idx_playlist_position", "playlist_id", "position"),
    )


class AnalysisJob(Base):
    """Track analysis job queue"""
    
    __tablename__ = "analysis_jobs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    track_id = Column(UUID(as_uuid=True), ForeignKey("tracks.id"), nullable=False)
    
    # Job metadata
    status = Column(String(20), default="queued", index=True)  # queued, processing, completed, failed
    priority = Column(Integer, default=0, index=True)  # Higher number = higher priority
    worker_id = Column(String(50), nullable=True)
    
    # Progress tracking
    progress_percent = Column(Integer, default=0)
    current_step = Column(String(100), nullable=True)
    
    # Performance metrics
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    processing_time = Column(Float, nullable=True)
    
    # Error handling
    error_message = Column(Text, nullable=True)
    retry_count = Column(Integer, default=0)
    max_retries = Column(Integer, default=3)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Indexes for job processing
    __table_args__ = (
        Index("idx_status_priority", "status", "priority"),
        Index("idx_worker_status", "worker_id", "status"),
    )


class CacheEntry(Base):
    """Generic cache table for features and computed results"""
    
    __tablename__ = "cache_entries"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    cache_key = Column(String(255), unique=True, nullable=False, index=True)
    cache_type = Column(String(50), nullable=False, index=True)  # features, similarity, etc.
    
    # Data
    data = Column(JSON, nullable=False)
    data_size = Column(Integer, nullable=True)  # Size in bytes
    
    # Metadata
    expires_at = Column(DateTime(timezone=True), nullable=True, index=True)
    access_count = Column(Integer, default=0)
    last_accessed = Column(DateTime(timezone=True), server_default=func.now())
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    __table_args__ = (
        Index("idx_type_expires", "cache_type", "expires_at"),
    )
