"""
Database repositories for domain entities.

This module implements the repository pattern for data persistence,
providing a clean abstraction over the database layer.
"""

import sqlite3
import json
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Iterator
from uuid import UUID
from datetime import datetime
from contextlib import contextmanager

from domain.entities import AudioFile, FeatureSet, Metadata, AnalysisResult, Playlist
from domain.repositories import (
    AudioFileRepository,
    FeatureSetRepository,
    MetadataRepository,
    AnalysisResultRepository,
    PlaylistRepository
)
from shared.exceptions import DatabaseError, EntityNotFoundError
from shared.config import get_config


class SQLiteAudioFileRepository(AudioFileRepository):
    """SQLite implementation of AudioFile repository."""
    
    def __init__(self, db_path: Optional[Path] = None):
        """Initialize repository with database path."""
        self.config = get_config()
        self.db_path = db_path or self.config.database.db_path
        self.logger = logging.getLogger(__name__)
        self._ensure_table_exists()
    
    def _ensure_table_exists(self):
        """Ensure the audio_files table exists."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audio_files (
                    id TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    file_name TEXT NOT NULL,
                    file_size_bytes INTEGER NOT NULL,
                    duration_seconds REAL,
                    bitrate_kbps INTEGER,
                    sample_rate_hz INTEGER,
                    channels INTEGER,
                    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    external_metadata TEXT,
                    UNIQUE(file_path)
                )
            """)
    
    @contextmanager
    def _get_connection(self):
        """Get database connection with proper error handling."""
        conn = None
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            yield conn
        except sqlite3.Error as e:
            self.logger.error(f"Database error: {e}")
            raise DatabaseError(f"Database operation failed: {e}")
        finally:
            if conn:
                conn.close()
    
    def save(self, audio_file: AudioFile) -> AudioFile:
        """Save audio file to database."""
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO audio_files 
                    (id, file_path, file_name, file_size_bytes, duration_seconds, 
                     bitrate_kbps, sample_rate_hz, channels, external_metadata, updated_date)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    str(audio_file.id),
                    str(audio_file.file_path),
                    audio_file.file_name,
                    audio_file.file_size_bytes,
                    audio_file.duration_seconds,
                    audio_file.bitrate_kbps,
                    audio_file.sample_rate_hz,
                    audio_file.channels,
                    json.dumps(audio_file.external_metadata),
                    datetime.now().isoformat()
                ))
                conn.commit()
                self.logger.debug(f"Saved audio file: {audio_file.id}")
                return audio_file
        except Exception as e:
            self.logger.error(f"Failed to save audio file {audio_file.id}: {e}")
            raise DatabaseError(f"Failed to save audio file: {e}")
    
    def find_by_id(self, audio_file_id: UUID) -> Optional[AudioFile]:
        """Find audio file by ID."""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("""
                    SELECT * FROM audio_files WHERE id = ?
                """, (str(audio_file_id),))
                row = cursor.fetchone()
                
                if row:
                    return self._row_to_audio_file(row)
                return None
        except Exception as e:
            self.logger.error(f"Failed to find audio file {audio_file_id}: {e}")
            raise DatabaseError(f"Failed to find audio file: {e}")
    
    def find_by_path(self, file_path: Path) -> Optional[AudioFile]:
        """Find audio file by file path."""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("""
                    SELECT * FROM audio_files WHERE file_path = ?
                """, (str(file_path),))
                row = cursor.fetchone()
                
                if row:
                    return self._row_to_audio_file(row)
                return None
        except Exception as e:
            self.logger.error(f"Failed to find audio file by path {file_path}: {e}")
            raise DatabaseError(f"Failed to find audio file by path: {e}")
    
    def find_all(self, limit: Optional[int] = None, offset: Optional[int] = None) -> List[AudioFile]:
        """Find all audio files with optional pagination."""
        try:
            with self._get_connection() as conn:
                query = "SELECT * FROM audio_files ORDER BY created_date DESC"
                params = []
                
                if limit is not None:
                    query += " LIMIT ?"
                    params.append(limit)
                
                if offset is not None:
                    query += " OFFSET ?"
                    params.append(offset)
                
                cursor = conn.execute(query, params)
                rows = cursor.fetchall()
                
                return [self._row_to_audio_file(row) for row in rows]
        except Exception as e:
            self.logger.error(f"Failed to find all audio files: {e}")
            raise DatabaseError(f"Failed to find all audio files: {e}")
    
    def delete(self, audio_file_id: UUID) -> bool:
        """Delete audio file by ID."""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("""
                    DELETE FROM audio_files WHERE id = ?
                """, (str(audio_file_id),))
                conn.commit()
                
                deleted = cursor.rowcount > 0
                if deleted:
                    self.logger.debug(f"Deleted audio file: {audio_file_id}")
                return deleted
        except Exception as e:
            self.logger.error(f"Failed to delete audio file {audio_file_id}: {e}")
            raise DatabaseError(f"Failed to delete audio file: {e}")
    
    def count(self) -> int:
        """Get total number of audio files."""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM audio_files")
                return cursor.fetchone()[0]
        except Exception as e:
            self.logger.error(f"Failed to count audio files: {e}")
            raise DatabaseError(f"Failed to count audio files: {e}")
    
    def _row_to_audio_file(self, row) -> AudioFile:
        """Convert database row to AudioFile entity."""
        external_metadata = json.loads(row['external_metadata']) if row['external_metadata'] else {}
        
        return AudioFile(
            id=UUID(row['id']),
            file_path=Path(row['file_path']),
            file_size_bytes=row['file_size_bytes'],
            duration_seconds=row['duration_seconds'],
            bitrate_kbps=row['bitrate_kbps'],
            sample_rate_hz=row['sample_rate_hz'],
            channels=row['channels'],
            external_metadata=external_metadata
        )


class SQLiteFeatureSetRepository(FeatureSetRepository):
    """SQLite implementation of FeatureSet repository."""
    
    def __init__(self, db_path: Optional[Path] = None):
        """Initialize repository with database path."""
        self.config = get_config()
        self.db_path = db_path or self.config.database.db_path
        self.logger = logging.getLogger(__name__)
        self._ensure_table_exists()
    
    def _ensure_table_exists(self):
        """Ensure the feature_sets table exists."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feature_sets (
                    id TEXT PRIMARY KEY,
                    audio_file_id TEXT NOT NULL,
                    bpm REAL,
                    energy REAL,
                    danceability REAL,
                    valence REAL,
                    acousticness REAL,
                    instrumentalness REAL,
                    speechiness REAL,
                    liveness REAL,
                    loudness REAL,
                    key TEXT,
                    mode TEXT,
                    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (audio_file_id) REFERENCES audio_files (id) ON DELETE CASCADE
                )
            """)
    
    @contextmanager
    def _get_connection(self):
        """Get database connection with proper error handling."""
        conn = None
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            yield conn
        except sqlite3.Error as e:
            self.logger.error(f"Database error: {e}")
            raise DatabaseError(f"Database operation failed: {e}")
        finally:
            if conn:
                conn.close()
    
    def save(self, feature_set: FeatureSet) -> FeatureSet:
        """Save feature set to database."""
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO feature_sets 
                    (id, audio_file_id, bpm, energy, danceability, valence, acousticness,
                     instrumentalness, speechiness, liveness, loudness, key, mode, updated_date)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    str(feature_set.id),
                    str(feature_set.audio_file_id),
                    feature_set.bpm,
                    feature_set.energy,
                    feature_set.danceability,
                    feature_set.valence,
                    feature_set.acousticness,
                    feature_set.instrumentalness,
                    feature_set.speechiness,
                    feature_set.liveness,
                    feature_set.loudness,
                    feature_set.key,
                    feature_set.mode,
                    datetime.now().isoformat()
                ))
                conn.commit()
                self.logger.debug(f"Saved feature set: {feature_set.id}")
                return feature_set
        except Exception as e:
            self.logger.error(f"Failed to save feature set {feature_set.id}: {e}")
            raise DatabaseError(f"Failed to save feature set: {e}")
    
    def find_by_id(self, feature_set_id: UUID) -> Optional[FeatureSet]:
        """Find feature set by ID."""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("""
                    SELECT * FROM feature_sets WHERE id = ?
                """, (str(feature_set_id),))
                row = cursor.fetchone()
                
                if row:
                    return self._row_to_feature_set(row)
                return None
        except Exception as e:
            self.logger.error(f"Failed to find feature set {feature_set_id}: {e}")
            raise DatabaseError(f"Failed to find feature set: {e}")
    
    def find_by_audio_file_id(self, audio_file_id: UUID) -> Optional[FeatureSet]:
        """Find feature set by audio file ID."""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("""
                    SELECT * FROM feature_sets WHERE audio_file_id = ?
                """, (str(audio_file_id),))
                row = cursor.fetchone()
                
                if row:
                    return self._row_to_feature_set(row)
                return None
        except Exception as e:
            self.logger.error(f"Failed to find feature set for audio file {audio_file_id}: {e}")
            raise DatabaseError(f"Failed to find feature set for audio file: {e}")
    
    def delete(self, feature_set_id: UUID) -> bool:
        """Delete feature set by ID."""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("""
                    DELETE FROM feature_sets WHERE id = ?
                """, (str(feature_set_id),))
                conn.commit()
                
                deleted = cursor.rowcount > 0
                if deleted:
                    self.logger.debug(f"Deleted feature set: {feature_set_id}")
                return deleted
        except Exception as e:
            self.logger.error(f"Failed to delete feature set {feature_set_id}: {e}")
            raise DatabaseError(f"Failed to delete feature set: {e}")
    
    def _row_to_feature_set(self, row) -> FeatureSet:
        """Convert database row to FeatureSet entity."""
        return FeatureSet(
            id=UUID(row['id']),
            audio_file_id=UUID(row['audio_file_id']),
            bpm=row['bpm'],
            energy=row['energy'],
            danceability=row['danceability'],
            valence=row['valence'],
            acousticness=row['acousticness'],
            instrumentalness=row['instrumentalness'],
            speechiness=row['speechiness'],
            liveness=row['liveness'],
            loudness=row['loudness'],
            key=row['key'],
            mode=row['mode']
        )


class SQLiteMetadataRepository(MetadataRepository):
    """SQLite implementation of Metadata repository."""
    
    def __init__(self, db_path: Optional[Path] = None):
        """Initialize repository with database path."""
        self.config = get_config()
        self.db_path = db_path or self.config.database.db_path
        self.logger = logging.getLogger(__name__)
        self._ensure_table_exists()
    
    def _ensure_table_exists(self):
        """Ensure the metadata table exists."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metadata (
                    id TEXT PRIMARY KEY,
                    audio_file_id TEXT NOT NULL,
                    title TEXT,
                    artist TEXT,
                    album TEXT,
                    year INTEGER,
                    genre TEXT,
                    track_number INTEGER,
                    disc_number INTEGER,
                    composer TEXT,
                    lyrics TEXT,
                    comment TEXT,
                    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (audio_file_id) REFERENCES audio_files (id) ON DELETE CASCADE
                )
            """)
    
    @contextmanager
    def _get_connection(self):
        """Get database connection with proper error handling."""
        conn = None
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            yield conn
        except sqlite3.Error as e:
            self.logger.error(f"Database error: {e}")
            raise DatabaseError(f"Database operation failed: {e}")
        finally:
            if conn:
                conn.close()
    
    def save(self, metadata: Metadata) -> Metadata:
        """Save metadata to database."""
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO metadata 
                    (id, audio_file_id, title, artist, album, year, genre, track_number,
                     disc_number, composer, lyrics, comment, updated_date)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    str(metadata.id),
                    str(metadata.audio_file_id),
                    metadata.title,
                    metadata.artist,
                    metadata.album,
                    metadata.year,
                    metadata.genre,
                    metadata.track_number,
                    metadata.disc_number,
                    metadata.composer,
                    metadata.lyrics,
                    metadata.comment,
                    datetime.now().isoformat()
                ))
                conn.commit()
                self.logger.debug(f"Saved metadata: {metadata.id}")
                return metadata
        except Exception as e:
            self.logger.error(f"Failed to save metadata {metadata.id}: {e}")
            raise DatabaseError(f"Failed to save metadata: {e}")
    
    def find_by_id(self, metadata_id: UUID) -> Optional[Metadata]:
        """Find metadata by ID."""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("""
                    SELECT * FROM metadata WHERE id = ?
                """, (str(metadata_id),))
                row = cursor.fetchone()
                
                if row:
                    return self._row_to_metadata(row)
                return None
        except Exception as e:
            self.logger.error(f"Failed to find metadata {metadata_id}: {e}")
            raise DatabaseError(f"Failed to find metadata: {e}")
    
    def find_by_audio_file_id(self, audio_file_id: UUID) -> Optional[Metadata]:
        """Find metadata by audio file ID."""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("""
                    SELECT * FROM metadata WHERE audio_file_id = ?
                """, (str(audio_file_id),))
                row = cursor.fetchone()
                
                if row:
                    return self._row_to_metadata(row)
                return None
        except Exception as e:
            self.logger.error(f"Failed to find metadata for audio file {audio_file_id}: {e}")
            raise DatabaseError(f"Failed to find metadata for audio file: {e}")
    
    def delete(self, metadata_id: UUID) -> bool:
        """Delete metadata by ID."""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("""
                    DELETE FROM metadata WHERE id = ?
                """, (str(metadata_id),))
                conn.commit()
                
                deleted = cursor.rowcount > 0
                if deleted:
                    self.logger.debug(f"Deleted metadata: {metadata_id}")
                return deleted
        except Exception as e:
            self.logger.error(f"Failed to delete metadata {metadata_id}: {e}")
            raise DatabaseError(f"Failed to delete metadata: {e}")
    
    def _row_to_metadata(self, row) -> Metadata:
        """Convert database row to Metadata entity."""
        return Metadata(
            id=UUID(row['id']),
            audio_file_id=UUID(row['audio_file_id']),
            title=row['title'],
            artist=row['artist'],
            album=row['album'],
            year=row['year'],
            genre=row['genre'],
            track_number=row['track_number'],
            disc_number=row['disc_number'],
            composer=row['composer'],
            lyrics=row['lyrics'],
            comment=row['comment']
        )


class SQLitePlaylistRepository(PlaylistRepository):
    """SQLite implementation of Playlist repository."""
    
    def __init__(self, db_path: Optional[Path] = None):
        """Initialize repository with database path."""
        self.config = get_config()
        self.db_path = db_path or self.config.database.playlist_db_path
        self.logger = logging.getLogger(__name__)
        self._ensure_table_exists()
    
    def _ensure_table_exists(self):
        """Ensure the playlists table exists."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS playlists (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    track_ids TEXT NOT NULL,
                    track_paths TEXT NOT NULL,
                    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
    
    @contextmanager
    def _get_connection(self):
        """Get database connection with proper error handling."""
        conn = None
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            yield conn
        except sqlite3.Error as e:
            self.logger.error(f"Database error: {e}")
            raise DatabaseError(f"Database operation failed: {e}")
        finally:
            if conn:
                conn.close()
    
    def save(self, playlist: Playlist) -> Playlist:
        """Save playlist to database."""
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO playlists 
                    (id, name, description, track_ids, track_paths, updated_date)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    str(playlist.id),
                    playlist.name,
                    playlist.description,
                    json.dumps([str(track_id) for track_id in playlist.track_ids]),
                    json.dumps(playlist.track_paths),
                    datetime.now().isoformat()
                ))
                conn.commit()
                self.logger.debug(f"Saved playlist: {playlist.id}")
                return playlist
        except Exception as e:
            self.logger.error(f"Failed to save playlist {playlist.id}: {e}")
            raise DatabaseError(f"Failed to save playlist: {e}")
    
    def find_by_id(self, playlist_id: UUID) -> Optional[Playlist]:
        """Find playlist by ID."""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("""
                    SELECT * FROM playlists WHERE id = ?
                """, (str(playlist_id),))
                row = cursor.fetchone()
                
                if row:
                    return self._row_to_playlist(row)
                return None
        except Exception as e:
            self.logger.error(f"Failed to find playlist {playlist_id}: {e}")
            raise DatabaseError(f"Failed to find playlist: {e}")
    
    def find_all(self, limit: Optional[int] = None, offset: Optional[int] = None) -> List[Playlist]:
        """Find all playlists with optional pagination."""
        try:
            with self._get_connection() as conn:
                query = "SELECT * FROM playlists ORDER BY created_date DESC"
                params = []
                
                if limit is not None:
                    query += " LIMIT ?"
                    params.append(limit)
                
                if offset is not None:
                    query += " OFFSET ?"
                    params.append(offset)
                
                cursor = conn.execute(query, params)
                rows = cursor.fetchall()
                
                return [self._row_to_playlist(row) for row in rows]
        except Exception as e:
            self.logger.error(f"Failed to find all playlists: {e}")
            raise DatabaseError(f"Failed to find all playlists: {e}")
    
    def delete(self, playlist_id: UUID) -> bool:
        """Delete playlist by ID."""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("""
                    DELETE FROM playlists WHERE id = ?
                """, (str(playlist_id),))
                conn.commit()
                
                deleted = cursor.rowcount > 0
                if deleted:
                    self.logger.debug(f"Deleted playlist: {playlist_id}")
                return deleted
        except Exception as e:
            self.logger.error(f"Failed to delete playlist {playlist_id}: {e}")
            raise DatabaseError(f"Failed to delete playlist: {e}")
    
    def count(self) -> int:
        """Get total number of playlists."""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM playlists")
                return cursor.fetchone()[0]
        except Exception as e:
            self.logger.error(f"Failed to count playlists: {e}")
            raise DatabaseError(f"Failed to count playlists: {e}")
    
    def _row_to_playlist(self, row) -> Playlist:
        """Convert database row to Playlist entity."""
        track_ids = [UUID(track_id) for track_id in json.loads(row['track_ids'])]
        track_paths = json.loads(row['track_paths'])
        
        return Playlist(
            id=UUID(row['id']),
            name=row['name'],
            description=row['description'],
            track_ids=track_ids,
            track_paths=track_paths
        ) 