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
            # Check if table exists and has the old schema
            cursor = conn.execute("PRAGMA table_info(audio_files)")
            columns = {row[1]: row[2] for row in cursor.fetchall()}
            
            if not columns:
                # Table doesn't exist, create it
                conn.execute("""
                    CREATE TABLE audio_files (
                        id TEXT PRIMARY KEY,
                        file_path TEXT NOT NULL,
                        file_name TEXT NOT NULL,
                        file_size_bytes INTEGER,
                        file_hash TEXT,
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
            else:
                # Table exists, check if it needs migration
                needs_migration = False
                migration_reason = []
                
                # Check for NOT NULL constraint on file_size_bytes
                if 'file_size_bytes' in columns and 'NOT NULL' in columns['file_size_bytes']:
                    needs_migration = True
                    migration_reason.append("file_size_bytes NOT NULL constraint")
                
                # Check for missing file_hash column
                if 'file_hash' not in columns:
                    needs_migration = True
                    migration_reason.append("missing file_hash column")
                
                if needs_migration:
                    self.logger.info(f"Migrating audio_files table: {', '.join(migration_reason)}")
                    
                    # Create temporary table with new schema
                    conn.execute("""
                        CREATE TABLE audio_files_new (
                            id TEXT PRIMARY KEY,
                            file_path TEXT NOT NULL,
                            file_name TEXT NOT NULL,
                            file_size_bytes INTEGER,
                            file_hash TEXT,
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
                    
                    # Copy data from old table to new table
                    if 'file_size_bytes' in columns:
                        # Handle file_size_bytes migration
                        conn.execute("""
                            INSERT INTO audio_files_new 
                            SELECT id, file_path, file_name, 
                                   CASE WHEN file_size_bytes = 0 THEN NULL ELSE file_size_bytes END,
                                   NULL, duration_seconds, bitrate_kbps, sample_rate_hz, channels,
                                   created_date, updated_date, external_metadata
                            FROM audio_files
                        """)
                    else:
                        # No file_size_bytes column, add it as NULL
                        conn.execute("""
                            INSERT INTO audio_files_new 
                            SELECT id, file_path, file_name, 
                                   NULL, NULL, duration_seconds, bitrate_kbps, sample_rate_hz, channels,
                                   created_date, updated_date, external_metadata
                            FROM audio_files
                        """)
                    
                    # Drop old table and rename new table
                    conn.execute("DROP TABLE audio_files")
                    conn.execute("ALTER TABLE audio_files_new RENAME TO audio_files")
    
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
                    (id, file_path, file_name, file_size_bytes, file_hash, duration_seconds, 
                     bitrate_kbps, sample_rate_hz, channels, external_metadata, updated_date)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    str(audio_file.id),
                    str(audio_file.file_path),
                    audio_file.file_name,
                    audio_file.file_size_bytes if audio_file.file_size_bytes else None,
                    audio_file.file_hash,
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
    
    def find_by_hash(self, file_hash: str) -> Optional[AudioFile]:
        """Find audio file by file hash."""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("""
                    SELECT * FROM audio_files WHERE file_hash = ?
                """, (file_hash,))
                row = cursor.fetchone()
                
                if row:
                    return self._row_to_audio_file(row)
                return None
        except Exception as e:
            self.logger.error(f"Failed to find audio file by hash {file_hash[:8]}...: {e}")
            raise DatabaseError(f"Failed to find audio file by hash: {e}")
    
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
            file_hash=row['file_hash'],
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
    
    def remove_audio_file_references(self, audio_file_id: UUID) -> int:
        """Remove references to an audio file from all playlists."""
        try:
            with self._get_connection() as conn:
                # Get all playlists that reference this audio file
                cursor = conn.execute("""
                    SELECT id, track_ids FROM playlists 
                    WHERE track_ids LIKE ?
                """, (f'%{str(audio_file_id)}%',))
                
                updated_count = 0
                
                for row in cursor.fetchall():
                    playlist_id = row['id']
                    track_ids = json.loads(row['track_ids']) if row['track_ids'] else []
                    
                    # Remove the audio file ID from track_ids
                    if str(audio_file_id) in track_ids:
                        track_ids.remove(str(audio_file_id))
                        
                        # Update the playlist
                        conn.execute("""
                            UPDATE playlists 
                            SET track_ids = ?, updated_date = ?
                            WHERE id = ?
                        """, (
                            json.dumps(track_ids),
                            datetime.now().isoformat(),
                            playlist_id
                        ))
                        updated_count += 1
                        self.logger.debug(f"Removed audio file {audio_file_id} from playlist {playlist_id}")
                
                conn.commit()
                return updated_count
                
        except Exception as e:
            self.logger.error(f"Failed to remove audio file references {audio_file_id}: {e}")
            raise DatabaseError(f"Failed to remove audio file references: {e}")
    
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


class SQLiteAnalysisResultRepository(AnalysisResultRepository):
    """SQLite implementation of AnalysisResult repository."""
    
    def __init__(self, db_path: Optional[Path] = None):
        """Initialize repository with database path."""
        self.config = get_config()
        self.db_path = db_path or self.config.database.db_path
        self.logger = logging.getLogger(__name__)
        self._ensure_table_exists()
    
    def _ensure_table_exists(self):
        """Ensure the analysis_results table exists."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS analysis_results (
                    id TEXT PRIMARY KEY,
                    audio_file_id TEXT NOT NULL,
                    feature_set_id TEXT,
                    metadata_id TEXT,
                    quality_score REAL,
                    is_successful BOOLEAN NOT NULL,
                    is_complete BOOLEAN NOT NULL,
                    processing_time_ms REAL,
                    error_message TEXT,
                    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (audio_file_id) REFERENCES audio_files (id),
                    FOREIGN KEY (feature_set_id) REFERENCES feature_sets (id),
                    FOREIGN KEY (metadata_id) REFERENCES metadata (id)
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
    
    def save(self, analysis_result: AnalysisResult) -> AnalysisResult:
        """Save analysis result to database."""
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO analysis_results 
                    (id, audio_file_id, feature_set_id, metadata_id, quality_score,
                     is_successful, is_complete, processing_time_ms, error_message, updated_date)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    str(analysis_result.id),
                    str(analysis_result.audio_file.id),
                    str(analysis_result.feature_set.id) if analysis_result.feature_set else None,
                    str(analysis_result.metadata.id) if analysis_result.metadata else None,
                    analysis_result.quality_score,
                    analysis_result.is_successful,
                    analysis_result.is_complete,
                    analysis_result.processing_time_ms,
                    analysis_result.error_message,
                    datetime.now().isoformat()
                ))
                conn.commit()
                self.logger.debug(f"Saved analysis result: {analysis_result.id}")
                return analysis_result
        except Exception as e:
            self.logger.error(f"Failed to save analysis result {analysis_result.id}: {e}")
            raise DatabaseError(f"Failed to save analysis result: {e}")
    
    def find_by_id(self, analysis_result_id: UUID) -> Optional[AnalysisResult]:
        """Find analysis result by ID."""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("""
                    SELECT * FROM analysis_results WHERE id = ?
                """, (str(analysis_result_id),))
                row = cursor.fetchone()
                
                if row:
                    return self._row_to_analysis_result(row)
                return None
        except Exception as e:
            self.logger.error(f"Failed to find analysis result {analysis_result_id}: {e}")
            raise DatabaseError(f"Failed to find analysis result: {e}")
    
    def find_by_audio_file_id(self, audio_file_id: UUID) -> Optional[AnalysisResult]:
        """Find analysis result by audio file ID."""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("""
                    SELECT * FROM analysis_results WHERE audio_file_id = ?
                """, (str(audio_file_id),))
                row = cursor.fetchone()
                
                if row:
                    return self._row_to_analysis_result(row)
                return None
        except Exception as e:
            self.logger.error(f"Failed to find analysis result by audio file {audio_file_id}: {e}")
            raise DatabaseError(f"Failed to find analysis result by audio file: {e}")
    
    def delete_by_audio_file_id(self, audio_file_id: UUID) -> bool:
        """Delete all analysis results for an audio file ID."""
        try:
            with self._get_connection() as conn:
                # First, get the analysis result to find related feature_set_id and metadata_id
                cursor = conn.execute("""
                    SELECT feature_set_id, metadata_id FROM analysis_results WHERE audio_file_id = ?
                """, (str(audio_file_id),))
                rows = cursor.fetchall()
                
                deleted_count = 0
                
                for row in rows:
                    feature_set_id = row['feature_set_id']
                    metadata_id = row['metadata_id']
                    
                    # Delete related feature set if it exists
                    if feature_set_id:
                        try:
                            conn.execute("DELETE FROM feature_sets WHERE id = ?", (feature_set_id,))
                        except Exception as e:
                            self.logger.warning(f"Failed to delete feature set {feature_set_id}: {e}")
                    
                    # Delete related metadata if it exists
                    if metadata_id:
                        try:
                            conn.execute("DELETE FROM metadata WHERE id = ?", (metadata_id,))
                        except Exception as e:
                            self.logger.warning(f"Failed to delete metadata {metadata_id}: {e}")
                
                # Delete all analysis results for this audio file
                cursor = conn.execute("""
                    DELETE FROM analysis_results WHERE audio_file_id = ?
                """, (str(audio_file_id),))
                
                deleted_count = cursor.rowcount
                conn.commit()
                
                if deleted_count > 0:
                    self.logger.debug(f"Deleted {deleted_count} analysis results for audio file: {audio_file_id}")
                
                return deleted_count > 0
                
        except Exception as e:
            self.logger.error(f"Failed to delete analysis results for audio file {audio_file_id}: {e}")
            raise DatabaseError(f"Failed to delete analysis results for audio file: {e}")
    
    def delete(self, analysis_result_id: UUID) -> bool:
        """Delete analysis result by ID."""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("""
                    DELETE FROM analysis_results WHERE id = ?
                """, (str(analysis_result_id),))
                conn.commit()
                
                deleted = cursor.rowcount > 0
                if deleted:
                    self.logger.debug(f"Deleted analysis result: {analysis_result_id}")
                return deleted
        except Exception as e:
            self.logger.error(f"Failed to delete analysis result {analysis_result_id}: {e}")
            raise DatabaseError(f"Failed to delete analysis result: {e}")
    
    def _row_to_analysis_result(self, row) -> AnalysisResult:
        """Convert database row to AnalysisResult entity."""
        # This is a simplified conversion - in a real implementation,
        # you'd need to load the related entities (AudioFile, FeatureSet, Metadata)
        from domain.entities.audio_file import AudioFile
        
        audio_file = AudioFile(file_path=Path(row['audio_file_id']))  # Simplified
        
        return AnalysisResult(
            id=UUID(row['id']),
            audio_file=audio_file,
            feature_set=None,  # Would need to load from feature_sets table
            metadata=None,      # Would need to load from metadata table
            quality_score=row['quality_score'],
            is_successful=bool(row['is_successful']),
            is_complete=bool(row['is_complete']),
            processing_time_ms=row['processing_time_ms'],
            error_message=row['error_message']
        ) 