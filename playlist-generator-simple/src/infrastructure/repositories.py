"""
SQLite repository implementations for Playlist Generator.
Implements domain interfaces with SQLite database.
"""

import sqlite3
import json
import os
from typing import List, Optional, Dict, Any
from datetime import datetime
from contextlib import contextmanager

from ..domain.interfaces import ITrackRepository, IAnalysisRepository, IPlaylistRepository
from ..domain.entities import Track, TrackMetadata, AnalysisResult, Playlist
from ..domain.exceptions import RepositoryException


class SQLiteTrackRepository(ITrackRepository):
    """SQLite implementation of track repository."""
    
    def __init__(self, db_path: str = "/app/cache/playlista.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize database tables."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tracks (
                    id TEXT PRIMARY KEY,
                    path TEXT UNIQUE NOT NULL,
                    title TEXT NOT NULL,
                    artist TEXT NOT NULL,
                    album TEXT,
                    duration REAL,
                    year INTEGER,
                    genre TEXT,
                    tags TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            conn.commit()
    
    @contextmanager
    def _get_connection(self):
        """Get database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def save(self, track: Track) -> bool:
        """Save a track to the repository."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO tracks 
                    (id, path, title, artist, album, duration, year, genre, tags, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    track.id,
                    track.path,
                    track.metadata.title,
                    track.metadata.artist,
                    track.metadata.album,
                    track.metadata.duration,
                    track.metadata.year,
                    track.metadata.genre,
                    json.dumps(track.metadata.tags),
                    track.created_at.isoformat(),
                    track.updated_at.isoformat()
                ))
                conn.commit()
                return True
        except Exception as e:
            raise RepositoryException(f"Failed to save track: {e}")
    
    def find_by_id(self, track_id: str) -> Optional[Track]:
        """Find a track by its ID."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM tracks WHERE id = ?", (track_id,))
                row = cursor.fetchone()
                return self._row_to_track(row) if row else None
        except Exception as e:
            raise RepositoryException(f"Failed to find track by ID: {e}")
    
    def find_by_path(self, path: str) -> Optional[Track]:
        """Find a track by its file path."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM tracks WHERE path = ?", (path,))
                row = cursor.fetchone()
                return self._row_to_track(row) if row else None
        except Exception as e:
            raise RepositoryException(f"Failed to find track by path: {e}")
    
    def find_all(self) -> List[Track]:
        """Find all tracks."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM tracks ORDER BY created_at DESC")
                rows = cursor.fetchall()
                return [self._row_to_track(row) for row in rows]
        except Exception as e:
            raise RepositoryException(f"Failed to find all tracks: {e}")
    
    def find_unanalyzed(self) -> List[Track]:
        """Find tracks that haven't been analyzed."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT t.* FROM tracks t
                    LEFT JOIN analysis_results ar ON t.id = ar.track_id
                    WHERE ar.track_id IS NULL
                    ORDER BY t.created_at DESC
                """)
                rows = cursor.fetchall()
                return [self._row_to_track(row) for row in rows]
        except Exception as e:
            raise RepositoryException(f"Failed to find unanalyzed tracks: {e}")
    
    def delete(self, track_id: str) -> bool:
        """Delete a track by ID."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM tracks WHERE id = ?", (track_id,))
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            raise RepositoryException(f"Failed to delete track: {e}")
    
    def count(self) -> int:
        """Get total number of tracks."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM tracks")
                return cursor.fetchone()[0]
        except Exception as e:
            raise RepositoryException(f"Failed to count tracks: {e}")
    
    def _row_to_track(self, row) -> Track:
        """Convert database row to Track entity."""
        metadata = TrackMetadata(
            title=row['title'],
            artist=row['artist'],
            album=row['album'] or "Unknown",
            duration=row['duration'],
            year=row['year'],
            genre=row['genre'],
            tags=json.loads(row['tags']) if row['tags'] else []
        )
        
        track = Track(row['id'], row['path'], metadata)
        track.created_at = datetime.fromisoformat(row['created_at'])
        track.updated_at = datetime.fromisoformat(row['updated_at'])
        
        return track


class SQLiteAnalysisRepository(IAnalysisRepository):
    """SQLite implementation of analysis repository."""
    
    def __init__(self, db_path: str = "/app/cache/playlista.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize database tables."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS analysis_results (
                    track_id TEXT PRIMARY KEY,
                    features TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    analysis_date TEXT NOT NULL,
                    processing_time REAL,
                    FOREIGN KEY (track_id) REFERENCES tracks (id)
                )
            """)
            conn.commit()
    
    @contextmanager
    def _get_connection(self):
        """Get database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def save_analysis(self, track_id: str, result: AnalysisResult) -> bool:
        """Save analysis result for a track."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO analysis_results 
                    (track_id, features, confidence, analysis_date, processing_time)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    track_id,
                    json.dumps(result.features),
                    result.confidence,
                    result.analysis_date.isoformat(),
                    result.processing_time
                ))
                conn.commit()
                return True
        except Exception as e:
            raise RepositoryException(f"Failed to save analysis: {e}")
    
    def get_analysis(self, track_id: str) -> Optional[AnalysisResult]:
        """Get analysis result for a track."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM analysis_results WHERE track_id = ?", (track_id,))
                row = cursor.fetchone()
                return self._row_to_analysis(row) if row else None
        except Exception as e:
            raise RepositoryException(f"Failed to get analysis: {e}")
    
    def delete_analysis(self, track_id: str) -> bool:
        """Delete analysis result for a track."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM analysis_results WHERE track_id = ?", (track_id,))
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            raise RepositoryException(f"Failed to delete analysis: {e}")
    
    def get_analysis_statistics(self) -> Dict[str, Any]:
        """Get analysis statistics."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) as total, AVG(confidence) as avg_confidence FROM analysis_results")
                row = cursor.fetchone()
                
                return {
                    "total_analyses": row['total'] or 0,
                    "average_confidence": row['avg_confidence'] or 0.0
                }
        except Exception as e:
            raise RepositoryException(f"Failed to get analysis statistics: {e}")
    
    def _row_to_analysis(self, row) -> AnalysisResult:
        """Convert database row to AnalysisResult entity."""
        return AnalysisResult(
            features=json.loads(row['features']),
            confidence=row['confidence'],
            analysis_date=datetime.fromisoformat(row['analysis_date']),
            processing_time=row['processing_time']
        )


class SQLitePlaylistRepository(IPlaylistRepository):
    """SQLite implementation of playlist repository."""
    
    def __init__(self, db_path: str = "/app/cache/playlista.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize database tables."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS playlists (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS playlist_tracks (
                    playlist_id TEXT NOT NULL,
                    track_id TEXT NOT NULL,
                    position INTEGER NOT NULL,
                    FOREIGN KEY (playlist_id) REFERENCES playlists (id),
                    FOREIGN KEY (track_id) REFERENCES tracks (id),
                    PRIMARY KEY (playlist_id, track_id)
                )
            """)
            conn.commit()
    
    @contextmanager
    def _get_connection(self):
        """Get database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def save_playlist(self, playlist: Playlist) -> bool:
        """Save a playlist."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Save playlist
                cursor.execute("""
                    INSERT OR REPLACE INTO playlists 
                    (id, name, created_at, updated_at)
                    VALUES (?, ?, ?, ?)
                """, (
                    playlist.id,
                    playlist.name,
                    playlist.created_at.isoformat(),
                    playlist.updated_at.isoformat()
                ))
                
                # Save playlist tracks
                cursor.execute("DELETE FROM playlist_tracks WHERE playlist_id = ?", (playlist.id,))
                
                for position, track in enumerate(playlist.tracks):
                    cursor.execute("""
                        INSERT INTO playlist_tracks (playlist_id, track_id, position)
                        VALUES (?, ?, ?)
                    """, (playlist.id, track.id, position))
                
                conn.commit()
                return True
        except Exception as e:
            raise RepositoryException(f"Failed to save playlist: {e}")
    
    def get_playlist(self, playlist_id: str) -> Optional[Playlist]:
        """Get a playlist by ID."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Get playlist
                cursor.execute("SELECT * FROM playlists WHERE id = ?", (playlist_id,))
                playlist_row = cursor.fetchone()
                if not playlist_row:
                    return None
                
                # Get tracks
                cursor.execute("""
                    SELECT t.* FROM tracks t
                    JOIN playlist_tracks pt ON t.id = pt.track_id
                    WHERE pt.playlist_id = ?
                    ORDER BY pt.position
                """, (playlist_id,))
                track_rows = cursor.fetchall()
                
                tracks = [self._row_to_track(row) for row in track_rows]
                
                playlist = Playlist(
                    playlist_row['id'],
                    playlist_row['name'],
                    tracks
                )
                playlist.created_at = datetime.fromisoformat(playlist_row['created_at'])
                playlist.updated_at = datetime.fromisoformat(playlist_row['updated_at'])
                
                return playlist
        except Exception as e:
            raise RepositoryException(f"Failed to get playlist: {e}")
    
    def get_all_playlists(self) -> List[Playlist]:
        """Get all playlists."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM playlists ORDER BY created_at DESC")
                playlist_rows = cursor.fetchall()
                
                playlists = []
                for playlist_row in playlist_rows:
                    playlist = Playlist(
                        playlist_row['id'],
                        playlist_row['name']
                    )
                    playlist.created_at = datetime.fromisoformat(playlist_row['created_at'])
                    playlist.updated_at = datetime.fromisoformat(playlist_row['updated_at'])
                    playlists.append(playlist)
                
                return playlists
        except Exception as e:
            raise RepositoryException(f"Failed to get all playlists: {e}")
    
    def delete_playlist(self, playlist_id: str) -> bool:
        """Delete a playlist."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM playlist_tracks WHERE playlist_id = ?", (playlist_id,))
                cursor.execute("DELETE FROM playlists WHERE id = ?", (playlist_id,))
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            raise RepositoryException(f"Failed to delete playlist: {e}")
    
    def count(self) -> int:
        """Get total number of playlists."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM playlists")
                return cursor.fetchone()[0]
        except Exception as e:
            raise RepositoryException(f"Failed to count playlists: {e}")
    
    def _row_to_track(self, row) -> Track:
        """Convert database row to Track entity."""
        metadata = TrackMetadata(
            title=row['title'],
            artist=row['artist'],
            album=row['album'] or "Unknown",
            duration=row['duration'],
            year=row['year'],
            genre=row['genre'],
            tags=json.loads(row['tags']) if row['tags'] else []
        )
        
        track = Track(row['id'], row['path'], metadata)
        track.created_at = datetime.fromisoformat(row['created_at'])
        track.updated_at = datetime.fromisoformat(row['updated_at'])
        
        return track 