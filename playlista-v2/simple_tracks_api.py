#!/usr/bin/env python3
"""
Simple working tracks API endpoint
"""

import os
import sys
sys.path.append('/app')

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import asyncpg
import json
from typing import List, Dict, Any

app = FastAPI(title="Simple Tracks API")

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def get_db_connection():
    """Get database connection"""
    return await asyncpg.connect(
        host='postgres',
        port=5432,
        user='playlista',
        password='playlista',
        database='playlista_v2'
    )

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "version": "2.0.0"}

@app.get("/api/library/stats")
async def get_library_stats():
    """Get library statistics"""
    conn = await get_db_connection()
    try:
        total_tracks = await conn.fetchval("SELECT COUNT(*) FROM tracks")
        analyzed_tracks = await conn.fetchval("SELECT COUNT(*) FROM tracks WHERE audio_features IS NOT NULL")
        total_playlists = await conn.fetchval("SELECT COUNT(*) FROM playlists")
        total_duration = await conn.fetchval("SELECT SUM((audio_features->>'duration')::float) FROM tracks WHERE audio_features->>'duration' IS NOT NULL") or 0
        
        return {
            "total_tracks": total_tracks or 0,
            "analyzed_tracks": analyzed_tracks or 0,
            "total_playlists": total_playlists or 0,
            "total_duration": float(total_duration)
        }
    finally:
        await conn.close()

@app.get("/api/library/tracks")
async def get_tracks(
    page: int = 1,
    per_page: int = 20,
    search: str = None,
    sort_by: str = "created_at",
    sort_order: str = "desc"
):
    """Get tracks with pagination"""
    conn = await get_db_connection()
    try:
        # Build query
        where_clause = ""
        params = []
        
        if search:
            where_clause = "WHERE (title ILIKE $1 OR filename ILIKE $1)"
            params.append(f"%{search}%")
        
        # Count total
        count_query = f"SELECT COUNT(*) FROM tracks {where_clause}"
        total = await conn.fetchval(count_query, *params)
        
        # Get tracks
        offset = (page - 1) * per_page
        order_clause = f"ORDER BY {sort_by} {sort_order.upper()}"
        
        if params:
            query = f"""
                SELECT id, file_path, filename, title, duration, file_size, audio_features, created_at
                FROM tracks 
                {where_clause}
                {order_clause}
                LIMIT ${len(params) + 1} OFFSET ${len(params) + 2}
            """
            params.extend([per_page, offset])
        else:
            query = f"""
                SELECT id, file_path, filename, title, duration, file_size, audio_features, created_at
                FROM tracks 
                {order_clause}
                LIMIT $1 OFFSET $2
            """
            params = [per_page, offset]
        
        rows = await conn.fetch(query, *params)
        
        # Format tracks
        tracks = []
        for row in rows:
            track = {
                "id": row['id'],
                "file_path": row['file_path'],
                "filename": row['filename'],
                "title": row['title'] or row['filename'],
                "artist": "Unknown",  # Could extract from audio_features if available
                "album": "Unknown",
                "duration": row['duration'] or 0,
                "file_size": row['file_size'] or 0,
                "audio_features": dict(row['audio_features']) if row['audio_features'] else None,
                "created_at": row['created_at'].isoformat() if row['created_at'] else None
            }
            
            # Extract features if available
            if track['audio_features']:
                features = track['audio_features']
                if 'tempo' in features:
                    track['tempo'] = features['tempo']
                if 'key' in features:
                    track['key'] = features['key']
                if 'energy' in features:
                    track['energy'] = features['energy']
            
            tracks.append(track)
        
        return {
            "tracks": tracks,
            "total": total,
            "page": page,
            "per_page": per_page,
            "total_pages": (total + per_page - 1) // per_page
        }
        
    finally:
        await conn.close()

if __name__ == "__main__":
    import uvicorn
    print("🌐 Starting Simple Tracks API on port 8001")
    uvicorn.run(app, host="0.0.0.0", port=8001)
