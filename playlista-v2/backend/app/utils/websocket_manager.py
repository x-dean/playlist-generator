"""
WebSocket manager for real-time updates
"""

import json
import uuid
from typing import Dict, List

from fastapi import WebSocket
from ..core.logging import get_logger

logger = get_logger("websocket")


class WebSocketManager:
    """Manages WebSocket connections for real-time updates"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        
    async def connect(self, websocket: WebSocket) -> str:
        """Accept a new WebSocket connection"""
        await websocket.accept()
        connection_id = str(uuid.uuid4())
        self.active_connections[connection_id] = websocket
        
        logger.info(f"WebSocket connected: {connection_id}")
        logger.info(f"Active connections: {len(self.active_connections)}")
        
        return connection_id
    
    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection"""
        connection_id = None
        for conn_id, conn in self.active_connections.items():
            if conn == websocket:
                connection_id = conn_id
                break
        
        if connection_id:
            del self.active_connections[connection_id]
            logger.info(f"WebSocket disconnected: {connection_id}")
            logger.info(f"Active connections: {len(self.active_connections)}")
    
    async def send_message(self, connection_id: str, message: dict):
        """Send message to specific connection"""
        websocket = self.active_connections.get(connection_id)
        if websocket:
            try:
                await websocket.send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Failed to send message to {connection_id}: {e}")
                self.disconnect(websocket)
    
    async def broadcast(self, message: dict):
        """Broadcast message to all active connections"""
        if not self.active_connections:
            return
        
        message_text = json.dumps(message)
        disconnected = []
        
        for connection_id, websocket in self.active_connections.items():
            try:
                await websocket.send_text(message_text)
            except Exception as e:
                logger.error(f"Failed to broadcast to {connection_id}: {e}")
                disconnected.append(websocket)
        
        # Clean up disconnected connections
        for websocket in disconnected:
            self.disconnect(websocket)
    
    async def send_analysis_progress(self, track_id: str, progress: int, step: str):
        """Send analysis progress update"""
        message = {
            "type": "analysis_progress",
            "track_id": track_id,
            "progress": progress,
            "step": step
        }
        await self.broadcast(message)
    
    async def send_analysis_complete(self, track_id: str, features: dict):
        """Send analysis completion notification"""
        message = {
            "type": "analysis_complete",
            "track_id": track_id,
            "features": features
        }
        await self.broadcast(message)
    
    async def send_playlist_update(self, playlist_id: str, action: str, data: dict):
        """Send playlist update notification"""
        message = {
            "type": "playlist_update",
            "playlist_id": playlist_id,
            "action": action,  # created, updated, deleted
            "data": data
        }
        await self.broadcast(message)
