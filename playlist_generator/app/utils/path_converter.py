import os
import logging
from typing import List, Optional

logger = logging.getLogger()

class PathConverter:
    """Handles path conversion between container and host paths."""
    
    def __init__(self, host_library: str, container_music: str = '/music'):
        """Initialize the path converter.
        
        Args:
            host_library (str): Host path to the music library
            container_music (str): Container path to the music directory (default: /music)
        """
        self.host_library = host_library
        self.container_music = container_music
    
    def container_to_host(self, container_path: str) -> str:
        """Convert a container path to host path.
        
        Args:
            container_path (str): Path within the container (e.g., /music/song.mp3)
            
        Returns:
            str: Host path (e.g., /root/music/library/song.mp3)
        """
        if not container_path:
            return container_path
            
        container_path = os.path.normpath(container_path)
        container_music = os.path.normpath(self.container_music)
        
        if not container_path.startswith(container_music):
            logger.warning(f"Container path {container_path} doesn't start with {container_music}")
            return container_path
            
        rel_path = os.path.relpath(container_path, container_music)
        host_path = os.path.join(self.host_library, rel_path)
        return host_path
    
    def host_to_container(self, host_path: str) -> str:
        """Convert a host path to container path.
        
        Args:
            host_path (str): Path on the host (e.g., /root/music/library/song.mp3)
            
        Returns:
            str: Container path (e.g., /music/song.mp3)
        """
        if not host_path:
            return host_path
            
        host_path = os.path.normpath(host_path)
        host_library = os.path.normpath(self.host_library)
        
        if not host_path.startswith(host_library):
            logger.warning(f"Host path {host_path} doesn't start with {host_library}")
            return host_path
            
        rel_path = os.path.relpath(host_path, host_library)
        container_path = os.path.join(self.container_music, rel_path)
        
        logger.debug(f"Converted {host_path} -> {container_path}")
        return container_path
    
    def convert_playlist_tracks(self, tracks: List[str]) -> List[str]:
        """Convert a list of track paths from container to host.
        
        Args:
            tracks (List[str]): List of container paths
            
        Returns:
            List[str]: List of host paths
        """
        return [self.container_to_host(track) for track in tracks]
    
    def convert_failed_files(self, failed_files: List[str]) -> List[str]:
        """Convert a list of failed file paths from container to host.
        
        Args:
            failed_files (List[str]): List of container paths
            
        Returns:
            List[str]: List of host paths
        """
        return [self.container_to_host(f) for f in failed_files]
    
    def is_container_path(self, path: str) -> bool:
        """Check if a path is a container path.
        
        Args:
            path (str): Path to check
            
        Returns:
            bool: True if it's a container path
        """
        return path.startswith(self.container_music)
    
    def is_host_path(self, path: str) -> bool:
        """Check if a path is a host path.
        
        Args:
            path (str): Path to check
            
        Returns:
            bool: True if it's a host path
        """
        return path.startswith(self.host_library)
    
    def get_path_info(self, path: str) -> dict:
        """Get information about a path.
        
        Args:
            path (str): Path to analyze
            
        Returns:
            dict: Information about the path
        """
        return {
            'original': path,
            'is_container': self.is_container_path(path),
            'is_host': self.is_host_path(path),
            'container_path': self.host_to_container(path) if self.is_host_path(path) else path,
            'host_path': self.container_to_host(path) if self.is_container_path(path) else path
        } 