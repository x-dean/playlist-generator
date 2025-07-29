"""
Playlist exporter service.

This module provides functionality to export playlists in various formats
including M3U, PLS, and XSPF.
"""

import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
from xml.etree import ElementTree as ET

from domain.entities import Playlist
from shared.config import get_config
from shared.exceptions import FileSystemError


class PlaylistExporter:
    """Service for exporting playlists in various formats."""
    
    def __init__(self):
        """Initialize the playlist exporter."""
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
        self.output_dir = self.config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def export_m3u(self, playlist: Playlist, filename: Optional[str] = None) -> Path:
        """Export playlist to M3U format."""
        try:
            if not filename:
                filename = f"{playlist.name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.m3u"
            
            output_path = self.output_dir / filename
            
            with open(output_path, 'w', encoding='utf-8') as f:
                # Write header
                f.write("#EXTM3U\n")
                f.write(f"# Playlist: {playlist.name}\n")
                if playlist.description:
                    f.write(f"# Description: {playlist.description}\n")
                f.write(f"# Created: {datetime.now().isoformat()}\n")
                f.write(f"# Tracks: {len(playlist.track_ids)}\n\n")
                
                # Write tracks
                for i, track_path in enumerate(playlist.track_paths, 1):
                    f.write(f"#EXTINF:-1,Track {i}\n")
                    f.write(f"{track_path}\n")
            
            self.logger.info(f"Exported playlist to M3U: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Failed to export playlist to M3U: {e}")
            raise FileSystemError(f"Failed to export playlist to M3U: {e}")
    
    def export_pls(self, playlist: Playlist, filename: Optional[str] = None) -> Path:
        """Export playlist to PLS format."""
        try:
            if not filename:
                filename = f"{playlist.name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pls"
            
            output_path = self.output_dir / filename
            
            with open(output_path, 'w', encoding='utf-8') as f:
                # Write header
                f.write("[playlist]\n")
                f.write(f"NumberOfEntries={len(playlist.track_ids)}\n")
                f.write(f"Title={playlist.name}\n")
                if playlist.description:
                    f.write(f"Description={playlist.description}\n")
                f.write(f"Created={datetime.now().isoformat()}\n\n")
                
                # Write tracks
                for i, track_path in enumerate(playlist.track_paths, 1):
                    f.write(f"File{i}={track_path}\n")
                    f.write(f"Title{i}=Track {i}\n")
                    f.write(f"Length{i}=-1\n\n")
            
            self.logger.info(f"Exported playlist to PLS: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Failed to export playlist to PLS: {e}")
            raise FileSystemError(f"Failed to export playlist to PLS: {e}")
    
    def export_xspf(self, playlist: Playlist, filename: Optional[str] = None) -> Path:
        """Export playlist to XSPF (XML Shareable Playlist Format)."""
        try:
            if not filename:
                filename = f"{playlist.name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xspf"
            
            output_path = self.output_dir / filename
            
            # Create XML structure
            root = ET.Element("playlist", version="1", xmlns="http://xspf.org/ns/0/")
            
            # Add title
            title_elem = ET.SubElement(root, "title")
            title_elem.text = playlist.name
            
            # Add description if available
            if playlist.description:
                annotation_elem = ET.SubElement(root, "annotation")
                annotation_elem.text = playlist.description
            
            # Add creator and date
            creator_elem = ET.SubElement(root, "creator")
            creator_elem.text = "Playlista"
            
            date_elem = ET.SubElement(root, "date")
            date_elem.text = datetime.now().isoformat()
            
            # Add track list
            track_list_elem = ET.SubElement(root, "trackList")
            
            for i, track_path in enumerate(playlist.track_paths, 1):
                track_elem = ET.SubElement(track_list_elem, "track")
                
                # Add track number
                track_num_elem = ET.SubElement(track_elem, "trackNum")
                track_num_elem.text = str(i)
                
                # Add location (file path)
                location_elem = ET.SubElement(track_elem, "location")
                location_elem.text = str(track_path)
                
                # Add title
                title_elem = ET.SubElement(track_elem, "title")
                title_elem.text = f"Track {i}"
            
            # Write XML to file
            tree = ET.ElementTree(root)
            ET.indent(tree, space="  ")  # Pretty print
            
            with open(output_path, 'w', encoding='utf-8') as f:
                tree.write(f, encoding='unicode', xml_declaration=True)
            
            self.logger.info(f"Exported playlist to XSPF: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Failed to export playlist to XSPF: {e}")
            raise FileSystemError(f"Failed to export playlist to XSPF: {e}")
    
    def export_json(self, playlist: Playlist, filename: Optional[str] = None) -> Path:
        """Export playlist to JSON format."""
        try:
            import json
            
            if not filename:
                filename = f"{playlist.name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            output_path = self.output_dir / filename
            
            # Create playlist data
            playlist_data = {
                "name": playlist.name,
                "description": playlist.description,
                "created_date": playlist.created_date.isoformat() if playlist.created_date else None,
                "track_count": len(playlist.track_ids),
                "tracks": [
                    {
                        "id": str(track_id),
                        "path": track_path
                    }
                    for track_id, track_path in zip(playlist.track_ids, playlist.track_paths)
                ],
                "export_info": {
                    "exported_at": datetime.now().isoformat(),
                    "format": "json",
                    "version": "1.0"
                }
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(playlist_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Exported playlist to JSON: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Failed to export playlist to JSON: {e}")
            raise FileSystemError(f"Failed to export playlist to JSON: {e}")
    
    def export_all_formats(self, playlist: Playlist, base_filename: Optional[str] = None) -> Dict[str, Path]:
        """Export playlist to all supported formats."""
        try:
            if not base_filename:
                base_filename = playlist.name.replace(' ', '_')
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            exports = {}
            
            # Export to all formats
            exports['m3u'] = self.export_m3u(playlist, f"{base_filename}_{timestamp}.m3u")
            exports['pls'] = self.export_pls(playlist, f"{base_filename}_{timestamp}.pls")
            exports['xspf'] = self.export_xspf(playlist, f"{base_filename}_{timestamp}.xspf")
            exports['json'] = self.export_json(playlist, f"{base_filename}_{timestamp}.json")
            
            self.logger.info(f"Exported playlist '{playlist.name}' to all formats")
            return exports
            
        except Exception as e:
            self.logger.error(f"Failed to export playlist to all formats: {e}")
            raise FileSystemError(f"Failed to export playlist to all formats: {e}")
    
    def get_export_formats(self) -> List[str]:
        """Get list of supported export formats."""
        return ['m3u', 'pls', 'xspf', 'json']
    
    def validate_playlist(self, playlist: Playlist) -> bool:
        """Validate playlist for export."""
        if not playlist.name:
            self.logger.error("Playlist must have a name")
            return False
        
        if not playlist.track_ids or not playlist.track_paths:
            self.logger.error("Playlist must have tracks")
            return False
        
        if len(playlist.track_ids) != len(playlist.track_paths):
            self.logger.error("Track IDs and paths must have the same length")
            return False
        
        return True 