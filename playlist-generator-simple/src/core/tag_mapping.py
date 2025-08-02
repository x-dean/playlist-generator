"""
Tag mapping utilities for Playlist Generator Simple.
Based on Navidrome's tag mappings for comprehensive metadata extraction.
"""

import re
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass

# Import universal logging
from .logging_setup import get_logger, log_universal

logger = get_logger(__name__)


@dataclass
class TagMapping:
    """Tag mapping configuration."""
    mutagen_key: str
    standard_key: str
    description: str
    data_type: str = "string"
    multiple: bool = False
    priority: int = 0


class TagMapper:
    """
    Comprehensive tag mapper based on Navidrome's mappings.
    
    Maps various mutagen tag formats to standardized keys for
    consistent metadata handling across different audio formats.
    """
    
    def __init__(self):
        """Initialize tag mapper with comprehensive mappings."""
        self.mappings = self._create_mappings()
        log_universal('INFO', 'TagMapper', f'Initialized with {len(self.mappings)} tag mappings')
    
    def _create_mappings(self) -> List[TagMapping]:
        """Create comprehensive tag mappings based on Navidrome's approach."""
        return [
            # Basic track information
            TagMapping('TIT2', 'title', 'Track title', priority=1),
            TagMapping('TITLE', 'title', 'Track title (alternative)', priority=1),
            TagMapping('TIT1', 'content_group', 'Content group', priority=2),
            TagMapping('TIT3', 'subtitle', 'Track subtitle', priority=2),
            
            # Artist information
            TagMapping('TPE1', 'artist', 'Lead performer', priority=1),
            TagMapping('ARTIST', 'artist', 'Lead performer (alternative)', priority=1),
            TagMapping('TPE2', 'album_artist', 'Album artist', priority=2),
            TagMapping('ALBUMARTIST', 'album_artist', 'Album artist (alternative)', priority=2),
            TagMapping('TPE3', 'conductor', 'Conductor', priority=3),
            TagMapping('TPE4', 'remixer', 'Remixer', priority=3),
            
            # Album information
            TagMapping('TALB', 'album', 'Album title', priority=1),
            TagMapping('ALBUM', 'album', 'Album title (alternative)', priority=1),
            TagMapping('TSOA', 'album_sort', 'Album sort order', priority=2),
            TagMapping('TSOP', 'artist_sort', 'Artist sort order', priority=2),
            TagMapping('TSOT', 'title_sort', 'Title sort order', priority=2),
            
            # Track numbering
            TagMapping('TRCK', 'track_number', 'Track number', data_type="int", priority=1),
            TagMapping('TRACK', 'track_number', 'Track number (alternative)', data_type="int", priority=1),
            TagMapping('TPOS', 'disc_number', 'Disc number', data_type="int", priority=2),
            TagMapping('DISC', 'disc_number', 'Disc number (alternative)', data_type="int", priority=2),
            
            # Year and date
            TagMapping('TDRC', 'year', 'Recording time', priority=1),
            TagMapping('TYER', 'year', 'Year', priority=1),
            TagMapping('YEAR', 'year', 'Year (alternative)', priority=1),
            TagMapping('TDRL', 'release_date', 'Release time', priority=2),
            TagMapping('TDOR', 'original_release_date', 'Original release time', priority=2),
            
            # Genre
            TagMapping('TCON', 'genre', 'Genre', multiple=True, priority=1),
            TagMapping('GENRE', 'genre', 'Genre (alternative)', multiple=True, priority=1),
            
            # Composer
            TagMapping('TCOM', 'composer', 'Composer', multiple=True, priority=2),
            TagMapping('COMPOSER', 'composer', 'Composer (alternative)', multiple=True, priority=2),
            
            # Lyrics
            TagMapping('TEXT', 'lyrics', 'Lyrics', priority=3),
            TagMapping('LYRICS', 'lyrics', 'Lyrics (alternative)', priority=3),
            TagMapping('USLT', 'lyrics', 'Unsynchronized lyrics', priority=3),
            
            # Comment
            TagMapping('COMM', 'comment', 'Comment', priority=3),
            TagMapping('COMMENT', 'comment', 'Comment (alternative)', priority=3),
            
            # Publisher
            TagMapping('TPUB', 'publisher', 'Publisher', priority=3),
            TagMapping('PUBLISHER', 'publisher', 'Publisher (alternative)', priority=3),
            
            # Copyright
            TagMapping('TCOP', 'copyright', 'Copyright', priority=3),
            TagMapping('COPYRIGHT', 'copyright', 'Copyright (alternative)', priority=3),
            
            # Language
            TagMapping('TLAN', 'language', 'Language', priority=3),
            TagMapping('LANGUAGE', 'language', 'Language (alternative)', priority=3),
            
            # BPM
            TagMapping('TBPM', 'bpm', 'BPM', data_type="float", priority=2),
            TagMapping('BPM', 'bpm', 'BPM (alternative)', data_type="float", priority=2),
            TagMapping('TEMPO', 'bpm', 'Tempo', data_type="float", priority=2),
            
            # Key
            TagMapping('TKEY', 'key', 'Key', priority=2),
            TagMapping('KEY', 'key', 'Key (alternative)', priority=2),
            
            # Mood
            TagMapping('TMOO', 'mood', 'Mood', priority=2),
            TagMapping('MOOD', 'mood', 'Mood (alternative)', priority=2),
            
            # Rating
            TagMapping('POPM', 'rating', 'Popularimeter', data_type="int", priority=2),
            TagMapping('RATING', 'rating', 'Rating', data_type="int", priority=2),
            
            # ReplayGain
            TagMapping('TXXX:REPLAYGAIN_TRACK_GAIN', 'replaygain_track_gain', 'ReplayGain track gain', data_type="float", priority=2),
            TagMapping('TXXX:REPLAYGAIN_ALBUM_GAIN', 'replaygain_album_gain', 'ReplayGain album gain', data_type="float", priority=2),
            TagMapping('TXXX:REPLAYGAIN_TRACK_PEAK', 'replaygain_track_peak', 'ReplayGain track peak', data_type="float", priority=2),
            TagMapping('TXXX:REPLAYGAIN_ALBUM_PEAK', 'replaygain_album_peak', 'ReplayGain album peak', data_type="float", priority=2),
            
            # MusicBrainz IDs
            TagMapping('TXXX:MUSICBRAINZ_TRACKID', 'musicbrainz_track_id', 'MusicBrainz track ID', priority=2),
            TagMapping('TXXX:MUSICBRAINZ_ARTISTID', 'musicbrainz_artist_id', 'MusicBrainz artist ID', priority=2),
            TagMapping('TXXX:MUSICBRAINZ_ALBUMID', 'musicbrainz_album_id', 'MusicBrainz album ID', priority=2),
            TagMapping('TXXX:MUSICBRAINZ_ALBUMARTISTID', 'musicbrainz_album_artist_id', 'MusicBrainz album artist ID', priority=2),
            TagMapping('TXXX:MUSICBRAINZ_RELEASEGROUPID', 'musicbrainz_release_group_id', 'MusicBrainz release group ID', priority=2),
            TagMapping('TXXX:MUSICBRAINZ_RECORDINGID', 'musicbrainz_recording_id', 'MusicBrainz recording ID', priority=2),
            TagMapping('TXXX:MUSICBRAINZ_WORKID', 'musicbrainz_work_id', 'MusicBrainz work ID', priority=2),
            
            # AcoustID
            TagMapping('TXXX:ACOUSTID_ID', 'acoustid_id', 'AcoustID', priority=2),
            TagMapping('TXXX:ACOUSTID_FINGERPRINT', 'acoustid_fingerprint', 'AcoustID fingerprint', priority=2),
            
            # ISRC
            TagMapping('TSRC', 'isrc', 'ISRC', priority=2),
            TagMapping('ISRC', 'isrc', 'ISRC (alternative)', priority=2),
            
            # Length
            TagMapping('TLEN', 'length', 'Length', data_type="int", priority=2),
            TagMapping('LENGTH', 'length', 'Length (alternative)', data_type="int", priority=2),
            
            # Encoder
            TagMapping('TENC', 'encoder', 'Encoder', priority=3),
            TagMapping('ENCODER', 'encoder', 'Encoder (alternative)', priority=3),
            
            # Original filename
            TagMapping('TOFN', 'original_filename', 'Original filename', priority=3),
            TagMapping('ORIGINALFILENAME', 'original_filename', 'Original filename (alternative)', priority=3),
            
            # Original year
            TagMapping('TOYE', 'original_year', 'Original year', data_type="int", priority=3),
            TagMapping('ORIGINALYEAR', 'original_year', 'Original year (alternative)', data_type="int", priority=3),
            
            # Playlist delay
            TagMapping('TDLY', 'playlist_delay', 'Playlist delay', data_type="int", priority=3),
            TagMapping('PLAYLISTDELAY', 'playlist_delay', 'Playlist delay (alternative)', data_type="int", priority=3),
            
            # File type
            TagMapping('TFLT', 'file_type', 'File type', priority=3),
            TagMapping('FILETYPE', 'file_type', 'File type (alternative)', priority=3),
            
            # Recording time
            TagMapping('TDRC', 'recording_time', 'Recording time', priority=3),
            TagMapping('RECORDINGTIME', 'recording_time', 'Recording time (alternative)', priority=3),
        ]
    
    def map_tags(self, mutagen_tags: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map mutagen tags to standardized format.
        
        Args:
            mutagen_tags: Raw mutagen tags dictionary
            
        Returns:
            Standardized metadata dictionary
        """
        if not mutagen_tags:
            return {}
        
        mapped_metadata = {}
        processed_keys = set()
        
        # Sort mappings by priority (highest first)
        sorted_mappings = sorted(self.mappings, key=lambda x: x.priority, reverse=True)
        
        for mapping in sorted_mappings:
            # Check if this mutagen key exists in the tags
            if mapping.mutagen_key in mutagen_tags:
                value = mutagen_tags[mapping.mutagen_key]
                
                # Skip if already processed (higher priority mapping took precedence)
                if mapping.standard_key in processed_keys:
                    continue
                
                # Process the value
                processed_value = self._process_value(value, mapping)
                if processed_value is not None:
                    mapped_metadata[mapping.standard_key] = processed_value
                    processed_keys.add(mapping.standard_key)
                    
                    log_universal('DEBUG', 'TagMapper', 
                                f'Mapped {mapping.mutagen_key} -> {mapping.standard_key}: {processed_value}')
        
        # Handle custom TXXX tags
        custom_tags = self._extract_custom_tags(mutagen_tags)
        if custom_tags:
            mapped_metadata['custom_tags'] = custom_tags
        
        log_universal('INFO', 'TagMapper', f'Mapped {len(mapped_metadata)} standard fields from {len(mutagen_tags)} raw tags')
        return mapped_metadata
    
    def _process_value(self, value: Any, mapping: TagMapping) -> Any:
        """
        Process a tag value according to its mapping configuration.
        
        Args:
            value: Raw tag value
            mapping: Tag mapping configuration
            
        Returns:
            Processed value
        """
        try:
            # Handle multiple values
            if mapping.multiple and isinstance(value, list):
                processed_values = []
                for item in value:
                    processed_item = self._convert_value(item, mapping.data_type)
                    if processed_item is not None:
                        processed_values.append(processed_item)
                return processed_values if processed_values else None
            elif mapping.multiple and isinstance(value, str):
                # Split comma-separated values
                values = [v.strip() for v in value.split(',') if v.strip()]
                processed_values = []
                for item in values:
                    processed_item = self._convert_value(item, mapping.data_type)
                    if processed_item is not None:
                        processed_values.append(processed_item)
                return processed_values if processed_values else None
            else:
                return self._convert_value(value, mapping.data_type)
                
        except Exception as e:
            log_universal('WARNING', 'TagMapper', f'Error processing {mapping.mutagen_key}: {e}')
            return None
    
    def _convert_value(self, value: Any, data_type: str) -> Any:
        """
        Convert value to specified data type.
        
        Args:
            value: Raw value
            data_type: Target data type
            
        Returns:
            Converted value
        """
        if value is None:
            return None
        
        try:
            if data_type == "int":
                return int(value)
            elif data_type == "float":
                return float(value)
            elif data_type == "string":
                return str(value).strip()
            else:
                return str(value).strip()
        except (ValueError, TypeError):
            return None
    
    def _extract_custom_tags(self, mutagen_tags: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract custom TXXX tags.
        
        Args:
            mutagen_tags: Raw mutagen tags
            
        Returns:
            Dictionary of custom tags
        """
        custom_tags = {}
        
        for key, value in mutagen_tags.items():
            if key.startswith('TXXX'):
                # Extract the custom tag name
                if isinstance(value, list) and len(value) > 0:
                    custom_name = str(value[0])
                    custom_value = value[1] if len(value) > 1 else ""
                elif isinstance(value, str):
                    # Handle string format "name\0value"
                    if '\0' in value:
                        parts = value.split('\0', 1)
                        custom_name = parts[0]
                        custom_value = parts[1] if len(parts) > 1 else ""
                    else:
                        custom_name = key
                        custom_value = value
                else:
                    custom_name = key
                    custom_value = str(value)
                
                custom_tags[custom_name] = custom_value
        
        return custom_tags
    
    def get_available_mappings(self) -> List[str]:
        """Get list of available standard keys."""
        return list(set(mapping.standard_key for mapping in self.mappings))
    
    def get_mapping_info(self) -> Dict[str, Any]:
        """Get detailed mapping information."""
        info = {
            'total_mappings': len(self.mappings),
            'standard_keys': self.get_available_mappings(),
            'priority_levels': {},
            'data_types': {}
        }
        
        for mapping in self.mappings:
            if mapping.priority not in info['priority_levels']:
                info['priority_levels'][mapping.priority] = []
            info['priority_levels'][mapping.priority].append(mapping.standard_key)
            
            if mapping.data_type not in info['data_types']:
                info['data_types'][mapping.data_type] = []
            info['data_types'][mapping.data_type].append(mapping.standard_key)
        
        return info


# Global tag mapper instance
_tag_mapper_instance = None

def get_tag_mapper() -> 'TagMapper':
    """Get the global tag mapper instance, creating it if necessary."""
    global _tag_mapper_instance
    if _tag_mapper_instance is None:
        _tag_mapper_instance = TagMapper()
    return _tag_mapper_instance 