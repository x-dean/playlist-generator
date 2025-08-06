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
            
            # Additional common tags
            TagMapping('TXXX:ARTIST', 'artist', 'Artist (TXXX)', priority=1),
            TagMapping('TXXX:TITLE', 'title', 'Title (TXXX)', priority=1),
            TagMapping('TXXX:ALBUM', 'album', 'Album (TXXX)', priority=1),
            TagMapping('TXXX:YEAR', 'year', 'Year (TXXX)', data_type="int", priority=1),
            TagMapping('TXXX:GENRE', 'genre', 'Genre (TXXX)', priority=1),
            TagMapping('TXXX:TRACK', 'track_number', 'Track number (TXXX)', data_type="int", priority=1),
            TagMapping('TXXX:DISC', 'disc_number', 'Disc number (TXXX)', data_type="int", priority=2),
            
            # Alternative formats (lowercase)
            TagMapping('artist', 'artist', 'Artist (lowercase)', priority=1),
            TagMapping('title', 'title', 'Title (lowercase)', priority=1),
            TagMapping('album', 'album', 'Album (lowercase)', priority=1),
            TagMapping('year', 'year', 'Year (lowercase)', data_type="int", priority=1),
            TagMapping('genre', 'genre', 'Genre (lowercase)', priority=1),
            TagMapping('track', 'track_number', 'Track number (lowercase)', data_type="int", priority=1),
            TagMapping('disc', 'disc_number', 'Disc number (lowercase)', data_type="int", priority=2),
            TagMapping('composer', 'composer', 'Composer (lowercase)', priority=2),
            TagMapping('conductor', 'conductor', 'Conductor (lowercase)', priority=3),
            TagMapping('remixer', 'remixer', 'Remixer (lowercase)', priority=3),
            TagMapping('publisher', 'publisher', 'Publisher (lowercase)', priority=3),
            TagMapping('copyright', 'copyright', 'Copyright (lowercase)', priority=3),
            TagMapping('language', 'language', 'Language (lowercase)', priority=3),
            TagMapping('mood', 'mood', 'Mood (lowercase)', priority=2),
            TagMapping('style', 'style', 'Style (lowercase)', priority=2),
            TagMapping('quality', 'quality', 'Quality (lowercase)', priority=3),
            TagMapping('encoder', 'encoder', 'Encoder (lowercase)', priority=3),
            TagMapping('encoder', 'encoded_by', 'Encoded by (lowercase)', priority=3),  # Also map to encoded_by
            TagMapping('bpm', 'bpm', 'BPM (lowercase)', data_type="float", priority=2),
            TagMapping('tempo', 'bpm', 'Tempo (lowercase)', data_type="float", priority=2),
            TagMapping('tempo', 'tempo', 'Tempo (lowercase)', data_type="float", priority=2),  # Also map to tempo
            TagMapping('key', 'key', 'Key (lowercase)', priority=2),
            
            # Additional lowercase mappings for missing fields
            TagMapping('original_artist', 'original_artist', 'Original artist (lowercase)', priority=3),
            TagMapping('original_album', 'original_album', 'Original album (lowercase)', priority=3),
            TagMapping('lyricist', 'lyricist', 'Lyricist (lowercase)', priority=3),
            TagMapping('band', 'band', 'Band (lowercase)', priority=3),
            
            # ID3v2.3 specific tags
            TagMapping('TIT1', 'content_group', 'Content group (ID3v2.3)', priority=2),
            TagMapping('TIT3', 'subtitle', 'Subtitle (ID3v2.3)', priority=2),
            TagMapping('TPE2', 'album_artist', 'Album artist (ID3v2.3)', priority=2),
            TagMapping('TPE3', 'conductor', 'Conductor (ID3v2.3)', priority=3),
            TagMapping('TPE4', 'remixer', 'Remixer (ID3v2.3)', priority=3),
            TagMapping('TALB', 'album', 'Album (ID3v2.3)', priority=1),
            TagMapping('TRCK', 'track_number', 'Track number (ID3v2.3)', data_type="int", priority=1),
            TagMapping('TPOS', 'disc_number', 'Disc number (ID3v2.3)', data_type="int", priority=2),
            TagMapping('TDRC', 'year', 'Year (ID3v2.3)', data_type="int", priority=1),
            TagMapping('TCON', 'genre', 'Genre (ID3v2.3)', priority=1),
            TagMapping('TCOM', 'composer', 'Composer (ID3v2.3)', priority=2),
            TagMapping('TEXT', 'lyrics', 'Lyrics (ID3v2.3)', priority=3),
            TagMapping('COMM', 'comment', 'Comment (ID3v2.3)', priority=3),
            TagMapping('TPUB', 'publisher', 'Publisher (ID3v2.3)', priority=3),
            TagMapping('TCOP', 'copyright', 'Copyright (ID3v2.3)', priority=3),
            TagMapping('TLAN', 'language', 'Language (ID3v2.3)', priority=3),
            TagMapping('TBPM', 'bpm', 'BPM (ID3v2.3)', data_type="float", priority=2),
            TagMapping('TKEY', 'key', 'Key (ID3v2.3)', priority=2),
            TagMapping('TMOO', 'mood', 'Mood (ID3v2.3)', priority=2),
            TagMapping('POPM', 'rating', 'Rating (ID3v2.3)', data_type="int", priority=2),
            TagMapping('TSRC', 'isrc', 'ISRC (ID3v2.3)', priority=2),
            TagMapping('TLEN', 'length', 'Length (ID3v2.3)', data_type="int", priority=2),
            TagMapping('TENC', 'encoder', 'Encoder (ID3v2.3)', priority=3),
            TagMapping('TENC', 'encoded_by', 'Encoded by (ID3v2.3)', priority=3),  # Also map to encoded_by
            TagMapping('TOFN', 'original_filename', 'Original filename (ID3v2.3)', priority=3),
            TagMapping('TOYE', 'original_year', 'Original year (ID3v2.3)', data_type="int", priority=3),
            TagMapping('TDLY', 'playlist_delay', 'Playlist delay (ID3v2.3)', data_type="int", priority=3),
            TagMapping('TFLT', 'file_type', 'File type (ID3v2.3)', priority=3),
            
            # Additional mappings for missing fields
            TagMapping('TXXX:ORIGINAL_ARTIST', 'original_artist', 'Original artist (TXXX)', priority=3),
            TagMapping('TXXX:ORIGINAL_ALBUM', 'original_album', 'Original album (TXXX)', priority=3),
            TagMapping('TXXX:LYRICIST', 'lyricist', 'Lyricist (TXXX)', priority=3),
            TagMapping('TXXX:BAND', 'band', 'Band (TXXX)', priority=3),
            TagMapping('TXXX:TEMPO', 'tempo', 'Tempo (TXXX)', data_type="float", priority=2),
            TagMapping('TXXX:BPM', 'tempo', 'BPM as tempo (TXXX)', data_type="float", priority=2),
            
            # Vorbis/FLAC specific tags
            TagMapping('ARTIST', 'artist', 'Artist (Vorbis)', priority=1),
            TagMapping('TITLE', 'title', 'Title (Vorbis)', priority=1),
            TagMapping('ALBUM', 'album', 'Album (Vorbis)', priority=1),
            TagMapping('DATE', 'year', 'Date (Vorbis)', data_type="int", priority=1),
            TagMapping('GENRE', 'genre', 'Genre (Vorbis)', priority=1),
            TagMapping('TRACKNUMBER', 'track_number', 'Track number (Vorbis)', data_type="int", priority=1),
            TagMapping('DISCNUMBER', 'disc_number', 'Disc number (Vorbis)', data_type="int", priority=2),
            TagMapping('COMPOSER', 'composer', 'Composer (Vorbis)', priority=2),
            TagMapping('CONDUCTOR', 'conductor', 'Conductor (Vorbis)', priority=3),
            TagMapping('REMIXER', 'remixer', 'Remixer (Vorbis)', priority=3),
            TagMapping('PUBLISHER', 'publisher', 'Publisher (Vorbis)', priority=3),
            TagMapping('COPYRIGHT', 'copyright', 'Copyright (Vorbis)', priority=3),
            TagMapping('LANGUAGE', 'language', 'Language (Vorbis)', priority=3),
            TagMapping('MOOD', 'mood', 'Mood (Vorbis)', priority=2),
            TagMapping('STYLE', 'style', 'Style (Vorbis)', priority=2),
            TagMapping('QUALITY', 'quality', 'Quality (Vorbis)', priority=3),
            TagMapping('ENCODER', 'encoder', 'Encoder (Vorbis)', priority=3),
            TagMapping('ENCODER', 'encoded_by', 'Encoded by (Vorbis)', priority=3),  # Also map to encoded_by
            TagMapping('BPM', 'bpm', 'BPM (Vorbis)', data_type="float", priority=2),
            TagMapping('TEMPO', 'bpm', 'Tempo (Vorbis)', data_type="float", priority=2),
            TagMapping('TEMPO', 'tempo', 'Tempo (Vorbis)', data_type="float", priority=2),  # Also map to tempo
            TagMapping('KEY', 'key', 'Key (Vorbis)', priority=2),
            TagMapping('RATING', 'rating', 'Rating (Vorbis)', data_type="int", priority=2),
            TagMapping('ISRC', 'isrc', 'ISRC (Vorbis)', priority=2),
            TagMapping('LENGTH', 'length', 'Length (Vorbis)', data_type="int", priority=2),
            TagMapping('ORIGINALFILENAME', 'original_filename', 'Original filename (Vorbis)', priority=3),
            TagMapping('ORIGINALYEAR', 'original_year', 'Original year (Vorbis)', data_type="int", priority=3),
            TagMapping('PLAYLISTDELAY', 'playlist_delay', 'Playlist delay (Vorbis)', data_type="int", priority=3),
            TagMapping('FILETYPE', 'file_type', 'File type (Vorbis)', priority=3),
            TagMapping('RECORDINGTIME', 'recording_time', 'Recording time (Vorbis)', priority=3),
            
                         # Additional Vorbis mappings for missing fields
             TagMapping('ORIGINALARTIST', 'original_artist', 'Original artist (Vorbis)', priority=3),
             TagMapping('ORIGINALALBUM', 'original_album', 'Original album (Vorbis)', priority=3),
             TagMapping('LYRICIST', 'lyricist', 'Lyricist (Vorbis)', priority=3),
             TagMapping('BAND', 'band', 'Band (Vorbis)', priority=3),
             
             # iTunes/M4A specific tags
             TagMapping('©nam', 'title', 'Title (iTunes/M4A)', priority=1),
             TagMapping('©ART', 'artist', 'Artist (iTunes/M4A)', priority=1),
             TagMapping('©alb', 'album', 'Album (iTunes/M4A)', priority=1),
             TagMapping('©gen', 'genre', 'Genre (iTunes/M4A)', priority=1),
             TagMapping('©day', 'year', 'Year (iTunes/M4A)', data_type="int", priority=1),
             TagMapping('trkn', 'track_number', 'Track number (iTunes/M4A)', data_type="int", priority=1),
             TagMapping('disk', 'disc_number', 'Disc number (iTunes/M4A)', data_type="int", priority=2),
             TagMapping('tmpo', 'bpm', 'BPM (iTunes/M4A)', data_type="float", priority=2),
             TagMapping('©wrt', 'composer', 'Composer (iTunes/M4A)', priority=2),
             TagMapping('©too', 'encoder', 'Encoder (iTunes/M4A)', priority=3),
             TagMapping('©too', 'encoded_by', 'Encoded by (iTunes/M4A)', priority=3),
             
             # iTunes custom tags (ReplayGain, etc.)
             TagMapping('----:com.apple.iTunes:replaygain_track_gain', 'replaygain_track_gain', 'ReplayGain track gain (iTunes)', data_type="float", priority=2),
             TagMapping('----:com.apple.iTunes:replaygain_album_gain', 'replaygain_album_gain', 'ReplayGain album gain (iTunes)', data_type="float", priority=2),
             TagMapping('----:com.apple.iTunes:replaygain_track_peak', 'replaygain_track_peak', 'ReplayGain track peak (iTunes)', data_type="float", priority=2),
             TagMapping('----:com.apple.iTunes:replaygain_album_peak', 'replaygain_album_peak', 'ReplayGain album peak (iTunes)', data_type="float", priority=2),
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
            # Check if this mutagen key exists in the tags (case-insensitive)
            found_key = None
            for tag_key in mutagen_tags.keys():
                # Ensure tag_key is a string before calling .lower()
                if isinstance(tag_key, str) and tag_key.lower() == mapping.mutagen_key.lower():
                    found_key = tag_key
                    break
            
            if found_key:
                value = mutagen_tags[found_key]
                
                # Skip if already processed (higher priority mapping took precedence)
                if mapping.standard_key in processed_keys:
                    continue
                
                # Process the value
                processed_value = self._process_value(value, mapping)
                if processed_value is not None:
                    mapped_metadata[mapping.standard_key] = processed_value
                    processed_keys.add(mapping.standard_key)
                    
                    log_universal('DEBUG', 'TagMapper', 
                                f'Mapped {found_key} -> {mapping.standard_key}: {processed_value}')
            else:
                # Debug: log when a mapping key is not found
                if mapping.priority <= 2:  # Only log high priority mappings to avoid spam
                    log_universal('DEBUG', 'TagMapper', 
                                f'Key not found: {mapping.mutagen_key} (expected for {mapping.standard_key})')
        
        # Handle custom TXXX tags
        custom_tags = self._extract_custom_tags(mutagen_tags)
        if custom_tags:
            mapped_metadata['custom_tags'] = custom_tags
            
            # Extract ReplayGain fields from custom tags
            replaygain_mappings = {
                'REPLAYGAIN_TRACK_GAIN': 'replaygain_track_gain',
                'replaygain_track_gain': 'replaygain_track_gain',
                'REPLAYGAIN_ALBUM_GAIN': 'replaygain_album_gain',
                'replaygain_album_gain': 'replaygain_album_gain',
                'REPLAYGAIN_TRACK_PEAK': 'replaygain_track_peak',
                'replaygain_track_peak': 'replaygain_track_peak',
                'REPLAYGAIN_ALBUM_PEAK': 'replaygain_album_peak',
                'replaygain_album_peak': 'replaygain_album_peak'
            }
            
            for custom_key, standard_key in replaygain_mappings.items():
                if custom_key in custom_tags:
                     value = custom_tags[custom_key]
                     log_universal('DEBUG', 'TagMapper', f'Found ReplayGain key: {custom_key}, value: "{value}"')
                     # Convert ReplayGain values (remove "dB" suffix for gain values)
                     if isinstance(value, str):
                         # Handle iTunes format: "b'-1.55 dB'" -> "-1.55"
                         if value.startswith("b'") and value.endswith("'"):
                             value = value[2:-1]  # Remove b' and '
                         # Remove "dB" suffix for gain values
                         if 'GAIN' in custom_key.upper():
                             value = value.replace(' dB', '').replace('dB', '')
                             log_universal('DEBUG', 'TagMapper', f'Removed dB suffix from {custom_key}: "{value}"')
                         log_universal('DEBUG', 'TagMapper', f'Processing ReplayGain {custom_key}: "{value}"')
                     try:
                         float_value = float(value)
                         mapped_metadata[standard_key] = float_value
                         log_universal('DEBUG', 'TagMapper', f'Extracted ReplayGain {custom_key} -> {standard_key}: {float_value}')
                     except (ValueError, TypeError):
                         log_universal('WARNING', 'TagMapper', f'Failed to convert ReplayGain {custom_key}: {value}')
        
        # Fallback: try to map unknown tags to common fields
        if not mapped_metadata:
            log_universal('DEBUG', 'TagMapper', 'No standard mappings found, trying fallback mappings')
            fallback_mappings = {
                'artist': ['artist', 'ARTIST', 'Artist', 'ARTISTNAME', 'PERFORMER'],
                'title': ['title', 'TITLE', 'Title', 'TRACKNAME', 'NAME'],
                'album': ['album', 'ALBUM', 'Album', 'ALBUMNAME'],
                'year': ['year', 'YEAR', 'Year', 'DATE', 'date'],
                'genre': ['genre', 'GENRE', 'Genre', 'STYLE', 'style'],
                'track_number': ['track', 'TRACK', 'Track', 'TRACKNUMBER', 'tracknumber'],
                'disc_number': ['disc', 'DISC', 'Disc', 'DISCNUMBER', 'discnumber'],
                'composer': ['composer', 'COMPOSER', 'Composer', 'WRITER'],
                'bpm': ['bpm', 'BPM', 'Bpm', 'TEMPO', 'tempo'],
                'key': ['key', 'KEY', 'Key', 'MUSICALKEY'],
                'rating': ['rating', 'RATING', 'Rating', 'SCORE']
            }
            
            for standard_key, possible_keys in fallback_mappings.items():
                if standard_key not in mapped_metadata:
                    for possible_key in possible_keys:
                        if possible_key in mutagen_tags:
                            value = mutagen_tags[possible_key]
                            processed_value = self._process_value(value, TagMapping(possible_key, standard_key, 'Fallback mapping'))
                            if processed_value is not None:
                                mapped_metadata[standard_key] = processed_value
                                log_universal('DEBUG', 'TagMapper', f'Fallback mapped {possible_key} -> {standard_key}: {processed_value}')
                                break
        
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
            # Handle lists (common in iTunes/M4A tags)
            if isinstance(value, list):
                if len(value) == 0:
                    return None
                elif len(value) == 1:
                    # Single value in list - extract it
                    single_value = value[0]
                    if mapping.multiple:
                        # Multiple values expected - return as list
                        processed_item = self._convert_value(single_value, mapping.data_type)
                        return [processed_item] if processed_item is not None else None
                    else:
                        # Single value expected - return the value
                        return self._convert_value(single_value, mapping.data_type)
                else:
                    # Multiple values in list
                    if mapping.multiple:
                        # Multiple values expected - process all
                        processed_values = []
                        for item in value:
                            processed_item = self._convert_value(item, mapping.data_type)
                            if processed_item is not None:
                                processed_values.append(processed_item)
                        return processed_values if processed_values else None
                    else:
                        # Single value expected - take first
                        return self._convert_value(value[0], mapping.data_type)
            
            # Handle multiple values in strings
            elif mapping.multiple and isinstance(value, str):
                # Split comma-separated values
                values = [v.strip() for v in value.split(',') if v.strip()]
                processed_values = []
                for item in values:
                    processed_item = self._convert_value(item, mapping.data_type)
                    if processed_item is not None:
                        processed_values.append(processed_item)
                return processed_values if processed_values else None
            
            # Handle single values
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
                # Handle iTunes track number format: "(1592, 0)" -> 1592
                if isinstance(value, str) and value.startswith('(') and value.endswith(')'):
                    # Extract first number from tuple format
                    parts = value.strip('()').split(',')
                    if parts:
                        return int(parts[0].strip())
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
        Extract custom TXXX tags and iTunes custom tags.
        
        Args:
            mutagen_tags: Raw mutagen tags
            
        Returns:
            Dictionary of custom tags
        """
        custom_tags = {}
        
        for key, value in mutagen_tags.items():
            # Handle TXXX tags (ID3v2.3 custom tags)
            if key.startswith('TXXX'):
                # Extract the custom tag name from the key
                # Format: TXXX:REPLAYGAIN_TRACK_GAIN -> REPLAYGAIN_TRACK_GAIN
                if ':' in key:
                    custom_name = key.split(':', 1)[1]
                else:
                    custom_name = key
                
                # Extract the value
                if isinstance(value, list) and len(value) > 0:
                    custom_value = str(value[0])
                elif isinstance(value, str):
                    # Handle string format "name\0value"
                    if '\0' in value:
                        parts = value.split('\0', 1)
                        custom_value = parts[1] if len(parts) > 1 else value
                    else:
                        custom_value = value
                else:
                    custom_value = str(value)
                
                custom_tags[custom_name] = custom_value
                
                log_universal('DEBUG', 'TagMapper', f'Extracted TXXX custom tag: {custom_name} = {custom_value}')
            
            # Handle iTunes custom tags (----:com.apple.iTunes:tag_name)
            elif key.startswith('----:com.apple.iTunes:'):
                # Extract the custom tag name from the key
                # Format: ----:com.apple.iTunes:replaygain_track_gain -> replaygain_track_gain
                custom_name = key.split(':', 2)[2] if key.count(':') >= 2 else key
                
                # Extract the value
                if isinstance(value, list) and len(value) > 0:
                    custom_value = str(value[0])
                elif isinstance(value, str):
                    custom_value = value
                else:
                    custom_value = str(value)
                
                custom_tags[custom_name] = custom_value
                
                log_universal('DEBUG', 'TagMapper', f'Extracted iTunes custom tag: {custom_name} = {custom_value}')
        
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