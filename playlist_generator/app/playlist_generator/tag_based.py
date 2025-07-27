import logging
from collections import defaultdict, Counter
import musicbrainzngs
import os
import requests

logger = logging.getLogger(__name__)


class TagBasedPlaylistGenerator:
    def __init__(self, min_tracks_per_genre=10, min_subgroup_size=10, large_group_threshold=40, db_file=None):
        self.min_tracks_per_genre = min_tracks_per_genre
        self.min_subgroup_size = min_subgroup_size
        self.large_group_threshold = large_group_threshold
        self.db_file = db_file
        logger.debug(
            f"Initialized TagBasedPlaylistGenerator: min_tracks_per_genre={min_tracks_per_genre}, min_subgroup_size={min_subgroup_size}, large_group_threshold={large_group_threshold}")

        # Set MusicBrainz user agent once
        musicbrainzngs.set_useragent(
            "PlaylistGenerator", "1.0", "noreply@example.com")
        self.lastfm_api_key = os.getenv("LASTFM_API_KEY")
        if self.lastfm_api_key:
            logger.debug("Last.fm API key found")
        else:
            logger.debug("Last.fm API key not found")

    def enrich_track_metadata(self, track):
        """
        Enrich track metadata using MusicBrainz and Last.fm.
        Fills in missing genres, years, etc. if possible.
        """
        logger.debug(
            f"Enriching metadata for track: {track.get('filepath', 'unknown')}")

        meta = track.get('metadata', {})
        artist = meta.get('artist')
        title = meta.get('title')

        # Only enrich if genre/year are missing
        needs_enrichment = not meta.get('genre') or not meta.get('year')
        if not needs_enrichment:
            logger.debug(
                f"Skipping enrichment for {track.get('filepath', 'unknown')} (metadata exists)")
            return track

        enriched = False

        # Try MusicBrainz first
        if artist and title:
            logger.debug(
                f"Attempting MusicBrainz enrichment for: {artist} - {title}")
            try:
                result = musicbrainzngs.search_recordings(
                    artist=artist, recording=title, limit=1)
                if result['recording-list']:
                    rec = result['recording-list'][0]
                    if 'first-release-date' in rec:
                        meta['year'] = rec['first-release-date'][:4]
                        logger.debug(
                            f"Extracted year from MusicBrainz: {meta['year']}")
                    if 'tag-list' in rec and rec['tag-list']:
                        meta['genre'] = [tag['name']
                                         for tag in rec['tag-list']]
                        enriched = True
                        logger.debug(
                            f"Extracted genres from MusicBrainz: {meta['genre']}")
                    if 'release-list' in rec and rec['release-list']:
                        meta['album'] = rec['release-list'][0]['title']
                        logger.debug(
                            f"Extracted album from MusicBrainz: {meta['album']}")
                    track['metadata'] = meta
                    logger.info(
                        f"Enriched metadata for {artist} - {title} from MusicBrainz.")
                else:
                    logger.info(
                        f"No MusicBrainz result for {artist} - {title}.")
            except Exception as e:
                logger.warning(
                    f"MusicBrainz enrichment failed for {artist} - {title}: {e}")
                import traceback
                logger.debug(
                    f"MusicBrainz enrichment error traceback: {traceback.format_exc()}")

        # If not enriched, try Last.fm
        if not enriched and artist and title and self.lastfm_api_key:
            logger.debug(
                f"Attempting Last.fm enrichment for: {artist} - {title}")
            try:
                url = (
                    "http://ws.audioscrobbler.com/2.0/"
                    "?method=track.getInfo"
                    f"&api_key={self.lastfm_api_key}"
                    f"&artist={requests.utils.quote(artist)}"
                    f"&track={requests.utils.quote(title)}"
                    "&format=json"
                )
                resp = requests.get(url, timeout=5)
                if resp.status_code == 200:
                    data = resp.json()
                    if 'track' in data:
                        track_info = data['track']
                        # Get tags
                        tags = track_info.get('toptags', {}).get('tag', [])
                        if tags:
                            meta['genre'] = [t['name']
                                             for t in tags if 'name' in t]
                            logger.debug(
                                f"Extracted genres from Last.fm: {meta['genre']}")
                        # Get album
                        if 'album' in track_info and 'title' in track_info['album']:
                            meta['album'] = track_info['album']['title']
                            logger.debug(
                                f"Extracted album from Last.fm: {meta['album']}")
                        # Get year (if available)
                        if 'wiki' in track_info and 'published' in track_info['wiki']:
                            import re
                            match = re.search(
                                r'(\\d{4})', track_info['wiki']['published'])
                            if match:
                                meta['year'] = match.group(1)
                                logger.debug(
                                    f"Extracted year from Last.fm: {meta['year']}")
                        track['metadata'] = meta
                        logger.info(
                            f"Enriched metadata for {artist} - {title} from Last.fm.")
                else:
                    logger.info(
                        f"Last.fm API returned status {resp.status_code} for {artist} - {title}.")
            except Exception as e:
                logger.warning(
                    f"Last.fm enrichment failed for {artist} - {title}: {e}")
                import traceback
                logger.debug(
                    f"Last.fm enrichment error traceback: {traceback.format_exc()}")
        elif not self.lastfm_api_key:
            logger.debug(
                "LASTFM_API_KEY not set; skipping Last.fm enrichment.")

        # Update database if db_file is set
        if self.db_file and (enriched or not meta.get('genre') or not meta.get('year')):
            logger.debug(
                f"Updating track metadata in database: {track.get('filepath', 'unknown')}")
            self.update_track_metadata_in_db(track.get('filepath'), meta)

        return track

    def update_track_metadata_in_db(self, filepath, metadata):
        """
        Update the metadata column for the given file in the audio_features table.
        """
        if not self.db_file or not filepath:
            logger.debug("Skipping database update: no db_file or filepath")
            return

        logger.debug(f"Updating metadata in database for: {filepath}")

        try:
            import sqlite3
            import json
            with sqlite3.connect(self.db_file) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE audio_features 
                    SET metadata = ? 
                    WHERE file_path = ?
                """, (json.dumps(metadata), filepath))

                if cursor.rowcount > 0:
                    logger.debug(
                        f"Successfully updated metadata for {filepath}")
                else:
                    logger.warning(f"No rows updated for {filepath}")

        except Exception as e:
            logger.error(
                f"Error updating metadata in database for {filepath}: {e}")
            import traceback
            logger.debug(
                f"Database update error traceback: {traceback.format_exc()}")

    def group_by_genre(self, features_list):
        """Group tracks by genre, handling various genre formats."""
        logger.debug(f"Grouping {len(features_list)} tracks by genre")

        try:
            genre_groups = defaultdict(list)
            processed_tracks = 0
            skipped_tracks = 0

            for track in features_list:
                if not track or 'metadata' not in track:
                    skipped_tracks += 1
                    logger.debug(f"Skipping track without metadata: {track}")
                    continue

                metadata = track['metadata']
                genres = metadata.get('genre', [])

                if not genres:
                    skipped_tracks += 1
                    logger.debug(
                        f"Skipping track without genre: {track.get('filepath', 'unknown')}")
                    continue

                # Handle both string and list genre formats
                if isinstance(genres, str):
                    genres = [genres]

                # Use the first genre for grouping
                primary_genre = self._normalize_genre(genres[0])
                if primary_genre:
                    genre_groups[primary_genre].append(track)
                    processed_tracks += 1
                    logger.debug(
                        f"Added track to genre group '{primary_genre}': {track.get('filepath', 'unknown')}")
                else:
                    skipped_tracks += 1
                    logger.debug(
                        f"Skipping track with invalid genre: {genres[0]}")

            logger.info(
                f"Genre grouping complete: {len(genre_groups)} groups, {processed_tracks} tracks processed, {skipped_tracks} skipped")
            logger.debug(f"Genre groups: {list(genre_groups.keys())}")

            return genre_groups

        except Exception as e:
            logger.error(f"Error in genre grouping: {str(e)}")
            import traceback
            logger.error(
                f"Genre grouping error traceback: {traceback.format_exc()}")
            return {}

    def group_by_decade(self, features_list):
        """Group tracks by decade based on year."""
        logger.debug(f"Grouping {len(features_list)} tracks by decade")

        try:
            decade_groups = defaultdict(list)
            processed_tracks = 0
            skipped_tracks = 0

            for track in features_list:
                if not track or 'metadata' not in track:
                    skipped_tracks += 1
                    logger.debug(f"Skipping track without metadata: {track}")
                    continue

                metadata = track['metadata']
                year = metadata.get('year')

                if not year:
                    skipped_tracks += 1
                    logger.debug(
                        f"Skipping track without year: {track.get('filepath', 'unknown')}")
                    continue

                try:
                    # Handle various year formats
                    year_int = int(str(year)[:4])
                    decade = self._get_decade(year_int)
                    decade_groups[decade].append(track)
                    processed_tracks += 1
                    logger.debug(
                        f"Added track to decade group '{decade}': {track.get('filepath', 'unknown')} (year: {year_int})")
                except (ValueError, TypeError):
                    skipped_tracks += 1
                    logger.debug(f"Skipping track with invalid year: {year}")

            logger.info(
                f"Decade grouping complete: {len(decade_groups)} groups, {processed_tracks} tracks processed, {skipped_tracks} skipped")
            logger.debug(f"Decade groups: {list(decade_groups.keys())}")

            return decade_groups

        except Exception as e:
            logger.error(f"Error in decade grouping: {str(e)}")
            import traceback
            logger.error(
                f"Decade grouping error traceback: {traceback.format_exc()}")
            return {}

    def _normalize_genre(self, genre):
        # Replace underscores/hyphens with spaces, title case, strip
        if not genre:
            return None
        normalized = genre.replace('_', ' ').replace('-', ' ').title().strip()
        logger.debug(f"Normalized genre: '{genre}' -> '{normalized}'")
        return normalized

    def _get_decade(self, year):
        decade_start = (year // 10) * 10
        return f"{decade_start}s"

    def _get_mood(self, track):
        # Use danceability, bpm, centroid for a simple mood/energy label
        danceability = track.get('danceability', 0)
        bpm = track.get('bpm', 0)
        centroid = track.get('centroid', 0)

        # Skip mood classification for tracks with failed BPM extraction
        if bpm == -1.0:
            return 'Unknown'  # Special mood for failed BPM tracks

        if danceability > 0.7 and bpm > 120:
            return 'Energetic'
        elif danceability < 0.3 and bpm < 80:
            return 'Calm'
        elif centroid > 4000:
            return 'Bright'
        elif centroid < 1500:
            return 'Warm'
        else:
            return 'Balanced'

    def _format_display_name(self, genre, decade, mood=None):
        """Format a human-readable playlist name."""
        if mood:
            return f"{genre} {decade} ({mood})"
        return f"{genre} {decade}"

    def _format_file_name(self, genre, decade, mood=None, part=None):
        """Format a filename-safe playlist name."""
        name_parts = [genre, decade]
        if mood:
            name_parts.append(mood)
        if part:
            name_parts.append(part)

        filename = '_'.join(name_parts).replace(
            ' ', '_').replace('(', '').replace(')', '')
        logger.debug(f"Formatted filename: {filename}")
        return filename

    def generate(self, features_list):
        # Normalize genres and count
        logger.debug(
            f"Starting tag-based playlist generation with {len(features_list)} tracks")

        try:
            # Enrich metadata first
            logger.debug("Enriching track metadata")
            enriched_tracks = []
            for track in features_list:
                enriched_track = self.enrich_track_metadata(track)
                enriched_tracks.append(enriched_track)

            # Group by genre
            genre_groups = self.group_by_genre(enriched_tracks)

            # Group by decade
            decade_groups = self.group_by_decade(enriched_tracks)

            # Generate playlists
            playlists = {}

            # Genre-based playlists
            for genre, tracks in genre_groups.items():
                if len(tracks) >= self.min_tracks_per_genre:
                    # Split large groups by decade
                    decade_subgroups = self.group_by_decade(tracks)

                    for decade, decade_tracks in decade_subgroups.items():
                        if len(decade_tracks) >= self.min_subgroup_size:
                            playlist_name = self._format_file_name(
                                genre, decade)
                            playlists[playlist_name] = {
                                'tracks': [t['filepath'] for t in decade_tracks],
                                'features': {
                                    'type': 'tag_based',
                                    'genre': genre,
                                    'decade': decade,
                                    'track_count': len(decade_tracks)
                                }
                            }
                            logger.debug(
                                f"Created genre-decade playlist: {playlist_name} with {len(decade_tracks)} tracks")

            # Decade-based playlists (for tracks without genre)
            for decade, tracks in decade_groups.items():
                if len(tracks) >= self.min_subgroup_size:
                    # Filter out tracks that already have genre
                    tracks_without_genre = [
                        t for t in tracks if not t.get('metadata', {}).get('genre')]

                    if len(tracks_without_genre) >= self.min_subgroup_size:
                        playlist_name = self._format_file_name(
                            'Unknown', decade)
                        playlists[playlist_name] = {
                            'tracks': [t['filepath'] for t in tracks_without_genre],
                            'features': {
                                'type': 'tag_based',
                                'genre': 'Unknown',
                                'decade': decade,
                                'track_count': len(tracks_without_genre)
                            }
                        }
                        logger.debug(
                            f"Created decade-only playlist: {playlist_name} with {len(tracks_without_genre)} tracks")

            logger.info(
                f"Tag-based playlist generation complete: {len(playlists)} playlists created")
            logger.debug(f"Generated playlists: {list(playlists.keys())}")

            return playlists

        except Exception as e:
            logger.error(f"Error in tag-based playlist generation: {str(e)}")
            import traceback
            logger.error(
                f"Tag-based generation error traceback: {traceback.format_exc()}")
            return {}
