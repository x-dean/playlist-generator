import logging
from collections import defaultdict, Counter
import musicbrainzngs
import os
import requests

logger = logging.getLogger(__name__)

class TagBasedPlaylistGenerator:
    def __init__(self, min_tracks_per_genre=10, min_subgroup_size=10, large_group_threshold=40, enrich_tags=False, db_file=None):
        self.min_tracks_per_genre = min_tracks_per_genre
        self.min_subgroup_size = min_subgroup_size
        self.large_group_threshold = large_group_threshold
        self.enrich_tags = enrich_tags
        self.db_file = db_file
        # Set MusicBrainz user agent once
        musicbrainzngs.set_useragent("PlaylistGenerator", "1.0", "noreply@example.com")
        self.lastfm_api_key = os.getenv("LASTFM_API_KEY")

    def _genre_needs_enrichment(self, genre):
        if not genre:
            return True
        if isinstance(genre, list):
            genre = genre[0] if genre else ''
        genre = str(genre).strip().lower()
        return genre in ('', 'other', 'unknowngenre', 'unknown', 'misc', 'various', 'n/a', 'none')

    def enrich_track_metadata(self, track):
        """
        Enrich track metadata using MusicBrainz and Last.fm.
        Fills in missing or placeholder genres, years, etc. if possible.
        Updates the database if enrichment occurs.
        """
        meta = track.get('metadata', {})
        artist = meta.get('artist')
        title = meta.get('title')
        genre = meta.get('genre')
        year = meta.get('year')
        genre_needs = self._genre_needs_enrichment(genre)
        year_needs = not year or str(year).strip() == ''
        enriched = False
        # Try MusicBrainz first if needed
        if (genre_needs or year_needs) and artist and title:
            try:
                result = musicbrainzngs.search_recordings(artist=artist, recording=title, limit=1)
                if result['recording-list']:
                    rec = result['recording-list'][0]
                    if year_needs and 'first-release-date' in rec:
                        meta['year'] = rec['first-release-date'][:4]
                        year_needs = False if meta['year'] else True
                        enriched = True
                    if genre_needs and 'tag-list' in rec and rec['tag-list']:
                        mb_genres = [tag['name'] for tag in rec['tag-list']]
                        if mb_genres and not self._genre_needs_enrichment(mb_genres):
                            meta['genre'] = mb_genres
                            genre_needs = False
                            enriched = True
                    if 'release-list' in rec and rec['release-list']:
                        meta['album'] = rec['release-list'][0]['title']
                    track['metadata'] = meta
                    logger.info(f"Enriched metadata for {artist} - {title} from MusicBrainz.")
                else:
                    logger.info(f"No MusicBrainz result for {artist} - {title}.")
            except Exception as e:
                logger.warning(f"MusicBrainz enrichment failed for {artist} - {title}: {e}")
        # If still missing genre or year, try Last.fm
        if (genre_needs or year_needs) and artist and title and self.lastfm_api_key:
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
                        # Get tags (genre)
                        if genre_needs:
                            tags = track_info.get('toptags', {}).get('tag', [])
                            if tags:
                                lastfm_genres = [t['name'] for t in tags if 'name' in t]
                                if lastfm_genres and not self._genre_needs_enrichment(lastfm_genres):
                                    meta['genre'] = lastfm_genres
                                    genre_needs = False
                                    enriched = True
                        # Get album
                        if 'album' in track_info and 'title' in track_info['album']:
                            meta['album'] = track_info['album']['title']
                        # Get year (if available)
                        if year_needs and 'wiki' in track_info and 'published' in track_info['wiki']:
                            import re
                            match = re.search(r'(\\d{4})', track_info['wiki']['published'])
                            if match:
                                meta['year'] = match.group(1)
                                year_needs = False
                                enriched = True
                        track['metadata'] = meta
                        logger.info(f"Enriched metadata for {artist} - {title} from Last.fm.")
                else:
                    logger.info(f"Last.fm API returned status {resp.status_code} for {artist} - {title}.")
            except Exception as e:
                logger.warning(f"Last.fm enrichment failed for {artist} - {title}: {e}")
        elif (genre_needs or year_needs) and not self.lastfm_api_key:
            logger.debug("LASTFM_API_KEY not set; skipping Last.fm enrichment.")
        # Update database if enrichment occurred
        if self.db_file and enriched:
            self.update_track_metadata_in_db(track.get('filepath'), meta)
        return track

    def update_track_metadata_in_db(self, filepath, metadata):
        """
        Update the metadata column for the given file in the audio_features table.
        """
        if not self.db_file or not filepath:
            return
        import sqlite3
        try:
            conn = sqlite3.connect(self.db_file, timeout=30)
            cur = conn.cursor()
            import json
            cur.execute(
                "UPDATE audio_features SET metadata = ? WHERE file_path = ?",
                (json.dumps(metadata), filepath)
            )
            conn.commit()
            conn.close()
            logger.info(f"Updated metadata in DB for {filepath}")
        except Exception as e:
            logger.warning(f"Failed to update metadata in DB for {filepath}: {e}")

    def group_by_genre(self, features_list):
        """
        Group tracks by each genre. Tracks with multiple genres appear in multiple playlists.
        Returns a dict: {genre_display_name: {'tracks': [...], 'features': {...}}}
        """
        genre_playlists = defaultdict(list)
        for track in features_list:
            meta = track.get('metadata', {})
            genres = meta.get('genre', 'UnknownGenre')
            if isinstance(genres, str):
                genres = [genres]
            norm_genres = [self._normalize_genre(g) for g in genres]
            for genre in norm_genres:
                genre_playlists[genre].append(track)
        result = {}
        for genre, tracks in genre_playlists.items():
            display_name = self._format_display_name(genre, None)
            file_name = self._format_file_name(genre, None)
            result[display_name] = {
                'tracks': [t['filepath'] for t in tracks],
                'features': {
                    'genre': genre,
                    'file_name': file_name
                }
            }
        return result

    def group_by_decade(self, features_list):
        """
        Group tracks by decade (or year if you want more granularity).
        Returns a dict: {decade_display_name: {'tracks': [...], 'features': {...}}}
        """
        decade_playlists = defaultdict(list)
        for track in features_list:
            meta = track.get('metadata', {})
            year = meta.get('date') or meta.get('year')
            decade = self._get_decade(year)
            decade_playlists[decade].append(track)
        result = {}
        for decade, tracks in decade_playlists.items():
            display_name = self._format_display_name(None, decade)
            file_name = self._format_file_name(None, decade)
            result[display_name] = {
                'tracks': [t['filepath'] for t in tracks],
                'features': {
                    'decade': decade,
                    'file_name': file_name
                }
            }
        return result

    def _normalize_genre(self, genre):
        # Replace underscores/hyphens with spaces, title case, strip
        if not genre:
            return "UnknownGenre"
        return genre.replace('_', ' ').replace('-', ' ').title().strip()

    def _get_decade(self, year):
        if not year or not str(year).isdigit() or len(str(year)) < 4:
            return "All Eras"
        y = str(year)
        if y == "0000":
            return "All Eras"
        return f"{y[:3]}0s"

    def _get_mood(self, track):
        # Use danceability, bpm, centroid for a simple mood/energy label
        dance = float(track.get('danceability', 0) or 0)
        bpm = float(track.get('bpm', 0) or 0)
        centroid = float(track.get('centroid', 0) or 0)
        # Simple rules for mood/energy
        if dance < 0.4 and bpm < 90:
            return "Chill"
        elif dance > 0.7 and bpm > 120:
            return "Energetic"
        elif centroid > 4000:
            return "Bright"
        elif bpm > 110:
            return "Upbeat"
        else:
            return "Balanced"

    def _format_display_name(self, genre, decade, mood=None):
        genre_disp = self._normalize_genre(genre)
        if not decade or decade in ("UnknownDecade", "0000s", "All Eras"):
            decade_disp = "All Eras"
        else:
            decade_disp = decade
        if mood:
            return f"{genre_disp} ({decade_disp}) - {mood}"
        else:
            return f"{genre_disp} ({decade_disp})"

    def _format_file_name(self, genre, decade, mood=None, part=None):
        import re
        genre_file = self._normalize_genre(genre).replace(' ', '_') if genre else ''
        if not decade or decade in ("UnknownDecade", "0000s", "All Eras"):
            decade_file = "All_Eras"
        else:
            decade_file = decade
        name = f"{genre_file}_{decade_file}"
        if mood:
            name += f"_{mood}"
        if part:
            name += f"_Part{part}"
        # Replace any non-alphanumeric character with underscore
        name = re.sub(r'[^A-Za-z0-9_-]+', '_', name)
        # Collapse multiple underscores
        name = re.sub(r'_+', '_', name)
        # Remove leading/trailing underscores
        name = name.strip('_')
        return name

    def generate(self, features_list, enrich_tags=None):
        # Only enrich if enabled (from argument or self)
        do_enrich = self.enrich_tags if enrich_tags is None else enrich_tags
        if do_enrich:
            for track in features_list:
                self.enrich_track_metadata(track)
        # Normalize genres and count
        genre_counter = Counter()
        track_genres = []
        for track in features_list:
            meta = track.get('metadata', {})
            genres = meta.get('genre', 'UnknownGenre')
            if isinstance(genres, str):
                genres = [genres]
            norm_genres = [self._normalize_genre(g) for g in genres]
            track_genres.append((track, norm_genres))
            genre_counter.update(norm_genres)
        # Only keep genres with enough tracks
        valid_genres = {g for g, count in genre_counter.items() if count >= self.min_tracks_per_genre}
        # Group by genre and decade
        genre_decade_groups = defaultdict(list)
        for track, genres in track_genres:
            meta = track.get('metadata', {})
            year = meta.get('date') or meta.get('year')
            decade = self._get_decade(year)
            for genre in genres:
                if genre in valid_genres:
                    genre_decade_groups[(genre, decade)].append(track)
        # For each genre+decade group, further split by mood/tempo if large
        playlists = {}
        for (genre, decade), tracks in genre_decade_groups.items():
            if len(tracks) >= self.large_group_threshold:
                # Subgroup by mood
                mood_groups = defaultdict(list)
                for t in tracks:
                    mood = self._get_mood(t)
                    mood_groups[mood].append(t)
                for mood, mood_tracks in mood_groups.items():
                    if len(mood_tracks) < self.min_subgroup_size:
                        # Merge small subgroups into a "Mixed" for this genre+decade
                        mixed_key = (genre, decade, "Mixed")
                        if mixed_key not in playlists:
                            playlists[mixed_key] = []
                        playlists[mixed_key].extend(mood_tracks)
                    else:
                        playlists[(genre, decade, mood)] = mood_tracks
            else:
                playlists[(genre, decade, None)] = tracks
        # Convert to expected playlist dict format
        result = {}
        for key, tracks in playlists.items():
            genre, decade, mood = key
            display_name = self._format_display_name(genre, decade, mood)
            file_name = self._format_file_name(genre, decade, mood)
            result[display_name] = {
                'tracks': [t['filepath'] for t in tracks],
                'features': {
                    'genre': genre,
                    'decade': decade,
                    'mood': mood,
                    'file_name': file_name
                }
            }
        return result 