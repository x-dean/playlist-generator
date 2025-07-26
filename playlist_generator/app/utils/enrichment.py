import musicbrainzngs
import requests
import logging
import os
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Configure MusicBrainz
musicbrainzngs.set_useragent(
    "playlist-generator", "1.0", "https://github.com/your-repo")


def enrich_musicbrainz(artist: str, title: str) -> Optional[Dict[str, Any]]:
    """Enrich track metadata using MusicBrainz API."""
    logger.debug(f"Starting MusicBrainz enrichment for: {artist} - {title}")

    try:
        # Search for the recording
        logger.debug(f"Searching MusicBrainz for: {artist} - {title}")
        result = musicbrainzngs.search_recordings(
            artist=artist, title=title, limit=1)

        if result and 'recording-list' in result and result['recording-list']:
            recording = result['recording-list'][0]
            logger.debug(
                f"Found MusicBrainz recording: {recording.get('id', 'unknown')}")

            # Extract relevant metadata
            metadata = {}

            # Basic info
            if 'title' in recording:
                metadata['title'] = recording['title']
                logger.debug(f"Extracted title: {metadata['title']}")

            if 'artist-credit' in recording:
                artists = []
                for artist_credit in recording['artist-credit']:
                    if 'artist' in artist_credit:
                        artists.append(artist_credit['artist']['name'])
                if artists:
                    metadata['artist'] = ', '.join(artists)
                    logger.debug(f"Extracted artists: {metadata['artist']}")

            # Release info
            if 'release-list' in recording and recording['release-list']:
                release = recording['release-list'][0]
                logger.debug(f"Found release: {release.get('id', 'unknown')}")

                if 'title' in release:
                    metadata['album'] = release['title']
                    logger.debug(f"Extracted album: {metadata['album']}")

                if 'date' in release:
                    metadata['year'] = release['date'][:4]  # Extract year
                    logger.debug(f"Extracted year: {metadata['year']}")

                # Label info
                if 'label-info-list' in release and release['label-info-list']:
                    label_info = release['label-info-list'][0]
                    if 'label' in label_info and 'name' in label_info['label']:
                        metadata['label'] = label_info['label']['name']
                        logger.debug(f"Extracted label: {metadata['label']}")

            # Genre info (if available)
            if 'tag-list' in recording:
                genres = []
                for tag in recording['tag-list']:
                    if tag.get('count', 0) > 0:  # Only include tags with votes
                        genres.append(tag['name'])
                if genres:
                    metadata['genre'] = ', '.join(
                        genres[:3])  # Limit to top 3 genres
                    logger.debug(f"Extracted genres: {metadata['genre']}")

            logger.info(
                f"Enriched metadata for {artist} - {title} from MusicBrainz.")
            logger.debug(f"Extracted metadata: {metadata}")
            return metadata
        else:
            logger.info(f"No MusicBrainz result for {artist} - {title}.")
            return None

    except Exception as e:
        logger.warning(
            f"MusicBrainz enrichment failed for {artist} - {title}: {e}")
        import traceback
        logger.debug(f"MusicBrainz error traceback: {traceback.format_exc()}")
        return None


def enrich_lastfm(artist: str, title: str) -> Optional[Dict[str, Any]]:
    """Enrich track metadata using Last.fm API."""
    logger.debug(f"Starting Last.fm enrichment for: {artist} - {title}")

    api_key = os.getenv('LASTFM_API_KEY')
    if not api_key:
        logger.debug("LASTFM_API_KEY not set, skipping Last.fm enrichment")
        return None

    try:
        # Search for track info
        url = "http://ws.audioscrobbler.com/2.0/"
        params = {
            'method': 'track.getInfo',
            'api_key': api_key,
            'artist': artist,
            'track': title,
            'format': 'json'
        }

        logger.debug(f"Making Last.fm API request for: {artist} - {title}")
        response = requests.get(url, params=params, timeout=10)

        if response.status_code == 200:
            data = response.json()
            logger.debug(
                f"Last.fm API returned status {response.status_code} for {artist} - {title}")

            if 'track' in data:
                track = data['track']
                metadata = {}

                # Basic info
                if 'name' in track:
                    metadata['title'] = track['name']
                    logger.debug(
                        f"Extracted Last.fm title: {metadata['title']}")

                if 'artist' in track and 'name' in track['artist']:
                    metadata['artist'] = track['artist']['name']
                    logger.debug(
                        f"Extracted Last.fm artist: {metadata['artist']}")

                # Album info
                if 'album' in track and 'title' in track['album']:
                    metadata['album'] = track['album']['title']
                    logger.debug(
                        f"Extracted Last.fm album: {metadata['album']}")

                # Tags/genres
                if 'toptags' in track and 'tag' in track['toptags']:
                    tags = []
                    for tag in track['toptags']['tag'][:5]:  # Top 5 tags
                        if tag.get('count', 0) > 0:
                            tags.append(tag['name'])
                    if tags:
                        metadata['genre'] = ', '.join(tags)
                        logger.debug(
                            f"Extracted Last.fm genres: {metadata['genre']}")

                # Year
                if 'wiki' in track and 'published' in track['wiki']:
                    published = track['wiki']['published']
                    if published:
                        try:
                            # Extract year from date
                            year = published.split(' ')[-1][:4]
                            if year.isdigit():
                                metadata['year'] = year
                                logger.debug(
                                    f"Extracted Last.fm year: {metadata['year']}")
                        except (IndexError, ValueError):
                            pass

                logger.info(
                    f"Enriched metadata for {artist} - {title} from Last.fm.")
                logger.debug(f"Extracted Last.fm metadata: {metadata}")
                return metadata
            else:
                logger.debug(
                    f"No track data found in Last.fm response for {artist} - {title}")
                return None
        else:
            logger.warning(
                f"Last.fm API returned status {response.status_code} for {artist} - {title}")
            return None

    except requests.exceptions.RequestException as e:
        logger.warning(
            f"Last.fm enrichment failed for {artist} - {title}: {e}")
        import traceback
        logger.debug(
            f"Last.fm request error traceback: {traceback.format_exc()}")
        return None
    except Exception as e:
        logger.warning(
            f"Last.fm enrichment failed for {artist} - {title}: {e}")
        import traceback
        logger.debug(f"Last.fm error traceback: {traceback.format_exc()}")
        return None


def enrich_metadata(artist: str, title: str) -> Dict[str, Any]:
    """Enrich track metadata using both MusicBrainz and Last.fm."""
    logger.debug(f"Starting metadata enrichment for: {artist} - {title}")

    enriched_data = {}

    # Try MusicBrainz first
    mb_data = enrich_musicbrainz(artist, title)
    if mb_data:
        enriched_data.update(mb_data)
        logger.debug(
            f"MusicBrainz enrichment successful, got {len(mb_data)} fields")

    # Try Last.fm as fallback or supplement
    lfm_data = enrich_lastfm(artist, title)
    if lfm_data:
        # Only add fields that weren't already found by MusicBrainz
        for key, value in lfm_data.items():
            if key not in enriched_data or not enriched_data[key]:
                enriched_data[key] = value
                logger.debug(f"Added Last.fm field: {key} = {value}")

    logger.debug(
        f"Metadata enrichment complete for {artist} - {title}: {len(enriched_data)} fields")
    return enriched_data


def merge_metadata(existing: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    """Merge existing metadata with new enriched data."""
    logger.debug(
        f"Merging metadata: existing={len(existing)} fields, new={len(new)} fields")

    merged = existing.copy()

    for key, value in new.items():
        if key not in merged or not merged[key]:
            merged[key] = value
            logger.debug(f"Merged field: {key} = {value}")
        elif merged[key] != value:
            logger.debug(
                f"Field {key} already exists, keeping existing value: {merged[key]}")

    logger.debug(f"Metadata merge complete: {len(merged)} total fields")
    return merged
