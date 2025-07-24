import musicbrainzngs
import requests
import logging

logger = logging.getLogger()

def enrich_with_musicbrainz(meta, artist, title):
    """
    Query MusicBrainz for genre, year, album. Return dict of updated fields.
    """
    result_meta = {}
    try:
        result = musicbrainzngs.search_recordings(artist=artist, recording=title, limit=1)
        if result['recording-list']:
            rec = result['recording-list'][0]
            if 'first-release-date' in rec:
                result_meta['year'] = rec['first-release-date'][:4]
            if 'tag-list' in rec and rec['tag-list']:
                result_meta['genre'] = [tag['name'] for tag in rec['tag-list']]
            if 'release-list' in rec and rec['release-list']:
                result_meta['album'] = rec['release-list'][0]['title']
            logger.info(f"Enriched metadata for {artist} - {title} from MusicBrainz.")
        else:
            logger.info(f"No MusicBrainz result for {artist} - {title}.")
    except Exception as e:
        logger.warning(f"MusicBrainz enrichment failed for {artist} - {title}: {e}")
    return result_meta

def enrich_with_lastfm(meta, artist, title, lastfm_api_key, genre_needed=True, year_needed=True):
    """
    Query Last.fm for genre and/or year. Return dict of updated fields.
    genre_needed: only fetch genre if True
    year_needed: only fetch year if True
    """
    result_meta = {}
    try:
        url = (
            "http://ws.audioscrobbler.com/2.0/"
            "?method=track.getInfo"
            f"&api_key={lastfm_api_key}"
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
                if genre_needed:
                    tags = track_info.get('toptags', {}).get('tag', [])
                    if tags:
                        result_meta['genre'] = [t['name'] for t in tags if 'name' in t]
                # Get album
                if 'album' in track_info and 'title' in track_info['album']:
                    result_meta['album'] = track_info['album']['title']
                # Get year (if available)
                if year_needed and 'wiki' in track_info and 'published' in track_info['wiki']:
                    import re
                    match = re.search(r'(\\d{4})', track_info['wiki']['published'])
                    if match:
                        result_meta['year'] = match.group(1)
        else:
            logger.info(f"Last.fm API returned status {resp.status_code} for {artist} - {title}.")
    except Exception as e:
        logger.warning(f"Last.fm enrichment failed for {artist} - {title}: {e}")
    return result_meta 