import argparse
import os
import sys
import json
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.console import Console
from playlist_generator.tag_based import TagBasedPlaylistGenerator
from database.db_manager import DatabaseManager
import logging

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Enrich music metadata in the database using MusicBrainz/Last.fm APIs.')
    parser.add_argument('--cache_dir', default='/app/cache', help='Cache directory containing audio_analysis.db')
    parser.add_argument('--db_file', default=None, help='Path to audio_analysis.db (overrides --cache_dir)')
    parser.add_argument('--enrich_tags', action='store_true', help='Enrich tags using MusicBrainz/Last.fm APIs for tracks missing genre or year (default: False)')
    args = parser.parse_args()

    if not args.enrich_tags:
        print("[ERROR] You must specify --enrich_tags to run enrichment.")
        sys.exit(1)

    cache_dir = args.cache_dir
    db_file = args.db_file or os.path.join(cache_dir, 'audio_analysis.db')

    if not os.path.exists(db_file):
        logger.debug(f"[ERROR] Database file not found: {db_file}")
        sys.exit(1)

    dbm = DatabaseManager(db_file)
    conn = dbm._get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT file_path, metadata FROM audio_features")
    rows = cursor.fetchall()
    total = len(rows)
    enriched = 0
    skipped = 0
    failed = 0
    tagger = TagBasedPlaylistGenerator(db_file=db_file, enrich_tags=True)
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        TextColumn("{task.fields[trackinfo]}", justify="right"),
        console=Console()
    )
    logger.debug(f"[INFO] Starting enrichment for {total} tracks in {db_file}")
    with progress:
        task_id = progress.add_task(f"Enriching 0/{total} tracks", total=total, trackinfo="")
        for i, row in enumerate(rows):
            filepath = row[0]
            try:
                meta = json.loads(row[1]) if row[1] else {}
                track_data = {'filepath': filepath, 'metadata': meta}
                before = dict(meta)
                needs_enrichment = not before.get('genre') or not before.get('year')
                if needs_enrichment:
                    result = tagger.enrich_track_metadata(track_data)
                    after = result.get('metadata', {})
                    filename = os.path.basename(filepath)
                    progress.update(
                        task_id,
                        advance=1,
                        description=f"Enriching {i+1}/{total} tracks",
                        trackinfo=f"{filename}"
                    )
                    if (not before.get('genre') and after.get('genre')) or (not before.get('year') and after.get('year')):
                        enriched += 1
                    else:
                        skipped += 1
                else:
                    skipped += 1
                    progress.update(
                        task_id,
                        advance=1,
                        description=f"Enriching {i+1}/{total} tracks",
                        trackinfo=f"{os.path.basename(filepath)} (skipped)"
                    )
            except Exception as e:
                failed += 1
                logger.debug(f"[red]Failed to enrich {filepath}: {e}")
    logger.debug(f"\nEnrichment complete. Total: {total}, Enriched: {enriched}, Skipped: {skipped}, Failed: {failed}")

if __name__ == "__main__":
    main() 