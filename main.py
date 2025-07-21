#!/usr/bin/env python3
import argparse
import os
import logging
from colorlog import ColoredFormatter
import multiprocessing as mp
import time
import traceback
import sys
from audio_analysis import AudioAnalyzer, get_all_features
from playlist_generation import PlaylistGenerator

def setup_colored_logging():
    """Configure colored logging"""
    formatter = ColoredFormatter(
        "%(log_color)s%(asctime)s %(levelname)-8s%(reset)s %(message)s",
        datefmt="%H:%M:%S",
        reset=True,
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    return logger

def main():
    logger = setup_colored_logging()
    parser = argparse.ArgumentParser(description='Music Playlist Generator')
    parser.add_argument('--music_dir', required=True, help='Music directory in container')
    parser.add_argument('--host_music_dir', required=True, help='Host music directory')
    parser.add_argument('--output_dir', default='./playlists', help='Output directory')
    parser.add_argument('--workers', type=int, default=None, help='Number of workers')
    parser.add_argument('--use_db', action='store_true', help='Use database only')
    parser.add_argument('--force_sequential', action='store_true', help='Force sequential processing')
    parser.add_argument('--update', action='store_true', help='Update existing playlists')
    parser.add_argument('--analyze_only', action='store_true', help='Only run audio analysis')
    parser.add_argument('--generate_only', action='store_true', help='Only generate playlists')
    args = parser.parse_args()

    generator = PlaylistGenerator()
    generator.host_music_dir = args.host_music_dir.rstrip('/')

    start_time = time.time()
    try:
        # Database cleanup
        missing_in_db = generator.cleanup_database()
        if missing_in_db:
            logger.info(f"Removed {len(missing_in_db)} missing files from database")
            generator.failed_files.extend(missing_in_db)

        # Analysis only mode
        if args.analyze_only:
            logger.info("Running audio analysis only")
            features = generator.analyze_directory(
                args.music_dir,
                args.workers,
                args.force_sequential
            )
            logger.info(f"Analysis completed. Processed {len(features)} files")
            return

        # Playlist generation modes
        if args.update:
            logger.info("Updating playlists from database")
            generator.update_playlists()
            playlists = generator.generate_playlists_from_db()
        elif args.generate_only:
            logger.info("Generating playlists from database")
            features_from_db = get_all_features()
            time_playlists = generator.generate_time_based_playlists(features_from_db)
            db_playlists = generator.generate_playlists_from_db()
            playlists = {**db_playlists, **time_playlists}
        else:
            logger.info("Analyzing directory and generating playlists")
            features = generator.analyze_directory(
                args.music_dir,
                args.workers,
                args.force_sequential
            )
            time_playlists = generator.generate_time_based_playlists(features)
            db_playlists = generator.generate_playlists_from_db()
            playlists = {**db_playlists, **time_playlists}
        
        # Save results
        generator.save_playlists(playlists, args.output_dir)
        logger.info(f"Generated {len(playlists)} playlists")

    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        logger.error(traceback.format_exc())
        raise
    finally:
        elapsed = time.time() - start_time
        logger.info(f"Completed in {elapsed:.2f} seconds")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()