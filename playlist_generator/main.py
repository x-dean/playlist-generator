import argparse
import os
import sys
import time
import traceback
import multiprocessing as mp
from logging_setup import setup_colored_logging
from music_analyzer.audio_analyzer import AudioAnalyzer
from database.playlist_db import PlaylistDatabase
from music_analyzer.parallel import ParallelProcessor
from music_analyzer.sequential import SequentialProcessor
from playlist_generator.time_based import TimeBasedScheduler
from playlist_generator.kmeans import KMeansPlaylistGenerator
from playlist_generator.cache import CacheBasedGenerator

logger = setup_colored_logging()

def get_audio_files(music_dir):
    file_list = []
    valid_ext = ('.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac', '.opus')
    
    for root, _, files in os.walk(music_dir):
        for file in files:
            file_lower = file.lower()
            if file_lower.endswith(valid_ext):
                file_list.append(os.path.join(root, file))
    
    logger.info(f"Found {len(file_list)} audio files in {music_dir}")
    return file_list

def convert_to_host_path(container_path, host_music_dir, container_music_dir):
    container_path = os.path.normpath(container_path)
    container_music_dir = os.path.normpath(container_music_dir)
    
    if not container_path.startswith(container_music_dir):
        return container_path
    
    rel_path = os.path.relpath(container_path, container_music_dir)
    return os.path.join(host_music_dir, rel_path)

def save_playlists(playlists, output_dir, host_music_dir, container_music_dir, failed_files):
    os.makedirs(output_dir, exist_ok=True)
    saved_count = 0

    for name, playlist_data in playlists.items():
        if 'tracks' not in playlist_data or not playlist_data['tracks']:
            continue
            
        host_songs = [
            convert_to_host_path(song, host_music_dir, container_music_dir) 
            for song in playlist_data['tracks']
        ]
        
        playlist_path = os.path.join(output_dir, f"{name}.m3u")
        with open(playlist_path, 'w') as f:
            f.write("\n".join(host_songs))
        
        saved_count += 1
        logger.info(f"Saved {name} with {len(host_songs)} tracks")

    # Save failed files
    if failed_files:
        failed_path = os.path.join(output_dir, "Failed_Files.m3u")
        with open(failed_path, 'w') as f:
            host_failed = [
                convert_to_host_path(p, host_music_dir, container_music_dir) 
                for p in failed_files
            ]
            f.write("\n".join(host_failed))
        logger.info(f"Saved {len(failed_files)} failed files")

def main():
    parser = argparse.ArgumentParser(description='Music Playlist Generator')
    parser.add_argument('--music_dir', required=True, help='Music directory in container')
    parser.add_argument('--host_music_dir', required=True, help='Host music directory')
    parser.add_argument('--output_dir', default='./playlists', help='Output directory')
    parser.add_argument('--num_playlists', type=int, default=8, help='Number of playlists')
    parser.add_argument('--workers', type=int, default=None, help='Number of workers (default: auto)')
    parser.add_argument('--force_sequential', action='store_true', help='Force sequential processing')
    parser.add_argument('--update', action='store_true', help='Update existing playlists')
    parser.add_argument('--analyze_only', action='store_true', help='Only run audio analysis without generating playlists')
    parser.add_argument('--generate_only', action='store_true', help='Only generate playlists from database without analysis')
    args = parser.parse_args()

    # Set cache file path
    cache_dir = os.getenv('CACHE_DIR', '/app/cache')
    os.environ['CACHE_DIR'] = cache_dir
    cache_file = os.path.join(cache_dir, 'audio_analysis.db')
    
    # Initialize components
    audio_db = AudioAnalyzer(cache_file)
    playlist_db = PlaylistDatabase(cache_file)
    time_scheduler = TimeBasedScheduler()
    kmeans_generator = KMeansPlaylistGenerator()
    cache_generator = CacheBasedGenerator(cache_file)
    
    container_music_dir = args.music_dir.rstrip('/')
    host_music_dir = args.host_music_dir.rstrip('/')
    failed_files = []

    if not os.path.exists(args.music_dir):
        logger.error(f"Music directory not found: {args.music_dir}")
        sys.exit(1)

    start_time = time.time()
    try:
        # Clean up database
        missing_in_db = audio_db.cleanup_database()
        if missing_in_db:
            logger.info(f"Removed {len(missing_in_db)} missing files from database")
            failed_files.extend(missing_in_db)

            if args.update:
                logger.info("Running in UPDATE mode")
                if not playlist_db.playlists_exist():
                    logger.info("No playlists found, running full analysis")
                    # Force full analysis and generation
                    file_list = get_audio_files(args.music_dir)
                    processor = ParallelProcessor()
                    processor.process(file_list, workers=args.workers or mp.cpu_count())
                
                # Always get ALL features
                features_from_db = audio_db.get_all_features()
                
                # Regenerate ALL playlists
                time_playlists = time_scheduler.generate_time_based_playlists(features_from_db)
                cache_playlists = cache_generator.generate(features_from_db)
                kmeans_playlists = kmeans_generator.generate(features_from_db, args.num_playlists)
                all_playlists = {**time_playlists, **cache_playlists, **kmeans_playlists}
                
                # Save to DB and files
                playlist_db.save_playlists(all_playlists)
                save_playlists(all_playlists, args.output_dir, host_music_dir, container_music_dir, failed_files)
            else:
                changed_files = playlist_db.get_changed_files()
                if changed_files:
                    logger.info(f"Updating all playlists for {len(changed_files)} changed files")
                    features_from_db = audio_db.get_all_features()
                    time_playlists = time_scheduler.generate_time_based_playlists(features_from_db)
                    cache_playlists = cache_generator.generate(features_from_db)  # Pass features
                    kmeans_playlists = kmeans_generator.generate(features_from_db, args.num_playlists)
                    all_playlists = {**time_playlists, **cache_playlists, **kmeans_playlists}
                    playlist_db.save_playlists(all_playlists)
                    save_playlists(all_playlists, args.output_dir, host_music_dir, container_music_dir, failed_files)
                else:
                    logger.info("No changed files, playlists remain up-to-date")
        
        elif args.analyze_only:
            logger.info("Running audio analysis only")
            file_list = get_audio_files(args.music_dir)
            
            if args.force_sequential or (args.workers and args.workers <= 1):
                processor = SequentialProcessor()
                features = processor.process(file_list)
                failed_files.extend(processor.failed_files)
            else:
                processor = ParallelProcessor()
                features = processor.process(
                    file_list, 
                    workers=args.workers or max(1, mp.cpu_count() // 2)
                )
                failed_files.extend(processor.failed_files)
            
            logger.info(f"Analysis completed. Processed {len(features)} files, {len(failed_files)} failed")
        
        elif args.generate_only:
            logger.info("Generating playlists from database")
            features_from_db = audio_db.get_all_features()
            
            time_playlists = time_scheduler.generate_time_based_playlists(features_from_db)
            cache_playlists = cache_generator.generate()
            kmeans_playlists = kmeans_generator.generate(features_from_db, args.num_playlists)  # ADD KMEANS
            all_playlists = {**time_playlists, **cache_playlists, **kmeans_playlists}  # COMBINE ALL
            
            save_playlists(all_playlists, args.output_dir, host_music_dir, container_music_dir, failed_files)
        
        else:
            logger.info("Full processing pipeline")
            file_list = get_audio_files(args.music_dir)
            
            # Analyze files
            processor = ParallelProcessor() if not args.force_sequential else SequentialProcessor()
            workers = args.workers or max(1, mp.cpu_count())
            analysis_results = processor.process(file_list, workers)
            failed_files.extend(processor.failed_files)
            
            # Get ALL features from DB (not just current run)
            all_features = audio_db.get_all_features()
            
            # Generate playlists using ALL features
            time_playlists = time_scheduler.generate_time_based_playlists(all_features)
            cache_playlists = cache_generator.generate(all_features)  # Pass features explicitly
            kmeans_playlists = kmeans_generator.generate(all_features, args.num_playlists)
            
            all_playlists = {**time_playlists, **cache_playlists, **kmeans_playlists}
            save_playlists(all_playlists, args.output_dir, host_music_dir, container_music_dir, failed_files)
    
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)
    finally:
        elapsed = time.time() - start_time
        logger.info(f"Completed in {elapsed:.2f} seconds")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()