#!/usr/bin/env python3
"""
Application runner for the playlist generator.
"""

import os
import sys
import time
import logging
import multiprocessing as mp
import traceback
from typing import List, Dict, Any, Optional
from datetime import datetime

from rich.console import Console
from rich.panel import Panel

from utils.cli import PlaylistGeneratorCLI, CLIContextManager
from playlist_generator.playlist_manager import PlaylistManager
from music_analyzer.analysis_manager import run_pipeline, run_analysis
from database.playlist_db import PlaylistDatabase
from music_analyzer.audio_analyzer import AudioAnalyzer
from utils.path_converter import PathConverter
from music_analyzer.file_discovery import FileDiscovery
from signal_handlers import cleanup_child_processes, handle_keyboard_interrupt


logger = logging.getLogger(__name__)


def get_audio_files(music_dir: str) -> List[str]:
    """Get audio files using the FileDiscovery module.

    Args:
        music_dir (str): Path to the music directory.

    Returns:
        List[str]: List of audio file paths.
    """
    # Use FileDiscovery to get all valid audio files
    file_discovery = FileDiscovery(music_dir=music_dir)
    file_list = file_discovery.discover_files()

    logger.info(f"Found {len(file_list)} audio files in {music_dir}")
    return file_list


def save_playlists(playlists: Dict[str, Dict[str, Any]], output_dir: str, library: str, music: str, 
                   failed_files: List[str], playlist_method: Optional[str] = None, host_library: str = None) -> None:
    """Saves generated playlists to disk.

    Args:
        playlists (Dict[str, Dict[str, Any]]): Dictionary of playlist names to their data.
        output_dir (str): Root output directory.
        library (str): Path to the host's music directory.
        music (str): Path to the container's music directory.
        failed_files (List[str]): List of failed file paths.
        playlist_method (str | None): The method used to generate playlists (e.g., 'time', 'kmeans').
        host_library (str | None): Host library path for path conversion.
    """
    # For time-based, create subfolders per slot
    def get_time_slot_from_name(name: str) -> Optional[str]:
        if name.startswith("TimeSlot_"):
            slot_part = name[len("TimeSlot_"):]
            if "_Part" in slot_part:
                slot = slot_part.split("_Part")[0]
            else:
                slot = slot_part
            return slot
        return None

    # Initialize path converter
    path_converter = PathConverter(host_library or library, '/music')

    saved_count = 0
    for name, playlist_data in playlists.items():
        if 'tracks' not in playlist_data or not playlist_data['tracks']:
            continue

        # Determine output path
        playlist_out_dir = output_dir
        if playlist_method == 'time':
            slot = get_time_slot_from_name(name)
            if slot:
                playlist_out_dir = os.path.join(output_dir, slot)
        os.makedirs(playlist_out_dir, exist_ok=True)

        # Convert container paths to host paths using the path converter
        host_songs = path_converter.convert_playlist_tracks(
            playlist_data['tracks'])
        playlist_path = os.path.join(playlist_out_dir, f"{name}.m3u")
        with open(playlist_path, 'w') as f:
            f.write("\n".join(host_songs))
        saved_count += 1
        logger.debug(
            f"Saved {name} with {len(host_songs)} tracks to {playlist_path}")

    # Save failed files (keep at root of method dir)
    os.makedirs(output_dir, exist_ok=True)
    if failed_files:
        failed_path = os.path.join(output_dir, "Failed_Files.m3u")
        with open(failed_path, 'w') as f:
            host_failed = path_converter.convert_failed_files(failed_files)
            f.write("\n".join(host_failed))
        logger.info(f"Saved {len(failed_files)} failed files to {failed_path}")


def initialize_databases(cache_file: str, library: str) -> tuple:
    """Initialize database connections.
    
    Args:
        cache_file (str): Path to the cache database file.
        library (str): Host library path.
        
    Returns:
        tuple: (playlist_db, audio_db) database instances.
    """
    logger.debug("Initializing database connections")
    playlist_db = PlaylistDatabase(cache_file)
    # AudioAnalyzer needs host library path for database storage, container music path for conversion
    audio_db = AudioAnalyzer(cache_file, library=library, music='/music')
    logger.debug("Database connections initialized")
    return playlist_db, audio_db


def initialize_playlist_manager(cache_file: str, playlist_method: str, min_tracks_per_genre: int) -> PlaylistManager:
    """Initialize playlist manager.
    
    Args:
        cache_file (str): Path to the cache database file.
        playlist_method (str): Playlist generation method.
        min_tracks_per_genre (int): Minimum tracks per genre for tag-based playlists.
        
    Returns:
        PlaylistManager: Initialized playlist manager.
    """
    logger.info(f"Initializing playlist manager with method: {playlist_method}")
    if playlist_method == 'tags':
        playlist_manager = PlaylistManager(
            cache_file, playlist_method, min_tracks_per_genre=min_tracks_per_genre)
        logger.debug(
            f"Using tag-based playlist generation with min_tracks_per_genre: {min_tracks_per_genre}")
    else:
        playlist_manager = PlaylistManager(cache_file, playlist_method)
    
    return playlist_manager


def run_update_mode(args, audio_db, playlist_db, cli, playlist_manager):
    """Run update mode - regenerate all playlists from database."""
    logger.info("Starting UPDATE mode - regenerating all playlists from database")
    cli.update_status("Running in UPDATE mode")
    features_from_db = audio_db.get_all_features()
    logger.info(f"Retrieved {len(features_from_db)} tracks from database for playlist generation")

    with CLIContextManager(cli, 3, "[green]Generating playlists...") as (progress, task_id):
        logger.debug("Generating playlists from database features")
        all_playlists = playlist_manager.generate_playlists(
            features_from_db, args.num_playlists)
        progress.update(task_id, advance=3)
        logger.info(f"Generated {len(all_playlists)} playlists")

    # Show playlist statistics
    logger.debug("Calculating and displaying playlist statistics")
    cli.show_playlist_stats(playlist_manager.get_playlist_stats())

    # Save playlists to DB and to disk
    logger.info("Saving playlists to database")
    playlist_db.save_playlists(all_playlists)
    if args.playlist_method == 'tags':
        method_dir = os.path.join(args.output_dir, 'tags')
    else:
        method_dir = os.path.join(args.output_dir, args.playlist_method)
    logger.info(f"Saving playlists to disk: {method_dir}")
    save_playlists(all_playlists, method_dir, args.library, '/music', [],
                   playlist_method=args.playlist_method, host_library=args.library)
    logger.info("UPDATE mode completed successfully")


def run_generate_only_mode(args, audio_db, cli, playlist_manager):
    """Run generate only mode - create playlists from existing analysis."""
    logger.info("Starting GENERATE_ONLY mode - creating playlists from existing analysis")
    cli.update_status("Generating playlists from database")
    features_from_db = audio_db.get_all_features()
    logger.info(f"Retrieved {len(features_from_db)} tracks from database for playlist generation")

    with CLIContextManager(cli, 3, "[green]Generating playlists...") as (progress, task_id):
        logger.debug("Generating playlists from database features")
        all_playlists = playlist_manager.generate_playlists(
            features_from_db, args.num_playlists)
        progress.update(task_id, advance=3)
        logger.info(f"Generated {len(all_playlists)} playlists")

    # Show playlist statistics
    logger.debug("Calculating and displaying playlist statistics")
    cli.show_playlist_stats(playlist_manager.get_playlist_stats())

    # Save playlists
    if args.playlist_method == 'tags':
        method_dir = os.path.join(args.output_dir, 'tags')
    else:
        method_dir = os.path.join(args.output_dir, args.playlist_method)
    logger.info(f"Saving playlists to disk: {method_dir}")
    save_playlists(all_playlists, method_dir, args.library, '/music', [],
                   playlist_method=args.playlist_method, host_library=args.library)
    logger.info("GENERATE_ONLY mode completed successfully")


def run_analysis_mode(args, audio_db, playlist_db, cli):
    """Run analysis mode - analyze audio files."""
    logger.info(f"Starting ANALYSIS mode: analyze={args.analyze}, failed={args.failed}, force={args.force}")
    failed_files = run_analysis(
        args, audio_db, playlist_db, cli, force_reextract=args.no_cache)
    logger.info(f"Analysis completed with {len(failed_files)} failed files")
    return failed_files


def run_pipeline_mode(args, audio_db, playlist_db, cli):
    """Run pipeline mode - analyze, force, then failed."""
    logger.info("Starting PIPELINE mode (default) - analyze, force, then failed")
    run_pipeline(args, audio_db, playlist_db, cli)
    logger.info("Pipeline mode completed")


def generate_final_summary(audio_db, failed_files, start_time):
    """Generate final summary statistics."""
    try:
        logger.debug("Calculating final summary statistics")
        features_from_db = audio_db.get_all_features()
        total_files = len(features_from_db)
        failed_count = len(failed_files)
        mb_count = 0
        no_mb_count = 0
        for f in features_from_db:
            meta = f.get('metadata', {})
            if meta.get('musicbrainz_id'):
                mb_count += 1
            else:
                no_mb_count += 1

        runtime = time.time() - start_time
        logger.info(
            f"Final summary: Processed={total_files}, Failed={failed_count}, With MusicBrainz={mb_count}, Without MusicBrainz={no_mb_count}, Runtime={runtime:.1f}s")

        logger.debug(f"Processed Files: {total_files}")
        logger.debug(f"Failed Files: {failed_count}")
        logger.debug(f"With MusicBrainz Info: {mb_count}")
        logger.debug(f"Without MusicBrainz Info: {no_mb_count}")
        logger.debug(f"Runtime: {runtime:.1f} seconds")

    except Exception as e:
        logger.error(f"Error generating summary: {e}")


def print_banner():
    """Print the Playlista ASCII art banner."""
    banner = r"""
  _                ___  __ ___     
 |_) |   /\ \_/ |   |  (_   |  /\  
 |   |_ /--\ |  |_ _|_ __)  | /--\ 
                                    
"""
    console = Console()
    console.print(f"[bold magenta]{banner}[/bold magenta]")


def run_application(args):
    """Main application runner."""
    logger.info("Starting Playlista application")

    # Clear the terminal screen only if running interactively
    if sys.stdout.isatty():
        os.system('cls' if os.name == 'nt' else 'clear')
    
    print_banner()

    logger.info("Scanning music library for audio files")
    total_files = len(get_audio_files('/music'))
    logger.info(f"Found {total_files} audio files in library")

    # Initialize playlist_db and audio_db early for all modes
    cache_file = os.getenv('CACHE_FILE', '/app/cache/audio_analysis.db')
    playlist_db, audio_db = initialize_databases(cache_file, args.library)

    total_in_db = len(audio_db.get_all_features())
    logger.info(f"Database contains {total_in_db} analyzed tracks")

    # Initialize CLI
    cli = PlaylistGeneratorCLI()

    if args.pipeline:
        run_pipeline_mode(args, audio_db, playlist_db, cli)
        return

    # Check if the container music directory exists (not the host path)
    if not os.path.exists('/music'):
        logger.error("Music directory not found: /music")
        cli.show_error(f"Music directory not found: /music")
        sys.exit(1)

    start_time = time.time()
    failed_files = []
    
    try:
        # Clean up database
        logger.info("Cleaning up database - removing missing files")
        missing_in_db = audio_db.cleanup_database()
        if missing_in_db:
            logger.info(f"Removed {len(missing_in_db)} missing files from database")
            cli.show_warning(f"Removed {len(missing_in_db)} missing files from database")
            failed_files.extend(missing_in_db)

        # Initialize playlist manager
        playlist_manager = initialize_playlist_manager(
            cache_file, args.playlist_method, args.min_tracks_per_genre)

        if args.update:
            run_update_mode(args, audio_db, playlist_db, cli, playlist_manager)
        elif args.analyze or args.failed or args.force:
            failed_files = run_analysis_mode(args, audio_db, playlist_db, cli)
        elif args.generate_only:
            run_generate_only_mode(args, audio_db, cli, playlist_manager)
        else:
            logger.info(f"DEBUG: No specific mode flags set. analyze={args.analyze}, generate_only={args.generate_only}, update={args.update}")
            run_pipeline_mode(args, audio_db, playlist_db, cli)

        # Show failed files if any
        if failed_files:
            logger.warning(f"Showing {len(failed_files)} failed files")
            cli.show_file_errors(failed_files)

    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        logger.error(f"Error traceback: {traceback.format_exc()}")
        cli.show_error(str(e), details=traceback.format_exc())
        sys.exit(1)
    except KeyboardInterrupt:
        handle_keyboard_interrupt()
    finally:
        # Only print summary for generate/update modes (not analyze_only)
        if not args.analyze:
            generate_final_summary(audio_db, failed_files, start_time)

        # Add session end marker
        logger.info("=" * 80)
        logger.info(f"PLAYLISTA SESSION ENDED - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 80) 