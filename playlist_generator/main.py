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
from playlist_generator.playlist_manager import PlaylistManager
import logging
from utils.cli import PlaylistGeneratorCLI, CLIContextManager
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.console import Console
from rich.panel import Panel
from utils.checkpoint import CheckpointManager
from typing import Optional
import multiprocessing
import threading
from rich.live import Live
from rich.spinner import Spinner
import essentia
essentia.log.infoActive = False
essentia.log.warningActive = False
essentia.log.errorActive = False

# Suppress glog output before importing Essentia or any C++ modules
os.environ["GLOG_minloglevel"] = "3"  # Only FATAL
os.environ["GLOG_logtostderr"] = "1"
os.environ["GLOG_stderrthreshold"] = "3"

logger = setup_colored_logging()

import logging as pylogging
pylogging.getLogger("musicbrainzngs").setLevel(pylogging.WARNING)

# Initialize system monitoring
# Remove SystemMonitor and monitor_performance imports
# Remove @monitor_performance decorators from get_audio_files and main
# Remove system_monitor initialization
checkpoint_manager = CheckpointManager()

# Initialize CLI
cli = PlaylistGeneratorCLI()

os.environ["ESSENTIA_LOGGING_LEVEL"] = "error"
os.environ["ESSENTIA_STREAM_LOGGING"] = "none"

cache_dir = os.getenv('CACHE_DIR', '/app/cache')
logfile_path = os.path.join(cache_dir, 'essentia_stderr.log')
logfile = open(logfile_path, 'w')
os.dup2(logfile.fileno(), 2)  # Redirect fd 2 (stderr) at the OS level
sys.stderr = logfile  # Also update Python's sys.stderr

def get_audio_files(music_dir: str) -> list[str]:
    """Recursively find all audio files in the given directory.

    Args:
        music_dir (str): Path to the music directory.

    Returns:
        list[str]: List of audio file paths.
    """
    file_list = []
    valid_ext = ('.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac', '.opus')
    
    for root, _, files in os.walk(music_dir):
        for file in files:
            file_lower = file.lower()
            if file_lower.endswith(valid_ext):
                file_list.append(os.path.join(root, file))
    
    logger.info(f"Found {len(file_list)} audio files in {music_dir}")
    return file_list

def convert_to_host_path(container_path: str, host_music_dir: str, container_music_dir: str) -> str:
    """Converts a path from the container to the host.

    Args:
        container_path (str): Path within the container.
        host_music_dir (str): Path to the host's music directory.
        container_music_dir (str): Path to the container's music directory.

    Returns:
        str: Path on the host.
    """
    container_path = os.path.normpath(container_path)
    container_music_dir = os.path.normpath(container_music_dir)
    
    if not container_path.startswith(container_music_dir):
        return container_path
    
    rel_path = os.path.relpath(container_path, container_music_dir)
    return os.path.join(host_music_dir, rel_path)

def save_playlists(playlists: dict[str, dict], output_dir: str, host_music_dir: str, container_music_dir: str, failed_files: list[str], playlist_method: Optional[str] = None) -> None:
    """Saves generated playlists to disk.

    Args:
        playlists (dict[str, dict]): Dictionary of playlist names to their data.
        output_dir (str): Root output directory.
        host_music_dir (str): Path to the host's music directory.
        container_music_dir (str): Path to the container's music directory.
        failed_files (list[str]): List of failed file paths.
        playlist_method (str | None): The method used to generate playlists (e.g., 'time', 'kmeans').
    """
    # For time-based, create subfolders per slot
    def get_time_slot_from_name(name: str) -> str | None:
        if name.startswith("TimeSlot_"):
            slot_part = name[len("TimeSlot_"):]
            if "_Part" in slot_part:
                slot = slot_part.split("_Part")[0]
            else:
                slot = slot_part
            return slot
        return None

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

        host_songs = [
            convert_to_host_path(song, host_music_dir, container_music_dir)
            for song in playlist_data['tracks']
        ]
        playlist_path = os.path.join(playlist_out_dir, f"{name}.m3u")
        with open(playlist_path, 'w') as f:
            f.write("\n".join(host_songs))
        saved_count += 1
        logger.debug(f"Saved {name} with {len(host_songs)} tracks to {playlist_path}")

    # Save failed files (keep at root of method dir)
    os.makedirs(output_dir, exist_ok=True)
    if failed_files:
        failed_path = os.path.join(output_dir, "Failed_Files.m3u")
        with open(failed_path, 'w') as f:
            host_failed = [
                convert_to_host_path(p, host_music_dir, container_music_dir)
                for p in failed_files
            ]
            f.write("\n".join(host_failed))
        logger.info(f"Saved {len(failed_files)} failed files to {failed_path}")

def main() -> None:
    """Main entry point for the Playlist Generator CLI application."""
    # Start CLI session
    cli.start_session()

    # Parse arguments
    parser = argparse.ArgumentParser(description='Music Playlist Generator')
    group = parser.add_mutually_exclusive_group(required=False)
    parser.add_argument('--music_dir', required=True, help='Music directory in container')
    parser.add_argument('--host_music_dir', required=True, help='Host music directory')
    parser.add_argument('--output_dir', default='./playlists', help='Output directory')
    parser.add_argument('--num_playlists', type=int, default=8, help='Number of playlists')
    parser.add_argument('--workers', type=int, default=None, help='Number of workers (default: auto)')
    parser.add_argument('--force_sequential', action='store_true', help='Force sequential processing')
    parser.add_argument('-a', '--analyze', action='store_true', help='Analyze files (see --failed and --force for options)')
    parser.add_argument('-g', '--generate_only', action='store_true', help='Only generate playlists from database (no analysis)')
    parser.add_argument('-u', '--update', action='store_true', help='Update all playlists from database (no analysis, regenerates all playlists)')
    parser.add_argument('--failed', action='store_true', help='With --analyze: only re-analyze files previously marked as failed')
    # Only one add_argument for --force
    parser.add_argument('-f', '--force', action='store_true', help='Force re-analyze or re-enrich (used with --analyze or --enrich_only)')
    parser.add_argument('--status', action='store_true', help='Show library/database statistics and exit')
    parser.add_argument('--resume', action='store_true', help='Resume from last checkpoint if available')
    parser.add_argument('-m', '--playlist_method', choices=['all', 'time', 'kmeans', 'cache', 'tags'], default='all',
                      help='Playlist generation method: all (feature-group, default), time, kmeans, cache, or tags (genre+decade)')
    parser.add_argument('--min_tracks_per_genre', type=int, default=10, help='Minimum number of tracks required for a genre to create a playlist (tags method only)')
    parser.add_argument('--enrich_tags', action='store_true', help='Enrich tags using MusicBrainz/Last.fm APIs (default: False)')
    parser.add_argument('--force_enrich_tags', action='store_true', help='Force re-enrichment of tags and overwrite metadata in the database (default: False)')
    parser.add_argument('--enrich_only', action='store_true', help='Enrich tags for all tracks in the database using MusicBrainz/Last.fm APIs (no analysis or playlist generation)')
    # The --force argument is now handled by -f/--force, so we don't need a separate --force argument here.
    args = parser.parse_args()

    # Set cache file path
    cache_file = os.path.join(cache_dir, 'audio_analysis.db')
    playlist_db = PlaylistDatabase(cache_file)

    # If --status is set, show statistics and exit
    if getattr(args, 'status', False):
        stats = playlist_db.get_library_statistics()
        # Add failed/skipped files count
        audio_db = AudioAnalyzer(cache_file)
        skipped_count = len([f for f in audio_db.get_all_features(include_failed=True) if f['failed']])
        cli.show_library_statistics(stats)
        print(f"Skipped (failed) files: {skipped_count}")
        sys.exit(0)

    # If no mutually exclusive mode is set, default to analyze_only
    if not (args.analyze or args.failed or args.update):
        args.analyze = True

    # Show configuration
    cli.show_config({
        'Music Directory': args.music_dir,
        'Output Directory': args.output_dir,
        'Number of Playlists': args.num_playlists,
        'Workers': args.workers or 'Auto',
        'Mode': 'Sequential' if args.force_sequential or (args.workers is not None and int(args.workers) <= 1) else 'Parallel',
        'Update Mode': args.update,
        'Analysis Only': args.analyze,
        'Generate Only': args.generate_only,
        'Resume': args.resume,
        'Playlist Method': args.playlist_method
    })
    
    # Initialize components
    audio_db = AudioAnalyzer(cache_file, host_music_dir=args.host_music_dir, container_music_dir=args.music_dir)
    # Pass min_tracks_per_genre and enrich_tags to PlaylistManager if using tags method
    if args.playlist_method == 'tags':
        playlist_manager = PlaylistManager(
            cache_file, args.playlist_method, min_tracks_per_genre=args.min_tracks_per_genre, enrich_tags=args.enrich_tags, force_enrich_tags=args.force_enrich_tags)
    else:
        playlist_manager = PlaylistManager(cache_file, args.playlist_method)
    
    container_music_dir = args.music_dir.rstrip('/')
    host_music_dir = args.host_music_dir.rstrip('/')
    failed_files = []

    if not os.path.exists(args.music_dir):
        cli.show_error(f"Music directory not found: {args.music_dir}")
        sys.exit(1)

    start_time = time.time()
    try:
        # Try to resume from checkpoint
        if args.resume:
            state = checkpoint_manager.get_recovery_state()
            if state:
                cli.update_status("Resuming from checkpoint")
                failed_files = state.get('failed_files', [])
                if 'progress' in state:
                    cli.update_status(f"Resuming from {state['progress']} stage")

        # Clean up database
        missing_in_db = audio_db.cleanup_database()
        if missing_in_db:
            cli.show_warning(f"Removed {len(missing_in_db)} missing files from database")
            failed_files.extend(missing_in_db)

        # Dedicated enrichment mode
        if args.enrich_only:
            from playlist_generator.tag_based import TagBasedPlaylistGenerator
            from database.db_manager import DatabaseManager
            import json
            db_file = cache_file
            dbm = DatabaseManager(db_file)
            # Load all tracks from DB
            conn = dbm._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT file_path, metadata FROM audio_features")
            rows = cursor.fetchall()
            total = len(rows)
            enriched = 0
            skipped = 0
            failed = 0
            tagger = TagBasedPlaylistGenerator(db_file=db_file, enrich_tags=True, force_enrich_tags=args.force)
            print(f"Starting enrichment for {total} tracks (force: {args.force})...")
            for row in rows:
                filepath = row[0]
                try:
                    meta = json.loads(row[1]) if row[1] else {}
                    track_data = {'filepath': filepath, 'metadata': meta}
                    before = dict(meta)
                    result = tagger.enrich_track_metadata(track_data, force_enrich_tags=args.force)
                    after = result.get('metadata', {})
                    # If force or if genre/year was missing and now present, count as enriched
                    if args.force or (not before.get('genre') and after.get('genre')) or (not before.get('year') and after.get('year')):
                        enriched += 1
                    else:
                        skipped += 1
                except Exception as e:
                    failed += 1
                    print(f"Failed to enrich {filepath}: {e}")
            print(f"\nEnrichment complete. Total: {total}, Enriched: {enriched}, Skipped: {skipped}, Failed: {failed}")
            exit(0)

        # Create a multiprocessing manager queue for long-running file notifications
        manager = multiprocessing.Manager()
        status_queue = manager.Queue()
        long_running_files = set()
        spinner_stop = threading.Event()

        def spinner_panel():
            if not long_running_files:
                return Panel("All files are processing normally.", title="Long-Running Files", border_style="green")
            return Panel(
                "\n".join([f"[yellow]{file}[/yellow]" for file in long_running_files]),
                title="[cyan]Long-Running Files (>5s) [spinner]",
                border_style="yellow"
            )

        def status_listener():
            while not spinner_stop.is_set():
                try:
                    filepath = status_queue.get(timeout=0.5)
                    long_running_files.add(filepath)
                except Exception:
                    continue

        listener_thread = threading.Thread(target=status_listener, daemon=True)
        listener_thread.start()

        if args.update:
            cli.update_status("Running in UPDATE mode")
            features_from_db = audio_db.get_all_features()
            with CLIContextManager(cli, 3, "[green]Generating playlists...") as (progress, task_id):
                all_playlists = playlist_manager.generate_playlists(features_from_db, args.num_playlists)
                progress.update(task_id, advance=3)
            # Show playlist statistics
            cli.show_playlist_stats(playlist_manager.get_playlist_stats())
            # Save playlists to DB and to disk
            playlist_db.save_playlists(all_playlists)
            if args.playlist_method == 'tags':
                method_dir = os.path.join(args.output_dir, 'tags')
            else:
                method_dir = os.path.join(args.output_dir, args.playlist_method)
            save_playlists(all_playlists, method_dir, host_music_dir, container_music_dir, failed_files, playlist_method=args.playlist_method)

        elif args.analyze:
            file_list = get_audio_files(args.music_dir)
            file_list = [convert_to_host_path(f, host_music_dir, container_music_dir) for f in file_list]
            # Normalize db_files and failed_files_db to host paths for comparison
            db_features = audio_db.get_all_features(include_failed=True)
            db_files = set(convert_to_host_path(f['filepath'], host_music_dir, container_music_dir) for f in db_features)
            failed_files_db = set(convert_to_host_path(f['filepath'], host_music_dir, container_music_dir) for f in db_features if f['failed'])
            # DEBUG: Print samples to check for path mismatches
            print("Sample from file_list:", file_list[:3])
            print("Sample from failed_files_db:", list(failed_files_db)[:3])
            print("Sample from db_files:", list(db_files)[:3])
            if args.failed:
                # Analyze files not in DB or previously failed
                files_to_analyze = [f for f in file_list if f not in db_files or f in failed_files_db]
            else:
                # Only analyze files not in DB
                files_to_analyze = [f for f in file_list if f not in db_files]
            if not files_to_analyze:
                skipped_count = len([f for f in audio_db.get_all_features(include_failed=True) if f['failed']])
                cli.show_success(f"Processed 0 files! Skipped {skipped_count} files due to errors (see database for details).")
                return
            BIG_FILE_SIZE_MB = 200
            def is_big_file(filepath):
                try:
                    return os.path.getsize(filepath) > BIG_FILE_SIZE_MB * 1024 * 1024
                except Exception:
                    return False
            big_files = [f for f in files_to_analyze if is_big_file(f)]
            normal_files = [f for f in files_to_analyze if not is_big_file(f)]
            failed_files = []
            processed_count = 0
            total_files = len(files_to_analyze)
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
            with progress:
                task_id = progress.add_task(f"Processed 0/{total_files} files", total=total_files, trackinfo="")
                # 1. Process normal files in parallel
                if normal_files:
                    if args.force_sequential or (args.workers and args.workers <= 1):
                        processor = SequentialProcessor()
                        process_iter = processor.process(normal_files, workers=args.workers or 1)
                        for filepath in normal_files:
                            filename = os.path.basename(filepath)
                            try:
                                size_mb = os.path.getsize(filepath) / (1024 * 1024)
                            except Exception:
                                size_mb = 0
                            features, _, _ = process_file_worker(filepath)
                            processed_count += 1
                            progress.update(
                                task_id,
                                advance=1,
                                description=f"Processed {processed_count}/{total_files} files",
                                trackinfo=f"{filename} ({size_mb:.1f} MB)"
                            )
                            logger.debug(f"Features: {features}")
                            if features and 'metadata' in features:
                                meta = features['metadata']
                                if meta.get('musicbrainz_id'):
                                    pass
                                else:
                                    pass
                        failed_files.extend(processor.failed_files)
                    else:
                        processor = ParallelProcessor()
                        process_iter = processor.process(normal_files, workers=args.workers or multiprocessing.cpu_count())
                        for features, filepath, db_write_success in process_iter:
                            filename = os.path.basename(filepath)
                            try:
                                size_mb = os.path.getsize(filepath) / (1024 * 1024)
                            except Exception:
                                size_mb = 0
                            processed_count += 1
                            progress.update(
                                task_id,
                                advance=1,
                                description=f"Processed {processed_count}/{total_files} files",
                                trackinfo=f"{filename} ({size_mb:.1f} MB)"
                            )
                            logger.debug(f"Features: {features}")
                            if features and 'metadata' in features:
                                meta = features['metadata']
                                if meta.get('musicbrainz_id'):
                                    pass
                                else:
                                    pass
                        failed_files.extend(processor.failed_files)
                # 2. Process big files sequentially
                if big_files:
                    processor = SequentialProcessor()
                    for filepath in big_files:
                        filename = os.path.basename(filepath)
                        try:
                            size_mb = os.path.getsize(filepath) / (1024 * 1024)
                        except Exception:
                            size_mb = 0
                        progress.update(
                            task_id,
                            description=f"Processing: {filename} | {processed_count}/{total_files} files",
                            trackinfo=f"{filename} ({size_mb:.1f} MB)"
                        )
                        for features, _ in processor.process([filepath], workers=1):
                            processed_count += 1
                            progress.update(
                                task_id,
                                advance=1,
                                description=f"Processed {processed_count}/{total_files} files (big file)",
                                trackinfo=f"{filename} ({size_mb:.1f} MB)"
                            )
                            logger.debug(f"Features: {features}")
                            if features and 'metadata' in features:
                                meta = features['metadata']
                                if meta.get('musicbrainz_id'):
                                    pass
                                else:
                                    pass
                    failed_files.extend(processor.failed_files)
            cli.show_success(f"Analysis completed. Processed {len(files_to_analyze)} files, {len(failed_files)} failed")
            # Print how many files were skipped due to errors
            skipped_count = len([f for f in audio_db.get_all_features(include_failed=True) if f['failed']])
            if skipped_count > 0:
                print(f"Skipped {skipped_count} files due to errors (see database for details).")
            runtime = time.time() - start_time
            console = Console()
            summary_text = f"""
[bold green]Analysis Summary (this run)[/bold green]

Processed Files: [cyan]{len(files_to_analyze)}[/cyan]
Failed Files: [red]{len(failed_files)}[/red]
Skipped Files (errors): [yellow]{skipped_count}[/yellow]
With MusicBrainz Info: [green]{0}[/green]
Without MusicBrainz Info: [yellow]{0}[/yellow]
Runtime: [magenta]{runtime:.1f} seconds[/magenta]
"""
            console.print(Panel(summary_text, title="📊 Analysis Summary", border_style="blue"))
            return

        elif args.generate_only:
            cli.update_status("Generating playlists from database")
            features_from_db = audio_db.get_all_features()
            
            with CLIContextManager(cli, 3, "[green]Generating playlists...") as (progress, task_id):
                all_playlists = playlist_manager.generate_playlists(features_from_db, args.num_playlists)
                progress.update(task_id, advance=3)
            
            # Show playlist statistics
            cli.show_playlist_stats(playlist_manager.get_playlist_stats())
            
            # Save playlists
            if args.playlist_method == 'tags':
                method_dir = os.path.join(args.output_dir, 'tags')
            else:
                method_dir = os.path.join(args.output_dir, args.playlist_method)
            save_playlists(all_playlists, method_dir, host_music_dir, container_music_dir, failed_files, playlist_method=args.playlist_method)

        else:
            cli.update_status("Running full processing pipeline")
            file_list = get_audio_files(args.music_dir)
            file_list = [convert_to_host_path(f, host_music_dir, container_music_dir) for f in file_list]
            
            # Analysis phase
            with CLIContextManager(cli, len(file_list), "[cyan]Analyzing audio files...") as (progress, task_id):
                processor = ParallelProcessor() if not args.force_sequential else SequentialProcessor()
                workers = args.workers or max(1, mp.cpu_count())
                
                for i, features in enumerate(processor.process(file_list, workers)):
                    # Show only the filename in the progress bar
                    if isinstance(features, tuple):
                        # SequentialProcessor yields (features, filepath)
                        _, filepath = features
                    else:
                        # ParallelProcessor yields features only, can't get filename
                        filepath = file_list[i] if i < len(file_list) else ""
                    filename = os.path.basename(filepath)
                    progress.update(task_id, advance=1, description=f"Analyzing: {filename} ({i+1}/{len(file_list)})")
                
                failed_files.extend(processor.failed_files)
            
            # Generation phase
            features_from_db = audio_db.get_all_features()
            
            with CLIContextManager(cli, 3, "[green]Generating playlists...") as (progress, task_id):
                all_playlists = playlist_manager.generate_playlists(features_from_db, args.num_playlists)
                progress.update(task_id, advance=3)
            
            # Show playlist statistics
            cli.show_playlist_stats(playlist_manager.get_playlist_stats())
            
            # Save playlists
            if args.playlist_method == 'tags':
                method_dir = os.path.join(args.output_dir, 'tags')
            else:
                method_dir = os.path.join(args.output_dir, args.playlist_method)
            save_playlists(all_playlists, method_dir, host_music_dir, container_music_dir, failed_files, playlist_method=args.playlist_method)

        # Show failed files if any
        if failed_files:
            cli.show_file_errors(failed_files)

    except Exception as e:
        cli.show_error(str(e), details=traceback.format_exc())
        
        # Save error state
        checkpoint_manager.save_checkpoint({
            'stage': 'error',
            'error': str(e),
            'traceback': traceback.format_exc(),
            'failed_files': failed_files
        }, name='error_state')
        
        sys.exit(1)
        
    finally:
        # Only print summary for generate/update modes (not analyze_only)
        if not args.analyze:
            try:
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
                print("\n=== Analysis Summary ===")
                print(f"Processed Files: {total_files}")
                print(f"Failed Files: {failed_count}")
                print(f"With MusicBrainz Info: {mb_count}")
                print(f"Without MusicBrainz Info: {no_mb_count}")
                print(f"Runtime: {runtime:.1f} seconds")
            except Exception as e:
                print(f"Error generating summary: {e}")

if __name__ == "__main__":
    main()