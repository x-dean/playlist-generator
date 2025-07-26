import os
import sys
import time
import logging
import multiprocessing as mp
import sqlite3
import json
import shutil
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.console import Console
from music_analyzer.parallel import ParallelProcessor
from music_analyzer.sequential import SequentialProcessor
from music_analyzer.feature_extractor import AudioAnalyzer
import psutil
import threading
from typing import List, Tuple, Optional
from utils.cli import CLIContextManager
from rich.status import Status
from rich.live import Live
from rich.panel import Panel

# Default logging level for workers
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
logging.getLogger().setLevel(getattr(logging, LOG_LEVEL.upper(), logging.DEBUG))

logger = logging.getLogger(__name__)
BIG_FILE_SIZE_MB = 50


def _update_progress_bar(progress, task_id, files_list, current_index, total_count,
                         mode_color="[cyan]", mode_text="Processing", status_dot="",
                         current_filepath=None, is_parallel=True, file_sizes=None):
    """Update progress bar with current file info and next file preview."""
    if current_index >= len(files_list):
        return

    item = files_list[current_index]
    file_path = item[0] if isinstance(item, tuple) else item
    current_filename = os.path.basename(file_path)
    max_len = 70  # Increased to make better use of available space
    if len(current_filename) > max_len:
        display_name = current_filename[:max_len-3] + "..."
    else:
        display_name = current_filename

    # Get file size from database or filesystem
    if file_sizes and file_path in file_sizes:
        # Use database file size
        size_bytes = file_sizes[file_path]
        size_mb = size_bytes / (1024 * 1024)
        file_size_info = f"{size_mb:.1f}MB"
        # Determine processing mode based on file size
        if size_mb > BIG_FILE_SIZE_MB:
            processing_mode = "Sequential"  # Big files always use sequential
        else:
            processing_mode = "Parallel" if is_parallel else "Sequential"
    else:
        # Fallback to filesystem check
        try:
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            file_size_info = f"{size_mb:.1f}MB"
            if size_mb > BIG_FILE_SIZE_MB:
                processing_mode = "Sequential"
            else:
                processing_mode = "Parallel" if is_parallel else "Sequential"
        except:
            file_size_info = "Unknown"
            processing_mode = "Sequential" if not is_parallel else "Parallel"

    # Calculate percentage
    percentage = (current_index / total_count * 100) if total_count > 0 else 0

    # Update progress bar
    if current_index < len(files_list) - 1:
        # Show next file being processed
        progress.update(
            task_id,
            advance=1 if current_index > 0 else 0,
            description=f"{mode_color}{processing_mode}: {display_name} ({file_size_info}) ({current_index}/{total_count}) {status_dot}",
            trackinfo=f"{percentage:.1f}%"
        )
    else:
        # Last file completed
        progress.update(
            task_id,
            advance=1,
            description=f"{mode_color}Completed: {display_name} ({file_size_info}) ({current_index}/{total_count}) {status_dot}",
            trackinfo=f"{percentage:.1f}%"
        )


def _get_status_dot(features, db_write_success):
    """Get status dot based on processing result."""
    if features and db_write_success:
        return "🟢"  # Green dot for success
    else:
        return "🔴"  # Red dot for failure

# --- File Selection ---


def select_files_for_analysis(args, audio_db):
    """Simplified file selection based on args and DB state."""
    logger.debug(f"DISCOVERY: select_files_for_analysis: force={args.force}, failed={args.failed}")
    
    # Get database state
    db_features = audio_db.get_all_features(include_failed=True)
    db_files = set(f['filepath'] for f in db_features)
    failed_db_files = set(f['filepath'] for f in db_features if f['failed'])
    
    logger.debug(f"DISCOVERY: files in db={len(db_files)}, failed in db={len(failed_db_files)}")
    if db_files:
        sample_db_files = list(db_files)[:3]
        logger.debug(f"DISCOVERY: Sample files from db: {sample_db_files}")
    
    if args.failed:
        # Failed mode: only process failed files that aren't in failed directory
        files_to_analyze = []
        for filepath in failed_db_files:
            if '/failed_files' not in filepath:
                files_to_analyze.append(filepath)
        logger.info(f"DISCOVERY: Failed mode - {len(files_to_analyze)} files to retry")
        return files_to_analyze, [], [], db_features
    
    elif args.force:
        # Force mode: process all files except failed ones
        files_to_analyze = audio_db.get_files_needing_analysis()
        # Filter out failed files
        files_to_analyze = [f[0] for f in files_to_analyze if f[0] not in failed_db_files]
        logger.info(f"DISCOVERY: Force mode - {len(files_to_analyze)} files to process")
        if files_to_analyze:
            sample_files = files_to_analyze[:3]
            logger.debug(f"DISCOVERY: Sample files to analyze (force): {sample_files}")
    else:
        # Normal mode: only new/modified files
        files_to_analyze = audio_db.get_files_needing_analysis()
        files_to_analyze = [f[0] for f in files_to_analyze]
        logger.info(f"DISCOVERY: Normal mode - {len(files_to_analyze)} files to process")
        if files_to_analyze:
            sample_files = files_to_analyze[:3]
            logger.debug(f"DISCOVERY: Sample files to analyze (normal): {sample_files}")
    
    # Get file sizes from database for classification
    file_sizes = audio_db.get_file_sizes_from_db(files_to_analyze)
    
    # Classify files by size
    big_files = []
    normal_files = []
    for file_path in files_to_analyze:
        file_size_bytes = file_sizes.get(file_path, 0)
        file_size_mb = file_size_bytes / (1024 * 1024)
        if file_size_mb > BIG_FILE_SIZE_MB:
            big_files.append(file_path)
        else:
            normal_files.append(file_path)
    
    logger.debug(f"DISCOVERY: normal_files={len(normal_files)}, big_files={len(big_files)}")
    return normal_files, big_files, [], db_features


def move_failed_files(audio_db, failed_dir=None):
    import os
    if failed_dir is None:
        failed_dir = '/music/failed_files'
    failed_dir_abs = os.path.abspath(failed_dir)
    if not os.path.exists(failed_dir_abs):
        os.makedirs(failed_dir_abs)
    db_features = audio_db.get_all_features(include_failed=True)
    moved = 0
    for f in db_features:
        if f.get('failed'):
            src = f['filepath']
            if not os.path.exists(src):
                continue
            dst = os.path.join(failed_dir_abs, os.path.basename(src))
            if os.path.abspath(src) == os.path.abspath(dst):
                logger.info(
                    f"File {src} is already in the failed_files directory, skipping move.")
                continue
            try:
                shutil.move(src, dst)
                logger.warning(f"Moved failed file to {dst}")
                moved += 1
            except Exception as e:
                logger.error(f"Failed to move {src} to {dst}: {e}")
    if moved > 0:
        logger.info(
            f"Moved {moved} failed files to '{failed_dir_abs}' and excluded them from analysis.")
    else:
        logger.info(f"No failed files to move to '{failed_dir_abs}'.")


def move_newly_failed_files(audio_db, newly_failed, failed_dir=None):
    import os
    if failed_dir is None:
        failed_dir = '/music/failed_files'
    failed_dir_abs = os.path.abspath(failed_dir)
    if not os.path.exists(failed_dir_abs):
        os.makedirs(failed_dir_abs)
    moved = 0
    moved_files = []
    for src in newly_failed:
        if not os.path.exists(src):
            continue
        dst = os.path.join(failed_dir_abs, os.path.basename(src))
        if os.path.abspath(src) == os.path.abspath(dst):
            logger.info(
                f"File {src} is already in the failed_files directory, skipping move.")
            continue
        try:
            if os.path.exists(dst):
                os.remove(dst)
                logger.info(
                    f"Existing file at {dst} deleted before replacement.")
            shutil.move(src, dst)
            logger.warning(f"Moved (or replaced) failed file to {dst}")
            moved += 1
            moved_files.append(src)
        except Exception as e:
            logger.error(f"Failed to move/replace {src} to {dst}: {e}")
    if moved > 0:
        logger.info(
            f"Moved {moved} newly failed files to '{failed_dir_abs}' and excluded them from analysis.")
        logger.info(f"Full paths: {moved_files}")
    else:
        logger.info(f"No newly failed files to move to '{failed_dir_abs}'.")

# --- Graceful Shutdown ---

# --- Worker Managers ---


class ParallelWorkerManager:
    def __init__(self):
        pass

    def process(self, files, workers, status_queue=None, **kwargs):
        processor = ParallelProcessor()
        return processor.process(files, workers=workers, status_queue=status_queue, **kwargs)


class SequentialWorkerManager:
    def __init__(self):
        pass

    def process(self, files, workers, **kwargs):
        processor = SequentialProcessor()
        return processor.process(files, workers=workers, **kwargs)


# BigFileWorkerManager removed - redundant with SequentialProcessor
# SequentialProcessor already handles large files with proper timeout protection

# --- Progress Bar ---


def create_progress_bar(total_files):
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        TextColumn("{task.fields[trackinfo]}", justify="right"),
        console=Console()
    )


# --- Main Orchestration ---
def run_analysis(args, audio_db, playlist_db, cli, force_reextract=False, pipeline_mode=False):
    """Run analysis with improved logic based on mode."""
    logger.info(
        f"Starting analysis with mode: analyze={args.analyze}, force={args.force}, failed={args.failed}, pipeline_mode={pipeline_mode}")

    if args.failed:
        return run_failed_mode(args, audio_db, cli)
    elif args.force:
        return run_force_mode(args, audio_db, cli)
    else:
        return run_analyze_mode(args, audio_db, cli, force_reextract)


def run_analyze_mode(args, audio_db, cli, force_reextract):
    """Run analysis mode - analyze files that need processing."""
    logger.info("Starting ANALYZE mode")

    # Get files that need analysis using simplified selection
    normal_files, big_files, _, db_features = select_files_for_analysis(args, audio_db)
    files_to_analyze = normal_files + big_files
    
    logger.debug(f"DISCOVERY: Files needing analysis: {len(files_to_analyze)}")

    if not files_to_analyze:
        logger.info("No files need analysis. All files are up to date.")
        return []

    # Get statistics
    total_files = len(audio_db.get_all_audio_files())
    total_in_db = audio_db.get_all_tracks()

    # Display statistics
    print(f"\n📊 Library Statistics:")
    print(f"   📁 Total music files: {total_files:,}")
    print(f"   💾 Tracks in database: {total_in_db:,}")
    print(f"   🔄 Files to process: {len(files_to_analyze):,}")
    print(
        f"   📈 Progress: {total_in_db}/{total_files} ({total_in_db/total_files*100:.1f}% complete)")
    print()  # Add spacing before progress bar

    # Get file sizes from database for progress bar
    file_sizes = audio_db.get_file_sizes_from_db(files_to_analyze)
    
    logger.info(f"DISCOVERY: File distribution: {len(normal_files)} normal files, {len(big_files)} big files (>50MB)")
    
    # Run analysis on files that need it
    failed_files = []

    with CLIContextManager(cli, len(files_to_analyze), f"[cyan]Analyzing {len(files_to_analyze)} files...") as (progress, task_id):
        processed_count = 0

        # Pre-update progress bar with first file
        if files_to_analyze:
            _update_progress_bar(progress, task_id, files_to_analyze, 0, len(files_to_analyze),
                                 "[cyan]", "", "", None, True, file_sizes)

        # Step 1: Process big files sequentially
        if big_files:
            logger.info(f"Processing {len(big_files)} big files sequentially...")
            sequential_processor = SequentialProcessor()
            workers = 1  # Sequential processing uses 1 worker
            
            for features, filepath, db_write_success in sequential_processor.process(big_files, workers, force_reextract=force_reextract):
                processed_count += 1
                filename = os.path.basename(filepath)
                
                # Get status dot for result
                status_dot = _get_status_dot(features, db_write_success)
                
                # Update progress bar
                _update_progress_bar(progress, task_id, files_to_analyze, processed_count, len(files_to_analyze),
                                     "[cyan]", "", status_dot, None, True, file_sizes)
                
                logger.debug(f"Processed {processed_count}/{len(files_to_analyze)}: {filename} ({status_dot})")
                
                if not features or not db_write_success:
                    failed_files.append(filepath)
                    logger.warning(f"Analysis failed for {filepath}")
                else:
                    logger.info(f"Analysis completed for {filepath}")
        
        # Step 2: Process normal files in parallel
        if normal_files:
            logger.info(f"Processing {len(normal_files)} normal files in parallel...")
            parallel_processor = ParallelProcessor()
            workers = args.workers or max(1, mp.cpu_count())
            
            for features, filepath, db_write_success in parallel_processor.process(normal_files, workers, force_reextract=force_reextract):
                processed_count += 1
                filename = os.path.basename(filepath)
                
                # Get status dot for result
                status_dot = _get_status_dot(features, db_write_success)
                
                # Update progress bar
                _update_progress_bar(progress, task_id, files_to_analyze, processed_count, len(files_to_analyze),
                                     "[cyan]", "", status_dot, None, True, file_sizes)
                
                logger.debug(f"Processed {processed_count}/{len(files_to_analyze)}: {filename} ({status_dot})")
                
                if not features or not db_write_success:
                    failed_files.append(filepath)
                    logger.warning(f"Analysis failed for {filepath}")
                else:
                    logger.info(f"Analysis completed for {filepath}")

        logger.info(f"Processing completed. Processed {processed_count} files out of {len(files_to_analyze)}")

    logger.info(
        f"ANALYZE mode completed with {len(failed_files)} failed files")
    return failed_files


def run_force_mode(args, audio_db, cli):
    """FORCE mode: Fast validation and retry of cached files."""
    logger.info("Starting FORCE mode - fast validation and retry")

    # Get invalid files that need retry
    invalid_files = audio_db.get_invalid_files_from_db()
    logger.debug(f"Invalid files found: {len(invalid_files) if invalid_files else 0}")

    if not invalid_files:
        logger.info("No invalid files found in database")
        return []

    # Display totals with better visual formatting
    total_files = len(audio_db.get_all_audio_files())
    total_in_db = len(audio_db.get_all_tracks())
    files_to_retry = len(invalid_files)

    print(f"\n📊 Force Mode Statistics:")
    print(f"   📁 Total music files: {total_files:,}")
    print(f"   💾 Tracks in database: {total_in_db:,}")
    print(f"   🔄 Files to retry: {files_to_retry:,}")
    print(
        f"   📈 Database health: {total_in_db}/{total_files} ({total_in_db/total_files*100:.1f}% valid)")
    print()  # Add spacing before progress bar

    # Retry analysis for invalid files
    failed_files = []
    with CLIContextManager(cli, len(invalid_files), f"[yellow]Retrying {len(invalid_files)} invalid files...") as (progress, task_id):
        # Log processor type for force mode (always sequential for retries)
        logger.info("Using Sequential processor for force mode retries")

        # Pre-update progress bar with first file
        if invalid_files:
            _update_progress_bar(progress, task_id, invalid_files, 0, len(invalid_files),
                                 # Handles tuple or str
                                 "[yellow]", "", "", None, True)

        # Prepare list of file paths only for processing
        file_paths_only = [item[0] if isinstance(
            item, tuple) else item for item in invalid_files]

        processed_count = 0
        for file_path in file_paths_only:
            processed_count += 1
            filename = os.path.basename(file_path)

            # Update progress bar with current file
            _update_progress_bar(progress, task_id, invalid_files, processed_count, len(invalid_files),
                                 # Handles tuple or str
                                 "[yellow]", "", "", None, True)

            logger.info(f"Retrying analysis for {file_path}")
            success, features = audio_db.retry_analysis_with_backoff(
                file_path, max_attempts=3)

            # Get status dot for result
            status_dot = _get_status_dot(features, success)

            if success:
                logger.info(f"Retry successful for {file_path}")
                audio_db.unmark_as_failed(file_path)
            else:
                logger.error(
                    f"Retry failed for {file_path} - marking as failed")
                file_info = audio_db._get_file_info(file_path)
                audio_db._mark_failed(file_info)
                failed_files.append(file_path)

            logger.debug(f"Retry result for {filename}: {status_dot}")

    logger.info(f"FORCE mode completed with {len(failed_files)} failed files")

    # Check if we were interrupted
    return failed_files


def run_failed_mode(args, audio_db, cli):
    """FAILED mode: Recovery of previously failed files."""
    logger.info("Starting FAILED mode - recovering failed files")

    # Get all failed files from database
    failed_files = audio_db.get_failed_files_from_db()
    logger.debug(f"Failed files found: {len(failed_files) if failed_files else 0}")

    if not failed_files:
        logger.info("No failed files found in database")
        return []

    # Display totals with better visual formatting
    total_files = len(audio_db.get_all_audio_files())
    total_in_db = len(audio_db.get_all_tracks())
    files_to_recover = len(failed_files)

    print(f"\n📊 Recovery Mode Statistics:")
    print(f"   📁 Total music files: {total_files:,}")
    print(f"   💾 Tracks in database: {total_in_db:,}")
    print(f"   🔄 Files to recover: {files_to_recover:,}")
    print(
        f"   📈 Success rate: {total_in_db}/{total_files} ({total_in_db/total_files*100:.1f}% successful)")
    print()  # Add spacing before progress bar

    # Retry each failed file up to 3 times
    still_failed = []
    with CLIContextManager(cli, len(failed_files), f"[red]Recovering {len(failed_files)} failed files...") as (progress, task_id):
        # Log processor type for failed mode (always sequential for retries)
        logger.info("Using Sequential processor for failed mode retries")

        # Pre-update progress bar with first file
        if failed_files:
            _update_progress_bar(progress, task_id, failed_files, 0, len(failed_files),
                                 # Handles tuple or str
                                 "[red]", "", "", None, False)

        # Prepare list of file paths only for processing
        file_paths_only = [item[0] if isinstance(
            item, tuple) else item for item in failed_files]

        processed_count = 0
        for file_path in file_paths_only:
            processed_count += 1
            filename = os.path.basename(file_path)

            # Update progress bar with current file
            _update_progress_bar(progress, task_id, failed_files, processed_count, len(failed_files),
                                 # Handles tuple or str
                                 "[red]", "", "", None, False)

            logger.info(f"Retrying failed file: {file_path}")
            success, features = audio_db.retry_analysis_with_backoff(
                file_path, max_attempts=3)

            # Get status dot for result
            status_dot = _get_status_dot(features, success)

            if success:
                logger.info(f"Recovery successful for {file_path}")
                audio_db.unmark_as_failed(file_path)
            else:
                logger.error(
                    f"Recovery failed for {file_path} - moving to failed directory")
                if audio_db.move_to_failed_directory(file_path):
                    still_failed.append(file_path)
                else:
                    logger.error(
                        f"Failed to move {file_path} to failed directory")

            logger.debug(f"Recovery result for {filename}: {status_dot}")

    logger.info(
        f"FAILED mode completed with {len(still_failed)} still failed files")

    # Check if we were interrupted
    return still_failed


def run_pipeline(args, audio_db, playlist_db, cli):
    from rich.table import Table
    from rich.console import Console
    results = []
    console = Console()
    
    logger.info("Starting pipeline execution")
    logger.debug(f"Pipeline args: force={args.force}, failed={args.failed}, workers={args.workers}")
    
    # Debug: Check initial state
    logger.debug(f"DISCOVERY: Pipeline starting - total files in db: {len(audio_db.get_all_features())}")
    logger.debug(f"DISCOVERY: Pipeline starting - files needing analysis: {len(audio_db.get_files_needing_analysis())}")
    
    console.print(
        "\n[bold cyan]PIPELINE: Starting default analysis[/bold cyan]")
    console.print("[dim]Analyze new files[/dim]")
    args.force = False
    args.failed = False
    logger.info("Pipeline Stage 1: Running default analysis (new files only)")
    logger.debug(f"DISCOVERY: Stage 1 - before run_analysis: force={args.force}, failed={args.failed}")
    res1 = run_analysis(args, audio_db, playlist_db, cli,
                        force_reextract=False, pipeline_mode=True)
    logger.info(f"Pipeline Stage 1 result: {len(res1) if res1 else 0} files processed")
    logger.debug(f"DISCOVERY: Stage 1 - after run_analysis: returned {len(res1) if res1 else 0} files")
    results.append(('Default', {'processed_this_run': len(res1) if res1 else 0, 'failed_this_run': len(res1) if res1 else 0}))
    console.print(
        "[green]PIPELINE: Default analysis complete (new files analyzed)[/green]")
    console.print(
        "[dim]───────────────────────────────────────────────[/dim]\n")

    console.print(
        "[bold cyan]PIPELINE: Enriching missing tags from MusicBrainz and Last.fm (if API provided)[/bold cyan]")
    console.print("[dim]Enriching tags from MusicBrainz and Last.fm[/dim]")
    args.force = True
    args.failed = False
    logger.info("Pipeline Stage 2: Running force analysis (re-enrich all files)")
    logger.debug(f"DISCOVERY: Stage 2 - before run_analysis: force={args.force}, failed={args.failed}")
    res2 = run_analysis(args, audio_db, playlist_db, cli,
                        force_reextract=True, pipeline_mode=True)
    logger.info(f"Pipeline Stage 2 result: {len(res2) if res2 else 0} files processed")
    logger.debug(f"DISCOVERY: Stage 2 - after run_analysis: returned {len(res2) if res2 else 0} files")
    results.append(('Force', {'processed_this_run': len(res2) if res2 else 0, 'failed_this_run': len(res2) if res2 else 0}))
    console.print(
        "[green]PIPELINE: Tags enriching complete (tags updated)[/green]")
    console.print(
        "[dim]───────────────────────────────────────────────[/dim]\n")

    console.print("[bold cyan]PIPELINE: Retrying failed files[/bold cyan]")
    console.print("[dim]Retry failed files[/dim]")
    args.force = False
    args.failed = True
    logger.info("Pipeline Stage 3: Running failed analysis (retry failed files)")
    logger.debug(f"DISCOVERY: Stage 3 - before run_analysis: force={args.force}, failed={args.failed}")
    res3 = run_analysis(args, audio_db, playlist_db, cli,
                        force_reextract=True, pipeline_mode=True)
    logger.info(f"Pipeline Stage 3 result: {len(res3) if res3 else 0} files processed")
    logger.debug(f"DISCOVERY: Stage 3 - after run_analysis: returned {len(res3) if res3 else 0} files")
    # Count files in /music/failed_files after failed step
    import os
    failed_dir = '/music/failed_files'
    try:
        moved_failed = len([f for f in os.listdir(
            failed_dir) if os.path.isfile(os.path.join(failed_dir, f))])
    except Exception:
        moved_failed = 0
    results.append(('Failed', {'processed_this_run': len(res3) if res3 else 0, 'failed_this_run': moved_failed}))
    console.print(
        "[green]PIPELINE: Failed files retry complete (failures handled)[/green]")
    console.print(
        "[dim]───────────────────────────────────────────────[/dim]\n")

    # Show a single summary table at the end
    console.print("\n")
    table = Table(title="Pipeline Summary")
    table.add_column("Stage")
    table.add_column("Processed")
    table.add_column("Failed")
    labels = {
        'Default': 'Analysis Step',
        'Force': 'Re-enriching Step',
        'Failed': 'Failed Retry Step'
    }
    for i, (stage, res) in enumerate(results):
        processed = res.get('processed_this_run', '-')
        if stage == 'Failed':
            failed = str(moved_failed)
        else:
            failed = res.get('failed_this_run', '-')
        table.add_row(labels.get(stage, stage), str(processed), str(failed))
    console.print(table)
    console.print(
        "\n[bold green]PIPELINE: Complete. Now you can start generating playlists![/bold green]\n")

# --- File Discovery Helper ---
# File discovery is now handled by the FileDiscovery class in file_discovery.py
