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
    logger.debug(f"PROGRESS: Updating progress bar - current_index={current_index}, total_count={total_count}")
    
    if current_index >= len(files_list):
        logger.debug(f"PROGRESS: current_index {current_index} >= len(files_list) {len(files_list)}, returning")
        return

    # Use the actual filepath being processed if provided, otherwise fall back to list lookup
    if current_filepath:
        file_path = current_filepath
    else:
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

    logger.debug(f"PROGRESS: Updating progress bar for {display_name} - {processing_mode} mode, {percentage:.1f}%")

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
    
    logger.debug(f"PROGRESS: Progress bar update completed")


def _get_status_dot(features, db_write_success):
    """Get status dot based on processing result."""
    if features and db_write_success:
        return "🟢"  # Green dot for success
    else:
        return "🔴"  # Red dot for failure

# --- File Selection ---


def select_files_for_analysis(args, audio_db):
    """Simplified file selection based on args and DB state."""
    logger.debug(
        f"DISCOVERY: select_files_for_analysis: force={args.force}, failed={args.failed}")

    # Get database state
    db_features = audio_db.get_all_features(include_failed=True)
    db_files = set(f['filepath'] for f in db_features)
    failed_db_files = set(f['filepath'] for f in db_features if f['failed'])

    logger.debug(
        f"DISCOVERY: files in db={len(db_files)}, failed in db={len(failed_db_files)}")
    if db_files:
        sample_db_files = list(db_files)[:3]
        logger.debug(f"DISCOVERY: Sample files from db: {sample_db_files}")

    if args.failed:
        # Failed mode: only process failed files that aren't in failed directory
        files_to_analyze = []
        for filepath in failed_db_files:
            if '/failed_files' not in filepath:
                files_to_analyze.append(filepath)
        logger.info(
            f"DISCOVERY: Failed mode - {len(files_to_analyze)} files to retry")
        return files_to_analyze, [], [], db_features

    elif args.force:
        # Force mode: process all files except failed ones
        files_to_analyze = audio_db.get_files_needing_analysis()
        # Filter out failed files
        files_to_analyze = [f[0]
                            for f in files_to_analyze if f[0] not in failed_db_files]
        logger.info(
            f"DISCOVERY: Force mode - {len(files_to_analyze)} files to process")
        if files_to_analyze:
            sample_files = files_to_analyze[:3]
            logger.debug(
                f"DISCOVERY: Sample files to analyze (force): {sample_files}")
    else:
        # Normal mode: only new/modified files
        files_to_analyze = audio_db.get_files_needing_analysis()
        files_to_analyze = [f[0] for f in files_to_analyze]
        logger.info(
            f"DISCOVERY: Normal mode - {len(files_to_analyze)} files to process")
        if files_to_analyze:
            sample_files = files_to_analyze[:3]
            logger.debug(
                f"DISCOVERY: Sample files to analyze (normal): {sample_files}")

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

    logger.debug(
        f"DISCOVERY: normal_files={len(normal_files)}, big_files={len(big_files)}")
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
    def __init__(self, audio_analyzer=None):
        self.audio_analyzer = audio_analyzer

    def process(self, files, workers, **kwargs):
        processor = SequentialProcessor(audio_analyzer=self.audio_analyzer)
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
    normal_files, big_files, _, db_features = select_files_for_analysis(
        args, audio_db)

    # Create a combined list that matches the processing order (big files first, then normal files)
    files_to_analyze = big_files + normal_files
    
    # Debug logging for file distribution
    logger.info(f"🔍 DEBUG: File distribution - normal_files={len(normal_files)}, big_files={len(big_files)}")
    if normal_files:
        logger.info(f"🔍 DEBUG: Sample normal files: {normal_files[:3]}")
    if big_files:
        logger.info(f"🔍 DEBUG: Sample big files: {big_files[:3]}")

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

    logger.info(
        f"DISCOVERY: File distribution: {len(normal_files)} normal files, {len(big_files)} big files (>50MB)")

    # Run analysis on files that need it
    failed_files = []

    # Create a single AudioAnalyzer instance for sequential processing
    audio_analyzer = AudioAnalyzer(cache_file=audio_db.cache_file, library=audio_db.library, music=audio_db.music)

    with CLIContextManager(cli, len(files_to_analyze), f"[cyan]Analyzing {len(files_to_analyze)} files...") as (progress, task_id):
        processed_count = 0
        file_start_times = {}  # Track individual file processing times
        
        # Ensure file_start_times is always a dictionary
        if not isinstance(file_start_times, dict):
            file_start_times = {}

        # Pre-update progress bar with first file
        if files_to_analyze:
            _update_progress_bar(progress, task_id, files_to_analyze, 0, len(files_to_analyze),
                                 "[cyan]", "", "", None, True, file_sizes)

        # Step 1: Process big files sequentially
        if big_files:
            logger.info(
                f"Processing {len(big_files)} big files sequentially...")
            sequential_processor = SequentialProcessor(audio_analyzer=audio_analyzer)
            workers = 1  # Sequential processing uses 1 worker

            # Add a periodic update for large files
            start_time = time.time()
            last_update_time = start_time
            
            # Initialize timing for the first file
            if big_files and isinstance(file_start_times, dict):
                first_file = big_files[0]
                first_filepath = first_file[0] if isinstance(first_file, tuple) else first_file
                file_start_times[first_filepath] = time.time()
                first_filename = os.path.basename(first_filepath)
                logger.info(f"SEQUENTIAL: Starting processing {first_filename}")

            # Add timeout and memory monitoring for large files
            def timeout_monitor():
                """Monitor for stuck processes and memory issues."""
                while True:
                    time.sleep(30)  # Check every 30 seconds
                    current_time = time.time()
                    
                    # Check for stuck processes - only warn if no progress for 20 minutes
                    # AND we're not in the middle of processing a large file
                    if current_time - last_update_time > 1200:  # 20 minutes without update
                        # Check if we're currently processing a large file
                        current_file_size = 0
                        current_file_start_time = 0
                        try:
                            if isinstance(file_start_times, dict) and len(file_start_times) > 0:
                                # Get the most recently started file
                                current_file = max(file_start_times.items(), key=lambda x: x[1])
                                current_filepath = current_file[0]
                                current_file_start_time = current_file[1]
                                if os.path.exists(current_filepath):
                                    current_file_size = os.path.getsize(current_filepath) / (1024 * 1024)  # MB
                        except Exception as e:
                            logger.debug(f"Could not check current file size: {e}")
                        
                        # Calculate how long the current file has been processing
                        file_processing_time = current_time - current_file_start_time
                        
                        # Only warn if it's not a large file (>50MB) or if it's been too long even for a large file
                        threshold_mb = int(os.getenv('LARGE_FILE_THRESHOLD', '50'))
                        max_time_for_large_file = 3600  # 1 hour for large files
                        
                        # Don't warn if we're processing a large file and it hasn't been too long
                        if current_file_size >= threshold_mb and file_processing_time < max_time_for_large_file:
                            logger.debug(f"Large file processing ({current_file_size:.1f}MB) for {file_processing_time:.1f}s - not stuck yet")
                        else:
                            logger.warning(f"PROCESSING: No progress for 20 minutes - file may be stuck (current file: {current_file_size:.1f}MB, processing for {file_processing_time:.1f}s)")
                    
                    # Check memory usage
                    try:
                        import psutil
                        memory_info = psutil.virtual_memory()
                        memory_percent = memory_info.percent
                        used_gb = memory_info.used / (1024**3)
                        available_gb = memory_info.available / (1024**3)
                        
                        logger.info(f"Memory usage: {memory_percent:.1f}% ({used_gb:.1f}GB used, {available_gb:.1f}GB available)")
                        
                        # If memory is critical (>90%), force cleanup
                        if memory_percent > 90:
                            logger.error(f"CRITICAL: Memory usage at {memory_percent:.1f}% - forcing cleanup")
                            import gc
                            gc.collect()
                            time.sleep(2)  # Give time for cleanup
                            
                            # Check again after cleanup
                            memory_info = psutil.virtual_memory()
                            memory_percent = memory_info.percent
                            logger.info(f"Memory after cleanup: {memory_percent:.1f}%")
                            
                            # If still critical, log warning
                            if memory_percent > 95:
                                logger.error(f"CRITICAL: Memory still at {memory_percent:.1f}% after cleanup")
                    except Exception as e:
                        logger.debug(f"Could not check memory: {e}")

            # Start timeout monitor in background
            timeout_thread = threading.Thread(target=timeout_monitor, daemon=True)
            timeout_thread.start()

            for features, filepath, db_write_success in sequential_processor.process(big_files, workers, force_reextract=force_reextract):
                processed_count += 1
                filename = os.path.basename(filepath)
                
                # Check for interrupt using global flag
                try:
                    from playlista import is_interrupt_requested
                    if is_interrupt_requested():
                        logger.warning("Interrupt received after completing file")
                        print(f"\n🛑 Interrupt received! Completed {filename}, stopping analysis...")
                        return failed_files
                except ImportError:
                    # If we can't import the function, continue normally
                    pass
                
                # Calculate individual file processing time
                if isinstance(file_start_times, dict) and filepath in file_start_times:
                    file_processing_time = time.time() - file_start_times[filepath]
                    logger.info(f"SEQUENTIAL: {filename} completed in {file_processing_time:.1f}s")
                else:
                    file_processing_time = 0

                # Get status dot for result
                status_dot = _get_status_dot(features, db_write_success)

                # Update progress bar with the file that was just processed
                logger.debug(f"PROGRESS: About to update progress bar for completed file {filename}")
                _update_progress_bar(progress, task_id, files_to_analyze, processed_count - 1, len(files_to_analyze),
                                     "[cyan]", "", status_dot, filepath, True, file_sizes)
                logger.debug(f"PROGRESS: Progress bar updated for {filename}")
                
                # Log completion with timing info
                logger.info(f"SEQUENTIAL: Completed processing {filename} ({file_processing_time:.1f}s)")
                
                # Check if we need to show a periodic update for long-running processes
                current_time = time.time()
                if current_time - last_update_time > 30:  # Update every 30 seconds
                    elapsed = current_time - start_time
                    logger.info(f"SEQUENTIAL: Still processing... ({elapsed:.0f}s elapsed)")
                    last_update_time = current_time
                
                # Update last_update_time for timeout monitoring
                last_update_time = current_time

                logger.debug(
                    f"Processed {processed_count}/{len(files_to_analyze)}: {filename} ({status_dot})")

                if not features or not db_write_success:
                    failed_files.append(filepath)
                    logger.warning(f"Analysis failed for {filepath}")
                else:
                    logger.info(f"Analysis completed for {filepath}")
                
                # Start timing the next file
                logger.debug(f"PROGRESS: Checking if there's a next file to process")
                if processed_count < len(files_to_analyze) and isinstance(file_start_times, dict):
                    next_file = files_to_analyze[processed_count]
                    next_filepath = next_file[0] if isinstance(next_file, tuple) else next_file
                    file_start_times[next_filepath] = time.time()
                    next_filename = os.path.basename(next_filepath)
                    logger.info(f"SEQUENTIAL: Starting processing {next_filename}")
                    logger.debug(f"PROGRESS: Next file timing started for {next_filename}")
                else:
                    logger.debug(f"PROGRESS: No next file to process - processed_count={processed_count}, total_files={len(files_to_analyze)}")

        # Step 2: Process normal files (parallel or sequential based on workers)
        if normal_files:
            # Determine if we should use parallel processing
            workers = getattr(args, 'workers', None)
            
            # If no workers specified, use automatic memory-aware selection
            if workers is None:
                try:
                    from music_analyzer.parallel import get_memory_aware_worker_count
                    workers = get_memory_aware_worker_count()
                    logger.info(f"🔄 AUTO: Using memory-aware worker count: {workers}")
                except Exception as e:
                    logger.warning(f"Could not determine memory-aware worker count: {e}")
                    workers = 1  # Fallback to sequential
            
            use_parallel = workers > 1
            
            # Debug logging
            logger.info(f"🔍 DEBUG: workers={workers}, use_parallel={use_parallel}")
            logger.info(f"🔍 DEBUG: args.workers={getattr(args, 'workers', 'NOT_SET')}")
            logger.info(f"🔍 DEBUG: type(workers)={type(workers)}")
            
            if use_parallel:
                logger.info(
                    f"🔄 PARALLEL MODE: Processing {len(normal_files)} normal files with {workers} workers...")
                logger.info(f"🔄 PARALLEL: Using multiprocessing with {workers} worker processes")
                if getattr(args, 'workers', None) is None:
                    logger.info(f"🔄 AUTO: Automatically selected {workers} workers based on memory")
                
                # Set parallel processing flag to prevent signal handler from triggering during pool termination
                try:
                    from playlista import set_parallel_processing_active, clear_parallel_processing_active
                    set_parallel_processing_active()
                except ImportError:
                    pass
                
                parallel_manager = ParallelWorkerManager()
                for result in parallel_manager.process(normal_files, workers=workers, force_reextract=force_reextract):
                    processed_count += 1
                    filename = os.path.basename(result[1]) # result[1] is the filepath
                    
                    # Check for interrupt using global flag
                    try:
                        from playlista import is_interrupt_requested
                        if is_interrupt_requested():
                            logger.warning("Interrupt received after completing file")
                            print(f"\n🛑 Interrupt received! Completed {filename}, stopping analysis...")
                            return failed_files
                    except ImportError:
                        # If we can't import the function, continue normally
                        pass

                    # Get status dot for result
                    status_dot = _get_status_dot(result[0], result[2]) # result[0] is features, result[1] is filepath, result[2] is db_write_success

                    # Update progress bar with the file that was just processed
                    _update_progress_bar(progress, task_id, files_to_analyze, processed_count - 1, len(files_to_analyze),
                                         "[cyan]", "", status_dot, result[1], True, file_sizes)

                    logger.debug(
                        f"Processed {processed_count}/{len(files_to_analyze)}: {filename} ({status_dot})")

                    if not result[0] or not result[2]:
                        failed_files.append(result[1])
                        logger.warning(f"Analysis failed for {result[1]}")
                    else:
                        logger.info(f"Analysis completed for {result[1]}")
                
                # Clear parallel processing flag
                try:
                    clear_parallel_processing_active()
                except NameError:
                    pass
            else:
                logger.info(
                    f"🐌 SEQUENTIAL MODE: Processing {len(normal_files)} normal files with 1 worker...")
                logger.info(f"🐌 SEQUENTIAL: Using single-threaded processing")
                if getattr(args, 'workers', None) is None:
                    logger.info(f"🐌 AUTO: Automatically selected sequential processing based on memory")
                sequential_manager = SequentialWorkerManager(audio_analyzer=audio_analyzer)
                for result in sequential_manager.process(normal_files, workers=1, force_reextract=force_reextract):
                    processed_count += 1
                    filename = os.path.basename(result[1]) # result[1] is the filepath
                    
                    # Check for interrupt using global flag
                    try:
                        from playlista import is_interrupt_requested
                        if is_interrupt_requested():
                            logger.warning("Interrupt received after completing file")
                            print(f"\n🛑 Interrupt received! Completed {filename}, stopping analysis...")
                            return failed_files
                    except ImportError:
                        # If we can't import the function, continue normally
                        pass

                    # Get status dot for result
                    status_dot = _get_status_dot(result[0], result[2]) # result[0] is features, result[1] is filepath, result[2] is db_write_success

                    # Update progress bar with the file that was just processed
                    _update_progress_bar(progress, task_id, files_to_analyze, processed_count - 1, len(files_to_analyze),
                                         "[cyan]", "", status_dot, result[1], True, file_sizes)

                    logger.debug(
                        f"Processed {processed_count}/{len(files_to_analyze)}: {filename} ({status_dot})")

                    if not result[0] or not result[2]:
                        failed_files.append(result[1])
                        logger.warning(f"Analysis failed for {result[1]}")
                    else:
                        logger.info(f"Analysis completed for {result[1]}")
        else:
            logger.debug("DISCOVERY: No normal files to process in parallel")

        logger.info(
            f"Processing completed. Processed {processed_count} files out of {len(files_to_analyze)}")

    logger.info(
        f"ANALYZE mode completed with {len(failed_files)} failed files")
    return failed_files


def run_force_mode(args, audio_db, cli):
    """FORCE mode: Fast validation and retry of cached files."""
    logger.info("Starting FORCE mode - fast validation and retry")

    # Get invalid files that need retry
    invalid_files = audio_db.get_invalid_files_from_db()
    logger.debug(
        f"Invalid files found: {len(invalid_files) if invalid_files else 0}")

    if not invalid_files:
        logger.info("No invalid files found in database")
        return []

    # Display totals with better visual formatting
    total_files = len(audio_db.get_all_audio_files())
    total_in_db = len(audio_db.get_all_tracks())
    files_to_retry = len(invalid_files)
    
    # Get breakdown of invalid files
    failed_files = audio_db.get_failed_files_from_db()
    skipped_musicnn_files = audio_db.get_files_with_skipped_musicnn()
    failed_count = len(failed_files)
    skipped_musicnn_count = len(skipped_musicnn_files)

    print(f"\n📊 Force Mode Statistics:")
    print(f"   📁 Total music files: {total_files:,}")
    print(f"   💾 Tracks in database: {total_in_db:,}")
    print(f"   🔄 Files to retry: {files_to_retry:,}")
    print(f"   ❌ Failed files: {failed_count:,}")
    print(f"   ⏭️  MusicNN skipped: {skipped_musicnn_count:,}")
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
                # Unmark both failed and MusicNN skipped status
                audio_db.unmark_as_failed(file_path)
                audio_db.unmark_musicnn_skipped(file_path)
            else:
                # Check if this was originally a failed file or just MusicNN skipped
                # Only mark as failed if it was originally a failed file
                cursor = audio_db.conn.cursor()
                cursor.execute("SELECT failed FROM audio_features WHERE file_path = ?", (file_path,))
                result = cursor.fetchone()
                was_failed = result and result[0] == 1 if result else False
                
                if was_failed:
                    logger.error(
                        f"Retry failed for {file_path} - marking as failed")
                    file_info = audio_db._get_file_info(file_path)
                    audio_db._mark_failed(file_info)
                    failed_files.append(file_path)
                else:
                    logger.warning(
                        f"Retry failed for {file_path} but was only MusicNN skipped - keeping as skipped")
                    # Don't mark as failed, just leave it as MusicNN skipped

            logger.debug(f"Retry result for {filename}: {status_dot}")

    logger.info(f"FORCE mode completed with {len(failed_files)} failed files")

    # Check if we were interrupted
    return failed_files


def run_failed_mode(args, audio_db, cli):
    """FAILED mode: Recovery of previously failed files."""
    logger.info("Starting FAILED mode - recovering failed files")

    # Get all failed files from database
    failed_files = audio_db.get_failed_files_from_db()
    logger.debug(
        f"Failed files found: {len(failed_files) if failed_files else 0}")

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
    logger.debug(
        f"Pipeline args: force={args.force}, failed={args.failed}, workers={args.workers}")

    # Debug: Check initial state
    logger.debug(
        f"DISCOVERY: Pipeline starting - total files in db: {len(audio_db.get_all_features())}")
    logger.debug(
        f"DISCOVERY: Pipeline starting - files needing analysis: {len(audio_db.get_files_needing_analysis())}")

    console.print(
        "\n[bold cyan]PIPELINE: Starting default analysis[/bold cyan]")
    console.print("[dim]Analyze new files[/dim]")
    args.force = False
    args.failed = False
    logger.info("Pipeline Stage 1: Running default analysis (new files only)")
    logger.debug(
        f"DISCOVERY: Stage 1 - before run_analysis: force={args.force}, failed={args.failed}")
    res1 = run_analysis(args, audio_db, playlist_db, cli,
                        force_reextract=False, pipeline_mode=True)
    logger.info(
        f"Pipeline Stage 1 result: {len(res1) if res1 else 0} files processed")
    logger.debug(
        f"DISCOVERY: Stage 1 - after run_analysis: returned {len(res1) if res1 else 0} files")
    results.append(('Default', {'processed_this_run': len(
        res1) if res1 else 0, 'failed_this_run': len(res1) if res1 else 0}))
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
    logger.debug(
        f"DISCOVERY: Stage 2 - before run_analysis: force={args.force}, failed={args.failed}")
    res2 = run_analysis(args, audio_db, playlist_db, cli,
                        force_reextract=True, pipeline_mode=True)
    logger.info(
        f"Pipeline Stage 2 result: {len(res2) if res2 else 0} files processed")
    logger.debug(
        f"DISCOVERY: Stage 2 - after run_analysis: returned {len(res2) if res2 else 0} files")
    results.append(('Force', {'processed_this_run': len(
        res2) if res2 else 0, 'failed_this_run': len(res2) if res2 else 0}))
    console.print(
        "[green]PIPELINE: Tags enriching complete (tags updated)[/green]")
    console.print(
        "[dim]───────────────────────────────────────────────[/dim]\n")

    console.print("[bold cyan]PIPELINE: Retrying failed files[/bold cyan]")
    console.print("[dim]Retry failed files[/dim]")
    args.force = False
    args.failed = True
    logger.info("Pipeline Stage 3: Running failed analysis (retry failed files)")
    logger.debug(
        f"DISCOVERY: Stage 3 - before run_analysis: force={args.force}, failed={args.failed}")
    res3 = run_analysis(args, audio_db, playlist_db, cli,
                        force_reextract=True, pipeline_mode=True)
    logger.info(
        f"Pipeline Stage 3 result: {len(res3) if res3 else 0} files processed")
    logger.debug(
        f"DISCOVERY: Stage 3 - after run_analysis: returned {len(res3) if res3 else 0} files")
    # Count files in /music/failed_files after failed step
    import os
    failed_dir = '/music/failed_files'
    try:
        moved_failed = len([f for f in os.listdir(
            failed_dir) if os.path.isfile(os.path.join(failed_dir, f))])
    except Exception:
        moved_failed = 0
    results.append(('Failed', {'processed_this_run': len(
        res3) if res3 else 0, 'failed_this_run': moved_failed}))
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
