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
from music_analyzer.parallel import ParallelProcessor, UserAbortException
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
import logging
logging.getLogger().setLevel(getattr(logging, LOG_LEVEL.upper(), logging.DEBUG))

logger = logging.getLogger(__name__)
BIG_FILE_SIZE_MB = 200

def _update_progress_bar(progress, task_id, files_list, current_index, total_count, 
                        mode_color="[cyan]", mode_text="Processing", status_dot="", 
                        current_filepath=None, is_parallel=True):
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
    
    # Get file size
    try:
        size_mb = os.path.getsize(file_path) / (1024 * 1024)
        file_size_info = f"{size_mb:.1f}MB"
        # Determine processing mode based on file size and context
        if size_mb > BIG_FILE_SIZE_MB:
            processing_mode = "Sequential"  # Big files always use sequential
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
    """Return (normal_files, big_files) to analyze based on args and DB state."""
    # Use container music path instead of host library path for file discovery
    file_list = get_audio_files('/music')
    db_features = audio_db.get_all_features(include_failed=True)
    db_files = set(f['filepath'] for f in db_features)
    failed_files_db = set(f['filepath'] for f in db_features if f['failed'])
    # Exclude files in the failed directory
    failed_dir = os.path.join(os.getenv('CACHE_DIR', '/app/cache'), 'failed_files')
    failed_dir_abs = os.path.abspath(failed_dir)
    file_list = [f for f in file_list if not os.path.abspath(f).startswith(failed_dir_abs)]
    
    # Debug logging
    logger.debug(f"select_files_for_analysis: args.force={args.force}, args.failed={args.failed}")
    logger.debug(f"select_files_for_analysis: total files found={len(file_list)}")
    logger.debug(f"select_files_for_analysis: files in db={len(db_files)}")
    logger.debug(f"select_files_for_analysis: failed files in db={len(failed_files_db)}")
    
    if args.force:
        files_to_analyze = [f for f in file_list if f not in failed_files_db]
        logger.debug(f"select_files_for_analysis: force=True, files_to_analyze={len(files_to_analyze)}")
        logger.debug(f"select_files_for_analysis: force=True, first 5 files={files_to_analyze[:5]}")
    elif args.failed:
        # Directly query the DB for failed files
        import sqlite3
        conn = sqlite3.connect(audio_db.cache_file)
        cur = conn.cursor()
        cur.execute("SELECT file_path FROM audio_features WHERE failed=1")
        files_to_analyze = [row[0] for row in cur.fetchall()]
        conn.close()
        logger.debug(f"select_files_for_analysis: failed=True, files_to_analyze={len(files_to_analyze)}")
        # All files to be processed sequentially
        return files_to_analyze, [], file_list, db_features
    else:
        files_to_analyze = [f for f in file_list if f not in db_files and f not in failed_files_db]
        logger.debug(f"select_files_for_analysis: default mode, files_to_analyze={len(files_to_analyze)}")
    
    def is_big_file(filepath):
        try:
            return os.path.getsize(filepath) > BIG_FILE_SIZE_MB * 1024 * 1024
        except Exception:
            return False
    big_files = [f for f in files_to_analyze if is_big_file(f)]
    normal_files = [f for f in files_to_analyze if not is_big_file(f)]
    logger.debug(f"select_files_for_analysis: normal_files={len(normal_files)}, big_files={len(big_files)}")
    return normal_files, big_files, file_list, db_features

def move_failed_files(audio_db, failed_dir=None):
    import os
    if failed_dir is None:
        failed_dir = '/app/failed_files'
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
                logger.info(f"File {src} is already in the failed_files directory, skipping move.")
                continue
            try:
                shutil.move(src, dst)
                logger.warning(f"Moved failed file to {dst}")
                moved += 1
            except Exception as e:
                logger.error(f"Failed to move {src} to {dst}: {e}")
    if moved > 0:
        logger.info(f"Moved {moved} failed files to '{failed_dir_abs}' and excluded them from analysis.")
    else:
        logger.info(f"No failed files to move to '{failed_dir_abs}'.")

def move_newly_failed_files(audio_db, newly_failed, failed_dir=None):
    import os
    if failed_dir is None:
        failed_dir = '/app/failed_files'
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
            logger.info(f"File {src} is already in the failed_files directory, skipping move.")
            continue
        try:
            if os.path.exists(dst):
                os.remove(dst)
                logger.info(f"Existing file at {dst} deleted before replacement.")
            shutil.move(src, dst)
            logger.warning(f"Moved (or replaced) failed file to {dst}")
            moved += 1
            moved_files.append(src)
        except Exception as e:
            logger.error(f"Failed to move/replace {src} to {dst}: {e}")
    if moved > 0:
        logger.info(f"Moved {moved} newly failed files to '{failed_dir_abs}' and excluded them from analysis.")
        logger.info(f"Full paths: {moved_files}")
    else:
        logger.info(f"No newly failed files to move to '{failed_dir_abs}'.")

# --- Graceful Shutdown ---
def setup_graceful_shutdown():
    stop_event = mp.Event()
    def handle_stop_signal(signum, frame):
        stop_event.set()
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        logger.debug(f"Stop event set: {stop_event.is_set()}")
        # Force cleanup of child processes
        cleanup_child_processes()
        # Force exit after cleanup
        import sys
        sys.exit(0)
    # Handle multiple signal types for Docker compatibility
    import signal
    signal.signal(signal.SIGINT, handle_stop_signal)
    signal.signal(signal.SIGTERM, handle_stop_signal)
    signal.signal(signal.SIGQUIT, handle_stop_signal)
    return stop_event

def cleanup_child_processes():
    parent = psutil.Process(os.getpid())
    for child in parent.children(recursive=True):
        try:
            child.terminate()
        except Exception:
            pass
    gone, alive = psutil.wait_procs(parent.children(recursive=True), timeout=3)
    for p in alive:
        try:
            p.kill()
        except Exception:
            pass
    for child in parent.children(recursive=True):
        try:
            os.kill(child.pid, signal.SIGKILL)
        except Exception:
            pass

# --- Worker Managers ---
class ParallelWorkerManager:
    def __init__(self, stop_event):
        self.stop_event = stop_event
    def process(self, files, workers, status_queue=None, **kwargs):
        processor = ParallelProcessor()
        return processor.process(files, workers=workers, status_queue=status_queue, stop_event=self.stop_event, **kwargs)

class SequentialWorkerManager:
    def __init__(self, stop_event):
        self.stop_event = stop_event
    def process(self, files, workers, **kwargs):
        processor = SequentialProcessor()
        return processor.process(files, workers=workers, stop_event=self.stop_event, **kwargs)

class BigFileWorkerManager:
    def __init__(self, stop_event, audio_db):
        self.stop_event = stop_event
        self.audio_db = audio_db
        self.processes = []
    def process(self, files):
        results = []
        for filepath in files:
            if self.stop_event.is_set():
                break
            queue = mp.Queue()
            def analyze_in_subprocess(filepath, cache_file, library, music, queue, stop_event):
                try:
                    if stop_event.is_set():
                        queue.put(None)
                        return
                    analyzer = AudioAnalyzer(cache_file, library, music)
                    result = analyzer.extract_features(filepath)
                    queue.put(result)
                except Exception as e:
                    queue.put(None)
            p = mp.Process(target=analyze_in_subprocess, args=(filepath, self.audio_db.cache_file, self.audio_db.library, self.audio_db.music, queue, self.stop_event))
            p.start()
            self.processes.append(p)
            try:
                result = queue.get(timeout=600)
                p.join()
            except Exception:
                p.terminate()
                p.join()
                result = None
                # Mark as failed if interrupted
                try:
                    analyzer = AudioAnalyzer(self.audio_db.cache_file, self.audio_db.library, self.audio_db.music)
                    file_info = analyzer._get_file_info(filepath)
                    analyzer._mark_failed(file_info)
                except Exception as e:
                    logger.error(f"Failed to mark interrupted file as failed: {filepath} ({e})")
            results.append((filepath, result))
        # Cleanup any still-alive processes
        for p in self.processes:
            if p.is_alive():
                p.terminate()
            p.join()
        return results

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

def _get_system_resources():
    """Get current system resource usage."""
    try:
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used_gb = memory.used / (1024**3)
        memory_total_gb = memory.total / (1024**3)
        return {
            'cpu_percent': cpu_percent,
            'memory_percent': memory_percent,
            'memory_used_gb': memory_used_gb,
            'memory_total_gb': memory_total_gb
        }
    except Exception as e:
        logger.debug(f"Error getting system resources: {e}")
        return {
            'cpu_percent': 0,
            'memory_percent': 0,
            'memory_used_gb': 0,
            'memory_total_gb': 0
        }

def _get_active_workers():
    """Get count of active worker processes."""
    try:
        current_pid = os.getpid()
        active_workers = 0
        
        # Count child processes that are likely workers
        for proc in psutil.process_iter(['pid', 'ppid', 'name']):
            try:
                proc_info = proc.info
                # Check if it's a child of our process
                if (proc_info['ppid'] == current_pid and 
                    proc_info['name'] and 
                    'python' in proc_info['name'].lower()):
                    active_workers += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied, KeyError):
                continue
        
        # If we can't detect workers properly, use a simple estimate
        if active_workers == 0:
            # Simple fallback - assume 1 worker if we can't detect
            active_workers = 1
        
        return active_workers
    except Exception as e:
        logger.debug(f"Error getting active workers: {e}")
        return 1  # Fallback to 1 worker

def _create_resource_panel(total_workers):
    """Create a resource panel for Rich Live display."""
    resources = _get_system_resources()
    active_workers = _get_active_workers()
    
    # Create resource info string
    cpu_info = f"CPU: {resources['cpu_percent']:.1f}%"
    memory_info = f"RAM: {resources['memory_used_gb']:.1f}GB/{resources['memory_total_gb']:.1f}GB ({resources['memory_percent']:.1f}%)"
    worker_info = f"Workers: {active_workers}/{total_workers}"
    
    resource_text = f"📊 {cpu_info} | {memory_info} | {worker_info}"
    return Panel(resource_text, title="System Resources", border_style="blue")

# --- Main Orchestration ---
def run_analysis(args, audio_db, playlist_db, cli, stop_event=None, force_reextract=False):
    """Run analysis with improved logic based on mode."""
    logger.info(f"Starting analysis with mode: analyze={args.analyze}, force={args.force}, failed={args.failed}")
    
    if args.failed:
        return run_failed_mode(args, audio_db, cli, stop_event)
    elif args.force:
        return run_force_mode(args, audio_db, cli, stop_event)
    else:
        return run_analyze_mode(args, audio_db, cli, stop_event, force_reextract)

def run_analyze_mode(args, audio_db, cli, stop_event, force_reextract):
    """ANALYZE mode: Smart cache comparison and analysis."""
    logger.info("Starting ANALYZE mode - smart cache comparison")
    
    # Get files that need analysis
    files_to_analyze = audio_db.get_files_needing_analysis()
    
    if not files_to_analyze:
        logger.info("No files need analysis - all files are up to date")
        return []
    
    # Display totals with better visual formatting
    total_files = len(audio_db.get_all_audio_files())
    total_in_db = audio_db.get_all_tracks()
    files_to_process = len(files_to_analyze)
    
    print(f"\n📊 Library Statistics:")
    print(f"   📁 Total music files: {total_files:,}")
    print(f"   💾 Tracks in database: {total_in_db:,}")
    print(f"   🔄 Files to process: {files_to_process:,}")
    print(f"   📈 Progress: {total_in_db}/{total_files} ({total_in_db/total_files*100:.1f}% complete)")
    print()  # Add spacing before progress bar
    
    # Run analysis on files that need it
    failed_files = []
    
    # Display resource panel first
    console = Console()
    console.print(_create_resource_panel(workers))
    
    with CLIContextManager(cli, len(files_to_analyze), f"[cyan]Analyzing {len(files_to_analyze)} files...") as (progress, task_id):
        processor = ParallelProcessor() if not args.force_sequential else SequentialProcessor()
        workers = args.workers or max(1, mp.cpu_count())
        
        # Log which processor is being used
        processor_type = "Sequential" if args.force_sequential else "Parallel"
        logger.info(f"Using {processor_type} processor with {workers} workers")
        logger.info(f"Starting to process {len(files_to_analyze)} files")
        
        # Pre-update progress bar with first file
        if files_to_analyze:
            _update_progress_bar(progress, task_id, files_to_analyze, 0, len(files_to_analyze), 
                               "[cyan]", "", "", None, not args.force_sequential)
        
        # Prepare list of file paths only for processing
        file_paths_only = [item[0] if isinstance(item, tuple) else item for item in files_to_analyze]
        
        processed_count = 0
        for features, filepath, db_write_success in processor.process(file_paths_only, workers, force_reextract=force_reextract):
            if stop_event and stop_event.is_set():
                break
            
            processed_count += 1
            filename = os.path.basename(filepath)
            
            # Get status dot for previous file result
            status_dot = _get_status_dot(features, db_write_success)
            
            # Update progress bar with next file
            _update_progress_bar(progress, task_id, files_to_analyze, processed_count, len(files_to_analyze), 
                               "[cyan]", "", status_dot, None, not args.force_sequential)
            
            logger.debug(f"Processed {processed_count}/{len(files_to_analyze)}: {filename} ({status_dot})")
            
            if not features or not db_write_success:
                failed_files.append(filepath)
                logger.warning(f"Analysis failed for {filepath}")
            else:
                logger.info(f"Analysis completed for {filepath}")
        
        logger.info(f"Processing loop completed. Processed {processed_count} files out of {len(files_to_analyze)}")
    
    logger.info(f"ANALYZE mode completed with {len(failed_files)} failed files")
    return failed_files

def run_force_mode(args, audio_db, cli, stop_event):
    """FORCE mode: Fast validation and retry of cached files."""
    logger.info("Starting FORCE mode - fast validation and retry")
    
    # Get invalid files that need retry
    invalid_files = audio_db.get_invalid_files_from_db()
    
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
    print(f"   📈 Database health: {total_in_db}/{total_files} ({total_in_db/total_files*100:.1f}% valid)")
    print()  # Add spacing before progress bar
    
    # Retry analysis for invalid files
    failed_files = []
    with CLIContextManager(cli, len(invalid_files), f"[yellow]Retrying {len(invalid_files)} invalid files...") as (progress, task_id):
        # Log processor type for force mode (always sequential for retries)
        logger.info("Using Sequential processor for force mode retries")
        
        # Pre-update progress bar with first file
        if invalid_files:
            _update_progress_bar(progress, task_id, invalid_files, 0, len(invalid_files), 
                               "[yellow]", "", "", None, True)  # Handles tuple or str
        
        # Prepare list of file paths only for processing
        file_paths_only = [item[0] if isinstance(item, tuple) else item for item in invalid_files]
        
        processed_count = 0
        for file_path in file_paths_only:
            if stop_event and stop_event.is_set():
                break
            
            processed_count += 1
            filename = os.path.basename(file_path)
            
            # Update progress bar with current file
            _update_progress_bar(progress, task_id, invalid_files, processed_count, len(invalid_files), 
                               "[yellow]", "", "", None, True)  # Handles tuple or str
            
            logger.info(f"Retrying analysis for {file_path}")
            success, features = audio_db.retry_analysis_with_backoff(file_path, max_attempts=3)
            
            # Get status dot for result
            status_dot = _get_status_dot(features, success)
            
            if success:
                logger.info(f"Retry successful for {file_path}")
                audio_db.unmark_as_failed(file_path)
            else:
                logger.error(f"Retry failed for {file_path} - marking as failed")
                file_info = audio_db._get_file_info(file_path)
                audio_db._mark_failed(file_info)
                failed_files.append(file_path)
            
            logger.debug(f"Retry result for {filename}: {status_dot}")
    
    logger.info(f"FORCE mode completed with {len(failed_files)} failed files")
    return failed_files

def run_failed_mode(args, audio_db, cli, stop_event):
    """FAILED mode: Recovery of previously failed files."""
    logger.info("Starting FAILED mode - recovering failed files")
    
    # Get all failed files from database
    failed_files = audio_db.get_failed_files_from_db()
    
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
    print(f"   📈 Success rate: {total_in_db}/{total_files} ({total_in_db/total_files*100:.1f}% successful)")
    print()  # Add spacing before progress bar
    
    # Retry each failed file up to 3 times
    still_failed = []
    with CLIContextManager(cli, len(failed_files), f"[red]Recovering {len(failed_files)} failed files...") as (progress, task_id):
        # Log processor type for failed mode (always sequential for retries)
        logger.info("Using Sequential processor for failed mode retries")
        
        # Pre-update progress bar with first file
        if failed_files:
            _update_progress_bar(progress, task_id, failed_files, 0, len(failed_files), 
                               "[red]", "", "", None, False)  # Handles tuple or str
        
        # Prepare list of file paths only for processing
        file_paths_only = [item[0] if isinstance(item, tuple) else item for item in failed_files]
        
        processed_count = 0
        for file_path in file_paths_only:
            if stop_event and stop_event.is_set():
                break
            
            processed_count += 1
            filename = os.path.basename(file_path)
            
            # Update progress bar with current file
            _update_progress_bar(progress, task_id, failed_files, processed_count, len(failed_files), 
                               "[red]", "", "", None, False)  # Handles tuple or str
            
            logger.info(f"Retrying failed file: {file_path}")
            success, features = audio_db.retry_analysis_with_backoff(file_path, max_attempts=3)
            
            # Get status dot for result
            status_dot = _get_status_dot(features, success)
            
            if success:
                logger.info(f"Recovery successful for {file_path}")
                audio_db.unmark_as_failed(file_path)
            else:
                logger.error(f"Recovery failed for {file_path} - moving to failed directory")
                if audio_db.move_to_failed_directory(file_path):
                    still_failed.append(file_path)
                else:
                    logger.error(f"Failed to move {file_path} to failed directory")
            
            logger.debug(f"Recovery result for {filename}: {status_dot}")
    
    logger.info(f"FAILED mode completed with {len(still_failed)} still failed files")
    return still_failed

def run_pipeline(args, audio_db, playlist_db, cli, stop_event=None):
    from rich.table import Table
    from rich.console import Console
    results = []
    console = Console()
    console.print("\n[bold cyan]PIPELINE: Starting default analysis[/bold cyan]")
    console.print("[dim]Analyze new files[/dim]")
    args.force = False
    args.failed = False
    res1 = run_analysis(args, audio_db, playlist_db, cli, stop_event=stop_event, force_reextract=False, pipeline_mode=True)
    results.append(('Default', res1))
    console.print("[green]PIPELINE: Default analysis complete (new files analyzed)[/green]")
    console.print("[dim]───────────────────────────────────────────────[/dim]\n")

    console.print("[bold cyan]PIPELINE: Enriching missing tags from MusicBrainz and Last.fm (if API provided)[/bold cyan]")
    console.print("[dim]Enriching tags from MusicBrainz and Last.fm[/dim]")
    args.force = True
    args.failed = False
    res2 = run_analysis(args, audio_db, playlist_db, cli, stop_event=stop_event, force_reextract=True, pipeline_mode=True)
    results.append(('Force', res2))
    console.print("[green]PIPELINE: Tags enriching complete (tags updated)[/green]")
    console.print("[dim]───────────────────────────────────────────────[/dim]\n")

    console.print("[bold cyan]PIPELINE: Retrying failed files[/bold cyan]")
    console.print("[dim]Retry failed files[/dim]")
    args.force = False
    args.failed = True
    res3 = run_analysis(args, audio_db, playlist_db, cli, stop_event=stop_event, force_reextract=True, pipeline_mode=True)
    # Count files in /app/failed_files after failed step
    import os
    failed_dir = '/app/failed_files'
    try:
        moved_failed = len([f for f in os.listdir(failed_dir) if os.path.isfile(os.path.join(failed_dir, f))])
    except Exception:
        moved_failed = 0
    results.append(('Failed', res3))
    console.print("[green]PIPELINE: Failed files retry complete (failures handled)[/green]")
    console.print("[dim]───────────────────────────────────────────────[/dim]\n")

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
    console.print("\n[bold green]PIPELINE: Complete. Now you can start generating playlists![/bold green]\n")

# --- File Discovery Helper ---
def get_audio_files(music: str) -> List[str]:
    import os
    file_list = []
    valid_ext = ('.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac', '.opus')
    failed_dir = os.path.abspath('/app/failed_files')
    for root, _, files in os.walk(music):
        abs_root = os.path.abspath(root)
        # Skip failed_files directory
        if abs_root.startswith(failed_dir):
            continue
        for file in files:
            file_lower = file.lower()
            if file_lower.endswith(valid_ext):
                file_list.append(os.path.join(root, file))
    logger.info(f"Found {len(file_list)} audio files in {music}")
    return file_list 