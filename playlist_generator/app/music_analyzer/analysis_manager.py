import os
import multiprocessing
import signal
import logging
import psutil
import sqlite3
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.console import Console
from music_analyzer.parallel import ParallelProcessor, UserAbortException
from music_analyzer.sequential import SequentialProcessor
from music_analyzer.feature_extractor import AudioAnalyzer
import json
import shutil
from utils.logging_setup import setup_colored_logging
import os
setup_colored_logging()
log_level = os.getenv('LOG_LEVEL', 'INFO')
import logging
logging.getLogger().setLevel(getattr(logging, log_level.upper(), logging.INFO))

logger = logging.getLogger()
BIG_FILE_SIZE_MB = 200

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
    stop_event = multiprocessing.Event()
    def handle_stop_signal(signum, frame):
        stop_event.set()
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        logger.debug(f"Stop event set: {stop_event.is_set()}")
        # Force cleanup of child processes
        cleanup_child_processes()
    # Handle multiple signal types for Docker compatibility
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
            queue = multiprocessing.Queue()
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
            p = multiprocessing.Process(target=analyze_in_subprocess, args=(filepath, self.audio_db.cache_file, self.audio_db.library, self.audio_db.music, queue, self.stop_event))
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

# --- Main Orchestration ---
def run_analysis(args, audio_db, playlist_db, cli, stop_event=None, force_reextract=False, pipeline_mode=False):
    if stop_event is None:
        stop_event = setup_graceful_shutdown()
    # Only move failed files before analysis if not --failed mode
    if not getattr(args, 'failed', False):
        move_failed_files(audio_db, failed_dir='/app/failed_files')
    normal_files, big_files, file_list, db_features = select_files_for_analysis(args, audio_db)
    failed_files = []
    processed_count = 0
    total_files = len(normal_files) + len(big_files)
    progress = create_progress_bar(total_files)
    MAX_SEQUENTIAL_RETRIES = 3
    failed_retries = {}
    with progress:
        task_id = progress.add_task(f"Analyzing: (0/{total_files})", total=total_files, trackinfo="")
        if args.failed:
            # Only process failed files sequentially, with in-memory retry counter
            retry_counter = {}
            processed_files_set = set()
            files_to_retry = normal_files[:]
            # Track files already failed before this run
            already_failed = set(f['filepath'] for f in db_features if f.get('failed'))
            last_error = {}
            failed_dir = '/app/failed_files'
            failed_dir_abs = os.path.abspath(failed_dir)
            if not os.path.exists(failed_dir_abs):
                os.makedirs(failed_dir_abs)
            moved_files = []
            error_summary = {}
            while files_to_retry:
                # Remove files that have already failed 3 times
                files_to_retry = [f for f in files_to_retry if retry_counter.get(f, 0) < 3]
                if not files_to_retry:
                    break
                # Log the files being retried and their retry count
                logger.info(f"Retry round: {[(os.path.basename(f), retry_counter.get(f, 0)) for f in files_to_retry]}")
                next_retry = []
                for filepath in files_to_retry:
                    count = retry_counter.get(filepath, 0)
                    # Only increment processed_count for unique files
                    if filepath not in processed_files_set:
                        processed_count += 1
                        processed_files_set.add(filepath)
                    if count >= 3:
                        # Mark as failed in DB
                        conn = sqlite3.connect(audio_db.cache_file)
                        cur = conn.cursor()
                        cur.execute("UPDATE audio_features SET failed = 1 WHERE file_path = ?", (filepath,))
                        conn.commit()
                        conn.close()
                        logger.warning(f"File {filepath} failed 3 times. Marking as failed and skipping for the rest of this run.")
                        # Move the file immediately
                        if os.path.exists(filepath):
                            dst = os.path.join(failed_dir_abs, os.path.basename(filepath))
                            if os.path.abspath(filepath) == os.path.abspath(dst):
                                logger.info(f"File {filepath} is already in the failed_files directory, skipping move.")
                            else:
                                try:
                                    if os.path.exists(dst):
                                        os.remove(dst)
                                        logger.info(f"Existing file at {dst} deleted before replacement.")
                                    shutil.move(filepath, dst)
                                    logger.warning(f"Moved (or replaced) failed file to {dst}")
                                    moved_files.append(filepath)
                                    error_summary[filepath] = last_error.get(filepath, 'No error captured.')
                                    # Delete from DB
                                    conn = sqlite3.connect(audio_db.cache_file)
                                    cur = conn.cursor()
                                    cur.execute("DELETE FROM audio_features WHERE file_path = ?", (filepath,))
                                    conn.commit()
                                    conn.close()
                                    logger.info(f"Deleted DB entry for {filepath}")
                                except Exception as e:
                                    logger.error(f"Failed to move/replace {filepath} to {dst}: {e}")
                        continue
                    filename = os.path.basename(filepath)
                    max_len = 50
                    if len(filename) > max_len:
                        display_name = filename[:max_len-3] + "..."
                    else:
                        display_name = filename
                    try:
                        size_mb = os.path.getsize(filepath) / (1024 * 1024)
                    except Exception:
                        size_mb = 0
                    progress.update(
                        task_id,
                        advance=1,
                        description=f"Analyzing: {display_name} ({processed_count}/{total_files})",
                        trackinfo=f"{size_mb:.1f} MB",
                        refresh=True
                    )
                    # Process the file
                    seq_manager = SequentialWorkerManager(stop_event)
                    features = None
                    error_msg = None
                    try:
                        for f, fp in seq_manager.process([filepath], workers=1, force_reextract=True):
                            features = f
                    except Exception as e:
                        error_msg = str(e)
                    if not features:
                        retry_counter[filepath] = count + 1
                        logger.warning(f"Retrying file {filepath} (retry count: {retry_counter[filepath]})")
                        last_error[filepath] = error_msg or 'Unknown error or extraction failure.'
                        if retry_counter[filepath] < 3:
                            next_retry.append(filepath)
                            logger.info(f"File {filepath} failed this round, will retry.")
                        else:
                            # Mark as failed in DB and move the file immediately
                            conn = sqlite3.connect(audio_db.cache_file)
                            cur = conn.cursor()
                            cur.execute("UPDATE audio_features SET failed = 1 WHERE file_path = ?", (filepath,))
                            conn.commit()
                            conn.close()
                            logger.warning(f"File {filepath} failed 3 times. Marking as failed and skipping for the rest of this run.")
                            if os.path.exists(filepath):
                                dst = os.path.join(failed_dir_abs, os.path.basename(filepath))
                                if os.path.abspath(filepath) == os.path.abspath(dst):
                                    logger.info(f"File {filepath} is already in the failed_files directory, skipping move.")
                                else:
                                    try:
                                        if os.path.exists(dst):
                                            os.remove(dst)
                                            logger.info(f"Existing file at {dst} deleted before replacement.")
                                        shutil.move(filepath, dst)
                                        logger.warning(f"Moved (or replaced) failed file to {dst}")
                                        moved_files.append(filepath)
                                        error_summary[filepath] = last_error.get(filepath, 'No error captured.')
                                        # Delete from DB
                                        conn = sqlite3.connect(audio_db.cache_file)
                                        cur = conn.cursor()
                                        cur.execute("DELETE FROM audio_features WHERE file_path = ?", (filepath,))
                                        conn.commit()
                                        conn.close()
                                        logger.info(f"Deleted DB entry for {filepath}")
                                    except Exception as e:
                                        logger.error(f"Failed to move/replace {filepath} to {dst}: {e}")
                        continue
                    elif features:
                        # Only reset failed if feature extraction is truly successful
                        conn = sqlite3.connect(audio_db.cache_file)
                        cur = conn.cursor()
                        cur.execute("UPDATE audio_features SET failed = 0 WHERE file_path = ?", (filepath,))
                        conn.commit()
                        conn.close()
                        logger.info(f"File {filepath} succeeded on retry {count+1 if count else 1}.")
                logger.info(f"End of retry round. Files to retry next: {[(os.path.basename(f), retry_counter.get(f, 0)) for f in next_retry]}")
                files_to_retry = next_retry
            # After retry loop, print summary
            if moved_files:
                logger.info(f"Moved {len(moved_files)} newly failed files to '{failed_dir_abs}' and excluded them from analysis.")
                logger.info(f"Full paths: {moved_files}")
                logger.info(f"Newly failed files and errors: {error_summary}")
            else:
                logger.info(f"No newly failed files to move to '{failed_dir_abs}'.")
            # Sequential for big files (with force_reextract)
            if big_files:
                from music_analyzer.feature_extractor import AudioAnalyzer
                analyzer = AudioAnalyzer(audio_db.cache_file)
                seq_manager = SequentialWorkerManager(stop_event)
                for features, filepath in seq_manager.process(big_files, workers=1, force_reextract=force_reextract):
                    processed_count += 1
                    filename = os.path.basename(filepath)
                    max_len = 50
                    if len(filename) > max_len:
                        display_name = filename[:max_len-3] + "..."
                    else:
                        display_name = filename
                    try:
                        size_mb = os.path.getsize(filepath) / (1024 * 1024)
                    except Exception:
                        size_mb = 0
                    progress.update(
                        task_id,
                        advance=1,
                        description=f"Analyzing: {display_name} ({processed_count}/{total_files})",
                        trackinfo=f"{size_mb:.1f} MB",
                        refresh=True
                    )
                    # After processing, check if metadata is present
                    file_info = analyzer._get_file_info(filepath)
                    meta = None
                    try:
                        conn = sqlite3.connect(audio_db.cache_file)
                        cur = conn.cursor()
                        cur.execute("SELECT metadata FROM audio_features WHERE file_path = ?", (filepath,))
                        row = cur.fetchone()
                        meta = json.loads(row[0]) if row and row[0] else {}
                        conn.close()
                    except Exception as e:
                        logger.error(f"Error reading metadata for {filepath}: {e}")
                        meta = {}
                    if not meta or not any(meta.values()):
                        # If metadata is still empty, set failed=1
                        conn = sqlite3.connect(audio_db.cache_file)
                        cur = conn.cursor()
                        cur.execute("UPDATE audio_features SET failed = 1 WHERE file_path = ?", (filepath,))
                        conn.commit()
                        conn.close()
                        logger.warning(f"Big file {filepath} has empty metadata after --force, marked as failed.")
        else:
            # Parallel for normal files (with force_reextract)
            par_manager = ParallelWorkerManager(stop_event)
            if normal_files:
                logger.debug(f"Starting parallel processing of {len(normal_files)} normal files")
                from music_analyzer.feature_extractor import AudioAnalyzer
                analyzer = AudioAnalyzer(audio_db.cache_file)
                for features, filepath in par_manager.process(normal_files, workers=args.workers or multiprocessing.cpu_count(), force_reextract=force_reextract, enforce_fail_limit=False):
                    if stop_event.is_set():
                        logger.debug("Stop event set, breaking from processing loop")
                        break
                    processed_count += 1
                    logger.debug(f"Processed file {processed_count}/{total_files}: {filepath}")
                    if filepath:
                        filename = os.path.basename(filepath)
                        max_len = 50
                        if len(filename) > max_len:
                            display_name = filename[:max_len-3] + "..."
                        else:
                            display_name = filename
                        try:
                            size_mb = os.path.getsize(filepath) / (1024 * 1024)
                        except Exception:
                            size_mb = 0
                        progress.update(
                            task_id,
                            advance=1,
                            description=f"Analyzing: {display_name} ({processed_count}/{total_files})",
                            trackinfo=f"{size_mb:.1f} MB",
                            refresh=True
                        )
                        logger.debug(f"Updated progress for file: {display_name}")
                    else:
                        filename = "Unknown"
                        size_mb = 0
                    progress.update(
                        task_id,
                        advance=1,
                        description=f"Analyzing: {display_name} ({processed_count}/{total_files})",
                        trackinfo=f"{size_mb:.1f} MB",
                        refresh=True
                    )
                    logger.debug(f"Features: {features}")
                    logger.debug(f"Completed processing file {processed_count}/{total_files}")
                
                logger.debug(f"Parallel processing completed. Processed {processed_count} files out of {total_files}")
                logger.debug(f"Big files to process: {len(big_files)}")
                
                # Sequential for big files (with force_reextract)
                if big_files:
                    logger.debug(f"Starting sequential processing of {len(big_files)} big files")
                    seq_manager = SequentialWorkerManager(stop_event)
                    for features, filepath in seq_manager.process(big_files, workers=1, force_reextract=force_reextract):
                        if stop_event.is_set():
                            logger.debug("Stop event set, breaking from sequential processing loop")
                            break
                        processed_count += 1
                        logger.debug(f"Processed big file {processed_count}/{total_files}: {filepath}")
                        filename = os.path.basename(filepath)
                        max_len = 50
                        if len(filename) > max_len:
                            display_name = filename[:max_len-3] + "..."
                        else:
                            display_name = filename
                        try:
                            size_mb = os.path.getsize(filepath) / (1024 * 1024)
                        except Exception:
                            size_mb = 0
                        progress.update(
                            task_id,
                            advance=1,
                            description=f"Analyzing: {display_name} ({processed_count}/{total_files})",
                            trackinfo=f"{size_mb:.1f} MB",
                            refresh=True
                        )
                        logger.debug(f"Completed processing big file {processed_count}/{total_files}")
                    
                    logger.debug(f"Sequential processing completed. Total processed: {processed_count}/{total_files}")
            # Big files in subprocesses
            if big_files:
                big_manager = BigFileWorkerManager(stop_event, audio_db)
                for filepath, result in big_manager.process(big_files):
                    if stop_event.is_set():
                        break
                    filename = os.path.basename(filepath)
                    max_len = 50
                    if len(filename) > max_len:
                        display_name = filename[:max_len-3] + "..."
                    else:
                        display_name = filename
                    try:
                        size_mb = os.path.getsize(filepath) / (1024 * 1024)
                    except Exception:
                        size_mb = 0
                    # Show which big file is being processed before starting
                    progress.update(
                        task_id,
                        description=f"Analyzing: {display_name} ({processed_count+1}/{total_files})",
                        trackinfo=f"{size_mb:.1f} MB",
                        advance=1,
                        refresh=True
                    )
                    import time
                    time.sleep(0.5)  # Give the bar a chance to render
                    # Start the subprocess and refresh the bar while waiting
                    p = None
                    for proc in big_manager.processes:
                        if proc.is_alive():
                            p = proc
                            break
                    start_time = time.time()
                    while p and p.is_alive():
                        elapsed = int(time.time() - start_time)
                        progress.update(
                            task_id,
                            description=f"Analyzing (big file): {display_name} ({size_mb:.1f} MB) ({processed_count+1}/{total_files}) [Elapsed: {elapsed}s]"
                        )
                        time.sleep(0.3)
                    processed_count += 1
                    progress.update(
                        task_id,
                        advance=1,
                        refresh=True
                    )
                    logger.debug(f"Features: {result}")
                    if not result or not result[0]:
                        failed_files.append(filepath)
    # After processing (always show summary)
    total_found = len(file_list)
    total_in_db = len(audio_db.get_all_features(include_failed=True))
    total_failed = len([f for f in audio_db.get_all_features(include_failed=True) if f['failed']])
    processed_this_run = processed_count
    failed_this_run = len(failed_files)
    stats = playlist_db.get_library_statistics()
    if not pipeline_mode:
        cli.show_analysis_summary(
            stats=stats,
            processed_this_run=processed_this_run,
            failed_this_run=failed_this_run,
            total_found=total_found,
            total_in_db=total_in_db,
            total_failed=total_failed
        )
    return {
        'processed_this_run': processed_this_run,
        'failed_this_run': failed_this_run,
        'total_found': total_found,
        'total_in_db': total_in_db,
        'total_failed': total_failed
    }

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
def get_audio_files(music: str) -> list[str]:
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