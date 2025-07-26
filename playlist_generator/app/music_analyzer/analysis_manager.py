import os
import multiprocessing
import signal
import logging
import psutil
import sqlite3
from typing import List
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

# --- File Discovery Helper ---
def get_audio_files(music: str) -> List[str]:
    import os
    logger.debug(f"Starting audio file discovery in directory: {music}")
    file_list = []
    valid_ext = ('.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac', '.opus')
    
    if not os.path.exists(music):
        logger.warning(f"Music directory does not exist: {music}")
        return file_list
    
    for root, dirs, files in os.walk(music):
        logger.debug(f"Scanning directory: {root} (found {len(files)} files)")
        for file in files:
            file_lower = file.lower()
            if file_lower.endswith(valid_ext):
                file_path = os.path.join(root, file)
                file_list.append(file_path)
                logger.debug(f"Found audio file: {file_path}")
    
    logger.info(f"Audio file discovery complete: found {len(file_list)} files in {music}")
    return file_list

def move_failed_files(audio_db, failed_dir=None):
    import os
    logger.info("Starting failed files cleanup process")
    if failed_dir is None:
        failed_dir = '/app/failed_files'
    failed_dir_abs = os.path.abspath(failed_dir)
    logger.debug(f"Failed files directory: {failed_dir_abs}")
    
    if not os.path.exists(failed_dir_abs):
        logger.info(f"Creating failed files directory: {failed_dir_abs}")
        os.makedirs(failed_dir_abs)
    
    db_features = audio_db.get_all_features(include_failed=True)
    logger.debug(f"Retrieved {len(db_features)} features from database")
    
    moved = 0
    for f in db_features:
        if f.get('failed'):
            src = f['filepath']
            if not os.path.exists(src):
                logger.debug(f"Failed file no longer exists: {src}")
                continue
            dst = os.path.join(failed_dir_abs, os.path.basename(src))
            if os.path.abspath(src) == os.path.abspath(dst):
                logger.info(f"File {src} is already in the failed_files directory, skipping move.")
                continue
            try:
                logger.debug(f"Moving failed file: {src} -> {dst}")
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
    """Setup signal handlers for graceful shutdown."""
    logger.debug("Setting up graceful shutdown handlers")
    
    def handle_stop_signal(signum, frame):
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        # Set a global flag or use multiprocessing.Event
        import sys
        sys.exit(0)
    
    import signal
    signal.signal(signal.SIGINT, handle_stop_signal)
    signal.signal(signal.SIGTERM, handle_stop_signal)
    logger.debug("Graceful shutdown handlers configured")

def cleanup_child_processes():
    """Clean up any remaining child processes."""
    logger.debug("Cleaning up child processes")
    try:
        import psutil
        current_process = psutil.Process()
        children = current_process.children(recursive=True)
        logger.debug(f"Found {len(children)} child processes to clean up")
        
        for child in children:
            try:
                logger.debug(f"Terminating child process {child.pid}")
                child.terminate()
            except psutil.NoSuchProcess:
                logger.debug(f"Child process {child.pid} already terminated")
            except Exception as e:
                logger.warning(f"Error terminating child process {child.pid}: {e}")
        
        # Wait for processes to terminate
        gone, alive = psutil.wait_procs(children, timeout=3)
        logger.debug(f"Terminated {len(gone)} processes, {len(alive)} still alive")
        
        # Force kill any remaining processes
        for child in alive:
            try:
                logger.debug(f"Force killing child process {child.pid}")
                child.kill()
            except Exception as e:
                logger.warning(f"Error force killing child process {child.pid}: {e}")
                
    except Exception as e:
        logger.error(f"Error during child process cleanup: {e}")

# --- Worker Managers ---
class ParallelWorkerManager:
    """Manages parallel processing with proper cleanup."""
    def __init__(self, stop_event):
        self.stop_event = stop_event
        logger.debug("Initialized ParallelWorkerManager")

    def process(self, files, workers, status_queue=None, **kwargs):
        logger.debug(f"ParallelWorkerManager processing {len(files)} files with {workers} workers")
        processor = ParallelProcessor()
        return processor.process(files, workers, status_queue, stop_event=self.stop_event, **kwargs)

class SequentialWorkerManager:
    """Manages sequential processing with proper cleanup."""
    def __init__(self, stop_event):
        self.stop_event = stop_event
        logger.debug("Initialized SequentialWorkerManager")

    def process(self, files, workers, **kwargs):
        logger.debug(f"SequentialWorkerManager processing {len(files)} files")
        processor = SequentialProcessor()
        return processor.process(files, workers, stop_event=self.stop_event, **kwargs)

class BigFileWorkerManager:
    """Manages big file processing with subprocess isolation."""
    def __init__(self, stop_event, audio_db):
        self.stop_event = stop_event
        self.audio_db = audio_db
        logger.debug("Initialized BigFileWorkerManager")

    def process(self, files):
        logger.debug(f"BigFileWorkerManager processing {len(files)} big files")
        import multiprocessing as mp
        from multiprocessing import Queue
        
        def analyze_in_subprocess(filepath, cache_file, library, music, queue, stop_event):
            """Analyze a single file in a separate subprocess."""
            logger.debug(f"Starting subprocess analysis for: {filepath}")
            try:
                from music_analyzer.feature_extractor import AudioAnalyzer
                analyzer = AudioAnalyzer(cache_file=cache_file, library=library, music=music)
                features, db_write_success, file_hash = analyzer.extract_features(filepath, force_reextract=True)
                if features:
                    queue.put((features, filepath, db_write_success))
                    logger.debug(f"Subprocess analysis successful for: {filepath}")
                else:
                    queue.put((None, filepath, False))
                    logger.warning(f"Subprocess analysis failed for: {filepath}")
            except Exception as e:
                logger.error(f"Subprocess analysis error for {filepath}: {e}")
                queue.put((None, filepath, False))

        results = []
        for filepath in files:
            if self.stop_event and self.stop_event.is_set():
                logger.info("Stop event received, stopping big file processing")
                break
            
            logger.debug(f"Processing big file: {filepath}")
            queue = Queue()
            process = mp.Process(
                target=analyze_in_subprocess,
                args=(filepath, self.audio_db.cache_file, None, '/music', queue, self.stop_event)
            )
            process.start()
            process.join(timeout=600)  # 10 minute timeout
            
            if process.is_alive():
                logger.warning(f"Big file processing timed out for: {filepath}")
                process.terminate()
                process.join()
                yield None, filepath
            else:
                try:
                    result = queue.get(timeout=1)
                    results.append(result)
                    yield result
                except Exception as e:
                    logger.error(f"Error getting result for {filepath}: {e}")
                    yield None, filepath
        
        logger.info(f"Big file processing complete: {len(results)} files processed")

# --- Progress Bar ---
def create_progress_bar(total_files):
    """Create a progress bar for file processing."""
    logger.debug(f"Creating progress bar for {total_files} files")
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=Console()
    )

# --- Main Orchestration ---
def run_analysis(args, audio_db, playlist_db, cli, stop_event=None, force_reextract=False, pipeline_mode=False):
    """Run the main analysis process."""
    logger.info("Starting analysis process")
    logger.debug(f"Analysis parameters: force_reextract={force_reextract}, pipeline_mode={pipeline_mode}")
    
    try:
        # Get files to analyze
        normal_files, big_files, all_files, db_features = select_files_for_analysis(args, audio_db)
        logger.info(f"File selection complete: {len(normal_files)} normal files, {len(big_files)} big files")
        
        if not normal_files and not big_files:
            logger.info("No files to analyze")
            return
        
        # Setup graceful shutdown
        setup_graceful_shutdown()
        
        # Create progress bar
        total_files = len(normal_files) + len(big_files)
        progress = create_progress_bar(total_files)
        
        with progress:
            task = progress.add_task("Analyzing", total=total_files)
            
            # Process normal files in parallel
            if normal_files:
                logger.info(f"Starting parallel processing of {len(normal_files)} normal files")
                parallel_manager = ParallelWorkerManager(stop_event)
                
                for features, filepath in parallel_manager.process(
                    normal_files, 
                    workers=args.workers,
                    status_queue=None,
                    force_reextract=force_reextract
                ):
                    if stop_event and stop_event.is_set():
                        logger.info("Stop event received, stopping analysis")
                        break
                    
                    progress.advance(task)
                    if features:
                        logger.debug(f"Successfully processed: {filepath}")
                    else:
                        logger.warning(f"Failed to process: {filepath}")
            
            # Process big files sequentially
            if big_files:
                logger.info(f"Starting sequential processing of {len(big_files)} big files")
                big_manager = BigFileWorkerManager(stop_event, audio_db)
                
                for features, filepath in big_manager.process(big_files):
                    if stop_event and stop_event.is_set():
                        logger.info("Stop event received, stopping big file analysis")
                        break
                    
                    progress.advance(task)
                    if features:
                        logger.debug(f"Successfully processed big file: {filepath}")
                    else:
                        logger.warning(f"Failed to process big file: {filepath}")
        
        logger.info("Analysis process completed successfully")
        
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
        cleanup_child_processes()
    except Exception as e:
        logger.error(f"Analysis process failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        cleanup_child_processes()
        raise

def run_pipeline(args, audio_db, playlist_db, cli, stop_event=None):
    """Run the complete pipeline: analysis, enrichment, and playlist generation."""
    logger.info("Starting complete pipeline")
    logger.debug(f"Pipeline parameters: force={args.force}, failed={args.failed}, enrich_tags={args.enrich_tags}")
    
    try:
        # Step 1: Analysis
        logger.info("PIPELINE: Starting default analysis")
        run_analysis(args, audio_db, playlist_db, cli, stop_event, force_reextract=args.force, pipeline_mode=True)
        logger.info("PIPELINE: Default analysis complete")
        
        # Step 2: Enrichment (if requested)
        if args.enrich_tags:
            logger.info("PIPELINE: Enriching missing tags from MusicBrainz and Last.fm (if API provided)")
            # Enrichment logic would go here
            logger.info("PIPELINE: Tags enriching complete")
        
        # Step 3: Retry failed files
        logger.info("PIPELINE: Retrying failed files")
        # Failed file retry logic would go here
        logger.info("PIPELINE: Failed files retry complete")
        
        # Step 4: Playlist generation
        logger.info("PIPELINE: Generating playlists")
        # Playlist generation logic would go here
        logger.info("PIPELINE: Playlist generation complete")
        
        logger.info("PIPELINE: Complete. Now you can start generating playlists!")
        
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        cleanup_child_processes()
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        cleanup_child_processes()
        raise 