import os
import multiprocessing
import signal
import logging
import psutil
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.console import Console
from music_analyzer.parallel import ParallelProcessor, UserAbortException
from music_analyzer.sequential import SequentialProcessor
from music_analyzer.feature_extractor import AudioAnalyzer

logger = logging.getLogger(__name__)
BIG_FILE_SIZE_MB = 200

# --- File Selection ---
def select_files_for_analysis(args, audio_db):
    """Return (normal_files, big_files) to analyze based on args and DB state."""
    file_list = get_audio_files(args.music_dir)
    db_features = audio_db.get_all_features(include_failed=True)
    db_files = set(f['filepath'] for f in db_features)
    failed_files_db = set(f['filepath'] for f in db_features if f['failed'])
    if args.force:
        files_to_analyze = file_list
    elif args.failed:
        files_to_analyze = [f for f in file_list if f not in db_files or f in failed_files_db]
    else:
        files_to_analyze = [f for f in file_list if f not in db_files]
    def is_big_file(filepath):
        try:
            return os.path.getsize(filepath) > BIG_FILE_SIZE_MB * 1024 * 1024
        except Exception:
            return False
    big_files = [f for f in files_to_analyze if is_big_file(f)]
    normal_files = [f for f in files_to_analyze if not is_big_file(f)]
    return normal_files, big_files, file_list, db_features

# --- Graceful Shutdown ---
def setup_graceful_shutdown():
    stop_event = multiprocessing.Event()
    def handle_stop_signal(signum, frame):
        stop_event.set()
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
    signal.signal(signal.SIGINT, handle_stop_signal)
    signal.signal(signal.SIGTERM, handle_stop_signal)
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
    def process(self, files, workers, status_queue=None):
        processor = ParallelProcessor()
        return processor.process(files, workers=workers, status_queue=status_queue, stop_event=self.stop_event)

class SequentialWorkerManager:
    def __init__(self, stop_event):
        self.stop_event = stop_event
    def process(self, files, workers):
        processor = SequentialProcessor()
        return processor.process(files, workers=workers, stop_event=self.stop_event)

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
            def analyze_in_subprocess(filepath, cache_file, host_music_dir, container_music_dir, queue, stop_event):
                try:
                    if stop_event.is_set():
                        queue.put(None)
                        return
                    analyzer = AudioAnalyzer(cache_file, host_music_dir, container_music_dir)
                    result = analyzer.extract_features(filepath)
                    queue.put(result)
                except Exception as e:
                    queue.put(None)
            p = multiprocessing.Process(target=analyze_in_subprocess, args=(filepath, self.audio_db.cache_file, self.audio_db.host_music_dir, self.audio_db.container_music_dir, queue, self.stop_event))
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
                    analyzer = AudioAnalyzer(self.audio_db.cache_file, self.audio_db.host_music_dir, self.audio_db.container_music_dir)
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
def run_analysis(args, audio_db, playlist_db, cli, stop_event=None):
    if stop_event is None:
        stop_event = setup_graceful_shutdown()
    normal_files, big_files, file_list, db_features = select_files_for_analysis(args, audio_db)
    failed_files = []
    processed_count = 0
    total_files = len(normal_files) + len(big_files)
    progress = create_progress_bar(total_files)
    with progress:
        task_id = progress.add_task(f"Processed 0/{total_files} files", total=total_files, trackinfo="")
        if args.force_sequential or (args.workers and args.workers <= 1):
            # All files sequential
            seq_manager = SequentialWorkerManager(stop_event)
            for features, filepath in seq_manager.process(normal_files + big_files, workers=args.workers or 1):
                if stop_event.is_set():
                    break
                filename = os.path.basename(filepath)
                try:
                    size_mb = os.path.getsize(filepath) / (1024 * 1024)
                except Exception:
                    size_mb = 0
                processed_count += 1
                progress.update(
                    task_id,
                    advance=1,
                    trackinfo=f"{filename} ({size_mb:.1f} MB)"
                )
                logger.debug(f"Features: {features}")
                if not features:
                    failed_files.append(filepath)
        else:
            # Parallel for normal files
            par_manager = ParallelWorkerManager(stop_event)
            if normal_files:
                for features in par_manager.process(normal_files, workers=args.workers or multiprocessing.cpu_count()):
                    if stop_event.is_set():
                        break
                    processed_count += 1
                    progress.update(
                        task_id,
                        advance=1,
                        description=f"Processed {processed_count}/{total_files} files",
                        trackinfo=""
                    )
                    logger.debug(f"Features: {features}")
                    if not features:
                        # No filepath in features for parallel, can't append to failed_files
                        pass
            # Big files in subprocesses
            if big_files:
                big_manager = BigFileWorkerManager(stop_event, audio_db)
                for filepath, result in big_manager.process(big_files):
                    if stop_event.is_set():
                        break
                    filename = os.path.basename(filepath)
                    try:
                        size_mb = os.path.getsize(filepath) / (1024 * 1024)
                    except Exception:
                        size_mb = 0
                    processed_count += 1
                    progress.update(
                        task_id,
                        advance=1,
                        description=f"Processed {processed_count}/{total_files} files (big file)",
                        trackinfo=f"{filename} ({size_mb:.1f} MB)"
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
    cli.show_analysis_summary(
        stats=stats,
        processed_this_run=processed_this_run,
        failed_this_run=failed_this_run,
        total_found=total_found,
        total_in_db=total_in_db,
        total_failed=total_failed
    )
    return failed_files

# --- File Discovery Helper ---
def get_audio_files(music_dir: str) -> list[str]:
    file_list = []
    valid_ext = ('.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac', '.opus')
    for root, _, files in os.walk(music_dir):
        for file in files:
            file_lower = file.lower()
            if file_lower.endswith(valid_ext):
                file_list.append(os.path.join(root, file))
    logger.info(f"Found {len(file_list)} audio files in {music_dir}")
    return file_list 