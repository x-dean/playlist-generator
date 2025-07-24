import os
import multiprocessing
import time
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.console import Console
import logging
import psutil
import signal

logger = logging.getLogger(__name__)

BIG_FILE_SIZE_MB = 200

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
    # Extra: send SIGKILL to any still alive
    for child in parent.children(recursive=True):
        try:
            os.kill(child.pid, signal.SIGKILL)
        except Exception:
            pass

def run_analysis(args, audio_db, playlist_db, cli):
    try:
        file_list = get_audio_files(args.music_dir)
        db_features = audio_db.get_all_features(include_failed=True)
        db_files = set(f['filepath'] for f in db_features)
        failed_files_db = set(f['filepath'] for f in db_features if f['failed'])
        logger.debug(f"Sample from file_list: {file_list[:3]}")
        logger.debug(f"Sample from failed_files_db: {list(failed_files_db)[:3]}")
        logger.debug(f"Sample from db_files: {list(db_files)[:3]}")
        logger.debug(f"Intersection (should be re-analyzed): {list(set(file_list) & failed_files_db)}")
        if args.force:
            files_to_analyze = file_list
        elif args.failed:
            files_to_analyze = [f for f in file_list if f not in db_files or f in failed_files_db]
        else:
            files_to_analyze = [f for f in file_list if f not in db_files]
        logger.debug(f"Files to analyze: {files_to_analyze[:10]}")
        logger.debug(f"About to process {len(files_to_analyze)} files")
        if not files_to_analyze:
            total_found = len(file_list)
            total_in_db = len(db_features)
            total_failed = len([f for f in db_features if f['failed']])
            processed_this_run = 0
            failed_this_run = 0
            stats = playlist_db.get_library_statistics()
            cli.show_analysis_summary(
                stats=stats,
                processed_this_run=processed_this_run,
                failed_this_run=failed_this_run,
                total_found=total_found,
                total_in_db=total_in_db,
                total_failed=total_failed
            )
            return []
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
        from music_analyzer.parallel import ParallelProcessor, UserAbortException
        from music_analyzer.sequential import SequentialProcessor
        with progress:
            task_id = progress.add_task(f"Processed 0/{total_files} files", total=total_files, trackinfo="")
            if args.force_sequential or (args.workers and args.workers <= 1):
                # Process all files sequentially (no multiprocessing, no subprocesses)
                processor = SequentialProcessor()
                for features, filepath in processor.process(files_to_analyze, workers=args.workers or 1):
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
                    if features and 'metadata' in features:
                        meta = features['metadata']
                        if meta.get('musicbrainz_id'):
                            pass
                        else:
                            pass
                failed_files.extend(processor.failed_files)
            else:
                # 1. Process normal files in parallel
                if normal_files:
                    processor = ParallelProcessor()
                    process_iter = processor.process(normal_files, workers=args.workers or multiprocessing.cpu_count())
                    for features in process_iter:
                        processed_count += 1
                        progress.update(
                            task_id,
                            advance=1,
                            description=f"Processed {processed_count}/{total_files} files",
                            trackinfo=""
                        )
                        logger.debug(f"Features: {features}")
                        if features and 'metadata' in features:
                            meta = features['metadata']
                            if meta.get('musicbrainz_id'):
                                pass
                            else:
                                pass
                    failed_files.extend(processor.failed_files)
                # 2. Process big files sequentially (now using subprocesses)
                if big_files:
                    from music_analyzer.audio_analyzer import AudioAnalyzer
                    bigfile_processes = []
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
                        # Use a subprocess for each big file
                        def analyze_in_subprocess(filepath, cache_file, host_music_dir, container_music_dir, queue):
                            try:
                                analyzer = AudioAnalyzer(cache_file, host_music_dir, container_music_dir)
                                result = analyzer.extract_features(filepath)
                                queue.put(result)
                            except Exception as e:
                                queue.put(None)
                        queue = multiprocessing.Queue()
                        p = multiprocessing.Process(target=analyze_in_subprocess, args=(filepath, audio_db.cache_file, audio_db.host_music_dir, audio_db.container_music_dir, queue))
                        p.start()
                        bigfile_processes.append(p)
                        try:
                            result = queue.get(timeout=600)  # 10 min timeout per big file
                            p.join()
                        except Exception:
                            p.terminate()
                            p.join()
                            result = None
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
                    # No need to extend failed_files from a processor, handled above
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
    except (KeyboardInterrupt, UserAbortException):
        logger.info("Analysis interrupted by user (Ctrl+C). Exiting analysis step.")
        cleanup_child_processes()
        return []

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