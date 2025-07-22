# music_analyzer/parallel.py
import multiprocessing as mp
import os
import sys
from tqdm import tqdm
import logging
import time
from .audio_analyzer import AudioAnalyzer

logger = logging.getLogger(__name__)

def process_file_worker(filepath):
    import os
    from .audio_analyzer import AudioAnalyzer
    audio_analyzer = AudioAnalyzer()
    max_retries = 2
    retry_count = 0
    backoff_time = 1  # Initial backoff time in seconds

    while retry_count <= max_retries:
        try:
            if not os.path.exists(filepath):
                logger.warning(f"File not found: {filepath}")
                return None, filepath
            
            if os.path.getsize(filepath) < 1024:
                logger.warning(f"Skipping small file: {filepath}")
                return None, filepath

            if not filepath.lower().endswith(('.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac')):
                return None, filepath

            result = audio_analyzer.extract_features(filepath)

            if result and result[0] is not None:
                features = result[0]
                for key in ['bpm', 'centroid', 'duration']:
                    if features.get(key) is None:
                        features[key] = 0.0
                return features, filepath
            
            # If we get here, result was None or features were None
            if retry_count < max_retries:
                retry_count += 1
                time.sleep(backoff_time)
                backoff_time *= 2  # Exponential backoff
                logger.warning(f"Retrying {filepath} (attempt {retry_count}/{max_retries})")
                continue
            
            return None, filepath

        except Exception as e:
            if retry_count < max_retries:
                retry_count += 1
                time.sleep(backoff_time)
                backoff_time *= 2
                logger.warning(f"Error processing {filepath}, retrying (attempt {retry_count}/{max_retries}): {str(e)}")
                continue
            
            logger.error(f"Error processing {filepath} after {max_retries} retries: {str(e)}", exc_info=True)
            return None, filepath

class ParallelProcessor:
    def __init__(self):
        self.failed_files = []
        self.batch_size = int(os.getenv('BATCH_SIZE', '50'))
        self.max_retries = int(os.getenv('MAX_RETRIES', '3'))
        self.min_workers = 2
        self.max_workers = int(os.getenv('MAX_WORKERS', str(mp.cpu_count())))

    def process(self, file_list, workers=None):
        if not file_list:
            return
        self.workers = max(self.min_workers, min(workers or self.max_workers, self.max_workers))
        self.batch_size = min(self.batch_size, len(file_list))
        yield from self._process_parallel(file_list)

    def _process_parallel(self, file_list):
        retries = 0
        remaining_files = file_list[:]

        while remaining_files and retries < self.max_retries:
            try:
                logger.info(f"Starting multiprocessing with {self.workers} workers (retry {retries})")
                ctx = mp.get_context('spawn')

                for i in range(0, len(remaining_files), self.batch_size):
                    batch = remaining_files[i:i+self.batch_size]
                    failed_in_batch = []

                    with ctx.Pool(processes=self.workers) as pool:
                        for features, filepath in pool.imap_unordered(process_file_worker, batch):
                            if features:
                                yield features
                            else:
                                failed_in_batch.append(filepath)
                    # Update remaining files for next iteration
                    remaining_files = remaining_files[i+self.batch_size:]
                    if failed_in_batch:
                        self.failed_files.extend(failed_in_batch)

                # If we get here, all batches processed successfully
                logger.info(f"Processing completed - yielded all successful, {len(self.failed_files)} failed")
                return

            except (mp.TimeoutError, BrokenPipeError, ConnectionResetError) as e:
                logger.error(f"Multiprocessing error: {str(e)}")
                retries += 1
                if retries < self.max_retries:
                    # Reduce workers on retry to handle potential resource constraints
                    self.workers = max(self.min_workers, self.workers // 2)
                    logger.warning(f"Reducing workers to {self.workers} and retrying with {len(remaining_files)} remaining files")
                    time.sleep(2 ** retries)  # Exponential backoff
                else:
                    logger.error("Max retries reached, switching to sequential for remaining files")
                    # Process remaining files sequentially
                    for filepath in remaining_files:
                        features, _ = process_file_worker(filepath)
                        if features:
                            yield features
                        else:
                            self.failed_files.append(filepath)
                    break

        return