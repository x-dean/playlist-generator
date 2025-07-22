# music_analyzer/parallel.py
import multiprocessing as mp
import os
import sys
from tqdm import tqdm
import logging
from .audio_analyzer import AudioAnalyzer
audio_analyzer = AudioAnalyzer()

logger = logging.getLogger(__name__)

def process_file_worker(filepath):
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

        if result is None or (result[0] is None and "timeout" in str(result).lower()):
            logger.warning(f"Retrying {filepath} after timeout")
            result = audio_analyzer.extract_features(filepath)

        if result and result[0] is not None:
            features = result[0]
            for key in ['bpm', 'centroid', 'duration']:
                if features.get(key) is None:
                    features[key] = 0.0
            return features, filepath
        return None, filepath
    except Exception as e:
        logger.error(f"Error processing {filepath}: {str(e)}")
        return None, filepath

class ParallelProcessor:
    def __init__(self):
        self.failed_files = []

    def process(self, file_list, workers):
        return self._process_parallel(file_list, workers)

    def _process_parallel(self, file_list, workers):
        results = []
        max_retries = 3
        retries = 0
        batch_size = min(50, len(file_list))

        while file_list and retries < max_retries:
            try:
                logger.info(f"Starting multiprocessing with {workers} workers (retry {retries})")
                ctx = mp.get_context('spawn')

                for i in range(0, len(file_list), batch_size):
                    batch = file_list[i:i+batch_size]
                    with ctx.Pool(processes=workers) as pool:
                        with tqdm(total=len(batch), desc=f"Processing batch {i//batch_size+1}",
                                  bar_format="{l_bar}{bar:40}{r_bar}", file=sys.stdout) as pbar:
                            
                            for features, filepath in pool.imap_unordered(process_file_worker, batch):
                                if features:
                                    results.append(features)
                                else:
                                    self.failed_files.append(filepath)
                                pbar.update(1)
                                pbar.set_postfix_str(f"OK: {len(results)}, Failed: {len(self.failed_files)}")
                    
                    pool.close()
                    pool.join()

                logger.info(f"Processing completed - {len(results)} successful, {len(self.failed_files)} failed")
                return results

            except (mp.TimeoutError, BrokenPipeError, ConnectionResetError) as e:
                logger.error(f"Multiprocessing error: {str(e)}")
                retries += 1
                file_list = file_list[i+batch_size:] if 'i' in locals() else file_list
                logger.warning(f"Retrying with {len(file_list)} remaining files")

        logger.error("Max retries reached, switching to sequential for remaining files")
        for filepath in file_list:
            features, _ = process_file_worker(filepath)
            if features:
                results.append(features)
            else:
                self.failed_files.append(filepath)
        return results