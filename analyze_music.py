import os
import signal
import essentia.standard as es
import numpy as np
from functools import lru_cache

class AudioAnalyzer:
    def __init__(self, base_timeout=30, long_file_threshold=300):
        self.base_timeout = base_timeout
        self.long_file_threshold = long_file_threshold  # seconds
        
    @lru_cache(maxsize=100)
    def _estimate_timeout(self, filepath):
        """Predict timeout based on file duration"""
        try:
            duration = os.path.getsize(filepath) / (44100 * 2 * 2)  # Rough estimate
            return min(120, max(self.base_timeout, int(duration * 0.5)))
        except:
            return self.base_timeout

    def extract_features(self, filepath):
        timeout = self._estimate_timeout(filepath)
        
        def handler(signum, frame):
            raise TimeoutError(f"Analysis timed out after {timeout}s")
        
        signal.signal(signal.SIGALRM, handler)
        signal.alarm(timeout)
        
        try:
            loader = es.MonoLoader(
                filename=filepath,
                sampleRate=44100,
                resampleQuality=2  # Faster resampling
            )
            audio = loader()
            
            # Skip analysis if file is too short
            if len(audio) < 44100:  # 1 second
                return None
                
            # Fast rhythm estimation
            rhythm = es.RhythmExtractor2013(
                method="degara",  # Faster algorithm
                minTempo=60,
                maxTempo=200
            )
            bpm, _, confidence, _, _ = rhythm(audio)
            
            # Simplified spectral analysis
            spectral = es.SpectralCentroid(sampleRate=44100)
            centroid = spectral(audio[:44100*30])  # Only analyze first 30 seconds
            
            return {
                'bpm': float(bpm),
                'duration': len(audio)/44100,
                'beat_confidence': float(np.mean(confidence)),
                'centroid': float(np.mean(centroid))
            }
            
        except TimeoutError:
            print(f"Skipped {os.path.basename(filepath)} (timeout: {timeout}s)")
            return None
        except Exception as e:
            print(f"Error in {os.path.basename(filepath)}: {str(e)}")
            return None
        finally:
            signal.alarm(0)