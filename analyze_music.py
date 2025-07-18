import os
import signal
import essentia.standard as es
import numpy as np
import sqlite3
from contextlib import contextmanager

class AudioAnalyzer:
    def __init__(self, timeout=30):
        self.timeout = timeout
        
    @contextmanager
    def _time_limit(self):
        signal.signal(signal.SIGALRM, self._raise_timeout)
        signal.alarm(self.timeout)
        try:
            yield
        finally:
            signal.alarm(0)
            
    def _raise_timeout(self, signum, frame):
        raise TimeoutError("Audio analysis timed out")

    def extract_features(self, filepath):
        try:
            with self._time_limit():
                # Load audio file
                loader = es.MonoLoader(
                    filename=filepath,
                    sampleRate=44100,
                    resampleQuality=4
                )
                audio = loader()
                
                # Skip if audio is too short
                if len(audio) < 44100:  # 1 second
                    return None
                
                # Extract features
                rhythm = es.RhythmExtractor2013()
                bpm, beats, confidence, _, _ = rhythm(audio)
                
                spectral = es.SpectralCentroidTime(sampleRate=44100)
                centroid = spectral(audio)
                
                return {
                    'bpm': float(bpm),
                    'duration': len(audio)/44100,
                    'beat_confidence': float(np.mean(confidence)),
                    'centroid': float(np.mean(centroid))
                }
                
        except TimeoutError:
            print(f"Timeout processing {os.path.basename(filepath)}")
            return None
        except Exception as e:
            print(f"Error processing {os.path.basename(filepath)}: {str(e)}")
            return None