import essentia.standard as es
import numpy as np
import os
import time
from contextlib import contextmanager

class AudioAnalyzer:
    def __init__(self, timeout=30):
        self.timeout = timeout
        
    @contextmanager
    def _time_limit(self):
        def signal_handler(signum, frame):
            raise TimeoutError()
        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(self.timeout)
        try:
            yield
        finally:
            signal.alarm(0)
    
    def extract_features(self, filepath):
        try:
            with self._time_limit():
                # Load audio
                loader = es.MonoLoader(filename=filepath, sampleRate=44100)
                audio = loader()
                
                # Extract features
                rhythm_extractor = es.RhythmExtractor2013()
                bpm, _, confidence, _, _ = rhythm_extractor(audio)
                
                spectral = es.SpectralCentroidTime(sampleRate=44100)
                centroid = spectral(audio)
                
                return {
                    'bpm': float(bpm),
                    'duration': len(audio)/44100.0,
                    'beat_confidence': float(np.mean(confidence)),
                    'centroid': float(np.mean(centroid))
                }
        except Exception as e:
            print(f"Error processing {filepath}: {str(e)}")
            return None