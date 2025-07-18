import os
import signal
import essentia.standard as es
import numpy as np

class AudioAnalyzer:
    def __init__(self, timeout=30):  # Add timeout parameter here
        self.timeout = timeout
        
    def _time_limit(self):
        """Context manager for timeout handling"""
        def signal_handler(signum, frame):
            raise TimeoutError(f"Analysis timed out after {self.timeout}s")
        
        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(self.timeout)
        
        return self
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        signal.alarm(0)
        
    def extract_features(self, filepath):
        try:
            with self._time_limit():
                loader = es.MonoLoader(
                    filename=filepath,
                    sampleRate=44100,
                    resampleQuality=2
                )
                audio = loader()
                
                if len(audio) < 44100:
                    return None
                    
                rhythm = es.RhythmExtractor2013(method="degara")
                bpm, _, confidence, _, _ = rhythm(audio)
                
                spectral = es.SpectralCentroid(sampleRate=44100)
                centroid = spectral(audio[:44100*30])  # First 30 seconds only
                
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