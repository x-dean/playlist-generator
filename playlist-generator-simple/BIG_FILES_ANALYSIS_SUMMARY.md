# Big Files Analysis Summary

## Overview
The playlist generator handles large audio files through a multi-tier approach that combines dynamic timeouts, feature skipping, streaming analysis, and invalid markers for failed features.

## File Size Thresholds

### **Sample-Based Thresholds (in samples at 44kHz)**
```python
LARGE_FILE_THRESHOLD = 100000000          # ~2.3 hours at 44kHz
EXTREMELY_LARGE_THRESHOLD = 200000000     # ~4.5 hours at 44kHz  
EXTREMELY_LARGE_PROCESSING_THRESHOLD = 500000000  # ~11.3 hours at 44kHz
```

### **MB-Based Thresholds (for streaming)**
```python
streaming_large_file_threshold_mb = 50MB  # Default from config
```

## Timeout Management

### **Dynamic Timeout Calculation**
```python
def get_timeout_for_file_size(audio_length: int) -> int:
    if audio_length > EXTREMELY_LARGE_PROCESSING_THRESHOLD:
        return TIMEOUT_EXTREMELY_LARGE      # 30 minutes
    elif audio_length > EXTREMELY_LARGE_THRESHOLD:
        return TIMEOUT_LARGE_FILES          # 20 minutes
    else:
        return DEFAULT_TIMEOUT_SECONDS      # 10 minutes
```

### **Timeout Constants**
```python
DEFAULT_TIMEOUT_SECONDS = 600      # 10 minutes
TIMEOUT_LARGE_FILES = 1200         # 20 minutes  
TIMEOUT_EXTREMELY_LARGE = 1800     # 30 minutes
```

## Feature Extraction Strategy

### **1. Extremely Large Files (>500M samples)**
**Skipped Features:**
- ❌ Rhythm features (BPM, tempo)
- ❌ Spectral features (centroid, rolloff, flatness)
- ❌ Loudness features
- ❌ Key detection
- ❌ MFCC features
- ❌ MusiCNN features

**Fallback Values:**
```python
features.setdefault('bpm', -999.0)                    # Invalid BPM
features.setdefault('spectral_centroid', -999.0)      # Invalid centroid
features.setdefault('loudness', -999.0)               # Invalid loudness
features.setdefault('key', 'INVALID')                 # Invalid key
features.setdefault('scale', 'INVALID')               # Invalid scale
features.setdefault('key_strength', -999.0)           # Invalid strength
```

**What's Still Extracted:**
- ✅ Duration calculation
- ✅ Basic metadata
- ✅ External API enrichment (if available)

### **2. Very Large Files (>200M samples)**
**Skipped Features:**
- ❌ MFCC features
- ❌ MusiCNN features

**Fallback Values:**
```python
features.setdefault('mfcc', [-999.0] * 13)           # Invalid MFCC
features.setdefault('musicnn_features', [-999.0] * 50)  # Invalid MusiCNN
```

**What's Still Extracted:**
- ✅ Rhythm features (BPM, tempo)
- ✅ Spectral features
- ✅ Loudness features
- ✅ Key detection
- ✅ Duration calculation
- ✅ Basic metadata
- ✅ External API enrichment

### **3. Large Files (>100M samples)**
**All features extracted normally** with extended timeouts.

## Audio Loading Strategy

### **Traditional Loading (<50MB)**
- Loads entire file into memory
- Standard analysis approach

### **Streaming Loading (50MB-100MB)**
- Uses streaming audio loader
- Processes in chunks
- Concatenates chunks for analysis

### **True Streaming Analysis (>100MB)**
```python
def _analyze_large_file_streaming(self, audio_path: str, streaming_loader):
    # Calculate segment parameters
    segment_duration = 120.0  # 2 minutes per segment
    num_segments = min(10, int(duration / segment_duration))  # Max 10 segments
    segment_interval = duration / (num_segments + 1)  # Evenly spaced
    
    # Sample segments throughout the file
    for i in range(num_segments):
        segment_start = (i + 1) * segment_interval
        chunk, sr = librosa.load(
            audio_path,
            sr=DEFAULT_SAMPLE_RATE,
            mono=True,
            offset=segment_start,
            duration=segment_duration
        )
        segments.append(chunk)
        gc.collect()  # Force garbage collection
```

**Strategy:**
- Samples 10 segments of 2 minutes each
- Evenly distributed throughout the file
- Better for mixed tracks with varying characteristics
- Forces garbage collection after each segment

## Invalid Markers System

### **Purpose**
Prevents failed/skipped features from being used in playlist generation.

### **Invalid Values**
```python
# Float features
-999.0  # For BPM, loudness, spectral features, key strength

# String features  
'INVALID'  # For key, scale

# Array features
[-999.0] * 13   # For MFCC (13 coefficients)
[-999.0] * 50   # For MusiCNN (50 features)
```

### **Validation**
```python
def _is_valid_for_playlist(self, features: Dict[str, Any]) -> bool:
    """Check if features are suitable for playlist generation."""
    invalid_markers = [-999.0, 'INVALID']
    
    for value in features.values():
        if isinstance(value, (int, float)) and value in invalid_markers:
            return False
        elif isinstance(value, str) and value in invalid_markers:
            return False
        elif isinstance(value, list) and any(v in invalid_markers for v in value):
            return False
    
    return True
```

## Cross-Platform Timeout Handling

### **Unix/Linux (SIGALRM)**
```python
def _handle_timeout(signum, frame):
    raise TimeoutException(error_message)

signal.signal(signal.SIGALRM, _handle_timeout)
signal.alarm(seconds)
```

### **Windows (Threading)**
```python
thread = threading.Thread(target=target)
thread.daemon = True
thread.start()
thread.join(seconds)

if thread.is_alive():
    raise TimeoutException(error_message)
```

## Error Handling

### **Timeout Recovery**
- Analysis fails gracefully with timeout message
- File is marked as failed in database
- Continues with next file

### **Feature Failure Recovery**
- Individual feature failures don't stop entire analysis
- Failed features get invalid markers
- Successful features are still extracted

### **Memory Management**
- Garbage collection after each segment
- Streaming prevents memory overflow
- Fallback to traditional loading if streaming fails

## Configuration

### **Relevant Config Settings**
```ini
# Timeouts
ANALYSIS_TIMEOUT_SECONDS=300

# Streaming
STREAMING_AUDIO_ENABLED=true
STREAMING_LARGE_FILE_THRESHOLD_MB=50
STREAMING_MEMORY_LIMIT_PERCENT=30
STREAMING_CHUNK_DURATION_SECONDS=10

# Feature extraction
EXTRACT_RHYTHM=true
EXTRACT_SPECTRAL=true
EXTRACT_LOUDNESS=true
EXTRACT_KEY=true
EXTRACT_MFCC=true
EXTRACT_MUSICNN=false
```

## Summary

The big file analysis system provides:

1. **Progressive Degradation**: Features are skipped based on file size
2. **Dynamic Timeouts**: Timeout increases with file size
3. **Streaming Analysis**: Memory-efficient processing for large files
4. **Invalid Markers**: Clear indication of failed/skipped features
5. **Cross-Platform Support**: Works on both Unix and Windows
6. **Graceful Failure**: Analysis continues even if individual features fail
7. **Memory Management**: Prevents memory overflow with garbage collection

This ensures that even extremely large files (>11 hours) can be processed, though with minimal feature extraction, while maintaining system stability and preventing crashes. 