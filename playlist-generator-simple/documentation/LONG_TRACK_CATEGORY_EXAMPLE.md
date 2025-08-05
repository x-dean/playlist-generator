# Long Track Category Addition Example

This document shows how a long track category is determined and added to the database during sequential processing.

## Example Scenario

**File**: `/music/radio_shows/bbc_radio_1_essential_mix_2024_01_15.mp3`
**Size**: 250MB (triggers sequential processing)
**Duration**: 120 minutes (estimated from file size)
**Content**: Electronic dance music radio show

## Step-by-Step Process

### 1. File Discovery and Size Check

```python
# File is discovered and size is checked
file_size_bytes = os.path.getsize(file_path)  # 262144000 bytes
file_size_mb = file_size_bytes / (1024 * 1024)  # 250.0 MB

# Since file_size_mb >= 200, it goes to sequential processing
if file_size_mb >= 200:
    # Route to sequential analyzer
    sequential_analyzer.process_files([file_path])
```

### 2. Sequential Analyzer Configuration

```python
# In sequential_analyzer.py _get_analysis_config()
def _get_analysis_config(self, file_path: str) -> Dict[str, Any]:
    file_size_mb = 250.0  # 250MB file
    
    # Sequential processing is for files >= 200MB
    if file_size_mb >= 200:
        analysis_type = 'basic'
        use_full_analysis = False
        enable_musicnn = False  # Disabled for files >= 200MB
        enable_lightweight_categorization = True
        
        log_universal('DEBUG', 'Sequential', 
                     f'File 250.0MB: Sequential + Multi-segment processing')
```

### 3. Long Audio Track Detection

```python
# In audio_analyzer.py _is_long_audio_track()
def _is_long_audio_track(self, file_path: str) -> bool:
    file_size_bytes = 262144000
    estimated_duration_seconds = (file_size_bytes * 8) / (320 * 1000)  # 7200 seconds
    estimated_duration_minutes = estimated_duration_seconds / 60  # 120 minutes
    
    long_threshold = 45  # minutes
    is_long = estimated_duration_minutes > long_threshold  # True (120 > 45)
    
    log_universal('DEBUG', 'Audio', 
                 f'File bbc_radio_1_essential_mix_2024_01_15.mp3: 120.0 minutes estimated, long_audio: True')
    
    return True
```

### 4. Lightweight Feature Extraction

```python
# Extract lightweight features from 15-second sample
sample_duration = 15  # seconds
audio_sample, sample_rate = self._load_audio_sample(file_path, sample_duration)

# Extract essential features for categorization
advanced_features = {
    'danceability': 0.85,      # High danceability (electronic music)
    'dynamic_complexity': 0.72, # Moderate complexity
    'silence_rate': 0.08,       # Low silence (continuous music)
    'zero_crossing_rate': 0.045,
    'harmonic_peaks_count': 156
}
```

### 5. Lightweight Category Creation

```python
# In audio_analyzer.py _create_lightweight_category()
def _create_lightweight_category(self, features: Dict[str, Any], metadata: Dict[str, Any]) -> Optional[str]:
    # Extract features
    danceability = features.get('danceability', 0.0)  # 0.85
    dynamic_complexity = features.get('dynamic_complexity', 0.0)  # 0.72
    silence_rate = features.get('silence_rate', 0.0)  # 0.08
    
    # Get metadata
    artist = metadata.get('artist', '').lower()  # 'bbc radio 1'
    title = metadata.get('title', '').lower()    # 'essential mix 2024-01-15'
    
    # Use simplified categorization
    category = self._simplified_categorization(artist, title, genre, features)
    
    log_universal('DEBUG', 'Audio', 
                 f'Lightweight category created: Radio/General (danceability: 0.85, complexity: 0.72)')
    
    return 'Radio/General'
```

### 6. Simplified Categorization Logic

```python
# In audio_analyzer.py _simplified_categorization()
def _simplified_categorization(self, artist: str, title: str, genre: str, features: Dict[str, Any]) -> str:
    # Step 1: Check for obvious content types first
    if self._is_radio_show(artist, title):  # True - contains 'radio'
        return 'Radio/General'
    elif self._is_podcast(artist, title):   # False
        return 'Podcast/General'
    elif self._is_mix(artist, title):       # False
        return 'Mix/General'
    
    # Radio show detected, return early
    return 'Radio/General'
```

### 7. Radio Show Detection

```python
# In audio_analyzer.py _is_radio_show()
def _is_radio_show(self, artist: str, title: str) -> bool:
    radio_indicators = [
        'radio', 'fm', 'am', 'broadcast', 'station', 'dj', 'disc jockey',
        'state of trance', 'asot', 'essential mix', 'bbc radio',
        'radio 1', 'radio 2', 'kiss fm', 'capital fm'
    ]
    
    # Check for radio show patterns
    for indicator in radio_indicators:
        if indicator in artist or indicator in title:
            return True  # 'radio' found in 'bbc radio 1'
    
    return False
```

### 8. Database Storage

```python
# In audio_analyzer.py main analysis flow
if file_size_mb > 200 and skip_musicnn_for_large:
    # Create lightweight category for very large files
    if advanced_features and enable_lightweight_categorization:
        lightweight_category = self._create_lightweight_category(advanced_features, metadata)
        if lightweight_category:
            # Store category in metadata and database
            metadata['lightweight_category'] = lightweight_category  # 'Radio/General'
            metadata['long_audio_category'] = lightweight_category   # 'Radio/General'
            
            # Save to database
            self.db_manager.save_metadata(file_path, metadata)
            
            log_universal('INFO', 'Audio', 
                         f'Created lightweight category for very large file: Radio/General')
            log_universal('INFO', 'Audio', 
                         f'Lightweight category assigned: Radio/General')
```

### 9. Database Insert Example

```sql
-- Final database record in tracks table
INSERT INTO tracks (
    file_path,
    file_hash,
    filename,
    file_size_bytes,
    title,
    artist,
    album,
    duration,
    analysis_type,
    analyzed,
    audio_type,
    long_audio_category,
    discovery_source,
    analysis_date,
    -- ... other fields
) VALUES (
    '/music/radio_shows/bbc_radio_1_essential_mix_2024_01_15.mp3',
    'a1b2c3d4e5f6...',
    'bbc_radio_1_essential_mix_2024_01_15.mp3',
    262144000,
    'Essential Mix 2024-01-15',
    'BBC Radio 1',
    'Essential Mix',
    7200.0,  -- 120 minutes in seconds
    'basic',
    1,  -- True
    'large_file',
    'Radio/General',
    'file_system',
    '2024-01-15 14:30:00'
);
```

## Database Query Examples

### Check Long Audio Categories

```sql
-- View all long audio tracks and their categories
SELECT 
    filename,
    artist,
    title,
    duration,
    long_audio_category,
    audio_type,
    file_size_bytes / (1024*1024) as size_mb
FROM tracks 
WHERE long_audio_category IS NOT NULL
ORDER BY duration DESC;
```

### Count Categories

```sql
-- Count tracks by long audio category
SELECT 
    long_audio_category,
    COUNT(*) as track_count,
    AVG(duration) as avg_duration_minutes,
    AVG(file_size_bytes / (1024*1024)) as avg_size_mb
FROM tracks 
WHERE long_audio_category IS NOT NULL
GROUP BY long_audio_category
ORDER BY track_count DESC;
```

### Find Radio Shows

```sql
-- Find all radio shows
SELECT 
    filename,
    artist,
    title,
    duration / 60 as duration_minutes,
    file_size_bytes / (1024*1024) as size_mb
FROM tracks 
WHERE long_audio_category = 'Radio/General'
ORDER BY duration DESC;
```

## Log Output Example

```
[INFO] Sequential: Processing file 1/1: bbc_radio_1_essential_mix_2024_01_15.mp3
[DEBUG] Sequential: File 250.0MB: Sequential + Multi-segment processing
[DEBUG] Sequential: MusiCNN enabled: False (disabled for files >= 200MB)
[DEBUG] Sequential: Lightweight categorization enabled: True
[INFO] Audio: Extracting lightweight features for large file: bbc_radio_1_essential_mix_2024_01_15.mp3
[INFO] Audio: Successfully loaded 15s sample for lightweight analysis
[DEBUG] Audio: File bbc_radio_1_essential_mix_2024_01_15.mp3: 120.0 minutes estimated, long_audio: True
[DEBUG] Audio: Lightweight category created: Radio/General (danceability: 0.85, complexity: 0.72)
[INFO] Audio: Created lightweight category for very large file: Radio/General
[INFO] Audio: Lightweight category assigned: Radio/General
[INFO] Audio: Category based on: 0.85 danceability, 0.72 complexity
[INFO] Sequential: Processing completed successfully
[DEBUG] Sequential: Memory cleanup completed: 2.1GB used
[DEBUG] Sequential: Cleared TensorFlow session
```

## Category Types

The system recognizes these lightweight categories:

1. **Radio/General** - Radio shows, broadcasts
2. **Podcast/General** - Podcasts, interviews, talk shows
3. **Mix/General** - DJ mixes, compilations
4. **Electronic/Dance** - High danceability (>0.7)
5. **Ambient/Chill** - Low danceability (<0.3)
6. **Rock/Metal** - High dynamic complexity (>0.8)
7. **Speech/Spoken** - High silence rate (>0.5)
8. **Pop/Indie** - Default for modern music

## Key Points

1. **File Size Threshold**: Only files >= 200MB use sequential processing
2. **Duration Threshold**: Only files > 45 minutes are considered "long audio"
3. **Lightweight Features**: Uses 15-second sample for categorization
4. **Database Storage**: Category stored in both `lightweight_category` and `long_audio_category` fields
5. **Memory Management**: TensorFlow sessions cleared after each file
6. **Logging**: Comprehensive logging for debugging and monitoring 