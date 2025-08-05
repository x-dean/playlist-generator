# Trance Track Categorization Example

This document shows how trance tracks are properly categorized and why they were previously misclassified as hip-hop.

## The Problem

**Issue**: Trance tracks were being categorized as "Hip-Hop/Rap" instead of "Electronic/Dance"

**Root Cause**: The `_simplified_categorization` method wasn't using the `_detect_music_genre` method, which contains proper trance detection logic.

## Fixed Categorization Process

### Example: Trance Track

**File**: `/music/trance/armin_van_buuren_state_of_trance_episode_1000.mp3`
**Size**: 180MB (uses parallel processing with multi-segment)
**Content**: Trance music, 138 BPM

## Step-by-Step Categorization

### 1. Metadata Analysis

```python
# Extract metadata
artist = "Armin van Buuren"  # Contains "armin" (trance artist)
title = "State of Trance Episode 1000"  # Contains "state of trance"
genre = "Trance"  # Direct genre tag

# Check for radio show first
if self._is_radio_show(artist, title):
    # "state of trance" is in radio indicators
    return 'Radio/General'  # This would be correct for a radio show
```

### 2. Genre Detection (If not radio show)

```python
# In _detect_music_genre()
electronic_keywords = {
    'trance': ['trance', 'asot', 'state of trance', 'armin', 'tiesto', 'paul van dyk'],
    'techno': ['techno', 'tech house', 'minimal', 'berlin techno', 'detroit techno'],
    'house': ['house', 'deep house', 'progressive house', 'acid house', 'chicago house'],
    # ... other electronic genres
}

# Check for trance keywords
for keyword in ['trance', 'asot', 'state of trance', 'armin']:
    if keyword in artist.lower() or keyword in title.lower():
        return 'Electronic/Dance'  # Correct categorization
```

### 3. Audio Feature Analysis (Fallback)

```python
# If no metadata clues, use audio features
features = {
    'danceability': 0.85,  # High danceability (typical for trance)
    'bpm': 138.0,         # Trance BPM range (125-150)
    'dynamic_complexity': 0.65
}

# BPM-based detection
if 115 <= bpm <= 150 and danceability > 0.6:
    return 'Electronic/Dance'  # Correct for trance
```

## Before vs After Fix

### Before Fix (Incorrect)
```python
def _simplified_categorization(self, artist, title, genre, features):
    # Step 1: Check for obvious content types
    if self._is_radio_show(artist, title):
        return 'Radio/General'
    elif self._is_podcast(artist, title):
        return 'Podcast/General'
    elif self._is_mix(artist, title):
        return 'Mix/General'
    
    # Step 2: Use audio features ONLY (no genre detection)
    if danceability > 0.7:
        return 'Electronic/Dance'  # This worked for some trance
    elif danceability < 0.3:
        return 'Ambient/Chill'
    elif dynamic_complexity > 0.8:
        return 'Rock/Metal'
    else:
        return 'Pop/Indie'  # Many trance tracks fell here
```

### After Fix (Correct)
```python
def _simplified_categorization(self, artist, title, genre, features):
    # Step 1: Check for obvious content types
    if self._is_radio_show(artist, title):
        return 'Radio/General'
    elif self._is_podcast(artist, title):
        return 'Podcast/General'
    elif self._is_mix(artist, title):
        return 'Mix/General'
    
    # Step 2: Use genre detection from metadata and audio features
    detected_genre = self._detect_music_genre(artist, title, genre, features)
    if detected_genre:
        return detected_genre  # Returns 'Electronic/Dance' for trance
    
    # Step 3: Fallback to audio features only
    # ... (same as before)
```

## Test Cases

### Case 1: Trance Track with Metadata
```python
artist = "Armin van Buuren"
title = "In and Out of Love"
genre = "Trance"
# Result: 'Electronic/Dance' (from metadata keywords)
```

### Case 2: Trance Track without Metadata
```python
artist = "Unknown Artist"
title = "Track 1"
genre = ""
features = {'danceability': 0.8, 'bpm': 138.0}
# Result: 'Electronic/Dance' (from BPM + danceability)
```

### Case 3: Trance Radio Show
```python
artist = "Armin van Buuren"
title = "State of Trance Episode 1000"
genre = "Trance"
# Result: 'Radio/General' (from radio show detection)
```

## Why Hip-Hop Misclassification Happened

The issue occurred because:

1. **No genre detection**: The simplified categorization didn't use the `_detect_music_genre` method
2. **Audio features only**: It relied solely on danceability, complexity, etc.
3. **Fallback to default**: Many trance tracks fell through to the default category
4. **Hip-hop detection**: The `_detect_music_genre` method has hip-hop keywords that might have been triggered incorrectly

## Improved Detection

### Enhanced Trance Keywords
```python
'trance': ['trance', 'asot', 'state of trance', 'armin', 'tiesto', 'paul van dyk']
```

### BPM-Based Detection
```python
# Trance: 125-150 BPM, Techno: 120-140 BPM, House: 115-130 BPM
if 115 <= bpm <= 150 and danceability > 0.6:
    return 'Electronic/Dance'
```

### Priority Order
1. **Radio/Podcast/Mix detection** (content type)
2. **Genre detection** (metadata keywords + BPM)
3. **Audio features fallback** (danceability, complexity)

## Expected Results

After the fix, trance tracks should be categorized as:
- **Electronic/Dance** - Most trance tracks
- **Radio/General** - Trance radio shows
- **Mix/General** - Trance mix compilations

No more misclassification as hip-hop! 