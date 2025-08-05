# Sequential Processing and Lightweight Categories Fixes

## Issues Identified

### 1. **Inconsistent Threshold Logic**
**Problem**: Multiple conflicting thresholds in configuration
- `BIG_FILE_SIZE_MB=200` vs `PARALLEL_MAX_FILE_SIZE_MB=100` vs `SEQUENTIAL_MAX_FILE_SIZE_MB=2000`
- `VERY_LARGE_FILE_THRESHOLD_MB=200` duplicated functionality

**Fix**: Aligned all thresholds to be consistent
- Files > 200MB use sequential processing
- Files 100-200MB use parallel with multi-segment
- Files < 100MB use parallel with full analysis

### 2. **Sequential Analyzer Logic Problems**
**Problem**: Redundant condition in `_get_analysis_config()`
```python
if file_size_mb >= 200:
    analysis_type = 'basic'
    use_full_analysis = False
else:  # Should not reach here
    analysis_type = 'basic'  # Same as above
    use_full_analysis = False
```

**Fix**: 
- Removed redundant condition
- Added proper warning for files that shouldn't be in sequential analyzer
- Disabled MusicNN for files >= 200MB to save memory
- Enabled lightweight categorization for sequential processing

### 3. **Lightweight Categorization Issues**
**Problem**: Overly complex categorization logic with too many fallbacks
- Complex nested conditions in `_smart_categorize_with_genre()`
- Heavy dependency on metadata that may be missing
- Inconsistent feature extraction methods

**Fix**: 
- Simplified to `_simplified_categorization()` with clear priority order
- Reduced categories to essential types: Radio, Podcast, Mix, Electronic/Dance, Ambient/Chill, Rock/Metal, Speech/Spoken, Pop/Indie
- Made categorization more reliable by using audio features as primary decision maker

### 4. **Memory Management Issues**
**Problem**: 
- No proper TensorFlow session cleanup
- Memory usage not monitored during sequential processing
- No warnings for high memory usage

**Fix**:
- Added TensorFlow session cleanup in `_cleanup_memory()`
- Added memory usage monitoring and warnings
- Improved garbage collection

### 5. **Database Integration Problems**
**Problem**:
- Lightweight categories not properly saved to database
- Inconsistent field mapping between lightweight and full analysis

**Fix**:
- Added proper database saving for lightweight categories
- Set both `lightweight_category` and `long_audio_category` fields
- Added error handling for missing features

## Configuration Changes

### Fixed Thresholds
```ini
# Aligned all thresholds
BIG_FILE_SIZE_MB=200
PARALLEL_MAX_FILE_SIZE_MB=200
SEQUENTIAL_MAX_FILE_SIZE_MB=2000
VERY_LARGE_FILE_THRESHOLD_MB=200
```

### Sequential Processing Settings
```ini
# Sequential processing now properly configured
ENABLE_LIGHTWEIGHT_CATEGORIZATION=true
SKIP_MUSICNN_FOR_VERY_LARGE_FILES=true
LONG_AUDIO_DURATION_THRESHOLD_MINUTES=45
```

## Code Changes

### Sequential Analyzer (`sequential_analyzer.py`)
1. **Fixed analysis configuration logic**
   - Removed redundant condition
   - Added proper MusicNN disabling for large files
   - Enabled lightweight categorization

2. **Improved memory management**
   - Added TensorFlow session cleanup
   - Added memory usage monitoring
   - Better error handling

### Audio Analyzer (`audio_analyzer.py`)
1. **Simplified lightweight categorization**
   - Replaced complex `_smart_categorize_with_genre()` with `_simplified_categorization()`
   - Reduced categories to essential types
   - Made audio features primary decision maker

2. **Fixed database integration**
   - Added proper saving of lightweight categories
   - Set both category fields for consistency
   - Added error handling

3. **Improved main analysis flow**
   - Better handling of very large files
   - Proper lightweight category creation and storage
   - Added warnings for missing features

## Testing Recommendations

1. **Test with various file sizes**:
   - Files < 100MB (should use parallel)
   - Files 100-200MB (should use parallel with multi-segment)
   - Files > 200MB (should use sequential with lightweight categorization)

2. **Test memory usage**:
   - Monitor memory during sequential processing
   - Verify TensorFlow session cleanup
   - Check for memory leaks

3. **Test categorization accuracy**:
   - Verify lightweight categories are assigned correctly
   - Check database storage of categories
   - Test with missing metadata

## Expected Behavior After Fixes

1. **Sequential Processing**:
   - Only handles files >= 200MB
   - Uses lightweight categorization instead of MusicNN
   - Proper memory cleanup between files
   - Clear logging of processing decisions

2. **Lightweight Categories**:
   - Simple, reliable categorization
   - Based on audio features when metadata is missing
   - Properly saved to database
   - Consistent category names

3. **Memory Management**:
   - TensorFlow sessions cleared after each file
   - Memory usage monitored and logged
   - Warnings when memory usage is high
   - Better garbage collection

## Monitoring

Check logs for:
- `"Sequential + Multi-segment processing"` for large files
- `"Lightweight category created"` for categorization
- `"Memory cleanup completed"` for memory management
- `"Cleared TensorFlow session"` for session cleanup
- `"Memory usage high"` warnings if memory limit exceeded 