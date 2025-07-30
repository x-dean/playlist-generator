# Progress Bar Feature

A simple rich-based progress bar implementation for the playlist generator simple version.

## Features

- **Visual Progress Bars**: Real-time progress tracking with rich library
- **File Processing Progress**: Shows current file being processed
- **Analysis Progress**: Separate progress for sequential and parallel analysis
- **Status Messages**: Colored status, success, warning, and error messages
- **Results Tables**: Beautiful result summaries with statistics
- **Logging Fallback**: Falls back to logging when progress bars are disabled

## Usage

### Basic Usage

```python
from core.progress_bar import get_progress_bar

# Get progress bar instance
progress_bar = get_progress_bar()

# Start file processing
progress_bar.start_file_processing(100, "Processing Music Files")

# Update progress
for i, file in enumerate(files, 1):
    progress_bar.update_file_progress(i, file)

# Complete with results
progress_bar.complete_file_processing(100, 95, 5)
```

### Analysis Progress

```python
# Start analysis
progress_bar.start_analysis(50, "Sequential Analysis")

# Update during processing
progress_bar.update_analysis_progress(i, current_file)

# Complete analysis
progress_bar.complete_analysis(50, 48, 2, "Sequential Analysis")
```

### Status Messages

```python
progress_bar.show_status("Starting analysis...", "blue")
progress_bar.show_success("Analysis completed!")
progress_bar.show_warning("Some files were skipped")
progress_bar.show_error("Failed to process file")
```

## Configuration

Add to `playlista.conf`:

```ini
# Progress Bar Settings
PROGRESS_BAR_ENABLED=true
PROGRESS_BAR_SHOW_PERCENTAGE=true
PROGRESS_BAR_SHOW_TIME_ELAPSED=true
PROGRESS_BAR_SHOW_CURRENT_FILE=true
```

## Integration

The progress bar is automatically integrated into:

- **Sequential Analyzer**: Shows progress for large file processing
- **Parallel Analyzer**: Shows progress for parallel file processing  
- **Analysis Manager**: Shows overall analysis progress

## Testing

Run the progress bar test:

```bash
python test_progress_bar.py
```

This demonstrates all features including file processing, analysis progress, and status messages. 