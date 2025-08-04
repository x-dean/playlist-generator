# Sequential Processing Failure Handling

## Problem

The original issue was that when a sequential process failed, the entire application would exit. This happened because the sequential analyzer ran in the same process as the main application, so any unhandled exception in the analysis would crash the entire app.

## Solution

The solution implements **process isolation** using multiprocessing to decouple the sequential analysis from the main application process.

### Key Changes

1. **Multiprocessing Isolation**: Each file is processed in a separate process using Python's `multiprocessing` module
2. **Timeout Handling**: Each analysis has a configurable timeout to prevent hanging processes
3. **Error Isolation**: Failures in the analysis process don't affect the main application
4. **Database Integration**: Failed files are properly marked in the database for later retry or cleanup

### Implementation Details

#### 1. Worker Function (`_worker_process_function`)

```python
def _worker_process_function(file_path: str, force_reextract: bool, timeout_seconds: int) -> Dict[str, Any]:
    """
    Worker function for multiprocessing - runs in isolated process.
    """
    # Set up timeout
    signal.signal(signal.SIGALRM, timeout_handler_worker)
    signal.alarm(timeout_seconds)
    
    try:
        # Import and initialize components in worker process
        from .audio_analyzer import AudioAnalyzer
        from .database import DatabaseManager
        
        # Process file and save to database
        result = analyzer.analyze_audio_file(file_path, force_reextract)
        
        if result:
            # Save to database
            success = db_manager.save_analysis_result(...)
            return {'success': success, 'error': None}
        else:
            return {'success': False, 'error': 'Analysis failed'}
            
    except Exception as e:
        return {'success': False, 'error': str(e)}
```

#### 2. Multiprocessing Method (`_process_single_file_multiprocessing`)

```python
def _process_single_file_multiprocessing(self, file_path: str, force_reextract: bool = False) -> bool:
    """
    Process a single file using multiprocessing for better isolation.
    """
    try:
        # Use multiprocessing to isolate the analysis
        with mp.Pool(processes=1) as pool:
            future = pool.apply_async(
                _worker_process_function, 
                args=(file_path, force_reextract, self.timeout_seconds)
            )
            
            # Wait for result with timeout
            result = future.get(timeout=self.timeout_seconds)
            
            if result['success']:
                return True
            else:
                # Mark as failed in database
                self.db_manager.mark_analysis_failed(file_path, filename, result['error'])
                return False
                
    except mp.TimeoutError:
        # Process timed out
        self.db_manager.mark_analysis_failed(file_path, filename, "Analysis timed out")
        return False
```

### Benefits

1. **Process Isolation**: Analysis failures don't crash the main application
2. **Timeout Protection**: Long-running analyses are automatically terminated
3. **Memory Management**: Each analysis runs in its own process with isolated memory
4. **Error Tracking**: Failed files are properly tracked in the database
5. **Unattended Processing**: The application can run unattended without crashing

### Configuration

The following settings can be configured:

- `timeout_seconds`: Maximum time for each analysis (default: 600 seconds)
- `memory_threshold_percent`: Memory threshold for cleanup (default: 85%)
- `rss_limit_gb`: RSS memory limit (default: 6.0 GB)

### Usage

The sequential analyzer now automatically uses process isolation:

```python
# Initialize analyzer
analyzer = SequentialAnalyzer()

# Process files - failures won't crash the app
results = analyzer.process_files(files, force_reextract=False)

# Check results
print(f"Success: {results['success_count']}")
print(f"Failed: {results['failed_count']}")
```

### Testing

Use the test script to verify the solution:

```bash
python test_sequential_failure_handling.py
```

This will test:
1. Basic failure handling with multiprocessing isolation
2. Process isolation to prevent main app crashes
3. Database integration for failed files

### Cleanup

Failed files can be cleaned up using the CLI:

```bash
# Show failed files
playlista cleanup --show-failed

# Move failed files to failed directory
playlista cleanup --move-to-failed-dir

# Clean up old failed analysis entries
playlista cleanup --max-retries 3
```

### Database Schema

Failed files are stored in the `analysis_cache` table:

```sql
CREATE TABLE analysis_cache (
    file_path TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    error_message TEXT,
    status TEXT DEFAULT 'failed',
    retry_count INTEGER DEFAULT 0,
    last_retry_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

This solution ensures that the application can run unattended without crashing when individual files fail to process. 