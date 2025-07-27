# Using Playlist Generator with Custom Paths

You can specify custom library, cache, and output directories when running the playlist generator.

## Basic Usage

```bash
playlista --analyze --library /path/to/music --cache_dir /path/to/cache --output_dir /path/to/output
```

## Available Path Arguments

- `--library`: Your music library directory (default: auto-detected)
- `--cache_dir`: Cache directory for analysis data (default: `/app/cache`)
- `--output_dir`: Output directory for generated playlists (default: `/app/playlists`)

## Example Commands

### 1. Analyze Your Test Library
```bash
playlista --analyze --library /tmp --cache_dir /tmp/playlist_cache --output_dir /tmp/playlist_output --workers 2
```

### 2. Generate Playlists Only (from existing analysis)
```bash
playlista --generate_only --library /tmp --cache_dir /tmp/playlist_cache --output_dir /tmp/playlist_output
```

### 3. Show Library Statistics
```bash
playlista --status --library /tmp --cache_dir /tmp/playlist_cache
```

### 4. Analyze with Specific Playlist Method
```bash
playlista --analyze --playlist_method tags --library /tmp --cache_dir /tmp/playlist_cache --output_dir /tmp/playlist_output
```

### 5. Force Re-analyze All Files
```bash
playlista --analyze --force --library /tmp --cache_dir /tmp/playlist_cache --output_dir /tmp/playlist_output
```

### 6. Re-analyze Only Failed Files
```bash
playlista --analyze --failed --library /tmp --cache_dir /tmp/playlist_cache --output_dir /tmp/playlist_output
```

### 7. Low Memory Mode (for large files)
```bash
playlista --analyze --low_memory --workers 1 --library /tmp --cache_dir /tmp/playlist_cache --output_dir /tmp/playlist_output
```

## Quick Test with Your Copied Files

If you used the `copy_audio_files.sh` script to copy files to `/tmp`, you can test with:

```bash
# Start the container
./run.sh

# Inside the container, analyze the copied files
playlista --analyze --library /tmp --cache_dir /tmp/playlist_cache --output_dir /tmp/playlist_output --workers 2

# Generate playlists
playlista --generate_only --library /tmp --cache_dir /tmp/playlist_cache --output_dir /tmp/playlist_output

# Check results
ls -la /tmp/playlist_output/
```

## Directory Structure

```
/tmp/                          # Your test music library
├── song1.mp3
├── song2.flac
└── ...

/tmp/playlist_cache/           # Analysis cache
├── audio_features.db
└── ...

/tmp/playlist_output/          # Generated playlists
├── playlist1.m3u
├── playlist2.m3u
└── ...
```

## Tips

1. **Use `/tmp` for testing** - Files in `/tmp` are temporary and will be cleaned up
2. **Start with small libraries** - Test with a few files first
3. **Use `--workers 1`** for debugging or low memory situations
4. **Use `--low_memory`** for large files or limited RAM
5. **Check the cache directory** to see analysis progress and database files 