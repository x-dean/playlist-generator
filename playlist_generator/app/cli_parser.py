#!/usr/bin/env python3
"""
CLI argument parser for the playlist generator application.
"""

import argparse
import os
import sys
from typing import Optional
from datetime import datetime


def get_host_library_path() -> str:
    """Get the host library path from Docker volume mount."""
    # Try to get from environment variable first
    host_library = os.getenv('HOST_LIBRARY_PATH')
    if host_library:
        return host_library

    # Try to detect from /proc/mounts (Linux containers)
    try:
        with open('/proc/mounts', 'r') as f:
            for line in f:
                if '/music' in line:
                    parts = line.split()
                    if len(parts) >= 2:
                        host_path = parts[0]
                        container_path = parts[1]
                        if container_path == '/music':
                            # Filter out loop devices and prefer actual host paths
                            if not host_path.startswith('/dev/loop'):
                                return host_path
                            # If we only find loop devices, try to resolve the real path
                            elif host_path.startswith('/dev/loop'):
                                # Try to get the real path from the loop device
                                try:
                                    import subprocess
                                    result = subprocess.run(['losetup', '-l'],
                                                            capture_output=True, text=True, timeout=5)
                                    if result.returncode == 0:
                                        for losetup_line in result.stdout.splitlines():
                                            if host_path in losetup_line:
                                                # Extract the backing file path
                                                parts = losetup_line.split()
                                                if len(parts) >= 7:  # Format: device backing_file
                                                    backing_file = parts[5]
                                                    # Extract the directory part
                                                    if '/music' in backing_file:
                                                        # Find the host path by removing the container path
                                                        host_dir = backing_file.replace(
                                                            '/music', '').rstrip('/')
                                                        if host_dir:
                                                            return host_dir
                                except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
                                    pass
    except (FileNotFoundError, PermissionError):
        pass

    # Try to get from Docker inspect (if available)
    try:
        import subprocess
        result = subprocess.run(['docker', 'inspect', '--format', '{{.Mounts}}', 'playlista'],
                                capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            # Parse the mount information
            mounts = result.stdout.strip()
            if '/music' in mounts:
                # Extract the host path from the mount info
                import re
                match = re.search(r'([^:]+):/music', mounts)
                if match:
                    return match.group(1)
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        pass

    # Try to get from /proc/self/mountinfo (more detailed mount info)
    try:
        with open('/proc/self/mountinfo', 'r') as f:
            for line in f:
                if '/music' in line:
                    parts = line.split()
                    if len(parts) >= 5:
                        # Format: ID MAJOR:MINOR ROOT MOUNT_POINT OPTIONS
                        mount_point = parts[4]
                        if mount_point == '/music':
                            # The host path is in the root field (parts[3])
                            # But we need to resolve it properly
                            root_path = parts[3]
                            # This is complex - let's try a different approach
                            pass
    except (FileNotFoundError, PermissionError):
        pass

    # If all else fails, use the default
    return '/root/music/library'


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description='''
Playlista: Fast, flexible music playlist generator and analyzer.

- Analyze your music library for audio features
- Generate playlists using multiple methods (feature-group, time, kmeans, cache, tags)
- Designed for Docker and large libraries
''',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog='''
Usage Tips:
  - All paths are fixed for Docker: /music, /app/playlists, /app/cache
  - Use --workers N for parallel processing (default: all CPUs)
  - Use --workers=1 for sequential processing (debugging/low memory)
  - Use --analyze to analyze new/changed files
  - Use --generate_only to generate playlists from existing analysis
  - Use --update to regenerate all playlists
  - Use --failed to re-analyze only failed tracks
  - Use --status to show library/database stats
  - Use --playlist_method to select playlist generation method
  - Use --min_tracks_per_genre for tag-based playlists

Examples:
  playlista --analyze --workers 8
  playlista --generate_only
  playlista --update
  playlista --analyze --workers=1
  playlista --playlist_method tags --min_tracks_per_genre 15
''')
    
    # -h/--help is available by default in argparse
    # Remove --music_dir argument
    # Rename --library to --library
    parser.add_argument('--library', default=get_host_library_path(),
                        help='Host music library directory (auto-detected from Docker volume mount)')
    # Output directory for playlists
    parser.add_argument(
        '--output_dir', default='/app/playlists', help='Output directory')
    # Cache directory
    parser.add_argument('--cache_dir', default='/app/cache',
                        help='Cache directory')
    # Number of playlists to generate
    parser.add_argument('--num_playlists', type=int, default=8,
                        help='Number of playlists to generate')
    # Number of workers for parallel processing
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of workers (default: auto, use --workers=1 for sequential processing)')
    
    # Memory management options
    parser.add_argument('--memory_limit', type=str, default=None,
                        help='Memory limit per worker (e.g., "2GB", "512MB")')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size for processing (default: equals number of workers)')
    parser.add_argument('--low_memory', action='store_true',
                        help='Enable low memory mode (reduces workers and batch size)')
    parser.add_argument('--large_file_threshold', type=int, default=50,
                        help='File size threshold in MB for separate process handling (default: 50MB)')
    parser.add_argument('--memory_aware', action='store_true',
                        help='Enable memory-aware processing (skip memory-intensive features when memory is low)')
    parser.add_argument('--rss_limit_gb', type=float, default=6.0,
                        help='Max total Python RSS (GB) before aborting/skipping (default: 6.0)')

    # Analyze files (default mode)
    parser.add_argument('-a', '--analyze', action='store_true',
                        help='Analyze files (see --failed and --force for options)')
    # Only generate playlists from database (no analysis)
    parser.add_argument('-g', '--generate_only', action='store_true',
                        help='Only generate playlists from database (no analysis)')
    # Update all playlists from database (no analysis, regenerates all playlists)
    parser.add_argument('-u', '--update', action='store_true',
                        help='Update all playlists from database (no analysis, regenerates all playlists)')
    # With --analyze: only re-analyze files previously marked as failed
    parser.add_argument('--failed', action='store_true',
                        help='With --analyze: only re-analyze files previously marked as failed')
    # Force re-analyze (used with --analyze)
    parser.add_argument('-f', '--force', action='store_true',
                        help='Force re-analyze (used with --analyze)')
    # Show library/database statistics and exit
    parser.add_argument('--status', action='store_true',
                        help='Show library/database statistics and exit')
    # Playlist generation method
    parser.add_argument('-m', '--playlist_method', choices=['all', 'time', 'kmeans', 'cache', 'tags'], default='all',
                        help='Playlist generation method: all (feature-group, default), time, kmeans, cache, or tags (genre+decade)')
    # Minimum number of tracks required for a genre to create a playlist (tags method only)
    parser.add_argument('--min_tracks_per_genre', type=int, default=10,
                        help='Minimum number of tracks required for a genre to create a playlist (tags method only)')
    # Set the logging level
    parser.add_argument('--log_level', default='INFO', choices=[
                        'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help='Set the logging level (default: INFO)')
    # Bypass the cache and re-extract features for all files
    parser.add_argument('--no_cache', action='store_true',
                        help='Bypass the cache and re-extract features for all files')
    # Run full pipeline: analyze, force, then failed
    parser.add_argument('--pipeline', action='store_true',
                        help='Run full pipeline: analyze, force, then failed')
    
    return parser


def parse_early_args() -> argparse.Namespace:
    """Parse early arguments (like --log_level) before full argument parsing."""
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument('--log_level', default='INFO',
                            choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    pre_args, _ = pre_parser.parse_known_args(sys.argv[1:])
    return pre_args


def validate_args(args: argparse.Namespace) -> None:
    """Validate parsed arguments."""
    # Validate that at least one mode is specified
    modes = [args.analyze, args.generate_only, args.update, args.status, args.pipeline]
    if not any(modes):
        # Default to analyze mode if no mode specified
        args.analyze = True
    
    # Validate workers count
    if args.workers is not None and args.workers < 1:
        raise ValueError("Workers count must be at least 1")
    
    # Validate num_playlists
    if args.num_playlists < 1:
        raise ValueError("Number of playlists must be at least 1")
    
    # Validate min_tracks_per_genre
    if args.min_tracks_per_genre < 1:
        raise ValueError("Minimum tracks per genre must be at least 1")
    
    # Validate large_file_threshold
    if args.large_file_threshold < 1:
        raise ValueError("Large file threshold must be at least 1MB")
    
    # Validate rss_limit_gb
    if args.rss_limit_gb <= 0:
        raise ValueError("RSS limit must be positive")


def setup_environment_vars(args: argparse.Namespace) -> None:
    """Set up environment variables based on parsed arguments."""
    # Set basic environment variables
    os.environ['LARGE_FILE_THRESHOLD'] = str(args.large_file_threshold)
    os.environ['MEMORY_AWARE'] = str(args.memory_aware).lower()
    os.environ['HOST_LIBRARY_PATH'] = args.library
    os.environ['MUSIC_PATH'] = '/music'
    
    # Set cache file environment variable
    cache_dir = os.getenv('CACHE_DIR', '/app/cache')
    cache_file = os.path.join(cache_dir, 'audio_analysis.db')
    os.environ['CACHE_FILE'] = cache_file 