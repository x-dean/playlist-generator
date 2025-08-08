"""
CLI command handlers for Playlist Generator Simple.
Split from enhanced_cli.py for better maintainability.
"""

import argparse
from typing import Dict, List, Any, Optional
from pathlib import Path

from core.analysis_manager import AnalysisManager
from core.resource_manager import ResourceManager
from core.audio_analyzer import AudioAnalyzer
from core.database import DatabaseManager, get_db_manager
from core.playlist_generator import PlaylistGenerator, PlaylistGenerationMethod
from core.logging_setup import get_logger

logger = get_logger('playlista.cli.commands')


class AnalysisCommands:
    """Handle analysis-related CLI commands."""
    
    @staticmethod
    def handle_analyze(args) -> int:
        """Handle analyze command."""
        try:
            music_path = args.music_path or '/music'
            force = args.force
            no_cache = args.no_cache
            workers = args.workers
            
            logger.info(f"Starting analysis of {music_path}")
            
            analysis_manager = AnalysisManager()
            
            # Select files for analysis
            files = analysis_manager.select_files_for_analysis(
                music_path=music_path,
                force_reextract=force,
                include_failed=False
            )
            
            if not files:
                logger.info("No files found for analysis")
                return 0
            
            logger.info(f"Found {len(files)} files to analyze")
            
            # Analyze the files
            result = analysis_manager.analyze_files(
                files=files,
                force_reextract=force,
                max_workers=workers
            )
            
            if result.get('success', False):
                logger.info("Analysis completed successfully")
                logger.info(f"Analyzed: {result.get('analyzed_count', 0)} files")
                logger.info(f"Failed: {result.get('failed_count', 0)} files")
                return 0
            else:
                logger.error("Analysis failed")
                return 1
                
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            return 1
    
    @staticmethod
    def handle_stats(args) -> int:
        """Handle stats command."""
        try:
            db_manager = get_db_manager()
            stats = db_manager.get_database_statistics()
            
            print(f"Total tracks: {stats.get('total_tracks', 0)}")
            print(f"Analyzed tracks: {stats.get('analyzed_tracks', 0)}")
            print(f"Failed tracks: {stats.get('failed_tracks', 0)}")
            print(f"Database size: {stats.get('db_size_mb', 0):.2f} MB")
            
            return 0
        except Exception as e:
            logger.error(f"Stats error: {e}")
            return 1
    
    @staticmethod
    def handle_retry_failed(args) -> int:
        """Handle retry-failed command."""
        try:
            analysis_manager = AnalysisManager()
            result = analysis_manager.retry_failed_analysis()
            
            if result:
                logger.info("Retry completed successfully")
                return 0
            else:
                logger.error("Retry failed")
                return 1
                
        except Exception as e:
            logger.error(f"Retry error: {e}")
            return 1


class PlaylistCommands:
    """Handle playlist-related CLI commands."""
    
    @staticmethod
    def handle_playlist(args) -> int:
        """Handle playlist generation command."""
        try:
            method = args.method
            num_playlists = args.num_playlists or 5
            use_advanced = args.use_advanced_features
            
            logger.info(f"Generating {num_playlists} playlists using {method} method")
            
            playlist_generator = PlaylistGenerator()
            result = playlist_generator.generate_playlists(
                method=method,
                num_playlists=num_playlists,
                use_advanced_features=use_advanced
            )
            
            if result:
                logger.info("Playlist generation completed successfully")
                return 0
            else:
                logger.error("Playlist generation failed")
                return 1
                
        except Exception as e:
            logger.error(f"Playlist generation error: {e}")
            return 1
    
    @staticmethod
    def handle_playlist_methods(args) -> int:
        """Handle playlist-methods command."""
        try:
            methods = PlaylistGenerationMethod.get_available_methods()
            print("Available playlist generation methods:")
            for method in methods:
                print(f"  - {method}")
            return 0
        except Exception as e:
            logger.error(f"Playlist methods error: {e}")
            return 1


class DatabaseCommands:
    """Handle database-related CLI commands."""
    
    @staticmethod
    def handle_db(args) -> int:
        """Handle database management commands."""
        try:
            db_manager = get_db_manager()
            
            if args.init:
                db_manager.initialize_database()
                logger.info("Database initialized successfully")
                return 0
            elif args.integrity_check:
                result = db_manager.check_integrity()
                if result:
                    logger.info("Database integrity check passed")
                    return 0
                else:
                    logger.error("Database integrity check failed")
                    return 1
            elif args.backup:
                db_manager.create_backup()
                logger.info("Database backup created successfully")
                return 0
            elif args.vacuum:
                db_manager.vacuum_database()
                logger.info("Database vacuum completed")
                return 0
            else:
                logger.error("No database operation specified")
                return 1
                
        except Exception as e:
            logger.error(f"Database operation error: {e}")
            return 1
    
    @staticmethod
    def handle_validate_database(args) -> int:
        """Handle validate-database command."""
        try:
            db_manager = get_db_manager()
            result = db_manager.validate_database_structure()
            
            if result:
                logger.info("Database validation passed")
                return 0
            else:
                logger.error("Database validation failed")
                return 1
                
        except Exception as e:
            logger.error(f"Database validation error: {e}")
            return 1


class UtilityCommands:
    """Handle utility CLI commands."""
    
    @staticmethod
    def handle_status(args) -> int:
        """Handle status command."""
        try:
            print("Playlist Generator Simple Status")
            print("=" * 40)
            
            # System status
            resource_manager = ResourceManager()
            resource_info = resource_manager.get_current_resources()
            memory_info = resource_info.get('memory', {})
            print(f"Memory usage: {memory_info.get('used_gb', 0):.1f} GB / {memory_info.get('total_gb', 0):.1f} GB")
            print(f"CPU usage: {resource_info.get('cpu_percent', 0):.1f}%")
            
            # Database status
            db_manager = get_db_manager()
            stats = db_manager.get_statistics()
            print(f"Database tracks: {stats.get('total_tracks', 0)}")
            
            return 0
        except Exception as e:
            logger.error(f"Status error: {e}")
            return 1
    
    @staticmethod
    def handle_cleanup(args) -> int:
        """Handle cleanup command."""
        try:
            analysis_manager = AnalysisManager()
            result = analysis_manager.cleanup_old_data()
            
            if result:
                logger.info("Cleanup completed successfully")
                return 0
            else:
                logger.error("Cleanup failed")
                return 1
                
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
            return 1 