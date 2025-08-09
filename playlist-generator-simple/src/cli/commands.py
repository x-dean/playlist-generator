"""
CLI command handlers for Playlist Generator Simple.
Split from enhanced_cli.py for better maintainability.
"""

import argparse
from typing import Dict, List, Any, Optional
from pathlib import Path

import sys
sys.path.append('/app')

from src.core.analysis_manager import get_analysis_manager
from src.core.single_analyzer import analyze_files
from src.core.resource_manager import ResourceManager
from src.core.database import DatabaseManager, get_db_manager
from src.core.playlist_generator import PlaylistGenerator, PlaylistGenerationMethod
from src.core.comprehensive_manager import get_comprehensive_manager
from src.core.logging_setup import get_logger

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
            
            log_universal('INFO', 'CLI', f"Starting analysis of {music_path}")
            
            analysis_manager = get_analysis_manager()
            
            # Select files for analysis
            files = analysis_manager.select_files_for_analysis(
                music_path=music_path,
                force_reextract=force,
                include_failed=False
            )
            
            if not files:
                log_universal('INFO', 'CLI', "No files found for analysis")
                return 0
            
            log_universal('INFO', 'CLI', f"Found {len(files)} files to analyze")
            
            # Analyze the files
            result = analysis_manager.analyze_files(
                files=files,
                force_reextract=force,
                max_workers=workers
            )
            
            if result.get('success', False):
                log_universal('INFO', 'CLI', "Analysis completed successfully")
                log_universal('INFO', 'CLI', f"Processed: {result.get('analyzed_count', 0)} files")
                if result.get('failed_count', 0) > 0:
                    log_universal('WARNING', 'CLI', f"Failed: {result.get('failed_count', 0)} files")
                return 0
            else:
                log_universal('ERROR', 'CLI', "Analysis failed")
                return 1
                
        except Exception as e:
            log_universal('ERROR', 'CLI', f"Analysis error: {str(e)}")
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
            log_universal('ERROR', 'CLI', f"Stats error: {str(e)}")
            return 1
    
    @staticmethod
    def handle_retry_failed(args) -> int:
        """Handle retry-failed command."""
        try:
            analysis_manager = AnalysisManager()
            result = analysis_manager.retry_failed_analysis()
            
            if result:
                log_universal('INFO', 'CLI', "Retry completed successfully")
                return 0
            else:
                log_universal('ERROR', 'CLI', "Retry failed")
                return 1
                
        except Exception as e:
            log_universal('ERROR', 'CLI', f"Retry error: {str(e)}")
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


class ManagerCommands:
    """Handle comprehensive PLAYLISTA Manager commands."""
    
    @staticmethod
    def handle_manager_status(args) -> int:
        """Handle comprehensive manager status command."""
        try:
            manager = get_comprehensive_manager()
            status = manager.get_comprehensive_status()
            
            print("=" * 80)
            print("🎯 COMPREHENSIVE PLAYLISTA MANAGER STATUS")
            print("=" * 80)
            
            # Manager info
            manager_info = status['manager_info']
            print(f"\n📊 Manager Status:")
            print(f"  State: {manager_info['state'].upper()}")
            print(f"  Current Mode: {manager_info.get('current_mode', 'None')}")
            print(f"  Uptime: {manager_info['uptime_seconds']:.1f} seconds")
            
            # PLAYLISTA compliance
            compliance = status['playlista_compliance']
            print(f"\n🎯 PLAYLISTA Pattern 4 Compliance: {compliance['compliance_score']}%")
            
            # System health
            health = status['system_health']
            print(f"\n📈 System Health: {health['overall_health'].upper()}")
            print(f"  Health Score: {health['health_score']:.1%}")
            print(f"  Memory Health: {health['memory_health']}")
            print(f"  CPU Health: {health['cpu_health']}")
            
            # File status
            file_status = status['file_status']
            print(f"\n💾 Database Status:")
            print(f"  Total Files: {file_status['total_files']}")
            print(f"  Completion Rate: {file_status['completion_rate']:.1f}%")
            print(f"  Failure Rate: {file_status['failure_rate']:.1f}%")
            
            # Resource status
            resource_status = status['resource_status']
            current_resources = resource_status['current_resources']
            print(f"\n🎛️ Resource Status:")
            print(f"  Available Memory: {current_resources.get('available_memory_gb', 0):.1f} GB")
            print(f"  CPU Cores: {current_resources.get('cpu_cores', 0)}")
            print(f"  Memory Usage: {current_resources.get('memory_used_percent', 0):.1f}%")
            print(f"  CPU Usage: {current_resources.get('cpu_usage_percent', 0):.1f}%")
            
            return 0
            
        except Exception as e:
            logger.error(f"Manager status command failed: {e}")
            return 1
    
    @staticmethod
    def handle_manager_analyze(args) -> int:
        """Handle comprehensive manager analysis command."""
        try:
            manager = get_comprehensive_manager()
            
            music_path = args.music_path or '/music'
            force = args.force
            
            print(f"🎯 Starting PLAYLISTA Manager Analysis")
            print(f"Music Path: {music_path}")
            print(f"Force Reanalysis: {force}")
            
            # Start analysis using comprehensive manager
            operation_id = manager.feed_analyzers_with_tracks(
                files=None,  # Auto-select files
                force_reanalysis=force
            )
            
            if operation_id == "no_files_needed":
                print("✅ No files need analysis")
                return 0
            
            print(f"🚀 Analysis pipeline started: {operation_id}")
            print("Use 'playlista manager status' to track progress")
            
            return 0
            
        except Exception as e:
            logger.error(f"Manager analysis command failed: {e}")
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