"""
Simplified CLI main interface for Playlist Generator Simple.
Replaces the massive enhanced_cli.py with a cleaner, modular approach.
"""

import os
import sys
import argparse
from typing import Optional

# Suppress external library logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['ESSENTIA_LOG_LEVEL'] = 'error'
os.environ['MUSICEXTRACTOR_LOG_LEVEL'] = 'error'
os.environ['TENSORFLOW_LOG_LEVEL'] = '2'
os.environ['LIBROSA_LOG_LEVEL'] = 'error'

import sys
sys.path.append('/app')

from src.core.logging_setup import get_logger, setup_logging, log_universal
from src.core.config_loader import config_loader
from src.core.startup_check import verify_database_config
from src.cli.commands import AnalysisCommands, PlaylistCommands, DatabaseCommands, UtilityCommands, ManagerCommands

logger = get_logger('playlista.cli.main')


class SimpleCLI:
    """Simplified CLI interface with modular command handling."""
    
    def __init__(self):
        self.parser = self._create_argument_parser()
        self.config = config_loader.get_config()
        self._setup_logging()  # Initial setup with config file defaults
        # Ensure PostgreSQL is configured
        verify_database_config()
    
    def _setup_logging(self, args=None):
        """Setup logging based on configuration and CLI flags."""
        logging_config = config_loader.get_logging_config()
        log_level = logging_config.get('LOG_LEVEL', 'INFO')
        console_logging = logging_config.get('LOG_CONSOLE_ENABLED', True)
        file_logging = logging_config.get('LOG_FILE_ENABLED', True)
        
        # Override log level based on CLI flags
        if args:
            if args.debug:
                log_level = 'DEBUG'
            elif args.verbose:
                log_level = 'INFO'
        
        setup_logging(
            log_level=log_level,
            console_logging=console_logging,
            file_logging=file_logging
        )
    
    def _create_argument_parser(self) -> argparse.ArgumentParser:
        """Create the argument parser with all commands."""
        parser = argparse.ArgumentParser(
            description='Playlist Generator Simple - Music analysis and playlist generation',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  playlista analyze --music-path /music
  playlista -vv analyze --music-path /music  # Debug mode
  playlista playlist --method kmeans --num-playlists 5
  playlista manager status
  playlista manager analyze --force
  playlista stats
  playlista status
            """
        )
        
        # Global flags
        parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output (INFO level)')
        parser.add_argument('-vv', '--debug', action='store_true', help='Debug output (DEBUG level)')
        
        subparsers = parser.add_subparsers(dest='command', help='Available commands')
        
        # Analysis commands
        analyze_parser = subparsers.add_parser('analyze', help='Analyze music files')
        analyze_parser.add_argument('--music-path', help='Path to music directory')
        analyze_parser.add_argument('--force', action='store_true', help='Force re-analysis')
        analyze_parser.add_argument('--no-cache', action='store_true', help='Skip cache')
        analyze_parser.add_argument('--workers', type=int, help='Number of workers')
        
        stats_parser = subparsers.add_parser('stats', help='Show analysis statistics')
        
        retry_parser = subparsers.add_parser('retry-failed', help='Retry failed analysis')
        
        # Playlist commands
        playlist_parser = subparsers.add_parser('playlist', help='Generate playlists')
        playlist_parser.add_argument('--method', default='all', help='Generation method')
        playlist_parser.add_argument('--num-playlists', type=int, default=5, help='Number of playlists')
        playlist_parser.add_argument('--use-advanced-features', action='store_true', help='Use advanced features')
        
        methods_parser = subparsers.add_parser('playlist-methods', help='List available methods')
        
        # Database commands
        db_parser = subparsers.add_parser('db', help='Database management')
        db_parser.add_argument('--init', action='store_true', help='Initialize database')
        db_parser.add_argument('--integrity-check', action='store_true', help='Check integrity')
        db_parser.add_argument('--backup', action='store_true', help='Create backup')
        db_parser.add_argument('--vacuum', action='store_true', help='Vacuum database')
        
        validate_parser = subparsers.add_parser('validate-database', help='Validate database structure')
        
        # Manager commands (Comprehensive PLAYLISTA Manager)
        manager_parser = subparsers.add_parser('manager', help='Comprehensive PLAYLISTA Manager')
        manager_subparsers = manager_parser.add_subparsers(dest='manager_command', help='Manager commands')
        
        manager_status_parser = manager_subparsers.add_parser('status', help='Show comprehensive manager status')
        
        manager_analyze_parser = manager_subparsers.add_parser('analyze', help='Start PLAYLISTA Manager analysis')
        manager_analyze_parser.add_argument('--music-path', help='Path to music directory')
        manager_analyze_parser.add_argument('--force', action='store_true', help='Force re-analysis')
        
        # Utility commands
        status_parser = subparsers.add_parser('status', help='Show system status')
        cleanup_parser = subparsers.add_parser('cleanup', help='Clean up old data')
        
        return parser
    
    def run(self, args: Optional[list] = None) -> int:
        """Run the CLI with the given arguments."""
        try:
            parsed_args = self.parser.parse_args(args)
            
            # Re-setup logging based on CLI flags
            self._setup_logging(parsed_args)
            
            if not parsed_args.command:
                self.parser.print_help()
                return 0
            
            # Route to appropriate command handler
            return self._handle_command(parsed_args)
            
        except KeyboardInterrupt:
            logger.info("Operation cancelled by user")
            return 1
        except Exception as e:
            logger.error(f"CLI error: {e}")
            return 1
    
    def _handle_command(self, args) -> int:
        """Route commands to appropriate handlers."""
        command = args.command
        
        # Analysis commands
        if command == 'analyze':
            return AnalysisCommands.handle_analyze(args)
        elif command == 'stats':
            return AnalysisCommands.handle_stats(args)
        elif command == 'retry-failed':
            return AnalysisCommands.handle_retry_failed(args)
        
        # Playlist commands
        elif command == 'playlist':
            return PlaylistCommands.handle_playlist(args)
        elif command == 'playlist-methods':
            return PlaylistCommands.handle_playlist_methods(args)
        
        # Database commands
        elif command == 'db':
            return DatabaseCommands.handle_db(args)
        elif command == 'validate-database':
            return DatabaseCommands.handle_validate_database(args)
        
        # Manager commands
        elif command == 'manager':
            manager_command = getattr(args, 'manager_command', None)
            if manager_command == 'status':
                return ManagerCommands.handle_manager_status(args)
            elif manager_command == 'analyze':
                return ManagerCommands.handle_manager_analyze(args)
            else:
                logger.error("No manager command specified. Use 'playlista manager --help'")
                return 1
        
        # Utility commands
        elif command == 'status':
            return UtilityCommands.handle_status(args)
        elif command == 'cleanup':
            return UtilityCommands.handle_cleanup(args)
        
        else:
            logger.error(f"Unknown command: {command}")
            return 1


def main():
    """Main entry point for the CLI."""
    cli = SimpleCLI()
    return cli.run()


if __name__ == '__main__':
    sys.exit(main())
