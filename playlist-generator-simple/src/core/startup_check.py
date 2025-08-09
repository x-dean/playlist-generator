"""
Startup Check - Ensure PostgreSQL is configured.

Clean architecture requires PostgreSQL to be properly configured.
"""

import os
import sys
from .logging_setup import log_universal
from .config_loader import config_loader


def verify_database_config():
    """Verify PostgreSQL configuration is present."""
    config = config_loader.get_config()
    
    # Required PostgreSQL settings
    required_settings = {
        'POSTGRES_HOST': config.get('POSTGRES_HOST') or os.getenv('POSTGRES_HOST'),
        'POSTGRES_DB': config.get('POSTGRES_DB') or os.getenv('POSTGRES_DB'),
        'POSTGRES_USER': config.get('POSTGRES_USER') or os.getenv('POSTGRES_USER'),
        'POSTGRES_PASSWORD': config.get('POSTGRES_PASSWORD') or os.getenv('POSTGRES_PASSWORD')
    }
    
    missing_settings = [key for key, value in required_settings.items() if not value]
    
    if missing_settings:
        log_universal('ERROR', 'Startup', 'PostgreSQL configuration missing!')
        log_universal('ERROR', 'Startup', f'Missing settings: {", ".join(missing_settings)}')
        log_universal('ERROR', 'Startup', '')
        log_universal('ERROR', 'Startup', 'Quick fix options:')
        log_universal('ERROR', 'Startup', '1. Start with Docker: docker-compose up -d')
        log_universal('ERROR', 'Startup', '2. Set environment variables:')
        log_universal('ERROR', 'Startup', '   export POSTGRES_HOST=localhost')
        log_universal('ERROR', 'Startup', '   export POSTGRES_DB=playlista')
        log_universal('ERROR', 'Startup', '   export POSTGRES_USER=playlista')
        log_universal('ERROR', 'Startup', '   export POSTGRES_PASSWORD=your_password')
        log_universal('ERROR', 'Startup', '3. Update playlista.conf with PostgreSQL settings')
        log_universal('ERROR', 'Startup', '')
        sys.exit(1)
    
    log_universal('INFO', 'Startup', f'PostgreSQL configured: {required_settings["POSTGRES_HOST"]}:{config.get("POSTGRES_PORT", 5432)}')
    

def test_database_connection():
    """Test if database connection works."""
    try:
        from .database import get_db_manager
        db = get_db_manager()
        count = db.get_track_count()
        log_universal('INFO', 'Startup', f'✓ Database connection OK - {count} tracks found')
        return True
        
    except Exception as e:
        log_universal('ERROR', 'Startup', f'Database connection failed: {str(e)}')
        log_universal('ERROR', 'Startup', 'Solutions:')
        log_universal('ERROR', 'Startup', '1. Start PostgreSQL: docker-compose up -d postgres')
        log_universal('ERROR', 'Startup', '2. Initialize database: python database/setup_postgresql.py')
        log_universal('ERROR', 'Startup', '3. Check PostgreSQL is running on configured host/port')
        return False


if __name__ == '__main__':
    verify_database_config()
    test_database_connection()
