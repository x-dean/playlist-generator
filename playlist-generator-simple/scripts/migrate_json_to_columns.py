#!/usr/bin/env python3
"""
Migration script to convert JSON blobs to proper columns in the tracks table.
This improves query performance and makes the data more accessible.
"""

import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.database import get_db_manager
from core.logging_setup import log_universal

def main():
    """Run the JSON to columns migration."""
    try:
        log_universal('INFO', 'Migration', 'Starting JSON to columns migration...')
        
        # Get database manager
        db_manager = get_db_manager()
        
        # Run migration
        stats = db_manager.migrate_json_to_columns()
        
        if 'error' in stats:
            log_universal('ERROR', 'Migration', f'Migration failed: {stats["error"]}')
            return False
        
        log_universal('INFO', 'Migration', f'Migration completed successfully:')
        log_universal('INFO', 'Migration', f'  - Tracks processed: {stats["tracks_processed"]}')
        log_universal('INFO', 'Migration', f'  - Values migrated: {stats["values_migrated"]}')
        log_universal('INFO', 'Migration', f'  - Errors: {stats["errors"]}')
        
        return True
        
    except Exception as e:
        log_universal('ERROR', 'Migration', f'Migration script failed: {e}')
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1) 