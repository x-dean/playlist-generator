#!/usr/bin/env python3
"""
Initialize database with complete schema.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.core.database import get_db_manager

def init_database():
    """Initialize database with complete schema."""
    print("Initializing database with complete schema...")
    
    try:
        # Get database manager
        db_manager = get_db_manager()
        
        # Initialize schema
        success = db_manager.initialize_schema()
        
        if success:
            print("✓ Database initialized successfully with complete schema")
            
            # Show schema information
            schema_info = db_manager.show_schema_info()
            print(f"✓ Schema created with {schema_info['total_columns']} columns")
            print(f"✓ {schema_info['total_indexes']} indexes created")
            print(f"✓ {schema_info['total_views']} views created")
            
            return True
        else:
            print("✗ Database initialization failed")
            return False
            
    except Exception as e:
        print(f"✗ Error initializing database: {e}")
        return False

if __name__ == "__main__":
    print("=== Database Initialization ===")
    
    success = init_database()
    
    if success:
        print("\n✓ Database is ready for use!")
        print("\nNext steps:")
        print("1. Run analysis: playlista analyze --music-path /path/to/music")
        print("2. Generate playlists: playlista playlist --method kmeans")
        print("3. Check status: playlista status")
    else:
        print("\n✗ Database initialization failed")
        sys.exit(1) 