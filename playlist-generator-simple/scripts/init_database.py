#!/usr/bin/env python3
"""
Database initialization script for Playlist Generator Simple.
Creates the optimized database schema for web UI performance.
"""

import sqlite3
import os
import sys
from pathlib import Path

def init_database(db_path: str) -> bool:
    """
    Initialize database with optimized schema for web UI performance.
    
    Args:
        db_path: Path to the database file
        
    Returns:
        True if initialization successful, False otherwise
    """
    print(f"Initializing database: {db_path}")
    
    # Ensure database directory exists
    db_dir = os.path.dirname(db_path)
    os.makedirs(db_dir, exist_ok=True)
    
    # Connect to database
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
    except Exception as e:
        print(f"Failed to connect to database: {e}")
        return False
    
    try:
        # Check if database already has tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='tracks'")
        if cursor.fetchone():
            print("Database schema already exists.")
            return True
        
        # Look for schema files in multiple locations
        script_dir = os.path.dirname(os.path.abspath(__file__))
        candidate_paths = [
            # Schema file
            os.path.join(script_dir, '..', 'database', 'database_schema.sql'),
            os.path.join(script_dir, '..', 'database_schema.sql'),
            os.path.join(script_dir, 'database_schema.sql'),
        ]
        
        schema_sql = None
        schema_path = None
        
        for path in candidate_paths:
            if os.path.exists(path):
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        schema_sql = f.read()
                    schema_path = path
                    print(f"Found schema file: {path}")
                    break
                except Exception as e:
                    print(f"Failed to read schema from {path}: {e}")
                    continue
        
        if not schema_sql:
            print("No schema file found in any of these locations:")
            for path in candidate_paths:
                print(f"  - {path}")
            return False
        
        # Execute schema creation
        print(f"Creating database schema from {schema_path}...")
        cursor.executescript(schema_sql)
        
        # Enable WAL mode for better performance
        print("Enabling WAL mode for better performance...")
        cursor.execute("PRAGMA journal_mode=WAL")
        
        # Set synchronous mode for better performance
        cursor.execute("PRAGMA synchronous=NORMAL")
        
        # Set cache size for better performance
        cursor.execute("PRAGMA cache_size=10000")
        
        # Set temp store to memory for better performance
        cursor.execute("PRAGMA temp_store=MEMORY")
        
        # Enable foreign keys
        cursor.execute("PRAGMA foreign_keys=ON")
        
        # Create initial statistics
        print("Creating initial statistics...")
        create_initial_statistics(cursor)
        
        conn.commit()
        print("Database initialized successfully!")
        print("Features enabled:")
        print("- Optimized schema for web UI performance")
        print("- Comprehensive indexing for fast queries")
        print("- Discovery tracking and caching")
        print("- Statistics for dashboards")
        print("- WAL mode for concurrent access")
        return True
        
    except Exception as e:
        print(f"Database initialization failed: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()

def create_initial_statistics(cursor):
    """Create initial statistics entries."""
    initial_stats = [
        ('tracks', 'total_count', 0),
        ('playlists', 'total_count', 0),
        ('analysis', 'successful_count', 0),
        ('analysis', 'failed_count', 0),
        ('discovery', 'directories_scanned', 0),
        ('discovery', 'files_found', 0),
        ('cache', 'total_entries', 0),
    ]
    
    for category, metric_name, initial_value in initial_stats:
        cursor.execute("""
            INSERT INTO statistics (category, metric_name, metric_value)
            VALUES (?, ?, ?)
        """, (category, metric_name, initial_value))

def main():
    """Main initialization function."""
    if len(sys.argv) != 2:
        print("Usage: python init_database.py <database_path>")
        print("Example: python init_database.py cache/playlista.db")
        sys.exit(1)
    
    db_path = sys.argv[1]
    
    if init_database(db_path):
        print("Database initialization completed successfully!")
        print("\nNext steps:")
        print("1. Run your analysis to populate the database")
        print("2. Access web UI for data visualization")
        print("3. Use caching for improved performance")
        sys.exit(0)
    else:
        print("Database initialization failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 