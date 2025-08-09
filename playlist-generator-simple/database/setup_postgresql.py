#!/usr/bin/env python3
"""
PostgreSQL Database Setup Script for Playlist Generator.

This script sets up the PostgreSQL database for the playlist generator,
including creating the database, user, and installing required extensions.
"""

import os
import sys
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.logging_setup import get_logger, log_universal
from src.core.config_loader import config_loader

logger = get_logger('playlista.db_setup')


def setup_database():
    """Set up PostgreSQL database for playlist generator."""
    
    config = config_loader.get_config()
    
    # Database configuration
    admin_config = {
        'host': config.get('POSTGRES_HOST', 'localhost'),
        'port': config.get('POSTGRES_PORT', 5432),
        'user': 'postgres',  # Admin user
        'password': os.getenv('POSTGRES_ADMIN_PASSWORD', 'postgres')
    }
    
    app_config = {
        'database': config.get('POSTGRES_DB', 'playlista'),
        'user': config.get('POSTGRES_USER', 'playlista'),
        'password': config.get('POSTGRES_PASSWORD', 'playlista_password')
    }
    
    try:
        log_universal('INFO', 'Setup', 'Starting PostgreSQL database setup...')
        
        # Connect as admin to create database and user
        log_universal('INFO', 'Setup', f'Connecting to PostgreSQL at {admin_config["host"]}:{admin_config["port"]}')
        admin_conn = psycopg2.connect(**admin_config)
        admin_conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        admin_cursor = admin_conn.cursor()
        
        # Create database user
        log_universal('INFO', 'Setup', f'Creating user: {app_config["user"]}')
        try:
            admin_cursor.execute(f"""
                CREATE USER {app_config['user']} WITH PASSWORD '{app_config['password']}';
            """)
            log_universal('INFO', 'Setup', f'✓ User {app_config["user"]} created')
        except psycopg2.errors.DuplicateObject:
            log_universal('INFO', 'Setup', f'User {app_config["user"]} already exists')
        
        # Create database
        log_universal('INFO', 'Setup', f'Creating database: {app_config["database"]}')
        try:
            admin_cursor.execute(f"""
                CREATE DATABASE {app_config['database']} 
                OWNER {app_config['user']} 
                ENCODING 'UTF8';
            """)
            log_universal('INFO', 'Setup', f'✓ Database {app_config["database"]} created')
        except psycopg2.errors.DuplicateDatabase:
            log_universal('INFO', 'Setup', f'Database {app_config["database"]} already exists')
        
        # Grant privileges
        admin_cursor.execute(f"""
            GRANT ALL PRIVILEGES ON DATABASE {app_config['database']} TO {app_config['user']};
        """)
        
        admin_cursor.close()
        admin_conn.close()
        
        # Connect to application database to set up extensions and schema
        app_conn_config = {**admin_config, **app_config}
        del app_conn_config['password']  # Remove admin password
        app_conn_config['password'] = app_config['password']  # Use app password
        
        log_universal('INFO', 'Setup', f'Connecting to application database: {app_config["database"]}')
        app_conn = psycopg2.connect(**app_conn_config)
        app_conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        app_cursor = app_conn.cursor()
        
        # Install required extensions
        log_universal('INFO', 'Setup', 'Installing PostgreSQL extensions...')
        
        extensions = [
            ('uuid-ossp', 'UUID generation'),
            ('pg_trgm', 'Fuzzy text search'),
            ('vector', 'Vector similarity (pgvector)')
        ]
        
        for ext_name, description in extensions:
            try:
                app_cursor.execute(f'CREATE EXTENSION IF NOT EXISTS "{ext_name}";')
                log_universal('INFO', 'Setup', f'✓ Extension {ext_name} installed ({description})')
            except Exception as e:
                if ext_name == 'vector':
                    log_universal('WARNING', 'Setup', f'pgvector extension not available: {str(e)}')
                    log_universal('WARNING', 'Setup', 'Music similarity features will be limited')
                else:
                    log_universal('ERROR', 'Setup', f'Failed to install {ext_name}: {str(e)}')
                    raise
        
        # Create schema
        log_universal('INFO', 'Setup', 'Creating database schema...')
        schema_file = os.path.join(os.path.dirname(__file__), 'postgresql_schema.sql')
        
        with open(schema_file, 'r') as f:
            schema_sql = f.read()
        
        app_cursor.execute(schema_sql)
        log_universal('INFO', 'Setup', '✓ Database schema created')
        
        app_cursor.close()
        app_conn.close()
        
        log_universal('INFO', 'Setup', '🎉 PostgreSQL database setup completed successfully!')
        log_universal('INFO', 'Setup', f'Database: {app_config["database"]}')
        log_universal('INFO', 'Setup', f'User: {app_config["user"]}')
        log_universal('INFO', 'Setup', f'Host: {admin_config["host"]}:{admin_config["port"]}')
        
        return True
        
    except Exception as e:
        log_universal('ERROR', 'Setup', f'Database setup failed: {str(e)}')
        return False


def check_database_connection():
    """Test database connection with application credentials."""
    try:
        config = config_loader.get_config()
        
        conn_config = {
            'host': config.get('POSTGRES_HOST', 'localhost'),
            'port': config.get('POSTGRES_PORT', 5432),
            'database': config.get('POSTGRES_DB', 'playlista'),
            'user': config.get('POSTGRES_USER', 'playlista'),
            'password': config.get('POSTGRES_PASSWORD', 'playlista_password'),
        }
        
        log_universal('INFO', 'Setup', 'Testing database connection...')
        conn = psycopg2.connect(**conn_config)
        cursor = conn.cursor()
        
        # Test basic query
        cursor.execute('SELECT version();')
        version = cursor.fetchone()[0]
        log_universal('INFO', 'Setup', f'✓ Connection successful: {version}')
        
        # Check tables
        cursor.execute("""
            SELECT table_name FROM information_schema.tables 
            WHERE table_schema = 'public' ORDER BY table_name;
        """)
        tables = [row[0] for row in cursor.fetchall()]
        log_universal('INFO', 'Setup', f'✓ Tables found: {", ".join(tables)}')
        
        cursor.close()
        conn.close()
        
        return True
        
    except Exception as e:
        log_universal('ERROR', 'Setup', f'Connection test failed: {str(e)}')
        return False


def main():
    """Main setup function."""
    print("🎵 Playlist Generator - PostgreSQL Database Setup")
    print("=" * 50)
    
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        # Test connection only
        success = check_database_connection()
    else:
        # Full setup
        success = setup_database()
        if success:
            success = check_database_connection()
    
    if success:
        print("\n✅ Database setup completed successfully!")
        print("\nNext steps:")
        print("1. Update your .env file with database credentials")
        print("2. Run analysis to populate the database")
        print("3. Start the web UI")
    else:
        print("\n❌ Database setup failed!")
        print("\nTroubleshooting:")
        print("1. Ensure PostgreSQL is running")
        print("2. Check admin credentials")
        print("3. Install pgvector extension if needed")
        sys.exit(1)


if __name__ == '__main__':
    main()
