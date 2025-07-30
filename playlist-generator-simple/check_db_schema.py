#!/usr/bin/env python3
"""
Check database schema to debug the 'no such column: artist' error.
"""

import sqlite3
import os

def check_schema():
    db_path = os.path.join('cache', 'playlista.db')
    
    if not os.path.exists(db_path):
        print(f"❌ Database file not found: {db_path}")
        return
    
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # Check if analysis_results table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='analysis_results'")
            if not cursor.fetchone():
                print("❌ analysis_results table does not exist")
                return
            
            # Get table schema
            cursor.execute("PRAGMA table_info(analysis_results)")
            columns = cursor.fetchall()
            
            print("📋 analysis_results table schema:")
            for col in columns:
                print(f"   {col[1]} ({col[2]}) - {'NOT NULL' if col[3] else 'NULL'}")
            
            # Check if artist column exists
            artist_col = [col for col in columns if col[1] == 'artist']
            if artist_col:
                print("✅ artist column exists")
            else:
                print("❌ artist column missing")
                
    except Exception as e:
        print(f"❌ Error checking schema: {e}")

if __name__ == "__main__":
    check_schema() 