#!/usr/bin/env python3
"""
Database setup script for Playlista v2
Creates tables and populates with sample data
"""

import asyncio
import sys
import os
sys.path.append('/app')

from app.database.models import Base, Track, AnalysisJob, Playlist, PlaylistItem
from app.database.connection import engine, get_db_session
from app.core.logging import get_logger

logger = get_logger("database_setup")

async def create_tables():
    """Create all database tables"""
    try:
        async with engine.begin() as conn:
            # Create all tables
            await conn.run_sync(Base.metadata.create_all)
        
        logger.info("Database tables created successfully")
        print("✅ Database tables created successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to create tables: {e}")
        print(f"❌ Failed to create tables: {e}")
        return False

async def check_tables():
    """Check what tables exist in the database"""
    try:
        from sqlalchemy import text
        
        async with engine.begin() as conn:
            result = await conn.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
                ORDER BY table_name;
            """))
            tables = [row[0] for row in result.fetchall()]
        
        print(f"\n📊 Database Tables:")
        if tables:
            for table in tables:
                print(f"   ✅ {table}")
        else:
            print("   ❌ No tables found")
        
        return tables
    except Exception as e:
        print(f"❌ Failed to check tables: {e}")
        return []

async def check_table_contents():
    """Check contents of existing tables"""
    try:
        async with get_db_session() as session:
            from sqlalchemy import text
            
            tables = ['tracks', 'analysis_jobs', 'playlists', 'playlist_items']
            
            print(f"\n📋 Table Contents:")
            for table in tables:
                try:
                    result = await session.execute(text(f"SELECT COUNT(*) FROM {table}"))
                    count = result.scalar()
                    print(f"   📄 {table}: {count} records")
                except Exception as e:
                    print(f"   ❌ {table}: Table doesn't exist or error: {e}")
                    
    except Exception as e:
        print(f"❌ Failed to check table contents: {e}")

async def add_sample_data():
    """Add some sample data for demonstration"""
    try:
        async with get_db_session() as session:
            # Check if we already have data
            from sqlalchemy import text
            result = await session.execute(text("SELECT COUNT(*) FROM tracks"))
            track_count = result.scalar()
            
            if track_count > 0:
                print(f"📊 Database already has {track_count} tracks")
                return
            
            # Add sample tracks (will be populated by actual analysis)
            sample_tracks = [
                Track(
                    id="sample_1",
                    file_path="/music/sample1.mp3",
                    filename="sample1.mp3",
                    title="Sample Track 1",
                    artist="Sample Artist",
                    album="Sample Album",
                    duration=180.0,
                    file_size=5242880,  # 5MB
                    audio_features={"tempo": 120, "key": "C", "loudness": -10.5}
                ),
                Track(
                    id="sample_2", 
                    file_path="/music/sample2.mp3",
                    filename="sample2.mp3",
                    title="Sample Track 2",
                    artist="Another Artist",
                    album="Another Album", 
                    duration=240.0,
                    file_size=7340032,  # 7MB
                    audio_features={"tempo": 140, "key": "G", "loudness": -8.2}
                )
            ]
            
            for track in sample_tracks:
                session.add(track)
            
            await session.commit()
            print("✅ Sample data added successfully")
            
    except Exception as e:
        print(f"❌ Failed to add sample data: {e}")

async def main():
    """Main setup function"""
    print("🎵 PLAYLISTA V2 DATABASE SETUP")
    print("=" * 50)
    
    # Create tables
    print("\n1️⃣ Creating database tables...")
    success = await create_tables()
    
    if not success:
        print("❌ Database setup failed")
        return
    
    # Check what was created
    print("\n2️⃣ Checking created tables...")
    tables = await check_tables()
    
    # Add sample data
    print("\n3️⃣ Adding sample data...")
    await add_sample_data()
    
    # Check final contents
    print("\n4️⃣ Checking table contents...")
    await check_table_contents()
    
    print("\n🎯 Database setup complete!")
    print("\n📡 Access Points:")
    print("   🔗 Backend API: http://localhost:8000")
    print("   📊 API Docs: http://localhost:8000/docs")
    print("   🌐 Web UI: http://localhost:3000")
    print("   🗄️ Database: localhost:5432 (playlista_v2)")

if __name__ == "__main__":
    asyncio.run(main())
