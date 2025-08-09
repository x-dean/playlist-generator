#!/usr/bin/env python3
"""Simple database setup for Playlista v2"""

import asyncio
import sys
sys.path.append('/app')

async def setup_database():
    """Simple database setup"""
    try:
        from app.database.connection import init_db
        from app.database.models import Base
        from sqlalchemy import create_engine, text
        from app.core.config import get_settings
        
        settings = get_settings()
        print(f"🔗 Connecting to: {settings.database_url}")
        
        # Initialize database connection
        await init_db()
        print("✅ Database connection established")
        
        # Create sync engine for table creation
        sync_url = settings.database_url.replace('+asyncpg', '')
        sync_engine = create_engine(sync_url)
        
        # Create all tables
        Base.metadata.create_all(sync_engine)
        print("✅ Database tables created")
        
        # Check tables
        with sync_engine.connect() as conn:
            result = conn.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
                ORDER BY table_name;
            """))
            tables = [row[0] for row in result.fetchall()]
        
        print("\n📊 Created Tables:")
        for table in tables:
            print(f"   ✅ {table}")
        
        sync_engine.dispose()
        
        return True
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_api_endpoints():
    """Test API endpoints"""
    try:
        import requests
        
        print("\n🔗 Testing API Endpoints:")
        
        # Test health endpoint
        try:
            response = requests.get("http://localhost:8000/api/health", timeout=5)
            if response.status_code == 200:
                print("   ✅ Health endpoint: WORKING")
            else:
                print(f"   ❌ Health endpoint: {response.status_code}")
        except Exception as e:
            print(f"   ❌ Health endpoint: {e}")
        
        # Test tracks endpoint
        try:
            response = requests.get("http://localhost:8000/api/library/tracks", timeout=5)
            if response.status_code == 200:
                data = response.json()
                print(f"   ✅ Tracks endpoint: {len(data.get('tracks', []))} tracks")
            else:
                print(f"   ❌ Tracks endpoint: {response.status_code}")
        except Exception as e:
            print(f"   ❌ Tracks endpoint: {e}")
            
    except Exception as e:
        print(f"❌ API test failed: {e}")

async def main():
    print("🎵 SIMPLE DATABASE SETUP")
    print("=" * 40)
    
    success = await setup_database()
    
    if success:
        await test_api_endpoints()
        
        print("\n🎯 Setup Complete!")
        print("\n📡 Access Points:")
        print("   🔗 Backend API: http://localhost:8000")
        print("   📊 API Documentation: http://localhost:8000/docs")
        print("   🌐 Web UI: http://localhost:3000")
        print("   📋 Health Check: http://localhost:8000/api/health")

if __name__ == "__main__":
    asyncio.run(main())
