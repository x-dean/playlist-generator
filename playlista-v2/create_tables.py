#!/usr/bin/env python3
"""
Create database tables for Playlista v2
"""

import asyncio
import asyncpg
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import create_async_engine

# Database connection details
DATABASE_URL = "postgresql://playlista:playlista@localhost:5432/playlista_v2"
ASYNC_DATABASE_URL = "postgresql+asyncpg://playlista:playlista@localhost:5432/playlista_v2"

async def create_tables():
    """Create all database tables"""
    print("🗄️  Creating database tables for Playlista v2...")
    
    try:
        # Import models to register them with SQLAlchemy
        import sys
        sys.path.append("backend")
        
        from backend.app.database.models import Base
        
        # Create async engine
        engine = create_async_engine(ASYNC_DATABASE_URL, echo=True)
        
        # Create all tables
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        print("✅ Database tables created successfully!")
        
        # Test connection
        async with engine.begin() as conn:
            result = await conn.execute(text("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"))
            tables = [row[0] for row in result.fetchall()]
            
        print(f"📊 Created {len(tables)} tables:")
        for table in sorted(tables):
            print(f"   - {table}")
            
        await engine.dispose()
        
    except Exception as e:
        print(f"❌ Failed to create tables: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = asyncio.run(create_tables())
    exit(0 if success else 1)

