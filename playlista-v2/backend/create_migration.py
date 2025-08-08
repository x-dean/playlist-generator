#!/usr/bin/env python3
"""
Create initial database migration for Playlista v2
"""

import asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from app.database.models import Base
from app.core.config import get_settings

async def create_tables():
    """Create database tables"""
    settings = get_settings()
    
    engine = create_async_engine(
        settings.database_url,
        echo=True
    )
    
    async with engine.begin() as conn:
        # Drop all tables first (for clean development)
        await conn.run_sync(Base.metadata.drop_all)
        
        # Create all tables
        await conn.run_sync(Base.metadata.create_all)
    
    await engine.dispose()
    print("✓ Database tables created successfully")

if __name__ == "__main__":
    asyncio.run(create_tables())
