"""
Async database connection management for Playlista v2
"""

import asyncio
from typing import AsyncGenerator, Optional

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool
import redis.asyncio as aioredis

from ..core.config import get_settings
from ..core.logging import get_logger

logger = get_logger("database")
settings = get_settings()

# Global database engine and session maker
engine: Optional[object] = None
async_session_maker: Optional[async_sessionmaker] = None
redis_client: Optional[aioredis.Redis] = None


async def init_db() -> None:
    """Initialize database connections"""
    global engine, async_session_maker, redis_client
    
    logger.info("Initializing database connections...")
    
    # PostgreSQL connection
    engine = create_async_engine(
        settings.database_url,
        echo=settings.debug,
        pool_size=settings.database_pool_size,
        max_overflow=settings.database_max_overflow,
        pool_pre_ping=True,
        poolclass=NullPool if settings.debug else None,  # Use NullPool in debug mode
    )
    
    async_session_maker = async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    
    # Redis connection
    redis_client = aioredis.from_url(
        settings.redis_url,
        max_connections=settings.redis_max_connections,
        retry_on_timeout=True,
        decode_responses=True,
    )
    
    # Test connections
    try:
        # Test PostgreSQL
        async with async_session_maker() as session:
            await session.execute("SELECT 1")
        logger.info("PostgreSQL connection established")
        
        # Test Redis
        await redis_client.ping()
        logger.info("Redis connection established")
        
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        raise


async def close_db() -> None:
    """Close database connections"""
    global engine, redis_client
    
    logger.info("Closing database connections...")
    
    if engine:
        await engine.dispose()
        logger.info("PostgreSQL connection closed")
    
    if redis_client:
        await redis_client.close()
        logger.info("Redis connection closed")


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Get async database session"""
    if not async_session_maker:
        raise RuntimeError("Database not initialized")
    
    async with async_session_maker() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def get_redis() -> aioredis.Redis:
    """Get Redis client"""
    if not redis_client:
        raise RuntimeError("Redis not initialized")
    return redis_client


class DatabaseManager:
    """Database operations manager with caching"""
    
    def __init__(self):
        self.logger = get_logger("database.manager")
    
    async def execute_query(self, query: str, params: dict = None) -> list:
        """Execute a raw SQL query with parameters"""
        async with async_session_maker() as session:
            try:
                result = await session.execute(query, params or {})
                await session.commit()
                return result.fetchall()
            except Exception as e:
                await session.rollback()
                self.logger.error(f"Query execution failed: {e}")
                raise
    
    async def cache_set(
        self, 
        key: str, 
        value: str, 
        ttl: int = None
    ) -> bool:
        """Set value in Redis cache"""
        try:
            redis = await get_redis()
            if ttl:
                return await redis.setex(key, ttl, value)
            else:
                return await redis.set(key, value)
        except Exception as e:
            self.logger.error(f"Cache set failed for key {key}: {e}")
            return False
    
    async def cache_get(self, key: str) -> Optional[str]:
        """Get value from Redis cache"""
        try:
            redis = await get_redis()
            return await redis.get(key)
        except Exception as e:
            self.logger.error(f"Cache get failed for key {key}: {e}")
            return None
    
    async def cache_delete(self, key: str) -> bool:
        """Delete value from Redis cache"""
        try:
            redis = await get_redis()
            return bool(await redis.delete(key))
        except Exception as e:
            self.logger.error(f"Cache delete failed for key {key}: {e}")
            return False
    
    async def cache_exists(self, key: str) -> bool:
        """Check if key exists in Redis cache"""
        try:
            redis = await get_redis()
            return bool(await redis.exists(key))
        except Exception as e:
            self.logger.error(f"Cache exists check failed for key {key}: {e}")
            return False


# Global database manager instance
db_manager = DatabaseManager()
