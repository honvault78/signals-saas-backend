"""
Database Connection Management

Provides async database engine, session factory, and dependency injection
for FastAPI endpoints.
"""

import logging
from typing import AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    AsyncEngine,
    create_async_engine,
    async_sessionmaker,
)
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import text

from .config import get_database_settings

logger = logging.getLogger(__name__)


# =============================================================================
# BASE MODEL
# =============================================================================

class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""
    pass


# =============================================================================
# ENGINE & SESSION FACTORY
# =============================================================================

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine:
    """
    Get or create the async database engine.
    Uses connection pooling for efficient resource usage.
    """
    global _engine
    
    if _engine is None:
        settings = get_database_settings()
        
        # Remove sslmode from URL (handled separately for asyncpg)
        db_url = settings.async_database_url
        if "sslmode=" in db_url:
            import re
            db_url = re.sub(r'[?&]sslmode=[^&]*', '', db_url)
            # Clean up any trailing ? or &
            db_url = db_url.rstrip('?&')
        
        _engine = create_async_engine(
            db_url,
            pool_size=settings.pool_size,
            max_overflow=settings.max_overflow,
            pool_timeout=settings.pool_timeout,
            pool_recycle=settings.pool_recycle,
            echo=settings.echo_sql,
            # SSL for asyncpg - use True for Neon
            connect_args={
                "ssl": True,
                # Disable prepared statement cache for Neon pooler compatibility
                "prepared_statement_cache_size": 0,
            },
        )
        
        logger.info("Database engine created")
    
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """
    Get or create the async session factory.
    """
    global _session_factory
    
    if _session_factory is None:
        engine = get_engine()
        
        _session_factory = async_sessionmaker(
            bind=engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autocommit=False,
            autoflush=False,
        )
        
        logger.info("Session factory created")
    
    return _session_factory


# =============================================================================
# DEPENDENCY INJECTION FOR FASTAPI
# =============================================================================

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency that provides a database session.
    
    Usage:
        @app.get("/items")
        async def get_items(db: AsyncSession = Depends(get_db)):
            ...
    
    The session is automatically closed after the request completes.
    """
    session_factory = get_session_factory()
    
    async with session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


@asynccontextmanager
async def get_db_context() -> AsyncGenerator[AsyncSession, None]:
    """
    Context manager for database sessions outside of FastAPI.
    
    Usage:
        async with get_db_context() as db:
            user = await get_user(db, user_id)
    """
    session_factory = get_session_factory()
    
    async with session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# =============================================================================
# HEALTH CHECK & LIFECYCLE
# =============================================================================

async def check_database_connection() -> bool:
    """
    Check if database connection is healthy.
    Returns True if connection works, False otherwise.
    """
    try:
        async with get_db_context() as db:
            await db.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error(f"Database connection check failed: {e}")
        return False


async def init_database() -> None:
    """
    Initialize database connection on application startup.
    Call this in FastAPI lifespan or startup event.
    """
    logger.info("Initializing database connection...")
    
    # Create engine and session factory
    get_engine()
    get_session_factory()
    
    # Verify connection
    if await check_database_connection():
        logger.info("Database connection verified")
    else:
        logger.error("Database connection failed!")
        raise RuntimeError("Could not connect to database")


async def close_database() -> None:
    """
    Close database connections on application shutdown.
    Call this in FastAPI lifespan or shutdown event.
    """
    global _engine, _session_factory
    
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory = None
        logger.info("Database connections closed")
