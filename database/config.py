"""
Database Configuration

Loads database settings from environment variables.
Supports both development and production configurations.
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings
from functools import lru_cache


class DatabaseSettings(BaseSettings):
    """Database configuration loaded from environment."""
    
    # Main database URL
    database_url: str = ""
    
    # Connection pool settings
    pool_size: int = 5
    max_overflow: int = 10
    pool_timeout: int = 30
    pool_recycle: int = 1800  # 30 minutes
    
    # Echo SQL statements (for debugging)
    echo_sql: bool = False
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"
    
    @property
    def async_database_url(self) -> str:
        """
        Convert standard PostgreSQL URL to async version.
        Ensures +asyncpg driver is specified.
        """
        url = self.database_url
        
        if not url:
            raise ValueError("DATABASE_URL environment variable is not set")
        
        # Handle various URL formats
        if url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql://", 1)
        
        if "postgresql://" in url and "+asyncpg" not in url:
            url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
        
        # Remove channel_binding parameter (not supported by asyncpg)
        if "channel_binding" in url:
            import re
            url = re.sub(r'[&?]channel_binding=[^&]*', '', url)
        
        return url
    
    @property
    def sync_database_url(self) -> str:
        """
        Get synchronous database URL for migrations.
        Uses psycopg2 driver.
        """
        url = self.database_url
        
        if not url:
            raise ValueError("DATABASE_URL environment variable is not set")
        
        if url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql://", 1)
        
        # Remove async driver if present
        url = url.replace("postgresql+asyncpg://", "postgresql://")
        
        # Remove channel_binding parameter
        if "channel_binding" in url:
            import re
            url = re.sub(r'[&?]channel_binding=[^&]*', '', url)
        
        return url


@lru_cache()
def get_database_settings() -> DatabaseSettings:
    """
    Get cached database settings.
    Uses lru_cache to avoid reloading on every call.
    """
    return DatabaseSettings()


# Convenience function
def get_database_url() -> str:
    """Get the async database URL."""
    return get_database_settings().async_database_url
