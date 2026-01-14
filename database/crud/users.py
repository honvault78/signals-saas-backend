"""
User CRUD Operations

Handles user creation, retrieval, and Clerk synchronization.
"""

import logging
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import User, UserPlan
from ..schemas import UserCreate, UserUpdate

logger = logging.getLogger(__name__)


# Usage limits per plan
USAGE_LIMITS = {
    UserPlan.FREE: 10,
    UserPlan.PRO: 100,
    UserPlan.ENTERPRISE: 1000,
}


async def get_user_by_clerk_id(
    db: AsyncSession,
    clerk_id: str,
) -> Optional[User]:
    """
    Get user by Clerk ID.
    
    Args:
        db: Database session
        clerk_id: Clerk user ID
        
    Returns:
        User if found, None otherwise
    """
    result = await db.execute(
        select(User).where(User.clerk_id == clerk_id)
    )
    return result.scalar_one_or_none()


async def get_user_by_id(
    db: AsyncSession,
    user_id: UUID,
) -> Optional[User]:
    """
    Get user by internal UUID.
    
    Args:
        db: Database session
        user_id: Internal user UUID
        
    Returns:
        User if found, None otherwise
    """
    result = await db.execute(
        select(User).where(User.id == user_id)
    )
    return result.scalar_one_or_none()


async def create_user(
    db: AsyncSession,
    user_data: UserCreate,
) -> User:
    """
    Create a new user.
    
    Args:
        db: Database session
        user_data: User creation data
        
    Returns:
        Created user
    """
    user = User(
        clerk_id=user_data.clerk_id,
        email=user_data.email,
        first_name=user_data.first_name,
        last_name=user_data.last_name,
        plan=UserPlan.FREE,
        usage_count=0,
        usage_reset_at=_get_next_reset_date(),
    )
    
    db.add(user)
    await db.flush()
    await db.refresh(user)
    
    logger.info(f"Created new user: {user.email} (clerk_id: {user.clerk_id})")
    return user


async def update_user(
    db: AsyncSession,
    user_id: UUID,
    user_data: UserUpdate,
) -> Optional[User]:
    """
    Update user fields.
    
    Args:
        db: Database session
        user_id: User UUID
        user_data: Fields to update
        
    Returns:
        Updated user if found
    """
    user = await get_user_by_id(db, user_id)
    if not user:
        return None
    
    update_data = user_data.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(user, field, value)
    
    await db.flush()
    await db.refresh(user)
    
    logger.info(f"Updated user: {user.email}")
    return user


async def sync_clerk_user(
    db: AsyncSession,
    clerk_id: str,
    email: str,
    first_name: Optional[str] = None,
    last_name: Optional[str] = None,
) -> User:
    """
    Sync user from Clerk - create if doesn't exist, update if does.
    
    This should be called on every authenticated request to ensure
    the user exists in our database and info is up-to-date.
    
    Args:
        db: Database session
        clerk_id: Clerk user ID
        email: User email
        first_name: First name
        last_name: Last name
        
    Returns:
        User (created or existing)
    """
    user = await get_user_by_clerk_id(db, clerk_id)
    
    if user is None:
        # Create new user
        user_data = UserCreate(
            clerk_id=clerk_id,
            email=email,
            first_name=first_name,
            last_name=last_name,
        )
        user = await create_user(db, user_data)
    else:
        # Update existing user info if changed
        changed = False
        if user.email != email:
            user.email = email
            changed = True
        if user.first_name != first_name:
            user.first_name = first_name
            changed = True
        if user.last_name != last_name:
            user.last_name = last_name
            changed = True
        
        if changed:
            await db.flush()
            logger.debug(f"Updated user info: {user.email}")
    
    # Update last active timestamp
    user.last_active_at = datetime.now(timezone.utc)
    await db.flush()
    
    return user


async def increment_usage(
    db: AsyncSession,
    user_id: UUID,
) -> tuple[int, int]:
    """
    Increment user's usage count.
    Resets count if past reset date.
    
    Args:
        db: Database session
        user_id: User UUID
        
    Returns:
        Tuple of (current_usage, limit)
        
    Raises:
        ValueError: If user not found
    """
    user = await get_user_by_id(db, user_id)
    if not user:
        raise ValueError(f"User not found: {user_id}")
    
    # Check if we need to reset
    now = datetime.now(timezone.utc)
    if user.usage_reset_at <= now:
        user.usage_count = 0
        user.usage_reset_at = _get_next_reset_date()
        logger.info(f"Reset usage count for user: {user.email}")
    
    # Increment usage
    user.usage_count += 1
    await db.flush()
    
    limit = USAGE_LIMITS.get(user.plan, 10)
    logger.debug(f"User {user.email} usage: {user.usage_count}/{limit}")
    
    return user.usage_count, limit


async def get_usage_limits(
    db: AsyncSession,
    user_id: UUID,
) -> dict:
    """
    Get user's current usage and limits.
    
    Args:
        db: Database session
        user_id: User UUID
        
    Returns:
        Dict with usage info
    """
    user = await get_user_by_id(db, user_id)
    if not user:
        raise ValueError(f"User not found: {user_id}")
    
    limit = USAGE_LIMITS.get(user.plan, 10)
    
    # Check if we need to reset (but don't actually reset here)
    now = datetime.now(timezone.utc)
    current_usage = 0 if user.usage_reset_at <= now else user.usage_count
    
    return {
        "usage_count": current_usage,
        "usage_limit": limit,
        "usage_reset_at": user.usage_reset_at,
        "plan": user.plan,
    }


def _get_next_reset_date() -> datetime:
    """Get the first day of next month (timezone-aware)."""
    now = datetime.now(timezone.utc)
    if now.month == 12:
        return datetime(now.year + 1, 1, 1, tzinfo=timezone.utc)
    else:
        return datetime(now.year, now.month + 1, 1, tzinfo=timezone.utc)
