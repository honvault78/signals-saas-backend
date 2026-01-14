"""
Portfolio CRUD Operations

Handles portfolio creation, retrieval, update, and deletion.
"""

import logging
from typing import Optional, List
from uuid import UUID

from sqlalchemy import select, update, delete, func, and_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ..models import Portfolio, PortfolioSnapshot
from ..schemas import PortfolioCreate, PortfolioUpdate

logger = logging.getLogger(__name__)


async def get_portfolio(
    db: AsyncSession,
    portfolio_id: UUID,
    user_id: Optional[UUID] = None,
) -> Optional[Portfolio]:
    """
    Get portfolio by ID.
    
    Args:
        db: Database session
        portfolio_id: Portfolio UUID
        user_id: If provided, verify ownership
        
    Returns:
        Portfolio if found (and owned by user if user_id provided)
    """
    query = select(Portfolio).where(Portfolio.id == portfolio_id)
    
    if user_id:
        query = query.where(Portfolio.user_id == user_id)
    
    result = await db.execute(query)
    return result.scalar_one_or_none()


async def get_user_portfolios(
    db: AsyncSession,
    user_id: UUID,
    include_snapshots: bool = False,
    limit: int = 50,
    offset: int = 0,
) -> tuple[List[Portfolio], int]:
    """
    Get all portfolios for a user.
    
    Args:
        db: Database session
        user_id: User UUID
        include_snapshots: Whether to load latest snapshot
        limit: Max portfolios to return
        offset: Pagination offset
        
    Returns:
        Tuple of (portfolios, total_count)
    """
    # Count total
    count_query = select(func.count(Portfolio.id)).where(
        Portfolio.user_id == user_id
    )
    total = await db.execute(count_query)
    total_count = total.scalar() or 0
    
    # Fetch portfolios
    query = (
        select(Portfolio)
        .where(Portfolio.user_id == user_id)
        .order_by(Portfolio.is_default.desc(), Portfolio.updated_at.desc())
        .limit(limit)
        .offset(offset)
    )
    
    result = await db.execute(query)
    portfolios = list(result.scalars().all())
    
    return portfolios, total_count


async def create_portfolio(
    db: AsyncSession,
    user_id: UUID,
    portfolio_data: PortfolioCreate,
) -> Portfolio:
    """
    Create a new portfolio.
    
    Args:
        db: Database session
        user_id: Owner user UUID
        portfolio_data: Portfolio creation data
        
    Returns:
        Created portfolio
    """
    # If this is marked as default, unset other defaults
    if portfolio_data.is_default:
        await _unset_default_portfolios(db, user_id)
    
    # Convert positions to dict format
    positions = [p.model_dump() for p in portfolio_data.positions]
    
    portfolio = Portfolio(
        user_id=user_id,
        name=portfolio_data.name,
        description=portfolio_data.description,
        positions=positions,
        is_default=portfolio_data.is_default,
        is_tracked=portfolio_data.is_tracked,
    )
    
    db.add(portfolio)
    await db.flush()
    await db.refresh(portfolio)
    
    logger.info(f"Created portfolio: {portfolio.name} for user {user_id}")
    return portfolio


async def update_portfolio(
    db: AsyncSession,
    portfolio_id: UUID,
    user_id: UUID,
    portfolio_data: PortfolioUpdate,
) -> Optional[Portfolio]:
    """
    Update a portfolio.
    
    Args:
        db: Database session
        portfolio_id: Portfolio UUID
        user_id: Owner user UUID (for verification)
        portfolio_data: Fields to update
        
    Returns:
        Updated portfolio if found and owned
    """
    portfolio = await get_portfolio(db, portfolio_id, user_id)
    if not portfolio:
        return None
    
    update_data = portfolio_data.model_dump(exclude_unset=True)
    
    # Handle default flag
    if update_data.get("is_default"):
        await _unset_default_portfolios(db, user_id)
    
    # Convert positions if provided
    if "positions" in update_data and update_data["positions"] is not None:
        update_data["positions"] = [p.model_dump() for p in update_data["positions"]]
    
    for field, value in update_data.items():
        setattr(portfolio, field, value)
    
    await db.flush()
    await db.refresh(portfolio)
    
    logger.info(f"Updated portfolio: {portfolio.name}")
    return portfolio


async def delete_portfolio(
    db: AsyncSession,
    portfolio_id: UUID,
    user_id: UUID,
) -> bool:
    """
    Delete a portfolio.
    
    Args:
        db: Database session
        portfolio_id: Portfolio UUID
        user_id: Owner user UUID (for verification)
        
    Returns:
        True if deleted, False if not found
    """
    portfolio = await get_portfolio(db, portfolio_id, user_id)
    if not portfolio:
        return False
    
    await db.delete(portfolio)
    await db.flush()
    
    logger.info(f"Deleted portfolio: {portfolio_id}")
    return True


async def set_default_portfolio(
    db: AsyncSession,
    portfolio_id: UUID,
    user_id: UUID,
) -> Optional[Portfolio]:
    """
    Set a portfolio as the default for a user.
    
    Args:
        db: Database session
        portfolio_id: Portfolio to set as default
        user_id: Owner user UUID
        
    Returns:
        Updated portfolio if found
    """
    portfolio = await get_portfolio(db, portfolio_id, user_id)
    if not portfolio:
        return None
    
    # Unset all other defaults
    await _unset_default_portfolios(db, user_id)
    
    # Set this one as default
    portfolio.is_default = True
    await db.flush()
    await db.refresh(portfolio)
    
    logger.info(f"Set default portfolio: {portfolio.name}")
    return portfolio


async def get_tracked_portfolios(
    db: AsyncSession,
    limit: int = 1000,
) -> List[Portfolio]:
    """
    Get all portfolios that are tracked (for daily updates).
    
    Args:
        db: Database session
        limit: Max portfolios to return
        
    Returns:
        List of tracked portfolios
    """
    query = (
        select(Portfolio)
        .where(Portfolio.is_tracked == True)
        .limit(limit)
    )
    
    result = await db.execute(query)
    return list(result.scalars().all())


async def get_user_default_portfolio(
    db: AsyncSession,
    user_id: UUID,
) -> Optional[Portfolio]:
    """
    Get user's default portfolio.
    
    Args:
        db: Database session
        user_id: User UUID
        
    Returns:
        Default portfolio if set
    """
    query = select(Portfolio).where(
        and_(
            Portfolio.user_id == user_id,
            Portfolio.is_default == True,
        )
    )
    
    result = await db.execute(query)
    return result.scalar_one_or_none()


async def _unset_default_portfolios(
    db: AsyncSession,
    user_id: UUID,
) -> None:
    """Unset all default portfolios for a user."""
    await db.execute(
        update(Portfolio)
        .where(
            and_(
                Portfolio.user_id == user_id,
                Portfolio.is_default == True,
            )
        )
        .values(is_default=False)
    )
