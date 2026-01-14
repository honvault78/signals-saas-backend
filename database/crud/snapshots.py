"""
Portfolio Snapshot CRUD Operations

Handles daily portfolio snapshots for tracking and charting.
"""

import logging
from typing import Optional, List
from uuid import UUID
from datetime import datetime, timedelta

from sqlalchemy import select, delete, func, and_
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import PortfolioSnapshot
from ..schemas import SnapshotCreate

logger = logging.getLogger(__name__)


async def get_snapshot(
    db: AsyncSession,
    snapshot_id: UUID,
) -> Optional[PortfolioSnapshot]:
    """
    Get snapshot by ID.
    
    Args:
        db: Database session
        snapshot_id: Snapshot UUID
        
    Returns:
        Snapshot if found
    """
    result = await db.execute(
        select(PortfolioSnapshot).where(PortfolioSnapshot.id == snapshot_id)
    )
    return result.scalar_one_or_none()


async def get_portfolio_snapshots(
    db: AsyncSession,
    portfolio_id: UUID,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    limit: int = 365,
) -> List[PortfolioSnapshot]:
    """
    Get snapshots for a portfolio.
    
    Args:
        db: Database session
        portfolio_id: Portfolio UUID
        start_date: Filter from this date
        end_date: Filter until this date
        limit: Max snapshots to return
        
    Returns:
        List of snapshots ordered by date
    """
    filters = [PortfolioSnapshot.portfolio_id == portfolio_id]
    
    if start_date:
        filters.append(PortfolioSnapshot.snapshot_date >= start_date)
    
    if end_date:
        filters.append(PortfolioSnapshot.snapshot_date <= end_date)
    
    query = (
        select(PortfolioSnapshot)
        .where(and_(*filters))
        .order_by(PortfolioSnapshot.snapshot_date.asc())
        .limit(limit)
    )
    
    result = await db.execute(query)
    return list(result.scalars().all())


async def get_latest_snapshot(
    db: AsyncSession,
    portfolio_id: UUID,
) -> Optional[PortfolioSnapshot]:
    """
    Get the most recent snapshot for a portfolio.
    
    Args:
        db: Database session
        portfolio_id: Portfolio UUID
        
    Returns:
        Latest snapshot if exists
    """
    query = (
        select(PortfolioSnapshot)
        .where(PortfolioSnapshot.portfolio_id == portfolio_id)
        .order_by(PortfolioSnapshot.snapshot_date.desc())
        .limit(1)
    )
    
    result = await db.execute(query)
    return result.scalar_one_or_none()


async def get_snapshot_by_date(
    db: AsyncSession,
    portfolio_id: UUID,
    snapshot_date: datetime,
) -> Optional[PortfolioSnapshot]:
    """
    Get snapshot for a specific date.
    
    Args:
        db: Database session
        portfolio_id: Portfolio UUID
        snapshot_date: Date to look up
        
    Returns:
        Snapshot if exists for that date
    """
    # Compare just the date part
    date_start = snapshot_date.replace(hour=0, minute=0, second=0, microsecond=0)
    date_end = date_start + timedelta(days=1)
    
    query = (
        select(PortfolioSnapshot)
        .where(
            and_(
                PortfolioSnapshot.portfolio_id == portfolio_id,
                PortfolioSnapshot.snapshot_date >= date_start,
                PortfolioSnapshot.snapshot_date < date_end,
            )
        )
        .limit(1)
    )
    
    result = await db.execute(query)
    return result.scalar_one_or_none()


async def create_snapshot(
    db: AsyncSession,
    snapshot_data: SnapshotCreate,
) -> PortfolioSnapshot:
    """
    Create a new snapshot.
    
    Args:
        db: Database session
        snapshot_data: Snapshot creation data
        
    Returns:
        Created snapshot
    """
    # Check if snapshot already exists for this date
    existing = await get_snapshot_by_date(
        db,
        snapshot_data.portfolio_id,
        snapshot_data.snapshot_date,
    )
    
    if existing:
        # Update existing snapshot instead of creating duplicate
        existing.cumulative_return = snapshot_data.cumulative_return
        existing.daily_return = snapshot_data.daily_return
        existing.portfolio_value = snapshot_data.portfolio_value
        existing.regime = snapshot_data.regime
        existing.signal = snapshot_data.signal
        existing.z_score = snapshot_data.z_score
        existing.rsi = snapshot_data.rsi
        existing.adf_pvalue = snapshot_data.adf_pvalue
        existing.trend_score = snapshot_data.trend_score
        
        await db.flush()
        await db.refresh(existing)
        
        logger.debug(f"Updated existing snapshot for portfolio {snapshot_data.portfolio_id}")
        return existing
    
    # Create new snapshot
    snapshot = PortfolioSnapshot(
        portfolio_id=snapshot_data.portfolio_id,
        snapshot_date=snapshot_data.snapshot_date,
        cumulative_return=snapshot_data.cumulative_return,
        daily_return=snapshot_data.daily_return,
        portfolio_value=snapshot_data.portfolio_value,
        regime=snapshot_data.regime,
        signal=snapshot_data.signal,
        z_score=snapshot_data.z_score,
        rsi=snapshot_data.rsi,
        adf_pvalue=snapshot_data.adf_pvalue,
        trend_score=snapshot_data.trend_score,
    )
    
    db.add(snapshot)
    await db.flush()
    await db.refresh(snapshot)
    
    logger.debug(f"Created snapshot for portfolio {snapshot_data.portfolio_id} @ {snapshot_data.snapshot_date}")
    return snapshot


async def delete_old_snapshots(
    db: AsyncSession,
    portfolio_id: UUID,
    keep_days: int = 365,
) -> int:
    """
    Delete snapshots older than specified days.
    
    Args:
        db: Database session
        portfolio_id: Portfolio UUID
        keep_days: Keep snapshots from the last N days
        
    Returns:
        Number of snapshots deleted
    """
    cutoff = datetime.utcnow() - timedelta(days=keep_days)
    
    result = await db.execute(
        delete(PortfolioSnapshot)
        .where(
            and_(
                PortfolioSnapshot.portfolio_id == portfolio_id,
                PortfolioSnapshot.snapshot_date < cutoff,
            )
        )
    )
    
    deleted_count = result.rowcount
    
    if deleted_count > 0:
        logger.info(f"Deleted {deleted_count} old snapshots for portfolio {portfolio_id}")
    
    return deleted_count


async def get_portfolio_performance(
    db: AsyncSession,
    portfolio_id: UUID,
    days: int = 30,
) -> dict:
    """
    Calculate portfolio performance metrics from snapshots.
    
    Args:
        db: Database session
        portfolio_id: Portfolio UUID
        days: Number of days to analyze
        
    Returns:
        Performance metrics dict
    """
    start_date = datetime.utcnow() - timedelta(days=days)
    
    snapshots = await get_portfolio_snapshots(
        db,
        portfolio_id,
        start_date=start_date,
    )
    
    if not snapshots:
        return {
            "has_data": False,
        }
    
    # Calculate metrics
    first = snapshots[0]
    last = snapshots[-1]
    
    total_return = last.cumulative_return - first.cumulative_return
    
    # Find max drawdown
    peak = first.portfolio_value
    max_drawdown = 0
    
    for snap in snapshots:
        if snap.portfolio_value > peak:
            peak = snap.portfolio_value
        drawdown = (peak - snap.portfolio_value) / peak if peak > 0 else 0
        max_drawdown = max(max_drawdown, drawdown)
    
    # Count signals
    buy_signals = sum(1 for s in snapshots if s.signal == "BUY")
    sell_signals = sum(1 for s in snapshots if s.signal == "SELL")
    
    return {
        "has_data": True,
        "period_days": days,
        "total_return": total_return,
        "latest_value": last.portfolio_value,
        "latest_regime": last.regime,
        "latest_signal": last.signal,
        "max_drawdown": max_drawdown,
        "buy_signals": buy_signals,
        "sell_signals": sell_signals,
        "snapshot_count": len(snapshots),
    }
