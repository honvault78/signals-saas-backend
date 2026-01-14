"""
Alert CRUD Operations

Handles alert creation, retrieval, and management.
"""

import logging
from typing import Optional, List
from uuid import UUID
from datetime import datetime

from sqlalchemy import select, update, delete, func, and_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ..models import Alert, AlertType, AlertSeverity, Portfolio
from ..schemas import AlertCreate

logger = logging.getLogger(__name__)


async def get_alert(
    db: AsyncSession,
    alert_id: UUID,
    user_id: Optional[UUID] = None,
) -> Optional[Alert]:
    """
    Get alert by ID.
    
    Args:
        db: Database session
        alert_id: Alert UUID
        user_id: If provided, verify ownership
        
    Returns:
        Alert if found
    """
    query = select(Alert).where(Alert.id == alert_id)
    
    if user_id:
        query = query.where(Alert.user_id == user_id)
    
    result = await db.execute(query)
    return result.scalar_one_or_none()


async def get_user_alerts(
    db: AsyncSession,
    user_id: UUID,
    unread_only: bool = False,
    alert_types: Optional[List[AlertType]] = None,
    portfolio_id: Optional[UUID] = None,
    limit: int = 50,
    offset: int = 0,
) -> tuple[List[Alert], int, int]:
    """
    Get alerts for a user.
    
    Args:
        db: Database session
        user_id: User UUID
        unread_only: Only return unread alerts
        alert_types: Filter by alert types
        portfolio_id: Filter by portfolio
        limit: Max alerts to return
        offset: Pagination offset
        
    Returns:
        Tuple of (alerts, total_count, unread_count)
    """
    # Base filters
    filters = [Alert.user_id == user_id]
    
    if unread_only:
        filters.append(Alert.is_read == False)
    
    if alert_types:
        filters.append(Alert.alert_type.in_(alert_types))
    
    if portfolio_id:
        filters.append(Alert.portfolio_id == portfolio_id)
    
    # Count total matching
    count_query = select(func.count(Alert.id)).where(and_(*filters))
    total = await db.execute(count_query)
    total_count = total.scalar() or 0
    
    # Count unread (all for this user)
    unread_query = select(func.count(Alert.id)).where(
        and_(
            Alert.user_id == user_id,
            Alert.is_read == False,
        )
    )
    unread = await db.execute(unread_query)
    unread_count = unread.scalar() or 0
    
    # Fetch alerts with portfolio info
    query = (
        select(Alert)
        .options(selectinload(Alert.portfolio))
        .where(and_(*filters))
        .order_by(Alert.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    
    result = await db.execute(query)
    alerts = list(result.scalars().all())
    
    return alerts, total_count, unread_count


async def get_unread_count(
    db: AsyncSession,
    user_id: UUID,
) -> int:
    """
    Get count of unread alerts for a user.
    
    Args:
        db: Database session
        user_id: User UUID
        
    Returns:
        Number of unread alerts
    """
    query = select(func.count(Alert.id)).where(
        and_(
            Alert.user_id == user_id,
            Alert.is_read == False,
        )
    )
    
    result = await db.execute(query)
    return result.scalar() or 0


async def create_alert(
    db: AsyncSession,
    user_id: UUID,
    alert_data: AlertCreate,
) -> Alert:
    """
    Create a new alert.
    
    Args:
        db: Database session
        user_id: User UUID
        alert_data: Alert creation data
        
    Returns:
        Created alert
    """
    alert = Alert(
        user_id=user_id,
        portfolio_id=alert_data.portfolio_id,
        alert_type=alert_data.alert_type,
        severity=alert_data.severity,
        signal_date=alert_data.signal_date,
        message=alert_data.message,
        portfolio_value=alert_data.portfolio_value,
        regime=alert_data.regime,
        z_score=alert_data.z_score,
        rsi=alert_data.rsi,
    )
    
    db.add(alert)
    await db.flush()
    await db.refresh(alert)
    
    logger.info(f"Created alert: {alert.alert_type.value} for portfolio {alert.portfolio_id}")
    return alert


async def create_signal_alert(
    db: AsyncSession,
    user_id: UUID,
    portfolio_id: UUID,
    signal: str,  # "BUY" or "SELL"
    signal_date: datetime,
    portfolio_value: float,
    regime: str,
    z_score: float,
    rsi: float,
) -> Alert:
    """
    Create a BUY or SELL signal alert.
    
    Convenience function for creating signal alerts with proper formatting.
    
    Args:
        db: Database session
        user_id: User UUID
        portfolio_id: Portfolio UUID
        signal: "BUY" or "SELL"
        signal_date: Date of the signal
        portfolio_value: Portfolio value at signal time
        regime: Current regime
        z_score: Current z-score
        rsi: Current RSI
        
    Returns:
        Created alert
    """
    alert_type = AlertType.BUY if signal.upper() == "BUY" else AlertType.SELL
    severity = AlertSeverity.CRITICAL  # Signals are important
    
    # Format message
    if signal.upper() == "BUY":
        message = f"🟢 BUY signal generated. Regime: {regime}, Z-Score: {z_score:.2f}, RSI: {rsi:.1f}"
    else:
        message = f"🔴 SELL signal generated. Regime: {regime}, Z-Score: {z_score:.2f}, RSI: {rsi:.1f}"
    
    alert_data = AlertCreate(
        portfolio_id=portfolio_id,
        alert_type=alert_type,
        severity=severity,
        signal_date=signal_date,
        message=message,
        portfolio_value=portfolio_value,
        regime=regime,
        z_score=z_score,
        rsi=rsi,
    )
    
    return await create_alert(db, user_id, alert_data)


async def mark_alert_read(
    db: AsyncSession,
    alert_id: UUID,
    user_id: UUID,
) -> Optional[Alert]:
    """
    Mark an alert as read.
    
    Args:
        db: Database session
        alert_id: Alert UUID
        user_id: User UUID (for verification)
        
    Returns:
        Updated alert if found
    """
    alert = await get_alert(db, alert_id, user_id)
    if not alert:
        return None
    
    alert.is_read = True
    alert.read_at = datetime.utcnow()
    await db.flush()
    await db.refresh(alert)
    
    return alert


async def mark_all_alerts_read(
    db: AsyncSession,
    user_id: UUID,
    portfolio_id: Optional[UUID] = None,
) -> int:
    """
    Mark all alerts as read for a user.
    
    Args:
        db: Database session
        user_id: User UUID
        portfolio_id: If provided, only mark alerts for this portfolio
        
    Returns:
        Number of alerts updated
    """
    filters = [
        Alert.user_id == user_id,
        Alert.is_read == False,
    ]
    
    if portfolio_id:
        filters.append(Alert.portfolio_id == portfolio_id)
    
    result = await db.execute(
        update(Alert)
        .where(and_(*filters))
        .values(is_read=True, read_at=datetime.utcnow())
    )
    
    updated_count = result.rowcount
    
    if updated_count > 0:
        logger.info(f"Marked {updated_count} alerts as read for user {user_id}")
    
    return updated_count


async def delete_alert(
    db: AsyncSession,
    alert_id: UUID,
    user_id: UUID,
) -> bool:
    """
    Delete an alert.
    
    Args:
        db: Database session
        alert_id: Alert UUID
        user_id: User UUID (for verification)
        
    Returns:
        True if deleted, False if not found
    """
    alert = await get_alert(db, alert_id, user_id)
    if not alert:
        return False
    
    await db.delete(alert)
    await db.flush()
    
    logger.info(f"Deleted alert: {alert_id}")
    return True


async def delete_read_alerts(
    db: AsyncSession,
    user_id: UUID,
) -> int:
    """
    Delete all read alerts for a user.
    
    Args:
        db: Database session
        user_id: User UUID
        
    Returns:
        Number of alerts deleted
    """
    result = await db.execute(
        delete(Alert)
        .where(
            and_(
                Alert.user_id == user_id,
                Alert.is_read == True,
            )
        )
    )
    
    deleted_count = result.rowcount
    
    if deleted_count > 0:
        logger.info(f"Deleted {deleted_count} read alerts for user {user_id}")
    
    return deleted_count


async def delete_old_alerts(
    db: AsyncSession,
    user_id: UUID,
    keep_days: int = 90,
) -> int:
    """
    Delete alerts older than specified days.
    
    Args:
        db: Database session
        user_id: User UUID
        keep_days: Keep alerts from the last N days
        
    Returns:
        Number of alerts deleted
    """
    from datetime import timedelta
    
    cutoff = datetime.utcnow() - timedelta(days=keep_days)
    
    result = await db.execute(
        delete(Alert)
        .where(
            and_(
                Alert.user_id == user_id,
                Alert.created_at < cutoff,
            )
        )
    )
    
    deleted_count = result.rowcount
    
    if deleted_count > 0:
        logger.info(f"Deleted {deleted_count} old alerts for user {user_id}")
    
    return deleted_count
