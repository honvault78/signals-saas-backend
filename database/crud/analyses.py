"""
Analysis CRUD Operations

Handles analysis history storage and retrieval.
"""

import logging
from typing import Optional, List
from uuid import UUID
from datetime import datetime

from sqlalchemy import select, delete, func, and_
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import Analysis
from ..schemas import AnalysisCreate

logger = logging.getLogger(__name__)


async def get_analysis(
    db: AsyncSession,
    analysis_id: UUID,
    user_id: Optional[UUID] = None,
    include_html: bool = False,
) -> Optional[Analysis]:
    """
    Get analysis by ID.
    
    Args:
        db: Database session
        analysis_id: Analysis UUID
        user_id: If provided, verify ownership
        include_html: Whether to include full HTML report
        
    Returns:
        Analysis if found (and owned by user if user_id provided)
    """
    query = select(Analysis).where(Analysis.id == analysis_id)
    
    if user_id:
        query = query.where(Analysis.user_id == user_id)
    
    result = await db.execute(query)
    analysis = result.scalar_one_or_none()
    
    # Optionally strip HTML to reduce payload
    if analysis and not include_html:
        # We return the analysis but mark that HTML exists
        pass  # Let the schema handle has_html_report
    
    return analysis


async def get_user_analyses(
    db: AsyncSession,
    user_id: UUID,
    portfolio_id: Optional[UUID] = None,
    limit: int = 20,
    offset: int = 0,
) -> tuple[List[Analysis], int]:
    """
    Get analyses for a user.
    
    Args:
        db: Database session
        user_id: User UUID
        portfolio_id: Filter by portfolio (optional)
        limit: Max analyses to return
        offset: Pagination offset
        
    Returns:
        Tuple of (analyses, total_count)
    """
    # Base filter
    filters = [Analysis.user_id == user_id]
    
    if portfolio_id:
        filters.append(Analysis.portfolio_id == portfolio_id)
    
    # Count total
    count_query = select(func.count(Analysis.id)).where(and_(*filters))
    total = await db.execute(count_query)
    total_count = total.scalar() or 0
    
    # Fetch analyses (without HTML to reduce payload)
    query = (
        select(Analysis)
        .where(and_(*filters))
        .order_by(Analysis.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    
    result = await db.execute(query)
    analyses = list(result.scalars().all())
    
    return analyses, total_count


async def get_portfolio_analyses(
    db: AsyncSession,
    portfolio_id: UUID,
    user_id: UUID,
    limit: int = 10,
) -> List[Analysis]:
    """
    Get analyses for a specific portfolio.
    
    Args:
        db: Database session
        portfolio_id: Portfolio UUID
        user_id: User UUID (for verification)
        limit: Max analyses to return
        
    Returns:
        List of analyses
    """
    query = (
        select(Analysis)
        .where(
            and_(
                Analysis.portfolio_id == portfolio_id,
                Analysis.user_id == user_id,
            )
        )
        .order_by(Analysis.created_at.desc())
        .limit(limit)
    )
    
    result = await db.execute(query)
    return list(result.scalars().all())


async def create_analysis(
    db: AsyncSession,
    user_id: UUID,
    analysis_data: AnalysisCreate,
) -> Analysis:
    """
    Create a new analysis record.
    
    Args:
        db: Database session
        user_id: User UUID
        analysis_data: Analysis creation data
        
    Returns:
        Created analysis
    """
    # Convert positions to dict format
    positions = [p.model_dump() for p in analysis_data.positions]
    
    analysis = Analysis(
        user_id=user_id,
        portfolio_id=analysis_data.portfolio_id,
        portfolio_name=analysis_data.portfolio_name,
        positions=positions,
        analysis_period_days=analysis_data.analysis_period_days,
        result_summary=analysis_data.result_summary,
        html_report=analysis_data.html_report,
        ai_memo=analysis_data.ai_memo,
        duration_ms=analysis_data.duration_ms,
    )
    
    db.add(analysis)
    await db.flush()
    await db.refresh(analysis)
    
    logger.info(f"Created analysis: {analysis.portfolio_name} for user {user_id}")
    return analysis


async def update_analysis_memo(
    db: AsyncSession,
    analysis_id: UUID,
    user_id: UUID,
    ai_memo: str,
) -> Optional[Analysis]:
    """
    Update an analysis with AI memo (generated on-demand).
    
    Args:
        db: Database session
        analysis_id: Analysis UUID
        user_id: User UUID (for verification)
        ai_memo: Generated AI memo
        
    Returns:
        Updated analysis if found
    """
    analysis = await get_analysis(db, analysis_id, user_id)
    if not analysis:
        return None
    
    analysis.ai_memo = ai_memo
    await db.flush()
    await db.refresh(analysis)
    
    logger.info(f"Updated analysis memo: {analysis_id}")
    return analysis


async def delete_analysis(
    db: AsyncSession,
    analysis_id: UUID,
    user_id: UUID,
) -> bool:
    """
    Delete an analysis.
    
    Args:
        db: Database session
        analysis_id: Analysis UUID
        user_id: User UUID (for verification)
        
    Returns:
        True if deleted, False if not found
    """
    analysis = await get_analysis(db, analysis_id, user_id)
    if not analysis:
        return False
    
    await db.delete(analysis)
    await db.flush()
    
    logger.info(f"Deleted analysis: {analysis_id}")
    return True


async def delete_old_analyses(
    db: AsyncSession,
    user_id: UUID,
    keep_count: int = 50,
) -> int:
    """
    Delete old analyses, keeping only the most recent ones.
    
    Args:
        db: Database session
        user_id: User UUID
        keep_count: Number of recent analyses to keep
        
    Returns:
        Number of analyses deleted
    """
    # Get IDs to keep
    keep_query = (
        select(Analysis.id)
        .where(Analysis.user_id == user_id)
        .order_by(Analysis.created_at.desc())
        .limit(keep_count)
    )
    
    keep_result = await db.execute(keep_query)
    keep_ids = [row[0] for row in keep_result.fetchall()]
    
    if not keep_ids:
        return 0
    
    # Delete others
    delete_query = (
        delete(Analysis)
        .where(
            and_(
                Analysis.user_id == user_id,
                Analysis.id.notin_(keep_ids),
            )
        )
    )
    
    result = await db.execute(delete_query)
    deleted_count = result.rowcount
    
    if deleted_count > 0:
        logger.info(f"Deleted {deleted_count} old analyses for user {user_id}")
    
    return deleted_count
