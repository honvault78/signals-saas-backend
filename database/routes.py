"""
API Routes for Persistent Storage (v2)

- Explicit save only (no auto-save)
- Clear delete functions
- User-controlled persistence
"""

import logging
from typing import Optional, List
from uuid import UUID
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel

from database import get_db, crud
from database.schemas import (
    # Users
    UserProfile,
    # Portfolios
    PortfolioCreate,
    PortfolioUpdate,
    Portfolio as PortfolioSchema,
    PortfolioList,
    PositionBase,
    # Analyses
    Analysis as AnalysisSchema,
    AnalysisFull,
    AnalysisList,
    AnalysisCreate,
    # Alerts
    Alert as AlertSchema,
    AlertList,
    AlertUpdate,
    # Snapshots
    Snapshot as SnapshotSchema,
    SnapshotList,
)

# Import your auth module
from engine.auth import ClerkUser, get_current_user

logger = logging.getLogger(__name__)

# Create router
router = APIRouter()


# =============================================================================
# HELPER: Sync Clerk user to database
# =============================================================================

async def get_db_user(
    db: AsyncSession,
    clerk_user: ClerkUser,
):
    """
    Get or create database user from Clerk user.
    Call this at the start of protected endpoints.
    """
    return await crud.sync_clerk_user(
        db,
        clerk_id=clerk_user.user_id,
        email=clerk_user.email or "",
        first_name=clerk_user.first_name,
        last_name=clerk_user.last_name,
    )


# =============================================================================
# USER ROUTES
# =============================================================================

@router.get("/me", response_model=UserProfile)
async def get_current_user_profile(
    clerk_user: ClerkUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get current user's profile and usage information.
    """
    user = await get_db_user(db, clerk_user)
    usage = await crud.get_usage_limits(db, user.id)
    
    return UserProfile(
        id=user.id,
        email=user.email,
        first_name=user.first_name,
        last_name=user.last_name,
        full_name=user.full_name,
        plan=user.plan,
        usage_count=usage["usage_count"],
        usage_limit=usage["usage_limit"],
        usage_reset_at=usage["usage_reset_at"],
        created_at=user.created_at,
    )


# =============================================================================
# PORTFOLIO ROUTES (Explicit Save)
# =============================================================================

@router.get("/portfolios", response_model=PortfolioList)
async def list_portfolios(
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    clerk_user: ClerkUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    List all saved portfolios for the current user.
    """
    user = await get_db_user(db, clerk_user)
    portfolios, total = await crud.get_user_portfolios(
        db, user.id, limit=limit, offset=offset
    )
    
    return PortfolioList(
        portfolios=[PortfolioSchema.model_validate(p) for p in portfolios],
        total=total,
    )


@router.post("/portfolios", response_model=PortfolioSchema)
async def save_portfolio(
    portfolio_data: PortfolioCreate,
    clerk_user: ClerkUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Save a new portfolio (explicit save action).
    User clicks "Save Portfolio" button to trigger this.
    """
    user = await get_db_user(db, clerk_user)
    
    portfolio = await crud.create_portfolio(db, user.id, portfolio_data)
    logger.info(f"User {user.email} saved portfolio: {portfolio.name}")
    
    return PortfolioSchema.model_validate(portfolio)


@router.get("/portfolios/{portfolio_id}", response_model=PortfolioSchema)
async def get_portfolio(
    portfolio_id: UUID,
    clerk_user: ClerkUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get a specific saved portfolio.
    """
    user = await get_db_user(db, clerk_user)
    
    portfolio = await crud.get_portfolio(db, portfolio_id, user.id)
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    
    return PortfolioSchema.model_validate(portfolio)


@router.put("/portfolios/{portfolio_id}", response_model=PortfolioSchema)
async def update_portfolio(
    portfolio_id: UUID,
    portfolio_data: PortfolioUpdate,
    clerk_user: ClerkUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Update a saved portfolio.
    """
    user = await get_db_user(db, clerk_user)
    
    portfolio = await crud.update_portfolio(db, portfolio_id, user.id, portfolio_data)
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    
    logger.info(f"User {user.email} updated portfolio: {portfolio.name}")
    return PortfolioSchema.model_validate(portfolio)


@router.delete("/portfolios/{portfolio_id}")
async def delete_portfolio(
    portfolio_id: UUID,
    clerk_user: ClerkUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Delete a saved portfolio.
    Also deletes associated snapshots and alerts.
    """
    user = await get_db_user(db, clerk_user)
    
    # Get portfolio name for logging
    portfolio = await crud.get_portfolio(db, portfolio_id, user.id)
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    
    portfolio_name = portfolio.name
    deleted = await crud.delete_portfolio(db, portfolio_id, user.id)
    
    logger.info(f"User {user.email} deleted portfolio: {portfolio_name}")
    return {"message": f"Portfolio '{portfolio_name}' deleted", "deleted": True}


@router.post("/portfolios/{portfolio_id}/default", response_model=PortfolioSchema)
async def set_default_portfolio(
    portfolio_id: UUID,
    clerk_user: ClerkUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Set a portfolio as the default (loaded on login).
    """
    user = await get_db_user(db, clerk_user)
    
    portfolio = await crud.set_default_portfolio(db, portfolio_id, user.id)
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    
    return PortfolioSchema.model_validate(portfolio)


@router.post("/portfolios/{portfolio_id}/tracking")
async def toggle_portfolio_tracking(
    portfolio_id: UUID,
    is_tracked: bool = Query(...),
    clerk_user: ClerkUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Enable or disable daily tracking for a portfolio.
    """
    user = await get_db_user(db, clerk_user)
    
    portfolio_data = PortfolioUpdate(is_tracked=is_tracked)
    portfolio = await crud.update_portfolio(db, portfolio_id, user.id, portfolio_data)
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    
    status = "enabled" if is_tracked else "disabled"
    logger.info(f"User {user.email} {status} tracking for portfolio: {portfolio.name}")
    
    return {"message": f"Tracking {status}", "is_tracked": is_tracked}


# =============================================================================
# ANALYSIS ROUTES (Explicit Save)
# =============================================================================

class SaveAnalysisRequest(BaseModel):
    """Request to save an analysis result."""
    portfolio_name: str
    positions: List[PositionBase]
    analysis_period_days: int
    result_summary: dict
    html_report: str
    ai_memo: Optional[str] = None
    duration_ms: Optional[int] = None
    portfolio_id: Optional[UUID] = None  # Link to saved portfolio if applicable


@router.get("/analyses", response_model=AnalysisList)
async def list_analyses(
    portfolio_id: Optional[UUID] = None,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    clerk_user: ClerkUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    List saved analysis history for the current user.
    """
    user = await get_db_user(db, clerk_user)
    
    analyses, total = await crud.get_user_analyses(
        db, user.id,
        portfolio_id=portfolio_id,
        limit=limit,
        offset=offset,
    )
    
    # Convert to schema and add has_html_report flag
    analysis_list = []
    for a in analyses:
        schema = AnalysisSchema.model_validate(a)
        schema.has_html_report = a.html_report is not None
        schema.has_ai_memo = a.ai_memo is not None
        analysis_list.append(schema)
    
    return AnalysisList(
        analyses=analysis_list,
        total=total,
    )


@router.post("/analyses", response_model=AnalysisSchema)
async def save_analysis(
    request: SaveAnalysisRequest,
    clerk_user: ClerkUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Save an analysis to history (explicit save action).
    
    User runs analysis -> views result -> clicks "Save to History" -> triggers this.
    The analysis is NOT auto-saved when running.
    """
    user = await get_db_user(db, clerk_user)
    
    analysis_data = AnalysisCreate(
        portfolio_name=request.portfolio_name,
        positions=request.positions,
        analysis_period_days=request.analysis_period_days,
        result_summary=request.result_summary,
        html_report=request.html_report,
        ai_memo=request.ai_memo,
        duration_ms=request.duration_ms,
        portfolio_id=request.portfolio_id,
    )
    
    analysis = await crud.create_analysis(db, user.id, analysis_data)
    logger.info(f"User {user.email} saved analysis: {analysis.portfolio_name}")
    
    schema = AnalysisSchema.model_validate(analysis)
    schema.has_html_report = True
    schema.has_ai_memo = analysis.ai_memo is not None
    
    return schema


@router.get("/analyses/{analysis_id}", response_model=AnalysisFull)
async def get_analysis(
    analysis_id: UUID,
    clerk_user: ClerkUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get a specific saved analysis with full HTML report.
    """
    user = await get_db_user(db, clerk_user)
    
    analysis = await crud.get_analysis(db, analysis_id, user.id, include_html=True)
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    schema = AnalysisFull.model_validate(analysis)
    schema.has_html_report = analysis.html_report is not None
    schema.has_ai_memo = analysis.ai_memo is not None
    
    return schema


@router.delete("/analyses/{analysis_id}")
async def delete_analysis(
    analysis_id: UUID,
    clerk_user: ClerkUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Delete a saved analysis from history.
    """
    user = await get_db_user(db, clerk_user)
    
    # Get analysis name for logging
    analysis = await crud.get_analysis(db, analysis_id, user.id)
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    analysis_name = analysis.portfolio_name
    deleted = await crud.delete_analysis(db, analysis_id, user.id)
    
    logger.info(f"User {user.email} deleted analysis: {analysis_name}")
    return {"message": f"Analysis '{analysis_name}' deleted", "deleted": True}


@router.delete("/analyses")
async def delete_all_analyses(
    confirm: bool = Query(..., description="Must be true to confirm deletion"),
    clerk_user: ClerkUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Delete all saved analyses for the current user.
    Requires confirm=true query parameter.
    """
    if not confirm:
        raise HTTPException(status_code=400, detail="Must confirm with confirm=true")
    
    user = await get_db_user(db, clerk_user)
    
    # Delete all (keep 0)
    deleted_count = await crud.delete_old_analyses(db, user.id, keep_count=0)
    
    logger.info(f"User {user.email} deleted all analyses ({deleted_count} total)")
    return {"message": f"Deleted {deleted_count} analyses", "deleted_count": deleted_count}


# =============================================================================
# ALERT ROUTES
# =============================================================================

@router.get("/alerts", response_model=AlertList)
async def list_alerts(
    unread_only: bool = Query(False),
    portfolio_id: Optional[UUID] = None,
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    clerk_user: ClerkUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    List alerts for the current user.
    """
    user = await get_db_user(db, clerk_user)
    
    alerts, total, unread_count = await crud.get_user_alerts(
        db, user.id,
        unread_only=unread_only,
        portfolio_id=portfolio_id,
        limit=limit,
        offset=offset,
    )
    
    # Convert and add portfolio name
    alert_list = []
    for a in alerts:
        schema = AlertSchema.model_validate(a)
        if a.portfolio:
            schema.portfolio_name = a.portfolio.name
        alert_list.append(schema)
    
    return AlertList(
        alerts=alert_list,
        total=total,
        unread_count=unread_count,
    )


@router.get("/alerts/unread-count")
async def get_unread_alert_count(
    clerk_user: ClerkUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get count of unread alerts.
    """
    user = await get_db_user(db, clerk_user)
    count = await crud.get_unread_count(db, user.id)
    return {"unread_count": count}


@router.patch("/alerts/{alert_id}", response_model=AlertSchema)
async def mark_alert_read(
    alert_id: UUID,
    clerk_user: ClerkUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Mark an alert as read.
    """
    user = await get_db_user(db, clerk_user)
    
    alert = await crud.mark_alert_read(db, alert_id, user.id)
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    
    return AlertSchema.model_validate(alert)


@router.post("/alerts/mark-all-read")
async def mark_all_alerts_read(
    portfolio_id: Optional[UUID] = None,
    clerk_user: ClerkUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Mark all alerts as read.
    """
    user = await get_db_user(db, clerk_user)
    
    count = await crud.mark_all_alerts_read(db, user.id, portfolio_id)
    return {"marked_read": count}


@router.delete("/alerts/{alert_id}")
async def delete_alert(
    alert_id: UUID,
    clerk_user: ClerkUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Delete an alert.
    """
    user = await get_db_user(db, clerk_user)
    
    deleted = await crud.delete_alert(db, alert_id, user.id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Alert not found")
    
    logger.info(f"User {user.email} deleted alert: {alert_id}")
    return {"message": "Alert deleted", "deleted": True}


@router.delete("/alerts")
async def delete_all_alerts(
    read_only: bool = Query(False, description="If true, only delete read alerts"),
    clerk_user: ClerkUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Delete alerts (all or just read ones).
    """
    user = await get_db_user(db, clerk_user)
    
    if read_only:
        # Delete only read alerts
        deleted_count = await crud.delete_read_alerts(db, user.id)
    else:
        # Delete all alerts (keep 0 days)
        deleted_count = await crud.delete_old_alerts(db, user.id, keep_days=0)
    
    logger.info(f"User {user.email} deleted {deleted_count} alerts")
    return {"message": f"Deleted {deleted_count} alerts", "deleted_count": deleted_count}


# =============================================================================
# SNAPSHOT ROUTES (Read-only, populated by daily job)
# =============================================================================

@router.get("/portfolios/{portfolio_id}/snapshots", response_model=SnapshotList)
async def get_portfolio_snapshots(
    portfolio_id: UUID,
    days: int = Query(90, ge=1, le=365),
    clerk_user: ClerkUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get daily snapshots for a portfolio (for charting).
    Snapshots are created by the daily tracking job.
    """
    user = await get_db_user(db, clerk_user)
    
    # Verify ownership
    portfolio = await crud.get_portfolio(db, portfolio_id, user.id)
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    
    from datetime import timedelta
    start_date = datetime.utcnow() - timedelta(days=days)
    
    snapshots = await crud.get_portfolio_snapshots(
        db, portfolio_id, start_date=start_date
    )
    
    return SnapshotList(
        snapshots=[SnapshotSchema.model_validate(s) for s in snapshots],
        total=len(snapshots),
    )


@router.get("/portfolios/{portfolio_id}/performance")
async def get_portfolio_performance(
    portfolio_id: UUID,
    days: int = Query(30, ge=1, le=365),
    clerk_user: ClerkUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get performance summary for a portfolio.
    """
    user = await get_db_user(db, clerk_user)
    
    # Verify ownership
    portfolio = await crud.get_portfolio(db, portfolio_id, user.id)
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    
    performance = await crud.get_portfolio_performance(db, portfolio_id, days)
    return performance


# =============================================================================
# DASHBOARD ROUTE
# =============================================================================

@router.get("/dashboard")
async def get_dashboard(
    clerk_user: ClerkUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get dashboard summary: portfolios, recent alerts, recent analyses.
    """
    user = await get_db_user(db, clerk_user)
    usage = await crud.get_usage_limits(db, user.id)
    
    # Get portfolios with latest snapshot info
    portfolios, _ = await crud.get_user_portfolios(db, user.id, limit=10)
    
    portfolio_summaries = []
    for p in portfolios:
        latest = await crud.get_latest_snapshot(db, p.id)
        portfolio_summaries.append({
            "id": str(p.id),
            "name": p.name,
            "positions_count": len(p.positions) if p.positions else 0,
            "is_default": p.is_default,
            "is_tracked": p.is_tracked,
            "latest_return": latest.cumulative_return if latest else None,
            "latest_regime": latest.regime if latest else None,
            "latest_signal": latest.signal if latest else None,
            "latest_snapshot_date": latest.snapshot_date.isoformat() if latest else None,
        })
    
    # Get recent alerts
    alerts, _, unread_count = await crud.get_user_alerts(db, user.id, limit=5)
    alert_summaries = [
        {
            "id": str(a.id),
            "alert_type": a.alert_type.value,
            "severity": a.severity.value,
            "message": a.message,
            "portfolio_name": a.portfolio.name if a.portfolio else None,
            "is_read": a.is_read,
            "created_at": a.created_at.isoformat(),
        }
        for a in alerts
    ]
    
    # Get recent analyses
    analyses, _ = await crud.get_user_analyses(db, user.id, limit=5)
    analysis_summaries = [
        {
            "id": str(a.id),
            "portfolio_name": a.portfolio_name,
            "result_summary": a.result_summary,
            "created_at": a.created_at.isoformat(),
        }
        for a in analyses
    ]
    
    return {
        "user": {
            "id": str(user.id),
            "email": user.email,
            "full_name": user.full_name,
            "plan": user.plan.value,
            "usage_count": usage["usage_count"],
            "usage_limit": usage["usage_limit"],
        },
        "portfolios": portfolio_summaries,
        "alerts": alert_summaries,
        "unread_alerts": unread_count,
        "recent_analyses": analysis_summaries,
    }
