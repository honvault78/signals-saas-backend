"""
Bavella Validity Engine — FastAPI Routes
=========================================

API endpoints for validity analysis.

Add to your main.py:
    from engine.validity.routes import router as validity_router
    app.include_router(validity_router, prefix="/api/validity", tags=["validity"])
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np

# Import validity engine
from .analyzer import ValidityAnalyzer, ValidityReport
from .detector import ValidityDetector, DetectorResult
from .episodes import EpisodeStore, get_episode_store, EpisodeSummary
from .core import FailureMode, ValidityState

logger = logging.getLogger(__name__)

router = APIRouter()


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class ValidityCheckRequest(BaseModel):
    """Request for validity check."""
    node_id: str = Field(..., description="Identifier for the series/relationship")
    owner_id: str = Field(default="default", description="User/tenant ID")
    returns: List[float] = Field(..., description="Daily returns as decimals")
    timestamps: Optional[List[str]] = Field(None, description="ISO timestamps (optional)")
    
    # From regime detection (optional but recommended)
    adf_pvalue: Optional[float] = Field(None, description="ADF p-value from regime detection")
    halflife: Optional[float] = Field(None, description="Mean reversion half-life")
    z_score: Optional[float] = Field(None, description="Current Z-score")
    regime: str = Field(default="unknown", description="Current regime string")


class ValidityCheckResponse(BaseModel):
    """Response from validity check."""
    # Core verdict
    validity_score: float
    validity_state: str
    is_valid: bool
    
    # Failure modes
    fm_count: int
    primary_fm: Optional[Dict[str, Any]]
    active_fms: List[Dict[str, Any]]
    
    # Context
    regime: str
    adf_pvalue: float
    halflife: float
    z_score: float
    
    # Episode
    has_active_episode: bool
    episode: Optional[Dict[str, Any]]
    episode_is_new: bool
    
    # Recovery
    recovery_estimate: Optional[Dict[str, Any]]
    
    # Insights
    insights: List[str]
    warnings: List[str]
    
    # Full report (for detailed view)
    full_report: Dict[str, Any]


class PortfolioValidityRequest(BaseModel):
    """Request for portfolio validity check (integrates with existing analysis)."""
    portfolio_name: str = Field(default="Portfolio")
    positions: List[Dict[str, Any]] = Field(..., description="Positions from portfolio analysis")
    analysis_period_days: int = Field(default=180)
    
    # Regime detection results (from your existing /analyze endpoint)
    regime_summary: Optional[Dict[str, Any]] = Field(None)


class EpisodeHistoryResponse(BaseModel):
    """Response for episode history."""
    episodes: List[Dict[str, Any]]
    summary: Dict[str, Any]


# =============================================================================
# ANALYZER DEPENDENCY
# =============================================================================

_analyzer: Optional[ValidityAnalyzer] = None


def get_analyzer() -> ValidityAnalyzer:
    """Get or create validity analyzer."""
    global _analyzer
    if _analyzer is None:
        _analyzer = ValidityAnalyzer()
    return _analyzer


# =============================================================================
# ENDPOINTS
# =============================================================================

@router.post("/check", response_model=ValidityCheckResponse)
async def check_validity(
    request: ValidityCheckRequest,
    analyzer: ValidityAnalyzer = Depends(get_analyzer),
) -> ValidityCheckResponse:
    """
    Check validity of a series/relationship.
    
    This is the main validity endpoint. Pass returns and regime detection output.
    """
    try:
        # Convert returns to pandas Series
        if request.timestamps:
            index = pd.to_datetime(request.timestamps)
            returns = pd.Series(request.returns, index=index)
        else:
            returns = pd.Series(request.returns)
        
        # Run analysis
        report = analyzer.analyze(
            node_id=request.node_id,
            returns=returns,
            owner_id=request.owner_id,
            adf_pvalue=request.adf_pvalue,
            halflife=request.halflife,
            z_score=request.z_score,
            regime=request.regime,
        )
        
        # Build response
        return ValidityCheckResponse(
            validity_score=round(report.verdict.validity_score, 1),
            validity_state=report.verdict.validity_state.value,
            is_valid=report.verdict.is_valid,
            fm_count=len(report.detection.active_fms),
            primary_fm=report.detection.primary_fm.to_dict() if report.detection.primary_fm else None,
            active_fms=[fm.to_dict() for fm in report.detection.active_fms],
            regime=report.detection.regime,
            adf_pvalue=round(report.detection.adf_pvalue, 4),
            halflife=round(report.detection.halflife, 1),
            z_score=round(report.detection.z_score, 2),
            has_active_episode=report.episode is not None and report.episode.state.value == "active",
            episode=report.episode.to_dict() if report.episode else None,
            episode_is_new=report.episode_is_new,
            recovery_estimate=report.recovery_estimate.to_dict() if report.recovery_estimate else None,
            insights=report.insights,
            warnings=report.warnings,
            full_report=report.to_dict(),
        )
        
    except Exception as e:
        logger.error(f"Validity check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{node_id}")
async def get_validity_status(
    node_id: str,
    owner_id: str = "default",
    analyzer: ValidityAnalyzer = Depends(get_analyzer),
) -> Dict[str, Any]:
    """
    Get current validity status for a node.
    
    Returns the most recent verdict and active episode info.
    """
    episode_store = analyzer.episode_store
    
    # Get active episode
    active_episode = episode_store.get_active_episode(node_id)
    
    # Get recent history
    history = episode_store.get_node_history(node_id, limit=5)
    
    return {
        "node_id": node_id,
        "has_active_episode": active_episode is not None,
        "active_episode": active_episode.to_dict() if active_episode else None,
        "recent_episodes": [ep.to_dict() for ep in history],
    }


@router.get("/episodes/{node_id}", response_model=EpisodeHistoryResponse)
async def get_episode_history(
    node_id: str,
    limit: int = 10,
    analyzer: ValidityAnalyzer = Depends(get_analyzer),
) -> EpisodeHistoryResponse:
    """
    Get episode history for a node.
    """
    episode_store = analyzer.episode_store
    
    episodes = episode_store.get_node_history(node_id, limit=limit)
    summary = episode_store.get_summary()
    
    return EpisodeHistoryResponse(
        episodes=[ep.to_dict() for ep in episodes],
        summary=summary.to_dict(),
    )


@router.get("/summary")
async def get_validity_summary(
    owner_id: str = "default",
    analyzer: ValidityAnalyzer = Depends(get_analyzer),
) -> Dict[str, Any]:
    """
    Get summary of all validity episodes.
    """
    episode_store = analyzer.episode_store
    summary = episode_store.get_summary(owner_id)
    
    return {
        "owner_id": owner_id,
        "summary": summary.to_dict(),
    }


@router.delete("/episodes/{node_id}")
async def clear_episodes(
    node_id: str,
    analyzer: ValidityAnalyzer = Depends(get_analyzer),
) -> Dict[str, str]:
    """
    Clear episode history for a node.
    
    Use with caution — this removes historical pattern matching data.
    """
    # Note: In production, you'd want more granular control
    analyzer.episode_store.clear()
    return {"status": "cleared", "node_id": node_id}


# =============================================================================
# INTEGRATION HELPER - Use with existing /analyze endpoint
# =============================================================================

def check_portfolio_validity(
    portfolio_returns: pd.Series,
    regime_summary: Dict[str, Any],
    portfolio_name: str = "Portfolio",
    owner_id: str = "default",
    z_score_override: Optional[float] = None,
) -> ValidityReport:
    """
    Check portfolio validity using results from existing /analyze endpoint.
    
    Integration example in your main.py /analyze endpoint:
    
        from engine.validity.routes import check_portfolio_validity
        
        # After running regime detection...
        validity_report = check_portfolio_validity(
            portfolio_returns=portfolio_returns.daily_returns,
            regime_summary=regime_summary.to_dict(),
            portfolio_name=request.portfolio_name,
            z_score_override=z_score,  # Pass the z_score from detect_regime
        )
        
        # Include in response
        return {
            ...
            "validity": validity_report.to_summary_dict(),
        }
    """
    analyzer = get_analyzer()
    
    # Extract values - they may be at top level or nested under 'metrics'
    metrics = regime_summary.get("metrics", regime_summary)
    
    # Try different key names for ADF p-value
    adf_pvalue = (
        regime_summary.get("adf_pvalue") or 
        metrics.get("adf_pvalue") or
        regime_summary.get("adf_p_value") or
        metrics.get("adf_p_value")
    )
    
    # Try different key names for half-life
    halflife = (
        regime_summary.get("halflife") or 
        metrics.get("halflife") or
        regime_summary.get("halflife_periods") or 
        metrics.get("halflife_periods") or
        regime_summary.get("mean_reversion_halflife") or
        metrics.get("mean_reversion_halflife")
    )
    
    # Use override if provided, otherwise try to find in regime_summary
    z_score = z_score_override
    if z_score is None:
        z_score = (
            regime_summary.get("z_score") or 
            metrics.get("z_score") or
            regime_summary.get("zscore") or
            metrics.get("zscore") or
            regime_summary.get("z_score_current") or
            metrics.get("z_score_current")
        )
    
    # Try different key names for regime
    regime = (
        regime_summary.get("current_regime") or
        regime_summary.get("regime") or
        regime_summary.get("primary_regime") or
        "unknown"
    )
    
    # Log what we found for debugging
    logger.info(f"Validity check - ADF: {adf_pvalue}, Halflife: {halflife}, Z-score: {z_score}, Regime: {regime}")
    
    return analyzer.analyze(
        node_id=portfolio_name,
        returns=portfolio_returns,
        owner_id=owner_id,
        adf_pvalue=adf_pvalue,
        halflife=halflife,
        z_score=z_score,
        regime=regime,
    )
