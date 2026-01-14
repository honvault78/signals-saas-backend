"""
Pydantic Schemas

API request and response models for validation and serialization.
Separate from SQLAlchemy models to maintain clean separation.
"""

from datetime import datetime
from typing import Optional, List, Any
from uuid import UUID
import enum

from pydantic import BaseModel, Field, ConfigDict


# =============================================================================
# ENUMS (matching database enums)
# =============================================================================

class UserPlan(str, enum.Enum):
    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"


class AlertType(str, enum.Enum):
    BUY = "buy"
    SELL = "sell"
    REGIME_CHANGE = "regime_change"
    DRAWDOWN = "drawdown"
    ZSCORE_EXTREME = "zscore_extreme"


class AlertSeverity(str, enum.Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


# =============================================================================
# POSITION SCHEMAS
# =============================================================================

class PositionBase(BaseModel):
    """Single position in a portfolio."""
    ticker: str = Field(..., min_length=1, max_length=20)
    amount: float = Field(..., description="USD amount, negative for short")


class PositionCreate(PositionBase):
    """Create a position."""
    pass


class Position(PositionBase):
    """Position response."""
    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# USER SCHEMAS
# =============================================================================

class UserBase(BaseModel):
    """Base user fields."""
    email: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None


class UserCreate(UserBase):
    """Create a user (from Clerk sync)."""
    clerk_id: str


class UserUpdate(BaseModel):
    """Update user fields."""
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    plan: Optional[UserPlan] = None


class User(UserBase):
    """User response."""
    id: UUID
    clerk_id: str
    plan: UserPlan
    usage_count: int
    usage_reset_at: datetime
    created_at: datetime
    last_active_at: Optional[datetime] = None
    
    model_config = ConfigDict(from_attributes=True)


class UserProfile(BaseModel):
    """User profile response (for /me endpoint)."""
    id: UUID
    email: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    full_name: str
    plan: UserPlan
    usage_count: int
    usage_limit: int
    usage_reset_at: datetime
    created_at: datetime
    
    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# PORTFOLIO SCHEMAS
# =============================================================================

class PortfolioBase(BaseModel):
    """Base portfolio fields."""
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    positions: List[PositionBase] = Field(default_factory=list)
    is_default: bool = False
    is_tracked: bool = True


class PortfolioCreate(PortfolioBase):
    """Create a portfolio."""
    pass


class PortfolioUpdate(BaseModel):
    """Update portfolio fields."""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    positions: Optional[List[PositionBase]] = None
    is_default: Optional[bool] = None
    is_tracked: Optional[bool] = None


class Portfolio(PortfolioBase):
    """Portfolio response."""
    id: UUID
    user_id: UUID
    created_at: datetime
    updated_at: datetime
    
    model_config = ConfigDict(from_attributes=True)


class PortfolioWithStats(Portfolio):
    """Portfolio with latest stats (from snapshots)."""
    latest_return: Optional[float] = None
    latest_regime: Optional[str] = None
    latest_signal: Optional[str] = None
    latest_snapshot_date: Optional[datetime] = None


class PortfolioList(BaseModel):
    """List of portfolios response."""
    portfolios: List[Portfolio]
    total: int


# =============================================================================
# ANALYSIS SCHEMAS
# =============================================================================

class AnalysisResultSummary(BaseModel):
    """Summary of analysis results."""
    regime: str
    signal: str
    trend_score: float
    z_score: float
    rsi: float
    adf_pvalue: float
    cumulative_return: float
    volatility: float
    sharpe_ratio: Optional[float] = None
    max_drawdown: float


class AnalysisBase(BaseModel):
    """Base analysis fields."""
    portfolio_name: str
    positions: List[PositionBase]
    analysis_period_days: int = 180


class AnalysisCreate(AnalysisBase):
    """Create an analysis record."""
    portfolio_id: Optional[UUID] = None
    result_summary: dict = Field(default_factory=dict)
    html_report: Optional[str] = None
    ai_memo: Optional[str] = None
    duration_ms: Optional[int] = None


class Analysis(AnalysisBase):
    """Analysis response (without HTML)."""
    id: UUID
    user_id: UUID
    portfolio_id: Optional[UUID] = None
    result_summary: dict
    duration_ms: Optional[int] = None
    created_at: datetime
    has_html_report: bool = False
    has_ai_memo: bool = False
    
    model_config = ConfigDict(from_attributes=True)


class AnalysisFull(Analysis):
    """Full analysis response (with HTML and memo)."""
    html_report: Optional[str] = None
    ai_memo: Optional[str] = None


class AnalysisList(BaseModel):
    """List of analyses response."""
    analyses: List[Analysis]
    total: int


# =============================================================================
# SNAPSHOT SCHEMAS
# =============================================================================

class SnapshotBase(BaseModel):
    """Base snapshot fields."""
    snapshot_date: datetime
    cumulative_return: float
    daily_return: float
    portfolio_value: float
    regime: str
    signal: str
    z_score: float
    rsi: float
    adf_pvalue: float
    trend_score: float


class SnapshotCreate(SnapshotBase):
    """Create a snapshot."""
    portfolio_id: UUID


class Snapshot(SnapshotBase):
    """Snapshot response."""
    id: UUID
    portfolio_id: UUID
    created_at: datetime
    
    model_config = ConfigDict(from_attributes=True)


class SnapshotList(BaseModel):
    """List of snapshots for charting."""
    snapshots: List[Snapshot]
    total: int


# =============================================================================
# ALERT SCHEMAS
# =============================================================================

class AlertBase(BaseModel):
    """Base alert fields."""
    alert_type: AlertType
    severity: AlertSeverity
    signal_date: datetime
    message: str
    portfolio_value: Optional[float] = None
    regime: Optional[str] = None
    z_score: Optional[float] = None
    rsi: Optional[float] = None


class AlertCreate(AlertBase):
    """Create an alert."""
    portfolio_id: UUID


class Alert(AlertBase):
    """Alert response."""
    id: UUID
    user_id: UUID
    portfolio_id: UUID
    is_read: bool
    created_at: datetime
    read_at: Optional[datetime] = None
    
    # Include portfolio name for display
    portfolio_name: Optional[str] = None
    
    model_config = ConfigDict(from_attributes=True)


class AlertUpdate(BaseModel):
    """Update alert (mark as read)."""
    is_read: bool = True


class AlertList(BaseModel):
    """List of alerts response."""
    alerts: List[Alert]
    total: int
    unread_count: int


# =============================================================================
# DASHBOARD SCHEMAS
# =============================================================================

class DashboardPortfolio(BaseModel):
    """Portfolio summary for dashboard."""
    id: UUID
    name: str
    positions_count: int
    latest_return: Optional[float] = None
    return_7d: Optional[float] = None
    regime: Optional[str] = None
    signal: Optional[str] = None
    has_alert: bool = False


class Dashboard(BaseModel):
    """Dashboard response."""
    user: UserProfile
    portfolios: List[DashboardPortfolio]
    recent_alerts: List[Alert]
    recent_analyses: List[Analysis]
