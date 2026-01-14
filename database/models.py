"""
Database Models

SQLAlchemy ORM models for the Signals SaaS application.
Defines the database schema and relationships.
"""

import uuid
from datetime import datetime
from typing import Optional, List

from sqlalchemy import (
    String,
    Integer,
    Float,
    Boolean,
    Text,
    DateTime,
    ForeignKey,
    Index,
    Enum as SQLEnum,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID, JSONB
import enum

from .connection import Base


# =============================================================================
# ENUMS
# =============================================================================

class UserPlan(str, enum.Enum):
    """User subscription plans."""
    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"


class AlertType(str, enum.Enum):
    """Types of alerts."""
    BUY = "buy"
    SELL = "sell"
    REGIME_CHANGE = "regime_change"
    DRAWDOWN = "drawdown"
    ZSCORE_EXTREME = "zscore_extreme"


class AlertSeverity(str, enum.Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


# Helper to get enum values (lowercase) instead of names (uppercase)
def _enum_values(enum_class):
    """Return enum values for SQLAlchemy to use lowercase in DB."""
    return [e.value for e in enum_class]


# =============================================================================
# USERS
# =============================================================================

class User(Base):
    """
    User model - synced from Clerk authentication.
    
    The clerk_id is the primary identifier from Clerk.
    We store a local copy for foreign key relationships and usage tracking.
    """
    __tablename__ = "users"
    
    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    
    # Clerk integration
    clerk_id: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        nullable=False,
        index=True,
    )
    
    # User info (synced from Clerk)
    email: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    first_name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    last_name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    
    # Subscription - use values_callable to send lowercase to DB
    plan: Mapped[UserPlan] = mapped_column(
        SQLEnum(UserPlan, values_callable=_enum_values),
        default=UserPlan.FREE,
        nullable=False,
    )
    
    # Usage tracking
    usage_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    usage_reset_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        nullable=False,
    )
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        nullable=False,
    )
    last_active_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    
    # Relationships
    portfolios: Mapped[List["Portfolio"]] = relationship(
        "Portfolio",
        back_populates="user",
        cascade="all, delete-orphan",
    )
    analyses: Mapped[List["Analysis"]] = relationship(
        "Analysis",
        back_populates="user",
        cascade="all, delete-orphan",
    )
    alerts: Mapped[List["Alert"]] = relationship(
        "Alert",
        back_populates="user",
        cascade="all, delete-orphan",
    )
    
    def __repr__(self) -> str:
        return f"<User {self.email}>"
    
    @property
    def full_name(self) -> str:
        """Get user's full name."""
        parts = [self.first_name, self.last_name]
        return " ".join(p for p in parts if p) or self.email


# =============================================================================
# PORTFOLIOS
# =============================================================================

class Portfolio(Base):
    """
    Portfolio model - stores saved portfolio configurations.
    
    Positions are stored as JSONB for flexibility:
    [{"ticker": "AAPL", "amount": 100000}, {"ticker": "MSFT", "amount": -50000}]
    """
    __tablename__ = "portfolios"
    
    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    
    # Foreign key to user
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    
    # Portfolio details
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Positions stored as JSONB
    positions: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
    )
    
    # Settings
    is_default: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_tracked: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )
    
    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="portfolios")
    analyses: Mapped[List["Analysis"]] = relationship(
        "Analysis",
        back_populates="portfolio",
        cascade="all, delete-orphan",
    )
    snapshots: Mapped[List["PortfolioSnapshot"]] = relationship(
        "PortfolioSnapshot",
        back_populates="portfolio",
        cascade="all, delete-orphan",
    )
    alerts: Mapped[List["Alert"]] = relationship(
        "Alert",
        back_populates="portfolio",
        cascade="all, delete-orphan",
    )
    
    # Indexes
    __table_args__ = (
        Index("ix_portfolios_user_default", "user_id", "is_default"),
    )
    
    def __repr__(self) -> str:
        return f"<Portfolio {self.name}>"


# =============================================================================
# ANALYSES
# =============================================================================

class Analysis(Base):
    """
    Analysis model - stores full analysis results.
    
    Each analysis is a point-in-time snapshot of a portfolio analysis.
    HTML report is stored for quick retrieval.
    """
    __tablename__ = "analyses"
    
    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    
    # Foreign keys
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    portfolio_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("portfolios.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    
    # Request parameters
    portfolio_name: Mapped[str] = mapped_column(String(255), nullable=False)
    positions: Mapped[dict] = mapped_column(JSONB, nullable=False)
    analysis_period_days: Mapped[int] = mapped_column(Integer, nullable=False)
    
    # Summary results (for quick display without parsing HTML)
    result_summary: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
    )
    
    # Full HTML report (compressed in production if needed)
    html_report: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # AI memo (stored separately, generated on-demand)
    ai_memo: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Metadata
    duration_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        nullable=False,
        index=True,
    )
    
    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="analyses")
    portfolio: Mapped[Optional["Portfolio"]] = relationship(
        "Portfolio",
        back_populates="analyses",
    )
    
    # Indexes
    __table_args__ = (
        Index("ix_analyses_user_created", "user_id", "created_at"),
    )
    
    def __repr__(self) -> str:
        return f"<Analysis {self.portfolio_name} @ {self.created_at}>"


# =============================================================================
# PORTFOLIO SNAPSHOTS (Daily Tracking)
# =============================================================================

class PortfolioSnapshot(Base):
    """
    Portfolio snapshot - daily metrics for tracked portfolios.
    
    Created by the daily update job for each tracked portfolio.
    Enables performance tracking and charting over time.
    """
    __tablename__ = "portfolio_snapshots"
    
    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    
    # Foreign key to portfolio
    portfolio_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("portfolios.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    
    # Snapshot date (trading day)
    snapshot_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
    )
    
    # Performance metrics
    cumulative_return: Mapped[float] = mapped_column(Float, nullable=False)
    daily_return: Mapped[float] = mapped_column(Float, nullable=False)
    portfolio_value: Mapped[float] = mapped_column(Float, nullable=False)
    
    # Regime & signals
    regime: Mapped[str] = mapped_column(String(50), nullable=False)
    signal: Mapped[str] = mapped_column(String(20), nullable=False)  # BUY, SELL, HOLD
    
    # Technical indicators
    z_score: Mapped[float] = mapped_column(Float, nullable=False)
    rsi: Mapped[float] = mapped_column(Float, nullable=False)
    adf_pvalue: Mapped[float] = mapped_column(Float, nullable=False)
    trend_score: Mapped[float] = mapped_column(Float, nullable=False)
    
    # Timestamp
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        nullable=False,
    )
    
    # Relationships
    portfolio: Mapped["Portfolio"] = relationship(
        "Portfolio",
        back_populates="snapshots",
    )
    
    # Indexes
    __table_args__ = (
        Index("ix_snapshots_portfolio_date", "portfolio_id", "snapshot_date"),
        Index("ix_snapshots_date", "snapshot_date"),
    )
    
    def __repr__(self) -> str:
        return f"<Snapshot {self.portfolio_id} @ {self.snapshot_date}>"


# =============================================================================
# ALERTS
# =============================================================================

class Alert(Base):
    """
    Alert model - notifications for users about portfolio events.
    
    Alerts are created when significant events occur:
    - New BUY/SELL signals
    - Regime changes
    - Significant drawdowns
    """
    __tablename__ = "alerts"
    
    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    
    # Foreign keys
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    portfolio_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("portfolios.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    
    # Alert details - use values_callable to send lowercase to DB
    alert_type: Mapped[AlertType] = mapped_column(
        SQLEnum(AlertType, values_callable=_enum_values),
        nullable=False,
        index=True,
    )
    severity: Mapped[AlertSeverity] = mapped_column(
        SQLEnum(AlertSeverity, values_callable=_enum_values),
        default=AlertSeverity.INFO,
        nullable=False,
    )
    
    # Signal date (when the event occurred in market)
    signal_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )
    
    # Message
    message: Mapped[str] = mapped_column(Text, nullable=False)
    
    # Context at time of alert
    portfolio_value: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    regime: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    z_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    rsi: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Status
    is_read: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        nullable=False,
        index=True,
    )
    read_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    
    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="alerts")
    portfolio: Mapped["Portfolio"] = relationship("Portfolio", back_populates="alerts")
    
    # Indexes
    __table_args__ = (
        Index("ix_alerts_user_unread", "user_id", "is_read"),
        Index("ix_alerts_user_created", "user_id", "created_at"),
    )
    
    def __repr__(self) -> str:
        return f"<Alert {self.alert_type.value} for {self.portfolio_id}>"
