"""
Database Package

Provides database connectivity, models, schemas, and CRUD operations.

Usage:
    from database import get_db, User, Portfolio
    from database.crud import get_user_by_clerk_id, create_portfolio
"""

# Connection management
from .connection import (
    Base,
    get_db,
    get_db_context,
    get_engine,
    get_session_factory,
    init_database,
    close_database,
    check_database_connection,
)

# Configuration
from .config import (
    get_database_settings,
    get_database_url,
    DatabaseSettings,
)

# Models
from .models import (
    User,
    Portfolio,
    Analysis,
    PortfolioSnapshot,
    Alert,
    UserPlan,
    AlertType,
    AlertSeverity,
)

# Schemas
from .schemas import (
    # Users
    UserCreate,
    UserUpdate,
    User as UserSchema,
    UserProfile,
    # Portfolios
    PortfolioCreate,
    PortfolioUpdate,
    Portfolio as PortfolioSchema,
    PortfolioWithStats,
    PortfolioList,
    PositionBase,
    # Analyses
    AnalysisCreate,
    Analysis as AnalysisSchema,
    AnalysisFull,
    AnalysisList,
    AnalysisResultSummary,
    # Snapshots
    SnapshotCreate,
    Snapshot as SnapshotSchema,
    SnapshotList,
    # Alerts
    AlertCreate,
    AlertUpdate,
    Alert as AlertSchema,
    AlertList,
    # Dashboard
    Dashboard,
    DashboardPortfolio,
)

# CRUD operations
from . import crud

__all__ = [
    # Connection
    "Base",
    "get_db",
    "get_db_context",
    "get_engine",
    "get_session_factory",
    "init_database",
    "close_database",
    "check_database_connection",
    # Config
    "get_database_settings",
    "get_database_url",
    "DatabaseSettings",
    # Models
    "User",
    "Portfolio",
    "Analysis",
    "PortfolioSnapshot",
    "Alert",
    "UserPlan",
    "AlertType",
    "AlertSeverity",
    # Schemas
    "UserCreate",
    "UserUpdate",
    "UserSchema",
    "UserProfile",
    "PortfolioCreate",
    "PortfolioUpdate",
    "PortfolioSchema",
    "PortfolioWithStats",
    "PortfolioList",
    "PositionBase",
    "AnalysisCreate",
    "AnalysisSchema",
    "AnalysisFull",
    "AnalysisList",
    "AnalysisResultSummary",
    "SnapshotCreate",
    "SnapshotSchema",
    "SnapshotList",
    "AlertCreate",
    "AlertUpdate",
    "AlertSchema",
    "AlertList",
    "Dashboard",
    "DashboardPortfolio",
    # CRUD
    "crud",
]
