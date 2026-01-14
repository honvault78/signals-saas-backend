"""
CRUD Operations

Database Create, Read, Update, Delete operations.
Each module handles a specific entity.
"""

from .users import (
    get_user_by_clerk_id,
    get_user_by_id,
    create_user,
    update_user,
    sync_clerk_user,
    increment_usage,
    get_usage_limits,
)

from .portfolios import (
    get_portfolio,
    get_user_portfolios,
    create_portfolio,
    update_portfolio,
    delete_portfolio,
    set_default_portfolio,
    get_tracked_portfolios,
)

from .analyses import (
    get_analysis,
    get_user_analyses,
    create_analysis,
    delete_analysis,
    get_portfolio_analyses,
    delete_old_analyses,
)

from .alerts import (
    get_alert,
    get_user_alerts,
    create_alert,
    mark_alert_read,
    mark_all_alerts_read,
    get_unread_count,
    delete_alert,
    delete_read_alerts,
    delete_old_alerts,
)

from .snapshots import (
    get_snapshot,
    get_portfolio_snapshots,
    create_snapshot,
    get_latest_snapshot,
    delete_old_snapshots,
    get_portfolio_performance,
)

__all__ = [
    # Users
    "get_user_by_clerk_id",
    "get_user_by_id",
    "create_user",
    "update_user",
    "sync_clerk_user",
    "increment_usage",
    "get_usage_limits",
    # Portfolios
    "get_portfolio",
    "get_user_portfolios",
    "create_portfolio",
    "update_portfolio",
    "delete_portfolio",
    "set_default_portfolio",
    "get_tracked_portfolios",
    # Analyses
    "get_analysis",
    "get_user_analyses",
    "create_analysis",
    "delete_analysis",
    "get_portfolio_analyses",
    "delete_old_analyses",
    # Alerts
    "get_alert",
    "get_user_alerts",
    "create_alert",
    "mark_alert_read",
    "mark_all_alerts_read",
    "get_unread_count",
    "delete_alert",
    "delete_read_alerts",
    "delete_old_alerts",
    # Snapshots
    "get_snapshot",
    "get_portfolio_snapshots",
    "create_snapshot",
    "get_latest_snapshot",
    "delete_old_snapshots",
    "get_portfolio_performance",
]
