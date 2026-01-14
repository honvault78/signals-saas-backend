"""
Signals SaaS Engine - Investment Analysis Engine

This package provides institutional-grade portfolio analysis including:
- Data fetching from Yahoo Finance
- Portfolio construction and returns calculation
- Performance backtesting
- Market regime detection
- Statistical analysis
- Chart generation
- AI-powered memo generation
- Professional HTML report generation
"""

from .data_fetcher import (
    fetch_ticker_data,
    fetch_multiple_tickers,
    calculate_returns,
    get_ticker_info,
    validate_tickers,
    DataFetchError,
    InsufficientDataError,
)

from .portfolio import (
    Position,
    PortfolioDefinition,
    PortfolioReturns,
    build_portfolio_returns,
    calculate_position_attribution,
)

from .backtest import (
    PerformanceMetrics,
    BacktestResult,
    calculate_drawdown,
    calculate_performance_metrics,
    run_backtest,
    run_full_backtest,
)

from .market_regime_detector import (
    MarketRegimeDetector,
    MarketRegime,
    RegimeAnalysis,
    RegimeMetrics,
)

from .regime import (
    StrategicSignal,
    TechnicalIndicators,
    RegimeSummary,
    calculate_technical_indicators,
    detect_regime,
    generate_trading_signals,
)

from .stats_analysis import (
    EnhancedStatistics,
    calculate_enhanced_statistics,
    calculate_memo_stats,
)

from .charts import (
    create_regime_chart,
    create_performance_chart,
    create_distribution_chart,
    create_all_charts,
)

from .memo import (
    generate_memo,
    build_tactical_memo_prompt,
)

from .report import (
    generate_html_report,
)

__all__ = [
    # Data
    "fetch_ticker_data",
    "fetch_multiple_tickers", 
    "calculate_returns",
    "get_ticker_info",
    "validate_tickers",
    "DataFetchError",
    "InsufficientDataError",
    # Portfolio
    "Position",
    "PortfolioDefinition",
    "PortfolioReturns",
    "build_portfolio_returns",
    "calculate_position_attribution",
    # Backtest
    "PerformanceMetrics",
    "BacktestResult",
    "calculate_drawdown",
    "calculate_performance_metrics",
    "run_backtest",
    "run_full_backtest",
    # Regime
    "MarketRegime",
    "StrategicSignal",
    "TechnicalIndicators",
    "RegimeMetrics",
    "RegimeSummary",
    "calculate_technical_indicators",
    "detect_regime",
    "generate_trading_signals",
    # Statistics
    "EnhancedStatistics",
    "calculate_enhanced_statistics",
    "calculate_memo_stats",
    # Charts
    "create_regime_chart",
    "create_performance_chart",
    "create_distribution_chart",
    "create_all_charts",
    # Memo
    "generate_memo",
    "build_tactical_memo_prompt",
    # Report
    "generate_html_report",
]
