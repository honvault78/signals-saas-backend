"""
Backtest Module - Performance Metrics Calculation

Calculates all performance metrics for the portfolio over multiple time periods.
Investment-grade implementation matching institutional standards.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .portfolio import PortfolioDefinition, PortfolioReturns, build_portfolio_returns

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for a single time period."""
    
    # Period info
    period_label: str
    period_days: int
    trading_days: int
    start_date: str
    end_date: str
    
    # Returns
    total_return: float  # As decimal (0.05 = 5%)
    annualized_return: float
    
    # Risk
    volatility: float  # Annualized
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float  # As decimal (-0.08 = -8%)
    
    # Win/Loss
    win_rate: float  # As decimal (0.55 = 55%)
    best_day: float
    worst_day: float
    
    # Attribution
    long_contribution: float
    short_contribution: float
    
    # Additional
    calmar_ratio: float
    avg_daily_return: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "period_label": self.period_label,
            "period_days": self.period_days,
            "trading_days": self.trading_days,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "total_return": round(self.total_return, 6),
            "total_return_pct": round(self.total_return * 100, 2),
            "annualized_return": round(self.annualized_return, 6),
            "annualized_return_pct": round(self.annualized_return * 100, 2),
            "volatility": round(self.volatility, 6),
            "volatility_pct": round(self.volatility * 100, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 3),
            "sortino_ratio": round(self.sortino_ratio, 3),
            "max_drawdown": round(self.max_drawdown, 6),
            "max_drawdown_pct": round(self.max_drawdown * 100, 2),
            "win_rate": round(self.win_rate, 4),
            "win_rate_pct": round(self.win_rate * 100, 1),
            "best_day": round(self.best_day, 6),
            "best_day_pct": round(self.best_day * 100, 2),
            "worst_day": round(self.worst_day, 6),
            "worst_day_pct": round(self.worst_day * 100, 2),
            "long_contribution": round(self.long_contribution, 6),
            "long_contribution_pct": round(self.long_contribution * 100, 2),
            "short_contribution": round(self.short_contribution, 6),
            "short_contribution_pct": round(self.short_contribution * 100, 2),
            "calmar_ratio": round(self.calmar_ratio, 3),
            "avg_daily_return": round(self.avg_daily_return, 6),
        }


def calculate_drawdown(cumulative: pd.Series) -> pd.Series:
    """
    Calculate drawdown series from cumulative returns.
    
    Parameters
    ----------
    cumulative : pd.Series
        Cumulative returns (starts at 1.0)
        
    Returns
    -------
    pd.Series
        Drawdown series (negative values, -0.05 = 5% drawdown)
    """
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    return drawdown


def calculate_performance_metrics(
    portfolio_returns: PortfolioReturns,
    portfolio: PortfolioDefinition,
    period_label: str,
    period_days: int,
    risk_free_rate: float = 0.0
) -> PerformanceMetrics:
    """
    Calculate comprehensive performance metrics for a portfolio.
    
    Parameters
    ----------
    portfolio_returns : PortfolioReturns
        Calculated portfolio returns.
    portfolio : PortfolioDefinition
        Portfolio definition.
    period_label : str
        Label for the period (e.g., "180_days")
    period_days : int
        Number of calendar days in the period.
    risk_free_rate : float
        Annualized risk-free rate for Sharpe calculation.
        
    Returns
    -------
    PerformanceMetrics
        Calculated performance metrics.
    """
    daily_returns = portfolio_returns.daily_returns
    cumulative = portfolio_returns.cumulative
    trading_days = len(daily_returns)
    
    # Returns
    total_return = cumulative.iloc[-1] - 1
    annualized_return = (cumulative.iloc[-1]) ** (252 / trading_days) - 1
    avg_daily = daily_returns.mean()
    
    # Volatility (annualized)
    daily_vol = daily_returns.std()
    volatility = daily_vol * np.sqrt(252)
    
    # Sharpe Ratio
    daily_rf = risk_free_rate / 252
    excess_returns = daily_returns - daily_rf
    if daily_vol > 0:
        sharpe = (excess_returns.mean() * 252) / volatility
    else:
        sharpe = 0.0
    
    # Sortino Ratio (downside deviation)
    downside_returns = daily_returns[daily_returns < 0]
    if len(downside_returns) > 0:
        downside_std = downside_returns.std() * np.sqrt(252)
        if downside_std > 0:
            sortino = (annualized_return - risk_free_rate) / downside_std
        else:
            sortino = 0.0
    else:
        sortino = 0.0
    
    # Drawdown
    drawdown = calculate_drawdown(cumulative)
    max_dd = drawdown.min()
    
    # Win rate
    win_rate = (daily_returns > 0).sum() / trading_days if trading_days > 0 else 0
    
    # Best/Worst
    best_day = daily_returns.max()
    worst_day = daily_returns.min()
    
    # Long/Short attribution
    long_contribution = 0.0
    short_contribution = 0.0
    
    if portfolio_returns.long_cumulative is not None:
        long_contribution = portfolio_returns.long_cumulative.iloc[-1] - 1
    
    if portfolio_returns.short_cumulative is not None:
        short_contribution = portfolio_returns.short_cumulative.iloc[-1] - 1
    
    # Calmar Ratio
    if max_dd != 0:
        calmar = annualized_return / abs(max_dd)
    else:
        calmar = 0.0
    
    return PerformanceMetrics(
        period_label=period_label,
        period_days=period_days,
        trading_days=trading_days,
        start_date=str(portfolio_returns.start_date.date()),
        end_date=str(portfolio_returns.end_date.date()),
        total_return=total_return,
        annualized_return=annualized_return,
        volatility=volatility,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        max_drawdown=max_dd,
        win_rate=win_rate,
        best_day=best_day,
        worst_day=worst_day,
        long_contribution=long_contribution,
        short_contribution=short_contribution,
        calmar_ratio=calmar,
        avg_daily_return=avg_daily,
    )


def run_backtest(
    stock_returns: pd.DataFrame,
    portfolio: PortfolioDefinition,
    periods: List[int] = [14, 30, 180],
    risk_free_rate: float = 0.0
) -> Dict[str, PerformanceMetrics]:
    """
    Run backtest for multiple time periods.
    
    Parameters
    ----------
    stock_returns : pd.DataFrame
        Stock returns in PERCENTAGE form (columns are tickers).
    portfolio : PortfolioDefinition
        Portfolio definition.
    periods : List[int]
        List of calendar day periods to analyze.
    risk_free_rate : float
        Annualized risk-free rate.
        
    Returns
    -------
    Dict[str, PerformanceMetrics]
        Dictionary mapping period label to metrics.
    """
    results = {}
    
    for days in periods:
        period_label = f"{days}_days"
        
        try:
            # Build returns for this period
            portfolio_returns = build_portfolio_returns(
                stock_returns=stock_returns,
                portfolio=portfolio,
                analysis_days=days
            )
            
            # Calculate metrics
            metrics = calculate_performance_metrics(
                portfolio_returns=portfolio_returns,
                portfolio=portfolio,
                period_label=period_label,
                period_days=days,
                risk_free_rate=risk_free_rate
            )
            
            results[period_label] = metrics
            
        except Exception as e:
            logger.error(f"Failed to calculate metrics for {period_label}: {e}")
            raise
    
    return results


@dataclass
class BacktestResult:
    """Complete backtest result with all periods and attribution."""
    
    portfolio: PortfolioDefinition
    metrics_by_period: Dict[str, PerformanceMetrics]
    primary_period_returns: PortfolioReturns  # Usually 180 days
    position_attribution: Dict[str, Dict]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "portfolio_type": self.portfolio.portfolio_type,
            "gross_exposure": self.portfolio.gross_exposure,
            "net_exposure": self.portfolio.net_exposure,
            "long_exposure": self.portfolio.long_exposure,
            "short_exposure": self.portfolio.short_exposure,
            "positions": [
                {
                    "ticker": p.ticker,
                    "amount": p.amount,
                    "type": p.position_type,
                }
                for p in self.portfolio.positions
            ],
            "long_weights": self.portfolio.long_weights,
            "short_weights": self.portfolio.short_weights,
            "metrics": {
                period: metrics.to_dict()
                for period, metrics in self.metrics_by_period.items()
            },
            "attribution": self.position_attribution,
        }


def run_full_backtest(
    stock_returns: pd.DataFrame,
    portfolio: PortfolioDefinition,
    primary_period: int = 180,
    additional_periods: List[int] = [14, 30],
    risk_free_rate: float = 0.0
) -> BacktestResult:
    """
    Run complete backtest with all metrics and attribution.
    
    Parameters
    ----------
    stock_returns : pd.DataFrame
        Stock returns in PERCENTAGE form.
    portfolio : PortfolioDefinition
        Portfolio definition.
    primary_period : int
        Primary analysis period (used for attribution).
    additional_periods : List[int]
        Additional periods to analyze.
    risk_free_rate : float
        Annualized risk-free rate.
        
    Returns
    -------
    BacktestResult
        Complete backtest results.
    """
    from .portfolio import calculate_position_attribution
    
    # All periods to analyze
    all_periods = sorted(set([primary_period] + additional_periods))
    
    # Run backtest for all periods
    metrics_by_period = run_backtest(
        stock_returns=stock_returns,
        portfolio=portfolio,
        periods=all_periods,
        risk_free_rate=risk_free_rate
    )
    
    # Get primary period returns for attribution
    primary_returns = build_portfolio_returns(
        stock_returns=stock_returns,
        portfolio=portfolio,
        analysis_days=primary_period
    )
    
    # Calculate position attribution
    attribution = calculate_position_attribution(
        portfolio_returns=primary_returns,
        portfolio=portfolio
    )
    
    return BacktestResult(
        portfolio=portfolio,
        metrics_by_period=metrics_by_period,
        primary_period_returns=primary_returns,
        position_attribution=attribution,
    )
