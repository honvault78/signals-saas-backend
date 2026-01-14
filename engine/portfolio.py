"""
Portfolio Module - Portfolio Construction and Returns Calculation

Handles single stocks, pairs, and baskets with proper long/short attribution.
Investment-grade implementation matching the notebook methodology exactly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Represents a single position in the portfolio."""
    ticker: str
    amount: float  # Positive = long, negative = short
    
    @property
    def is_long(self) -> bool:
        return self.amount > 0
    
    @property
    def is_short(self) -> bool:
        return self.amount < 0
    
    @property
    def abs_amount(self) -> float:
        return abs(self.amount)
    
    @property
    def position_type(self) -> str:
        return "LONG" if self.is_long else "SHORT"


@dataclass
class PortfolioDefinition:
    """
    Defines a portfolio with its positions and computed exposures.
    """
    positions: List[Position]
    
    @property
    def long_positions(self) -> List[Position]:
        return [p for p in self.positions if p.is_long]
    
    @property
    def short_positions(self) -> List[Position]:
        return [p for p in self.positions if p.is_short]
    
    @property
    def tickers(self) -> List[str]:
        return [p.ticker for p in self.positions]
    
    @property
    def long_exposure(self) -> float:
        return sum(p.amount for p in self.long_positions)
    
    @property
    def short_exposure(self) -> float:
        return abs(sum(p.amount for p in self.short_positions))
    
    @property
    def gross_exposure(self) -> float:
        return sum(p.abs_amount for p in self.positions)
    
    @property
    def net_exposure(self) -> float:
        return sum(p.amount for p in self.positions)
    
    @property
    def portfolio_type(self) -> str:
        """Determine portfolio type based on positions."""
        n_long = len(self.long_positions)
        n_short = len(self.short_positions)
        
        if n_short == 0:
            if n_long == 1:
                return "SINGLE_LONG"
            return "LONG_ONLY_BASKET"
        elif n_long == 0:
            if n_short == 1:
                return "SINGLE_SHORT"
            return "SHORT_ONLY_BASKET"
        else:
            if n_long == 1 and n_short == 1:
                return "PAIR_TRADE"
            return "LONG_SHORT_BASKET"
    
    @property
    def long_weights(self) -> Dict[str, float]:
        """Normalized long weights (sum to 100%)."""
        total = self.long_exposure
        if total == 0:
            return {}
        return {p.ticker: p.amount / total for p in self.long_positions}
    
    @property
    def short_weights(self) -> Dict[str, float]:
        """Normalized short weights (sum to 100%)."""
        total = self.short_exposure
        if total == 0:
            return {}
        return {p.ticker: p.abs_amount / total for p in self.short_positions}
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to {ticker: amount} dictionary."""
        return {p.ticker: p.amount for p in self.positions}
    
    @classmethod
    def from_dict(cls, positions_dict: Dict[str, float]) -> "PortfolioDefinition":
        """Create from {ticker: amount} dictionary."""
        positions = [
            Position(ticker=ticker, amount=amount)
            for ticker, amount in positions_dict.items()
        ]
        return cls(positions=positions)


@dataclass
class PortfolioReturns:
    """
    Contains calculated portfolio returns and attribution.
    """
    # Time series
    daily_returns: pd.Series  # Daily portfolio returns (decimal form, 0.01 = 1%)
    cumulative: pd.Series  # Cumulative returns (starts at 1.0)
    daily_pnl: pd.DataFrame  # P&L by position
    
    # Metadata
    start_date: datetime
    end_date: datetime
    trading_days: int
    gross_exposure: float
    
    # Attribution
    long_returns: Optional[pd.Series] = None
    short_returns: Optional[pd.Series] = None
    long_cumulative: Optional[pd.Series] = None
    short_cumulative: Optional[pd.Series] = None


def build_portfolio_returns(
    stock_returns: pd.DataFrame,
    portfolio: PortfolioDefinition,
    analysis_days: Optional[int] = None
) -> PortfolioReturns:
    """
    Build portfolio returns from stock returns and portfolio definition.
    
    This function exactly replicates the notebook methodology:
    - Returns are expected in PERCENTAGE form (1.5 = 1.5%)
    - Converts to decimal form internally (0.015)
    - Calculates dollar P&L then converts to portfolio return
    - Returns are based on gross exposure
    
    Parameters
    ----------
    stock_returns : pd.DataFrame
        DataFrame with tickers as columns, returns in PERCENTAGE form.
    portfolio : PortfolioDefinition
        Portfolio definition with positions.
    analysis_days : int, optional
        If specified, only use the last N calendar days.
        
    Returns
    -------
    PortfolioReturns
        Calculated portfolio returns and attribution.
    """
    # Make a copy and ensure datetime index
    returns = stock_returns.copy()
    returns.index = pd.to_datetime(returns.index)
    
    # Convert from percentage to decimal form (1.5% -> 0.015)
    returns = returns / 100
    
    # Apply clipping for extreme outliers (same as notebook)
    returns = returns.clip(lower=-0.5, upper=0.5)
    
    # Filter to analysis period if specified
    if analysis_days is not None:
        end_date = returns.index.max()
        start_date = end_date - timedelta(days=analysis_days)
        mask = (returns.index > start_date) & (returns.index <= end_date)
        returns = returns[mask].copy()
    
    # Verify all tickers are present
    missing = [t for t in portfolio.tickers if t not in returns.columns]
    if missing:
        raise ValueError(f"Missing tickers in returns data: {missing}")
    
    # Calculate daily P&L for each position
    positions_dict = portfolio.to_dict()
    daily_pnl = pd.DataFrame(index=returns.index)
    
    for ticker, amount in positions_dict.items():
        stock_ret = returns[ticker].fillna(0)
        daily_pnl[ticker] = amount * stock_ret
    
    # Total daily P&L
    daily_pnl["TOTAL"] = daily_pnl[portfolio.tickers].sum(axis=1)
    
    # Convert to portfolio return (based on gross exposure)
    gross = portfolio.gross_exposure
    portfolio_daily_returns = daily_pnl["TOTAL"] / gross
    
    # Cumulative returns (rebased to start at 1.0)
    portfolio_cumulative = (1 + portfolio_daily_returns).cumprod()
    
    # Long attribution
    long_returns = None
    long_cumulative = None
    if portfolio.long_positions:
        long_pnl = daily_pnl[[p.ticker for p in portfolio.long_positions]].sum(axis=1)
        long_returns = long_pnl / portfolio.long_exposure
        long_cumulative = (1 + long_returns).cumprod()
    
    # Short attribution
    short_returns = None
    short_cumulative = None
    if portfolio.short_positions:
        short_pnl = daily_pnl[[p.ticker for p in portfolio.short_positions]].sum(axis=1)
        short_returns = short_pnl / portfolio.short_exposure
        short_cumulative = (1 + short_returns).cumprod()
    
    return PortfolioReturns(
        daily_returns=portfolio_daily_returns,
        cumulative=portfolio_cumulative,
        daily_pnl=daily_pnl,
        start_date=returns.index.min(),
        end_date=returns.index.max(),
        trading_days=len(returns),
        gross_exposure=gross,
        long_returns=long_returns,
        short_returns=short_returns,
        long_cumulative=long_cumulative,
        short_cumulative=short_cumulative,
    )


def calculate_position_attribution(
    portfolio_returns: PortfolioReturns,
    portfolio: PortfolioDefinition
) -> Dict[str, Dict]:
    """
    Calculate detailed attribution for each position.
    
    Returns
    -------
    Dict[str, Dict]
        Dictionary mapping ticker to attribution metrics.
    """
    attribution = {}
    
    cumulative_pnl = portfolio_returns.daily_pnl.cumsum()
    
    for position in portfolio.positions:
        ticker = position.ticker
        
        if ticker not in cumulative_pnl.columns:
            continue
        
        final_pnl = cumulative_pnl[ticker].iloc[-1]
        
        # Position return (based on position size)
        if position.is_long:
            position_return = final_pnl / position.amount
        else:
            # For shorts, positive P&L means stock went down
            position_return = final_pnl / position.abs_amount
        
        attribution[ticker] = {
            "type": position.position_type,
            "initial_amount": position.amount,
            "final_pnl": final_pnl,
            "position_return": position_return,
            "contribution_to_portfolio": final_pnl / portfolio.gross_exposure,
        }
    
    return attribution
