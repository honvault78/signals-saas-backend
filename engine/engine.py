"""
Engine Module - Core Portfolio Analysis Engine

This module combines all the analysis functionality from the Jupyter notebook
into a production-grade SaaS engine. It provides:
- Portfolio definition and construction
- Returns calculation and backtesting
- Regime detection using MarketRegimeDetector
- Trading signal generation
- Chart generation
- Statistical analysis

Investment-grade implementation matching the Jupyter notebook exactly.
"""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from io import BytesIO
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import stats

# Import our modules
from data_fetcher import (
    fetch_ticker_data,
    fetch_multiple_tickers,
    calculate_returns,
    DataFetchError,
    InsufficientDataError,
)
from stats_analysis import (
    calculate_enhanced_statistics,
    calculate_memo_stats,
    EnhancedStatistics,
)
from memo import generate_memo, build_tactical_memo_prompt
from report import generate_html_report

# Import market regime detector
from market_regime_detector import MarketRegimeDetector, MarketRegime

logger = logging.getLogger(__name__)

# Re-export for main.py imports
__all__ = [
    'PortfolioDefinition',
    'Position',
    'PortfolioReturns',
    'BacktestResult',
    'PeriodMetrics',
    'RegimeSummary',
    'build_portfolio_returns',
    'run_full_backtest',
    'detect_regime',
    'generate_trading_signals',
    'create_all_charts',
    'calculate_enhanced_statistics',
    'calculate_memo_stats',
    'generate_memo',
    'generate_html_report',
    'fetch_ticker_data',
    'fetch_multiple_tickers',
    'calculate_returns',
    'DataFetchError',
    'InsufficientDataError',
]


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Position:
    """A single portfolio position."""
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


@dataclass
class PortfolioDefinition:
    """
    Complete portfolio definition with long/short positions.
    """
    positions: List[Position]
    
    @classmethod
    def from_dict(cls, positions_dict: Dict[str, float]) -> 'PortfolioDefinition':
        """Create portfolio from {ticker: amount} dictionary."""
        positions = [Position(ticker=t, amount=a) for t, a in positions_dict.items()]
        return cls(positions=positions)
    
    @property
    def tickers(self) -> List[str]:
        """All tickers in the portfolio."""
        return [p.ticker for p in self.positions]
    
    @property
    def long_positions(self) -> List[Position]:
        """All long positions."""
        return [p for p in self.positions if p.is_long]
    
    @property
    def short_positions(self) -> List[Position]:
        """All short positions."""
        return [p for p in self.positions if p.is_short]
    
    @property
    def gross_exposure(self) -> float:
        """Total gross exposure (sum of absolute values)."""
        return sum(p.abs_amount for p in self.positions)
    
    @property
    def net_exposure(self) -> float:
        """Net exposure (longs - shorts)."""
        return sum(p.amount for p in self.positions)
    
    @property
    def long_exposure(self) -> float:
        """Total long exposure."""
        return sum(p.amount for p in self.long_positions)
    
    @property
    def short_exposure(self) -> float:
        """Total short exposure (as positive number)."""
        return sum(p.abs_amount for p in self.short_positions)
    
    @property
    def long_weights(self) -> Dict[str, float]:
        """Long position weights (normalized to sum to 1)."""
        total = self.long_exposure
        if total == 0:
            return {}
        return {p.ticker: p.amount / total for p in self.long_positions}
    
    @property
    def short_weights(self) -> Dict[str, float]:
        """Short position weights (normalized to sum to 1, as positive values)."""
        total = self.short_exposure
        if total == 0:
            return {}
        return {p.ticker: p.abs_amount / total for p in self.short_positions}
    
    @property
    def portfolio_type(self) -> str:
        """Determine portfolio type."""
        if self.long_positions and self.short_positions:
            if abs(self.net_exposure) < self.gross_exposure * 0.1:
                return "MARKET_NEUTRAL"
            return "LONG_SHORT"
        elif self.long_positions:
            return "LONG_ONLY"
        elif self.short_positions:
            return "SHORT_ONLY"
        return "EMPTY"
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to {ticker: amount} dictionary."""
        return {p.ticker: p.amount for p in self.positions}


@dataclass
class PortfolioReturns:
    """Portfolio returns data."""
    daily_returns: pd.Series  # Daily returns in decimal form
    cumulative: pd.Series  # Cumulative returns (starts at 1.0)
    daily_pnl: pd.DataFrame  # P&L by position
    
    @property
    def total_return(self) -> float:
        """Total return over the period."""
        return self.cumulative.iloc[-1] - 1
    
    @property
    def trading_days(self) -> int:
        """Number of trading days."""
        return len(self.daily_returns)


@dataclass
class PeriodMetrics:
    """Performance metrics for a specific period."""
    period_days: int
    trading_days: int
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    best_day: float
    worst_day: float
    long_contribution: float
    short_contribution: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "period_days": self.period_days,
            "trading_days": self.trading_days,
            "total_return_pct": round(self.total_return * 100, 2),
            "annualized_return_pct": round(self.annualized_return * 100, 2),
            "volatility_pct": round(self.volatility * 100, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 3),
            "max_drawdown_pct": round(self.max_drawdown * 100, 2),
            "win_rate_pct": round(self.win_rate * 100, 1),
            "best_day_pct": round(self.best_day * 100, 2),
            "worst_day_pct": round(self.worst_day * 100, 2),
            "long_contribution_pct": round(self.long_contribution * 100, 2),
            "short_contribution_pct": round(self.short_contribution * 100, 2),
        }


@dataclass
class BacktestResult:
    """Complete backtest results."""
    portfolio: PortfolioDefinition
    metrics_by_period: Dict[str, PeriodMetrics]
    primary_period_days: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "primary_period_days": self.primary_period_days,
            "metrics": {k: v.to_dict() for k, v in self.metrics_by_period.items()},
        }


@dataclass
class RegimeSummary:
    """Summary of current market regime."""
    current_regime: str
    confidence: float
    risk_level: str
    trend_score: float
    mr_score: float
    z_score: float
    rsi: float
    adf_pvalue: float
    hurst_exponent: float
    halflife: float
    strategy: str
    strategic_signal: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "current_regime": self.current_regime,
            "confidence": round(self.confidence, 1),
            "risk_level": self.risk_level,
            "trend_score": round(self.trend_score, 2),
            "mr_score": round(self.mr_score, 1),
            "z_score": round(self.z_score, 2),
            "rsi": round(self.rsi, 1),
            "adf_pvalue": round(self.adf_pvalue, 4),
            "hurst_exponent": round(self.hurst_exponent, 4),
            "halflife": round(self.halflife, 1),
            "strategy": self.strategy,
            "strategic_signal": self.strategic_signal,
        }


# =============================================================================
# PORTFOLIO RETURNS CALCULATION
# =============================================================================

def build_portfolio_returns(
    stock_returns: pd.DataFrame,
    portfolio: PortfolioDefinition,
    analysis_days: int = 180,
) -> PortfolioReturns:
    """
    Build portfolio returns from stock returns.
    
    Parameters
    ----------
    stock_returns : pd.DataFrame
        Stock returns in PERCENTAGE form (1.5 = 1.5%)
    portfolio : PortfolioDefinition
        Portfolio definition
    analysis_days : int
        Number of days to analyze
        
    Returns
    -------
    PortfolioReturns
        Portfolio returns data
    """
    # Convert to decimal form
    stock_returns_decimal = stock_returns / 100.0
    
    # Get date range
    end_date = stock_returns_decimal.index.max()
    start_date = end_date - timedelta(days=analysis_days)
    
    mask = (stock_returns_decimal.index > start_date) & (stock_returns_decimal.index <= end_date)
    period_returns = stock_returns_decimal[mask].copy()
    
    # Calculate daily P&L for each position
    daily_pnl = pd.DataFrame(index=period_returns.index)
    portfolio_dict = portfolio.to_dict()
    
    for ticker, amount in portfolio_dict.items():
        if ticker in period_returns.columns:
            stock_ret = period_returns[ticker].fillna(0)
            daily_pnl[ticker] = amount * stock_ret
    
    # Total portfolio P&L and returns
    gross_exposure = portfolio.gross_exposure
    daily_pnl['TOTAL'] = daily_pnl[list(portfolio_dict.keys())].sum(axis=1)
    portfolio_daily_returns = daily_pnl['TOTAL'] / gross_exposure
    
    # Cumulative returns (rebased to 1.0)
    portfolio_cumulative = (1 + portfolio_daily_returns).cumprod()
    
    return PortfolioReturns(
        daily_returns=portfolio_daily_returns,
        cumulative=portfolio_cumulative,
        daily_pnl=daily_pnl,
    )


# =============================================================================
# BACKTESTING
# =============================================================================

def _calculate_period_metrics(
    portfolio_dict: Dict[str, float],
    stock_returns_decimal: pd.DataFrame,
    days_ago: int,
) -> Optional[PeriodMetrics]:
    """Calculate metrics for a single period."""
    
    end_date = stock_returns_decimal.index.max()
    start_date = end_date - timedelta(days=days_ago)
    
    mask = (stock_returns_decimal.index > start_date) & (stock_returns_decimal.index <= end_date)
    period_returns = stock_returns_decimal[mask].copy()
    
    if len(period_returns) == 0:
        return None
    
    # Calculate portfolio returns
    daily_pnl = pd.DataFrame(index=period_returns.index)
    
    for ticker, dollar_amount in portfolio_dict.items():
        if ticker in period_returns.columns:
            stock_ret = period_returns[ticker].fillna(0)
            daily_pnl[ticker] = dollar_amount * stock_ret
    
    total_daily_pnl = daily_pnl.sum(axis=1)
    gross_exposure = sum(abs(v) for v in portfolio_dict.values())
    port_returns = total_daily_pnl / gross_exposure
    
    # Calculate metrics
    cumulative = (1 + port_returns).cumprod()
    total_return = cumulative.iloc[-1] - 1
    
    trading_days = len(port_returns)
    years = trading_days / 252
    
    if years > 0:
        annualized_return = (1 + total_return) ** (1/years) - 1
    else:
        annualized_return = 0
    
    daily_vol = port_returns.std()
    volatility = daily_vol * np.sqrt(252)
    sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
    
    # Drawdown
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Other metrics
    win_rate = (port_returns > 0).sum() / len(port_returns) if len(port_returns) > 0 else 0
    best_day = port_returns.max()
    worst_day = port_returns.min()
    
    # Long/Short attribution
    long_positions = {k: v for k, v in portfolio_dict.items() if v > 0}
    short_positions = {k: v for k, v in portfolio_dict.items() if v < 0}
    
    long_contribution = 0
    if long_positions:
        long_pnl = daily_pnl[[k for k in long_positions.keys() if k in daily_pnl.columns]].sum(axis=1)
        long_gross = sum(long_positions.values())
        long_returns = long_pnl / long_gross
        long_cumulative = (1 + long_returns).cumprod()
        long_contribution = long_cumulative.iloc[-1] - 1
    
    short_contribution = 0
    if short_positions:
        short_pnl = daily_pnl[[k for k in short_positions.keys() if k in daily_pnl.columns]].sum(axis=1)
        short_gross = sum(abs(v) for v in short_positions.values())
        short_returns = short_pnl / short_gross
        short_cumulative = (1 + short_returns).cumprod()
        short_contribution = short_cumulative.iloc[-1] - 1
    
    return PeriodMetrics(
        period_days=days_ago,
        trading_days=trading_days,
        total_return=total_return,
        annualized_return=annualized_return,
        volatility=volatility,
        sharpe_ratio=sharpe_ratio,
        max_drawdown=max_drawdown,
        win_rate=win_rate,
        best_day=best_day,
        worst_day=worst_day,
        long_contribution=long_contribution,
        short_contribution=short_contribution,
    )


def run_full_backtest(
    stock_returns: pd.DataFrame,
    portfolio: PortfolioDefinition,
    primary_period: int = 180,
    additional_periods: List[int] = None,
) -> BacktestResult:
    """
    Run complete portfolio backtest for multiple periods.
    
    Parameters
    ----------
    stock_returns : pd.DataFrame
        Stock returns in PERCENTAGE form
    portfolio : PortfolioDefinition
        Portfolio definition
    primary_period : int
        Primary analysis period in days
    additional_periods : List[int]
        Additional periods to analyze
        
    Returns
    -------
    BacktestResult
        Complete backtest results
    """
    if additional_periods is None:
        additional_periods = [14, 30]
    
    # Convert to decimal
    stock_returns_decimal = stock_returns / 100.0
    stock_returns_decimal = stock_returns_decimal.clip(lower=-0.5, upper=0.5)
    
    portfolio_dict = portfolio.to_dict()
    
    # Calculate metrics for all periods
    all_periods = sorted(set([primary_period] + additional_periods))
    metrics_by_period = {}
    
    for days in all_periods:
        metrics = _calculate_period_metrics(portfolio_dict, stock_returns_decimal, days)
        if metrics:
            period_label = f"{days}_days"
            metrics_by_period[period_label] = metrics
    
    return BacktestResult(
        portfolio=portfolio,
        metrics_by_period=metrics_by_period,
        primary_period_days=primary_period,
    )


# =============================================================================
# TECHNICAL INDICATORS
# =============================================================================

def calculate_technical_indicators(cumulative_values: pd.Series) -> Dict[str, pd.Series]:
    """
    Calculate technical indicators for the portfolio.
    
    Matches the notebook implementation exactly.
    """
    close = cumulative_values.copy()
    
    # EMAs
    ema50 = close.ewm(span=50, adjust=False).mean()
    ema100 = close.ewm(span=100, adjust=False).mean()
    
    # RSI (14-period)
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    avg_gain = gain.ewm(span=14, adjust=False).mean()
    avg_loss = loss.ewm(span=14, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.replace([np.inf, -np.inf], np.nan)
    
    # MACD (12, 26, 9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    macd_histogram = macd_line - macd_signal
    
    # Bollinger Bands (20-period, 2 std)
    bb_middle = close.rolling(window=20).mean()
    bb_std = close.rolling(window=20).std()
    bb_upper = bb_middle + (2 * bb_std)
    bb_lower = bb_middle - (2 * bb_std)
    
    return {
        'ema50': ema50,
        'ema100': ema100,
        'rsi': rsi,
        'macd_line': macd_line,
        'macd_signal': macd_signal,
        'macd_histogram': macd_histogram,
        'bb_upper': bb_upper,
        'bb_middle': bb_middle,
        'bb_lower': bb_lower,
    }


# =============================================================================
# REGIME DETECTION
# =============================================================================

def _run_rolling_regime_analysis(
    cumulative_values: pd.Series,
    window: int = 60,
) -> pd.DataFrame:
    """
    Run MarketRegimeDetector on rolling windows to get regime history.
    """
    detector = MarketRegimeDetector(
        lookback_periods=window,
        min_periods=20,
        volatility_window=12,
        trend_window=12,
        asset_type="equity"
    )
    
    results = []
    prices = cumulative_values.values
    dummy_volume = np.ones(len(prices)) * 1000000
    
    for i in range(window, len(prices)):
        price_window = prices[i-window:i+1]
        vol_window = dummy_volume[i-window:i+1]
        
        analysis = detector.analyze(price_window, vol_window)
        
        results.append({
            'date': cumulative_values.index[i],
            'regime': analysis.primary_regime,
            'confidence': analysis.confidence,
            'trend_score': analysis.metrics.trend_score,
            'mean_reversion_score': analysis.metrics.mean_reversion_score,
            'breakout_probability': min(100, analysis.metrics.breakout_probability),
            'hurst': analysis.metrics.hurst_exponent,
            'adf_pvalue': analysis.metrics.adf_pvalue,
            'halflife': analysis.metrics.halflife_periods,
            'momentum': analysis.metrics.price_momentum,
            'volatility': analysis.metrics.realized_volatility,
            'range_position': analysis.metrics.range_position,
            'risk_level': analysis.risk_level,
            'strategy': analysis.suggested_strategy,
            'transition_prob': min(100, analysis.transition_probability),
        })
    
    return pd.DataFrame(results).set_index('date')


def _calculate_strategic_adf_overlay(regime_df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate strategic ADF overlay to show long-term mean-reversion character.
    """
    adf_values = regime_df['adf_pvalue'].values
    
    # Rolling average of ADF to smooth out noise
    adf_smooth = pd.Series(adf_values, index=regime_df.index).rolling(window=10, min_periods=5).mean()
    
    # Strategic assessment
    strategic_signal = pd.Series(index=regime_df.index, dtype='object')
    
    for date in regime_df.index:
        adf = adf_smooth.loc[date] if date in adf_smooth.index else regime_df.loc[date, 'adf_pvalue']
        
        if pd.isna(adf):
            strategic_signal.loc[date] = 'INSUFFICIENT_DATA'
        elif adf < 0.05:
            strategic_signal.loc[date] = 'STRONG_MEAN_REVERTING'
        elif adf < 0.20:
            strategic_signal.loc[date] = 'MEAN_REVERTING'
        elif adf < 0.50:
            strategic_signal.loc[date] = 'WEAKLY_MEAN_REVERTING'
        else:
            strategic_signal.loc[date] = 'TRENDING'
    
    return adf_smooth, strategic_signal


def detect_regime(
    cumulative: pd.Series,
    daily_returns: pd.Series,
    window: int = 60,
) -> Tuple[pd.DataFrame, RegimeSummary, Dict[str, pd.Series], pd.Series]:
    """
    Run complete regime detection analysis.
    
    Parameters
    ----------
    cumulative : pd.Series
        Cumulative portfolio values
    daily_returns : pd.Series
        Daily returns
    window : int
        Rolling window for regime detection
        
    Returns
    -------
    Tuple containing:
        - regime_df: DataFrame with regime history
        - regime_summary: RegimeSummary object
        - indicators: Dict of technical indicators
        - z_score: Z-score series
    """
    logger.info(f"Running regime detection with {window}-day window")
    
    # Calculate technical indicators
    indicators = calculate_technical_indicators(cumulative)
    
    # Run rolling regime analysis
    regime_df = _run_rolling_regime_analysis(cumulative, window=window)
    
    # Calculate fair value and z-score
    rolling_mean = cumulative.rolling(window=window).mean()
    rolling_std = cumulative.rolling(window=window).std()
    z_score = (cumulative - rolling_mean) / rolling_std
    
    # Calculate strategic ADF overlay
    adf_smooth, strategic_signal = _calculate_strategic_adf_overlay(regime_df)
    regime_df['adf_smooth'] = adf_smooth
    regime_df['strategic_signal'] = strategic_signal
    
    # Build summary
    latest = regime_df.iloc[-1]
    latest_z = z_score.iloc[-1] if not pd.isna(z_score.iloc[-1]) else 0
    latest_rsi = indicators['rsi'].iloc[-1] if not pd.isna(indicators['rsi'].iloc[-1]) else 50
    
    # Get regime value (handle both enum and string)
    regime_value = latest['regime'].value if hasattr(latest['regime'], 'value') else str(latest['regime'])
    
    regime_summary = RegimeSummary(
        current_regime=regime_value,
        confidence=latest['confidence'],
        risk_level=latest['risk_level'],
        trend_score=latest['trend_score'],
        mr_score=latest['mean_reversion_score'],
        z_score=latest_z,
        rsi=latest_rsi,
        adf_pvalue=latest['adf_pvalue'],
        hurst_exponent=latest['hurst'],
        halflife=latest['halflife'],
        strategy=latest['strategy'],
        strategic_signal=latest['strategic_signal'],
    )
    
    return regime_df, regime_summary, indicators, z_score


# =============================================================================
# TRADING SIGNALS
# =============================================================================

def generate_trading_signals(
    regime_df: pd.DataFrame,
    indicators: Dict[str, pd.Series],
    cumulative: pd.Series,
    z_score: pd.Series,
) -> pd.Series:
    """
    Generate BUY/SELL/HOLD signals based on regime and indicators.
    
    Implements the 5-mode ADF-based signal logic from the notebook.
    """
    signals = pd.Series(index=regime_df.index, dtype='object')
    signals[:] = 'HOLD'
    
    # Calculate rolling mean/std for z-score
    window = 60
    rolling_mean = cumulative.rolling(window=window).mean()
    rolling_std = cumulative.rolling(window=window).std()
    
    # Track previous values
    macd_hist_prev = indicators['macd_histogram'].shift(1)
    regime_prev = regime_df['regime'].shift(1)
    trend_score_prev = regime_df['trend_score'].shift(1)
    
    bullish_regimes = ['trending_up', 'breakout', 'accumulation']
    bearish_regimes = ['trending_down', 'breakdown', 'distribution']
    
    ema50 = indicators['ema50']
    
    for i, (date, row) in enumerate(regime_df.iterrows()):
        # Skip if indicators not available
        if date not in indicators['macd_histogram'].index:
            continue
        if date not in indicators['rsi'].index:
            continue
        if date not in cumulative.index:
            continue
        
        # Gather data
        adf_pvalue = row['adf_pvalue']
        trend_score = row['trend_score']
        mr_score = row['mean_reversion_score']
        current_regime = row['regime'].value if hasattr(row['regime'], 'value') else str(row['regime'])
        
        # Previous values
        prev_regime = None
        if date in regime_prev.index and pd.notna(regime_prev.loc[date]):
            prev_reg = regime_prev.loc[date]
            prev_regime = prev_reg.value if hasattr(prev_reg, 'value') else str(prev_reg)
        
        prev_trend_score = trend_score_prev.loc[date] if date in trend_score_prev.index else None
        
        # Technical indicators
        rsi = indicators['rsi'].loc[date]
        macd_hist = indicators['macd_histogram'].loc[date]
        macd_prev = macd_hist_prev.loc[date] if date in macd_hist_prev.index else 0
        
        # Price vs EMA50
        price = cumulative.loc[date]
        ema50_val = ema50.loc[date] if date in ema50.index else None
        price_above_ema50 = price > ema50_val if ema50_val is not None else None
        
        # Z-score
        fair_val = rolling_mean.loc[date] if date in rolling_mean.index else None
        std_val = rolling_std.loc[date] if date in rolling_std.index else None
        
        if fair_val is not None and std_val is not None and std_val > 0:
            z = (price - fair_val) / std_val
        else:
            z = 0
        
        # Skip if NaN
        if pd.isna(rsi) or pd.isna(macd_hist) or pd.isna(adf_pvalue):
            continue
        
        # MACD direction
        macd_turning_up = macd_hist > macd_prev if not pd.isna(macd_prev) else False
        macd_turning_down = macd_hist < macd_prev if not pd.isna(macd_prev) else False
        macd_positive = macd_hist > 0
        macd_negative = macd_hist < 0
        
        # Regime transition detection
        transition_buy_bias = False
        transition_sell_bias = False
        
        if prev_regime is not None and prev_regime != current_regime:
            if current_regime in bullish_regimes and prev_regime not in bullish_regimes:
                transition_buy_bias = True
            if current_regime in bearish_regimes and prev_regime not in bearish_regimes:
                transition_sell_bias = True
        
        # Trend score flip detection
        trend_flipped_bearish = False
        trend_flipped_bullish = False
        
        if prev_trend_score is not None and not pd.isna(prev_trend_score):
            if prev_trend_score > 0 and trend_score < 0:
                trend_flipped_bearish = True
            if prev_trend_score < 0 and trend_score > 0:
                trend_flipped_bullish = True
        
        # Trend adjustment factors
        trend_buy_adj = 0.3 if trend_score > 0.5 else (0.15 if trend_score > 0 else 0)
        trend_sell_adj = 0.3 if trend_score < -0.5 else (0.15 if trend_score < 0 else 0)
        
        # MODE 5: PURE TREND (ADF > 0.85)
        if adf_pvalue > 0.85:
            if trend_score > 0:
                if price_above_ema50 and macd_turning_up and rsi < 60:
                    signals.loc[date] = 'BUY'
                if transition_buy_bias:
                    signals.loc[date] = 'BUY'
                if trend_flipped_bearish:
                    signals.loc[date] = 'SELL'
                elif transition_sell_bias:
                    signals.loc[date] = 'SELL'
                elif price_above_ema50 == False and macd_negative and macd_turning_down:
                    signals.loc[date] = 'SELL'
            elif trend_score < 0:
                if price_above_ema50 == False and macd_turning_down and rsi > 40:
                    signals.loc[date] = 'SELL'
                if transition_sell_bias:
                    signals.loc[date] = 'SELL'
                if trend_flipped_bullish:
                    signals.loc[date] = 'BUY'
                elif transition_buy_bias:
                    signals.loc[date] = 'BUY'
                elif price_above_ema50 and macd_positive and macd_turning_up:
                    signals.loc[date] = 'BUY'
            else:
                if transition_buy_bias and macd_turning_up:
                    signals.loc[date] = 'BUY'
                elif transition_sell_bias and macd_turning_down:
                    signals.loc[date] = 'SELL'
        
        # MODE 4: TREND FOLLOWING (ADF 0.60 - 0.85)
        elif adf_pvalue > 0.60:
            if trend_score > 0.5:
                if rsi < 50 and (macd_positive or macd_turning_up):
                    signals.loc[date] = 'BUY'
                if trend_flipped_bearish or transition_sell_bias or \
                   (price_above_ema50 == False and macd_turning_down and rsi > 60):
                    signals.loc[date] = 'SELL'
            elif trend_score < -0.5:
                if rsi > 50 and (macd_negative or macd_turning_down):
                    signals.loc[date] = 'SELL'
                if trend_flipped_bullish or transition_buy_bias or \
                   (price_above_ema50 and macd_turning_up and rsi < 40):
                    signals.loc[date] = 'BUY'
            else:
                if rsi < 30 and macd_turning_up:
                    signals.loc[date] = 'BUY'
                elif rsi > 70 and macd_turning_down:
                    signals.loc[date] = 'SELL'
                elif transition_buy_bias and macd_turning_up:
                    signals.loc[date] = 'BUY'
                elif transition_sell_bias and macd_turning_down:
                    signals.loc[date] = 'SELL'
        
        # MODE 3: MIXED MODE (ADF 0.30 - 0.60)
        elif adf_pvalue > 0.30:
            if trend_score > 0.5:
                if rsi < 50 and (macd_turning_up or macd_positive) and z < 0.5:
                    signals.loc[date] = 'BUY'
                elif rsi > 70 and z > 1.0 and macd_turning_down:
                    signals.loc[date] = 'SELL'
            elif trend_score < -0.5:
                if rsi > 50 and (macd_turning_down or macd_negative) and z > -0.5:
                    signals.loc[date] = 'SELL'
                elif rsi < 30 and z < -1.0 and macd_turning_up:
                    signals.loc[date] = 'BUY'
            else:
                if rsi < 30 and macd_turning_up and z < -0.5:
                    signals.loc[date] = 'BUY'
                elif rsi > 70 and macd_turning_down and z > 0.5:
                    signals.loc[date] = 'SELL'
                elif transition_buy_bias and rsi < 45 and macd_turning_up:
                    signals.loc[date] = 'BUY'
                elif transition_sell_bias and rsi > 55 and macd_turning_down:
                    signals.loc[date] = 'SELL'
        
        # MODE 2: MEAN REVERSION (ADF 0.10 - 0.30)
        elif adf_pvalue > 0.10:
            z_buy_thresh = -1.0 + trend_buy_adj
            z_sell_thresh = 1.0 - trend_sell_adj
            
            if z < z_buy_thresh and rsi < 40 and mr_score > 40 and \
               (macd_turning_up or trend_score > 0 or transition_buy_bias):
                signals.loc[date] = 'BUY'
            elif z > z_sell_thresh and rsi > 60 and \
                 (macd_turning_down or trend_score < 0 or transition_sell_bias):
                signals.loc[date] = 'SELL'
        
        # MODE 1: STRONG MEAN REVERSION (ADF < 0.10)
        else:
            z_buy_thresh = -0.8 + trend_buy_adj
            z_sell_thresh = 0.8 - trend_sell_adj
            
            if z < z_buy_thresh and rsi < 45 and \
               (macd_turning_up or macd_positive or transition_buy_bias):
                signals.loc[date] = 'BUY'
            elif z > z_sell_thresh and rsi > 55 and \
                 (macd_turning_down or macd_negative or transition_sell_bias):
                signals.loc[date] = 'SELL'
    
    return signals


# =============================================================================
# CHART GENERATION
# =============================================================================

def _fig_to_base64(fig: plt.Figure) -> str:
    """Convert matplotlib figure to base64 string."""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', 
                facecolor='#e8f4fc', edgecolor='none')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return img_base64


def _create_regime_chart(
    cumulative: pd.Series,
    regime_df: pd.DataFrame,
    indicators: Dict[str, pd.Series],
    z_score: pd.Series,
    signals: pd.Series,
    portfolio_name: str,
) -> str:
    """Create the 6-panel regime analysis chart."""
    
    fig, axes = plt.subplots(6, 1, figsize=(18, 24))
    fig.patch.set_facecolor('#e8f4fc')
    for ax in axes:
        ax.set_facecolor('#e8f4fc')
    plt.subplots_adjust(hspace=0.35)
    
    # Calculate rolling mean for fair value
    window = 60
    rolling_mean = cumulative.rolling(window=window).mean()
    
    # Regime colors
    regime_colors = {
        MarketRegime.RANGING: 'lightgreen',
        MarketRegime.TRENDING_UP: 'green',
        MarketRegime.TRENDING_DOWN: 'red',
        MarketRegime.BREAKOUT: 'lime',
        MarketRegime.BREAKDOWN: 'salmon',
        MarketRegime.ACCUMULATION: 'lightblue',
        MarketRegime.DISTRIBUTION: 'orange',
        MarketRegime.VOLATILE_EXPANSION: 'yellow',
        MarketRegime.VOLATILE_CONTRACTION: 'lightyellow',
        MarketRegime.UNCERTAIN: 'lightgray',
        MarketRegime.UNKNOWN: 'white',
    }
    
    def add_regime_shading(ax, regime_series):
        prev_regime = None
        start_date = None
        
        for date, regime in regime_series.items():
            if regime != prev_regime:
                if prev_regime is not None and start_date is not None:
                    color = regime_colors.get(prev_regime, 'white')
                    ax.axvspan(start_date, date, alpha=0.3, color=color)
                prev_regime = regime
                start_date = date
        
        if prev_regime is not None and start_date is not None:
            color = regime_colors.get(prev_regime, 'white')
            ax.axvspan(start_date, regime_series.index[-1], alpha=0.3, color=color)
    
    # Panel 1: Price with EMAs, Bollinger Bands, Fair Value, Signals
    ax1 = axes[0]
    add_regime_shading(ax1, regime_df['regime'])
    
    ax1.fill_between(cumulative.index, indicators['bb_lower'], indicators['bb_upper'],
                     alpha=0.15, color='blue', label='BB (20,2)')
    ax1.plot(indicators['bb_middle'], 'b--', alpha=0.4, linewidth=1)
    ax1.plot(cumulative.index, cumulative.values, 'k-', linewidth=1.5, label='Portfolio')
    ax1.plot(indicators['ema50'], 'orange', linewidth=1.2, label='EMA50', alpha=0.8)
    ax1.plot(indicators['ema100'], 'purple', linewidth=1.2, label='EMA100', alpha=0.8)
    ax1.plot(rolling_mean, 'g-', linewidth=1.5, label='Fair Value', alpha=0.9)
    
    # BUY/SELL signals
    buy_dates = signals[signals == 'BUY'].index
    sell_dates = signals[signals == 'SELL'].index
    if len(buy_dates) > 0:
        ax1.scatter(buy_dates, cumulative.reindex(buy_dates), marker='^',
                   color='lime', s=150, label='BUY', zorder=5, edgecolors='darkgreen', linewidths=2)
    if len(sell_dates) > 0:
        ax1.scatter(sell_dates, cumulative.reindex(sell_dates), marker='v',
                   color='red', s=150, label='SELL', zorder=5, edgecolors='darkred', linewidths=2)
    
    ax1.set_ylabel('Value', fontsize=11)
    ax1.set_title(f'{portfolio_name}: Price, Indicators & Regime', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=8, ncol=4)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=1, color='black', linestyle='--', alpha=0.3)
    
    # Panel 2: Z-Score
    ax2 = axes[1]
    add_regime_shading(ax2, regime_df['regime'])
    ax2.plot(z_score.index, z_score.values, 'b-', linewidth=1.5, label='Z-Score')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax2.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Overbought')
    ax2.axhline(y=-1, color='green', linestyle='--', alpha=0.5, label='Oversold')
    ax2.axhline(y=2, color='darkred', linestyle=':', alpha=0.5)
    ax2.axhline(y=-2, color='darkgreen', linestyle=':', alpha=0.5)
    ax2.fill_between(z_score.index, z_score.values, 0, where=z_score > 0, alpha=0.3, color='red')
    ax2.fill_between(z_score.index, z_score.values, 0, where=z_score < 0, alpha=0.3, color='green')
    ax2.set_ylabel('Z-Score (σ)', fontsize=11)
    ax2.set_title('Z-Score from Fair Value', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-3, 3)
    
    # Panel 3: RSI
    ax3 = axes[2]
    add_regime_shading(ax3, regime_df['regime'])
    ax3.plot(indicators['rsi'].index, indicators['rsi'].values, 'purple', linewidth=1.5, label='RSI(14)')
    ax3.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Overbought (70)')
    ax3.axhline(y=30, color='green', linestyle='--', alpha=0.7, label='Oversold (30)')
    ax3.axhline(y=50, color='gray', linestyle='-', alpha=0.5)
    ax3.fill_between(indicators['rsi'].index, indicators['rsi'].values, 70,
                     where=indicators['rsi'] > 70, alpha=0.3, color='red')
    ax3.fill_between(indicators['rsi'].index, indicators['rsi'].values, 30,
                     where=indicators['rsi'] < 30, alpha=0.3, color='green')
    ax3.set_ylabel('RSI', fontsize=11)
    ax3.set_title('RSI (14)', fontsize=13, fontweight='bold')
    ax3.legend(loc='upper left', fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 100)
    
    # Panel 4: MACD
    ax4 = axes[3]
    add_regime_shading(ax4, regime_df['regime'])
    ax4.plot(indicators['macd_line'].index, indicators['macd_line'].values, 'b-', linewidth=1.5, label='MACD')
    ax4.plot(indicators['macd_signal'].index, indicators['macd_signal'].values, 'r-', linewidth=1.2, label='Signal')
    colors = ['green' if x >= 0 else 'red' for x in indicators['macd_histogram'].values]
    ax4.bar(indicators['macd_histogram'].index, indicators['macd_histogram'].values, color=colors, alpha=0.5, width=1)
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax4.set_ylabel('MACD', fontsize=11)
    ax4.set_title('MACD (12, 26, 9)', fontsize=13, fontweight='bold')
    ax4.legend(loc='upper left', fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    # Panel 5: Trend Score
    ax5 = axes[4]
    add_regime_shading(ax5, regime_df['regime'])
    ax5.plot(regime_df.index, regime_df['trend_score'].values, 'b-', linewidth=1.5, label='Trend Score')
    ax5.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax5.fill_between(regime_df.index, regime_df['trend_score'].values, 0,
                     where=regime_df['trend_score'] > 0, alpha=0.3, color='green')
    ax5.fill_between(regime_df.index, regime_df['trend_score'].values, 0,
                     where=regime_df['trend_score'] < 0, alpha=0.3, color='red')
    ax5.set_ylabel('Trend Score', fontsize=11)
    ax5.set_title('Trend Score', fontsize=13, fontweight='bold')
    ax5.legend(loc='upper left', fontsize=9)
    ax5.grid(True, alpha=0.3)
    
    # Panel 6: ADF p-value (Mean Reversion Strength)
    ax6 = axes[5]
    add_regime_shading(ax6, regime_df['regime'])
    ax6.plot(regime_df.index, regime_df['adf_pvalue'].values, 'purple', linewidth=1.5, label='ADF p-value')
    if 'adf_smooth' in regime_df.columns:
        ax6.plot(regime_df.index, regime_df['adf_smooth'].values, 'orange', linewidth=2, label='ADF Smoothed', alpha=0.8)
    ax6.axhline(y=0.05, color='green', linestyle='--', alpha=0.7, label='Strong MR (0.05)')
    ax6.axhline(y=0.20, color='blue', linestyle='--', alpha=0.7, label='MR Valid (0.20)')
    ax6.axhline(y=0.50, color='orange', linestyle='--', alpha=0.7, label='Weak MR (0.50)')
    ax6.fill_between(regime_df.index, regime_df['adf_pvalue'].values, 0.05,
                     where=regime_df['adf_pvalue'] < 0.05, alpha=0.3, color='green')
    ax6.fill_between(regime_df.index, regime_df['adf_pvalue'].values, 0.50,
                     where=regime_df['adf_pvalue'] > 0.50, alpha=0.3, color='red')
    ax6.set_ylabel('ADF p-value', fontsize=11)
    ax6.set_xlabel('Date', fontsize=11)
    ax6.set_title('ADF Test (Mean Reversion Validation)', fontsize=13, fontweight='bold')
    ax6.legend(loc='upper left', fontsize=9)
    ax6.grid(True, alpha=0.3)
    ax6.set_ylim(0, 1)
    
    plt.tight_layout()
    
    result = _fig_to_base64(fig)
    plt.close(fig)
    return result


def _create_performance_chart(
    portfolio_cumulative: pd.Series,
    portfolio_returns: pd.Series,
) -> str:
    """Create the 3-panel performance chart."""
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [2, 1, 1]})
    fig.patch.set_facecolor('#e8f4fc')
    for ax in axes:
        ax.set_facecolor('#e8f4fc')
    plt.subplots_adjust(hspace=0.3)
    
    # Panel 1: Cumulative Performance
    ax1 = axes[0]
    running_max = portfolio_cumulative.cummax()
    
    ax1.fill_between(portfolio_cumulative.index, portfolio_cumulative, running_max,
                     where=portfolio_cumulative < running_max, alpha=0.3, color='red')
    ax1.plot(portfolio_cumulative.index, portfolio_cumulative.values, 'b-', linewidth=1.5, label='Portfolio Value')
    ax1.plot(running_max.index, running_max.values, 'g--', linewidth=1, alpha=0.7, label='Running Maximum')
    ax1.axhline(y=1.0, color='black', linestyle='--', alpha=0.3)
    ax1.set_ylabel('Cumulative Value', fontsize=11)
    ax1.set_title('Portfolio Performance', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Daily Returns
    ax2 = axes[1]
    colors = ['green' if r >= 0 else 'red' for r in portfolio_returns.values]
    ax2.bar(portfolio_returns.index, portfolio_returns.values * 100, color=colors, alpha=0.7, width=1)
    ax2.axhline(y=0, color='black', linewidth=0.5)
    ax2.set_ylabel('Daily Return %', fontsize=11)
    ax2.set_title('Daily Returns', fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Drawdown
    ax3 = axes[2]
    drawdown = (portfolio_cumulative / running_max - 1) * 100
    ax3.fill_between(drawdown.index, drawdown.values, 0, alpha=0.5, color='red')
    ax3.plot(drawdown.index, drawdown.values, 'r-', linewidth=1)
    max_dd = drawdown.min()
    ax3.axhline(y=max_dd, color='darkred', linestyle='--', alpha=0.7, label=f'Max DD: {max_dd:.1f}%')
    ax3.set_ylabel('Drawdown %', fontsize=11)
    ax3.set_xlabel('Date', fontsize=11)
    ax3.set_title('Drawdown from Peak', fontsize=11)
    ax3.legend(loc='lower left', fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    result = _fig_to_base64(fig)
    plt.close(fig)
    return result


def _create_distribution_chart(portfolio_returns: pd.Series) -> str:
    """Create the returns distribution histogram."""
    
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor('#e8f4fc')
    ax.set_facecolor('#e8f4fc')
    
    returns_pct = portfolio_returns * 100
    n, bins, patches = ax.hist(returns_pct.dropna(), bins=30, color='steelblue',
                                edgecolor='white', alpha=0.8)
    
    # Color negative returns red
    for patch, left_edge in zip(patches, bins[:-1]):
        if left_edge < 0:
            patch.set_facecolor('indianred')
    
    ax.axvline(x=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.axvline(x=returns_pct.mean(), color='green', linestyle='-', linewidth=2,
               alpha=0.8, label=f'Mean: {returns_pct.mean():.3f}%')
    
    ax.set_xlabel('Daily Return (%)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Daily Returns Distribution', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add stats box
    stats_text = f'Mean: {returns_pct.mean():.3f}%\nStd: {returns_pct.std():.3f}%'
    ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    result = _fig_to_base64(fig)
    plt.close(fig)
    return result


def create_all_charts(
    cumulative: pd.Series,
    daily_returns: pd.Series,
    regime_df: pd.DataFrame,
    indicators: Dict[str, pd.Series],
    z_score: pd.Series,
    signals: pd.Series,
    portfolio_name: str = "Portfolio",
) -> Dict[str, str]:
    """
    Create all charts for the HTML report.
    
    Returns
    -------
    Dict[str, str]
        Dictionary mapping chart name to base64-encoded PNG
    """
    logger.info("Generating charts...")
    
    charts = {}
    
    # Regime analysis chart
    try:
        charts['regime'] = _create_regime_chart(
            cumulative, regime_df, indicators, z_score, signals, portfolio_name
        )
        logger.info("  - Regime chart generated")
    except Exception as e:
        logger.error(f"Failed to create regime chart: {e}")
        charts['regime'] = ""
    
    # Performance chart
    try:
        charts['performance'] = _create_performance_chart(cumulative, daily_returns)
        logger.info("  - Performance chart generated")
    except Exception as e:
        logger.error(f"Failed to create performance chart: {e}")
        charts['performance'] = ""
    
    # Distribution chart
    try:
        charts['distribution'] = _create_distribution_chart(daily_returns)
        logger.info("  - Distribution chart generated")
    except Exception as e:
        logger.error(f"Failed to create distribution chart: {e}")
        charts['distribution'] = ""
    
    return charts
