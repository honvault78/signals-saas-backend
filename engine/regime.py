"""
Regime Detection Module - Market Regime Analysis

Uses the professional MarketRegimeDetector class for institutional-grade regime detection.
Includes the full 5-mode signal generation logic for tactical trading.

Signal Modes (based on ADF p-value):
- Mode 1: STRONG MEAN REVERSION (ADF < 0.10) - Trade MR confidently
- Mode 2: MEAN REVERSION (ADF 0.10-0.30) - MR with trend adjustment
- Mode 3: MIXED MODE (ADF 0.30-0.60) - Blend MR and trend signals
- Mode 4: TREND FOLLOWING (ADF 0.60-0.85) - Follow trend, relaxed MR
- Mode 5: PURE TREND (ADF > 0.85) - Ignore RSI/Z-score, follow structure only
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Import the user's MarketRegimeDetector
from .market_regime_detector import MarketRegimeDetector, MarketRegime, RegimeAnalysis

logger = logging.getLogger(__name__)


class StrategicSignal(Enum):
    """Strategic assessment based on ADF."""
    STRONG_MEAN_REVERTING = "strong_mean_reverting"  # ADF < 0.05
    MEAN_REVERTING = "mean_reverting"  # ADF 0.05-0.20
    WEAKLY_MEAN_REVERTING = "weakly_mean_reverting"  # ADF 0.20-0.50
    TRENDING = "trending"  # ADF > 0.50


@dataclass
class TechnicalIndicators:
    """Technical indicators for the portfolio."""
    # EMAs
    ema50: pd.Series
    ema100: pd.Series
    
    # RSI
    rsi: pd.Series
    
    # MACD
    macd_line: pd.Series
    macd_signal: pd.Series
    macd_histogram: pd.Series
    
    # Bollinger Bands
    bb_upper: pd.Series
    bb_middle: pd.Series
    bb_lower: pd.Series


@dataclass 
class RegimeSummary:
    """Summary of current market regime and recommended action."""
    current_regime: str
    confidence: float
    risk_level: str
    strategic_signal: str
    trend_score: float
    mean_reversion_score: float
    breakout_probability: float
    hurst_exponent: float
    adf_pvalue: float
    adf_smoothed: float
    halflife: float
    z_score: float
    rsi: float
    strategy: str
    transition_probability: float
    
    def to_dict(self) -> Dict:
        return {
            "current_regime": self.current_regime,
            "confidence": round(self.confidence, 1),
            "risk_level": self.risk_level,
            "strategic_signal": self.strategic_signal,
            "trend_score": round(self.trend_score, 2),
            "mean_reversion_score": round(self.mean_reversion_score, 1),
            "breakout_probability": round(min(100, self.breakout_probability), 1),
            "hurst_exponent": round(self.hurst_exponent, 4),
            "adf_pvalue": round(self.adf_pvalue, 4),
            "adf_smoothed": round(self.adf_smoothed, 4),
            "halflife": round(self.halflife, 1),
            "z_score": round(self.z_score, 2),
            "rsi": round(self.rsi, 1),
            "strategy": self.strategy,
            "transition_probability": round(min(100, self.transition_probability), 1),
        }


def calculate_technical_indicators(cumulative: pd.Series) -> TechnicalIndicators:
    """
    Calculate all technical indicators for the portfolio cumulative returns.
    
    Parameters
    ----------
    cumulative : pd.Series
        Cumulative portfolio value (starts at 1.0)
        
    Returns
    -------
    TechnicalIndicators
        All calculated indicators.
    """
    close = cumulative.copy()
    
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
    
    return TechnicalIndicators(
        ema50=ema50,
        ema100=ema100,
        rsi=rsi,
        macd_line=macd_line,
        macd_signal=macd_signal,
        macd_histogram=macd_histogram,
        bb_upper=bb_upper,
        bb_middle=bb_middle,
        bb_lower=bb_lower,
    )


def run_rolling_regime_analysis(
    cumulative: pd.Series,
    window: int = 60
) -> pd.DataFrame:
    """
    Run MarketRegimeDetector on rolling windows to get regime history.
    
    Uses the user's MarketRegimeDetector class.
    """
    detector = MarketRegimeDetector(
        lookback_periods=window,
        min_periods=20,
        volatility_window=12,
        trend_window=12,
        asset_type="equity"  # Use equity mode for portfolio analysis
    )
    
    results = []
    prices = cumulative.values
    dummy_volume = np.ones(len(prices)) * 1000000  # Volume not critical for spreads
    
    for i in range(window, len(prices)):
        price_window = prices[i-window:i+1]
        vol_window = dummy_volume[i-window:i+1]
        
        analysis = detector.analyze(price_window, vol_window)
        
        results.append({
            'date': cumulative.index[i],
            'regime': analysis.primary_regime.value,
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
            'regime_strength': analysis.regime_strength,
        })
    
    df = pd.DataFrame(results).set_index('date')
    
    # Add smoothed ADF
    df['adf_smooth'] = df['adf_pvalue'].rolling(window=10, min_periods=3).mean()
    
    return df


def detect_regime(
    cumulative: pd.Series,
    daily_returns: pd.Series,
    window: int = 60
) -> Tuple[pd.DataFrame, RegimeSummary, TechnicalIndicators, pd.Series]:
    """
    Run complete regime detection analysis.
    
    Parameters
    ----------
    cumulative : pd.Series
        Cumulative portfolio value (starts at 1.0)
    daily_returns : pd.Series
        Daily returns (decimal form)
    window : int
        Rolling window for analysis
        
    Returns
    -------
    Tuple[pd.DataFrame, RegimeSummary, TechnicalIndicators, pd.Series]
        (regime_history, current_summary, indicators, z_score)
    """
    # Calculate technical indicators
    indicators = calculate_technical_indicators(cumulative)
    
    # Rolling fair value (mean) for Z-score
    rolling_mean = cumulative.rolling(window=window).mean()
    rolling_std = cumulative.rolling(window=window).std()
    
    # Z-score from fair value
    z_score = (cumulative - rolling_mean) / rolling_std
    z_score = z_score.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Run rolling regime analysis using MarketRegimeDetector
    regime_df = run_rolling_regime_analysis(cumulative, window)
    
    # Get latest values
    latest = regime_df.iloc[-1]
    latest_rsi = indicators.rsi.iloc[-1] if not pd.isna(indicators.rsi.iloc[-1]) else 50
    latest_z = z_score.iloc[-1] if not pd.isna(z_score.iloc[-1]) else 0
    
    # Strategic signal based on smoothed ADF
    adf_smooth = latest['adf_smooth'] if not pd.isna(latest['adf_smooth']) else latest['adf_pvalue']
    if adf_smooth < 0.05:
        strategic = StrategicSignal.STRONG_MEAN_REVERTING.value
    elif adf_smooth < 0.20:
        strategic = StrategicSignal.MEAN_REVERTING.value
    elif adf_smooth < 0.50:
        strategic = StrategicSignal.WEAKLY_MEAN_REVERTING.value
    else:
        strategic = StrategicSignal.TRENDING.value
    
    summary = RegimeSummary(
        current_regime=latest['regime'],
        confidence=latest['confidence'],
        risk_level=latest['risk_level'],
        strategic_signal=strategic,
        trend_score=latest['trend_score'],
        mean_reversion_score=latest['mean_reversion_score'],
        breakout_probability=latest['breakout_probability'],
        hurst_exponent=latest['hurst'],
        adf_pvalue=latest['adf_pvalue'],
        adf_smoothed=adf_smooth,
        halflife=latest['halflife'],
        z_score=latest_z,
        rsi=latest_rsi,
        strategy=latest['strategy'],
        transition_probability=latest['transition_prob'],
    )
    
    return regime_df, summary, indicators, z_score


def generate_trading_signals(
    regime_df: pd.DataFrame,
    indicators: TechnicalIndicators,
    cumulative: pd.Series,
    z_score: pd.Series
) -> pd.Series:
    """
    Generate BUY/SELL/HOLD signals based on regime and indicators.
    
    This implements the full 5-mode signal logic from the notebook:
    
    DECISION TREE (based on ADF p-value):
    =====================================
    ADF < 0.10  → STRONG MEAN REVERSION MODE
    ADF 0.10-0.30 → MEAN REVERSION MODE  
    ADF 0.30-0.60 → MIXED MODE
    ADF 0.60-0.85 → TREND FOLLOWING MODE (relaxed MR signals)
    ADF > 0.85  → PURE TREND MODE (NO mean-reversion signals!)
    """
    signals = pd.Series(index=regime_df.index, dtype='object')
    signals[:] = 'HOLD'
    
    # Track previous values for direction and regime transitions
    macd_hist_prev = indicators.macd_histogram.shift(1)
    regime_prev = regime_df['regime'].shift(1)
    trend_score_prev = regime_df['trend_score'].shift(1)
    
    # Define bullish and bearish regimes
    bullish_regimes = ['trending_up', 'breakout', 'accumulation']
    bearish_regimes = ['trending_down', 'breakdown', 'distribution']
    
    # EMA for trend structure
    ema50 = indicators.ema50
    
    # Rolling mean for fair value
    rolling_mean = cumulative.rolling(window=60).mean()
    rolling_std = cumulative.rolling(window=60).std()
    
    for date, row in regime_df.iterrows():
        # Skip if indicators not available
        if date not in indicators.macd_histogram.index:
            continue
        if date not in indicators.rsi.index:
            continue
        if date not in cumulative.index:
            continue
        
        # =================================================================
        # GATHER DATA
        # =================================================================
        adf_pvalue = row['adf_smooth'] if 'adf_smooth' in row and not pd.isna(row['adf_smooth']) else row['adf_pvalue']
        trend_score = row['trend_score']
        mr_score = row['mean_reversion_score']
        current_regime = row['regime']
        
        # Previous values
        prev_regime = None
        if date in regime_prev.index and pd.notna(regime_prev.loc[date]):
            prev_regime = regime_prev.loc[date]
        
        prev_trend_score = trend_score_prev.loc[date] if date in trend_score_prev.index else None
        
        # Technical indicators
        rsi = indicators.rsi.loc[date]
        macd_hist = indicators.macd_histogram.loc[date]
        macd_prev = macd_hist_prev.loc[date] if date in macd_hist_prev.index else 0
        
        # Price vs EMA50 (trend structure)
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
        
        # =================================================================
        # REGIME TRANSITION DETECTION
        # =================================================================
        transition_buy_bias = False
        transition_sell_bias = False
        
        if prev_regime is not None and prev_regime != current_regime:
            if current_regime in bullish_regimes and prev_regime not in bullish_regimes:
                transition_buy_bias = True
            if current_regime in bearish_regimes and prev_regime not in bearish_regimes:
                transition_sell_bias = True
        
        # =================================================================
        # TREND SCORE FLIP DETECTION (for pure trend mode)
        # =================================================================
        trend_flipped_bearish = False
        trend_flipped_bullish = False
        
        if prev_trend_score is not None and not pd.isna(prev_trend_score):
            if prev_trend_score > 0 and trend_score < 0:
                trend_flipped_bearish = True
            if prev_trend_score < 0 and trend_score > 0:
                trend_flipped_bullish = True
        
        # =================================================================
        # TREND SCORE ADJUSTMENT FACTOR
        # =================================================================
        trend_buy_adj = 0.3 if trend_score > 0.5 else (0.15 if trend_score > 0 else 0)
        trend_sell_adj = 0.3 if trend_score < -0.5 else (0.15 if trend_score < 0 else 0)
        
        # =================================================================
        # MODE 5: PURE TREND (ADF > 0.85)
        # =================================================================
        if adf_pvalue > 0.85:
            # ----- UPTREND (trend_score > 0) -----
            if trend_score > 0:
                # BUY: Pullback in strong uptrend
                if (price_above_ema50 and macd_turning_up and rsi < 60):
                    signals.loc[date] = 'BUY'
                if transition_buy_bias:
                    signals.loc[date] = 'BUY'
                
                # SELL: Only when trend structure breaks
                if trend_flipped_bearish:
                    signals.loc[date] = 'SELL'
                elif transition_sell_bias:
                    signals.loc[date] = 'SELL'
                elif (price_above_ema50 == False and macd_negative and macd_turning_down):
                    signals.loc[date] = 'SELL'
            
            # ----- DOWNTREND (trend_score < 0) -----
            elif trend_score < 0:
                if (price_above_ema50 == False and macd_turning_down and rsi > 40):
                    signals.loc[date] = 'SELL'
                if transition_sell_bias:
                    signals.loc[date] = 'SELL'
                
                if trend_flipped_bullish:
                    signals.loc[date] = 'BUY'
                elif transition_buy_bias:
                    signals.loc[date] = 'BUY'
                elif (price_above_ema50 and macd_positive and macd_turning_up):
                    signals.loc[date] = 'BUY'
            
            # ----- NO CLEAR TREND -----
            else:
                if transition_buy_bias and macd_turning_up:
                    signals.loc[date] = 'BUY'
                elif transition_sell_bias and macd_turning_down:
                    signals.loc[date] = 'SELL'
        
        # =================================================================
        # MODE 4: TREND FOLLOWING (ADF 0.60 - 0.85)
        # =================================================================
        elif adf_pvalue > 0.60:
            if trend_score > 0.5:
                if (rsi < 50 and (macd_positive or macd_turning_up)):
                    signals.loc[date] = 'BUY'
                if (trend_flipped_bearish or transition_sell_bias or
                    (price_above_ema50 == False and macd_turning_down and rsi > 60)):
                    signals.loc[date] = 'SELL'
            
            elif trend_score < -0.5:
                if (rsi > 50 and (macd_negative or macd_turning_down)):
                    signals.loc[date] = 'SELL'
                if (trend_flipped_bullish or transition_buy_bias or
                    (price_above_ema50 and macd_turning_up and rsi < 40)):
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
        
        # =================================================================
        # MODE 3: MIXED MODE (ADF 0.30 - 0.60)
        # =================================================================
        elif adf_pvalue > 0.30:
            if trend_score > 0.5:
                if (rsi < 50 and (macd_turning_up or macd_positive) and z < 0.5):
                    signals.loc[date] = 'BUY'
                elif (rsi > 70 and z > 1.0 and macd_turning_down):
                    signals.loc[date] = 'SELL'
            
            elif trend_score < -0.5:
                if (rsi > 50 and (macd_turning_down or macd_negative) and z > -0.5):
                    signals.loc[date] = 'SELL'
                elif (rsi < 30 and z < -1.0 and macd_turning_up):
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
        
        # =================================================================
        # MODE 2: MEAN REVERSION (ADF 0.10 - 0.30)
        # =================================================================
        elif adf_pvalue > 0.10:
            z_buy_thresh = -1.0 + trend_buy_adj
            z_sell_thresh = 1.0 - trend_sell_adj
            
            if (z < z_buy_thresh and rsi < 40 and mr_score > 40 and
                (macd_turning_up or trend_score > 0 or transition_buy_bias)):
                signals.loc[date] = 'BUY'
            
            elif (z > z_sell_thresh and rsi > 60 and
                  (macd_turning_down or trend_score < 0 or transition_sell_bias)):
                signals.loc[date] = 'SELL'
        
        # =================================================================
        # MODE 1: STRONG MEAN REVERSION (ADF < 0.10)
        # =================================================================
        else:
            z_buy_thresh = -1.5 + trend_buy_adj
            z_sell_thresh = 1.5 - trend_sell_adj
            
            if (z < z_buy_thresh and rsi < 35 and
                (macd_turning_up or trend_score > 0)):
                signals.loc[date] = 'BUY'
            
            elif (z > z_sell_thresh and rsi > 65 and
                  (macd_turning_down or trend_score < 0)):
                signals.loc[date] = 'SELL'
    
    return signals
