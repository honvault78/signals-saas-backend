"""
Market Regime Detection System - VISUALIZER-MATCHED VERSION
==========================================================
This version uses SIMPLIFIED calculations to match the display/visualizer exactly.

Version: 2.2 - Visualizer-Matched
Author: Professional Trading Systems

CHANGES FROM ORIGINAL:
----------------------
1. Simplified momentum calculation (single period, not weighted)
2. Simplified trend score (SMA comparison, not complex formula)
3. Shorter default windows (12 instead of 20)
4. Shorter default lookback (40 instead of 100)
5. min_regime_persistence = 1 (instant response)
6. min_confidence_threshold = 20 (sensitive)

This matches the embedded detector in your visualizer script for consistency!
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union
import warnings
from scipy import stats
from collections import deque

# Statistical tests for mean reversion validation
try:
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools import add_constant
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("Warning: statsmodels not available. ADF test and half-life calculation disabled.")

warnings.filterwarnings('ignore')

# =====================================================
# ENUMS AND DATA STRUCTURES
# =====================================================

class MarketRegime(Enum):
    """Primary market regimes"""
    RANGING = "ranging"                    # Sideways, mean-reverting market
    TRENDING_UP = "trending_up"            # Strong upward trend
    TRENDING_DOWN = "trending_down"        # Strong downward trend
    VOLATILE_EXPANSION = "volatile_expansion"  # High volatility, unclear direction
    VOLATILE_CONTRACTION = "volatile_contraction"  # Volatility decreasing
    BREAKOUT = "breakout"                  # Price breaking key levels
    BREAKDOWN = "breakdown"                # Price breaking down
    ACCUMULATION = "accumulation"          # Smart money accumulating
    DISTRIBUTION = "distribution"          # Smart money distributing
    UNCERTAIN = "uncertain"                # Low confidence in any regime
    UNKNOWN = "unknown"                    # Insufficient data or unclear

class TrendStrength(Enum):
    """Trend strength classification"""
    NONE = 0
    WEAK = 1
    MODERATE = 2
    STRONG = 3
    EXTREME = 4

class VolatilityState(Enum):
    """Volatility classification"""
    DORMANT = "dormant"          # Extremely low volatility
    LOW = "low"                  # Below average volatility
    NORMAL = "normal"            # Average volatility
    ELEVATED = "elevated"        # Above average volatility
    HIGH = "high"                # High volatility
    EXTREME = "extreme"          # Extreme volatility (news/events)

@dataclass
class MarketMicrostructure:
    """Market microstructure metrics"""
    avg_spread_bps: float          # Average bid-ask spread in basis points
    volume_imbalance: float        # Buy vs sell volume imbalance (-1 to 1)
    trade_intensity: float         # Trades per minute
    large_trade_ratio: float       # Ratio of large trades to total
    price_impact: float            # Average price impact of trades

@dataclass
class RegimeMetrics:
    """Detailed metrics for regime detection"""
    # Price metrics
    price_momentum: float          # Short-term momentum
    price_acceleration: float      # Rate of momentum change
    trend_consistency: float       # How consistent is the trend (0-1)
    range_position: float          # Position within recent range (0-1)
    
    # Volatility metrics
    realized_volatility: float     # Actual price volatility
    volatility_regime: VolatilityState
    volatility_percentile: float   # Current vol vs historical (0-100)
    volatility_trend: float        # Is volatility increasing/decreasing
    
    # Volume metrics
    volume_trend: float            # Volume trend vs average
    volume_volatility: float       # How stable is volume
    smart_money_flow: float        # Large order flow direction
    
    # Statistical metrics
    hurst_exponent: float          # Trending vs mean reverting (0.5 = random)
    autocorrelation: float         # Price serial correlation
    efficiency_ratio: float        # Directional movement efficiency
    fractal_dimension: float       # Market complexity (1-2)
    
    # Market structure
    support_strength: float        # Strength of nearby support (0-1)
    resistance_strength: float     # Strength of nearby resistance (0-1)
    level_clustering: float        # How many S/R levels nearby
    
    # Composite scores
    trend_score: float             # Overall trend strength (-100 to 100)
    mean_reversion_score: float    # Likelihood of mean reversion (0-100)
    breakout_probability: float    # Probability of breakout (0-100)
    regime_stability: float        # How stable is current regime (0-1)
    
    # Statistical validation metrics (NEW)
    adf_statistic: float = 0.0     # ADF test statistic
    adf_pvalue: float = 1.0        # ADF p-value (< 0.05 = stationary/mean-reverting)
    halflife_periods: float = 999.0  # Mean reversion half-life in periods
    is_mean_reverting: bool = False  # Combined validation result

@dataclass
class RegimeAnalysis:
    """Complete regime analysis output"""
    primary_regime: MarketRegime
    confidence: float              # Confidence in regime classification (0-100)
    
    # Detailed components
    metrics: RegimeMetrics
    microstructure: Optional[MarketMicrostructure]
    
    # Trading implications
    suggested_strategy: str        # Recommended trading approach
    risk_level: str               # Current risk level (low/medium/high/extreme)
    
    # Regime characteristics
    regime_duration: int          # How long in current regime (periods)
    regime_strength: float        # How strong is the regime (0-1)
    transition_probability: float  # Probability of regime change (0-100)
    
    # Key levels
    key_support: float            # Most important support level
    key_resistance: float         # Most important resistance level
    pivot_point: float            # Current pivot point
    
    # Additional context
    volatility_forecast: float    # Expected volatility next period
    trend_forecast: str           # Expected trend direction
    warnings: List[str]           # Any warning messages

# =====================================================
# MARKET REGIME DETECTOR
# =====================================================

class MarketRegimeDetector:
    """
    Institutional-grade market regime detection system.
    
    This system analyzes price, volume, volatility, and microstructure
    data across multiple timeframes to determine the current market regime
    with high confidence.
    """
    
    def __init__(self, 
                 lookback_periods: int = 40,        # Changed from 100 to match visualizer
                 min_periods: int = 20,             # Changed from 30
                 volatility_window: int = 12,       # Changed from 20 to match visualizer
                 trend_window: int = 12,            # Changed from 20 to match visualizer
                 asset_type: str = "crypto"):
        """
        Initialize the regime detector.
        """
        self.lookback_periods = lookback_periods
        self.min_periods = min_periods
        self.volatility_window = volatility_window
        self.trend_window = trend_window
        self.asset_type = asset_type

        # Historical data storage
        self.price_history = deque(maxlen=lookback_periods * 2)
        self.volume_history = deque(maxlen=lookback_periods * 2)
        self.regime_history = deque(maxlen=50)  # Store more history for persistence
        
        # NEW: Regime persistence tracking
        self.current_regime = MarketRegime.UNCERTAIN
        self.regime_start_time = datetime.now()
        self.regime_persistence_count = 0
        self.min_regime_persistence = 0  # FIXED: Changed from 3 to 1 for instant response

        # Calibration parameters for CRYPTO (Solana specifically)
        if asset_type == "crypto":
            # ULTRA SENSITIVE thresholds for crypto markets (from improved visualizer)
            self.momentum_threshold = 1.0          # 1% minimum for trend
            self.strong_momentum_threshold = 3.0   # 3% for strong trend
            self.extreme_momentum_threshold = 7.0  # 7% for extreme trend
            
            self.trend_score_threshold = 2       # Minimum score to declare trend
            self.strong_trend_threshold = 5      # Strong trend threshold
            
            self.consistency_threshold = 0.55     # Lower consistency needed
            self.efficiency_threshold = 0.3       # Lower efficiency needed
            
            # Confidence thresholds
            self.min_confidence_threshold = 20    # FIXED: Lowered from 30 for faster signals
            
            self.volatility_thresholds = {
                'dormant': 0.05,      
                'low': 0.25,          # Raised
                'normal': 0.50,       
                'elevated': 0.70,     # Lowered
                'high': 0.85,         # Lowered
                'extreme': 0.95       # Lowered
            }
            
            # Crypto-specific volatility levels (NON-ANNUALIZED %)
            self.extreme_volatility_threshold = 8.0   # 8% volatility for "extreme"
            self.high_volatility_threshold = 5.0      # 5% volatility for "high"
            self.normal_volatility_threshold = 3.0    # 3% volatility is "normal"
            self.typical_crypto_vol = 4.0             # Baseline for ratio calculations
            
        else:
            # Traditional asset thresholds
            self.momentum_threshold = 0.5
            self.strong_momentum_threshold = 1.5
            self.extreme_momentum_threshold = 3.0
            
            self.trend_score_threshold = 15
            self.strong_trend_threshold = 30
            
            self.consistency_threshold = 0.65
            self.efficiency_threshold = 0.35
            
            self.min_confidence_threshold = 40
            
            self.volatility_thresholds = {
                'dormant': 0.1,
                'low': 0.25,
                'normal': 0.5,
                'elevated': 0.75,
                'high': 0.9,
                'extreme': 0.98
            }

    def analyze(self, 
                prices: np.ndarray,
                volumes: np.ndarray,
                highs: Optional[np.ndarray] = None,
                lows: Optional[np.ndarray] = None,
                timestamps: Optional[np.ndarray] = None) -> RegimeAnalysis:
        """
        Perform comprehensive market regime analysis.
        """
        
        # Validate inputs
        if len(prices) < self.min_periods:
            return self._insufficient_data_response()
            
        # Calculate all metrics
        metrics = self._calculate_metrics(prices, volumes, highs, lows)
        
        # Determine primary regime with persistence
        primary_regime, confidence = self._classify_regime_with_persistence(metrics)
        
        # Calculate regime characteristics
        regime_duration = self._calculate_regime_duration(primary_regime)
        regime_strength = self._calculate_regime_strength(metrics, primary_regime)
        transition_prob = self._calculate_transition_probability(metrics, regime_duration)
        
        # Identify key levels
        key_support, key_resistance, pivot = self._identify_key_levels(prices, volumes)
        
        # Generate forecasts and recommendations
        vol_forecast = self._forecast_volatility(metrics)
        trend_forecast = self._forecast_trend(metrics)
        strategy = self._suggest_strategy(primary_regime, metrics)
        risk_level = self._assess_risk_level(metrics, primary_regime)
        
        # Check for warnings
        warnings = self._check_warnings(metrics, primary_regime)
        
        return RegimeAnalysis(
            primary_regime=primary_regime,
            confidence=confidence,
            metrics=metrics,
            microstructure=None,
            suggested_strategy=strategy,
            risk_level=risk_level,
            regime_duration=regime_duration,
            regime_strength=regime_strength,
            transition_probability=transition_prob,
            key_support=key_support,
            key_resistance=key_resistance,
            pivot_point=pivot,
            volatility_forecast=vol_forecast,
            trend_forecast=trend_forecast,
            warnings=warnings
        )
    
    def _calculate_metrics(self, prices: np.ndarray, volumes: np.ndarray,
                          highs: np.ndarray = None, lows: np.ndarray = None) -> RegimeMetrics:
        """Calculate all regime detection metrics"""
        
        # Price metrics
        returns = np.diff(prices) / prices[:-1]
        momentum = self._calculate_momentum(prices)
        acceleration = self._calculate_acceleration(prices)
        trend_consistency = self._calculate_trend_consistency(returns)
        range_position = self._calculate_range_position(prices)
        
        # Volatility metrics
        realized_vol = self._calculate_realized_volatility(returns)
        vol_regime = self._classify_volatility_regime(realized_vol, returns)
        vol_percentile = self._calculate_volatility_percentile(realized_vol, returns)
        vol_trend = self._calculate_volatility_trend(returns)
        
        # Volume metrics
        volume_trend = self._calculate_volume_trend(volumes)
        volume_volatility = self._calculate_volume_volatility(volumes)
        smart_money_flow = self._calculate_smart_money_flow(prices, volumes)
        
        # Statistical metrics
        hurst = self._calculate_hurst_exponent(prices)
        autocorr = self._calculate_autocorrelation(returns)
        efficiency = self._calculate_efficiency_ratio(prices)
        fractal_dim = self._calculate_fractal_dimension(prices)
        
        # NEW: Statistical validation metrics
        adf_stat, adf_pvalue = self._calculate_adf_test(prices)
        halflife = self._calculate_halflife(prices)
        is_mr_valid, mr_validation_reason = self._validate_mean_reversion(
            prices, hurst, adf_pvalue, halflife
        )
        
        # Market structure
        support_strength, resistance_strength = self._calculate_sr_strength(prices, volumes)
        level_clustering = self._calculate_level_clustering(prices)
        
        # Composite scores - SIMPLIFIED TO MATCH VISUALIZER
        trend_score = self._calculate_trend_score_simple(prices)  # Use simple SMA method
        mr_score = self._calculate_mean_reversion_score(hurst, autocorr, range_position)
        breakout_prob = self._calculate_breakout_probability(
            range_position, vol_regime, volume_trend, level_clustering
        )
        regime_stability = self._calculate_regime_stability(trend_consistency, vol_trend)
        
        return RegimeMetrics(
            price_momentum=momentum,
            price_acceleration=acceleration,
            trend_consistency=trend_consistency,
            range_position=range_position,
            realized_volatility=realized_vol,
            volatility_regime=vol_regime,
            volatility_percentile=vol_percentile,
            volatility_trend=vol_trend,
            volume_trend=volume_trend,
            volume_volatility=volume_volatility,
            smart_money_flow=smart_money_flow,
            hurst_exponent=hurst,
            autocorrelation=autocorr,
            efficiency_ratio=efficiency,
            fractal_dimension=fractal_dim,
            support_strength=support_strength,
            resistance_strength=resistance_strength,
            level_clustering=level_clustering,
            trend_score=trend_score,
            mean_reversion_score=mr_score,
            breakout_probability=breakout_prob,
            regime_stability=regime_stability,
            # NEW: Statistical validation
            adf_statistic=adf_stat,
            adf_pvalue=adf_pvalue,
            halflife_periods=halflife,
            is_mean_reverting=is_mr_valid
        )
    
    def _calculate_momentum(self, prices: np.ndarray) -> float:
        """Calculate price momentum - SIMPLIFIED to match visualizer"""
        if len(prices) < 2:
            return 0.0
        
        # SIMPLIFIED: Use single period like visualizer (not weighted average)
        if len(prices) >= 3:
            # Very short-term momentum - matches visualizer
            price_momentum = ((prices[-1] / prices[-3] - 1) * 100)
        elif len(prices) >= 2:
            price_momentum = ((prices[-1] / prices[-2] - 1) * 100)
        else:
            return 0.0
        
        return price_momentum
    
    def _calculate_acceleration(self, prices: np.ndarray) -> float:
        """Calculate price acceleration (momentum of momentum)"""
        if len(prices) < self.trend_window + 5:
            return 0.0
            
        # Calculate momentum at different points
        mom_current = self._calculate_momentum(prices)
        mom_previous = self._calculate_momentum(prices[:-5])
        
        return mom_current - mom_previous
    
    def _calculate_trend_consistency(self, returns: np.ndarray) -> float:
        """Calculate how consistent the trend direction is (0-1)"""
        if len(returns) < self.trend_window:
            return 0.5

        recent_returns = returns[-self.trend_window:]

        # Separate positive and negative returns
        positive_returns = recent_returns[recent_returns > 0]
        negative_returns = recent_returns[recent_returns < 0]

        # Calculate directional consistency
        if len(positive_returns) > len(negative_returns):
            # Uptrend consistency
            consistency = len(positive_returns) / len(recent_returns)
            # Boost if returns are consistently positive
            avg_positive = np.mean(positive_returns) if len(positive_returns) > 0 else 0
            avg_negative = abs(np.mean(negative_returns)) if len(negative_returns) > 0 else 0
            if avg_positive > avg_negative * 2:  # Strong positive bias
                consistency = min(1.0, consistency * 1.2)
        else:
            # Downtrend consistency
            consistency = len(negative_returns) / len(recent_returns)
            # Boost if returns are consistently negative
            avg_negative = abs(np.mean(negative_returns)) if len(negative_returns) > 0 else 0
            avg_positive = np.mean(positive_returns) if len(positive_returns) > 0 else 0
            if avg_negative > avg_positive * 2:  # Strong negative bias
                consistency = min(1.0, consistency * 1.2)

        # Also check for consecutive moves
        max_consecutive = 1
        current_consecutive = 1

        for i in range(1, len(recent_returns)):
            if (recent_returns[i] > 0 and recent_returns[i-1] > 0) or \
               (recent_returns[i] < 0 and recent_returns[i-1] < 0):
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 1

        # Combine both metrics
        consecutive_bonus = min(0.3, max_consecutive / len(recent_returns))

        return min(1.0, consistency + consecutive_bonus)
    
    def _calculate_range_position(self, prices: np.ndarray) -> float:
        """Calculate position within recent range (0-1)"""
        if len(prices) < self.lookback_periods:
            window = len(prices)
        else:
            window = self.lookback_periods
            
        recent_high = np.max(prices[-window:])
        recent_low = np.min(prices[-window:])
        
        if recent_high == recent_low:
            return 0.5
            
        return (prices[-1] - recent_low) / (recent_high - recent_low)
    
    def _calculate_realized_volatility(self, returns: np.ndarray) -> float:
        """Calculate realized volatility - NON-ANNUALIZED percentage"""
        # Use shorter window for crypto volatility
        vol_window = min(10, len(returns))
        
        if len(returns) >= vol_window:
            # Non-annualized percentage volatility (more relevant for short-term)
            realized_vol = np.std(returns[-vol_window:]) * 100
        else:
            realized_vol = np.std(returns) * 100 if len(returns) > 0 else 2.0
        
        return realized_vol
    
    def _classify_volatility_regime(self, current_vol: float, returns: np.ndarray) -> VolatilityState:
        """Classify current volatility regime"""
        if len(returns) < self.lookback_periods:
            historical_vols = [np.std(returns[max(0, i-self.volatility_window):i]) * np.sqrt(252) * 100 
                              for i in range(self.volatility_window, len(returns))]
        else:
            historical_vols = [np.std(returns[i-self.volatility_window:i]) * np.sqrt(252) * 100 
                              for i in range(self.volatility_window, len(returns))]
        
        if not historical_vols:
            return VolatilityState.NORMAL
            
        percentile = stats.percentileofscore(historical_vols, current_vol) / 100
        
        if percentile < self.volatility_thresholds['dormant']:
            return VolatilityState.DORMANT
        elif percentile < self.volatility_thresholds['low']:
            return VolatilityState.LOW
        elif percentile < self.volatility_thresholds['normal']:
            return VolatilityState.NORMAL
        elif percentile < self.volatility_thresholds['elevated']:
            return VolatilityState.ELEVATED
        elif percentile < self.volatility_thresholds['high']:
            return VolatilityState.HIGH
        else:
            return VolatilityState.EXTREME
    
    def _calculate_volatility_percentile(self, current_vol: float, returns: np.ndarray) -> float:
        """Calculate volatility percentile - IMPROVED with better rolling calculation"""
        vol_percentile = 50  # Default
        
        if len(returns) >= 20:
            # Use consistent rolling window
            roll_window = min(10, len(returns) // 3)
            rolling_vols = []
            
            # Calculate historical rolling volatilities
            for i in range(roll_window, len(returns) + 1):
                window_returns = returns[max(0, i-roll_window):i]
                if len(window_returns) >= 3:
                    # Non-annualized percentage volatility
                    rolling_vols.append(np.std(window_returns) * 100)
            
            if len(rolling_vols) >= 5:
                # Calculate percentile
                vol_percentile = (sum(1 for v in rolling_vols if v <= current_vol) / len(rolling_vols)) * 100
                
                # Smooth extreme values
                vol_percentile = max(5, min(95, vol_percentile))
        
        return vol_percentile
    
    def _calculate_volatility_trend(self, returns: np.ndarray) -> float:
        """Calculate if volatility is increasing or decreasing"""
        if len(returns) < self.volatility_window * 2:
            return 0.0
            
        # Calculate rolling volatility
        vols = []
        for i in range(self.volatility_window, len(returns)):
            vol = np.std(returns[i-self.volatility_window:i])
            vols.append(vol)
            
        if len(vols) < 2:
            return 0.0
            
        # Trend of volatility
        recent_vols = vols[-10:] if len(vols) >= 10 else vols
        if len(recent_vols) < 2:
            return 0.0
            
        x = np.arange(len(recent_vols))
        slope, _, _, _, _ = stats.linregress(x, recent_vols)
        
        return slope * 1000  # Scale for readability
    
    def _calculate_volume_trend(self, volumes: np.ndarray) -> float:
        """Calculate volume trend vs average"""
        if len(volumes) < self.trend_window:
            return 1.0
            
        avg_volume = np.mean(volumes[-self.lookback_periods:]) if len(volumes) >= self.lookback_periods else np.mean(volumes)
        recent_avg = np.mean(volumes[-self.trend_window:])
        
        return recent_avg / avg_volume if avg_volume > 0 else 1.0
    
    def _calculate_volume_volatility(self, volumes: np.ndarray) -> float:
        """Calculate how stable volume is"""
        if len(volumes) < self.volatility_window:
            return 0.5
            
        recent_volumes = volumes[-self.volatility_window:]
        return np.std(recent_volumes) / np.mean(recent_volumes) if np.mean(recent_volumes) > 0 else 0.5
    
    def _calculate_smart_money_flow(self, prices: np.ndarray, volumes: np.ndarray) -> float:
        """Calculate smart money flow (simplified without tick data)"""
        if len(prices) < 21 or len(volumes) < 21:
            return 0.0

        # Calculate price changes as percentage
        recent_prices = prices[-21:]  # Get last 21 prices
        price_changes = (recent_prices[1:] - recent_prices[:-1]) / recent_prices[:-1]  # 20 changes

        # Get corresponding volumes
        recent_volumes = volumes[-20:]  # Last 20 volumes to match price changes
        avg_volume = np.mean(recent_volumes)

        if avg_volume == 0:
            return 0.0

        volume_ratios = recent_volumes / avg_volume

        # High volume on up moves vs down moves
        up_volume = np.sum(volume_ratios[price_changes > 0])
        down_volume = np.sum(volume_ratios[price_changes < 0])
        total_volume = up_volume + down_volume

        if total_volume == 0:
            return 0.0

        return (up_volume - down_volume) / total_volume
    
    def _calculate_hurst_exponent(self, prices: np.ndarray) -> float:
        """Calculate Hurst exponent (trending vs mean reverting)"""
        if len(prices) < 100:
            return 0.5
            
        try:
            # Simplified R/S analysis
            returns = np.diff(np.log(prices))
            
            # Calculate for different lag values
            lags = range(2, min(20, len(returns) // 2))
            tau = []
            
            for lag in lags:
                # Calculate R/S for this lag
                subseries = [returns[i:i+lag] for i in range(0, len(returns)-lag+1, lag)]
                
                rs_values = []
                for series in subseries:
                    if len(series) < 2:
                        continue
                    mean = np.mean(series)
                    deviations = series - mean
                    Z = np.cumsum(deviations)
                    R = np.max(Z) - np.min(Z)
                    S = np.std(series, ddof=1)
                    if S > 0:
                        rs_values.append(R / S)
                
                if rs_values:
                    tau.append(np.mean(rs_values))
            
            if len(tau) < 2:
                return 0.5
                
            # Fit log-log regression
            log_lags = np.log(list(lags)[:len(tau)])
            log_tau = np.log(tau)
            hurst, _, _, _, _ = stats.linregress(log_lags, log_tau)
            
            return max(0, min(1, hurst))
            
        except:
            return 0.5
    
    def _calculate_adf_test(self, prices: np.ndarray) -> Tuple[float, float]:
        """
        Augmented Dickey-Fuller test for stationarity.
        
        Returns:
            (adf_statistic, p_value)
            
        Interpretation:
            - p_value < 0.05: Reject null hypothesis → series IS stationary (mean-reverting)
            - p_value >= 0.05: Cannot reject null → series may have unit root (trending/random walk)
        """
        if not STATSMODELS_AVAILABLE:
            return 0.0, 1.0  # Default to "not stationary"
        
        if len(prices) < 50:
            return 0.0, 1.0
        
        try:
            # Run ADF test
            result = adfuller(prices, maxlag=None, regression='c', autolag='AIC')
            adf_statistic = result[0]
            p_value = result[1]
            
            return float(adf_statistic), float(p_value)
            
        except Exception as e:
            return 0.0, 1.0
    
    def _calculate_halflife(self, prices: np.ndarray) -> float:
        """
        Calculate mean reversion half-life using Ornstein-Uhlenbeck process.
        
        Uses OLS regression: ΔP(t) = θ * (μ - P(t-1)) + ε
        Half-life = ln(2) / θ
        
        Returns:
            Half-life in periods (number of candles)
            Returns 9999 if no mean reversion detected
        """
        if not STATSMODELS_AVAILABLE:
            return 9999.0
        
        if len(prices) < 50:
            return 9999.0
        
        try:
            # Convert to pandas for easier manipulation
            prices_series = pd.Series(prices)
            
            # Create lagged price
            lag = prices_series.shift(1)
            
            # Price change
            delta = prices_series - lag
            
            # Remove NaN
            lag = lag.iloc[1:].values
            delta = delta.iloc[1:].values
            
            # Add constant for regression
            lag_with_const = add_constant(lag)
            
            # Run OLS: delta = alpha + theta * lag + epsilon
            model = OLS(delta, lag_with_const).fit()
            
            # theta is the coefficient on the lagged price
            theta = model.params[1]
            
            # If theta >= 0, no mean reversion (price is trending or random walk)
            if theta >= 0:
                return 9999.0
            
            # Half-life = -ln(2) / theta
            halflife = -np.log(2) / theta
            
            # Sanity check
            if halflife < 1 or halflife > 10000:
                return 9999.0
            
            return float(halflife)
            
        except Exception as e:
            return 9999.0
    
    def _validate_mean_reversion(self, prices: np.ndarray, 
                                  hurst: float,
                                  adf_pvalue: float,
                                  halflife: float,
                                  max_halflife: float = 360.0) -> Tuple[bool, str]:
        """
        Strategic validation: Is mean reversion viable for this price series?
        
        Args:
            prices: Price array
            hurst: Pre-calculated Hurst exponent
            adf_pvalue: Pre-calculated ADF p-value
            halflife: Pre-calculated half-life in periods
            max_halflife: Maximum acceptable half-life (default 360 = 30 hours on 5M)
            
        Returns:
            (is_valid, reason_string)
        """
        reasons = []
        
        # Check 1: ADF test (stationarity)
        if adf_pvalue > 0.05:
            reasons.append(f"ADF p={adf_pvalue:.3f}>0.05 (non-stationary)")
        
        # Check 2: Hurst exponent
        if hurst > 0.55:
            reasons.append(f"Hurst={hurst:.2f}>0.55 (trending)")
        
        # Check 3: Half-life
        if halflife > max_halflife:
            reasons.append(f"HalfLife={halflife:.0f}>{max_halflife:.0f} (too slow)")
        
        is_valid = len(reasons) == 0
        
        if is_valid:
            reason = f"VALID: ADF p={adf_pvalue:.3f}, Hurst={hurst:.2f}, HL={halflife:.0f}"
        else:
            reason = f"INVALID: {'; '.join(reasons)}"
        
        return is_valid, reason
    
    def _calculate_autocorrelation(self, returns: np.ndarray, lag: int = 1) -> float:
        """Calculate return autocorrelation"""
        if len(returns) < lag + 10:
            return 0.0
            
        return np.corrcoef(returns[:-lag], returns[lag:])[0, 1]
    
    def _calculate_efficiency_ratio(self, prices: np.ndarray) -> float:
        """Calculate Kaufman's Efficiency Ratio"""
        if len(prices) < self.trend_window:
            return 0.5
            
        period = min(self.trend_window, len(prices) - 1)
        
        # Net change over period
        net_change = abs(prices[-1] - prices[-period-1])
        
        # Sum of absolute changes
        sum_changes = np.sum(np.abs(np.diff(prices[-period-1:])))
        
        if sum_changes == 0:
            return 0.0
            
        return net_change / sum_changes
    
    def _calculate_fractal_dimension(self, prices: np.ndarray) -> float:
        """Calculate fractal dimension using box counting"""
        if len(prices) < 50:
            return 1.5
            
        try:
            # Normalize prices
            normalized = (prices - np.min(prices)) / (np.max(prices) - np.min(prices))
            
            # Box counting method
            n = len(normalized)
            max_box_size = n // 4
            box_sizes = [2**i for i in range(2, int(np.log2(max_box_size)))]
            
            counts = []
            for box_size in box_sizes:
                count = 0
                for i in range(0, n, box_size):
                    if i + box_size > n:
                        break
                    segment = normalized[i:i+box_size]
                    price_range = np.max(segment) - np.min(segment)
                    boxes_needed = int(np.ceil(price_range * box_size)) if price_range > 0 else 1
                    count += boxes_needed
                counts.append(count)
            
            if len(counts) < 2:
                return 1.5
                
            # Log-log regression
            log_sizes = np.log(box_sizes[:len(counts)])
            log_counts = np.log(counts)
            slope, _, _, _, _ = stats.linregress(log_sizes, log_counts)
            
            return max(1, min(2, -slope))
            
        except:
            return 1.5
    
    def _calculate_sr_strength(self, prices: np.ndarray, volumes: np.ndarray) -> Tuple[float, float]:
        """Calculate support and resistance strength"""
        if len(prices) < 20:
            return 0.5, 0.5
            
        current_price = prices[-1]
        
        # Find recent peaks and troughs
        highs = []
        lows = []
        
        for i in range(2, len(prices) - 2):
            if prices[i] > prices[i-1] and prices[i] > prices[i-2] and \
               prices[i] > prices[i+1] and prices[i] > prices[i+2]:
                highs.append((i, prices[i]))
            elif prices[i] < prices[i-1] and prices[i] < prices[i-2] and \
                 prices[i] < prices[i+1] and prices[i] < prices[i+2]:
                lows.append((i, prices[i]))
        
        # Find nearest support (below current price)
        support_strength = 0.0
        supports = [low[1] for low in lows if low[1] < current_price]
        if supports:
            nearest_support = max(supports)
            # Count touches
            touches = sum(1 for p in prices if abs(p - nearest_support) / nearest_support < 0.002)
            support_strength = min(1.0, touches / 10)
        
        # Find nearest resistance (above current price)
        resistance_strength = 0.0
        resistances = [high[1] for high in highs if high[1] > current_price]
        if resistances:
            nearest_resistance = min(resistances)
            # Count touches
            touches = sum(1 for p in prices if abs(p - nearest_resistance) / nearest_resistance < 0.002)
            resistance_strength = min(1.0, touches / 10)
        
        return support_strength, resistance_strength
    
    def _calculate_level_clustering(self, prices: np.ndarray) -> float:
        """Calculate how many S/R levels are nearby"""
        if len(prices) < 50:
            return 0.5
            
        current_price = prices[-1]
        
        # Find all potential levels
        levels = []
        
        # Round numbers
        round_base = round(current_price / 10) * 10
        for i in range(-5, 6):
            level = round_base + i * 10
            if 0.9 * current_price <= level <= 1.1 * current_price:
                levels.append(level)
        
        # Historical pivots
        for i in range(5, len(prices) - 5, 5):
            segment = prices[i-5:i+5]
            pivot = (np.max(segment) + np.min(segment) + prices[i]) / 3
            if 0.95 * current_price <= pivot <= 1.05 * current_price:
                levels.append(pivot)
        
        # Count levels within 2% of current price
        nearby_levels = sum(1 for level in levels if abs(level - current_price) / current_price <= 0.02)
        
        return min(1.0, nearby_levels / 10)
    
    def _calculate_trend_score_simple(self, prices: np.ndarray) -> float:
        """Calculate trend score using simple SMA comparison - MATCHES VISUALIZER"""
        
        # Use ultra-short SMAs like visualizer
        very_short = min(3, len(prices) // 4)
        short_window = min(5, len(prices) // 3)
        medium_window = min(10, len(prices) // 2)
        
        if len(prices) < short_window:
            return 0.0
        
        # Calculate SMAs
        sma_very_short = np.mean(prices[-very_short:]) if len(prices) >= very_short else prices[-1]
        sma_short = np.mean(prices[-short_window:]) if len(prices) >= short_window else sma_very_short
        sma_medium = np.mean(prices[-medium_window:]) if len(prices) >= medium_window else sma_short
        
        # Most sensitive trend score - use shortest comparison
        trend_score = ((sma_very_short / sma_short - 1) * 100) if sma_short != 0 else 0
        
        # Alternative trend calculation for confirmation
        if abs(trend_score) < 1 and len(prices) >= medium_window:
            trend_score = ((sma_short / sma_medium - 1) * 100) if sma_medium != 0 else 0
        
        return trend_score
    
    def _calculate_trend_score_crypto(self, momentum: float, acceleration: float, 
                                     consistency: float, efficiency: float) -> float:
        """Calculate trend score - COMPLEX VERSION (kept for backward compatibility)"""
        
        # Direct trend score based primarily on momentum with lighter modifiers
        score = momentum * 1.5  # Base from momentum
        
        # Add consistency contribution (simpler)
        if consistency > 0.5:
            score += (consistency - 0.5) * 20 * np.sign(momentum) if momentum != 0 else 0
        
        # Add efficiency contribution (simpler)
        if efficiency > 0.3:
            score += (efficiency - 0.3) * 15 * np.sign(momentum) if momentum != 0 else 0
        
        # Add acceleration if confirming
        if (momentum > 0 and acceleration > 0) or (momentum < 0 and acceleration < 0):
            score += acceleration * 0.2
        
        return max(-100, min(100, score))
    
    def _calculate_mean_reversion_score(self, hurst: float, autocorr: float, 
                                       range_position: float) -> float:
        """Calculate mean reversion likelihood (0-100)"""
        
        # Hurst < 0.5 suggests mean reversion
        hurst_score = max(0, (0.5 - hurst) * 200)
        
        # Negative autocorrelation suggests mean reversion
        autocorr_score = max(0, -autocorr * 100)
        
        # Extreme range positions suggest reversion
        range_score = 0
        if range_position > 0.8:
            range_score = (range_position - 0.8) * 500
        elif range_position < 0.2:
            range_score = (0.2 - range_position) * 500
        
        score = (hurst_score * 0.4 + autocorr_score * 0.3 + range_score * 0.3)
        
        return min(100, score)
    
    def _calculate_breakout_probability(self, range_position: float, 
                                       vol_regime: VolatilityState,
                                       volume_trend: float, 
                                       level_clustering: float) -> float:
        """Calculate breakout probability (0-100)"""
        
        # Base probability from range position
        if range_position > 0.8:
            base_prob = 30 + (range_position - 0.8) * 200
        elif range_position < 0.2:
            base_prob = 30 + (0.2 - range_position) * 200
        else:
            base_prob = 10
        
        # Adjust for volatility
        vol_multiplier = 1.0
        if vol_regime == VolatilityState.DORMANT:
            vol_multiplier = 1.5  # Dormant vol often precedes breakout
        elif vol_regime == VolatilityState.LOW:
            vol_multiplier = 1.2
        elif vol_regime in [VolatilityState.HIGH, VolatilityState.EXTREME]:
            vol_multiplier = 0.8  # Already breaking out?
        
        # Adjust for volume
        if volume_trend > 1.5:
            vol_multiplier *= 1.3
        
        # Adjust for level clustering
        level_multiplier = 1 + (1 - level_clustering) * 0.5
        
        probability = base_prob * vol_multiplier * level_multiplier
        
        return min(100, probability)
    
    def _calculate_regime_stability(self, trend_consistency: float, 
                                   volatility_trend: float) -> float:
        """Calculate how stable the current regime is (0-1)"""
        
        # High consistency = stable
        consistency_score = trend_consistency
        
        # Stable volatility = stable regime
        vol_stability = 1 - min(1, abs(volatility_trend) / 10)
        
        return (consistency_score * 0.6 + vol_stability * 0.4)
    
    def _classify_regime_with_persistence(self, metrics: RegimeMetrics) -> Tuple[MarketRegime, float]:
        """Classify regime with persistence to prevent flip-flopping"""
        
        # Get current classification
        new_regime, new_confidence = self._classify_regime(metrics)
        
        # Apply minimum confidence threshold
        if new_confidence < self.min_confidence_threshold:
            return MarketRegime.UNCERTAIN, new_confidence
        
        # Check if this is a regime change
        if new_regime != self.current_regime:
            # For trend changes, require stronger evidence
            if self.current_regime in [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN] and \
               new_regime in [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN]:
                # Switching between trends requires very high confidence
                if new_confidence < 70:
                    return self.current_regime, 50.0  # Stay in current regime with lower confidence
            
            # Count consecutive signals for new regime
            if len(self.regime_history) > 0 and self.regime_history[-1] == new_regime:
                self.regime_persistence_count += 1
            else:
                self.regime_persistence_count = 1
            
            # Need minimum persistence to change regime
            if self.regime_persistence_count < self.min_regime_persistence:
                # Not enough persistence, stay in current regime
                return self.current_regime, max(40.0, new_confidence - 20)
            
            # Regime change confirmed
            self.current_regime = new_regime
            self.regime_start_time = datetime.now()
            self.regime_persistence_count = 0
        
        # Add to history
        self.regime_history.append(new_regime)
        
        return self.current_regime, new_confidence
    
    def _classify_regime(self, metrics: RegimeMetrics) -> Tuple[MarketRegime, float]:
        """Classify market regime - MOMENTUM FIRST with ultra-sensitive thresholds (from visualizer)"""
        
        trend = metrics.trend_score
        vol_pct = metrics.volatility_percentile
        momentum = metrics.price_momentum
        
        # PRIORITY 1: Check momentum for immediate regime (most responsive)
        if abs(momentum) > 2:  # Very low threshold for ultra-sensitivity
            if momentum > 2:
                if vol_pct > 70:
                    return MarketRegime.BREAKOUT, 75.0
                elif trend > 1:
                    return MarketRegime.TRENDING_UP, 70.0
                else:
                    return MarketRegime.ACCUMULATION, 60.0
            else:  # momentum < -2
                if vol_pct > 70:
                    return MarketRegime.BREAKDOWN, 75.0
                elif trend < -1:
                    return MarketRegime.TRENDING_DOWN, 70.0
                else:
                    return MarketRegime.DISTRIBUTION, 60.0
        
        # PRIORITY 2: Check trend
        if abs(trend) > 2:  # Very sensitive
            confidence = min(70.0, 50.0 + abs(trend) * 2)
            if trend > 2:
                return MarketRegime.TRENDING_UP, confidence
            else:
                return MarketRegime.TRENDING_DOWN, confidence
        
        # PRIORITY 3: Volatility-based regimes
        if vol_pct > 80:
            return MarketRegime.VOLATILE_EXPANSION, 65.0
        elif vol_pct < 25:
            return MarketRegime.VOLATILE_CONTRACTION, 55.0
        
        # PRIORITY 4: Range detection with momentum bias
        if abs(trend) < 1 and abs(momentum) < 1:
            return MarketRegime.RANGING, 60.0
        
        # PRIORITY 5: Accumulation/Distribution based on slight biases
        if trend > 0.5 or momentum > 0.5:
            return MarketRegime.ACCUMULATION, 50.0
        elif trend < -0.5 or momentum < -0.5:
            return MarketRegime.DISTRIBUTION, 50.0
        
        # Default to ranging
        return MarketRegime.RANGING, 45.0
    
    def _calculate_regime_duration(self, current_regime: MarketRegime) -> int:
        """Calculate how long we've been in current regime"""
        if not self.regime_history:
            return 1
            
        duration = 1
        for regime in reversed(self.regime_history):
            if regime == current_regime:
                duration += 1
            else:
                break
                
        return duration
    
    def _calculate_regime_strength(self, metrics: RegimeMetrics, 
                                  regime: MarketRegime) -> float:
        """Calculate strength of current regime (0-1)"""
        
        if regime == MarketRegime.RANGING:
            # For ranging, consider mean reversion score and lack of trend
            ranging_strength = metrics.mean_reversion_score / 100
            trend_weakness = 1 - min(1, abs(metrics.trend_score) / 50)
            return (ranging_strength + trend_weakness) / 2
            
        elif regime in [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN]:
            return min(1.0, abs(metrics.trend_score) / 100)
            
        elif regime in [MarketRegime.VOLATILE_EXPANSION, MarketRegime.VOLATILE_CONTRACTION]:
            return min(1.0, abs(metrics.volatility_trend) / 5)
            
        elif regime in [MarketRegime.BREAKOUT, MarketRegime.BREAKDOWN]:
            return min(1.0, metrics.breakout_probability / 100)
            
        elif regime == MarketRegime.UNCERTAIN:
            return 0.2  # Low strength for uncertain regime
            
        else:
            return 0.5
    
    def _calculate_transition_probability(self, metrics: RegimeMetrics, 
                                        duration: int) -> float:
        """Calculate probability of regime change (0-100)"""
        
        # Base probability increases with duration
        base_prob = min(50, duration * 2)
        
        # Adjust for regime stability
        stability_adjustment = (1 - metrics.regime_stability) * 50
        
        # Adjust for extreme conditions
        extreme_adjustment = 0
        if metrics.range_position > 0.9 or metrics.range_position < 0.1:
            extreme_adjustment = 20
        if metrics.volatility_percentile > 90 or metrics.volatility_percentile < 10:
            extreme_adjustment += 20
            
        probability = base_prob + stability_adjustment + extreme_adjustment
        
        return min(100, probability)
    
    def _identify_key_levels(self, prices: np.ndarray, volumes: np.ndarray) -> Tuple[float, float, float]:
        """Identify key support/resistance levels"""
        if len(prices) < 20:
            # Not enough data, use simple min/max
            return np.min(prices), np.max(prices), np.median(prices)

        # Find local minima and maxima
        support_levels = []
        resistance_levels = []

        # Look for swing points with proper bounds checking
        for i in range(5, min(50, len(prices) - 5)):
            # Get the window around this point
            start_idx = max(0, i - 5)
            end_idx = min(len(prices), i + 6)  # +6 because slice is exclusive at end
            window = prices[start_idx:end_idx]

            if len(window) == 0:
                continue

            # Check if it's a local minimum (support)
            if prices[i] == np.min(window):
                support_levels.append(prices[i])

            # Check if it's a local maximum (resistance)
            if prices[i] == np.max(window):
                resistance_levels.append(prices[i])

        # Get key levels
        if support_levels:
            key_support = np.percentile(support_levels, 20)  # Strong support
        else:
            key_support = np.min(prices[-20:])

        if resistance_levels:
            key_resistance = np.percentile(resistance_levels, 80)  # Strong resistance
        else:
            key_resistance = np.max(prices[-20:])

        # Volume-weighted average price as pivot
        recent_prices = prices[-20:]
        recent_volumes = volumes[-20:] if len(volumes) >= 20 else volumes

        if np.sum(recent_volumes) > 0:
            pivot = np.sum(recent_prices * recent_volumes) / np.sum(recent_volumes)
        else:
            pivot = np.mean(recent_prices)

        return float(key_support), float(key_resistance), float(pivot)
    
    def _forecast_volatility(self, metrics: RegimeMetrics) -> float:
        """Forecast next period volatility"""
        
        # Base forecast is current volatility
        forecast = metrics.realized_volatility
        
        # Adjust for volatility trend
        if metrics.volatility_trend > 0:
            forecast *= 1 + min(0.2, metrics.volatility_trend / 10)
        else:
            forecast *= 1 + max(-0.2, metrics.volatility_trend / 10)
        
        # Adjust for regime
        if metrics.volatility_regime == VolatilityState.DORMANT:
            forecast *= 1.2  # Expect increase from dormant
        elif metrics.volatility_regime == VolatilityState.EXTREME:
            forecast *= 0.9  # Expect decrease from extreme
        
        return forecast
    
    def _forecast_trend(self, metrics: RegimeMetrics) -> str:
        """Forecast trend direction"""
        
        # Higher thresholds for crypto
        if self.asset_type == "crypto":
            if metrics.trend_score > 40 and metrics.price_momentum > self.momentum_threshold:
                return "up"
            elif metrics.trend_score < -40 and metrics.price_momentum < -self.momentum_threshold:
                return "down"
            else:
                return "neutral"
        else:
            if metrics.trend_score > 20 and metrics.price_momentum > 0:
                return "up"
            elif metrics.trend_score < -20 and metrics.price_momentum < 0:
                return "down"
            else:
                return "neutral"
    
    def _suggest_strategy(self, regime: MarketRegime, metrics: RegimeMetrics) -> str:
        """Suggest trading strategy based on regime"""
        
        strategies = {
            MarketRegime.RANGING: "Grid trading optimal. Buy support, sell resistance. Mean reversion setups.",
            MarketRegime.TRENDING_UP: "Buy dips, trail stops. Avoid counter-trend shorts. Trend following.",
            MarketRegime.TRENDING_DOWN: "Sell rallies, trail stops. Avoid counter-trend longs. Trend following.",
            MarketRegime.VOLATILE_EXPANSION: "Reduce position size. Wide stops. Wait for stability.",
            MarketRegime.VOLATILE_CONTRACTION: "Prepare for breakout. Set alerts at range boundaries.",
            MarketRegime.BREAKOUT: "Follow breakout with stops below breakout level.",
            MarketRegime.BREAKDOWN: "Follow breakdown with stops above breakdown level.",
            MarketRegime.ACCUMULATION: "Accumulate on dips. Smart money is buying.",
            MarketRegime.DISTRIBUTION: "Reduce longs on rallies. Smart money is selling.",
            MarketRegime.UNCERTAIN: "Low confidence regime. Use smaller positions or wait.",
            MarketRegime.UNKNOWN: "Insufficient data. Wait for clearer signals."
        }
        
        base_strategy = strategies.get(regime, "Monitor closely.")
        
        # Add specific recommendations
        if metrics.mean_reversion_score > 70:
            base_strategy += " Strong mean reversion setup."
        if metrics.breakout_probability > 70:
            base_strategy += " High breakout probability - set alerts."
        if metrics.regime_stability < 0.3:
            base_strategy += " Regime unstable - expect change soon."
            
        return base_strategy
    
    def _assess_risk_level(self, metrics: RegimeMetrics, regime: MarketRegime) -> str:
        """Assess current risk level"""

        risk_score = 0

        # Adjust thresholds based on asset type
        if hasattr(self, 'asset_type') and self.asset_type == "crypto":
            # Much less sensitive for crypto
            if metrics.volatility_percentile > 95:  # Only top 5% is extreme
                risk_score += 40
            elif metrics.volatility_percentile > 85:  # Top 15% is high
                risk_score += 25
            elif metrics.volatility_percentile > 70:  # Top 30% is elevated
                risk_score += 15

            # Actual volatility check
            if metrics.realized_volatility > self.extreme_volatility_threshold:
                risk_score += 30
            elif metrics.realized_volatility > self.high_volatility_threshold:
                risk_score += 20
            elif metrics.realized_volatility > self.normal_volatility_threshold:
                risk_score += 10
        else:
            # Original logic for traditional assets
            if metrics.volatility_regime == VolatilityState.EXTREME:
                risk_score += 40
            elif metrics.volatility_regime == VolatilityState.HIGH:
                risk_score += 25
            elif metrics.volatility_regime == VolatilityState.ELEVATED:
                risk_score += 15
        
        # Regime risk
        if regime in [MarketRegime.VOLATILE_EXPANSION, MarketRegime.BREAKDOWN]:
            risk_score += 30
        elif regime in [MarketRegime.BREAKOUT, MarketRegime.DISTRIBUTION]:
            risk_score += 20
        elif regime == MarketRegime.UNCERTAIN:
            risk_score += 15  # Add risk for uncertain regime
        
        # Stability risk
        if metrics.regime_stability < 0.3:
            risk_score += 20
        
        # Extreme position risk
        if metrics.range_position > 0.9 or metrics.range_position < 0.1:
            risk_score += 15
        
        if hasattr(self, 'asset_type') and self.asset_type == "crypto":
            if risk_score >= 80:   # Was 60
                return "extreme"
            elif risk_score >= 60: # Was 40
                return "high"
            elif risk_score >= 40: # Was 20
                return "medium"
            else:
                return "low"
        else:
            # Original thresholds
            if risk_score >= 60:
                return "extreme"
            elif risk_score >= 40:
                return "high"
            elif risk_score >= 20:
                return "medium"
            else:
                return "low"
    
    def _check_warnings(self, metrics: RegimeMetrics, regime: MarketRegime) -> List[str]:
        """Check for any warning conditions"""
        
        warnings = []
        
        if self.asset_type == "crypto":
            # Crypto-specific warnings
            if metrics.realized_volatility > self.extreme_volatility_threshold:
                warnings.append(f"Extreme volatility detected ({metrics.realized_volatility:.1f}%) - reduce position sizes")
            elif metrics.realized_volatility > self.high_volatility_threshold:
                warnings.append(f"High volatility ({metrics.realized_volatility:.1f}%) - use wider stops")
        else:
            if metrics.volatility_regime == VolatilityState.EXTREME:
                warnings.append("Extreme volatility detected - reduce position sizes")
            
        if metrics.regime_stability < 0.2:
            warnings.append("Regime highly unstable - expect sudden changes")
            
        if metrics.breakout_probability > 80:
            warnings.append("High breakout probability - set stops carefully")
            
        if abs(metrics.smart_money_flow) > 0.5:
            direction = "buying" if metrics.smart_money_flow > 0 else "selling"
            warnings.append(f"Strong smart money {direction} detected")
            
        if metrics.hurst_exponent > 0.7:
            warnings.append("Strong trending behavior - avoid mean reversion trades")
        elif metrics.hurst_exponent < 0.3:
            warnings.append("Strong mean reverting behavior - avoid trend following")
            
        if metrics.volume_volatility > 2:
            warnings.append("Erratic volume - possible manipulation or news")
            
        if regime == MarketRegime.UNCERTAIN:
            warnings.append("Low regime confidence - trade with caution")
            
        return warnings
    
    def _insufficient_data_response(self) -> RegimeAnalysis:
        """Return response when insufficient data"""
        
        return RegimeAnalysis(
            primary_regime=MarketRegime.UNKNOWN,
            confidence=0.0,
            metrics=RegimeMetrics(
                price_momentum=0, price_acceleration=0, trend_consistency=0.5,
                range_position=0.5, realized_volatility=0, 
                volatility_regime=VolatilityState.NORMAL,
                volatility_percentile=50, volatility_trend=0,
                volume_trend=1, volume_volatility=0, smart_money_flow=0,
                hurst_exponent=0.5, autocorrelation=0, efficiency_ratio=0.5,
                fractal_dimension=1.5, support_strength=0.5, resistance_strength=0.5,
                level_clustering=0.5, trend_score=0, mean_reversion_score=50,
                breakout_probability=0, regime_stability=0.5,
                adf_statistic=0.0, adf_pvalue=1.0, halflife_periods=9999.0,
                is_mean_reverting=False
            ),
            microstructure=None,
            suggested_strategy="Insufficient data for analysis. Need more price history.",
            risk_level="unknown",
            regime_duration=0,
            regime_strength=0,
            transition_probability=0,
            key_support=0,
            key_resistance=0,
            pivot_point=0,
            volatility_forecast=0,
            trend_forecast="unknown",
            warnings=["Insufficient data for reliable analysis"]
        )

# =====================================================
# USAGE EXAMPLE
# =====================================================

def example_usage():
    """Example of how to use the market regime detector"""
    
    # Initialize detector for crypto
    detector = MarketRegimeDetector(
        lookback_periods=100,
        min_periods=30,
        volatility_window=20,
        trend_window=20,
        asset_type="crypto"  # Important for crypto markets
    )
    
    # Example data (you would use real market data)
    prices = np.random.randn(200).cumsum() + 100
    volumes = np.random.rand(200) * 1000000
    
    # Perform analysis
    analysis = detector.analyze(prices, volumes)
    
    # Print results
    print(f"Market Regime: {analysis.primary_regime.value}")
    print(f"Confidence: {analysis.confidence:.1f}%")
    print(f"Risk Level: {analysis.risk_level}")
    print(f"Strategy: {analysis.suggested_strategy}")
    print(f"\nKey Levels:")
    print(f"  Support: ${analysis.key_support:.2f}")
    print(f"  Resistance: ${analysis.key_resistance:.2f}")
    print(f"  Pivot: ${analysis.pivot_point:.2f}")
    print(f"\nMetrics:")
    print(f"  Trend Score: {analysis.metrics.trend_score:.1f}")
    print(f"  Momentum: {analysis.metrics.price_momentum:.1f}%")
    print(f"  Mean Reversion Score: {analysis.metrics.mean_reversion_score:.1f}")
    print(f"  Breakout Probability: {analysis.metrics.breakout_probability:.1f}%")
    print(f"  Volatility: {analysis.metrics.realized_volatility:.1f}%")
    
    if analysis.warnings:
        print(f"\nWarnings:")
        for warning in analysis.warnings:
            print(f"  ⚠️  {warning}")

if __name__ == "__main__":
    example_usage()