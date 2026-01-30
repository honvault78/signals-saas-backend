"""
Bavella v2 — Normalized Innovation Space (NIS) Transforms
==========================================================

This module implements the core transformation layer that converts any
time series into a unitless, comparable representation.

Key principle: "All statistical analysis should operate on increments
relative to variability, not relative to level."

The transform pipeline:
    RAW SERIES → DIFFERENCING → SCALE ESTIMATION → NORMALIZATION → ANALYSIS

Copyright 2024-2026 Bavella Technologies Sarl
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import periodogram, find_peaks
from typing import Tuple, Optional, Dict, Any, List
from dataclasses import dataclass, field
import warnings

# Conditional imports for optional dependencies
try:
    from statsmodels.tsa.seasonal import STL
    from statsmodels.tsa.stattools import adfuller, kpss
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

from .models import (
    SeriesType, ScaleMethod, SeasonalityHandling, TransformSpec
)


# =============================================================================
# CONSTANTS
# =============================================================================

MIN_OBSERVATIONS = 30
RECOMMENDED_OBSERVATIONS = 100
SEASONALITY_DETECTION_THRESHOLD = 0.15
EWMA_SPAN = 20
ROBUST_SCALE_FACTOR = 1.4826


# =============================================================================
# TRANSFORM RESULT DATACLASS
# =============================================================================

@dataclass
class NISTransformResult:
    """
    Complete output of the NIS transformation pipeline.
    
    Contains all artifacts needed for downstream analysis:
    - z_t: Normalized innovations (shocks relative to local variability)
    - Z_t: Cumulative normalized process (for regime analysis)
    - Components: trend, seasonal, residual
    - Metadata: transform parameters and diagnostics
    """
    # Primary outputs
    z_t: pd.Series  # Normalized innovations
    Z_t: pd.Series  # Cumulative normalized process
    
    # Decomposition components
    trend: Optional[pd.Series] = None
    seasonal: Optional[pd.Series] = None
    residual: Optional[pd.Series] = None
    
    # Scale estimates
    scale_series: Optional[pd.Series] = None
    
    # Original data reference
    original_series: Optional[pd.Series] = None
    differences: Optional[pd.Series] = None
    
    # Transform metadata
    transform_spec: Optional[TransformSpec] = None
    series_type_detected: SeriesType = SeriesType.UNKNOWN
    type_confidence: float = 0.0
    
    # Diagnostics
    seasonality_detected: bool = False
    dominant_period: Optional[int] = None
    seasonality_strength: float = 0.0
    
    # Warnings
    warnings: List[str] = field(default_factory=list)
    
    @property
    def is_valid(self) -> bool:
        """Check if transform produced valid outputs."""
        return (
            self.z_t is not None 
            and len(self.z_t) > 0 
            and not self.z_t.isna().all()
        )


# =============================================================================
# SERIES TYPE DETECTION
# =============================================================================

class SeriesTypeDetector:
    """
    Auto-detects series behavior type to determine optimal transform.
    """
    
    def __init__(self, series: pd.Series):
        self.series = series.dropna()
        self.n = len(self.series)
        
    def detect(self) -> Tuple[SeriesType, float, Dict[str, Any]]:
        """Detect series type with confidence score."""
        if self.n < MIN_OBSERVATIONS:
            return SeriesType.UNKNOWN, 0.0, {"error": "Insufficient observations"}
        
        evidence = {}
        scores = {st: 0.0 for st in SeriesType}
        
        # Test 1: Can be negative?
        min_val = self.series.min()
        max_val = self.series.max()
        evidence["can_be_negative"] = min_val < 0
        evidence["crosses_zero"] = (min_val < 0) and (max_val > 0)
        
        if min_val < 0:
            scores[SeriesType.MULTIPLICATIVE_PRICE] -= 100
            scores[SeriesType.ADDITIVE_FLOW] += 30
            scores[SeriesType.CUMULATIVE_BALANCE] += 20
            
        # Test 2: Variance-level relationship
        if min_val > 0:
            var_level_corr = self._test_variance_level_relationship()
            evidence["variance_level_correlation"] = var_level_corr
            
            if var_level_corr > 0.5:
                scores[SeriesType.MULTIPLICATIVE_PRICE] += 40
            elif var_level_corr < 0.2:
                scores[SeriesType.ADDITIVE_FLOW] += 20
                
        # Test 3: Seasonality
        seasonality_strength, dominant_period = self._detect_seasonality()
        evidence["seasonality_strength"] = seasonality_strength
        
        if seasonality_strength > SEASONALITY_DETECTION_THRESHOLD:
            scores[SeriesType.SEASONAL_BILL] += 30
                
        # Determine winner
        best_type = max(scores, key=scores.get)
        max_score = scores[best_type]
        
        if max_score <= 0:
            best_type = SeriesType.IRREGULAR_SIGNAL
            confidence = 0.5
        else:
            total_positive = sum(s for s in scores.values() if s > 0)
            confidence = min(0.95, max_score / max(total_positive, 1))
            
        return best_type, confidence, evidence
    
    def _test_variance_level_relationship(self) -> float:
        """Test if variance is proportional to level."""
        window = min(20, self.n // 5)
        if window < 5:
            return 0.0
            
        rolling_mean = self.series.rolling(window).mean()
        rolling_std = self.series.rolling(window).std()
        
        valid_idx = ~(rolling_mean.isna() | rolling_std.isna())
        if valid_idx.sum() < 10:
            return 0.0
            
        corr = np.corrcoef(
            rolling_mean[valid_idx].values,
            rolling_std[valid_idx].values
        )[0, 1]
        
        return float(corr) if not np.isnan(corr) else 0.0
    
    def _detect_seasonality(self) -> Tuple[float, Optional[int]]:
        """Detect dominant periodicity using spectral analysis."""
        if self.n < 50:
            return 0.0, None
            
        try:
            detrended = self.series - self.series.rolling(
                min(20, self.n // 5), center=True
            ).mean()
            detrended = detrended.dropna()
            
            if len(detrended) < 30:
                return 0.0, None
            
            freqs, power = periodogram(detrended.values, scaling='spectrum')
            
            if len(power) < 3:
                return 0.0, None
                
            peaks, _ = find_peaks(power[1:], height=np.median(power[1:]) * 2)
            
            if len(peaks) == 0:
                return 0.0, None
                
            peak_idx = peaks[np.argmax(power[1:][peaks])] + 1
            dominant_freq = freqs[peak_idx]
            
            if dominant_freq > 0:
                dominant_period = int(round(1 / dominant_freq))
            else:
                return 0.0, None
                
            seasonality_strength = power[peak_idx] / power.sum()
            
            return float(seasonality_strength), dominant_period
            
        except Exception:
            return 0.0, None


# =============================================================================
# SCALE ESTIMATION
# =============================================================================

class ScaleEstimator:
    """Estimates local variability for normalization."""
    
    def __init__(
        self,
        method: ScaleMethod = ScaleMethod.EWMA,
        lookback: int = EWMA_SPAN,
        min_periods: int = 5
    ):
        self.method = method
        self.lookback = lookback
        self.min_periods = min_periods
        
    def estimate(self, series: pd.Series) -> pd.Series:
        """Estimate local scale (volatility)."""
        if self.method == ScaleMethod.EWMA:
            return series.ewm(span=self.lookback, min_periods=self.min_periods).std()
        elif self.method == ScaleMethod.ROLLING_STD:
            return series.rolling(window=self.lookback, min_periods=self.min_periods).std()
        elif self.method == ScaleMethod.MAD:
            def mad(x):
                median = np.median(x)
                return np.median(np.abs(x - median)) * ROBUST_SCALE_FACTOR
            return series.rolling(window=self.lookback, min_periods=self.min_periods).apply(mad, raw=True)
        else:
            return series.ewm(span=self.lookback, min_periods=self.min_periods).std()


# =============================================================================
# MAIN TRANSFORM PIPELINE
# =============================================================================

class NISTransformer:
    """
    Main transformation pipeline: Raw Series → Normalized Innovation Space.
    """
    
    def __init__(self, spec: Optional[TransformSpec] = None):
        self.spec = spec or TransformSpec()
        
    def transform(
        self,
        series: pd.Series,
        series_type_override: Optional[SeriesType] = None
    ) -> NISTransformResult:
        """Execute the full NIS transformation pipeline."""
        result = NISTransformResult(
            z_t=pd.Series(dtype=float),
            Z_t=pd.Series(dtype=float),
            original_series=series.copy(),
            transform_spec=self.spec
        )
        
        series = series.dropna()
        n = len(series)
        
        if n < MIN_OBSERVATIONS:
            result.warnings.append(f"Insufficient observations ({n}). Minimum {MIN_OBSERVATIONS} required.")
            return result
            
        if n < RECOMMENDED_OBSERVATIONS:
            result.warnings.append(f"Limited observations ({n}). Recommend {RECOMMENDED_OBSERVATIONS}+.")
            
        # Step 1: Detect series type
        if series_type_override:
            series_type = series_type_override
            type_confidence = 1.0
        else:
            detector = SeriesTypeDetector(series)
            series_type, type_confidence, _ = detector.detect()
            
        result.series_type_detected = series_type
        result.type_confidence = type_confidence
        
        # Step 2: Apply differencing
        if series_type == SeriesType.MULTIPLICATIVE_PRICE and (series > 0).all():
            differences = np.log(series).diff()
        else:
            differences = series.diff()
            
        differences = differences.dropna()
        result.differences = differences
        
        # Step 3: Handle seasonality (simplified)
        to_normalize = differences
        
        # Step 4: Estimate local scale
        scale_estimator = ScaleEstimator(
            method=self.spec.scale_method,
            lookback=self.spec.scale_lookback
        )
        scale = scale_estimator.estimate(to_normalize)
        result.scale_series = scale
        
        # Step 5: Normalize to produce z_t
        scale_safe = scale.replace(0, np.nan).fillna(scale.median())
        
        if scale_safe.isna().all() or (scale_safe == 0).all():
            global_scale = max(to_normalize.std(), 1e-10)
            z_t = to_normalize / global_scale
            result.warnings.append("Constant local scale detected. Using global normalization.")
        else:
            z_t = to_normalize / scale_safe
            
        z_t = z_t.replace([np.inf, -np.inf], np.nan).fillna(0)
        result.z_t = z_t
        
        # Step 6: Compute cumulative Z_t
        result.Z_t = z_t.cumsum()
        
        return result


# =============================================================================
# STATISTICAL TESTS
# =============================================================================

class TransformedSeriesAnalyzer:
    """Runs statistical tests on the transformed (NIS) series."""
    
    def __init__(self, transform_result: NISTransformResult):
        self.result = transform_result
        self.z_t = transform_result.z_t
        self.Z_t = transform_result.Z_t
        
    def compute_all_statistics(self) -> Dict[str, Any]:
        """Compute comprehensive statistics for Series Passport."""
        return {
            "z_t_stats": self._compute_basic_stats(self.z_t),
            "stationarity": self._test_stationarity(),
            "mean_reversion": self._compute_mean_reversion_metrics(),
            "tail_risk": self._compute_tail_metrics(),
            "jumps": self._compute_jump_metrics(),
        }
        
    def _compute_basic_stats(self, series: pd.Series) -> Dict[str, float]:
        series = series.dropna()
        if len(series) == 0:
            return {}
        return {
            "mean": float(series.mean()),
            "std": float(series.std()),
            "skewness": float(stats.skew(series)),
            "kurtosis": float(stats.kurtosis(series)),
            "min": float(series.min()),
            "max": float(series.max()),
            "median": float(series.median()),
        }
        
    def _test_stationarity(self) -> Dict[str, Any]:
        result = {"is_stationary": False, "adf_pvalue": 1.0, "kpss_pvalue": 0.0}
        
        if not HAS_STATSMODELS:
            result["error"] = "statsmodels not available"
            return result
        
        series = self.Z_t.dropna()
        if len(series) < 30:
            result["error"] = "Insufficient observations"
            return result
            
        try:
            adf_result = adfuller(series, autolag='AIC')
            result["adf_statistic"] = float(adf_result[0])
            result["adf_pvalue"] = float(adf_result[1])
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                kpss_result = kpss(series, regression='c', nlags='auto')
            result["kpss_statistic"] = float(kpss_result[0])
            result["kpss_pvalue"] = float(kpss_result[1])
            
            result["is_stationary"] = (result["adf_pvalue"] < 0.05) and (result["kpss_pvalue"] > 0.05)
        except Exception as e:
            result["error"] = str(e)
            
        return result
        
    def _compute_mean_reversion_metrics(self) -> Dict[str, Any]:
        result = {"hurst_exponent": 0.5, "half_life": None, "is_mean_reverting": False}
        
        series = self.Z_t.dropna()
        if len(series) < 50:
            result["error"] = "Insufficient observations"
            return result
            
        try:
            result["hurst_exponent"] = self._compute_hurst(series)
            result["is_mean_reverting"] = result["hurst_exponent"] < 0.5
            result["half_life"] = self._compute_half_life(series)
        except Exception as e:
            result["error"] = str(e)
            
        return result
        
    def _compute_hurst(self, series: pd.Series) -> float:
        n = len(series)
        max_k = min(n // 2, 100)
        
        if max_k < 10:
            return 0.5
            
        rs_list, n_list = [], []
        
        for k in range(10, max_k + 1, max(1, max_k // 20)):
            rs_values = []
            for i in range(0, n - k, k):
                window = series.iloc[i:i+k].values
                mean = np.mean(window)
                std = np.std(window, ddof=1)
                
                if std == 0:
                    continue
                    
                cumsum = np.cumsum(window - mean)
                r = np.max(cumsum) - np.min(cumsum)
                rs_values.append(r / std)
                
            if rs_values:
                rs_list.append(np.mean(rs_values))
                n_list.append(k)
                
        if len(rs_list) < 3:
            return 0.5
            
        slope, _, _, _, _ = stats.linregress(np.log(n_list), np.log(rs_list))
        return float(np.clip(slope, 0, 1))
        
    def _compute_half_life(self, series: pd.Series) -> Optional[float]:
        if len(series) < 30:
            return None
            
        y = series.values
        y_lag = y[:-1]
        dy = np.diff(y)
        
        y_lag_demeaned = y_lag - np.mean(y_lag)
        
        if np.var(y_lag_demeaned) == 0:
            return None
            
        theta = np.sum(dy * y_lag_demeaned) / np.sum(y_lag_demeaned ** 2)
        
        if theta >= 0 or 1 + theta <= 0:
            return None
            
        half_life = -np.log(2) / np.log(1 + theta)
        
        if half_life < 0 or half_life > len(series):
            return None
            
        return float(half_life)
        
    def _compute_tail_metrics(self) -> Dict[str, Any]:
        series = self.z_t.dropna()
        n = len(series)
        
        if n < 30:
            return {"error": "Insufficient observations"}
            
        values = series.values
        kurtosis = stats.kurtosis(values)
        
        if kurtosis < 0:
            tail_class = "light"
        elif kurtosis < 3:
            tail_class = "normal"
        elif kurtosis < 10:
            tail_class = "heavy"
        else:
            tail_class = "extreme"
            
        return {
            "var_95_left": float(np.percentile(values, 5)),
            "kurtosis": float(kurtosis),
            "tail_classification": tail_class,
        }
        
    def _compute_jump_metrics(self) -> Dict[str, Any]:
        series = self.z_t.dropna()
        n = len(series)
        
        if n < 20:
            return {"error": "Insufficient observations"}
            
        values = series.values
        median = np.median(values)
        mad = np.median(np.abs(values - median))
        
        if mad == 0:
            return {"jump_count": 0, "jump_frequency": 0.0}
            
        threshold = 3 * mad * ROBUST_SCALE_FACTOR
        jumps = np.abs(values - median) > threshold
        
        return {
            "jump_count": int(jumps.sum()),
            "jump_frequency": float(jumps.sum() / n),
            "largest_jump": float(np.max(np.abs(values - median)) / (mad * ROBUST_SCALE_FACTOR)),
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def transform_series(
    series: pd.Series,
    series_type: Optional[SeriesType] = None,
    scale_method: ScaleMethod = ScaleMethod.EWMA,
) -> NISTransformResult:
    """Convenience function to transform a series with default settings."""
    spec = TransformSpec(scale_method=scale_method)
    transformer = NISTransformer(spec)
    return transformer.transform(series, series_type_override=series_type)


def quick_passport(series: pd.Series) -> Dict[str, Any]:
    """Generate a quick Series Passport summary."""
    result = transform_series(series)
    analyzer = TransformedSeriesAnalyzer(result)
    stats_dict = analyzer.compute_all_statistics()
    
    return {
        "series_type": result.series_type_detected.value,
        "type_confidence": result.type_confidence,
        "observation_count": len(series),
        "seasonality_detected": result.seasonality_detected,
        "is_stationary": stats_dict.get("stationarity", {}).get("is_stationary", False),
        "hurst_exponent": stats_dict.get("mean_reversion", {}).get("hurst_exponent", 0.5),
        "half_life": stats_dict.get("mean_reversion", {}).get("half_life"),
        "tail_risk": stats_dict.get("tail_risk", {}).get("tail_classification", "unknown"),
        "warnings": result.warnings,
    }
