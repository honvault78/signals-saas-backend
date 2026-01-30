"""
Bavella Validity Engine — Failure Mode Detector
================================================

Detects FM1-FM7 failure modes using your existing MarketRegimeDetector output
plus additional statistical tests.

Integration point: Takes regime analysis + raw returns → failure mode detection
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats

from .core import (
    FailureMode,
    ValidityState,
    ValidityVerdict,
    FM_INFO,
    calculate_validity_score,
)

# Try to import statsmodels for ADF test
try:
    from statsmodels.tsa.stattools import adfuller
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False


# =============================================================================
# DETECTOR RESULT
# =============================================================================

@dataclass
class ActiveFailureMode:
    """An active failure mode with its detection details."""
    failure_mode: FailureMode
    severity: float  # 0-100
    confidence: float  # 0-1
    triggered_at: datetime
    evidence: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "failure_mode": self.failure_mode.value,
            "name": FM_INFO[self.failure_mode]["name"],
            "description": FM_INFO[self.failure_mode]["description"],
            "severity": round(self.severity, 1),
            "confidence": round(self.confidence, 2),
            "triggered_at": self.triggered_at.isoformat(),
            "evidence": self.evidence,
            "typically_reversible": FM_INFO[self.failure_mode]["typically_reversible"],
        }


@dataclass
class DetectorResult:
    """Complete result from validity detection."""
    # Core verdict
    validity_score: float
    validity_state: ValidityState
    
    # Active failure modes
    active_fms: List[ActiveFailureMode]
    primary_fm: Optional[ActiveFailureMode]
    
    # Confidence
    overall_confidence: float
    
    # Context from regime detection
    regime: str
    adf_pvalue: float
    halflife: float
    z_score: float
    
    # Metadata
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "validity_score": round(self.validity_score, 1),
            "validity_state": self.validity_state.value,
            "active_failure_modes": [fm.to_dict() for fm in self.active_fms],
            "primary_failure_mode": self.primary_fm.to_dict() if self.primary_fm else None,
            "overall_confidence": round(self.overall_confidence, 2),
            "context": {
                "regime": self.regime,
                "adf_pvalue": round(self.adf_pvalue, 4),
                "halflife": round(self.halflife, 1),
                "z_score": round(self.z_score, 2),
            },
            "detected_at": self.detected_at.isoformat(),
            "fm_count": len(self.active_fms),
            "is_valid": self.validity_state == ValidityState.VALID,
        }
    
    def to_verdict(self, node_id: str) -> ValidityVerdict:
        """Convert to immutable ValidityVerdict."""
        return ValidityVerdict.create(
            node_id=node_id,
            validity_score=self.validity_score,
            active_fms=[fm.failure_mode for fm in self.active_fms],
            primary_fm=self.primary_fm.failure_mode if self.primary_fm else None,
            confidence=self.overall_confidence,
            regime=self.regime,
            evidence_summary=self._build_evidence_summary(),
        )
    
    def _build_evidence_summary(self) -> str:
        if not self.active_fms:
            return "No failure modes detected. Relationship is epistemically valid."
        
        parts = [f"{len(self.active_fms)} failure mode(s) active:"]
        for fm in self.active_fms[:3]:  # Top 3
            parts.append(f"• {FM_INFO[fm.failure_mode]['name']} (severity={fm.severity:.0f})")
        
        if len(self.active_fms) > 3:
            parts.append(f"• ...and {len(self.active_fms) - 3} more")
        
        return " ".join(parts)


# =============================================================================
# VALIDITY DETECTOR
# =============================================================================

class ValidityDetector:
    """
    Detects validity failure modes from market data and regime analysis.
    
    Integration:
        Takes output from your MarketRegimeDetector and adds FM1-FM7 detection.
    
    Usage:
        detector = ValidityDetector()
        result = detector.detect(
            returns=daily_returns,
            regime_analysis=regime_analysis,  # From MarketRegimeDetector
        )
    """
    
    # Detection thresholds
    FM1_VOL_RATIO_THRESHOLD = 1.8      # 80% vol increase triggers FM1
    FM2_DRIFT_ZSCORE = 2.5             # Z-score of mean drift
    FM3_CORR_CHANGE_THRESHOLD = 0.4    # Correlation change magnitude
    FM4_ADF_THRESHOLD = 0.30           # ADF p-value for stationarity (tighter for structural)
    FM5_ADF_THRESHOLD = 0.50           # ADF p-value for stationarity loss
    FM6_TAIL_ZSCORE = 3.5              # Z-score for tail event
    FM7_HALFLIFE_THRESHOLD = 60        # Half-life above this = dependency break
    
    def __init__(self):
        self.lookback_short = 20   # Short window
        self.lookback_long = 60    # Long window
    
    def detect(
        self,
        returns: pd.Series,
        regime_analysis: Optional[Any] = None,
        adf_pvalue: Optional[float] = None,
        halflife: Optional[float] = None,
        z_score: Optional[float] = None,
        regime: str = "unknown",
        reference_returns: Optional[pd.Series] = None,
    ) -> DetectorResult:
        """
        Detect all failure modes.
        
        Args:
            returns: Daily returns series
            regime_analysis: Output from MarketRegimeDetector (optional)
            adf_pvalue: ADF test p-value (from regime detection)
            halflife: Mean reversion half-life (from regime detection)
            z_score: Current Z-score (from regime detection)
            regime: Current regime string
            reference_returns: Reference series for correlation (FM3, FM7)
        """
        now = datetime.now(timezone.utc)
        active_fms: List[ActiveFailureMode] = []
        
        # Extract values from regime_analysis if provided
        if regime_analysis is not None:
            metrics = getattr(regime_analysis, 'metrics', None)
            if metrics:
                adf_pvalue = adf_pvalue or getattr(metrics, 'adf_pvalue', None)
                halflife = halflife or getattr(metrics, 'halflife_periods', None)
            regime = regime_analysis.primary_regime.value if hasattr(regime_analysis, 'primary_regime') else regime
        
        # Default values
        adf_pvalue = adf_pvalue if adf_pvalue is not None else 0.5
        halflife = halflife if halflife is not None else 30
        z_score = z_score if z_score is not None else 0
        
        # Ensure we have enough data
        if len(returns) < self.lookback_long:
            return self._insufficient_data_result(regime, adf_pvalue, halflife, z_score)
        
        # Clean returns
        returns = returns.dropna()
        
        # ===== FM1: Variance Regime Shift =====
        fm1_result = self._detect_fm1_variance(returns, now)
        if fm1_result:
            active_fms.append(fm1_result)
        
        # ===== FM2: Mean Drift =====
        fm2_result = self._detect_fm2_drift(returns, now)
        if fm2_result:
            active_fms.append(fm2_result)
        
        # ===== FM3: Correlation Flip =====
        if reference_returns is not None:
            fm3_result = self._detect_fm3_correlation(returns, reference_returns, now)
            if fm3_result:
                active_fms.append(fm3_result)
        
        # ===== FM4: Structural Break =====
        fm4_result = self._detect_fm4_structural(returns, adf_pvalue, now)
        if fm4_result:
            active_fms.append(fm4_result)
        
        # ===== FM5: Stationarity Loss =====
        fm5_result = self._detect_fm5_stationarity(returns, adf_pvalue, now)
        if fm5_result:
            active_fms.append(fm5_result)
        
        # ===== FM6: Tail Event =====
        fm6_result = self._detect_fm6_tail(returns, now)
        if fm6_result:
            active_fms.append(fm6_result)
        
        # ===== FM7: Dependency Break =====
        fm7_result = self._detect_fm7_dependency(halflife, adf_pvalue, now)
        if fm7_result:
            active_fms.append(fm7_result)
        
        # Calculate validity score
        fm_severities = [(fm.failure_mode, fm.severity) for fm in active_fms]
        validity_score = calculate_validity_score(fm_severities)
        validity_state = ValidityState.from_score(validity_score)
        
        # Determine primary FM (highest severity)
        primary_fm = None
        if active_fms:
            primary_fm = max(active_fms, key=lambda x: x.severity)
        
        # Overall confidence (average of FM confidences, weighted by severity)
        if active_fms:
            total_weight = sum(fm.severity for fm in active_fms)
            if total_weight > 0:
                overall_confidence = sum(
                    fm.confidence * fm.severity for fm in active_fms
                ) / total_weight
            else:
                overall_confidence = 0.8
        else:
            overall_confidence = 0.9  # High confidence in validity when no FMs
        
        return DetectorResult(
            validity_score=validity_score,
            validity_state=validity_state,
            active_fms=active_fms,
            primary_fm=primary_fm,
            overall_confidence=overall_confidence,
            regime=regime,
            adf_pvalue=adf_pvalue,
            halflife=halflife,
            z_score=z_score,
        )
    
    def _detect_fm1_variance(
        self, returns: pd.Series, now: datetime
    ) -> Optional[ActiveFailureMode]:
        """FM1: Variance Regime Shift"""
        recent_vol = returns.iloc[-self.lookback_short:].std()
        long_vol = returns.iloc[-self.lookback_long:].std()
        
        if long_vol > 0:
            vol_ratio = recent_vol / long_vol
            
            if vol_ratio > self.FM1_VOL_RATIO_THRESHOLD:
                severity = min(100, (vol_ratio - 1) * 50)
                return ActiveFailureMode(
                    failure_mode=FailureMode.FM1_VARIANCE_REGIME,
                    severity=severity,
                    confidence=0.85,
                    triggered_at=now,
                    evidence={
                        "recent_vol": float(recent_vol),
                        "long_vol": float(long_vol),
                        "vol_ratio": float(vol_ratio),
                    },
                )
            elif vol_ratio < 1 / self.FM1_VOL_RATIO_THRESHOLD:
                # Vol compression can also be a regime shift
                severity = min(50, (1 / vol_ratio - 1) * 25)
                return ActiveFailureMode(
                    failure_mode=FailureMode.FM1_VARIANCE_REGIME,
                    severity=severity,
                    confidence=0.7,
                    triggered_at=now,
                    evidence={
                        "recent_vol": float(recent_vol),
                        "long_vol": float(long_vol),
                        "vol_ratio": float(vol_ratio),
                        "type": "compression",
                    },
                )
        
        return None
    
    def _detect_fm2_drift(
        self, returns: pd.Series, now: datetime
    ) -> Optional[ActiveFailureMode]:
        """FM2: Mean Drift"""
        cumulative = (1 + returns).cumprod()
        
        # Rolling mean
        long_mean = cumulative.iloc[-self.lookback_long:].mean()
        recent_mean = cumulative.iloc[-self.lookback_short:].mean()
        
        # Standard deviation of cumulative
        long_std = cumulative.iloc[-self.lookback_long:].std()
        
        if long_std > 0:
            drift_zscore = abs(recent_mean - long_mean) / long_std
            
            if drift_zscore > self.FM2_DRIFT_ZSCORE:
                severity = min(80, (drift_zscore - self.FM2_DRIFT_ZSCORE) * 20 + 30)
                return ActiveFailureMode(
                    failure_mode=FailureMode.FM2_MEAN_DRIFT,
                    severity=severity,
                    confidence=0.75,
                    triggered_at=now,
                    evidence={
                        "drift_zscore": float(drift_zscore),
                        "recent_mean": float(recent_mean),
                        "long_mean": float(long_mean),
                    },
                )
        
        return None
    
    def _detect_fm3_correlation(
        self, returns: pd.Series, reference: pd.Series, now: datetime
    ) -> Optional[ActiveFailureMode]:
        """FM3: Correlation Flip"""
        # Align series
        aligned = pd.concat([returns, reference], axis=1).dropna()
        if len(aligned) < self.lookback_long:
            return None
        
        ret1, ret2 = aligned.iloc[:, 0], aligned.iloc[:, 1]
        
        # Long-term correlation
        long_corr = ret1.iloc[-self.lookback_long:].corr(ret2.iloc[-self.lookback_long:])
        
        # Recent correlation
        recent_corr = ret1.iloc[-self.lookback_short:].corr(ret2.iloc[-self.lookback_short:])
        
        corr_change = abs(recent_corr - long_corr)
        
        if corr_change > self.FM3_CORR_CHANGE_THRESHOLD:
            # Sign flip is more severe
            sign_flip = (long_corr * recent_corr) < 0
            severity = min(100, corr_change * 100 + (30 if sign_flip else 0))
            
            return ActiveFailureMode(
                failure_mode=FailureMode.FM3_CORRELATION_FLIP,
                severity=severity,
                confidence=0.8,
                triggered_at=now,
                evidence={
                    "long_correlation": float(long_corr),
                    "recent_correlation": float(recent_corr),
                    "change": float(corr_change),
                    "sign_flip": sign_flip,
                },
            )
        
        return None
    
    def _detect_fm4_structural(
        self, returns: pd.Series, adf_pvalue: float, now: datetime
    ) -> Optional[ActiveFailureMode]:
        """FM4: Structural Break (permanent)"""
        # Use tighter ADF threshold + additional structural tests
        
        # Check for sudden level shift (Chow-like test simplified)
        cumulative = (1 + returns).cumprod()
        midpoint = len(cumulative) // 2
        
        first_half_mean = cumulative.iloc[:midpoint].mean()
        second_half_mean = cumulative.iloc[midpoint:].mean()
        overall_std = cumulative.std()
        
        if overall_std > 0:
            level_shift = abs(second_half_mean - first_half_mean) / overall_std
            
            # Structural break if: large level shift AND non-stationary
            if level_shift > 2.5 and adf_pvalue > self.FM4_ADF_THRESHOLD:
                severity = min(100, level_shift * 20 + adf_pvalue * 50)
                return ActiveFailureMode(
                    failure_mode=FailureMode.FM4_STRUCTURAL_BREAK,
                    severity=severity,
                    confidence=0.85,
                    triggered_at=now,
                    evidence={
                        "level_shift_zscore": float(level_shift),
                        "adf_pvalue": float(adf_pvalue),
                        "first_half_mean": float(first_half_mean),
                        "second_half_mean": float(second_half_mean),
                    },
                )
        
        return None
    
    def _detect_fm5_stationarity(
        self, returns: pd.Series, adf_pvalue: float, now: datetime
    ) -> Optional[ActiveFailureMode]:
        """FM5: Stationarity Loss"""
        # High ADF p-value = non-stationary = stationarity loss
        if adf_pvalue > self.FM5_ADF_THRESHOLD:
            # Severity scales with how non-stationary
            severity = min(80, (adf_pvalue - self.FM5_ADF_THRESHOLD) * 160)
            
            return ActiveFailureMode(
                failure_mode=FailureMode.FM5_STATIONARITY_LOSS,
                severity=severity,
                confidence=0.9 if STATSMODELS_AVAILABLE else 0.7,
                triggered_at=now,
                evidence={
                    "adf_pvalue": float(adf_pvalue),
                    "threshold": self.FM5_ADF_THRESHOLD,
                },
            )
        
        return None
    
    def _detect_fm6_tail(
        self, returns: pd.Series, now: datetime
    ) -> Optional[ActiveFailureMode]:
        """FM6: Tail Event (recent extreme move)"""
        # Check last few days for extreme moves
        recent_returns = returns.iloc[-5:]
        long_std = returns.iloc[-self.lookback_long:].std()
        long_mean = returns.iloc[-self.lookback_long:].mean()
        
        if long_std > 0:
            z_scores = (recent_returns - long_mean) / long_std
            max_zscore = z_scores.abs().max()
            
            if max_zscore > self.FM6_TAIL_ZSCORE:
                severity = min(60, (max_zscore - self.FM6_TAIL_ZSCORE) * 15 + 20)
                return ActiveFailureMode(
                    failure_mode=FailureMode.FM6_TAIL_EVENT,
                    severity=severity,
                    confidence=0.95,
                    triggered_at=now,
                    evidence={
                        "max_zscore": float(max_zscore),
                        "extreme_return": float(recent_returns.iloc[z_scores.abs().argmax()]),
                        "days_ago": int(len(recent_returns) - z_scores.abs().argmax() - 1),
                    },
                )
        
        return None
    
    def _detect_fm7_dependency(
        self, halflife: float, adf_pvalue: float, now: datetime
    ) -> Optional[ActiveFailureMode]:
        """FM7: Dependency Break"""
        # Long half-life + non-stationarity = dependency break
        if halflife > self.FM7_HALFLIFE_THRESHOLD and adf_pvalue > 0.3:
            severity = min(90, (halflife / self.FM7_HALFLIFE_THRESHOLD) * 30 + adf_pvalue * 40)
            
            return ActiveFailureMode(
                failure_mode=FailureMode.FM7_DEPENDENCY_BREAK,
                severity=severity,
                confidence=0.8,
                triggered_at=now,
                evidence={
                    "halflife": float(halflife),
                    "adf_pvalue": float(adf_pvalue),
                    "threshold": self.FM7_HALFLIFE_THRESHOLD,
                },
            )
        
        return None
    
    def _insufficient_data_result(
        self, regime: str, adf_pvalue: float, halflife: float, z_score: float
    ) -> DetectorResult:
        """Return result when insufficient data."""
        return DetectorResult(
            validity_score=50,  # Uncertain
            validity_state=ValidityState.DEGRADED,
            active_fms=[],
            primary_fm=None,
            overall_confidence=0.3,
            regime=regime,
            adf_pvalue=adf_pvalue,
            halflife=halflife,
            z_score=z_score,
        )
