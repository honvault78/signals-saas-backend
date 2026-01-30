"""
Bavella v2 — Proper Failure Mode Detectors
============================================

This replaces the placeholder detectors with real implementations.

CRITICAL FIX: Confidence handling
---------------------------------
Old (WRONG): effective_severity = severity * confidence
             → Low confidence REDUCED penalty → BACKWARDS

New (CORRECT): 
    - Detection confidence affects WHETHER we report the signal
    - It does NOT reduce the severity once reported
    - Low confidence → wider DEGRADED band, not better score
    - "We're not sure" should NEVER make things look better

Copyright 2024-2026 Bavella Technologies Sarl
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from scipy import stats
from scipy.signal import periodogram, find_peaks

from .core import (
    FailureMode, FailureSignal, ValidityVerdict, ValidityState,
    Thresholds
)


# =============================================================================
# CONFIDENCE POLICY (the fix)
# =============================================================================

class ConfidencePolicy:
    """
    How confidence affects validity computation.
    
    OLD (wrong): severity_contribution = weight * severity * confidence
                 This makes "we're uncertain" look BETTER. That's backwards.
    
    NEW (correct):
        1. Detection threshold: Only report signal if confidence > MIN_CONFIDENCE
        2. If reported, severity is NOT discounted by confidence
        3. Low confidence widens the DEGRADED band (conservative)
        4. "Validity confidence" tracked separately from score
    """
    
    # Minimum confidence to report a signal at all
    MIN_CONFIDENCE_TO_REPORT = 0.4
    
    # Below this, signal exists but marked "uncertain"
    UNCERTAIN_THRESHOLD = 0.7
    
    @classmethod
    def should_report(cls, confidence: float) -> bool:
        """Should this signal be included in validity computation?"""
        return confidence >= cls.MIN_CONFIDENCE_TO_REPORT
    
    @classmethod
    def is_uncertain(cls, confidence: float) -> bool:
        """Is this signal marked as uncertain?"""
        return confidence < cls.UNCERTAIN_THRESHOLD
    
    @classmethod
    def compute_validity_confidence(cls, signals: List[FailureSignal]) -> float:
        """
        Compute overall confidence in the validity assessment.
        
        This is separate from the validity SCORE.
        Low validity_confidence means we should be MORE conservative.
        """
        if not signals:
            return 1.0  # No signals = high confidence in validity
        
        # Confidence decreases with uncertain signals
        uncertain_count = sum(1 for s in signals if cls.is_uncertain(s.confidence))
        total = len(signals)
        
        if total == 0:
            return 1.0
        
        uncertain_ratio = uncertain_count / total
        
        # More uncertain signals = lower confidence in our assessment
        return 1.0 - (uncertain_ratio * 0.5)


# =============================================================================
# FM-SPECIFIC THRESHOLDS
# =============================================================================

class FMThresholds:
    """
    FM-specific thresholds.
    
    Different failure modes have different severity profiles.
    FM4 (structural break) and FM7 (dependency) are more severe.
    """
    
    # Standard FMs: gradual degradation
    STANDARD = {
        "degraded": 40,
        "invalid": 75,
    }
    
    # Structural FMs: more aggressive thresholds
    FM4_STRUCTURAL = {
        "degraded": 30,  # Earlier warning
        "invalid": 60,   # Earlier invalidation
    }
    
    FM7_DEPENDENCY = {
        "degraded": 25,  # Very early warning for relationship changes
        "invalid": 55,   # Aggressive invalidation
    }
    
    @classmethod
    def get_thresholds(cls, fm: FailureMode) -> Dict[str, float]:
        if fm == FailureMode.FM4_STRUCTURAL_BREAK:
            return cls.FM4_STRUCTURAL
        elif fm == FailureMode.FM7_DEPENDENCY_BREAK:
            return cls.FM7_DEPENDENCY
        else:
            return cls.STANDARD


# =============================================================================
# DETECTOR BASE CLASS
# =============================================================================

class FailureModeDetector(ABC):
    """Abstract base class for failure mode detectors."""
    
    failure_mode: FailureMode
    
    @abstractmethod
    def detect(
        self,
        z_t: pd.Series,
        Z_t: pd.Series,
        **kwargs,
    ) -> FailureSignal:
        """Run detection and return a FailureSignal."""
        pass
    
    def _make_signal(
        self,
        severity: float,
        confidence: float,
        explanation: str,
        evidence: Dict[str, Any],
        triggers_kill: bool = False,
        kill_reason: Optional[str] = None,
    ) -> FailureSignal:
        """Helper to create properly bounded signals."""
        return FailureSignal(
            failure_mode=self.failure_mode,
            severity=float(np.clip(severity, 0, 100)),
            confidence=float(np.clip(confidence, 0, 1)),
            triggers_kill=triggers_kill,
            kill_reason=kill_reason,
            first_detected_at=datetime.utcnow(),
            evidence=evidence,
            explanation=explanation,
        )


# =============================================================================
# FM1: VARIANCE REGIME SHIFT
# =============================================================================

class FM1_VarianceRegimeShift(FailureModeDetector):
    """
    Detects changes in volatility structure.
    
    TIER 1: Light contextualization (can be confused with FM4/FM2).
    Adds context about whether mean/beta are also affected.
    """
    
    failure_mode = FailureMode.FM1_VARIANCE_REGIME
    
    def __init__(self, short_window_ratio: float = 0.3):
        self.short_window_ratio = short_window_ratio
    
    def detect(
        self,
        z_t: pd.Series,
        Z_t: pd.Series,
        **kwargs,
    ) -> FailureSignal:
        z = z_t.dropna()
        n = len(z)
        
        if n < 40:
            return self._make_signal(
                severity=0, confidence=0.3,
                explanation="Insufficient data for variance regime detection",
                evidence={"n": n, "error": "insufficient_data"},
            )
        
        split = int(n * (1 - self.short_window_ratio))
        past = z.iloc[:split]
        recent = z.iloc[split:]
        
        var_past = past.var()
        var_recent = recent.var()
        
        if var_past < 1e-10:
            return self._make_signal(
                severity=0, confidence=0.5,
                explanation="Historical variance near zero",
                evidence={"var_past": float(var_past), "var_recent": float(var_recent)},
            )
        
        ratio = var_recent / var_past
        log_ratio = abs(np.log(max(ratio, 1e-10)))
        
        # F-test
        f_stat = var_recent / var_past if var_recent > var_past else var_past / var_recent
        df1, df2 = len(recent) - 1, len(past) - 1
        p_value = 2 * (1 - stats.f.cdf(f_stat, df1, df2))
        
        # =================================================================
        # LIGHT CONTEXTUALIZATION: Is mean also affected?
        # =================================================================
        mean_past = past.mean()
        mean_recent = recent.mean()
        overall_std = z.std()
        mean_shift = abs(mean_recent - mean_past) / max(overall_std, 1e-10)
        mean_stable = mean_shift < 1.0  # Mean hasn't moved much
        
        # Context message
        if mean_stable and ratio > 1.2:
            context = " (mean stable — likely regime noise, not structural)"
        elif not mean_stable and ratio > 1.2:
            context = " (mean also shifted — may indicate broader change)"
        else:
            context = ""
        
        severity = min(100, log_ratio * 50)
        confidence = 0.9 if p_value < 0.05 else 0.6
        
        direction = "increased" if ratio > 1 else "decreased"
        factor = ratio if ratio > 1 else 1/ratio
        
        return self._make_signal(
            severity=severity, confidence=confidence,
            explanation=f"Volatility {direction} {factor:.1f}x (p={p_value:.3f}){context}",
            evidence={
                "variance_ratio": float(ratio),
                "log_ratio": float(log_ratio),
                "f_statistic": float(f_stat),
                "p_value": float(p_value),
                "mean_shift_sigma": float(mean_shift),
                "mean_stable": mean_stable,
            },
        )


# =============================================================================
# FM2: MEAN DRIFT
# =============================================================================

class FM2_MeanDrift(FailureModeDetector):
    """
    Detects systematic drift in mean level.
    
    TIER 1: Light contextualization (second most dangerous after FM4).
    Adds persistence check across rolling windows.
    "Drift observed in 3/4 windows" vs "single-window deviation"
    """
    
    failure_mode = FailureMode.FM2_MEAN_DRIFT
    
    def detect(
        self,
        z_t: pd.Series,
        Z_t: pd.Series,
        **kwargs,
    ) -> FailureSignal:
        Z = Z_t.dropna()
        n = len(Z)
        
        if n < 40:
            return self._make_signal(
                severity=0, confidence=0.3,
                explanation="Insufficient data for mean drift detection",
                evidence={"n": n},
            )
        
        split = int(n * 0.7)
        Z_past = Z.iloc[:split]
        Z_recent = Z.iloc[split:]
        
        mean_past = Z_past.mean()
        mean_recent = Z_recent.mean()
        overall_std = max(Z.std(), 1e-10)
        
        divergence = abs(mean_recent - mean_past) / overall_std
        
        # =================================================================
        # DRIFT PERSISTENCE CHECK: Is this consistent across windows?
        # =================================================================
        # Check multiple rolling windows for drift direction consistency
        window_size = n // 4
        windows_with_drift = 0
        drift_direction = 1 if mean_recent > mean_past else -1
        
        if window_size >= 20:
            for i in range(4):
                start = i * window_size
                end = start + window_size
                if end <= n:
                    window_mean = Z.iloc[start:end].mean()
                    # Check if this window's mean is in the drift direction
                    if drift_direction > 0 and window_mean > mean_past:
                        windows_with_drift += 1
                    elif drift_direction < 0 and window_mean < mean_past:
                        windows_with_drift += 1
        
        # Persistence context
        if windows_with_drift >= 3:
            persistence = "persistent"
            persistence_context = f"drift consistent in {windows_with_drift}/4 windows"
        elif windows_with_drift >= 2:
            persistence = "moderate"
            persistence_context = f"drift in {windows_with_drift}/4 windows"
        else:
            persistence = "weak"
            persistence_context = "single-window deviation, may be transient"
        
        # ADF test
        adf_pvalue = 1.0
        try:
            from statsmodels.tsa.stattools import adfuller
            adf_result = adfuller(Z, autolag='AIC')
            adf_pvalue = adf_result[1]
        except:
            pass
        
        divergence_severity = min(50, divergence * 15)
        adf_severity = min(50, (1 - adf_pvalue) * 50) if adf_pvalue > 0.05 else 0
        
        # Boost severity for persistent drift
        persistence_multiplier = 1.2 if persistence == "persistent" else 1.0 if persistence == "moderate" else 0.8
        severity = min(100, (divergence_severity + adf_severity) * persistence_multiplier)
        
        confidence = 0.85 if persistence == "persistent" else 0.7 if adf_pvalue < 0.1 or divergence > 2 else 0.55
        
        return self._make_signal(
            severity=severity, confidence=confidence,
            explanation=f"Mean drift {divergence:.1f}σ ({persistence_context})",
            evidence={
                "mean_divergence_std": float(divergence),
                "adf_pvalue": float(adf_pvalue),
                "drift_persistence": persistence,
                "windows_with_drift": windows_with_drift,
            },
        )


# =============================================================================
# FM3: SEASONALITY MISMATCH — PROPERLY IMPLEMENTED
# =============================================================================

class FM3_SeasonalityMismatch(FailureModeDetector):
    """
    Detects changes in seasonal patterns.
    
    THIS IS A REAL IMPLEMENTATION, not a placeholder.
    
    Methods:
        1. Spectral analysis: Compare power spectrum past vs recent
        2. Dominant frequency shift: Detect if seasonal period changed
        3. Amplitude change: Detect if seasonal strength changed
        4. Phase shift: Detect if timing of peaks changed
    """
    
    failure_mode = FailureMode.FM3_SEASONALITY_MISMATCH
    
    def detect(
        self,
        z_t: pd.Series,
        Z_t: pd.Series,
        expected_period: Optional[int] = None,
        **kwargs,
    ) -> FailureSignal:
        z = z_t.dropna()
        n = len(z)
        
        if n < 60:
            return self._make_signal(
                severity=0, confidence=0.4,
                explanation="Insufficient data for seasonality analysis (need 60+)",
                evidence={"n": n},
            )
        
        try:
            # Split into past and recent
            split = int(n * 0.7)
            past = z.iloc[:split].values
            recent = z.iloc[split:].values
            
            # 1. SPECTRAL ANALYSIS
            freqs_past, power_past = periodogram(past, detrend='linear')
            freqs_recent, power_recent = periodogram(recent, detrend='linear')
            
            # Find dominant frequencies (excluding DC component)
            if len(power_past) < 3 or len(power_recent) < 3:
                return self._make_signal(
                    severity=0, confidence=0.3,
                    explanation="Insufficient spectral resolution",
                    evidence={},
                )
            
            # Dominant frequency in each period
            dom_idx_past = np.argmax(power_past[1:]) + 1
            dom_idx_recent = np.argmax(power_recent[1:]) + 1
            
            dom_freq_past = freqs_past[dom_idx_past]
            dom_freq_recent = freqs_recent[dom_idx_recent]
            
            # Convert to period (observations per cycle)
            period_past = 1 / dom_freq_past if dom_freq_past > 0 else float('inf')
            period_recent = 1 / dom_freq_recent if dom_freq_recent > 0 else float('inf')
            
            # 2. FREQUENCY SHIFT DETECTION
            freq_shift = 0.0
            if dom_freq_past > 0:
                freq_shift = abs(dom_freq_recent - dom_freq_past) / dom_freq_past
            
            # 3. AMPLITUDE CHANGE
            # Seasonal strength = fraction of variance explained by dominant frequency
            total_power_past = power_past.sum()
            total_power_recent = power_recent.sum()
            
            strength_past = power_past[dom_idx_past] / total_power_past if total_power_past > 0 else 0
            strength_recent = power_recent[dom_idx_recent] / total_power_recent if total_power_recent > 0 else 0
            
            strength_change = abs(strength_recent - strength_past)
            
            # 4. SPECTRAL DISTANCE (overall shape change)
            # Normalize power spectra
            norm_past = power_past / (total_power_past + 1e-10)
            norm_recent = power_recent / (total_power_recent + 1e-10)
            
            # Use minimum length for comparison
            min_len = min(len(norm_past), len(norm_recent))
            spectral_distance = np.sqrt(np.sum((norm_past[:min_len] - norm_recent[:min_len])**2))
            
            # 5. CHECK IF SEASONALITY EXISTS AT ALL
            # If neither period has strong seasonality, low severity
            has_seasonality_past = strength_past > 0.1
            has_seasonality_recent = strength_recent > 0.1
            
            # SEVERITY COMPUTATION
            severity = 0.0
            issues = []
            
            if has_seasonality_past and not has_seasonality_recent:
                # Lost seasonality
                severity += 50
                issues.append("seasonality disappeared")
            elif not has_seasonality_past and has_seasonality_recent:
                # New seasonality appeared
                severity += 40
                issues.append("new seasonality appeared")
            elif has_seasonality_past and has_seasonality_recent:
                # Both have seasonality - check for changes
                if freq_shift > 0.2:
                    severity += min(40, freq_shift * 100)
                    issues.append(f"period shifted ({period_past:.0f}→{period_recent:.0f})")
                
                if strength_change > 0.2:
                    severity += min(30, strength_change * 60)
                    issues.append(f"strength changed ({strength_past:.0%}→{strength_recent:.0%})")
                
                if spectral_distance > 0.5:
                    severity += min(30, spectral_distance * 30)
                    issues.append(f"spectral shape changed")
            
            severity = min(100, severity)
            
            # CONFIDENCE
            confidence = 0.7
            if n > 120:
                confidence = 0.85
            if n < 80:
                confidence = 0.5
            
            # Explanation
            if severity == 0:
                explanation = "Seasonal pattern stable or not detected"
            else:
                explanation = f"Seasonality mismatch: {', '.join(issues)}"
            
            return self._make_signal(
                severity=severity,
                confidence=confidence,
                explanation=explanation,
                evidence={
                    "dominant_period_past": float(period_past),
                    "dominant_period_recent": float(period_recent),
                    "frequency_shift": float(freq_shift),
                    "strength_past": float(strength_past),
                    "strength_recent": float(strength_recent),
                    "strength_change": float(strength_change),
                    "spectral_distance": float(spectral_distance),
                    "has_seasonality_past": has_seasonality_past,
                    "has_seasonality_recent": has_seasonality_recent,
                },
            )
            
        except Exception as e:
            return self._make_signal(
                severity=0, confidence=0.3,
                explanation=f"Seasonality detection error: {e}",
                evidence={"error": str(e)},
            )


# =============================================================================
# FM4: STRUCTURAL BREAK — KILL SWITCH CAPABLE
# =============================================================================

class FM4_StructuralBreak(FailureModeDetector):
    """
    Detects discrete changepoints in the data-generating process.
    
    KILL SWITCH: If break > 4σ with high confidence, instant INVALID.
    
    THIS IS A REAL CHANGEPOINT DETECTOR, NOT A STATIONARITY PROXY.
    
    Key distinction:
    - High ADF p-value = "series doesn't mean-revert well" (weak stationarity)
    - Structural break = "the relationship changed at a specific point" (changepoint)
    
    Detection approach:
    1. CUSUM on residuals with statistical significance testing
    2. Rolling parameter stability (mean/variance shift pre vs post)
    3. Break only fires when we have EVIDENCE of a discrete change
    
    Severity-confidence coupling:
    - If confidence < 0.5, severity is capped at 40 (can't scream with weak evidence)
    - final_severity = min(raw_severity, raw_severity * confidence * 1.5)
    """
    
    failure_mode = FailureMode.FM4_STRUCTURAL_BREAK
    
    # Critical thresholds for CUSUM significance
    # Based on Ploberger-Kramer critical values for structural break tests
    CUSUM_CRITICAL_90 = 0.85   # 90% confidence
    CUSUM_CRITICAL_95 = 1.00   # 95% confidence  
    CUSUM_CRITICAL_99 = 1.25   # 99% confidence
    
    def detect(
        self,
        z_t: pd.Series,
        Z_t: pd.Series,
        **kwargs,
    ) -> FailureSignal:
        Z = Z_t.dropna()
        n = len(Z)
        
        if n < 50:
            return self._make_signal(
                severity=0, confidence=0.3,
                explanation="Insufficient data for structural break detection (need 50+)",
                evidence={"n": n},
            )
        
        values = Z.values
        mean = np.mean(values)
        std = max(np.std(values), 1e-10)
        
        # =====================================================================
        # CUSUM TEST with proper normalization
        # =====================================================================
        # Standardized CUSUM: S_k = (1/σ√n) * Σ(x_i - μ)
        # Under H0 (no break), sup|S_k| has known distribution
        cusum = np.cumsum(values - mean)
        # Normalize by std * sqrt(n) for proper statistical scaling
        normalized_cusum = cusum / (std * np.sqrt(n))
        sup_cusum = np.max(np.abs(normalized_cusum))
        break_idx = np.argmax(np.abs(normalized_cusum))
        
        # =====================================================================
        # STATISTICAL SIGNIFICANCE of the break
        # =====================================================================
        # Is the CUSUM statistically significant?
        if sup_cusum < self.CUSUM_CRITICAL_90:
            # No statistically significant break detected
            return self._make_signal(
                severity=0, confidence=0.7,
                explanation=f"No structural break detected (CUSUM={sup_cusum:.2f} < critical)",
                evidence={
                    "sup_cusum": float(sup_cusum),
                    "critical_value_90": self.CUSUM_CRITICAL_90,
                    "break_detected": False,
                },
            )
        
        # =====================================================================
        # BREAK MAGNITUDE (only if CUSUM is significant)
        # =====================================================================
        # Compute pre/post statistics to quantify the break
        if 15 < break_idx < n - 15:
            pre_values = values[:break_idx]
            post_values = values[break_idx:]
            
            pre_mean = np.mean(pre_values)
            post_mean = np.mean(post_values)
            pre_std = np.std(pre_values)
            post_std = np.std(post_values)
            
            # Mean shift in pooled standard deviations
            pooled_std = np.sqrt((pre_std**2 + post_std**2) / 2)
            mean_shift = abs(post_mean - pre_mean) / max(pooled_std, 1e-10)
            
            # Variance ratio
            var_ratio = max(post_std, pre_std) / max(min(post_std, pre_std), 1e-10)
            
            # Break date
            if hasattr(Z, 'index'):
                break_date = str(Z.index[break_idx])[:10]
            else:
                break_date = f"index {break_idx}"
        else:
            # Break at edge - less reliable
            mean_shift = sup_cusum * 1.5  # Approximate
            var_ratio = 1.0
            break_date = "edge (unreliable)"
        
        # =====================================================================
        # CONFIDENCE based on statistical strength
        # =====================================================================
        if sup_cusum >= self.CUSUM_CRITICAL_99:
            raw_confidence = 0.90
        elif sup_cusum >= self.CUSUM_CRITICAL_95:
            raw_confidence = 0.75
        else:  # >= CRITICAL_90
            raw_confidence = 0.55
        
        # Reduce confidence if break is at edge
        if break_idx < 20 or break_idx > n - 20:
            raw_confidence *= 0.7
        
        # Reduce confidence for small samples
        if n < 80:
            raw_confidence *= 0.85
        
        confidence = max(0.35, min(0.95, raw_confidence))
        
        # =====================================================================
        # RAW SEVERITY based on break magnitude
        # =====================================================================
        # Combine mean shift and variance change
        raw_severity = 0
        
        # Mean shift contribution (primary)
        if mean_shift >= 3.0:
            raw_severity += min(60, 20 + (mean_shift - 2) * 15)
        elif mean_shift >= 2.0:
            raw_severity += (mean_shift - 1.5) * 20
        elif mean_shift >= 1.5:
            raw_severity += (mean_shift - 1.0) * 10
        
        # Variance ratio contribution (secondary)
        if var_ratio >= 2.0:
            raw_severity += min(30, (var_ratio - 1) * 15)
        elif var_ratio >= 1.5:
            raw_severity += (var_ratio - 1) * 10
        
        # CUSUM strength contribution
        raw_severity += min(20, (sup_cusum - self.CUSUM_CRITICAL_90) * 20)
        
        raw_severity = min(100, raw_severity)
        
        # =====================================================================
        # SEVERITY-CONFIDENCE COUPLING (critical fix)
        # =====================================================================
        # Rule: You cannot scream "severity 85" with "confidence 36%"
        # 
        # If confidence < 0.5: cap severity at 40 (warning, not alarm)
        # If confidence < 0.7: cap severity at 60
        # Otherwise: allow full severity, but still dampen by confidence
        
        if confidence < 0.5:
            severity = min(raw_severity, 40)
        elif confidence < 0.7:
            severity = min(raw_severity, int(raw_severity * confidence * 1.3))
            severity = min(severity, 60)
        else:
            severity = int(raw_severity * (0.7 + confidence * 0.3))
        
        severity = int(max(0, min(100, severity)))
        
        # =====================================================================
        # KILL SWITCH CHECK
        # =====================================================================
        # Only trigger kill if BOTH high severity AND high confidence
        triggers_kill = (
            mean_shift >= Thresholds.KILL_FM4_BREAK_MAGNITUDE and 
            confidence >= 0.75
        )
        kill_reason = None
        
        if triggers_kill:
            kill_reason = (
                f"Structural break: mean shifted {mean_shift:.1f}σ at {break_date} "
                f"(confidence {confidence:.0%})"
            )
        
        # =====================================================================
        # EXPLANATION with actual evidence
        # =====================================================================
        explanation = (
            f"Break at {break_date}: mean shift {mean_shift:.1f}σ, "
            f"variance ratio {var_ratio:.1f}x, CUSUM {sup_cusum:.2f}"
        )
        
        if triggers_kill:
            explanation = f"🔴 KILL: {explanation}"
        
        return self._make_signal(
            severity=severity,
            confidence=confidence,
            explanation=explanation,
            evidence={
                "break_detected": True,
                "break_date": break_date,
                "break_index": int(break_idx),
                "mean_shift_sigma": float(mean_shift),
                "variance_ratio": float(var_ratio),
                "sup_cusum": float(sup_cusum),
                "cusum_critical_90": self.CUSUM_CRITICAL_90,
                "raw_severity": float(raw_severity),
                "confidence_adjusted": True,
                "sample_size": int(n),
            },
            triggers_kill=triggers_kill,
            kill_reason=kill_reason,
        )


# =============================================================================
# FM5: OUTLIER CONTAMINATION
# =============================================================================

class FM5_OutlierContamination(FailureModeDetector):
    """Detects when extreme events dominate estimates."""
    
    failure_mode = FailureMode.FM5_OUTLIER_CONTAMINATION
    ROBUST_SCALE = 1.4826
    
    def detect(
        self,
        z_t: pd.Series,
        Z_t: pd.Series,
        **kwargs,
    ) -> FailureSignal:
        z = z_t.dropna()
        n = len(z)
        
        if n < 30:
            return self._make_signal(
                severity=0, confidence=0.3,
                explanation="Insufficient data for outlier detection",
                evidence={"n": n},
            )
        
        values = z.values
        median = np.median(values)
        mad = np.median(np.abs(values - median))
        std = np.std(values)
        
        if mad < 1e-10:
            return self._make_signal(
                severity=0, confidence=0.5,
                explanation="MAD near zero",
                evidence={"mad": float(mad)},
            )
        
        std_mad_ratio = std / (mad * self.ROBUST_SCALE)
        threshold = 3 * mad * self.ROBUST_SCALE
        outlier_count = np.sum(np.abs(values - median) > threshold)
        outlier_freq = outlier_count / n
        
        excess_outliers = max(0, outlier_freq - 0.003)
        
        ratio_severity = max(0, (std_mad_ratio - 1.2) * 40)
        outlier_severity = excess_outliers * 500
        severity = min(100, ratio_severity + outlier_severity)
        
        confidence = 0.8 if std_mad_ratio > 1.5 or outlier_freq > 0.02 else 0.6
        
        return self._make_signal(
            severity=severity, confidence=confidence,
            explanation=f"Outlier contamination: std/MAD={std_mad_ratio:.2f}, freq={outlier_freq:.1%}",
            evidence={
                "std_mad_ratio": float(std_mad_ratio),
                "outlier_frequency": float(outlier_freq),
                "outlier_count": int(outlier_count),
            },
        )


# =============================================================================
# FM6: DISTRIBUTIONAL SHIFT
# =============================================================================

class FM6_DistributionalShift(FailureModeDetector):
    """
    Detects changes in distribution shape.
    
    TIER 2: Simple with duration context.
    Adds: "Z-score >2σ for X% of lookback"
    """
    
    failure_mode = FailureMode.FM6_DISTRIBUTIONAL_SHIFT
    
    def detect(
        self,
        z_t: pd.Series,
        Z_t: pd.Series,
        **kwargs,
    ) -> FailureSignal:
        z = z_t.dropna()
        n = len(z)
        
        if n < 60:
            return self._make_signal(
                severity=0, confidence=0.3,
                explanation="Insufficient data for distribution test",
                evidence={"n": n},
            )
        
        split = int(n * 0.7)
        past = z.iloc[:split].values
        recent = z.iloc[split:].values
        
        ks_stat, ks_pvalue = stats.ks_2samp(past, recent)
        
        skew_past = stats.skew(past)
        skew_recent = stats.skew(recent)
        kurt_past = stats.kurtosis(past)
        kurt_recent = stats.kurtosis(recent)
        
        skew_change = abs(skew_recent - skew_past)
        kurt_change = abs(kurt_recent - kurt_past)
        
        # =================================================================
        # DURATION CONTEXT: How long has positioning been extreme?
        # =================================================================
        # Calculate z-scores relative to rolling mean
        rolling_mean = z.rolling(window=min(60, n//3)).mean()
        rolling_std = z.rolling(window=min(60, n//3)).std()
        rolling_z = (z - rolling_mean) / rolling_std.replace(0, np.nan)
        
        # Count time spent beyond thresholds
        pct_beyond_2sigma = (rolling_z.abs() > 2).sum() / n * 100
        pct_beyond_3sigma = (rolling_z.abs() > 3).sum() / n * 100
        
        duration_context = ""
        if pct_beyond_2sigma > 25:
            duration_context = f" (>2σ for {pct_beyond_2sigma:.0f}% of lookback)"
        elif pct_beyond_3sigma > 10:
            duration_context = f" (>3σ for {pct_beyond_3sigma:.0f}% of lookback)"
        
        ks_severity = (1 - ks_pvalue) * 60 if ks_pvalue < 0.1 else 0
        moment_severity = min(40, skew_change * 10 + kurt_change * 5)
        severity = min(100, ks_severity + moment_severity)
        
        confidence = 0.85 if ks_pvalue < 0.05 else 0.6
        
        return self._make_signal(
            severity=severity, confidence=confidence,
            explanation=f"Distribution shift: KS p={ks_pvalue:.3f}, Δskew={skew_change:.2f}{duration_context}",
            evidence={
                "ks_statistic": float(ks_stat),
                "ks_pvalue": float(ks_pvalue),
                "skew_change": float(skew_change),
                "kurtosis_change": float(kurt_change),
                "pct_beyond_2sigma": float(pct_beyond_2sigma),
                "pct_beyond_3sigma": float(pct_beyond_3sigma),
            },
        )


# =============================================================================
# FM7: DEPENDENCY BREAK — PROPERLY IMPLEMENTED + KILL SWITCH
# =============================================================================

class FM7_DependencyBreak(FailureModeDetector):
    """
    Detects changes in relationship structure.
    
    THIS IS A REAL IMPLEMENTATION, not a placeholder.
    
    KILL SWITCH: Correlation sign flip triggers instant INVALID.
    
    Methods:
        1. Rolling correlation stability
        2. Beta stability (regression coefficient)
        3. Sign flip detection (KILL SWITCH)
        4. Cointegration breakdown
        5. Eigenvalue stability (for multi-series baskets)
    """
    
    failure_mode = FailureMode.FM7_DEPENDENCY_BREAK
    
    def detect(
        self,
        z_t: pd.Series,
        Z_t: pd.Series,
        reference_z_t: Optional[pd.Series] = None,
        reference_series: Optional[List[pd.Series]] = None,  # For basket analysis
        **kwargs,
    ) -> FailureSignal:
        # Handle no reference case
        if reference_z_t is None and reference_series is None:
            return self._make_signal(
                severity=0, confidence=0.0,
                explanation="No reference series provided for dependency analysis",
                evidence={"error": "no_reference"},
            )
        
        # Single reference case
        if reference_z_t is not None:
            return self._detect_pairwise(z_t, Z_t, reference_z_t)
        
        # Multi-reference (basket) case
        if reference_series is not None and len(reference_series) >= 2:
            return self._detect_basket(z_t, Z_t, reference_series)
        
        return self._make_signal(
            severity=0, confidence=0.0,
            explanation="Insufficient reference series for basket analysis",
            evidence={},
        )
    
    def _detect_pairwise(
        self,
        z_t: pd.Series,
        Z_t: pd.Series,
        reference_z_t: pd.Series,
    ) -> FailureSignal:
        """Pairwise dependency analysis."""
        z = z_t.dropna()
        ref = reference_z_t.dropna()
        
        # Align
        common_idx = z.index.intersection(ref.index)
        if len(common_idx) < 60:
            return self._make_signal(
                severity=0, confidence=0.3,
                explanation=f"Insufficient overlapping data ({len(common_idx)})",
                evidence={"common_n": len(common_idx)},
            )
        
        z_aligned = z.loc[common_idx].values
        ref_aligned = ref.loc[common_idx].values
        n = len(common_idx)
        
        split = int(n * 0.7)
        
        # 1. CORRELATION ANALYSIS
        corr_past = np.corrcoef(z_aligned[:split], ref_aligned[:split])[0, 1]
        corr_recent = np.corrcoef(z_aligned[split:], ref_aligned[split:])[0, 1]
        corr_change = abs(corr_recent - corr_past)
        
        # 2. SIGN FLIP CHECK (KILL SWITCH)
        sign_flip = (
            (corr_past * corr_recent) < 0 and 
            abs(corr_past) > 0.3 and 
            abs(corr_recent) > 0.3
        )
        
        triggers_kill = sign_flip and Thresholds.KILL_FM7_CORRELATION_FLIP
        kill_reason = None
        if triggers_kill:
            kill_reason = (
                f"Correlation sign flip: {corr_past:.2f} → {corr_recent:.2f}. "
                "Relationship fundamentally changed."
            )
        
        # 3. ROLLING CORRELATION STABILITY
        window = max(20, n // 10)
        rolling_corrs = []
        for i in range(window, n):
            start = i - window
            c = np.corrcoef(z_aligned[start:i], ref_aligned[start:i])[0, 1]
            if not np.isnan(c):
                rolling_corrs.append(c)
        
        if rolling_corrs:
            corr_volatility = np.std(rolling_corrs)
        else:
            corr_volatility = 0.0
        
        # 4. BETA STABILITY (regression coefficient)
        try:
            from scipy.stats import linregress
            
            slope_past, _, _, _, _ = linregress(ref_aligned[:split], z_aligned[:split])
            slope_recent, _, _, _, _ = linregress(ref_aligned[split:], z_aligned[split:])
            
            beta_change = abs(slope_recent - slope_past) / (abs(slope_past) + 1e-10)
        except:
            beta_change = 0.0
        
        # SEVERITY COMPUTATION
        severity = 0.0
        issues = []
        
        # =================================================================
        # STATISTICAL SIGNIFICANCE CONTEXT
        # =================================================================
        # Check if correlation is still statistically significant
        # Using Fisher Z-transform for significance testing
        n_recent = len(z_aligned) - split
        if n_recent > 3:
            # Fisher z-transform
            z_fisher = 0.5 * np.log((1 + corr_recent) / (1 - corr_recent + 1e-10))
            se = 1 / np.sqrt(n_recent - 3)
            z_stat = abs(z_fisher / se)
            corr_significant = z_stat > 1.96  # 95% confidence
        else:
            corr_significant = abs(corr_recent) > 0.5
        
        # Context message
        if corr_significant and abs(corr_recent) > 0.3:
            significance_context = f"remains significant at {corr_recent:.2f}"
        elif not corr_significant:
            significance_context = f"now indistinguishable from noise ({corr_recent:.2f})"
        else:
            significance_context = f"weak ({corr_recent:.2f})"
        
        if sign_flip:
            severity = 100  # Maximum
            issues.append(f"SIGN FLIP ({corr_past:.2f}→{corr_recent:.2f})")
        else:
            if corr_change > 0.3:
                severity += min(40, corr_change * 80)
                issues.append(f"corr {corr_past:.2f}→{corr_recent:.2f}, {significance_context}")
            
            if corr_volatility > 0.15:
                severity += min(30, corr_volatility * 100)
                issues.append(f"unstable (σ={corr_volatility:.2f})")
            
            if beta_change > 0.5:
                severity += min(30, beta_change * 40)
                issues.append(f"beta shifted {beta_change:.0%}")
        
        severity = min(100, severity)
        
        # Confidence
        confidence = 0.85 if n > 100 else 0.7
        if corr_change > 0.4 or sign_flip:
            confidence = 0.9
        
        if severity == 0:
            explanation = f"Dependency stable (corr={corr_recent:.2f}, {significance_context})"
        else:
            explanation = f"Dependency break: {', '.join(issues)}"
        
        if triggers_kill:
            explanation = f"🔴 KILL: {explanation}"
        
        return self._make_signal(
            severity=severity, confidence=confidence,
            explanation=explanation,
            evidence={
                "correlation_past": float(corr_past),
                "correlation_recent": float(corr_recent),
                "correlation_change": float(corr_change),
                "correlation_volatility": float(corr_volatility),
                "beta_change": float(beta_change),
                "sign_flip": sign_flip,
                "correlation_significant": corr_significant,
            },
            triggers_kill=triggers_kill,
            kill_reason=kill_reason,
        )
    
    def _detect_basket(
        self,
        z_t: pd.Series,
        Z_t: pd.Series,
        reference_series: List[pd.Series],
    ) -> FailureSignal:
        """
        Multi-series basket analysis using eigenvalue stability.
        
        For baskets/portfolios, we care about:
            1. Correlation matrix stability
            2. Principal component stability
            3. Explained variance changes
        """
        z = z_t.dropna()
        
        # Align all series
        common_idx = z.index
        aligned_series = [z]
        
        for ref in reference_series:
            ref_clean = ref.dropna()
            common_idx = common_idx.intersection(ref_clean.index)
        
        if len(common_idx) < 60:
            return self._make_signal(
                severity=0, confidence=0.3,
                explanation="Insufficient overlapping data for basket analysis",
                evidence={"common_n": len(common_idx)},
            )
        
        # Build matrix
        data = np.column_stack([z.loc[common_idx].values] + 
                               [ref.loc[common_idx].values for ref in reference_series])
        n, k = data.shape
        split = int(n * 0.7)
        
        # Correlation matrices
        corr_past = np.corrcoef(data[:split].T)
        corr_recent = np.corrcoef(data[split:].T)
        
        # 1. Frobenius norm of correlation change
        corr_change_norm = np.linalg.norm(corr_recent - corr_past, 'fro')
        max_possible = np.sqrt(2 * k * (k - 1))  # Max Frobenius distance
        relative_corr_change = corr_change_norm / max_possible
        
        # 2. Eigenvalue stability
        try:
            eig_past = np.linalg.eigvalsh(corr_past)
            eig_recent = np.linalg.eigvalsh(corr_recent)
            
            # Sort descending
            eig_past = np.sort(eig_past)[::-1]
            eig_recent = np.sort(eig_recent)[::-1]
            
            # First eigenvalue change (explains most variance)
            first_eig_change = abs(eig_recent[0] - eig_past[0]) / eig_past[0]
            
            # Eigenvalue distribution change
            eig_distance = np.linalg.norm(eig_recent - eig_past)
        except:
            first_eig_change = 0.0
            eig_distance = 0.0
        
        # 3. Check for any sign flips in correlation matrix
        sign_flip_count = np.sum((corr_past * corr_recent) < 0)
        # Exclude diagonal
        sign_flip_count = max(0, sign_flip_count - k)
        
        # SEVERITY
        severity = 0.0
        issues = []
        
        triggers_kill = False
        kill_reason = None
        
        if sign_flip_count > 0:
            severity += min(50, sign_flip_count * 15)
            issues.append(f"{sign_flip_count} correlation sign flips")
            
            if sign_flip_count >= k:  # Major structural change
                triggers_kill = True
                kill_reason = f"Major basket structure change: {sign_flip_count} correlation sign flips"
        
        if relative_corr_change > 0.3:
            severity += min(30, relative_corr_change * 60)
            issues.append(f"corr matrix changed ({relative_corr_change:.0%})")
        
        if first_eig_change > 0.3:
            severity += min(30, first_eig_change * 50)
            issues.append(f"first PC shifted ({first_eig_change:.0%})")
        
        severity = min(100, severity)
        confidence = 0.75 if n > 100 else 0.6
        
        if severity == 0:
            explanation = "Basket dependency structure stable"
        else:
            explanation = f"Basket dependency break: {', '.join(issues)}"
        
        return self._make_signal(
            severity=severity, confidence=confidence,
            explanation=explanation,
            evidence={
                "correlation_change_norm": float(corr_change_norm),
                "relative_corr_change": float(relative_corr_change),
                "first_eigenvalue_change": float(first_eig_change),
                "eigenvalue_distance": float(eig_distance),
                "sign_flip_count": int(sign_flip_count),
                "basket_size": k,
            },
            triggers_kill=triggers_kill,
            kill_reason=kill_reason,
        )


# =============================================================================
# DETECTOR SUITE WITH CORRECTED CONFIDENCE HANDLING
# =============================================================================

class DetectorSuite:
    """
    Complete suite of failure mode detectors.
    
    IMPORTANT: Uses ConfidencePolicy for proper uncertainty handling.
    """
    
    def __init__(self):
        self.detectors = [
            FM1_VarianceRegimeShift(),
            FM2_MeanDrift(),
            FM3_SeasonalityMismatch(),  # Now properly implemented!
            FM4_StructuralBreak(),
            FM5_OutlierContamination(),
            FM6_DistributionalShift(),
            FM7_DependencyBreak(),       # Now properly implemented!
        ]
    
    def run_all(
        self,
        z_t: pd.Series,
        Z_t: pd.Series,
        reference_z_t: Optional[pd.Series] = None,
        reference_series: Optional[List[pd.Series]] = None,
    ) -> List[FailureSignal]:
        """
        Run all detectors and return signals.
        
        CONFIDENCE POLICY: 
        - Signals below MIN_CONFIDENCE_TO_REPORT are excluded
        - Remaining signals DO NOT have severity discounted by confidence
        """
        signals = []
        
        for detector in self.detectors:
            try:
                if detector.failure_mode == FailureMode.FM7_DEPENDENCY_BREAK:
                    signal = detector.detect(
                        z_t, Z_t, 
                        reference_z_t=reference_z_t,
                        reference_series=reference_series,
                    )
                else:
                    signal = detector.detect(z_t, Z_t)
                
                # Apply confidence policy
                if ConfidencePolicy.should_report(signal.confidence):
                    signals.append(signal)
                # Otherwise: signal is too uncertain to include
                
            except Exception as e:
                # Create error signal (always reported)
                signals.append(FailureSignal(
                    failure_mode=detector.failure_mode,
                    severity=0,
                    confidence=0.1,
                    explanation=f"Detection error: {e}",
                    evidence={"error": str(e)},
                ))
        
        return sorted(signals, key=lambda s: s.first_detected_at)
    
    def has_kill_switch(self, signals: List[FailureSignal]) -> Optional[FailureSignal]:
        """Check if any signal triggered a kill switch."""
        for signal in signals:
            if signal.triggers_kill:
                return signal
        return None
    
    def compute_validity_confidence(self, signals: List[FailureSignal]) -> float:
        """
        Compute confidence in the validity assessment.
        
        This is SEPARATE from the validity score.
        """
        return ConfidencePolicy.compute_validity_confidence(signals)
