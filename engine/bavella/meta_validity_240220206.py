"""
Bavella v2 — Detector Meta-Validity
====================================

THE MISSING LAYER: How confident are we that the detector is right?

Current state:
    FM1 fires → severity 62%
    
Missing question:
    "How sure are you that it ACTUALLY broke?"

This module adds:
    1. DetectionConfidence - second-order signal quality
    2. Sample sufficiency assessment
    3. Statistical power estimation
    4. Decision boundary proximity
    5. Window stability metrics

Now every signal carries:
    - severity (how bad is the failure)
    - detection_confidence (how sure are we it's real)

This is epistemic honesty at institutional scale.

Copyright 2024-2026 Bavella Technologies Sarl
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from scipy import stats


# =============================================================================
# DETECTION CONFIDENCE (second-order signal)
# =============================================================================

class ConfidenceLevel(Enum):
    """Qualitative confidence levels."""
    DECISIVE = "decisive"      # Very high confidence, clear signal
    CONFIDENT = "confident"    # High confidence
    MODERATE = "moderate"      # Reasonable confidence
    UNCERTAIN = "uncertain"    # Low confidence, noisy signal
    SPECULATIVE = "speculative"  # Very low confidence


@dataclass(frozen=True)
class DetectionConfidence:
    """
    Second-order confidence in the detection itself.
    
    This answers: "How sure are we that the detector is right?"
    
    Components:
        sample_sufficiency: Was sample size adequate for this test?
        statistical_power: Could we detect an effect of this size?
        window_stability: Was the analysis window stable?
        decision_boundary: How far from threshold? (margin)
        noise_ratio: Signal-to-noise quality
    """
    # Overall confidence (0-100)
    overall: float
    
    # Components (each 0-100)
    sample_sufficiency: float
    statistical_power: float
    window_stability: float
    decision_boundary_margin: float  # How far from threshold
    noise_ratio: float  # Signal quality
    
    # Qualitative level
    level: ConfidenceLevel
    
    # Specific concerns
    concerns: Tuple[str, ...] = ()
    
    @classmethod
    def compute(
        cls,
        sample_size: int,
        effect_size: float,
        test_statistic: float,
        threshold: float,
        variance_of_statistic: float,
        min_sample_required: int = 30,
    ) -> "DetectionConfidence":
        """
        Compute detection confidence from statistical properties.
        """
        concerns = []
        
        # 1. Sample sufficiency
        if sample_size < min_sample_required:
            sample_sufficiency = (sample_size / min_sample_required) * 50
            concerns.append(f"Sample size {sample_size} below minimum {min_sample_required}")
        elif sample_size < min_sample_required * 2:
            sample_sufficiency = 50 + (sample_size - min_sample_required) / min_sample_required * 30
        else:
            sample_sufficiency = min(100, 80 + np.log(sample_size / min_sample_required) * 10)
        
        # 2. Statistical power (simplified)
        # Power increases with effect size and sample size
        if effect_size > 0:
            noncentrality = effect_size * np.sqrt(sample_size)
            # Approximate power for t-test-like scenario
            statistical_power = min(100, 50 + noncentrality * 10)
        else:
            statistical_power = 50
        
        # 3. Window stability (based on variance of the statistic)
        if variance_of_statistic > 0:
            cv = np.sqrt(variance_of_statistic) / (abs(test_statistic) + 1e-10)
            window_stability = max(0, min(100, 100 - cv * 100))
        else:
            window_stability = 80
        
        # 4. Decision boundary margin
        margin = abs(test_statistic - threshold) / (threshold + 1e-10)
        decision_boundary_margin = min(100, margin * 100)
        
        if margin < 0.1:
            concerns.append("Near decision boundary - result may flip with small changes")
        
        # 5. Noise ratio (effect vs noise)
        if variance_of_statistic > 0:
            snr = abs(effect_size) / np.sqrt(variance_of_statistic)
            noise_ratio = min(100, snr * 25)
        else:
            noise_ratio = 70
        
        if noise_ratio < 30:
            concerns.append("High noise relative to signal")
        
        # Overall (weighted average)
        overall = (
            sample_sufficiency * 0.25 +
            statistical_power * 0.25 +
            window_stability * 0.20 +
            decision_boundary_margin * 0.15 +
            noise_ratio * 0.15
        )
        
        # Determine level
        if overall >= 85:
            level = ConfidenceLevel.DECISIVE
        elif overall >= 70:
            level = ConfidenceLevel.CONFIDENT
        elif overall >= 50:
            level = ConfidenceLevel.MODERATE
        elif overall >= 30:
            level = ConfidenceLevel.UNCERTAIN
        else:
            level = ConfidenceLevel.SPECULATIVE
        
        return cls(
            overall=overall,
            sample_sufficiency=sample_sufficiency,
            statistical_power=statistical_power,
            window_stability=window_stability,
            decision_boundary_margin=decision_boundary_margin,
            noise_ratio=noise_ratio,
            level=level,
            concerns=tuple(concerns),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall": self.overall,
            "level": self.level.value,
            "components": {
                "sample_sufficiency": self.sample_sufficiency,
                "statistical_power": self.statistical_power,
                "window_stability": self.window_stability,
                "decision_boundary_margin": self.decision_boundary_margin,
                "noise_ratio": self.noise_ratio,
            },
            "concerns": list(self.concerns),
        }


# =============================================================================
# ENHANCED FAILURE SIGNAL (with meta-validity)
# =============================================================================

@dataclass(frozen=True)
class EnhancedFailureSignal:
    """
    Failure mode signal with full meta-validity.
    
    Now includes:
        - severity: How bad is the failure (0-100)
        - detection_confidence: How sure are we it's real (DetectionConfidence)
        - effective_severity: severity weighted by confidence
    """
    # Core signal
    failure_mode: Any  # FailureMode enum
    severity: float  # 0-100
    
    # META-VALIDITY (the missing layer)
    detection_confidence: DetectionConfidence
    
    # Context
    explanation: str
    evidence: Dict[str, Any] = field(default_factory=dict)
    
    # Kill switch
    triggers_kill: bool = False
    kill_reason: Optional[str] = None
    
    # Timestamp
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def effective_severity(self) -> float:
        """
        Severity weighted by detection confidence.
        
        High severity + low confidence = lower effective severity.
        This prevents false alarms from dominating.
        """
        confidence_factor = self.detection_confidence.overall / 100
        return self.severity * confidence_factor
    
    @property
    def is_decisive(self) -> bool:
        """Is this a decisive detection (high confidence)?"""
        return self.detection_confidence.level in (
            ConfidenceLevel.DECISIVE,
            ConfidenceLevel.CONFIDENT,
        )
    
    @property
    def is_speculative(self) -> bool:
        """Is this detection speculative (low confidence)?"""
        return self.detection_confidence.level in (
            ConfidenceLevel.UNCERTAIN,
            ConfidenceLevel.SPECULATIVE,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "failure_mode": self.failure_mode.name if hasattr(self.failure_mode, 'name') else str(self.failure_mode),
            "severity": self.severity,
            "detection_confidence": self.detection_confidence.to_dict(),
            "effective_severity": self.effective_severity,
            "is_decisive": self.is_decisive,
            "explanation": self.explanation,
            "triggers_kill": self.triggers_kill,
        }


# =============================================================================
# META-VALIDITY ASSESSOR
# =============================================================================

class MetaValidityAssessor:
    """
    Assesses the validity of validity detections.
    
    This is the "confidence in confidence" layer that institutions require.
    """
    
    @staticmethod
    def assess_variance_detection(
        data: np.ndarray,
        variance_ratio: float,
        f_statistic: float,
        p_value: float,
    ) -> DetectionConfidence:
        """Assess confidence in variance regime detection."""
        n = len(data)
        effect_size = abs(np.log(max(variance_ratio, 0.01)))
        
        # Variance of F-statistic (approximation)
        var_f = 2 * f_statistic**2 / (n - 2) if n > 2 else 1.0
        
        return DetectionConfidence.compute(
            sample_size=n,
            effect_size=effect_size,
            test_statistic=f_statistic,
            threshold=1.5,  # F > 1.5 is notable
            variance_of_statistic=var_f,
            min_sample_required=40,
        )
    
    @staticmethod
    def assess_structural_break(
        data: np.ndarray,
        break_magnitude: float,
        cusum_max: float,
    ) -> DetectionConfidence:
        """Assess confidence in structural break detection."""
        n = len(data)
        
        # CUSUM variance under null
        var_cusum = n / 3  # Approximate for standardized CUSUM
        
        concerns = []
        if n < 60:
            concerns.append("Short series reduces break detection reliability")
        
        conf = DetectionConfidence.compute(
            sample_size=n,
            effect_size=break_magnitude,
            test_statistic=cusum_max,
            threshold=3.0,  # CUSUM > 3 is notable
            variance_of_statistic=var_cusum,
            min_sample_required=50,
        )
        
        # Add any extra concerns
        if concerns:
            conf = DetectionConfidence(
                overall=conf.overall,
                sample_sufficiency=conf.sample_sufficiency,
                statistical_power=conf.statistical_power,
                window_stability=conf.window_stability,
                decision_boundary_margin=conf.decision_boundary_margin,
                noise_ratio=conf.noise_ratio,
                level=conf.level,
                concerns=conf.concerns + tuple(concerns),
            )
        
        return conf
    
    @staticmethod
    def assess_correlation_break(
        n_observations: int,
        correlation_change: float,
        correlation_past: float,
        correlation_recent: float,
    ) -> DetectionConfidence:
        """Assess confidence in correlation/dependency break detection."""
        
        # Fisher z-transformation for correlation comparison
        def fisher_z(r):
            r = np.clip(r, -0.999, 0.999)
            return 0.5 * np.log((1 + r) / (1 - r))
        
        z_past = fisher_z(correlation_past)
        z_recent = fisher_z(correlation_recent)
        z_diff = abs(z_recent - z_past)
        
        # Standard error of z difference
        n_half = n_observations // 2
        se_diff = np.sqrt(2 / (n_half - 3)) if n_half > 3 else 1.0
        
        return DetectionConfidence.compute(
            sample_size=n_observations,
            effect_size=z_diff,
            test_statistic=z_diff / se_diff,
            threshold=1.96,  # ~95% significance
            variance_of_statistic=se_diff**2,
            min_sample_required=60,
        )
    
    @staticmethod
    def assess_distributional_shift(
        n_observations: int,
        ks_statistic: float,
        ks_pvalue: float,
    ) -> DetectionConfidence:
        """Assess confidence in distributional shift detection."""
        
        # KS statistic variance approximation
        n_half = n_observations // 2
        var_ks = (1 / n_half + 1 / n_half) / 4 if n_half > 0 else 1.0
        
        return DetectionConfidence.compute(
            sample_size=n_observations,
            effect_size=ks_statistic,
            test_statistic=ks_statistic,
            threshold=0.1,  # KS > 0.1 is notable
            variance_of_statistic=var_ks,
            min_sample_required=60,
        )


# =============================================================================
# VALIDITY WITH META-VALIDITY
# =============================================================================

@dataclass(frozen=True)
class MetaValidityVerdict:
    """
    Validity verdict with meta-validity information.
    
    This answers both:
        1. "Is this system valid?" (validity score)
        2. "How sure are you?" (meta-validity)
    """
    # Primary validity
    score: float
    state: str  # VALID, DEGRADED, INVALID
    
    # META-VALIDITY (the missing layer)
    assessment_confidence: float  # 0-100
    assessment_level: ConfidenceLevel
    
    # Detailed breakdown
    decisive_signals: int  # How many high-confidence signals
    speculative_signals: int  # How many low-confidence signals
    total_signals: int
    
    # Concerns about the assessment itself
    meta_concerns: Tuple[str, ...] = ()
    
    # Attribution (with confidence)
    attributions: Tuple[Tuple[Any, float, float], ...] = ()  # (FM, contribution%, confidence)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "validity": {
                "score": self.score,
                "state": self.state,
            },
            "meta_validity": {
                "assessment_confidence": self.assessment_confidence,
                "assessment_level": self.assessment_level.value,
                "decisive_signals": self.decisive_signals,
                "speculative_signals": self.speculative_signals,
                "total_signals": self.total_signals,
            },
            "concerns": list(self.meta_concerns),
            "attributions": [
                {"failure_mode": fm.name if hasattr(fm, 'name') else str(fm), 
                 "contribution_pct": contrib, 
                 "confidence": conf}
                for fm, contrib, conf in self.attributions
            ],
        }


def compute_meta_validity(
    signals: List[EnhancedFailureSignal],
    score: float,
    state: str,
) -> MetaValidityVerdict:
    """
    Compute validity verdict with meta-validity.
    """
    if not signals:
        return MetaValidityVerdict(
            score=score,
            state=state,
            assessment_confidence=90.0,
            assessment_level=ConfidenceLevel.CONFIDENT,
            decisive_signals=0,
            speculative_signals=0,
            total_signals=0,
        )
    
    # Count signal types
    decisive = sum(1 for s in signals if s.is_decisive)
    speculative = sum(1 for s in signals if s.is_speculative)
    total = len(signals)
    
    # Overall assessment confidence
    confidences = [s.detection_confidence.overall for s in signals]
    avg_confidence = np.mean(confidences)
    
    # Concerns
    concerns = []
    
    if speculative / total > 0.5:
        concerns.append(f"Majority of signals ({speculative}/{total}) are speculative")
        avg_confidence *= 0.8  # Penalize
    
    if decisive == 0 and total > 0:
        concerns.append("No decisive signals - assessment is uncertain")
        avg_confidence *= 0.7
    
    # Collect all detector concerns
    for s in signals:
        for concern in s.detection_confidence.concerns:
            if concern not in concerns:
                concerns.append(concern)
    
    # Determine level
    if avg_confidence >= 85:
        level = ConfidenceLevel.DECISIVE
    elif avg_confidence >= 70:
        level = ConfidenceLevel.CONFIDENT
    elif avg_confidence >= 50:
        level = ConfidenceLevel.MODERATE
    elif avg_confidence >= 30:
        level = ConfidenceLevel.UNCERTAIN
    else:
        level = ConfidenceLevel.SPECULATIVE
    
    # Attribution with confidence
    total_effective = sum(s.effective_severity for s in signals)
    if total_effective > 0:
        attributions = tuple(
            (s.failure_mode, 
             (s.effective_severity / total_effective) * 100,
             s.detection_confidence.overall)
            for s in signals
            if s.effective_severity > 0
        )
    else:
        attributions = ()
    
    return MetaValidityVerdict(
        score=score,
        state=state,
        assessment_confidence=avg_confidence,
        assessment_level=level,
        decisive_signals=decisive,
        speculative_signals=speculative,
        total_signals=total,
        meta_concerns=tuple(concerns),
        attributions=attributions,
    )


# =============================================================================
# TESTS
# =============================================================================

def test_detection_confidence_basic():
    """Test basic detection confidence computation."""
    # Good detection
    conf = DetectionConfidence.compute(
        sample_size=200,
        effect_size=2.0,
        test_statistic=3.5,
        threshold=1.96,
        variance_of_statistic=0.5,
    )
    
    assert conf.overall > 70, f"Should be confident, got {conf.overall}"
    assert conf.level in (ConfidenceLevel.CONFIDENT, ConfidenceLevel.DECISIVE)
    print(f"Good detection: {conf.overall:.1f}% ({conf.level.value})")
    
    # Poor detection (small sample)
    conf_poor = DetectionConfidence.compute(
        sample_size=15,
        effect_size=0.5,
        test_statistic=2.0,
        threshold=1.96,
        variance_of_statistic=2.0,
    )
    
    assert conf_poor.overall < 60, f"Should be uncertain, got {conf_poor.overall}"
    assert len(conf_poor.concerns) > 0
    print(f"Poor detection: {conf_poor.overall:.1f}% ({conf_poor.level.value})")
    print(f"Concerns: {conf_poor.concerns}")
    
    print("✓ test_detection_confidence_basic passed")


def test_effective_severity():
    """Test that low confidence reduces effective severity."""
    from .core import FailureMode
    
    high_conf = DetectionConfidence(
        overall=90.0,
        sample_sufficiency=90,
        statistical_power=90,
        window_stability=90,
        decision_boundary_margin=90,
        noise_ratio=90,
        level=ConfidenceLevel.DECISIVE,
    )
    
    low_conf = DetectionConfidence(
        overall=30.0,
        sample_sufficiency=30,
        statistical_power=30,
        window_stability=30,
        decision_boundary_margin=30,
        noise_ratio=30,
        level=ConfidenceLevel.SPECULATIVE,
    )
    
    # Same severity, different confidence
    signal_high = EnhancedFailureSignal(
        failure_mode=FailureMode.FM1_VARIANCE_REGIME,
        severity=80.0,
        detection_confidence=high_conf,
        explanation="High confidence",
    )
    
    signal_low = EnhancedFailureSignal(
        failure_mode=FailureMode.FM1_VARIANCE_REGIME,
        severity=80.0,
        detection_confidence=low_conf,
        explanation="Low confidence",
    )
    
    assert signal_high.effective_severity > signal_low.effective_severity, \
        "High confidence should have higher effective severity"
    
    print(f"High conf effective: {signal_high.effective_severity:.1f}")
    print(f"Low conf effective: {signal_low.effective_severity:.1f}")
    print("✓ test_effective_severity passed")


def test_meta_validity_verdict():
    """Test meta-validity verdict computation."""
    from .core import FailureMode
    
    # Mix of decisive and speculative signals
    signals = [
        EnhancedFailureSignal(
            failure_mode=FailureMode.FM1_VARIANCE_REGIME,
            severity=60,
            detection_confidence=DetectionConfidence(
                overall=85, sample_sufficiency=90, statistical_power=85,
                window_stability=80, decision_boundary_margin=85, noise_ratio=85,
                level=ConfidenceLevel.CONFIDENT,
            ),
            explanation="Confident",
        ),
        EnhancedFailureSignal(
            failure_mode=FailureMode.FM4_STRUCTURAL_BREAK,
            severity=70,
            detection_confidence=DetectionConfidence(
                overall=25, sample_sufficiency=30, statistical_power=20,
                window_stability=30, decision_boundary_margin=20, noise_ratio=25,
                level=ConfidenceLevel.SPECULATIVE,
                concerns=("Near decision boundary",),
            ),
            explanation="Speculative",
        ),
    ]
    
    verdict = compute_meta_validity(signals, score=65, state="DEGRADED")
    
    assert verdict.decisive_signals == 1
    assert verdict.speculative_signals == 1
    assert len(verdict.meta_concerns) > 0
    
    print(f"Meta verdict: {verdict.assessment_confidence:.1f}% ({verdict.assessment_level.value})")
    print(f"Concerns: {verdict.meta_concerns}")
    print("✓ test_meta_validity_verdict passed")


def run_all_meta_validity_tests():
    print("\n" + "=" * 60)
    print("DETECTOR META-VALIDITY TESTS")
    print("=" * 60 + "\n")
    
    test_detection_confidence_basic()
    print()
    test_effective_severity()
    print()
    test_meta_validity_verdict()
    
    print("\n" + "=" * 60)
    print("ALL META-VALIDITY TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    run_all_meta_validity_tests()
