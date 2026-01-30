"""
Bavella v2 — Corrected Validity Computer
=========================================

CRITICAL FIX: Confidence handling

OLD (wrong):
    effective_severity = weight * severity * confidence
    → Low confidence REDUCED penalty
    → "We're uncertain" made things look BETTER
    → THIS IS BACKWARDS

NEW (correct):
    1. Confidence determines IF signal is included (threshold)
    2. Once included, severity is NOT discounted
    3. validity_confidence tracked separately from validity_score
    4. Low validity_confidence → wider DEGRADED band

ALSO FIXED:
    - Kill switches override scoring
    - FM-specific thresholds
    - Non-linear amplification for correlated failures
    - Dominant failure logic

Copyright 2024-2026 Bavella Technologies Sarl
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from .core import (
    FailureMode, FailureSignal, ValidityVerdict, ValidityState,
    Thresholds, Weights
)
from .detectors_proper import (
    ConfidencePolicy, FMThresholds, DetectorSuite
)


# =============================================================================
# AGGREGATION MODES
# =============================================================================

class AggregationMode:
    """
    How to aggregate multiple failure modes into a validity score.
    """
    
    @staticmethod
    def linear(signals: List[FailureSignal]) -> float:
        """
        Simple weighted sum.
        
        Score = 100 - Σ(weight_i * severity_i)
        
        Note: Confidence is NOT used to discount severity.
        """
        total_penalty = 0.0
        
        for signal in signals:
            weight = signal.failure_mode.get_weight()
            total_penalty += weight * signal.severity
        
        return max(0, 100 - total_penalty)
    
    @staticmethod
    def with_amplification(signals: List[FailureSignal]) -> float:
        """
        Weighted sum WITH amplification for correlated failures.
        
        If multiple failures are severe, the penalty is amplified.
        This handles non-linear interactions.
        """
        total_penalty = 0.0
        severe_count = 0
        
        for signal in signals:
            weight = signal.failure_mode.get_weight()
            penalty = weight * signal.severity
            total_penalty += penalty
            
            if signal.severity > 50:
                severe_count += 1
        
        # Amplification: multiple severe failures compound
        if severe_count >= 2:
            amplification = 1 + (severe_count - 1) * 0.2  # 20% per additional severe
            total_penalty *= amplification
        
        return max(0, 100 - total_penalty)
    
    @staticmethod
    def with_dominance(signals: List[FailureSignal]) -> Tuple[float, Optional[FailureMode]]:
        """
        Score with dominant failure tracking.
        
        Returns (score, dominant_failure).
        The dominant failure is the one contributing most to the penalty.
        """
        penalties: List[Tuple[FailureMode, float]] = []
        
        for signal in signals:
            weight = signal.failure_mode.get_weight()
            penalty = weight * signal.severity
            penalties.append((signal.failure_mode, penalty))
        
        if not penalties:
            return 100.0, None
        
        total_penalty = sum(p for _, p in penalties)
        dominant = max(penalties, key=lambda x: x[1])
        
        return max(0, 100 - total_penalty), dominant[0]


# =============================================================================
# VALIDITY COMPUTER (corrected)
# =============================================================================

class ValidityComputer:
    """
    Computes validity scores from failure mode signals.
    
    CORRECTED BEHAVIOR:
        1. Confidence does NOT discount severity
        2. Kill switches override scoring entirely
        3. FM-specific thresholds apply
        4. Non-linear amplification for multiple severe failures
        5. validity_confidence tracked separately
    """
    
    def __init__(
        self,
        use_amplification: bool = True,
        use_fm_specific_thresholds: bool = True,
    ):
        self.use_amplification = use_amplification
        self.use_fm_specific_thresholds = use_fm_specific_thresholds
    
    def compute(
        self,
        signals: List[FailureSignal],
        upstream_verdicts: Optional[List[ValidityVerdict]] = None,
    ) -> ValidityVerdict:
        """
        Compute validity from signals.
        
        Args:
            signals: Failure mode detection signals
            upstream_verdicts: Parent node verdicts (for inheritance)
            
        Returns:
            ValidityVerdict with score, state, attribution, etc.
        """
        # Step 0: Check for kill switches FIRST
        kill_signal = next((s for s in signals if s.triggers_kill), None)
        
        if kill_signal:
            return self._create_killed_verdict(kill_signal, signals)
        
        # Step 1: Check upstream inheritance
        if upstream_verdicts:
            invalid_upstream = [v for v in upstream_verdicts if v.state == ValidityState.INVALID]
            if invalid_upstream:
                return self._create_inherited_invalid_verdict(invalid_upstream[0])
        
        # Step 2: Compute score using appropriate aggregation
        if self.use_amplification:
            score, dominant = AggregationMode.with_dominance(signals)
            
            # Apply amplification
            severe_count = sum(1 for s in signals if s.severity > 50)
            if severe_count >= 2:
                amplification = 1 + (severe_count - 1) * 0.2
                penalty = 100 - score
                score = max(0, 100 - penalty * amplification)
        else:
            score = AggregationMode.linear(signals)
            dominant = None
            if signals:
                penalties = [(s.failure_mode, s.failure_mode.get_weight() * s.severity) for s in signals]
                if penalties:
                    dominant = max(penalties, key=lambda x: x[1])[0]
        
        # Step 3: Apply inheritance cap (hard min)
        inherited_from = None
        pre_inheritance_score = score
        
        if upstream_verdicts:
            for upstream in upstream_verdicts:
                if upstream.score < score:
                    score = upstream.score
                    inherited_from = upstream.verdict_id[:8]  # Short ID
        
        # Step 4: Determine state
        state = self._determine_state(score, signals)
        
        # Step 5: Compute attribution (must sum to 100% when score < 100)
        attributions = self._compute_attribution(signals, score)
        
        # Step 6: Compute validity confidence (separate from score!)
        validity_confidence = ConfidencePolicy.compute_validity_confidence(signals)
        
        # Step 7: Causal ordering (by first_detected_at)
        causal_order = tuple(
            s.failure_mode for s in sorted(signals, key=lambda x: x.first_detected_at)
        )
        
        return ValidityVerdict(
            score=score,
            state=state,
            attributions=tuple(attributions),
            causal_order=causal_order,
            inherited_from=inherited_from,
            pre_inheritance_score=pre_inheritance_score if inherited_from else None,
        )
    
    def _determine_state(
        self,
        score: float,
        signals: List[FailureSignal],
    ) -> ValidityState:
        """
        Determine validity state from score.
        
        Uses FM-specific thresholds if enabled.
        """
        if not self.use_fm_specific_thresholds:
            return ValidityState.from_score(score)
        
        # Check if any FM has stricter thresholds that would trigger
        for signal in signals:
            thresholds = FMThresholds.get_thresholds(signal.failure_mode)
            
            # This FM's contribution to score loss
            penalty = signal.failure_mode.get_weight() * signal.severity
            
            # If this FM alone would cause INVALID under its threshold
            if penalty > (100 - thresholds["invalid"]):
                return ValidityState.INVALID
            
            # If this FM alone would cause DEGRADED under its threshold
            if penalty > (100 - thresholds["degraded"]):
                if score < Thresholds.VALID_MIN:
                    return ValidityState.DEGRADED
        
        # Fall back to standard thresholds
        return ValidityState.from_score(score)
    
    def _compute_attribution(
        self,
        signals: List[FailureSignal],
        score: float,
    ) -> List[Tuple[FailureMode, float]]:
        """
        Compute attribution that sums to 100%.
        
        IMPORTANT: This does NOT use confidence to discount.
        """
        if score >= 100:
            return []
        
        validity_loss = 100.0 - score
        
        # Compute raw penalties
        penalties: List[Tuple[FailureMode, float]] = []
        total_penalty = 0.0
        
        for signal in signals:
            weight = signal.failure_mode.get_weight()
            penalty = weight * signal.severity
            if penalty > 0:
                penalties.append((signal.failure_mode, penalty))
                total_penalty += penalty
        
        if total_penalty == 0:
            return []
        
        # Normalize to 100%
        return [
            (fm, (penalty / total_penalty) * 100.0)
            for fm, penalty in penalties
        ]
    
    def _create_killed_verdict(
        self,
        kill_signal: FailureSignal,
        all_signals: List[FailureSignal],
    ) -> ValidityVerdict:
        """Create a verdict for a kill switch activation."""
        return ValidityVerdict(
            score=0.0,
            state=ValidityState.INVALID,
            killed_by=kill_signal.failure_mode,
            kill_reason=kill_signal.kill_reason,
            attributions=((kill_signal.failure_mode, 100.0),),
            causal_order=tuple(
                s.failure_mode for s in sorted(all_signals, key=lambda x: x.first_detected_at)
            ),
        )
    
    def _create_inherited_invalid_verdict(
        self,
        upstream: ValidityVerdict,
    ) -> ValidityVerdict:
        """Create a verdict inherited from an INVALID upstream."""
        return ValidityVerdict(
            score=0.0,
            state=ValidityState.INVALID,
            kill_reason=f"Inherited INVALID from upstream (verdict {upstream.verdict_id[:8]})",
            inherited_from=upstream.verdict_id[:8],
            pre_inheritance_score=None,  # Never computed own score
        )


# =============================================================================
# VALIDITY ENGINE (orchestrates everything)
# =============================================================================

class ValidityEngine:
    """
    Complete validity assessment engine.
    
    Orchestrates:
        1. Detector execution
        2. Validity computation
        3. Inheritance tracking
        4. History management
    """
    
    def __init__(self):
        self.detector_suite = DetectorSuite()
        self.computer = ValidityComputer()
        self._upstream_verdicts: Dict[str, ValidityVerdict] = {}
    
    def register_upstream(self, node_id: str, verdict: ValidityVerdict) -> None:
        """Register an upstream node's verdict for inheritance."""
        self._upstream_verdicts[node_id] = verdict
    
    def assess(
        self,
        z_t: Any,  # pd.Series
        Z_t: Any,  # pd.Series
        target_id: str = "target",
        reference_z_t: Any = None,
        reference_series: List[Any] = None,
        upstream_ids: Optional[List[str]] = None,
    ) -> Tuple[ValidityVerdict, List[FailureSignal], float]:
        """
        Run complete validity assessment.
        
        Returns:
            (verdict, signals, validity_confidence)
        """
        # Run detectors
        signals = self.detector_suite.run_all(
            z_t=z_t,
            Z_t=Z_t,
            reference_z_t=reference_z_t,
            reference_series=reference_series,
        )
        
        # Get upstream verdicts
        upstream_verdicts = None
        if upstream_ids:
            upstream_verdicts = [
                self._upstream_verdicts[uid]
                for uid in upstream_ids
                if uid in self._upstream_verdicts
            ]
        
        # Compute validity
        verdict = self.computer.compute(signals, upstream_verdicts)
        
        # Compute validity confidence (separate from score)
        validity_confidence = self.detector_suite.compute_validity_confidence(signals)
        
        return verdict, signals, validity_confidence


# =============================================================================
# TESTS FOR CORRECTED BEHAVIOR
# =============================================================================

def test_confidence_not_discounting():
    """
    Test that confidence does NOT reduce severity.
    
    OLD BUG: severity=100, confidence=0.5 → effective=50
    CORRECT: severity=100, confidence=0.5 → penalty based on 100
    """
    from .core import FailureMode, FailureSignal
    
    # Signal with low confidence but high severity
    signal = FailureSignal(
        failure_mode=FailureMode.FM1_VARIANCE_REGIME,
        severity=100,
        confidence=0.5,  # LOW confidence
        explanation="Test",
    )
    
    computer = ValidityComputer(use_amplification=False)
    verdict = computer.compute([signal])
    
    # Expected: 100 - (0.18 * 100) = 82
    # NOT: 100 - (0.18 * 100 * 0.5) = 91
    expected_score = 100 - (Weights.FM1_VARIANCE * 100)
    
    assert abs(verdict.score - expected_score) < 0.1, \
        f"Confidence should NOT discount severity. Got {verdict.score}, expected {expected_score}"
    
    print("✓ test_confidence_not_discounting passed")


def test_kill_switch_overrides():
    """Test that kill switches override normal scoring."""
    from .core import FailureMode, FailureSignal
    
    # Kill switch signal with moderate severity
    signals = [
        FailureSignal(
            failure_mode=FailureMode.FM1_VARIANCE_REGIME,
            severity=10,  # Low
            confidence=0.9,
            explanation="Minor",
        ),
        FailureSignal(
            failure_mode=FailureMode.FM4_STRUCTURAL_BREAK,
            severity=50,  # Moderate
            confidence=0.9,
            triggers_kill=True,
            kill_reason="Break exceeds threshold",
            explanation="Kill",
        ),
    ]
    
    computer = ValidityComputer()
    verdict = computer.compute(signals)
    
    assert verdict.state == ValidityState.INVALID, "Kill switch should force INVALID"
    assert verdict.is_killed, "Should be marked as killed"
    assert verdict.killed_by == FailureMode.FM4_STRUCTURAL_BREAK
    
    print("✓ test_kill_switch_overrides passed")


def test_amplification():
    """Test non-linear amplification for multiple severe failures."""
    from .core import FailureMode, FailureSignal
    
    # Two severe failures
    signals = [
        FailureSignal(
            failure_mode=FailureMode.FM1_VARIANCE_REGIME,
            severity=60,
            confidence=0.9,
            explanation="Severe",
        ),
        FailureSignal(
            failure_mode=FailureMode.FM2_MEAN_DRIFT,
            severity=60,
            confidence=0.9,
            explanation="Severe",
        ),
    ]
    
    # Without amplification
    computer_no_amp = ValidityComputer(use_amplification=False)
    verdict_no_amp = computer_no_amp.compute(signals)
    
    # With amplification
    computer_amp = ValidityComputer(use_amplification=True)
    verdict_amp = computer_amp.compute(signals)
    
    # Amplified score should be LOWER (more penalty)
    assert verdict_amp.score < verdict_no_amp.score, \
        f"Amplification should reduce score. Amp={verdict_amp.score}, NoAmp={verdict_no_amp.score}"
    
    print("✓ test_amplification passed")


def test_attribution_sums_to_100():
    """Test that attribution always sums to 100% when score < 100."""
    from .core import FailureMode, FailureSignal
    
    signals = [
        FailureSignal(
            failure_mode=FailureMode.FM1_VARIANCE_REGIME,
            severity=30,
            confidence=0.9,
            explanation="A",
        ),
        FailureSignal(
            failure_mode=FailureMode.FM2_MEAN_DRIFT,
            severity=40,
            confidence=0.8,
            explanation="B",
        ),
        FailureSignal(
            failure_mode=FailureMode.FM5_OUTLIER_CONTAMINATION,
            severity=20,
            confidence=0.7,
            explanation="C",
        ),
    ]
    
    computer = ValidityComputer()
    verdict = computer.compute(signals)
    
    if verdict.score < 100:
        total = sum(pct for _, pct in verdict.attributions)
        assert 99.0 <= total <= 101.0, f"Attribution should sum to 100%, got {total}%"
    
    print("✓ test_attribution_sums_to_100 passed")


def run_all_compute_tests():
    """Run all validity computer tests."""
    print("\n" + "=" * 60)
    print("VALIDITY COMPUTER TESTS")
    print("=" * 60 + "\n")
    
    test_confidence_not_discounting()
    test_kill_switch_overrides()
    test_amplification()
    test_attribution_sums_to_100()
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    run_all_compute_tests()
