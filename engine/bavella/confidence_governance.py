"""
Bavella v2 — Confidence-Governed Validity
==========================================

PROBLEM: validity_confidence is computed but not used to govern behavior.

SOLUTION: Deterministic rules that make low confidence → conservative behavior.

Rules (frozen, no debate):
    1. If score in [65-75] AND confidence < 0.7 → force DEGRADED
    2. If score in [25-35] AND confidence < 0.7 → force INVALID  
    3. Exports require confidence ≥ 0.8 unless explicitly requested
    4. Predictions (L2) require confidence ≥ 0.7

This is how Bavella becomes TRUST INFRASTRUCTURE.

Copyright 2024-2026 Bavella Technologies Sarl
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple

from .core import (
    ValidityState, ValidityVerdict, FailureMode, FailureSignal,
    Thresholds
)


# =============================================================================
# CONFIDENCE GOVERNANCE RULES (FROZEN)
# =============================================================================

class ConfidenceGovernanceRules:
    """
    Frozen rules for how confidence affects governance.
    
    These rules are NOT configurable. They are the product.
    Changing them requires a new version.
    
    Philosophy:
        - "We're uncertain" should NEVER make things look better
        - Low confidence in borderline cases → be conservative
        - Exports require high confidence
        - Predictions require medium-high confidence
    """
    
    # Version for audit trail
    VERSION = "1.0.0"
    
    # Threshold zones
    VALID_DEGRADED_BOUNDARY = Thresholds.VALID_MIN  # 70
    DEGRADED_INVALID_BOUNDARY = Thresholds.DEGRADED_MIN  # 30
    
    # Zone widths for confidence adjustment
    UNCERTAINTY_ZONE_WIDTH = 10.0
    
    # Confidence thresholds
    MIN_CONFIDENCE_FOR_VALID = 0.7
    MIN_CONFIDENCE_FOR_EXPORTS = 0.8
    MIN_CONFIDENCE_FOR_PREDICTIONS = 0.7
    MIN_CONFIDENCE_TO_AVOID_WATERMARK = 0.8
    
    # Watermark thresholds
    UNCERTAIN_WATERMARK_THRESHOLD = 0.6
    
    @classmethod
    def is_in_valid_uncertainty_zone(cls, score: float) -> bool:
        """Is score in the zone where uncertainty could push to DEGRADED?"""
        lower = cls.VALID_DEGRADED_BOUNDARY - cls.UNCERTAINTY_ZONE_WIDTH
        upper = cls.VALID_DEGRADED_BOUNDARY + cls.UNCERTAINTY_ZONE_WIDTH
        return lower <= score <= upper
    
    @classmethod
    def is_in_degraded_uncertainty_zone(cls, score: float) -> bool:
        """Is score in the zone where uncertainty could push to INVALID?"""
        lower = cls.DEGRADED_INVALID_BOUNDARY - cls.UNCERTAINTY_ZONE_WIDTH
        upper = cls.DEGRADED_INVALID_BOUNDARY + cls.UNCERTAINTY_ZONE_WIDTH
        return lower <= score <= upper
    
    @classmethod
    def should_downgrade_valid_to_degraded(
        cls,
        score: float,
        confidence: float,
    ) -> bool:
        """
        Should a VALID score be downgraded to DEGRADED?
        
        Rule: If score in [65-80] AND confidence < 0.7 → DEGRADED
        """
        if score < cls.VALID_DEGRADED_BOUNDARY:
            return False  # Already not VALID
        
        if not cls.is_in_valid_uncertainty_zone(score):
            return False  # Not in uncertainty zone
        
        return confidence < cls.MIN_CONFIDENCE_FOR_VALID
    
    @classmethod
    def should_downgrade_degraded_to_invalid(
        cls,
        score: float,
        confidence: float,
    ) -> bool:
        """
        Should a DEGRADED score be downgraded to INVALID?
        
        Rule: If score in [25-40] AND confidence < 0.7 → INVALID
        """
        if score >= cls.VALID_DEGRADED_BOUNDARY:
            return False  # Not DEGRADED
        
        if score < cls.DEGRADED_INVALID_BOUNDARY:
            return False  # Already INVALID
        
        if not cls.is_in_degraded_uncertainty_zone(score):
            return False
        
        return confidence < cls.MIN_CONFIDENCE_FOR_VALID
    
    @classmethod
    def can_export(cls, score: float, confidence: float) -> Tuple[bool, str]:
        """
        Can we export full data?
        
        Requires:
            - VALID state (score >= 70)
            - High confidence (>= 0.8)
        """
        if score < cls.VALID_DEGRADED_BOUNDARY:
            return False, f"Export requires VALID state (score >= {cls.VALID_DEGRADED_BOUNDARY})"
        
        if confidence < cls.MIN_CONFIDENCE_FOR_EXPORTS:
            return False, f"Export requires confidence >= {cls.MIN_CONFIDENCE_FOR_EXPORTS}"
        
        return True, ""
    
    @classmethod
    def can_emit_predictions(
        cls,
        score: float,
        confidence: float,
    ) -> Tuple[bool, str]:
        """
        Can we emit predictions (L2 outputs)?
        
        Requires:
            - VALID state
            - Medium-high confidence
        """
        if score < cls.VALID_DEGRADED_BOUNDARY:
            return False, "Predictions require VALID state"
        
        if confidence < cls.MIN_CONFIDENCE_FOR_PREDICTIONS:
            return False, f"Predictions require confidence >= {cls.MIN_CONFIDENCE_FOR_PREDICTIONS}"
        
        return True, ""
    
    @classmethod
    def requires_uncertainty_watermark(
        cls,
        score: float,
        confidence: float,
    ) -> bool:
        """Does this output require an UNCERTAINTY watermark?"""
        # DEGRADED always requires watermark
        if score < cls.VALID_DEGRADED_BOUNDARY:
            return True
        
        # Low confidence VALID requires watermark
        if confidence < cls.MIN_CONFIDENCE_TO_AVOID_WATERMARK:
            return True
        
        return False


# =============================================================================
# CONFIDENCE-GOVERNED VERDICT
# =============================================================================

@dataclass(frozen=True)
class ConfidenceGovernedVerdict:
    """
    Validity verdict with confidence governance applied.
    
    This wraps a raw ValidityVerdict and applies confidence rules.
    """
    # Raw verdict (before confidence governance)
    raw_score: float
    raw_state: ValidityState
    
    # Confidence
    validity_confidence: float
    
    # Governed verdict (after confidence rules)
    governed_score: float
    governed_state: ValidityState
    
    # What happened
    was_downgraded: bool
    downgrade_reason: Optional[str]
    
    # Export/prediction permissions
    can_export: bool
    export_denial_reason: Optional[str]
    
    can_emit_predictions: bool
    prediction_denial_reason: Optional[str]
    
    # Watermark
    requires_watermark: bool
    watermark_text: str
    
    # Original verdict data
    attributions: Tuple[Tuple[FailureMode, float], ...] = ()
    killed_by: Optional[FailureMode] = None
    kill_reason: Optional[str] = None
    
    # Metadata
    rules_version: str = ConfidenceGovernanceRules.VERSION
    computed_at: datetime = field(default_factory=datetime.utcnow)


def apply_confidence_governance(
    verdict: ValidityVerdict,
    validity_confidence: float,
) -> ConfidenceGovernedVerdict:
    """
    Apply confidence governance rules to a verdict.
    
    This is the key function that makes confidence GOVERNING, not just informative.
    """
    rules = ConfidenceGovernanceRules
    
    raw_score = verdict.score
    raw_state = verdict.state
    
    # Start with raw values
    governed_score = raw_score
    governed_state = raw_state
    was_downgraded = False
    downgrade_reason = None
    
    # Apply downgrade rules
    if raw_state == ValidityState.VALID:
        if rules.should_downgrade_valid_to_degraded(raw_score, validity_confidence):
            governed_state = ValidityState.DEGRADED
            was_downgraded = True
            downgrade_reason = (
                f"Score {raw_score:.1f} is in uncertainty zone [{rules.VALID_DEGRADED_BOUNDARY - rules.UNCERTAINTY_ZONE_WIDTH:.0f}-{rules.VALID_DEGRADED_BOUNDARY + rules.UNCERTAINTY_ZONE_WIDTH:.0f}] "
                f"with low confidence ({validity_confidence:.2f} < {rules.MIN_CONFIDENCE_FOR_VALID}). "
                "Downgraded to DEGRADED for safety."
            )
    
    elif raw_state == ValidityState.DEGRADED:
        if rules.should_downgrade_degraded_to_invalid(raw_score, validity_confidence):
            governed_state = ValidityState.INVALID
            was_downgraded = True
            downgrade_reason = (
                f"Score {raw_score:.1f} is in uncertainty zone [{rules.DEGRADED_INVALID_BOUNDARY - rules.UNCERTAINTY_ZONE_WIDTH:.0f}-{rules.DEGRADED_INVALID_BOUNDARY + rules.UNCERTAINTY_ZONE_WIDTH:.0f}] "
                f"with low confidence ({validity_confidence:.2f} < {rules.MIN_CONFIDENCE_FOR_VALID}). "
                "Downgraded to INVALID for safety."
            )
    
    # Check export permission
    can_export, export_denial = rules.can_export(governed_score, validity_confidence)
    
    # Check prediction permission
    can_pred, pred_denial = rules.can_emit_predictions(governed_score, validity_confidence)
    
    # Watermark
    requires_watermark = rules.requires_uncertainty_watermark(governed_score, validity_confidence)
    
    watermark_text = ""
    if requires_watermark:
        parts = []
        if governed_state == ValidityState.DEGRADED:
            parts.append(f"VALIDITY DEGRADED ({governed_score:.0f}/100)")
        if validity_confidence < rules.MIN_CONFIDENCE_TO_AVOID_WATERMARK:
            parts.append(f"UNCERTAIN (confidence: {validity_confidence:.0%})")
        if was_downgraded:
            parts.append("Conservatively downgraded")
        watermark_text = " | ".join(parts)
    
    return ConfidenceGovernedVerdict(
        raw_score=raw_score,
        raw_state=raw_state,
        validity_confidence=validity_confidence,
        governed_score=governed_score,
        governed_state=governed_state,
        was_downgraded=was_downgraded,
        downgrade_reason=downgrade_reason,
        can_export=can_export,
        export_denial_reason=export_denial if not can_export else None,
        can_emit_predictions=can_pred,
        prediction_denial_reason=pred_denial if not can_pred else None,
        requires_watermark=requires_watermark,
        watermark_text=watermark_text,
        attributions=verdict.attributions,
        killed_by=verdict.killed_by,
        kill_reason=verdict.kill_reason,
    )


# =============================================================================
# CONFIDENCE-AWARE GOVERNOR
# =============================================================================

class ConfidenceAwareGovernor:
    """
    Governor that uses confidence to make emission decisions.
    
    This is the upgraded Governor that treats confidence as a GOVERNING input.
    """
    
    def __init__(self):
        self._verdicts: Dict[str, ConfidenceGovernedVerdict] = {}
        self._emission_log: List[Dict[str, Any]] = []
    
    def register_verdict(
        self,
        node_id: str,
        raw_verdict: ValidityVerdict,
        validity_confidence: float,
    ) -> ConfidenceGovernedVerdict:
        """Register a verdict with confidence governance applied."""
        governed = apply_confidence_governance(raw_verdict, validity_confidence)
        self._verdicts[node_id] = governed
        return governed
    
    def can_emit_metric(self, node_id: str) -> Tuple[bool, str]:
        """Can we emit a metric (L1) from this node?"""
        verdict = self._verdicts.get(node_id)
        if verdict is None:
            return False, "No verdict registered"
        
        if verdict.governed_state == ValidityState.INVALID:
            return False, f"Node is INVALID: {verdict.downgrade_reason or 'score too low'}"
        
        return True, ""
    
    def can_emit_prediction(self, node_id: str) -> Tuple[bool, str]:
        """Can we emit a prediction (L2) from this node?"""
        verdict = self._verdicts.get(node_id)
        if verdict is None:
            return False, "No verdict registered"
        
        if not verdict.can_emit_predictions:
            return False, verdict.prediction_denial_reason or "Predictions not permitted"
        
        return True, ""
    
    def can_export(self, node_id: str) -> Tuple[bool, str]:
        """Can we do a full export from this node?"""
        verdict = self._verdicts.get(node_id)
        if verdict is None:
            return False, "No verdict registered"
        
        if not verdict.can_export:
            return False, verdict.export_denial_reason or "Export not permitted"
        
        return True, ""
    
    def get_watermark(self, node_id: str) -> Optional[str]:
        """Get watermark text if required."""
        verdict = self._verdicts.get(node_id)
        if verdict is None:
            return None
        
        if verdict.requires_watermark:
            return verdict.watermark_text
        
        return None
    
    def emit_with_governance(
        self,
        node_id: str,
        output_type: str,
        value: Any,
    ) -> Tuple[bool, Optional[Any], Optional[str], Optional[str]]:
        """
        Attempt to emit a value with full governance.
        
        Returns:
            (permitted, value_or_none, watermark_or_none, denial_reason_or_none)
        """
        if output_type == "metric":
            permitted, reason = self.can_emit_metric(node_id)
        elif output_type == "prediction":
            permitted, reason = self.can_emit_prediction(node_id)
        elif output_type == "export":
            permitted, reason = self.can_export(node_id)
        else:
            permitted, reason = True, ""  # L0 always permitted
        
        # Log the emission attempt
        self._emission_log.append({
            "node_id": node_id,
            "output_type": output_type,
            "permitted": permitted,
            "denial_reason": reason if not permitted else None,
            "timestamp": datetime.utcnow().isoformat(),
        })
        
        if not permitted:
            return False, None, None, reason
        
        watermark = self.get_watermark(node_id)
        return True, value, watermark, None


# =============================================================================
# TESTS
# =============================================================================

def test_confidence_downgrade_valid_to_degraded():
    """Test that low confidence downgrades borderline VALID to DEGRADED."""
    from .core import ValidityState
    
    # Borderline VALID (72) with low confidence (0.5)
    raw_verdict = ValidityVerdict(
        score=72.0,
        state=ValidityState.VALID,
    )
    
    governed = apply_confidence_governance(raw_verdict, validity_confidence=0.5)
    
    assert governed.raw_state == ValidityState.VALID
    assert governed.governed_state == ValidityState.DEGRADED
    assert governed.was_downgraded
    assert "uncertainty zone" in governed.downgrade_reason.lower()
    
    print(f"Raw: {governed.raw_state.value}, Governed: {governed.governed_state.value}")
    print(f"Reason: {governed.downgrade_reason}")
    print("✓ test_confidence_downgrade_valid_to_degraded passed")


def test_confidence_downgrade_degraded_to_invalid():
    """Test that low confidence downgrades borderline DEGRADED to INVALID."""
    from .core import ValidityState
    
    # Borderline DEGRADED (32) with low confidence (0.5)
    raw_verdict = ValidityVerdict(
        score=32.0,
        state=ValidityState.DEGRADED,
    )
    
    governed = apply_confidence_governance(raw_verdict, validity_confidence=0.5)
    
    assert governed.raw_state == ValidityState.DEGRADED
    assert governed.governed_state == ValidityState.INVALID
    assert governed.was_downgraded
    
    print(f"Raw: {governed.raw_state.value}, Governed: {governed.governed_state.value}")
    print("✓ test_confidence_downgrade_degraded_to_invalid passed")


def test_high_confidence_no_downgrade():
    """Test that high confidence doesn't downgrade even in uncertainty zone."""
    from .core import ValidityState
    
    # Borderline VALID (72) with HIGH confidence (0.9)
    raw_verdict = ValidityVerdict(
        score=72.0,
        state=ValidityState.VALID,
    )
    
    governed = apply_confidence_governance(raw_verdict, validity_confidence=0.9)
    
    assert governed.raw_state == ValidityState.VALID
    assert governed.governed_state == ValidityState.VALID
    assert not governed.was_downgraded
    
    print("✓ test_high_confidence_no_downgrade passed")


def test_export_requires_high_confidence():
    """Test that exports require high confidence."""
    from .core import ValidityState
    
    # VALID with low confidence
    raw_verdict = ValidityVerdict(
        score=85.0,
        state=ValidityState.VALID,
    )
    
    governed = apply_confidence_governance(raw_verdict, validity_confidence=0.6)
    
    assert not governed.can_export
    assert "confidence" in governed.export_denial_reason.lower()
    
    print(f"Export denied: {governed.export_denial_reason}")
    print("✓ test_export_requires_high_confidence passed")


def test_predictions_require_valid_and_confidence():
    """Test that predictions require both VALID state and confidence."""
    from .core import ValidityState
    
    # DEGRADED - should deny predictions
    degraded = apply_confidence_governance(
        ValidityVerdict(score=50.0, state=ValidityState.DEGRADED),
        validity_confidence=0.9
    )
    assert not degraded.can_emit_predictions
    
    # VALID with low confidence - should deny predictions
    low_conf = apply_confidence_governance(
        ValidityVerdict(score=85.0, state=ValidityState.VALID),
        validity_confidence=0.5
    )
    assert not low_conf.can_emit_predictions
    
    # VALID with high confidence - should allow
    high_conf = apply_confidence_governance(
        ValidityVerdict(score=85.0, state=ValidityState.VALID),
        validity_confidence=0.9
    )
    assert high_conf.can_emit_predictions
    
    print("✓ test_predictions_require_valid_and_confidence passed")


def test_watermark_for_uncertainty():
    """Test that low confidence triggers watermark."""
    from .core import ValidityState
    
    # VALID with low confidence - should require watermark
    governed = apply_confidence_governance(
        ValidityVerdict(score=85.0, state=ValidityState.VALID),
        validity_confidence=0.6
    )
    
    assert governed.requires_watermark
    assert "uncertain" in governed.watermark_text.lower()
    
    print(f"Watermark: {governed.watermark_text}")
    print("✓ test_watermark_for_uncertainty passed")


def run_all_confidence_tests():
    print("\n" + "=" * 60)
    print("CONFIDENCE GOVERNANCE TESTS")
    print("=" * 60 + "\n")
    
    test_confidence_downgrade_valid_to_degraded()
    print()
    test_confidence_downgrade_degraded_to_invalid()
    print()
    test_high_confidence_no_downgrade()
    print()
    test_export_requires_high_confidence()
    print()
    test_predictions_require_valid_and_confidence()
    print()
    test_watermark_for_uncertainty()
    
    print("\n" + "=" * 60)
    print("ALL CONFIDENCE GOVERNANCE TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    run_all_confidence_tests()
