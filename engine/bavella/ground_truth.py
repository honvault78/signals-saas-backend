"""
Bavella v2 — Ground Truth Regimes
==================================

Enterprise ML reality:
    - Ground truth may be DELAYED (outcomes known weeks/months later)
    - Ground truth may be PARTIAL (only some predictions can be evaluated)
    - Ground truth may be ABSENT (no way to know if model is right)

This module implements:
    1. Ground truth mode tracking (none/delayed/partial/full)
    2. Coverage disclosure
    3. Residual audit status (inactive/active/partial)
    4. Governor enforcement of disclosure requirements

Without this, ML review is NOT enterprise-safe.

Copyright 2024-2026 Bavella Technologies Sarl
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Tuple
import json


# =============================================================================
# GROUND TRUTH MODE
# =============================================================================

class GroundTruthMode(Enum):
    """
    How ground truth is available for a model's predictions.
    
    FULL: All predictions can be evaluated immediately
    DELAYED: Outcomes available after a delay (e.g., loan defaults)
    PARTIAL: Only some predictions have outcomes
    NONE: No way to evaluate predictions (audit on inputs/outputs only)
    """
    FULL = "full"
    DELAYED = "delayed"
    PARTIAL = "partial"
    NONE = "none"


class ResidualAuditStatus(Enum):
    """
    Status of residual (error) analysis.
    
    ACTIVE: Residuals are being computed and analyzed
    PARTIAL: Some residuals available but not full coverage
    INACTIVE: Residual analysis not possible (no ground truth)
    BACKDATED: Running retrospective analysis on delayed outcomes
    """
    ACTIVE = "active"
    PARTIAL = "partial"
    INACTIVE = "inactive"
    BACKDATED = "backdated"


# =============================================================================
# OUTCOME STREAM (describes how ground truth arrives)
# =============================================================================

@dataclass
class OutcomeStream:
    """
    Describes how ground truth/outcomes arrive for a model.
    
    This is essential for:
        - Knowing when we can compute residuals
        - Determining audit completeness
        - Disclosure requirements
    """
    stream_id: str
    model_node_id: str
    
    # Mode
    mode: GroundTruthMode
    
    # Delay (for DELAYED mode)
    delay_days: Optional[int] = None
    delay_description: Optional[str] = None
    
    # Coverage (for PARTIAL mode)
    coverage_pct: float = 100.0
    coverage_description: Optional[str] = None
    
    # Availability
    first_available_at: Optional[datetime] = None
    last_available_at: Optional[datetime] = None
    
    # For PARTIAL: which predictions have outcomes?
    has_outcome_filter: Optional[str] = None  # e.g., "approved_loans_only"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "stream_id": self.stream_id,
            "model_node_id": self.model_node_id,
            "mode": self.mode.value,
            "delay_days": self.delay_days,
            "coverage_pct": self.coverage_pct,
            "coverage_description": self.coverage_description,
        }
    
    @property
    def can_compute_residuals(self) -> bool:
        """Can we compute residuals with this outcome stream?"""
        return self.mode != GroundTruthMode.NONE
    
    @property
    def residual_audit_status(self) -> ResidualAuditStatus:
        """Current status of residual analysis."""
        if self.mode == GroundTruthMode.NONE:
            return ResidualAuditStatus.INACTIVE
        elif self.mode == GroundTruthMode.PARTIAL:
            return ResidualAuditStatus.PARTIAL
        elif self.mode == GroundTruthMode.DELAYED:
            return ResidualAuditStatus.BACKDATED
        else:
            return ResidualAuditStatus.ACTIVE


# =============================================================================
# GROUND TRUTH REGIME (attached to model nodes)
# =============================================================================

@dataclass
class GroundTruthRegime:
    """
    Complete ground truth specification for a model.
    
    This gets attached to ML model nodes and:
        - Determines what residual analysis is possible
        - Triggers disclosure requirements
        - Affects validity interpretation
    """
    regime_id: str
    model_node_id: str
    
    # Outcome streams (a model may have multiple)
    outcome_streams: List[OutcomeStream] = field(default_factory=list)
    
    # Current status
    residual_audit_status: ResidualAuditStatus = ResidualAuditStatus.INACTIVE
    
    # Coverage metrics
    overall_coverage_pct: float = 0.0
    predictions_without_outcomes: int = 0
    predictions_with_outcomes: int = 0
    
    # Timestamps
    last_outcome_received_at: Optional[datetime] = None
    oldest_pending_prediction_at: Optional[datetime] = None
    
    # Configuration
    requires_disclosure: bool = True
    
    def update_coverage(
        self,
        with_outcomes: int,
        without_outcomes: int,
    ) -> None:
        """Update coverage metrics."""
        self.predictions_with_outcomes = with_outcomes
        self.predictions_without_outcomes = without_outcomes
        
        total = with_outcomes + without_outcomes
        if total > 0:
            self.overall_coverage_pct = (with_outcomes / total) * 100
        else:
            self.overall_coverage_pct = 0.0
    
    def get_effective_mode(self) -> GroundTruthMode:
        """Get the effective mode based on all outcome streams."""
        if not self.outcome_streams:
            return GroundTruthMode.NONE
        
        modes = [s.mode for s in self.outcome_streams]
        
        # Worst case wins
        if GroundTruthMode.NONE in modes:
            return GroundTruthMode.NONE
        if GroundTruthMode.PARTIAL in modes:
            return GroundTruthMode.PARTIAL
        if GroundTruthMode.DELAYED in modes:
            return GroundTruthMode.DELAYED
        return GroundTruthMode.FULL
    
    def to_disclosure(self) -> Dict[str, Any]:
        """Generate disclosure information for API responses."""
        return {
            "ground_truth_mode": self.get_effective_mode().value,
            "residual_audit_status": self.residual_audit_status.value,
            "coverage_pct": self.overall_coverage_pct,
            "predictions_auditable": self.predictions_with_outcomes,
            "predictions_unauditable": self.predictions_without_outcomes,
            "outcome_streams": [s.to_dict() for s in self.outcome_streams],
            "warnings": self._generate_warnings(),
        }
    
    def _generate_warnings(self) -> List[str]:
        """Generate warnings based on ground truth status."""
        warnings = []
        
        mode = self.get_effective_mode()
        
        if mode == GroundTruthMode.NONE:
            warnings.append(
                "⚠️ NO GROUND TRUTH: This model cannot be evaluated against actual outcomes. "
                "Validity assessment is based on input/output analysis only."
            )
        elif mode == GroundTruthMode.PARTIAL:
            warnings.append(
                f"⚠️ PARTIAL COVERAGE: Only {self.overall_coverage_pct:.0f}% of predictions "
                "can be evaluated against outcomes. Selection bias may affect residual analysis."
            )
        elif mode == GroundTruthMode.DELAYED:
            max_delay = max(
                (s.delay_days for s in self.outcome_streams if s.delay_days),
                default=0
            )
            warnings.append(
                f"⚠️ DELAYED OUTCOMES: Ground truth available after ~{max_delay} days. "
                "Recent predictions cannot yet be evaluated."
            )
        
        if self.residual_audit_status == ResidualAuditStatus.INACTIVE:
            warnings.append(
                "⚠️ RESIDUAL AUDIT INACTIVE: FM6 (distributional shift) analysis on residuals "
                "is disabled due to lack of ground truth."
            )
        
        return warnings


# =============================================================================
# GROUND TRUTH MANAGER
# =============================================================================

class GroundTruthManager:
    """
    Manages ground truth regimes for all model nodes.
    
    Responsibilities:
        - Track outcome streams
        - Compute coverage
        - Generate disclosures
        - Enforce disclosure requirements
    """
    
    def __init__(self):
        self._regimes: Dict[str, GroundTruthRegime] = {}
        self._pending_outcomes: Dict[str, List[Dict[str, Any]]] = {}
    
    def register_model(
        self,
        model_node_id: str,
        outcome_streams: List[OutcomeStream],
    ) -> GroundTruthRegime:
        """Register a model with its outcome streams."""
        regime = GroundTruthRegime(
            regime_id=f"gt_{model_node_id}",
            model_node_id=model_node_id,
            outcome_streams=outcome_streams,
        )
        
        # Determine initial status
        mode = regime.get_effective_mode()
        if mode == GroundTruthMode.NONE:
            regime.residual_audit_status = ResidualAuditStatus.INACTIVE
        elif mode == GroundTruthMode.PARTIAL:
            regime.residual_audit_status = ResidualAuditStatus.PARTIAL
        elif mode == GroundTruthMode.DELAYED:
            regime.residual_audit_status = ResidualAuditStatus.BACKDATED
        else:
            regime.residual_audit_status = ResidualAuditStatus.ACTIVE
        
        self._regimes[model_node_id] = regime
        return regime
    
    def record_prediction(
        self,
        model_node_id: str,
        prediction_id: str,
        prediction_ts: datetime,
    ) -> None:
        """Record a prediction awaiting outcome."""
        if model_node_id not in self._pending_outcomes:
            self._pending_outcomes[model_node_id] = []
        
        self._pending_outcomes[model_node_id].append({
            "prediction_id": prediction_id,
            "prediction_ts": prediction_ts,
            "outcome_received": False,
        })
    
    def record_outcome(
        self,
        model_node_id: str,
        prediction_id: str,
        outcome_ts: datetime,
        outcome_value: Any,
    ) -> None:
        """Record an outcome for a prediction."""
        if model_node_id not in self._pending_outcomes:
            return
        
        # Find and update the prediction
        for pred in self._pending_outcomes[model_node_id]:
            if pred["prediction_id"] == prediction_id:
                pred["outcome_received"] = True
                pred["outcome_ts"] = outcome_ts
                pred["outcome_value"] = outcome_value
                break
        
        # Update regime coverage
        regime = self._regimes.get(model_node_id)
        if regime:
            with_outcomes = sum(
                1 for p in self._pending_outcomes[model_node_id]
                if p["outcome_received"]
            )
            without_outcomes = sum(
                1 for p in self._pending_outcomes[model_node_id]
                if not p["outcome_received"]
            )
            regime.update_coverage(with_outcomes, without_outcomes)
            regime.last_outcome_received_at = outcome_ts
    
    def get_regime(self, model_node_id: str) -> Optional[GroundTruthRegime]:
        """Get the ground truth regime for a model."""
        return self._regimes.get(model_node_id)
    
    def get_disclosure(self, model_node_id: str) -> Dict[str, Any]:
        """Get disclosure information for a model."""
        regime = self._regimes.get(model_node_id)
        if regime is None:
            return {
                "ground_truth_mode": "unknown",
                "warnings": ["No ground truth regime registered for this model."],
            }
        return regime.to_disclosure()
    
    def can_run_residual_analysis(self, model_node_id: str) -> Tuple[bool, str]:
        """Check if residual analysis can be run."""
        regime = self._regimes.get(model_node_id)
        if regime is None:
            return False, "No ground truth regime registered"
        
        if regime.residual_audit_status == ResidualAuditStatus.INACTIVE:
            return False, "No ground truth available"
        
        if regime.residual_audit_status == ResidualAuditStatus.PARTIAL:
            if regime.overall_coverage_pct < 10:
                return False, f"Coverage too low ({regime.overall_coverage_pct:.0f}%)"
            return True, f"Partial coverage ({regime.overall_coverage_pct:.0f}%)"
        
        return True, "Ground truth available"


# =============================================================================
# GROUND TRUTH AWARE VALIDITY
# =============================================================================

@dataclass
class GroundTruthAwareValidity:
    """
    Validity assessment that includes ground truth disclosure.
    
    This is the final output that includes:
        - Standard validity score/state
        - Ground truth regime information
        - Disclosure requirements
        - Warnings about audit limitations
    """
    # Standard validity
    validity_score: float
    validity_state: str
    
    # Ground truth disclosure
    ground_truth_mode: str
    residual_audit_status: str
    coverage_pct: float
    
    # Audit limitations
    residual_analysis_possible: bool
    residual_analysis_limitation: Optional[str]
    
    # Warnings (MUST be shown to users)
    disclosure_warnings: List[str]
    
    # Flag for whether this assessment is "complete"
    is_complete_assessment: bool
    incompleteness_reason: Optional[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "validity": {
                "score": self.validity_score,
                "state": self.validity_state,
            },
            "ground_truth": {
                "mode": self.ground_truth_mode,
                "residual_audit_status": self.residual_audit_status,
                "coverage_pct": self.coverage_pct,
                "residual_analysis_possible": self.residual_analysis_possible,
                "residual_analysis_limitation": self.residual_analysis_limitation,
            },
            "disclosure": {
                "warnings": self.disclosure_warnings,
                "is_complete_assessment": self.is_complete_assessment,
                "incompleteness_reason": self.incompleteness_reason,
            },
        }


def create_ground_truth_aware_validity(
    validity_score: float,
    validity_state: str,
    gt_manager: GroundTruthManager,
    model_node_id: str,
) -> GroundTruthAwareValidity:
    """
    Create a validity assessment with ground truth disclosure.
    """
    regime = gt_manager.get_regime(model_node_id)
    disclosure = gt_manager.get_disclosure(model_node_id)
    
    can_run_residual, limitation = gt_manager.can_run_residual_analysis(model_node_id)
    
    # Determine if assessment is "complete"
    is_complete = True
    incompleteness_reason = None
    
    if regime:
        mode = regime.get_effective_mode()
        if mode == GroundTruthMode.NONE:
            is_complete = False
            incompleteness_reason = (
                "No ground truth available. Assessment based on input/output analysis only. "
                "Model accuracy cannot be verified."
            )
        elif mode == GroundTruthMode.PARTIAL and regime.overall_coverage_pct < 50:
            is_complete = False
            incompleteness_reason = (
                f"Low ground truth coverage ({regime.overall_coverage_pct:.0f}%). "
                "Assessment may not be representative."
            )
    else:
        is_complete = False
        incompleteness_reason = "No ground truth regime registered."
    
    return GroundTruthAwareValidity(
        validity_score=validity_score,
        validity_state=validity_state,
        ground_truth_mode=disclosure.get("ground_truth_mode", "unknown"),
        residual_audit_status=disclosure.get("residual_audit_status", "unknown"),
        coverage_pct=disclosure.get("coverage_pct", 0.0),
        residual_analysis_possible=can_run_residual,
        residual_analysis_limitation=limitation if not can_run_residual else None,
        disclosure_warnings=disclosure.get("warnings", []),
        is_complete_assessment=is_complete,
        incompleteness_reason=incompleteness_reason,
    )


# =============================================================================
# GOVERNOR ENFORCEMENT OF DISCLOSURE
# =============================================================================

class GroundTruthGovernor:
    """
    Governor that enforces ground truth disclosure requirements.
    
    Rules:
        1. All model outputs MUST include ground truth disclosure
        2. If NO ground truth, output MUST include prominent warning
        3. If PARTIAL coverage < 30%, exports restricted
        4. Residual-based metrics suppressed when inactive
    """
    
    def __init__(self, gt_manager: GroundTruthManager):
        self.gt_manager = gt_manager
    
    def can_emit_without_disclosure(self, model_node_id: str) -> Tuple[bool, str]:
        """Check if we can emit without disclosure (answer: NO)."""
        return False, "Ground truth disclosure is always required for model outputs"
    
    def can_export(self, model_node_id: str) -> Tuple[bool, str]:
        """Check if full export is permitted."""
        regime = self.gt_manager.get_regime(model_node_id)
        
        if regime is None:
            return False, "Cannot export: no ground truth regime registered"
        
        mode = regime.get_effective_mode()
        
        if mode == GroundTruthMode.NONE:
            return False, (
                "Export restricted: No ground truth available. "
                "Model accuracy cannot be verified."
            )
        
        if mode == GroundTruthMode.PARTIAL and regime.overall_coverage_pct < 30:
            return False, (
                f"Export restricted: Ground truth coverage too low "
                f"({regime.overall_coverage_pct:.0f}% < 30%)"
            )
        
        return True, ""
    
    def can_emit_residual_metrics(self, model_node_id: str) -> Tuple[bool, str]:
        """Check if residual-based metrics can be emitted."""
        can_run, reason = self.gt_manager.can_run_residual_analysis(model_node_id)
        
        if not can_run:
            return False, f"Residual metrics suppressed: {reason}"
        
        return True, ""
    
    def build_disclosure_envelope(
        self,
        model_node_id: str,
        output: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Wrap output in disclosure envelope.
        
        This is how we ensure disclosure is NEVER forgotten.
        """
        disclosure = self.gt_manager.get_disclosure(model_node_id)
        
        return {
            "output": output,
            "ground_truth_disclosure": disclosure,
            "_disclosure_version": "1.0.0",
            "_disclosure_required": True,
        }


# =============================================================================
# TESTS
# =============================================================================

def test_no_ground_truth_disclosure():
    """Test that NO ground truth mode generates proper warnings."""
    manager = GroundTruthManager()
    
    # Register model with no ground truth
    regime = manager.register_model(
        "credit_model",
        outcome_streams=[
            OutcomeStream(
                stream_id="none",
                model_node_id="credit_model",
                mode=GroundTruthMode.NONE,
            )
        ]
    )
    
    assert regime.residual_audit_status == ResidualAuditStatus.INACTIVE
    
    disclosure = manager.get_disclosure("credit_model")
    assert disclosure["ground_truth_mode"] == "none"
    assert any("NO GROUND TRUTH" in w for w in disclosure["warnings"])
    
    can_run, reason = manager.can_run_residual_analysis("credit_model")
    assert not can_run
    
    print(f"Mode: {disclosure['ground_truth_mode']}")
    print(f"Warnings: {disclosure['warnings']}")
    print("✓ test_no_ground_truth_disclosure passed")


def test_delayed_ground_truth():
    """Test delayed ground truth mode."""
    manager = GroundTruthManager()
    
    # Register model with delayed outcomes (e.g., loan defaults)
    regime = manager.register_model(
        "loan_model",
        outcome_streams=[
            OutcomeStream(
                stream_id="defaults",
                model_node_id="loan_model",
                mode=GroundTruthMode.DELAYED,
                delay_days=90,
                delay_description="Default status known after 90 days",
            )
        ]
    )
    
    assert regime.residual_audit_status == ResidualAuditStatus.BACKDATED
    
    disclosure = manager.get_disclosure("loan_model")
    assert disclosure["ground_truth_mode"] == "delayed"
    assert any("DELAYED" in w for w in disclosure["warnings"])
    
    can_run, reason = manager.can_run_residual_analysis("loan_model")
    assert can_run
    
    print(f"Mode: {disclosure['ground_truth_mode']}")
    print(f"Status: {disclosure['residual_audit_status']}")
    print("✓ test_delayed_ground_truth passed")


def test_partial_coverage():
    """Test partial ground truth with coverage tracking."""
    manager = GroundTruthManager()
    
    # Register model with partial outcomes
    regime = manager.register_model(
        "approval_model",
        outcome_streams=[
            OutcomeStream(
                stream_id="outcomes",
                model_node_id="approval_model",
                mode=GroundTruthMode.PARTIAL,
                coverage_pct=60.0,
                coverage_description="Only approved applications have outcome",
                has_outcome_filter="approved_only",
            )
        ]
    )
    
    # Simulate predictions
    from datetime import datetime
    manager.record_prediction("approval_model", "p1", datetime.utcnow())
    manager.record_prediction("approval_model", "p2", datetime.utcnow())
    manager.record_prediction("approval_model", "p3", datetime.utcnow())
    
    # Record outcomes for some
    manager.record_outcome("approval_model", "p1", datetime.utcnow(), "good")
    manager.record_outcome("approval_model", "p2", datetime.utcnow(), "bad")
    
    # Check coverage
    regime = manager.get_regime("approval_model")
    assert abs(regime.overall_coverage_pct - 66.67) < 1  # 2/3
    
    disclosure = manager.get_disclosure("approval_model")
    assert any("PARTIAL COVERAGE" in w for w in disclosure["warnings"])
    
    print(f"Coverage: {regime.overall_coverage_pct:.0f}%")
    print("✓ test_partial_coverage passed")


def test_governor_export_restrictions():
    """Test that governor restricts exports based on ground truth."""
    manager = GroundTruthManager()
    governor = GroundTruthGovernor(manager)
    
    # No ground truth - should block exports
    manager.register_model(
        "black_box",
        outcome_streams=[
            OutcomeStream(
                stream_id="none",
                model_node_id="black_box",
                mode=GroundTruthMode.NONE,
            )
        ]
    )
    
    can_export, reason = governor.can_export("black_box")
    assert not can_export
    assert "cannot be verified" in reason.lower()
    
    # Full ground truth - should allow exports
    manager.register_model(
        "auditable",
        outcome_streams=[
            OutcomeStream(
                stream_id="full",
                model_node_id="auditable",
                mode=GroundTruthMode.FULL,
            )
        ]
    )
    
    can_export, reason = governor.can_export("auditable")
    assert can_export
    
    print("✓ test_governor_export_restrictions passed")


def test_disclosure_envelope():
    """Test that outputs are wrapped in disclosure envelope."""
    manager = GroundTruthManager()
    governor = GroundTruthGovernor(manager)
    
    manager.register_model(
        "model1",
        outcome_streams=[
            OutcomeStream(
                stream_id="outcomes",
                model_node_id="model1",
                mode=GroundTruthMode.PARTIAL,
                coverage_pct=45.0,
            )
        ]
    )
    
    output = {"prediction": 0.85, "confidence": 0.9}
    envelope = governor.build_disclosure_envelope("model1", output)
    
    assert "output" in envelope
    assert "ground_truth_disclosure" in envelope
    assert envelope["_disclosure_required"] == True
    
    print(f"Envelope keys: {list(envelope.keys())}")
    print("✓ test_disclosure_envelope passed")


def run_all_ground_truth_tests():
    print("\n" + "=" * 60)
    print("GROUND TRUTH REGIME TESTS")
    print("=" * 60 + "\n")
    
    test_no_ground_truth_disclosure()
    print()
    test_delayed_ground_truth()
    print()
    test_partial_coverage()
    print()
    test_governor_export_restrictions()
    print()
    test_disclosure_envelope()
    
    print("\n" + "=" * 60)
    print("ALL GROUND TRUTH TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    run_all_ground_truth_tests()
