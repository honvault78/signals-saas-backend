"""
Bavella v2 — Epistemic Cost & Irreversibility
===============================================

THE MISSING CONCEPT: Not all validity breaks are equal.

Current model (wrong):
    Validity degrades → recovers → OK
    
Reality:
    - Some breaks are cheap (temporary volatility spike)
    - Some are catastrophic (structural break in target definition)
    - Some permanently invalidate historical comparability

This module implements:
    1. EpistemicCost per failure mode event
    2. Reversibility classification
    3. Historical contamination tracking
    4. Path-dependent recovery
    5. Trust recovery penalty

Now two systems with the same score may NOT be equally trusted.

Example output:
    VALID (81%)
    ⚠️ Historical comparability permanently compromised (FM4 @ 2024-10-15)

This is extremely powerful — and very hard to fake.

Copyright 2024-2026 Bavella Technologies Sarl
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Tuple, Set
import numpy as np

from .core import FailureMode, ValidityState


# =============================================================================
# REVERSIBILITY CLASSIFICATION
# =============================================================================

class Reversibility(Enum):
    """
    How reversible is the epistemic damage from this failure mode?
    """
    
    REVERSIBLE = "reversible"
    # Full recovery possible
    # Example: Temporary volatility spike
    # Trust returns to baseline after recovery
    
    PARTIALLY_REVERSIBLE = "partially_reversible"
    # Partial recovery possible, but some damage persists
    # Example: Distribution shift that was corrected
    # Trust recovers but with penalty
    
    IRREVERSIBLE = "irreversible"
    # Permanent damage to comparability/trust
    # Example: Structural break in target definition
    # Historical data before break cannot be compared
    
    CONTAMINATING = "contaminating"
    # Not just irreversible, but actively contaminates other nodes
    # Example: Model trained on invalid regime
    # All downstream nodes are contaminated


class HistoricalImpact(Enum):
    """Impact on historical data interpretation."""
    
    NONE = "none"
    # No impact on historical interpretation
    
    WINDOW_LIMITED = "window_limited"
    # Only recent window affected
    
    COMPARISON_IMPAIRED = "comparison_impaired"
    # Cannot compare pre/post failure
    
    FULLY_COMPROMISED = "fully_compromised"
    # All historical analysis compromised


# =============================================================================
# EPISTEMIC COST (per failure mode)
# =============================================================================

@dataclass(frozen=True)
class EpistemicCost:
    """
    The epistemic cost of a failure mode event.
    
    This captures what we LOSE when this failure occurs,
    not just how severe it is.
    """
    # Classification
    reversibility: Reversibility
    historical_impact: HistoricalImpact
    
    # Numeric costs (0-100)
    recovery_penalty: float  # How much harder is future recovery?
    trust_damage: float      # How much trust is permanently lost?
    comparison_cost: float   # How much historical comparison is lost?
    
    # Flags
    contaminates_downstream: bool = False
    invalidates_training_data: bool = False
    requires_rebaseline: bool = False
    
    # Description
    damage_description: str = ""
    recovery_requirements: Tuple[str, ...] = ()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "reversibility": self.reversibility.value,
            "historical_impact": self.historical_impact.value,
            "recovery_penalty": self.recovery_penalty,
            "trust_damage": self.trust_damage,
            "comparison_cost": self.comparison_cost,
            "contaminates_downstream": self.contaminates_downstream,
            "invalidates_training_data": self.invalidates_training_data,
            "requires_rebaseline": self.requires_rebaseline,
            "damage_description": self.damage_description,
            "recovery_requirements": list(self.recovery_requirements),
        }


# =============================================================================
# EPISTEMIC COST TABLE (FROZEN)
# =============================================================================

class EpistemicCostTable:
    """
    Defines the epistemic cost for each failure mode.
    
    THIS IS FROZEN CANON. Changes require version bump.
    """
    
    VERSION = "1.0.0"
    
    _COSTS: Dict[FailureMode, EpistemicCost] = {
        
        FailureMode.FM1_VARIANCE_REGIME: EpistemicCost(
            reversibility=Reversibility.REVERSIBLE,
            historical_impact=HistoricalImpact.WINDOW_LIMITED,
            recovery_penalty=10.0,
            trust_damage=5.0,
            comparison_cost=15.0,
            damage_description="Variance regime change - typically recoverable",
            recovery_requirements=("Return to stable variance for 5+ periods",),
        ),
        
        FailureMode.FM2_MEAN_DRIFT: EpistemicCost(
            reversibility=Reversibility.PARTIALLY_REVERSIBLE,
            historical_impact=HistoricalImpact.COMPARISON_IMPAIRED,
            recovery_penalty=20.0,
            trust_damage=15.0,
            comparison_cost=30.0,
            damage_description="Mean drift - pre/post comparison impaired",
            recovery_requirements=(
                "Mean stabilization for 10+ periods",
                "Rebaseline if permanent shift",
            ),
        ),
        
        FailureMode.FM3_SEASONALITY_MISMATCH: EpistemicCost(
            reversibility=Reversibility.REVERSIBLE,
            historical_impact=HistoricalImpact.WINDOW_LIMITED,
            recovery_penalty=15.0,
            trust_damage=10.0,
            comparison_cost=20.0,
            damage_description="Seasonality mismatch - seasonal patterns recoverable",
            recovery_requirements=("Full seasonal cycle observation",),
        ),
        
        FailureMode.FM4_STRUCTURAL_BREAK: EpistemicCost(
            reversibility=Reversibility.IRREVERSIBLE,
            historical_impact=HistoricalImpact.FULLY_COMPROMISED,
            recovery_penalty=60.0,
            trust_damage=50.0,
            comparison_cost=80.0,
            contaminates_downstream=True,
            requires_rebaseline=True,
            damage_description="Structural break - historical comparability permanently compromised",
            recovery_requirements=(
                "Full system rebaseline required",
                "New validity baseline from break point",
                "Historical pre-break data incomparable",
            ),
        ),
        
        FailureMode.FM5_OUTLIER_CONTAMINATION: EpistemicCost(
            reversibility=Reversibility.PARTIALLY_REVERSIBLE,
            historical_impact=HistoricalImpact.WINDOW_LIMITED,
            recovery_penalty=25.0,
            trust_damage=20.0,
            comparison_cost=25.0,
            damage_description="Outlier contamination - estimates may be biased",
            recovery_requirements=(
                "Outlier remediation or filtering",
                "Recomputation of affected statistics",
            ),
        ),
        
        FailureMode.FM6_DISTRIBUTIONAL_SHIFT: EpistemicCost(
            reversibility=Reversibility.PARTIALLY_REVERSIBLE,
            historical_impact=HistoricalImpact.COMPARISON_IMPAIRED,
            recovery_penalty=35.0,
            trust_damage=30.0,
            comparison_cost=45.0,
            damage_description="Distributional shift - probabilistic guarantees compromised",
            recovery_requirements=(
                "Distribution stabilization",
                "Quantile recalibration if persistent",
            ),
        ),
        
        FailureMode.FM7_DEPENDENCY_BREAK: EpistemicCost(
            reversibility=Reversibility.IRREVERSIBLE,
            historical_impact=HistoricalImpact.FULLY_COMPROMISED,
            recovery_penalty=70.0,
            trust_damage=65.0,
            comparison_cost=90.0,
            contaminates_downstream=True,
            invalidates_training_data=True,
            requires_rebaseline=True,
            damage_description="Dependency break - relationship assumptions permanently invalid",
            recovery_requirements=(
                "Complete relationship revalidation",
                "Model retraining if ML-dependent",
                "Historical pair analysis invalid",
            ),
        ),
    }
    
    @classmethod
    def get_cost(cls, fm: FailureMode) -> EpistemicCost:
        """Get the epistemic cost for a failure mode."""
        return cls._COSTS.get(fm, EpistemicCost(
            reversibility=Reversibility.PARTIALLY_REVERSIBLE,
            historical_impact=HistoricalImpact.WINDOW_LIMITED,
            recovery_penalty=20.0,
            trust_damage=15.0,
            comparison_cost=25.0,
            damage_description="Unknown failure mode",
        ))
    
    @classmethod
    def is_irreversible(cls, fm: FailureMode) -> bool:
        """Check if failure mode causes irreversible damage."""
        cost = cls.get_cost(fm)
        return cost.reversibility in (
            Reversibility.IRREVERSIBLE,
            Reversibility.CONTAMINATING,
        )


# =============================================================================
# DAMAGE RECORD (tracks accumulated damage)
# =============================================================================

@dataclass
class DamageEvent:
    """A single damage event from a failure mode."""
    event_id: str
    failure_mode: FailureMode
    occurred_at: datetime
    severity: float
    cost: EpistemicCost
    
    # Context
    node_id: str
    episode_id: Optional[str] = None
    
    # Was this recovered from?
    recovered: bool = False
    recovered_at: Optional[datetime] = None
    
    # Residual damage after recovery (0-100)
    residual_damage: float = 0.0


@dataclass
class DamageRecord:
    """
    Accumulated damage record for a node.
    
    This is what makes trust path-dependent:
        - Two nodes with same current score
        - May have different damage history
        - Therefore different trust levels
    """
    node_id: str
    owner_id: str
    
    # Events
    events: List[DamageEvent] = field(default_factory=list)
    
    # Accumulated damage (never fully recovers for irreversible)
    accumulated_trust_damage: float = 0.0
    accumulated_comparison_cost: float = 0.0
    
    # Flags
    has_irreversible_damage: bool = False
    requires_rebaseline: bool = False
    training_data_compromised: bool = False
    
    # Contamination tracking
    contaminated_by: Optional[str] = None  # node_id that contaminated this
    contamination_date: Optional[datetime] = None
    
    # Historical comparability markers
    comparability_broken_at: Optional[datetime] = None
    pre_break_data_usable: bool = True
    
    def record_event(self, event: DamageEvent) -> None:
        """Record a damage event."""
        self.events.append(event)
        
        cost = event.cost
        
        # Accumulate damage
        if cost.reversibility == Reversibility.IRREVERSIBLE:
            self.accumulated_trust_damage += cost.trust_damage
            self.accumulated_comparison_cost += cost.comparison_cost
            self.has_irreversible_damage = True
            
            if self.comparability_broken_at is None:
                self.comparability_broken_at = event.occurred_at
                self.pre_break_data_usable = True  # Pre-break is usable, post is not comparable
        
        elif cost.reversibility == Reversibility.CONTAMINATING:
            self.accumulated_trust_damage += cost.trust_damage
            self.accumulated_comparison_cost = 100.0  # Fully compromised
            self.has_irreversible_damage = True
            self.training_data_compromised = cost.invalidates_training_data
        
        elif cost.reversibility == Reversibility.PARTIALLY_REVERSIBLE:
            # Partial damage accumulates but can be partially recovered
            self.accumulated_trust_damage += cost.trust_damage * 0.3
            self.accumulated_comparison_cost += cost.comparison_cost * 0.5
        
        if cost.requires_rebaseline:
            self.requires_rebaseline = True
    
    def record_recovery(self, event_id: str, residual_damage: float = 0.0) -> None:
        """Record that an event was recovered from."""
        for event in self.events:
            if event.event_id == event_id:
                event.recovered = True
                event.recovered_at = datetime.now(timezone.utc)
                event.residual_damage = residual_damage
                break
    
    def get_effective_trust_penalty(self) -> float:
        """
        Get the effective trust penalty.
        
        This is applied AFTER validity recovery.
        Even if current validity is VALID, trust may be impaired.
        """
        base_penalty = min(50, self.accumulated_trust_damage)
        
        # Add penalty for unrecovered events
        for event in self.events:
            if not event.recovered:
                base_penalty += event.cost.recovery_penalty * 0.1
        
        return min(80, base_penalty)
    
    def get_warnings(self) -> List[str]:
        """Get warnings based on damage history."""
        warnings = []
        
        if self.has_irreversible_damage and self.comparability_broken_at:
            warnings.append(
                f"⚠️ Historical comparability permanently compromised "
                f"(since {self.comparability_broken_at.strftime('%Y-%m-%d')})"
            )
        
        if self.requires_rebaseline:
            warnings.append("⚠️ System rebaseline required")
        
        if self.training_data_compromised:
            warnings.append("⚠️ Training data validity compromised - model retraining may be required")
        
        if self.contaminated_by:
            warnings.append(
                f"⚠️ Contaminated by upstream node {self.contaminated_by} "
                f"(since {self.contamination_date.strftime('%Y-%m-%d') if self.contamination_date else 'unknown'})"
            )
        
        return warnings
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "accumulated_trust_damage": self.accumulated_trust_damage,
            "accumulated_comparison_cost": self.accumulated_comparison_cost,
            "has_irreversible_damage": self.has_irreversible_damage,
            "requires_rebaseline": self.requires_rebaseline,
            "training_data_compromised": self.training_data_compromised,
            "comparability_broken_at": self.comparability_broken_at.isoformat() if self.comparability_broken_at else None,
            "effective_trust_penalty": self.get_effective_trust_penalty(),
            "warnings": self.get_warnings(),
            "event_count": len(self.events),
        }


# =============================================================================
# TRUST-ADJUSTED VALIDITY
# =============================================================================

@dataclass(frozen=True)
class TrustAdjustedValidity:
    """
    Validity with trust adjustment based on damage history.
    
    This is why two systems with the same score may not be equally trusted.
    """
    # Raw validity
    raw_score: float
    raw_state: str
    
    # Trust adjustment
    trust_penalty: float  # 0-100
    adjusted_score: float  # raw_score - trust_penalty
    adjusted_state: str
    
    # Damage context
    has_irreversible_damage: bool
    comparability_broken_at: Optional[datetime]
    
    # Warnings
    warnings: Tuple[str, ...] = ()
    
    # Recovery path
    can_fully_recover: bool = True
    recovery_requirements: Tuple[str, ...] = ()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "validity": {
                "raw_score": self.raw_score,
                "raw_state": self.raw_state,
                "trust_adjusted_score": self.adjusted_score,
                "trust_adjusted_state": self.adjusted_state,
                "trust_penalty": self.trust_penalty,
            },
            "damage": {
                "has_irreversible_damage": self.has_irreversible_damage,
                "comparability_broken_at": self.comparability_broken_at.isoformat() if self.comparability_broken_at else None,
                "can_fully_recover": self.can_fully_recover,
            },
            "warnings": list(self.warnings),
            "recovery_requirements": list(self.recovery_requirements),
        }


def compute_trust_adjusted_validity(
    raw_score: float,
    raw_state: str,
    damage_record: DamageRecord,
) -> TrustAdjustedValidity:
    """
    Compute trust-adjusted validity.
    
    Even if raw validity is VALID, trust penalty may downgrade it.
    """
    from .core import ValidityState, Thresholds
    
    trust_penalty = damage_record.get_effective_trust_penalty()
    adjusted_score = max(0, raw_score - trust_penalty)
    
    # Determine adjusted state
    if adjusted_score >= Thresholds.VALID_MIN:
        adjusted_state = "VALID"
    elif adjusted_score >= Thresholds.DEGRADED_MIN:
        adjusted_state = "DEGRADED"
    else:
        adjusted_state = "INVALID"
    
    # Can we fully recover?
    can_fully_recover = not damage_record.has_irreversible_damage
    
    # Collect recovery requirements
    recovery_reqs = []
    for event in damage_record.events:
        if not event.recovered:
            recovery_reqs.extend(event.cost.recovery_requirements)
    
    return TrustAdjustedValidity(
        raw_score=raw_score,
        raw_state=raw_state,
        trust_penalty=trust_penalty,
        adjusted_score=adjusted_score,
        adjusted_state=adjusted_state,
        has_irreversible_damage=damage_record.has_irreversible_damage,
        comparability_broken_at=damage_record.comparability_broken_at,
        warnings=tuple(damage_record.get_warnings()),
        can_fully_recover=can_fully_recover,
        recovery_requirements=tuple(set(recovery_reqs)),
    )


# =============================================================================
# PROBATIONARY STATUS
# =============================================================================

class ProbationaryStatus(Enum):
    """Status after recovery."""
    NONE = "none"
    PROBATION = "probation"
    RESTRICTED = "restricted"


@dataclass
class RecoveryStatus:
    """
    Status of a node that has recovered from INVALID/DEGRADED.
    
    Prevents "bouncing" - quick recovery followed by re-degradation.
    """
    node_id: str
    
    # Recovery timeline
    recovered_at: datetime
    probation_ends_at: datetime
    
    # Status
    status: ProbationaryStatus
    
    # During probation
    outputs_stamped: bool = True  # All outputs marked with probation stamp
    exports_restricted: bool = False
    
    @property
    def is_on_probation(self) -> bool:
        return (
            self.status == ProbationaryStatus.PROBATION and
            datetime.now(timezone.utc) < self.probation_ends_at
        )
    
    @property
    def days_remaining(self) -> float:
        if not self.is_on_probation:
            return 0
        return (self.probation_ends_at - datetime.now(timezone.utc)).days
    
    def get_stamp(self) -> Optional[str]:
        if self.is_on_probation:
            return f"PROBATION ({self.days_remaining:.0f} days remaining)"
        return None


class ProbationaryTracker:
    """
    Tracks probationary status for recovered nodes.
    
    Rules:
        - After recovery to VALID: 7 day probation
        - During probation: outputs stamped
        - If re-degradation during probation: extended probation
    """
    
    DEFAULT_PROBATION_DAYS = 7
    EXTENDED_PROBATION_DAYS = 14
    
    def __init__(self):
        self._statuses: Dict[str, RecoveryStatus] = {}
    
    def record_recovery(
        self,
        node_id: str,
        from_state: str,
    ) -> RecoveryStatus:
        """Record that a node recovered."""
        now = datetime.now(timezone.utc)
        
        # Check if already on probation (re-recovery)
        existing = self._statuses.get(node_id)
        if existing and existing.is_on_probation:
            # Extend probation
            probation_days = self.EXTENDED_PROBATION_DAYS
        else:
            probation_days = self.DEFAULT_PROBATION_DAYS
        
        status = RecoveryStatus(
            node_id=node_id,
            recovered_at=now,
            probation_ends_at=now + timedelta(days=probation_days),
            status=ProbationaryStatus.PROBATION,
            outputs_stamped=True,
            exports_restricted=(from_state == "INVALID"),
        )
        
        self._statuses[node_id] = status
        return status
    
    def get_status(self, node_id: str) -> Optional[RecoveryStatus]:
        """Get probationary status for a node."""
        status = self._statuses.get(node_id)
        if status and not status.is_on_probation:
            # Probation ended
            status.status = ProbationaryStatus.NONE
        return status
    
    def is_on_probation(self, node_id: str) -> bool:
        """Check if node is on probation."""
        status = self.get_status(node_id)
        return status.is_on_probation if status else False


# =============================================================================
# TESTS
# =============================================================================

def test_epistemic_cost_classification():
    """Test that epistemic costs are correctly classified."""
    # FM4 should be irreversible
    fm4_cost = EpistemicCostTable.get_cost(FailureMode.FM4_STRUCTURAL_BREAK)
    assert fm4_cost.reversibility == Reversibility.IRREVERSIBLE
    assert fm4_cost.requires_rebaseline
    assert fm4_cost.contaminates_downstream
    
    # FM1 should be reversible
    fm1_cost = EpistemicCostTable.get_cost(FailureMode.FM1_VARIANCE_REGIME)
    assert fm1_cost.reversibility == Reversibility.REVERSIBLE
    
    print(f"FM4: {fm4_cost.reversibility.value}, penalty={fm4_cost.recovery_penalty}")
    print(f"FM1: {fm1_cost.reversibility.value}, penalty={fm1_cost.recovery_penalty}")
    print("✓ test_epistemic_cost_classification passed")


def test_damage_accumulation():
    """Test that damage accumulates correctly."""
    record = DamageRecord(node_id="test", owner_id="owner")
    
    # Record a reversible event
    record.record_event(DamageEvent(
        event_id="e1",
        failure_mode=FailureMode.FM1_VARIANCE_REGIME,
        occurred_at=datetime.now(timezone.utc),
        severity=50,
        cost=EpistemicCostTable.get_cost(FailureMode.FM1_VARIANCE_REGIME),
        node_id="test",
    ))
    
    assert not record.has_irreversible_damage
    
    # Record an irreversible event
    record.record_event(DamageEvent(
        event_id="e2",
        failure_mode=FailureMode.FM4_STRUCTURAL_BREAK,
        occurred_at=datetime.now(timezone.utc),
        severity=80,
        cost=EpistemicCostTable.get_cost(FailureMode.FM4_STRUCTURAL_BREAK),
        node_id="test",
    ))
    
    assert record.has_irreversible_damage
    assert record.requires_rebaseline
    assert record.comparability_broken_at is not None
    
    print(f"Accumulated trust damage: {record.accumulated_trust_damage}")
    print(f"Warnings: {record.get_warnings()}")
    print("✓ test_damage_accumulation passed")


def test_trust_adjusted_validity():
    """Test that trust penalty affects validity."""
    record = DamageRecord(node_id="test", owner_id="owner")
    
    # Record significant damage
    record.record_event(DamageEvent(
        event_id="e1",
        failure_mode=FailureMode.FM4_STRUCTURAL_BREAK,
        occurred_at=datetime.now(timezone.utc),
        severity=80,
        cost=EpistemicCostTable.get_cost(FailureMode.FM4_STRUCTURAL_BREAK),
        node_id="test",
    ))
    
    # Raw validity is VALID (75)
    adjusted = compute_trust_adjusted_validity(
        raw_score=75.0,
        raw_state="VALID",
        damage_record=record,
    )
    
    # Should be downgraded due to trust penalty
    assert adjusted.trust_penalty > 0
    assert adjusted.adjusted_score < adjusted.raw_score
    
    print(f"Raw: {adjusted.raw_score} ({adjusted.raw_state})")
    print(f"Adjusted: {adjusted.adjusted_score} ({adjusted.adjusted_state})")
    print(f"Trust penalty: {adjusted.trust_penalty}")
    print(f"Can fully recover: {adjusted.can_fully_recover}")
    
    print("✓ test_trust_adjusted_validity passed")


def test_probationary_status():
    """Test probationary status tracking."""
    tracker = ProbationaryTracker()
    
    # Record recovery
    status = tracker.record_recovery("node_1", from_state="INVALID")
    
    assert status.is_on_probation
    assert status.outputs_stamped
    assert status.exports_restricted
    
    print(f"On probation: {status.is_on_probation}")
    print(f"Stamp: {status.get_stamp()}")
    print(f"Days remaining: {status.days_remaining}")
    
    print("✓ test_probationary_status passed")


def run_all_epistemic_cost_tests():
    print("\n" + "=" * 60)
    print("EPISTEMIC COST & IRREVERSIBILITY TESTS")
    print("=" * 60 + "\n")
    
    test_epistemic_cost_classification()
    print()
    test_damage_accumulation()
    print()
    test_trust_adjusted_validity()
    print()
    test_probationary_status()
    
    print("\n" + "=" * 60)
    print("ALL EPISTEMIC COST TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    run_all_epistemic_cost_tests()
