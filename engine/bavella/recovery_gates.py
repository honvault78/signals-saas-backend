"""
Bavella v2 — Recovery Gates & Statistics
=========================================

THE "NOT MEAN-ONLY" RECOVERY INFRASTRUCTURE

Instead of: "Average recovery: 38 days"
We get:    "Median: 32d, IQR: [18, 52], 70% full recovery rate, 
            estimate confidence: 0.72 (n=8, moderate dispersion)"

This module defines:
    1. RecoveryGate - measurable, versioned recovery conditions
    2. RecoveryGateEvaluator - deterministic gate checking
    3. RecoveryStatistics - distribution-aware stats with uncertainty
    4. RecoveryEstimate - the "answer" with proper confidence

Copyright 2024-2026 Bavella Technologies Sarl
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
import uuid


from .audit_infrastructure import (
    BavellaVersions, RecoveryType, EvidenceType, EvidenceArtifact,
)


# =============================================================================
# RECOVERY GATES (versioned, measurable)
# =============================================================================

class GateOperator(Enum):
    """Comparison operators for gates."""
    GTE = ">="
    LTE = "<="
    GT = ">"
    LT = "<"
    EQ = "=="
    NEQ = "!="


class GateMetric(Enum):
    """Metrics that can be gated on."""
    VALIDITY_SCORE = "validity_score"
    SEVERITY_MAX = "severity_max"
    SEVERITY_CURRENT = "severity_current"
    CONSECUTIVE_CLEAN = "consecutive_clean"
    DAYS_BELOW_THRESHOLD = "days_below_threshold"
    TRUST_PENALTY = "trust_penalty"
    CONFIDENCE = "confidence"


@dataclass(frozen=True)
class RecoveryGateDefinition:
    """
    Definition of a recovery gate.
    
    Gates must be versioned and deterministic.
    """
    gate_id: str
    gate_name: str
    gate_version: str = BavellaVersions.RECOVERY_GATES_VERSION
    
    # What we're checking
    metric: GateMetric = GateMetric.VALIDITY_SCORE
    operator: GateOperator = GateOperator.GTE
    threshold: float = 70.0
    
    # Context
    description: str = ""
    required_for: Tuple[RecoveryType, ...] = (RecoveryType.FULL,)
    
    def evaluate(self, actual_value: float) -> bool:
        """Evaluate gate against actual value."""
        if self.operator == GateOperator.GTE:
            return actual_value >= self.threshold
        elif self.operator == GateOperator.LTE:
            return actual_value <= self.threshold
        elif self.operator == GateOperator.GT:
            return actual_value > self.threshold
        elif self.operator == GateOperator.LT:
            return actual_value < self.threshold
        elif self.operator == GateOperator.EQ:
            return actual_value == self.threshold
        elif self.operator == GateOperator.NEQ:
            return actual_value != self.threshold
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "gate_id": self.gate_id,
            "gate_name": self.gate_name,
            "gate_version": self.gate_version,
            "metric": self.metric.value,
            "operator": self.operator.value,
            "threshold": self.threshold,
            "description": self.description,
            "required_for": [rt.value for rt in self.required_for],
        }


@dataclass
class GateEvaluationResult:
    """Result of evaluating a single gate."""
    gate: RecoveryGateDefinition
    passed: bool
    actual_value: float
    evaluated_at: datetime
    evidence_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "gate_id": self.gate.gate_id,
            "gate_name": self.gate.gate_name,
            "passed": self.passed,
            "actual_value": self.actual_value,
            "threshold": self.gate.threshold,
            "operator": self.gate.operator.value,
            "evaluated_at": self.evaluated_at.isoformat(),
            "evidence_id": self.evidence_id,
        }


# =============================================================================
# STANDARD GATE DEFINITIONS (FROZEN CANON)
# =============================================================================

class StandardRecoveryGates:
    """
    Standard recovery gates. FROZEN - changes require version bump.
    """
    VERSION = "1.0.0"
    
    # Full recovery gates
    VALIDITY_RESTORED = RecoveryGateDefinition(
        gate_id="GATE_001",
        gate_name="validity_restored",
        metric=GateMetric.VALIDITY_SCORE,
        operator=GateOperator.GTE,
        threshold=70.0,
        description="Validity score must be >= 70 (VALID state)",
        required_for=(RecoveryType.FULL,),
    )
    
    SEVERITY_CLEARED = RecoveryGateDefinition(
        gate_id="GATE_002",
        gate_name="severity_cleared",
        metric=GateMetric.SEVERITY_CURRENT,
        operator=GateOperator.LTE,
        threshold=15.0,
        description="Current max severity must be <= 15",
        required_for=(RecoveryType.FULL, RecoveryType.PARTIAL),
    )
    
    CONSECUTIVE_CLEAN = RecoveryGateDefinition(
        gate_id="GATE_003",
        gate_name="consecutive_clean",
        metric=GateMetric.CONSECUTIVE_CLEAN,
        operator=GateOperator.GTE,
        threshold=3.0,
        description="Must have >= 3 consecutive clean detections",
        required_for=(RecoveryType.FULL,),
    )
    
    # Partial recovery gates
    VALIDITY_DEGRADED_OK = RecoveryGateDefinition(
        gate_id="GATE_004",
        gate_name="validity_degraded_acceptable",
        metric=GateMetric.VALIDITY_SCORE,
        operator=GateOperator.GTE,
        threshold=50.0,
        description="Validity score must be >= 50 for partial recovery",
        required_for=(RecoveryType.PARTIAL,),
    )
    
    # Trust penalty gate
    TRUST_PENALTY_CAPPED = RecoveryGateDefinition(
        gate_id="GATE_005",
        gate_name="trust_penalty_manageable",
        metric=GateMetric.TRUST_PENALTY,
        operator=GateOperator.LTE,
        threshold=30.0,
        description="Trust penalty must be <= 30 for partial recovery",
        required_for=(RecoveryType.PARTIAL,),
    )
    
    @classmethod
    def get_gates_for_recovery_type(cls, recovery_type: RecoveryType) -> List[RecoveryGateDefinition]:
        """Get all gates required for a recovery type."""
        all_gates = [
            cls.VALIDITY_RESTORED,
            cls.SEVERITY_CLEARED,
            cls.CONSECUTIVE_CLEAN,
            cls.VALIDITY_DEGRADED_OK,
            cls.TRUST_PENALTY_CAPPED,
        ]
        return [g for g in all_gates if recovery_type in g.required_for]


# =============================================================================
# GATE EVALUATOR
# =============================================================================

@dataclass
class RecoveryGateContext:
    """Context values for gate evaluation."""
    validity_score: float = 0.0
    severity_current: float = 0.0
    severity_max: float = 0.0
    consecutive_clean: int = 0
    days_below_threshold: int = 0
    trust_penalty: float = 0.0
    confidence: float = 0.0
    
    def get_metric_value(self, metric: GateMetric) -> float:
        """Get value for a metric."""
        mapping = {
            GateMetric.VALIDITY_SCORE: self.validity_score,
            GateMetric.SEVERITY_CURRENT: self.severity_current,
            GateMetric.SEVERITY_MAX: self.severity_max,
            GateMetric.CONSECUTIVE_CLEAN: float(self.consecutive_clean),
            GateMetric.DAYS_BELOW_THRESHOLD: float(self.days_below_threshold),
            GateMetric.TRUST_PENALTY: self.trust_penalty,
            GateMetric.CONFIDENCE: self.confidence,
        }
        return mapping.get(metric, 0.0)


class RecoveryGateEvaluator:
    """
    Evaluates recovery gates deterministically.
    """
    
    def __init__(self):
        self.gates_version = StandardRecoveryGates.VERSION
    
    def evaluate_all(
        self,
        context: RecoveryGateContext,
        gates: Optional[List[RecoveryGateDefinition]] = None,
    ) -> Tuple[List[GateEvaluationResult], List[GateEvaluationResult]]:
        """
        Evaluate all gates.
        
        Returns: (passed_gates, failed_gates)
        """
        if gates is None:
            gates = [
                StandardRecoveryGates.VALIDITY_RESTORED,
                StandardRecoveryGates.SEVERITY_CLEARED,
                StandardRecoveryGates.CONSECUTIVE_CLEAN,
            ]
        
        now = datetime.now(timezone.utc)
        passed = []
        failed = []
        
        for gate in gates:
            actual = context.get_metric_value(gate.metric)
            result = GateEvaluationResult(
                gate=gate,
                passed=gate.evaluate(actual),
                actual_value=actual,
                evaluated_at=now,
            )
            if result.passed:
                passed.append(result)
            else:
                failed.append(result)
        
        return passed, failed
    
    def determine_recovery_type(
        self,
        context: RecoveryGateContext,
        has_irreversible_damage: bool = False,
    ) -> Tuple[RecoveryType, List[GateEvaluationResult], List[GateEvaluationResult]]:
        """
        Determine recovery type based on gate evaluation.
        """
        # If irreversible, can't have FULL recovery
        if has_irreversible_damage:
            # Check for partial
            partial_gates = StandardRecoveryGates.get_gates_for_recovery_type(RecoveryType.PARTIAL)
            passed, failed = self.evaluate_all(context, partial_gates)
            
            if not failed:
                return RecoveryType.PARTIAL, passed, failed
            else:
                return RecoveryType.REBASELINE_REQUIRED, passed, failed
        
        # Try full recovery
        full_gates = StandardRecoveryGates.get_gates_for_recovery_type(RecoveryType.FULL)
        passed, failed = self.evaluate_all(context, full_gates)
        
        if not failed:
            return RecoveryType.FULL, passed, failed
        
        # Try partial
        partial_gates = StandardRecoveryGates.get_gates_for_recovery_type(RecoveryType.PARTIAL)
        passed_partial, failed_partial = self.evaluate_all(context, partial_gates)
        
        if not failed_partial:
            return RecoveryType.PARTIAL, passed_partial, failed_partial
        
        return RecoveryType.NO_RECOVERY, passed, failed


# =============================================================================
# RECOVERY STATISTICS (distribution-aware)
# =============================================================================

@dataclass
class RecoveryDistribution:
    """
    Distribution of recovery outcomes.
    
    This is NOT just a mean - it's a proper distribution.
    """
    # Sample info
    n_episodes: int = 0
    
    # Time distribution (days)
    recovery_time_median: Optional[float] = None
    recovery_time_p25: Optional[float] = None
    recovery_time_p75: Optional[float] = None
    recovery_time_p10: Optional[float] = None
    recovery_time_p90: Optional[float] = None
    recovery_time_min: Optional[float] = None
    recovery_time_max: Optional[float] = None
    recovery_time_std: Optional[float] = None
    
    # Outcome distribution
    pct_full_recovery: float = 0.0
    pct_partial_recovery: float = 0.0
    pct_rebaseline: float = 0.0
    pct_no_recovery: float = 0.0
    
    # IQR (interquartile range)
    @property
    def iqr(self) -> Optional[float]:
        if self.recovery_time_p25 is not None and self.recovery_time_p75 is not None:
            return self.recovery_time_p75 - self.recovery_time_p25
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_episodes": self.n_episodes,
            "recovery_time": {
                "median": self.recovery_time_median,
                "p25": self.recovery_time_p25,
                "p75": self.recovery_time_p75,
                "p10": self.recovery_time_p10,
                "p90": self.recovery_time_p90,
                "min": self.recovery_time_min,
                "max": self.recovery_time_max,
                "std": self.recovery_time_std,
                "iqr": self.iqr,
            },
            "outcome_distribution": {
                "full": self.pct_full_recovery,
                "partial": self.pct_partial_recovery,
                "rebaseline": self.pct_rebaseline,
                "no_recovery": self.pct_no_recovery,
            },
        }


def compute_recovery_distribution(
    recovery_times: List[float],
    recovery_types: List[RecoveryType],
) -> RecoveryDistribution:
    """
    Compute recovery distribution from historical data.
    """
    n = len(recovery_times)
    
    if n == 0:
        return RecoveryDistribution(n_episodes=0)
    
    times = np.array(recovery_times)
    
    # Outcome distribution
    type_counts = {rt: 0 for rt in RecoveryType}
    for rt in recovery_types:
        type_counts[rt] = type_counts.get(rt, 0) + 1
    
    return RecoveryDistribution(
        n_episodes=n,
        recovery_time_median=float(np.median(times)),
        recovery_time_p25=float(np.percentile(times, 25)),
        recovery_time_p75=float(np.percentile(times, 75)),
        recovery_time_p10=float(np.percentile(times, 10)),
        recovery_time_p90=float(np.percentile(times, 90)),
        recovery_time_min=float(np.min(times)),
        recovery_time_max=float(np.max(times)),
        recovery_time_std=float(np.std(times)) if n > 1 else 0,
        pct_full_recovery=type_counts.get(RecoveryType.FULL, 0) / n * 100,
        pct_partial_recovery=type_counts.get(RecoveryType.PARTIAL, 0) / n * 100,
        pct_rebaseline=type_counts.get(RecoveryType.REBASELINE_REQUIRED, 0) / n * 100,
        pct_no_recovery=type_counts.get(RecoveryType.NO_RECOVERY, 0) / n * 100,
    )


# =============================================================================
# CONFIDENCE COMPUTATION (deterministic, not arbitrary)
# =============================================================================

@dataclass
class EstimateConfidence:
    """
    Confidence in a recovery estimate.
    
    Computed from explicit factors, not arbitrary defaults.
    """
    overall: float = 0.0  # 0-1
    
    # Contributing factors
    sample_factor: float = 0.0      # Based on n
    dispersion_factor: float = 0.0   # Based on IQR/std
    match_quality_factor: float = 0.0  # Based on similarity scores
    integrity_factor: float = 0.0    # Based on analysis integrity
    
    # Caps applied
    caps_applied: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall": self.overall,
            "factors": {
                "sample": self.sample_factor,
                "dispersion": self.dispersion_factor,
                "match_quality": self.match_quality_factor,
                "integrity": self.integrity_factor,
            },
            "caps_applied": self.caps_applied,
        }


def compute_estimate_confidence(
    n_precedents: int,
    avg_match_similarity: float,
    distribution: RecoveryDistribution,
    analysis_integrity: str = "PASS",
    min_n_for_confidence: int = 3,
) -> EstimateConfidence:
    """
    Compute confidence in recovery estimate.
    
    Rules are deterministic and recorded.
    """
    caps = []
    
    # Sample factor: increases with n, saturates around 10
    if n_precedents == 0:
        sample_factor = 0.0
    elif n_precedents < min_n_for_confidence:
        sample_factor = 0.3 * (n_precedents / min_n_for_confidence)
        caps.append(f"n_precedents < {min_n_for_confidence}")
    else:
        sample_factor = min(1.0, 0.5 + 0.05 * n_precedents)
    
    # Dispersion factor: lower IQR = higher confidence
    if distribution.iqr is not None and distribution.recovery_time_median:
        cv = distribution.iqr / distribution.recovery_time_median
        dispersion_factor = max(0, 1 - cv)
    else:
        dispersion_factor = 0.5
    
    # Match quality factor
    match_factor = avg_match_similarity
    
    # Integrity factor
    if analysis_integrity == "PASS":
        integrity_factor = 1.0
    elif analysis_integrity == "WARN":
        integrity_factor = 0.7
        caps.append("integrity=WARN")
    else:
        integrity_factor = 0.5
        caps.append("integrity=FAIL")
    
    # Combine (weighted geometric mean)
    factors = [sample_factor, dispersion_factor, match_factor, integrity_factor]
    weights = [0.35, 0.25, 0.25, 0.15]
    
    # Handle zeros
    if any(f == 0 for f in factors):
        overall = 0.2 * min(factors)
    else:
        log_weighted = sum(w * np.log(f) for w, f in zip(weights, factors))
        overall = np.exp(log_weighted)
    
    # Apply hard caps
    if n_precedents < min_n_for_confidence:
        overall = min(overall, 0.5)
    
    if analysis_integrity == "FAIL":
        overall = min(overall, 0.6)
    
    return EstimateConfidence(
        overall=round(overall, 3),
        sample_factor=round(sample_factor, 3),
        dispersion_factor=round(dispersion_factor, 3),
        match_quality_factor=round(match_factor, 3),
        integrity_factor=round(integrity_factor, 3),
        caps_applied=caps,
    )


# =============================================================================
# RECOVERY ESTIMATE (the final answer)
# =============================================================================

@dataclass
class RecoveryEstimate:
    """
    The complete recovery estimate with uncertainty.
    
    This is the institutional-grade answer to "how long until recovery?"
    """
    estimate_id: str
    run_id: str
    created_at: datetime
    
    # Conditioning signature (what we matched on)
    conditioning_signature: Dict[str, Any] = field(default_factory=dict)
    
    # Precedents
    precedent_episode_ids: List[str] = field(default_factory=list)
    n_precedents: int = 0
    
    # Distribution
    distribution: Optional[RecoveryDistribution] = None
    
    # Confidence
    confidence: Optional[EstimateConfidence] = None
    
    # Point estimates (for convenience, but distribution is primary)
    estimated_remaining_days_median: Optional[float] = None
    estimated_remaining_days_p25: Optional[float] = None
    estimated_remaining_days_p75: Optional[float] = None
    
    # Current progress
    current_duration_days: float = 0.0
    
    # Likely outcome
    most_likely_outcome: Optional[RecoveryType] = None
    outcome_confidence: float = 0.0
    
    # Warnings
    warnings: List[str] = field(default_factory=list)
    
    # Versions
    gates_version: str = StandardRecoveryGates.VERSION
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "estimate_id": self.estimate_id,
            "run_id": self.run_id,
            "created_at": self.created_at.isoformat(),
            "conditioning_signature": self.conditioning_signature,
            "precedents": {
                "episode_ids": self.precedent_episode_ids,
                "n": self.n_precedents,
            },
            "distribution": self.distribution.to_dict() if self.distribution else None,
            "confidence": self.confidence.to_dict() if self.confidence else None,
            "point_estimates": {
                "remaining_median": self.estimated_remaining_days_median,
                "remaining_p25": self.estimated_remaining_days_p25,
                "remaining_p75": self.estimated_remaining_days_p75,
            },
            "current_duration_days": self.current_duration_days,
            "most_likely_outcome": self.most_likely_outcome.value if self.most_likely_outcome else None,
            "outcome_confidence": self.outcome_confidence,
            "warnings": self.warnings,
            "gates_version": self.gates_version,
        }
    
    def to_narrative(self) -> str:
        """Generate human-readable narrative."""
        if self.n_precedents == 0:
            return "No precedent episodes found. Cannot estimate recovery."
        
        if self.n_precedents < 3:
            return f"Only {self.n_precedents} precedent(s) found — estimate is speculative."
        
        if not self.distribution:
            return "Distribution not computed."
        
        d = self.distribution
        c = self.confidence
        
        parts = []
        
        # Timing
        parts.append(
            f"Based on {self.n_precedents} similar episodes: "
            f"median recovery {d.recovery_time_median:.0f} days "
            f"(IQR: {d.recovery_time_p25:.0f}–{d.recovery_time_p75:.0f})"
        )
        
        # Progress
        if self.current_duration_days > 0:
            if d.recovery_time_median:
                pct_through = min(100, self.current_duration_days / d.recovery_time_median * 100)
                parts.append(f"You are on day {self.current_duration_days:.0f} ({pct_through:.0f}% of median)")
        
        # Remaining
        if self.estimated_remaining_days_median is not None:
            parts.append(
                f"Estimated remaining: {self.estimated_remaining_days_median:.0f} days "
                f"(range: {self.estimated_remaining_days_p25:.0f}–{self.estimated_remaining_days_p75:.0f})"
            )
        
        # Outcome
        if self.most_likely_outcome:
            parts.append(
                f"Most likely outcome: {self.most_likely_outcome.value} "
                f"({d.pct_full_recovery:.0f}% of precedents fully recovered)"
            )
        
        # Confidence
        if c:
            parts.append(f"Estimate confidence: {c.overall:.0%}")
            if c.caps_applied:
                parts.append(f"  Confidence limited by: {', '.join(c.caps_applied)}")
        
        return ". ".join(parts)


def build_recovery_estimate(
    run_id: str,
    conditioning_signature: Dict[str, Any],
    precedent_data: List[Tuple[str, float, RecoveryType]],  # (episode_id, duration_days, recovery_type)
    current_duration_days: float,
    avg_match_similarity: float,
    analysis_integrity: str = "PASS",
) -> RecoveryEstimate:
    """
    Build complete recovery estimate.
    """
    now = datetime.now(timezone.utc)
    estimate_id = str(uuid.uuid4())
    
    warnings = []
    
    if not precedent_data:
        return RecoveryEstimate(
            estimate_id=estimate_id,
            run_id=run_id,
            created_at=now,
            conditioning_signature=conditioning_signature,
            n_precedents=0,
            warnings=["No precedent episodes found"],
        )
    
    # Extract data
    episode_ids = [p[0] for p in precedent_data]
    recovery_times = [p[1] for p in precedent_data]
    recovery_types = [p[2] for p in precedent_data]
    
    n = len(precedent_data)
    
    if n < 3:
        warnings.append(f"Only {n} precedent(s) — estimate is speculative")
    
    # Compute distribution
    distribution = compute_recovery_distribution(recovery_times, recovery_types)
    
    # Compute confidence
    confidence = compute_estimate_confidence(
        n_precedents=n,
        avg_match_similarity=avg_match_similarity,
        distribution=distribution,
        analysis_integrity=analysis_integrity,
    )
    
    # Point estimates (remaining time)
    remaining_median = None
    remaining_p25 = None
    remaining_p75 = None
    
    if distribution.recovery_time_median:
        remaining_median = max(0, distribution.recovery_time_median - current_duration_days)
    if distribution.recovery_time_p25:
        remaining_p25 = max(0, distribution.recovery_time_p25 - current_duration_days)
    if distribution.recovery_time_p75:
        remaining_p75 = max(0, distribution.recovery_time_p75 - current_duration_days)
    
    # Most likely outcome
    outcome_pcts = {
        RecoveryType.FULL: distribution.pct_full_recovery,
        RecoveryType.PARTIAL: distribution.pct_partial_recovery,
        RecoveryType.REBASELINE_REQUIRED: distribution.pct_rebaseline,
        RecoveryType.NO_RECOVERY: distribution.pct_no_recovery,
    }
    most_likely = max(outcome_pcts, key=outcome_pcts.get)
    
    return RecoveryEstimate(
        estimate_id=estimate_id,
        run_id=run_id,
        created_at=now,
        conditioning_signature=conditioning_signature,
        precedent_episode_ids=episode_ids,
        n_precedents=n,
        distribution=distribution,
        confidence=confidence,
        estimated_remaining_days_median=remaining_median,
        estimated_remaining_days_p25=remaining_p25,
        estimated_remaining_days_p75=remaining_p75,
        current_duration_days=current_duration_days,
        most_likely_outcome=most_likely,
        outcome_confidence=outcome_pcts[most_likely],
        warnings=warnings,
    )


# =============================================================================
# TESTS
# =============================================================================

def test_gate_evaluation():
    """Test recovery gate evaluation."""
    evaluator = RecoveryGateEvaluator()
    
    # Good recovery context
    good_context = RecoveryGateContext(
        validity_score=75,
        severity_current=10,
        consecutive_clean=5,
        trust_penalty=10,
    )
    
    passed, failed = evaluator.evaluate_all(good_context)
    print(f"Good context: {len(passed)} passed, {len(failed)} failed")
    assert len(failed) == 0
    
    # Bad recovery context
    bad_context = RecoveryGateContext(
        validity_score=45,
        severity_current=40,
        consecutive_clean=1,
        trust_penalty=50,
    )
    
    passed, failed = evaluator.evaluate_all(bad_context)
    print(f"Bad context: {len(passed)} passed, {len(failed)} failed")
    assert len(failed) > 0
    
    print("✓ test_gate_evaluation passed")


def test_recovery_type_determination():
    """Test recovery type determination."""
    evaluator = RecoveryGateEvaluator()
    
    # Full recovery
    full_ctx = RecoveryGateContext(validity_score=80, severity_current=5, consecutive_clean=5)
    rt, passed, failed = evaluator.determine_recovery_type(full_ctx, has_irreversible_damage=False)
    print(f"Full recovery context → {rt.value}")
    assert rt == RecoveryType.FULL
    
    # Partial (with irreversible damage)
    partial_ctx = RecoveryGateContext(validity_score=60, severity_current=10, consecutive_clean=3, trust_penalty=20)
    rt, _, _ = evaluator.determine_recovery_type(partial_ctx, has_irreversible_damage=True)
    print(f"Partial recovery context → {rt.value}")
    assert rt == RecoveryType.PARTIAL
    
    # No recovery
    no_ctx = RecoveryGateContext(validity_score=30, severity_current=50, consecutive_clean=0)
    rt, _, _ = evaluator.determine_recovery_type(no_ctx)
    print(f"No recovery context → {rt.value}")
    assert rt == RecoveryType.NO_RECOVERY
    
    print("✓ test_recovery_type_determination passed")


def test_recovery_distribution():
    """Test recovery distribution computation."""
    times = [20, 25, 30, 35, 45, 50, 60, 90]
    types = [
        RecoveryType.FULL, RecoveryType.FULL, RecoveryType.FULL,
        RecoveryType.PARTIAL, RecoveryType.FULL, RecoveryType.PARTIAL,
        RecoveryType.REBASELINE_REQUIRED, RecoveryType.NO_RECOVERY,
    ]
    
    dist = compute_recovery_distribution(times, types)
    
    print(f"N: {dist.n_episodes}")
    print(f"Median: {dist.recovery_time_median}")
    print(f"IQR: [{dist.recovery_time_p25}, {dist.recovery_time_p75}]")
    print(f"Full recovery: {dist.pct_full_recovery:.0f}%")
    
    assert dist.n_episodes == 8
    assert dist.recovery_time_median == 40  # median of 8 values
    assert dist.iqr is not None
    
    print("✓ test_recovery_distribution passed")


def test_confidence_computation():
    """Test confidence computation."""
    dist = RecoveryDistribution(
        n_episodes=8,
        recovery_time_median=40,
        recovery_time_p25=27.5,
        recovery_time_p75=52.5,
    )
    
    # Good conditions
    conf_good = compute_estimate_confidence(
        n_precedents=8,
        avg_match_similarity=0.85,
        distribution=dist,
        analysis_integrity="PASS",
    )
    print(f"Good conditions confidence: {conf_good.overall:.2f}")
    assert conf_good.overall > 0.6
    
    # Poor conditions
    conf_poor = compute_estimate_confidence(
        n_precedents=2,
        avg_match_similarity=0.5,
        distribution=dist,
        analysis_integrity="WARN",
    )
    print(f"Poor conditions confidence: {conf_poor.overall:.2f}")
    print(f"Caps: {conf_poor.caps_applied}")
    assert conf_poor.overall < conf_good.overall
    assert len(conf_poor.caps_applied) > 0
    
    print("✓ test_confidence_computation passed")


def test_recovery_estimate():
    """Test full recovery estimate."""
    precedents = [
        ("ep1", 25, RecoveryType.FULL),
        ("ep2", 30, RecoveryType.FULL),
        ("ep3", 45, RecoveryType.PARTIAL),
        ("ep4", 35, RecoveryType.FULL),
        ("ep5", 60, RecoveryType.REBASELINE_REQUIRED),
    ]
    
    estimate = build_recovery_estimate(
        run_id="run_123",
        conditioning_signature={"fm_set": ["FM4"], "root_fm": "FM4"},
        precedent_data=precedents,
        current_duration_days=10,
        avg_match_similarity=0.78,
    )
    
    print(f"Estimate ID: {estimate.estimate_id}")
    print(f"N precedents: {estimate.n_precedents}")
    print(f"Median remaining: {estimate.estimated_remaining_days_median:.0f} days")
    print(f"IQR remaining: [{estimate.estimated_remaining_days_p25:.0f}, {estimate.estimated_remaining_days_p75:.0f}]")
    print(f"Confidence: {estimate.confidence.overall:.2f}")
    print(f"Most likely: {estimate.most_likely_outcome.value}")
    print()
    print("Narrative:")
    print(estimate.to_narrative())
    
    assert estimate.n_precedents == 5
    assert estimate.confidence.overall > 0.5
    
    print("\n✓ test_recovery_estimate passed")


def run_all_recovery_tests():
    print("\n" + "=" * 60)
    print("RECOVERY GATES & STATISTICS TESTS")
    print("=" * 60 + "\n")
    
    test_gate_evaluation()
    print()
    test_recovery_type_determination()
    print()
    test_recovery_distribution()
    print()
    test_confidence_computation()
    print()
    test_recovery_estimate()
    
    print("\n" + "=" * 60)
    print("ALL RECOVERY TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    run_all_recovery_tests()
