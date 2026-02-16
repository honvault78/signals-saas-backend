"""
Bavella Validity Engine — Core Types
=====================================

Defines the fundamental types for validity analysis:
- Failure Modes (FM1-FM7)
- Validity States
- Verdicts

These map directly to the full Bavella v2 specification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid


# =============================================================================
# FAILURE MODES
# =============================================================================

class FailureMode(Enum):
    """
    The 7 Failure Modes that can invalidate an analytical relationship.
    
    When ANY of these fire, the relationship is epistemically compromised.
    """
    FM1_VARIANCE_REGIME = "FM1_VARIANCE_REGIME"     # Volatility regime shift
    FM2_MEAN_DRIFT = "FM2_MEAN_DRIFT"               # Mean/fair value drifting
    FM3_CORRELATION_FLIP = "FM3_CORRELATION_FLIP"   # Correlation structure changed
    FM4_STRUCTURAL_BREAK = "FM4_STRUCTURAL_BREAK"   # Permanent structural change
    FM5_STATIONARITY_LOSS = "FM5_STATIONARITY_LOSS" # Non-stationarity detected
    FM6_TAIL_EVENT = "FM6_TAIL_EVENT"               # Extreme outlier event
    FM7_DEPENDENCY_BREAK = "FM7_DEPENDENCY_BREAK"   # Dependency relationship broke


# Convenience aliases
FM1_VARIANCE_REGIME = FailureMode.FM1_VARIANCE_REGIME
FM2_MEAN_DRIFT = FailureMode.FM2_MEAN_DRIFT
FM3_CORRELATION_FLIP = FailureMode.FM3_CORRELATION_FLIP
FM4_STRUCTURAL_BREAK = FailureMode.FM4_STRUCTURAL_BREAK
FM5_STATIONARITY_LOSS = FailureMode.FM5_STATIONARITY_LOSS
FM6_TAIL_EVENT = FailureMode.FM6_TAIL_EVENT
FM7_DEPENDENCY_BREAK = FailureMode.FM7_DEPENDENCY_BREAK


# FM metadata
FM_INFO = {
    FailureMode.FM1_VARIANCE_REGIME: {
        "name": "Variance Regime Shift",
        "description": "Volatility has changed regime — historical vol assumptions invalid",
        "severity_weight": 0.18,
        "typically_reversible": True,
    },
    FailureMode.FM2_MEAN_DRIFT: {
        "name": "Mean Drift",
        "description": "Fair value is drifting — mean reversion targets compromised",
        "severity_weight": 0.18,
        "typically_reversible": True,
    },
    FailureMode.FM3_CORRELATION_FLIP: {
        "name": "Correlation Flip",
        "description": "Correlation structure changed — hedge ratios need recalibration",
        "severity_weight": 0.08,
        "typically_reversible": True,
    },
    FailureMode.FM4_STRUCTURAL_BREAK: {
        "name": "Structural Break",
        "description": "Permanent structural change — historical relationship invalidated",
        "severity_weight": 0.20,
        "typically_reversible": False,
    },
    FailureMode.FM5_STATIONARITY_LOSS: {
        "name": "Stationarity Loss",
        "description": "Non-stationary behavior — mean reversion assumptions violated",
        "severity_weight": 0.15,
        "typically_reversible": True,
    },
    FailureMode.FM6_TAIL_EVENT: {
        "name": "Tail Event",
        "description": "Extreme outlier — statistical models temporarily unreliable",
        "severity_weight": 0.15,
        "typically_reversible": True,
    },
    FailureMode.FM7_DEPENDENCY_BREAK: {
        "name": "Dependency Break",
        "description": "Dependency relationship broke — spread dynamics unreliable",
        "severity_weight": 0.06,
        "typically_reversible": True,
    },
}


# =============================================================================
# VALIDITY STATES
# =============================================================================

class ValidityState(Enum):
    """Validity state of an analytical relationship."""
    VALID = "valid"           # Safe to trade on
    DEGRADED = "degraded"     # Caution advised
    INVALID = "invalid"       # Do not rely on
    
    @classmethod
    def from_score(cls, score: float) -> "ValidityState":
        """Convert validity score (0-100) to state."""
        if score >= 70:
            return cls.VALID
        elif score >= 30:
            return cls.DEGRADED
        else:
            return cls.INVALID


# =============================================================================
# VALIDITY VERDICT
# =============================================================================

@dataclass(frozen=True)
class ValidityVerdict:
    """
    The answer to "Is this relationship epistemically valid?"
    
    Immutable record of a validity determination.
    """
    # Identity
    verdict_id: str
    timestamp: datetime
    
    # Node being analyzed
    node_id: str
    
    # The verdict
    validity_score: float  # 0-100
    validity_state: ValidityState
    
    # Active failure modes
    active_fms: tuple  # Tuple[FailureMode, ...]
    
    # Fields with defaults must come after fields without defaults
    node_type: str = "series"
    primary_fm: Optional[FailureMode] = None  # Root cause
    confidence: float = 0.0  # 0-1
    regime: str = "unknown"
    evidence_summary: str = ""
    
    @classmethod
    def create(
        cls,
        node_id: str,
        validity_score: float,
        active_fms: List[FailureMode],
        primary_fm: Optional[FailureMode] = None,
        confidence: float = 0.8,
        regime: str = "unknown",
        evidence_summary: str = "",
        node_type: str = "series",
    ) -> "ValidityVerdict":
        """Factory method to create a verdict."""
        return cls(
            verdict_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            node_id=node_id,
            node_type=node_type,
            validity_score=validity_score,
            validity_state=ValidityState.from_score(validity_score),
            active_fms=tuple(active_fms),
            primary_fm=primary_fm or (active_fms[0] if active_fms else None),
            confidence=confidence,
            regime=regime,
            evidence_summary=evidence_summary,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "verdict_id": self.verdict_id,
            "timestamp": self.timestamp.isoformat(),
            "node_id": self.node_id,
            "node_type": self.node_type,
            "validity_score": round(self.validity_score, 1),
            "validity_state": self.validity_state.value,
            "active_fms": [fm.value for fm in self.active_fms],
            "primary_fm": self.primary_fm.value if self.primary_fm else None,
            "confidence": round(self.confidence, 2),
            "regime": self.regime,
            "evidence_summary": self.evidence_summary,
        }
    
    @property
    def is_valid(self) -> bool:
        return self.validity_state == ValidityState.VALID
    
    @property
    def fm_count(self) -> int:
        return len(self.active_fms)

# =============================================================================
# SEVERITY CALCULATION (v2.2.1 — corrected)
# =============================================================================

# Co-firing amplification: when multiple FMs fire, total penalty increases
CO_FIRE_THRESHOLD = 2          # Minimum active FMs to trigger amplification
CO_FIRE_AMPLIFICATION = 0.25   # 25% penalty increase per additional active FM
ACTIVE_FM_MIN_SEVERITY = 15.0  # Minimum severity to count as "active"

# Confidence penalty: low attribution confidence penalises score
CONFIDENCE_BASELINE = 0.70     # No penalty above this
CONFIDENCE_PENALTY_RATE = 15.0 # Points per 0.1 below baseline


def calculate_fm_severity(
    fm: FailureMode,
    raw_severity: float,  # 0-100 from detector
) -> float:
    """
    Calculate weighted severity for a failure mode.

    Weights are normalized (sum to 1.0) so that max theoretical
    penalty from all 7 FMs at severity 100 = 100 points.
    """
    weight = FM_INFO[fm]["severity_weight"]
    return raw_severity * weight


def calculate_validity_score(
    active_fms: List[tuple],  # List of (FailureMode, severity)
    confidence: float = 0.8,  # Attribution confidence (0-1)
) -> float:
    """
    Calculate overall validity score from active failure modes.

    Formula (v2.2.1):
        base_penalty   = sum(weight_i * severity_i)
        amplified      = base_penalty * co_fire_multiplier
        conf_penalty   = f(attribution_confidence)
        score          = clip(100 - amplified - conf_penalty, 0, 100)

    Fixes over v2.2.0:
        - Weighted sum (not average) -- each FM adds independent penalty
        - Co-firing amplification -- 3+ FMs compound non-linearly
        - Confidence penalty -- uncertain attribution penalises score
    """
    if not active_fms:
        return 100.0

    # -- Step 1: Weighted sum of penalties --
    total_penalty = 0.0
    active_count = 0

    for fm, severity in active_fms:
        weighted = calculate_fm_severity(fm, severity)
        total_penalty += weighted
        if severity >= ACTIVE_FM_MIN_SEVERITY:
            active_count += 1

    # -- Step 2: Co-firing amplification --
    # Multiple simultaneous failures are worse than the sum of parts
    co_fire_multiplier = 1.0
    if active_count >= CO_FIRE_THRESHOLD:
        extra = active_count - CO_FIRE_THRESHOLD
        co_fire_multiplier = 1.0 + extra * CO_FIRE_AMPLIFICATION

    amplified_penalty = total_penalty * co_fire_multiplier

    # -- Step 3: Confidence penalty --
    # Low attribution confidence = engine can't identify root cause
    # This uncertainty is itself a risk signal
    confidence_penalty = 0.0
    if confidence < CONFIDENCE_BASELINE:
        gap = CONFIDENCE_BASELINE - confidence
        confidence_penalty = gap * CONFIDENCE_PENALTY_RATE

    # -- Step 4: Final score --
    validity = max(0, 100 - amplified_penalty - confidence_penalty)
    return validity
