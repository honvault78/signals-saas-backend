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
        "severity_weight": 1.0,
        "typically_reversible": True,
    },
    FailureMode.FM2_MEAN_DRIFT: {
        "name": "Mean Drift",
        "description": "Fair value is drifting — mean reversion targets compromised",
        "severity_weight": 0.8,
        "typically_reversible": True,
    },
    FailureMode.FM3_CORRELATION_FLIP: {
        "name": "Correlation Flip",
        "description": "Correlation structure changed — hedge ratios need recalibration",
        "severity_weight": 1.2,
        "typically_reversible": True,
    },
    FailureMode.FM4_STRUCTURAL_BREAK: {
        "name": "Structural Break",
        "description": "Permanent structural change — historical relationship invalidated",
        "severity_weight": 2.0,
        "typically_reversible": False,
    },
    FailureMode.FM5_STATIONARITY_LOSS: {
        "name": "Stationarity Loss",
        "description": "Non-stationary behavior — mean reversion assumptions violated",
        "severity_weight": 1.5,
        "typically_reversible": True,
    },
    FailureMode.FM6_TAIL_EVENT: {
        "name": "Tail Event",
        "description": "Extreme outlier — statistical models temporarily unreliable",
        "severity_weight": 0.6,
        "typically_reversible": True,
    },
    FailureMode.FM7_DEPENDENCY_BREAK: {
        "name": "Dependency Break",
        "description": "Dependency relationship broke — spread dynamics unreliable",
        "severity_weight": 1.8,
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
# SEVERITY CALCULATION
# =============================================================================

def calculate_fm_severity(
    fm: FailureMode,
    raw_severity: float,  # 0-100 from detector
) -> float:
    """
    Calculate weighted severity for a failure mode.
    
    Different FMs have different impact weights.
    """
    weight = FM_INFO[fm]["severity_weight"]
    return raw_severity * weight


def calculate_validity_score(
    active_fms: List[tuple],  # List of (FailureMode, severity)
) -> float:
    """
    Calculate overall validity score from active failure modes.
    
    Validity = 100 - Σ(weighted_severity)
    """
    if not active_fms:
        return 100.0
    
    total_penalty = 0.0
    for fm, severity in active_fms:
        weighted = calculate_fm_severity(fm, severity)
        total_penalty += weighted
    
    # Cap penalty at 100
    validity = max(0, 100 - total_penalty)
    return validity
