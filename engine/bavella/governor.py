"""
Bavella v2 — The Governor
==========================

The Governor is the single authority for emissions.

Nothing escapes the system without Governor clearance.

This is not a "suppression layer" - this IS the output system.
There is no bypass. There is no .force(). There is no backdoor.

Copyright 2024-2026 Bavella Technologies Sarl
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import (
    Any, Callable, Dict, Generic, List, Optional, 
    TypeVar, Union, Protocol, runtime_checkable
)
import json

from .core import (
    ValidityState, ValidityVerdict, GovernedValue, GovernorRefusal,
    AnalysisNode, NodeIdentity, FailureMode, Thresholds
)


# =============================================================================
# OUTPUT TYPES (what the Governor can emit)
# =============================================================================

class OutputLevel(Enum):
    """
    Output levels per Engineering Addendum.
    
    L0: Always visible (validity itself, raw series identity)
    L1: Suppressed when INVALID
    L2: Suppressed when DEGRADED
    """
    L0_ALWAYS = auto()  # Validity score, state, attribution, raw series ID
    L1_VALID_OR_DEGRADED = auto()  # Statistics, metrics, regime labels
    L2_VALID_ONLY = auto()  # Predictions, recommendations, comparisons


@dataclass(frozen=True)
class OutputRequest:
    """A request to emit a value from the system."""
    value: Any
    output_level: OutputLevel
    description: str
    source_node_id: str
    
    # Optional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OutputDecision:
    """The Governor's decision on an output request."""
    permitted: bool
    value: Optional[Any]  # None if not permitted
    
    # If permitted with conditions
    requires_watermark: bool = False
    watermark_text: str = ""
    
    # If not permitted
    refusal_reason: Optional[str] = None
    
    # Audit trail
    decision_id: str = field(default_factory=lambda: str(__import__('uuid').uuid4()))
    decided_at: datetime = field(default_factory=datetime.utcnow)
    
    # The verdict that governed this decision
    governing_verdict: Optional[ValidityVerdict] = None


# =============================================================================
# EMISSION RECORD (immutable audit trail)
# =============================================================================

@dataclass(frozen=True)
class EmissionRecord:
    """
    Immutable record of every emission attempt.
    
    This creates a complete audit trail of:
        - What was requested
        - What was permitted
        - What was refused
        - Why
    """
    request: OutputRequest
    decision: OutputDecision
    
    # Context
    session_id: str
    sequence_number: int
    
    def to_audit_dict(self) -> Dict[str, Any]:
        """Convert to audit-friendly dictionary."""
        return {
            "emission_id": self.decision.decision_id,
            "timestamp": self.decision.decided_at.isoformat(),
            "session": self.session_id,
            "sequence": self.sequence_number,
            "source_node": self.request.source_node_id,
            "output_level": self.request.output_level.name,
            "permitted": self.decision.permitted,
            "refusal_reason": self.decision.refusal_reason,
            "watermarked": self.decision.requires_watermark,
            "validity_score": (
                self.decision.governing_verdict.score 
                if self.decision.governing_verdict else None
            ),
            "validity_state": (
                self.decision.governing_verdict.state.value
                if self.decision.governing_verdict else None
            ),
        }


# =============================================================================
# THE GOVERNOR
# =============================================================================

class Governor:
    """
    The Governor is the single authority for emissions.
    
    Responsibilities:
        1. Receive all output requests
        2. Consult validity verdicts
        3. Apply suppression rules
        4. Emit or refuse
        5. Log everything
    
    There is no way to emit without going through the Governor.
    """
    
    def __init__(self, session_id: Optional[str] = None):
        self._session_id = session_id or str(__import__('uuid').uuid4())
        self._emission_log: List[EmissionRecord] = []
        self._sequence = 0
        
        # Node registry for verdict lookup
        self._node_verdicts: Dict[str, ValidityVerdict] = {}
    
    @property
    def session_id(self) -> str:
        return self._session_id
    
    @property
    def emission_count(self) -> int:
        return len(self._emission_log)
    
    def register_verdict(self, node_id: str, verdict: ValidityVerdict) -> None:
        """Register a node's validity verdict."""
        self._node_verdicts[node_id] = verdict
    
    def request_emission(
        self,
        request: OutputRequest,
    ) -> OutputDecision:
        """
        Request to emit a value.
        
        This is THE gate. Everything must pass through here.
        """
        # Get the governing verdict
        verdict = self._node_verdicts.get(request.source_node_id)
        
        if verdict is None:
            # No verdict registered - cannot emit
            decision = OutputDecision(
                permitted=False,
                value=None,
                refusal_reason="No validity verdict registered for source node",
                governing_verdict=None,
            )
        else:
            # Apply suppression rules
            decision = self._apply_suppression_rules(request, verdict)
        
        # Log the emission attempt
        self._sequence += 1
        record = EmissionRecord(
            request=request,
            decision=decision,
            session_id=self._session_id,
            sequence_number=self._sequence,
        )
        self._emission_log.append(record)
        
        return decision
    
    def _apply_suppression_rules(
        self,
        request: OutputRequest,
        verdict: ValidityVerdict,
    ) -> OutputDecision:
        """
        Apply the suppression matrix.
        
        Rules:
            L0 (ALWAYS): Always permitted
            L1 (VALID_OR_DEGRADED): Suppressed when INVALID
            L2 (VALID_ONLY): Suppressed when DEGRADED or INVALID
        """
        state = verdict.state
        level = request.output_level
        
        # L0: Always permitted
        if level == OutputLevel.L0_ALWAYS:
            return OutputDecision(
                permitted=True,
                value=request.value,
                requires_watermark=False,
                governing_verdict=verdict,
            )
        
        # INVALID: Only L0 permitted
        if state == ValidityState.INVALID:
            return OutputDecision(
                permitted=False,
                value=None,
                refusal_reason=self._format_refusal(verdict),
                governing_verdict=verdict,
            )
        
        # L2: Only VALID permitted
        if level == OutputLevel.L2_VALID_ONLY:
            if state == ValidityState.VALID:
                return OutputDecision(
                    permitted=True,
                    value=request.value,
                    requires_watermark=False,
                    governing_verdict=verdict,
                )
            else:  # DEGRADED
                return OutputDecision(
                    permitted=False,
                    value=None,
                    refusal_reason="Output level L2 requires VALID state; current state is DEGRADED",
                    governing_verdict=verdict,
                )
        
        # L1: VALID or DEGRADED permitted
        if level == OutputLevel.L1_VALID_OR_DEGRADED:
            if state == ValidityState.VALID:
                return OutputDecision(
                    permitted=True,
                    value=request.value,
                    requires_watermark=False,
                    governing_verdict=verdict,
                )
            else:  # DEGRADED
                return OutputDecision(
                    permitted=True,
                    value=request.value,
                    requires_watermark=True,
                    watermark_text=self._generate_watermark(verdict),
                    governing_verdict=verdict,
                )
        
        # Fallback: deny
        return OutputDecision(
            permitted=False,
            value=None,
            refusal_reason="Unknown output level",
            governing_verdict=verdict,
        )
    
    def _format_refusal(self, verdict: ValidityVerdict) -> str:
        """Format a human-readable refusal reason."""
        if verdict.is_killed:
            return (
                f"INVALID (kill switch): {verdict.killed_by.name} - {verdict.kill_reason}"
            )
        
        dominant = verdict.dominant_failure
        if dominant:
            return f"INVALID (score={verdict.score:.1f}): dominant failure is {dominant.name}"
        
        return f"INVALID (score={verdict.score:.1f})"
    
    def _generate_watermark(self, verdict: ValidityVerdict) -> str:
        """Generate watermark text for DEGRADED outputs."""
        dominant = verdict.dominant_failure
        if dominant:
            return (
                f"⚠️ VALIDITY DEGRADED ({verdict.score:.0f}/100) - "
                f"Primary concern: {dominant.name.replace('_', ' ').title()}"
            )
        return f"⚠️ VALIDITY DEGRADED ({verdict.score:.0f}/100)"
    
    # =========================================================================
    # CONVENIENCE METHODS FOR COMMON EMISSIONS
    # =========================================================================
    
    def emit_metric(
        self,
        node_id: str,
        name: str,
        value: float,
        **metadata,
    ) -> OutputDecision:
        """Emit a statistical metric (L1)."""
        return self.request_emission(OutputRequest(
            value={"name": name, "value": value, **metadata},
            output_level=OutputLevel.L1_VALID_OR_DEGRADED,
            description=f"Metric: {name}",
            source_node_id=node_id,
            metadata=metadata,
        ))
    
    def emit_prediction(
        self,
        node_id: str,
        prediction: Any,
        **metadata,
    ) -> OutputDecision:
        """Emit a prediction (L2 - VALID only)."""
        return self.request_emission(OutputRequest(
            value=prediction,
            output_level=OutputLevel.L2_VALID_ONLY,
            description="Prediction",
            source_node_id=node_id,
            metadata=metadata,
        ))
    
    def emit_validity(
        self,
        node_id: str,
        verdict: ValidityVerdict,
    ) -> OutputDecision:
        """Emit validity information (L0 - always)."""
        return self.request_emission(OutputRequest(
            value={
                "score": verdict.score,
                "state": verdict.state.value,
                "attributions": [
                    {"failure_mode": fm.name, "pct": pct}
                    for fm, pct in verdict.attributions
                ],
                "killed_by": verdict.killed_by.name if verdict.killed_by else None,
                "kill_reason": verdict.kill_reason,
            },
            output_level=OutputLevel.L0_ALWAYS,
            description="Validity verdict",
            source_node_id=node_id,
        ))
    
    # =========================================================================
    # AUDIT INTERFACE
    # =========================================================================
    
    def get_emission_log(self) -> List[Dict[str, Any]]:
        """Get the complete emission log for audit."""
        return [r.to_audit_dict() for r in self._emission_log]
    
    def get_refusals(self) -> List[EmissionRecord]:
        """Get all refused emissions."""
        return [r for r in self._emission_log if not r.decision.permitted]
    
    def get_permissions(self) -> List[EmissionRecord]:
        """Get all permitted emissions."""
        return [r for r in self._emission_log if r.decision.permitted]
    
    def summary(self) -> Dict[str, Any]:
        """Get a summary of Governor activity."""
        total = len(self._emission_log)
        permitted = sum(1 for r in self._emission_log if r.decision.permitted)
        refused = total - permitted
        watermarked = sum(
            1 for r in self._emission_log 
            if r.decision.permitted and r.decision.requires_watermark
        )
        
        return {
            "session_id": self._session_id,
            "total_requests": total,
            "permitted": permitted,
            "refused": refused,
            "watermarked": watermarked,
            "refusal_rate": refused / total if total > 0 else 0.0,
        }


# =============================================================================
# GOVERNED API RESPONSE BUILDER
# =============================================================================

class GovernedResponseBuilder:
    """
    Builds API responses that respect Governor decisions.
    
    This is how external APIs emit values. They don't
    access data directly - they go through this builder
    which consults the Governor.
    """
    
    def __init__(self, governor: Governor):
        self._governor = governor
        self._response: Dict[str, Any] = {}
        self._suppressed_fields: List[str] = []
    
    def add_always(self, key: str, node_id: str, value: Any) -> 'GovernedResponseBuilder':
        """Add a field that's always included (L0)."""
        decision = self._governor.request_emission(OutputRequest(
            value=value,
            output_level=OutputLevel.L0_ALWAYS,
            description=f"Response field: {key}",
            source_node_id=node_id,
        ))
        
        if decision.permitted:
            self._response[key] = decision.value
        
        return self
    
    def add_metric(self, key: str, node_id: str, value: Any) -> 'GovernedResponseBuilder':
        """Add a metric field (L1)."""
        decision = self._governor.request_emission(OutputRequest(
            value=value,
            output_level=OutputLevel.L1_VALID_OR_DEGRADED,
            description=f"Response field: {key}",
            source_node_id=node_id,
        ))
        
        if decision.permitted:
            if decision.requires_watermark:
                self._response[key] = {
                    "value": decision.value,
                    "_watermark": decision.watermark_text,
                }
            else:
                self._response[key] = decision.value
        else:
            self._suppressed_fields.append(key)
        
        return self
    
    def add_prediction(self, key: str, node_id: str, value: Any) -> 'GovernedResponseBuilder':
        """Add a prediction field (L2)."""
        decision = self._governor.request_emission(OutputRequest(
            value=value,
            output_level=OutputLevel.L2_VALID_ONLY,
            description=f"Response field: {key}",
            source_node_id=node_id,
        ))
        
        if decision.permitted:
            self._response[key] = decision.value
        else:
            self._suppressed_fields.append(key)
        
        return self
    
    def build(self) -> Dict[str, Any]:
        """Build the final response."""
        result = dict(self._response)
        
        if self._suppressed_fields:
            result["_suppressed"] = self._suppressed_fields
            result["_suppression_note"] = (
                "Some fields have been suppressed due to validity constraints. "
                "This is expected behavior when inference assumptions are violated."
            )
        
        return result


# =============================================================================
# GOVERNED EXPORT
# =============================================================================

class GovernedExporter:
    """
    Exports data with Governor enforcement.
    
    Used for CSV, JSON, PDF exports.
    All exports go through here - no direct data access.
    """
    
    def __init__(self, governor: Governor):
        self._governor = governor
    
    def can_export(self, node_id: str) -> bool:
        """Check if full export is permitted for a node."""
        verdict = self._governor._node_verdicts.get(node_id)
        if verdict is None:
            return False
        
        # Full exports require VALID state
        return verdict.state == ValidityState.VALID
    
    def export_with_governance(
        self,
        node_id: str,
        data: Dict[str, Any],
        format: str = "json",
    ) -> Optional[str]:
        """
        Export data with governance enforcement.
        
        Returns None if export not permitted.
        """
        verdict = self._governor._node_verdicts.get(node_id)
        
        if verdict is None:
            return None
        
        if verdict.state == ValidityState.INVALID:
            return None
        
        # Request export permission
        decision = self._governor.request_emission(OutputRequest(
            value=data,
            output_level=OutputLevel.L2_VALID_ONLY,
            description=f"Full export ({format})",
            source_node_id=node_id,
        ))
        
        if not decision.permitted:
            return None
        
        if format == "json":
            return json.dumps(data, indent=2, default=str)
        
        # Add other formats as needed
        return None


# =============================================================================
# GLOBAL GOVERNOR INSTANCE
# =============================================================================

_global_governor: Optional[Governor] = None

def get_governor() -> Governor:
    """Get the global Governor instance."""
    global _global_governor
    if _global_governor is None:
        _global_governor = Governor()
    return _global_governor

def new_session() -> Governor:
    """Create a new Governor session."""
    global _global_governor
    _global_governor = Governor()
    return _global_governor
