"""
Bavella v2 — Core Ontology (Rebuilt)
=====================================

THIS IS THE ARCHITECTURAL INVERSION.

Old: Analysis → Validity Score → Maybe Suppress (UI)
New: Validity Governor → Decides if Analysis Can Emit → Nothing Escapes Ungoverned

Key principles enforced at the type level:
    1. Everything is an AnalysisNode in a DAG
    2. Nodes cannot emit without Governor clearance
    3. INVALID nodes produce NOTHING (not hidden values - NO values)
    4. Inheritance is hard min(), not soft influence
    5. History is append-only, immutable, versioned
    6. Kill switches override additive scoring

Copyright 2024-2026 Bavella Technologies Sarl
"""

from __future__ import annotations

import hashlib
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import (
    Any, Dict, FrozenSet, Generic, List, Optional, 
    Set, Tuple, TypeVar, Union, Callable
)
import json


# =============================================================================
# FROZEN CONSTANTS (per Engineering Addendum v1.2)
# =============================================================================

class Thresholds:
    """Immutable validity thresholds. NEVER CHANGE AT RUNTIME."""
    VALID_MIN = 70.0
    DEGRADED_MIN = 30.0
    INVALID_MAX = 29.99
    
    # Kill switch thresholds (instant INVALID regardless of score)
    KILL_FM4_BREAK_MAGNITUDE = 4.0  # σ
    KILL_FM7_CORRELATION_FLIP = True  # sign reversal
    KILL_UPSTREAM_INVALID_COUNT = 1  # any invalid parent kills child


class Weights:
    """Frozen detector weights. Changes require new version."""
    FM1_VARIANCE = 0.18
    FM2_MEAN_DRIFT = 0.18
    FM3_SEASONALITY = 0.10
    FM4_STRUCTURAL = 0.20
    FM5_OUTLIERS = 0.12
    FM6_DISTRIBUTION = 0.12
    FM7_DEPENDENCY = 0.10
    
    VERSION = "1.0.0"  # Weight version for audit trail


# =============================================================================
# VALIDITY STATE (the only states that exist)
# =============================================================================

class ValidityState(Enum):
    """
    The three states of epistemic validity.
    
    VALID: All assumptions hold. Full inference permitted.
    DEGRADED: Some assumptions weakened. Inference with watermark.
    INVALID: Critical assumptions violated. NO INFERENCE EMITTED.
    
    Note: There is no UNKNOWN. If we cannot determine validity,
    the node does not exist yet.
    """
    VALID = "valid"
    DEGRADED = "degraded"
    INVALID = "invalid"
    
    @classmethod
    def from_score(cls, score: float) -> ValidityState:
        """Determine state from score. Kill switches bypass this."""
        if score >= Thresholds.VALID_MIN:
            return cls.VALID
        elif score >= Thresholds.DEGRADED_MIN:
            return cls.DEGRADED
        else:
            return cls.INVALID


# =============================================================================
# FAILURE MODE ENUMERATION
# =============================================================================

class FailureMode(Enum):
    """The seven canonical failure modes."""
    FM1_VARIANCE_REGIME = auto()
    FM2_MEAN_DRIFT = auto()
    FM3_SEASONALITY_MISMATCH = auto()
    FM4_STRUCTURAL_BREAK = auto()
    FM5_OUTLIER_CONTAMINATION = auto()
    FM6_DISTRIBUTIONAL_SHIFT = auto()
    FM7_DEPENDENCY_BREAK = auto()
    
    def is_kill_switch_capable(self) -> bool:
        """Can this FM trigger instant INVALID regardless of score?"""
        return self in {
            FailureMode.FM4_STRUCTURAL_BREAK,
            FailureMode.FM7_DEPENDENCY_BREAK,
        }
    
    def get_weight(self) -> float:
        """Get frozen weight for this failure mode."""
        return {
            FailureMode.FM1_VARIANCE_REGIME: Weights.FM1_VARIANCE,
            FailureMode.FM2_MEAN_DRIFT: Weights.FM2_MEAN_DRIFT,
            FailureMode.FM3_SEASONALITY_MISMATCH: Weights.FM3_SEASONALITY,
            FailureMode.FM4_STRUCTURAL_BREAK: Weights.FM4_STRUCTURAL,
            FailureMode.FM5_OUTLIER_CONTAMINATION: Weights.FM5_OUTLIERS,
            FailureMode.FM6_DISTRIBUTIONAL_SHIFT: Weights.FM6_DISTRIBUTION,
            FailureMode.FM7_DEPENDENCY_BREAK: Weights.FM7_DEPENDENCY,
        }[self]


# =============================================================================
# FAILURE MODE SIGNAL (immutable detection result)
# =============================================================================

@dataclass(frozen=True)
class FailureSignal:
    """
    Immutable record of a failure mode detection.
    
    frozen=True ensures this cannot be modified after creation.
    """
    failure_mode: FailureMode
    severity: float  # 0-100
    confidence: float  # 0-1
    
    # Kill switch evidence (if applicable)
    triggers_kill: bool = False
    kill_reason: Optional[str] = None
    
    # Causal ordering
    first_detected_at: datetime = field(default_factory=datetime.utcnow)
    
    # Raw evidence (for audit)
    evidence: Dict[str, Any] = field(default_factory=dict)
    
    # Human explanation
    explanation: str = ""
    
    def __post_init__(self):
        # Validate bounds (frozen dataclass uses object.__setattr__)
        if not 0 <= self.severity <= 100:
            raise ValueError(f"Severity must be 0-100, got {self.severity}")
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"Confidence must be 0-1, got {self.confidence}")


# =============================================================================
# VALIDITY VERDICT (immutable, attributed, ordered)
# =============================================================================

@dataclass(frozen=True)
class ValidityVerdict:
    """
    Immutable validity determination.
    
    This is NOT just a score. It encodes:
        - The numeric score (for gradation)
        - The state (for action decisions)
        - Kill switch status (overrides score)
        - Attribution (mandatory when score < 100)
        - Causal ordering (which FM was detected first)
        - Inheritance cap (from upstream)
    """
    # Core determination
    score: float
    state: ValidityState
    
    # Kill switch status
    killed_by: Optional[FailureMode] = None
    kill_reason: Optional[str] = None
    
    # Attribution (MUST sum to 100% when score < 100)
    attributions: Tuple[Tuple[FailureMode, float], ...] = ()
    
    # Causal ordering (first-detected to last)
    causal_order: Tuple[FailureMode, ...] = ()
    
    # Inheritance
    inherited_from: Optional[str] = None  # node_id that capped us
    pre_inheritance_score: Optional[float] = None
    
    # Metadata
    computed_at: datetime = field(default_factory=datetime.utcnow)
    verdict_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    @property
    def is_killed(self) -> bool:
        """Was this verdict forced by a kill switch?"""
        return self.killed_by is not None
    
    @property
    def dominant_failure(self) -> Optional[FailureMode]:
        """The failure mode most responsible for validity loss."""
        if self.killed_by:
            return self.killed_by
        if self.attributions:
            return max(self.attributions, key=lambda x: x[1])[0]
        return None
    
    def validate_attribution(self) -> bool:
        """Check that attribution sums to ~100% when required."""
        if self.score >= 100:
            return True
        if not self.attributions:
            return False
        total = sum(pct for _, pct in self.attributions)
        return 99.0 <= total <= 101.0


# =============================================================================
# NODE IDENTITY (immutable, content-addressed)
# =============================================================================

@dataclass(frozen=True)
class NodeIdentity:
    """
    Immutable identity of an analysis node.
    
    Content-addressed: same inputs → same identity.
    This enables deduplication and audit trails.
    """
    node_id: str
    node_type: str
    created_at: datetime
    
    # Content hash (SHA-256 of inputs)
    content_hash: str
    
    # Lineage
    parent_ids: FrozenSet[str] = frozenset()
    
    # Version tracking
    schema_version: str = "2.0.0"
    
    def __hash__(self):
        return hash(self.node_id)


# =============================================================================
# GOVERNED VALUE (the key abstraction)
# =============================================================================

T = TypeVar('T')

class GovernedValue(Generic[T]):
    """
    A value that can only be accessed if validity permits.
    
    THIS IS THE CORE ABSTRACTION.
    
    - VALID: .get() returns the value
    - DEGRADED: .get() returns the value (caller must handle watermark)
    - INVALID: .get() raises GovernorRefusal
    
    There is no .get_anyway() or .force(). INVALID means NO VALUE.
    """
    
    def __init__(
        self,
        value: T,
        verdict: ValidityVerdict,
        node_identity: NodeIdentity,
    ):
        self._value = value
        self._verdict = verdict
        self._identity = node_identity
        self._access_log: List[datetime] = []
    
    @property
    def validity(self) -> ValidityVerdict:
        """Always accessible - you can always ask about validity."""
        return self._verdict
    
    @property
    def identity(self) -> NodeIdentity:
        """Always accessible - you can always ask about identity."""
        return self._identity
    
    @property
    def state(self) -> ValidityState:
        """Convenience accessor for validity state."""
        return self._verdict.state
    
    def get(self) -> T:
        """
        Get the governed value.
        
        Raises:
            GovernorRefusal: If validity state is INVALID
            
        Returns:
            The value if VALID or DEGRADED
        """
        if self._verdict.state == ValidityState.INVALID:
            raise GovernorRefusal(
                node_id=self._identity.node_id,
                verdict=self._verdict,
                message="Cannot access value: node is INVALID"
            )
        
        self._access_log.append(datetime.utcnow())
        return self._value
    
    def get_if_valid(self) -> Optional[T]:
        """
        Get value only if VALID (not DEGRADED).
        
        Returns None for both DEGRADED and INVALID.
        Use this when you want strict validity only.
        """
        if self._verdict.state == ValidityState.VALID:
            self._access_log.append(datetime.utcnow())
            return self._value
        return None
    
    def requires_watermark(self) -> bool:
        """Does accessing this value require a watermark?"""
        return self._verdict.state == ValidityState.DEGRADED
    
    def map(self, f: Callable[[T], Any]) -> 'GovernedValue':
        """
        Transform the value while preserving governance.
        
        The transformed value inherits the same validity.
        """
        if self._verdict.state == ValidityState.INVALID:
            # Cannot map over invalid - create a new invalid governed value
            return GovernedValue(
                value=None,  # No value to transform
                verdict=self._verdict,
                node_identity=self._identity,
            )
        
        new_value = f(self._value)
        return GovernedValue(
            value=new_value,
            verdict=self._verdict,
            node_identity=self._identity,
        )


class GovernorRefusal(Exception):
    """
    Raised when attempting to access an INVALID governed value.
    
    This is not an error - this is the system working correctly.
    INVALID means no value exists to return.
    """
    
    def __init__(
        self,
        node_id: str,
        verdict: ValidityVerdict,
        message: str,
    ):
        self.node_id = node_id
        self.verdict = verdict
        super().__init__(f"[{node_id}] {message}")


# =============================================================================
# ANALYSIS NODE (base class for all nodes in the DAG)
# =============================================================================

class AnalysisNode(ABC):
    """
    Abstract base class for all nodes in the analysis DAG.
    
    Every piece of analysis - raw series, transforms, metrics,
    models, predictions - is a node.
    
    Nodes:
        - Have immutable identity
        - Have validity verdict
        - Can only emit governed values
        - Inherit validity from parents (hard min)
    """
    
    def __init__(
        self,
        node_id: str,
        node_type: str,
        parent_nodes: Optional[List['AnalysisNode']] = None,
    ):
        self._parent_nodes = parent_nodes or []
        self._verdict: Optional[ValidityVerdict] = None
        self._created_at = datetime.utcnow()
        
        # Compute content hash from inputs
        content = self._compute_content_for_hash()
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        
        self._identity = NodeIdentity(
            node_id=node_id,
            node_type=node_type,
            created_at=self._created_at,
            content_hash=content_hash,
            parent_ids=frozenset(p.identity.node_id for p in self._parent_nodes),
        )
    
    @property
    def identity(self) -> NodeIdentity:
        return self._identity
    
    @property
    def verdict(self) -> Optional[ValidityVerdict]:
        return self._verdict
    
    @property
    def parents(self) -> List['AnalysisNode']:
        return self._parent_nodes
    
    @abstractmethod
    def _compute_content_for_hash(self) -> str:
        """Compute content string for identity hash."""
        pass
    
    @abstractmethod
    def _compute_own_validity(self) -> Tuple[float, List[FailureSignal]]:
        """
        Compute this node's own validity (before inheritance).
        
        Returns:
            (score, signals) where score is 0-100 and signals
            are the failure mode detections.
        """
        pass
    
    def compute_validity(self) -> ValidityVerdict:
        """
        Compute final validity including inheritance.
        
        This enforces:
            1. Own validity from failure mode detection
            2. Kill switch override
            3. Hard min() inheritance from parents
        """
        # Step 1: Check if any parent is INVALID (instant kill)
        invalid_parents = [
            p for p in self._parent_nodes
            if p.verdict and p.verdict.state == ValidityState.INVALID
        ]
        
        if invalid_parents:
            # Kill switch: any invalid parent → this node is invalid
            parent = invalid_parents[0]
            self._verdict = ValidityVerdict(
                score=0.0,
                state=ValidityState.INVALID,
                killed_by=None,  # Killed by inheritance, not FM
                kill_reason=f"Inherited INVALID from parent: {parent.identity.node_id}",
                inherited_from=parent.identity.node_id,
                pre_inheritance_score=None,  # Never computed
            )
            return self._verdict
        
        # Step 2: Compute own validity
        own_score, signals = self._compute_own_validity()
        
        # Step 3: Check for kill switches
        kill_signal = next(
            (s for s in signals if s.triggers_kill),
            None
        )
        
        if kill_signal:
            self._verdict = ValidityVerdict(
                score=0.0,
                state=ValidityState.INVALID,
                killed_by=kill_signal.failure_mode,
                kill_reason=kill_signal.kill_reason,
                attributions=((kill_signal.failure_mode, 100.0),),
                causal_order=tuple(s.failure_mode for s in sorted(
                    signals, key=lambda x: x.first_detected_at
                )),
            )
            return self._verdict
        
        # Step 4: Compute attribution (if score < 100)
        attributions = self._compute_attribution(own_score, signals)
        
        # Step 5: Apply inheritance (hard min)
        parent_scores = [
            (p.identity.node_id, p.verdict.score)
            for p in self._parent_nodes
            if p.verdict is not None
        ]
        
        inherited_from = None
        pre_inheritance_score = own_score
        final_score = own_score
        
        for parent_id, parent_score in parent_scores:
            if parent_score < final_score:
                final_score = parent_score
                inherited_from = parent_id
        
        # Step 6: Determine state
        state = ValidityState.from_score(final_score)
        
        self._verdict = ValidityVerdict(
            score=final_score,
            state=state,
            attributions=tuple(attributions),
            causal_order=tuple(s.failure_mode for s in sorted(
                signals, key=lambda x: x.first_detected_at
            )),
            inherited_from=inherited_from,
            pre_inheritance_score=pre_inheritance_score if inherited_from else None,
        )
        
        return self._verdict
    
    def _compute_attribution(
        self,
        score: float,
        signals: List[FailureSignal],
    ) -> List[Tuple[FailureMode, float]]:
        """Compute attribution that sums to 100%."""
        if score >= 100:
            return []
        
        validity_loss = 100.0 - score
        
        # Compute raw contributions
        contributions = []
        total_raw = 0.0
        
        for signal in signals:
            if signal.severity > 0:
                weight = signal.failure_mode.get_weight()
                raw = weight * signal.severity * signal.confidence
                contributions.append((signal.failure_mode, raw))
                total_raw += raw
        
        if total_raw == 0:
            return []
        
        # Normalize to 100%
        return [
            (fm, (raw / total_raw) * 100.0)
            for fm, raw in contributions
        ]
    
    def emit(self, value: T) -> GovernedValue[T]:
        """
        Emit a governed value from this node.
        
        The value is wrapped with this node's validity verdict.
        If INVALID, the GovernedValue will refuse access.
        """
        if self._verdict is None:
            raise RuntimeError("Cannot emit before validity is computed")
        
        return GovernedValue(
            value=value,
            verdict=self._verdict,
            node_identity=self._identity,
        )


# =============================================================================
# VALIDITY HISTORY (append-only, immutable)
# =============================================================================

class ValidityHistory:
    """
    Append-only history of validity verdicts.
    
    Invariants:
        - Verdicts can only be appended, never modified
        - Each verdict has a unique ID
        - History is ordered by computation time
        - Amendments create new entries, don't modify old ones
    """
    
    def __init__(self, node_id: str):
        self._node_id = node_id
        self._entries: List[ValidityVerdict] = []
        self._amendments: Dict[str, str] = {}  # verdict_id → amendment_id
    
    def append(self, verdict: ValidityVerdict) -> None:
        """Append a new verdict. Cannot modify existing."""
        self._entries.append(verdict)
    
    def amend(self, original_id: str, amendment: ValidityVerdict) -> None:
        """
        Create an amendment for an existing verdict.
        
        The original is NOT modified. A new entry is added
        and linked as an amendment.
        """
        # Verify original exists
        if not any(e.verdict_id == original_id for e in self._entries):
            raise ValueError(f"No verdict with ID {original_id}")
        
        self._entries.append(amendment)
        self._amendments[original_id] = amendment.verdict_id
    
    def latest(self) -> Optional[ValidityVerdict]:
        """Get the most recent verdict."""
        return self._entries[-1] if self._entries else None
    
    def get_by_id(self, verdict_id: str) -> Optional[ValidityVerdict]:
        """Get a specific verdict by ID."""
        return next(
            (e for e in self._entries if e.verdict_id == verdict_id),
            None
        )
    
    def all_entries(self) -> List[ValidityVerdict]:
        """Get all entries (read-only view)."""
        return list(self._entries)
    
    def __len__(self) -> int:
        return len(self._entries)


# =============================================================================
# DAG REGISTRY (the global node graph)
# =============================================================================

class DAGRegistry:
    """
    Registry of all analysis nodes.
    
    Enforces:
        - Unique node IDs
        - Dependency tracking
        - Inheritance propagation
        - No orphan nodes
    """
    
    def __init__(self):
        self._nodes: Dict[str, AnalysisNode] = {}
        self._histories: Dict[str, ValidityHistory] = {}
    
    def register(self, node: AnalysisNode) -> None:
        """Register a node in the DAG."""
        node_id = node.identity.node_id
        
        if node_id in self._nodes:
            raise ValueError(f"Node {node_id} already registered")
        
        # Verify all parents are registered
        for parent in node.parents:
            if parent.identity.node_id not in self._nodes:
                raise ValueError(
                    f"Parent {parent.identity.node_id} not registered"
                )
        
        self._nodes[node_id] = node
        self._histories[node_id] = ValidityHistory(node_id)
    
    def get(self, node_id: str) -> Optional[AnalysisNode]:
        """Get a node by ID."""
        return self._nodes.get(node_id)
    
    def get_history(self, node_id: str) -> Optional[ValidityHistory]:
        """Get validity history for a node."""
        return self._histories.get(node_id)
    
    def record_verdict(self, node_id: str, verdict: ValidityVerdict) -> None:
        """Record a verdict in the node's history."""
        history = self._histories.get(node_id)
        if history:
            history.append(verdict)
    
    def get_children(self, node_id: str) -> List[AnalysisNode]:
        """Get all nodes that depend on this node."""
        return [
            node for node in self._nodes.values()
            if node_id in node.identity.parent_ids
        ]
    
    def propagate_invalidity(self, node_id: str) -> List[str]:
        """
        When a node becomes INVALID, propagate to children.
        
        Returns list of affected node IDs.
        """
        affected = []
        children = self.get_children(node_id)
        
        for child in children:
            # Force recomputation of child validity
            child.compute_validity()
            
            if child.verdict and child.verdict.state == ValidityState.INVALID:
                affected.append(child.identity.node_id)
                # Recursive propagation
                affected.extend(self.propagate_invalidity(child.identity.node_id))
        
        return affected


# Global registry instance
_global_registry = DAGRegistry()

def get_registry() -> DAGRegistry:
    """Get the global DAG registry."""
    return _global_registry
