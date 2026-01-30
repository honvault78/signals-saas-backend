"""
Bavella v2 — Audit Infrastructure
==================================

THE "NOBODY CAN CALL IT VIBES" LAYER

Every Bavella output must be:
    - Deterministic given (data, config, code_version)
    - Reproducible months later from stored artifacts
    - Explainable via structured evidence (not narrative-only)

This module defines:
    1. AnalysisRun - one execution with full version tracking
    2. DataRef - what was analyzed (court-proof)
    3. EpisodeRecord - immutable episode with lifecycle
    4. EvidenceArtifact - auditable evidence
    5. MatchResult - immutable match record
    6. TrustLedgerEntry - path-dependent trust tracking

Every "answer" references immutable records with IDs.

Copyright 2024-2026 Bavella Technologies Sarl
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, FrozenSet
import numpy as np


# =============================================================================
# VERSIONS (FROZEN - changes require version bump)
# =============================================================================

class BavellaVersions:
    """
    Central version registry. FROZEN CANON.
    
    Every output must carry these versions for reproducibility.
    """
    ONTOLOGY_VERSION = "2.1.0"           # FM definitions, thresholds
    DETECTOR_BUNDLE_VERSION = "2.1.0"    # Detector implementations
    FINGERPRINT_VERSION = "1.0.0"        # Fingerprint schema
    MATCHER_VERSION = "1.0.0"            # Similarity algorithm
    CONFLICT_RESOLUTION_VERSION = "1.0.0"  # Precedence graph
    TRUST_MODEL_VERSION = "1.0.0"        # Epistemic cost table
    SEVERITY_CURVE_VERSION = "1.0.0"     # Curve descriptor algorithm
    RECOVERY_GATES_VERSION = "1.0.0"     # Recovery gate definitions
    ROUTER_VERSION = "1.0.0"             # Query routing
    
    CODE_COMMIT = "dev"  # Set from git in production
    
    @classmethod
    def to_dict(cls) -> Dict[str, str]:
        return {
            "ontology": cls.ONTOLOGY_VERSION,
            "detector_bundle": cls.DETECTOR_BUNDLE_VERSION,
            "fingerprint": cls.FINGERPRINT_VERSION,
            "matcher": cls.MATCHER_VERSION,
            "conflict_resolution": cls.CONFLICT_RESOLUTION_VERSION,
            "trust_model": cls.TRUST_MODEL_VERSION,
            "severity_curve": cls.SEVERITY_CURVE_VERSION,
            "recovery_gates": cls.RECOVERY_GATES_VERSION,
            "router": cls.ROUTER_VERSION,
            "code_commit": cls.CODE_COMMIT,
        }
    
    @classmethod
    def compute_config_hash(cls, config: Dict[str, Any]) -> str:
        """Compute deterministic hash of configuration."""
        serialized = json.dumps(config, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()[:16]


# =============================================================================
# ENUMS
# =============================================================================

class NodeType(Enum):
    """Type of analysis node."""
    SERIES = "series"
    PAIR = "pair"
    FACTOR = "factor"
    MODEL = "model"
    PIPELINE_NODE = "pipeline_node"


class EpisodeStatus(Enum):
    """Episode lifecycle status."""
    ACTIVE = "active"
    RESOLVED = "resolved"


class RecoveryType(Enum):
    """How an episode was resolved."""
    FULL = "full"                     # Complete recovery
    PARTIAL = "partial"               # Partial recovery, some damage
    REBASELINE_REQUIRED = "rebaseline"  # Must rebaseline
    NO_RECOVERY = "no_recovery"       # Did not recover


class FailureModeRole(Enum):
    """Role of a failure mode within an episode."""
    ROOT = "root"       # Root cause
    SYMPTOM = "symptom"  # Downstream symptom
    INDEPENDENT = "independent"  # Independent co-occurrence


class IntegrityLevel(Enum):
    """Analysis integrity level."""
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


class EvidenceType(Enum):
    """Type of evidence artifact."""
    DETECTOR_TRIGGER = "detector_trigger"
    WINDOW_STATS = "window_stats"
    CHANGEPOINT = "changepoint"
    RESIDUAL_TEST = "residual_test"
    CORRELATION_SHIFT = "correlation_shift"
    DRIFT_METRIC = "drift_metric"
    RECOVERY_GATE = "recovery_gate"
    CAUSAL_ORDERING = "causal_ordering"
    MANUAL_NOTE = "manual_note"


class TrustEventType(Enum):
    """Type of trust ledger event."""
    EPISODE_START = "episode_start"
    EPISODE_ESCALATION = "episode_escalation"
    IRREVERSIBLE_FLAG = "irreversible_flag"
    RECOVERY = "recovery"
    REBASELINE = "rebaseline"
    DECAY = "decay"


# =============================================================================
# DATA REFERENCE (what was analyzed - court-proof)
# =============================================================================

@dataclass(frozen=True)
class DataRef:
    """
    Reference to the exact data that was analyzed.
    
    This makes analysis reproducible and auditable.
    """
    # Hash of actual numeric payload after normalization
    data_hash: str
    
    # Source information
    source_type: str  # UPLOAD, API, DB, VENDOR, COMPUTED
    source_details: str  # Opaque string, can be redacted
    
    # How data was preprocessed
    normalization_spec: str  # e.g., "log_returns", "pct_change", "raw"
    
    # Data quality
    n_points: int
    n_missing: int
    missingness_pct: float
    sampling_frequency: str  # "1D", "1H", "tick", etc.
    sampling_irregularity: float  # 0 = perfectly regular
    
    # Time bounds
    start_time: datetime
    end_time: datetime
    
    # Schema
    schema_version: str = "1.0.0"
    
    @classmethod
    def from_array(
        cls,
        data: np.ndarray,
        source_type: str = "UPLOAD",
        source_details: str = "",
        normalization_spec: str = "raw",
        timestamps: Optional[List[datetime]] = None,
    ) -> "DataRef":
        """Create DataRef from numpy array."""
        # Compute hash of normalized data
        data_bytes = data.tobytes()
        data_hash = hashlib.sha256(data_bytes).hexdigest()[:32]
        
        # Count missing
        n_missing = int(np.isnan(data).sum())
        n_points = len(data)
        
        # Time bounds
        now = datetime.now(timezone.utc)
        if timestamps:
            start_time = timestamps[0]
            end_time = timestamps[-1]
            # Compute irregularity
            if len(timestamps) > 1:
                diffs = [(timestamps[i+1] - timestamps[i]).total_seconds() 
                         for i in range(len(timestamps)-1)]
                mean_diff = np.mean(diffs)
                std_diff = np.std(diffs)
                irregularity = std_diff / mean_diff if mean_diff > 0 else 0
            else:
                irregularity = 0.0
        else:
            start_time = now
            end_time = now
            irregularity = 0.0
        
        return cls(
            data_hash=data_hash,
            source_type=source_type,
            source_details=source_details,
            normalization_spec=normalization_spec,
            n_points=n_points,
            n_missing=n_missing,
            missingness_pct=round(n_missing / n_points * 100, 2) if n_points > 0 else 0,
            sampling_frequency="inferred",
            sampling_irregularity=round(irregularity, 4),
            start_time=start_time,
            end_time=end_time,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "data_hash": self.data_hash,
            "source_type": self.source_type,
            "normalization_spec": self.normalization_spec,
            "n_points": self.n_points,
            "n_missing": self.n_missing,
            "missingness_pct": self.missingness_pct,
            "sampling_frequency": self.sampling_frequency,
            "sampling_irregularity": self.sampling_irregularity,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "schema_version": self.schema_version,
        }


# =============================================================================
# EVIDENCE ARTIFACT
# =============================================================================

@dataclass(frozen=True)
class EvidenceArtifact:
    """
    Auditable evidence supporting a claim.
    
    Every claim must have evidence refs. If not, it's not institutional-grade.
    """
    evidence_id: str
    evidence_type: EvidenceType
    created_at: datetime
    
    # Links
    run_id: Optional[str] = None
    episode_id: Optional[str] = None
    
    # Content
    summary: str = ""  # Human readable
    payload: Dict[str, Any] = field(default_factory=dict)
    payload_hash: str = ""
    
    # Redaction
    redaction_level: str = "NONE"  # NONE, REDACTED, AGGREGATED
    
    # Source
    source_refs: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def create(
        cls,
        evidence_type: EvidenceType,
        summary: str,
        payload: Dict[str, Any],
        run_id: Optional[str] = None,
        episode_id: Optional[str] = None,
    ) -> "EvidenceArtifact":
        """Create new evidence artifact."""
        evidence_id = str(uuid.uuid4())
        payload_hash = hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode()
        ).hexdigest()[:16]
        
        return cls(
            evidence_id=evidence_id,
            evidence_type=evidence_type,
            created_at=datetime.now(timezone.utc),
            run_id=run_id,
            episode_id=episode_id,
            summary=summary,
            payload=payload,
            payload_hash=payload_hash,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "evidence_type": self.evidence_type.value,
            "created_at": self.created_at.isoformat(),
            "run_id": self.run_id,
            "episode_id": self.episode_id,
            "summary": self.summary,
            "payload_hash": self.payload_hash,
            "redaction_level": self.redaction_level,
        }


# =============================================================================
# FAILURE MODE ENTRY (within episode)
# =============================================================================

@dataclass(frozen=True)
class FailureModeEntry:
    """A failure mode's participation in an episode."""
    fm_code: str  # e.g., "FM4_STRUCTURAL_BREAK"
    role: FailureModeRole
    
    # Timing
    onset_time: datetime
    offset_time: Optional[datetime] = None  # None if still active
    
    # Severity
    peak_severity: float = 0.0
    peak_time: Optional[datetime] = None
    end_severity: float = 0.0
    
    # Confidence
    detection_confidence: float = 0.0
    
    # Evidence
    evidence_ids: Tuple[str, ...] = ()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "fm_code": self.fm_code,
            "role": self.role.value,
            "onset_time": self.onset_time.isoformat(),
            "offset_time": self.offset_time.isoformat() if self.offset_time else None,
            "peak_severity": self.peak_severity,
            "peak_time": self.peak_time.isoformat() if self.peak_time else None,
            "end_severity": self.end_severity,
            "detection_confidence": self.detection_confidence,
            "evidence_ids": list(self.evidence_ids),
        }


# =============================================================================
# CAUSAL ORDERING
# =============================================================================

@dataclass(frozen=True)
class CausalOrdering:
    """
    Causal structure of failure modes within an episode.
    
    This is what makes attribution defensible.
    """
    # FMs ordered by onset time
    onset_ranked_list: Tuple[str, ...]  # FM codes in order of appearance
    
    # Root cause
    root_cause_fm: str
    
    # Dominance relationships that fired
    dominance_edges: Tuple[Tuple[str, str], ...]  # (dominator, dominated)
    
    # Evidence for ordering
    ordering_evidence_id: Optional[str] = None
    
    # Confidence in causal structure
    causal_confidence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "onset_ranked_list": list(self.onset_ranked_list),
            "root_cause_fm": self.root_cause_fm,
            "dominance_edges": [list(e) for e in self.dominance_edges],
            "ordering_evidence_id": self.ordering_evidence_id,
            "causal_confidence": self.causal_confidence,
        }


# =============================================================================
# RECOVERY GATE
# =============================================================================

@dataclass(frozen=True)
class RecoveryGate:
    """
    A measurable condition for recovery.
    
    Gates must be versioned and deterministic.
    """
    gate_id: str
    gate_name: str
    gate_version: str
    
    # Condition
    metric_name: str  # e.g., "validity_score", "severity", "consecutive_clean"
    operator: str     # ">=", "<=", "==", ">"
    threshold: float
    
    # Result
    passed: bool
    actual_value: float
    evaluated_at: datetime
    
    # Evidence
    evidence_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "gate_id": self.gate_id,
            "gate_name": self.gate_name,
            "gate_version": self.gate_version,
            "metric_name": self.metric_name,
            "operator": self.operator,
            "threshold": self.threshold,
            "passed": self.passed,
            "actual_value": self.actual_value,
            "evaluated_at": self.evaluated_at.isoformat(),
            "evidence_id": self.evidence_id,
        }


# =============================================================================
# EPISODE RESOLUTION
# =============================================================================

@dataclass(frozen=True)
class EpisodeResolution:
    """
    How an episode was resolved.
    
    Only present for RESOLVED episodes.
    """
    recovery_type: RecoveryType
    recovery_time: datetime
    
    # Gates that were evaluated
    gates_passed: Tuple[RecoveryGate, ...]
    gates_failed: Tuple[RecoveryGate, ...]
    
    # Post-recovery state
    post_recovery_validity: float
    post_recovery_baseline_id: Optional[str] = None
    
    # Evidence
    resolution_evidence_ids: Tuple[str, ...] = ()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "recovery_type": self.recovery_type.value,
            "recovery_time": self.recovery_time.isoformat(),
            "gates_passed": [g.to_dict() for g in self.gates_passed],
            "gates_failed": [g.to_dict() for g in self.gates_failed],
            "post_recovery_validity": self.post_recovery_validity,
            "post_recovery_baseline_id": self.post_recovery_baseline_id,
            "resolution_evidence_ids": list(self.resolution_evidence_ids),
        }


# =============================================================================
# EPISODE RECORD (immutable backbone)
# =============================================================================

@dataclass
class EpisodeRecord:
    """
    Immutable episode record - the backbone of institutional memory.
    
    Episodes cannot change retrospectively except by appending
    a new immutable revision (event-sourcing).
    """
    # Identity
    episode_id: str
    revision: int = 1  # Increments on updates
    
    # Ownership
    owner_id: str = ""
    node_id: str = ""
    node_type: NodeType = NodeType.SERIES
    
    # Status
    status: EpisodeStatus = EpisodeStatus.ACTIVE
    
    # Timing
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: Optional[datetime] = None
    duration_hours: float = 0.0
    
    # Failure modes with full detail
    failure_modes: List[FailureModeEntry] = field(default_factory=list)
    root_cause_fm: Optional[str] = None
    
    # Causal structure
    causal_ordering: Optional[CausalOrdering] = None
    
    # Severity curve (see severity_curve.py for full descriptor)
    severity_curve_descriptor: Dict[str, Any] = field(default_factory=dict)
    
    # Flags
    irreversibility_flag: bool = False
    rebaseline_required: bool = False
    
    # Trust damage
    trust_penalty_applied: float = 0.0
    trust_penalty_reason: str = ""
    trust_penalty_persistence: str = "PERSISTS"  # DECAYS or PERSISTS
    
    # Resolution (only if RESOLVED)
    resolution: Optional[EpisodeResolution] = None
    
    # Evidence
    evidence_ids: List[str] = field(default_factory=list)
    
    # Versions (for reproducibility)
    ontology_version: str = BavellaVersions.ONTOLOGY_VERSION
    fingerprint_version: str = BavellaVersions.FINGERPRINT_VERSION
    trust_model_version: str = BavellaVersions.TRUST_MODEL_VERSION
    
    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def __post_init__(self):
        # Validate invariants
        if self.root_cause_fm:
            fm_codes = [fm.fm_code for fm in self.failure_modes]
            if self.root_cause_fm not in fm_codes:
                raise ValueError(f"root_cause_fm {self.root_cause_fm} not in failure_modes")
        
        if self.resolution:
            if self.resolution.recovery_type == RecoveryType.REBASELINE_REQUIRED:
                if not self.rebaseline_required:
                    raise ValueError("recovery_type=REBASELINE_REQUIRED requires rebaseline_required=True")
    
    @classmethod
    def create(
        cls,
        owner_id: str,
        node_id: str,
        node_type: NodeType = NodeType.SERIES,
    ) -> "EpisodeRecord":
        """Create new episode."""
        return cls(
            episode_id=str(uuid.uuid4()),
            owner_id=owner_id,
            node_id=node_id,
            node_type=node_type,
        )
    
    def add_failure_mode(self, entry: FailureModeEntry) -> None:
        """Add a failure mode to the episode."""
        self.failure_modes.append(entry)
        self.revision += 1
        self.updated_at = datetime.now(timezone.utc)
    
    def set_causal_ordering(self, ordering: CausalOrdering) -> None:
        """Set causal ordering."""
        self.causal_ordering = ordering
        self.root_cause_fm = ordering.root_cause_fm
        self.revision += 1
        self.updated_at = datetime.now(timezone.utc)
    
    def resolve(self, resolution: EpisodeResolution) -> None:
        """Resolve the episode."""
        self.status = EpisodeStatus.RESOLVED
        self.resolution = resolution
        self.end_time = resolution.recovery_time
        
        if self.start_time:
            delta = resolution.recovery_time - self.start_time
            self.duration_hours = delta.total_seconds() / 3600
        
        self.revision += 1
        self.updated_at = datetime.now(timezone.utc)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "revision": self.revision,
            "owner_id": self.owner_id,
            "node_id": self.node_id,
            "node_type": self.node_type.value,
            "status": self.status.value,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_hours": self.duration_hours,
            "failure_modes": [fm.to_dict() for fm in self.failure_modes],
            "root_cause_fm": self.root_cause_fm,
            "causal_ordering": self.causal_ordering.to_dict() if self.causal_ordering else None,
            "severity_curve_descriptor": self.severity_curve_descriptor,
            "irreversibility_flag": self.irreversibility_flag,
            "rebaseline_required": self.rebaseline_required,
            "trust_penalty_applied": self.trust_penalty_applied,
            "trust_penalty_persistence": self.trust_penalty_persistence,
            "resolution": self.resolution.to_dict() if self.resolution else None,
            "evidence_ids": self.evidence_ids,
            "versions": {
                "ontology": self.ontology_version,
                "fingerprint": self.fingerprint_version,
                "trust_model": self.trust_model_version,
            },
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


# =============================================================================
# TRUST LEDGER ENTRY
# =============================================================================

@dataclass(frozen=True)
class TrustLedgerEntry:
    """
    Entry in the trust penalty ledger.
    
    Makes path-dependent trust reconstructible.
    """
    entry_id: str
    timestamp: datetime
    
    # Context
    owner_id: str
    node_id: str
    
    # Event
    event_type: TrustEventType
    
    # Trust change
    trust_penalty_delta: float  # Can be negative (reduction)
    trust_penalty_level: float  # Level after this event
    
    # Links
    episode_id: Optional[str] = None
    
    # Explanation
    reason_code: str = ""
    reason_detail: str = ""
    
    # Versions
    trust_model_version: str = BavellaVersions.TRUST_MODEL_VERSION
    config_hash: str = ""
    
    @classmethod
    def create(
        cls,
        owner_id: str,
        node_id: str,
        event_type: TrustEventType,
        delta: float,
        new_level: float,
        reason_code: str,
        episode_id: Optional[str] = None,
    ) -> "TrustLedgerEntry":
        return cls(
            entry_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            owner_id=owner_id,
            node_id=node_id,
            event_type=event_type,
            trust_penalty_delta=delta,
            trust_penalty_level=new_level,
            episode_id=episode_id,
            reason_code=reason_code,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "timestamp": self.timestamp.isoformat(),
            "owner_id": self.owner_id,
            "node_id": self.node_id,
            "event_type": self.event_type.value,
            "trust_penalty_delta": self.trust_penalty_delta,
            "trust_penalty_level": self.trust_penalty_level,
            "episode_id": self.episode_id,
            "reason_code": self.reason_code,
            "trust_model_version": self.trust_model_version,
        }


# =============================================================================
# ANALYSIS RUN
# =============================================================================

@dataclass
class AnalysisRun:
    """
    One execution of analysis.
    
    Complete record for reproducibility.
    """
    run_id: str
    created_at: datetime
    
    # Context
    owner_id: str
    node_id: str
    node_type: NodeType
    
    # Request
    request_context: Dict[str, Any] = field(default_factory=dict)  # UI/API/LLM, query_type
    
    # Data
    data_ref: Optional[DataRef] = None
    
    # Analysis window
    analysis_window_start: Optional[datetime] = None
    analysis_window_end: Optional[datetime] = None
    
    # Versions (full)
    pipeline_versions: Dict[str, str] = field(default_factory=dict)
    config_hash: str = ""
    code_commit: str = BavellaVersions.CODE_COMMIT
    runtime_env: str = ""
    
    # Integrity
    integrity: IntegrityLevel = IntegrityLevel.PASS
    integrity_reasons: List[str] = field(default_factory=list)
    
    # Outputs
    report_id: Optional[str] = None
    episode_ids_created: List[str] = field(default_factory=list)
    episode_ids_updated: List[str] = field(default_factory=list)
    match_result_ids: List[str] = field(default_factory=list)
    evidence_ids: List[str] = field(default_factory=list)
    
    # Duration
    duration_ms: float = 0.0
    
    @classmethod
    def create(
        cls,
        owner_id: str,
        node_id: str,
        node_type: NodeType = NodeType.SERIES,
        request_context: Optional[Dict[str, Any]] = None,
    ) -> "AnalysisRun":
        import sys
        
        return cls(
            run_id=str(uuid.uuid4()),
            created_at=datetime.now(timezone.utc),
            owner_id=owner_id,
            node_id=node_id,
            node_type=node_type,
            request_context=request_context or {},
            pipeline_versions=BavellaVersions.to_dict(),
            runtime_env=f"python_{sys.version_info.major}.{sys.version_info.minor}",
        )
    
    def set_data_ref(self, data_ref: DataRef) -> None:
        self.data_ref = data_ref
        self.analysis_window_start = data_ref.start_time
        self.analysis_window_end = data_ref.end_time
    
    def add_integrity_issue(self, reason: str, level: IntegrityLevel = IntegrityLevel.WARN) -> None:
        self.integrity_reasons.append(reason)
        if level == IntegrityLevel.FAIL:
            self.integrity = IntegrityLevel.FAIL
        elif level == IntegrityLevel.WARN and self.integrity != IntegrityLevel.FAIL:
            self.integrity = IntegrityLevel.WARN
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "created_at": self.created_at.isoformat(),
            "owner_id": self.owner_id,
            "node_id": self.node_id,
            "node_type": self.node_type.value,
            "request_context": self.request_context,
            "data_ref": self.data_ref.to_dict() if self.data_ref else None,
            "analysis_window": {
                "start": self.analysis_window_start.isoformat() if self.analysis_window_start else None,
                "end": self.analysis_window_end.isoformat() if self.analysis_window_end else None,
            },
            "pipeline_versions": self.pipeline_versions,
            "config_hash": self.config_hash,
            "code_commit": self.code_commit,
            "runtime_env": self.runtime_env,
            "integrity": self.integrity.value,
            "integrity_reasons": self.integrity_reasons,
            "outputs": {
                "report_id": self.report_id,
                "episode_ids_created": self.episode_ids_created,
                "episode_ids_updated": self.episode_ids_updated,
                "match_result_ids": self.match_result_ids,
                "evidence_ids": self.evidence_ids,
            },
            "duration_ms": self.duration_ms,
        }


# =============================================================================
# TESTS
# =============================================================================

def test_data_ref():
    """Test DataRef creation."""
    data = np.array([1.0, 2.0, 3.0, np.nan, 5.0])
    ref = DataRef.from_array(data, source_type="TEST")
    
    assert ref.n_points == 5
    assert ref.n_missing == 1
    assert ref.missingness_pct == 20.0
    assert len(ref.data_hash) == 32
    
    print(f"DataRef: {ref.to_dict()}")
    print("✓ test_data_ref passed")


def test_episode_record():
    """Test EpisodeRecord creation and updates."""
    now = datetime.now(timezone.utc)
    
    episode = EpisodeRecord.create("owner", "node")
    
    # Add failure mode
    fm_entry = FailureModeEntry(
        fm_code="FM4_STRUCTURAL_BREAK",
        role=FailureModeRole.ROOT,
        onset_time=now,
        peak_severity=85.0,
        detection_confidence=0.92,
    )
    episode.add_failure_mode(fm_entry)
    
    # Set causal ordering
    ordering = CausalOrdering(
        onset_ranked_list=("FM4_STRUCTURAL_BREAK",),
        root_cause_fm="FM4_STRUCTURAL_BREAK",
        dominance_edges=(),
        causal_confidence=0.95,
    )
    episode.set_causal_ordering(ordering)
    
    assert episode.revision == 3  # Initial + 2 updates
    assert episode.root_cause_fm == "FM4_STRUCTURAL_BREAK"
    
    print(f"Episode ID: {episode.episode_id}")
    print(f"Revision: {episode.revision}")
    print("✓ test_episode_record passed")


def test_trust_ledger():
    """Test trust ledger entries."""
    entry = TrustLedgerEntry.create(
        owner_id="owner",
        node_id="node",
        event_type=TrustEventType.EPISODE_START,
        delta=15.0,
        new_level=15.0,
        reason_code="FM4_STRUCTURAL_BREAK",
        episode_id="ep_123",
    )
    
    assert entry.trust_penalty_delta == 15.0
    assert entry.trust_penalty_level == 15.0
    
    print(f"Ledger entry: {entry.to_dict()}")
    print("✓ test_trust_ledger passed")


def test_analysis_run():
    """Test AnalysisRun creation."""
    run = AnalysisRun.create("owner", "node", NodeType.FACTOR)
    
    # Add data ref
    data = np.random.randn(100)
    ref = DataRef.from_array(data)
    run.set_data_ref(ref)
    
    # Add integrity issue
    run.add_integrity_issue("FM3 detector failed", IntegrityLevel.WARN)
    
    assert run.integrity == IntegrityLevel.WARN
    assert len(run.integrity_reasons) == 1
    assert "ontology" in run.pipeline_versions
    
    print(f"Run: {run.run_id}")
    print(f"Versions: {run.pipeline_versions}")
    print("✓ test_analysis_run passed")


def run_all_audit_tests():
    print("\n" + "=" * 60)
    print("AUDIT INFRASTRUCTURE TESTS")
    print("=" * 60 + "\n")
    
    test_data_ref()
    print()
    test_episode_record()
    print()
    test_trust_ledger()
    print()
    test_analysis_run()
    
    print("\n" + "=" * 60)
    print("ALL AUDIT TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    run_all_audit_tests()
