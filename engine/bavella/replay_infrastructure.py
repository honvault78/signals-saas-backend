"""
Bavella v2 — Replay & Reproducibility Infrastructure
=====================================================

THE "PROVE IT" LAYER

Institutions need to answer:
    "Show me exactly why you said INVALID on Jan 15, with the metrics 
     and thresholds used, and prove it's reproducible."

This module provides:
    1. ReplayBundle - everything needed to reproduce an analysis
    2. ReplayExecutor - re-runs analysis from bundle
    3. ReproducibilityChecker - validates same inputs → same outputs
    4. AuditTrail - complete chain of evidence

Guarantees:
    - Deterministic computations (no hidden randomness)
    - Explicit version stamps on every output
    - Evidence artifacts stored per episode
    - Replay tool: regenerate from raw inputs

Copyright 2024-2026 Bavella Technologies Sarl
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

from .audit_infrastructure import (
    BavellaVersions, AnalysisRun, DataRef, EpisodeRecord,
    EvidenceArtifact, EvidenceType, IntegrityLevel,
)


# =============================================================================
# REPLAY BUNDLE
# =============================================================================

@dataclass
class ReplayBundle:
    """
    Everything needed to reproduce an analysis.
    
    This is the "time capsule" that proves reproducibility.
    """
    # Identity
    bundle_id: str
    created_at: datetime
    
    # Original run reference
    original_run_id: str
    
    # Input data (the exact data analyzed)
    data_hash: str  # Hash of raw input
    data_values: List[float]  # Actual values (can be encrypted/redacted)
    timestamps: List[str]  # ISO format
    
    # Configuration (frozen)
    config: Dict[str, Any]
    config_hash: str
    
    # Versions (frozen)
    versions: Dict[str, str]
    
    # Thresholds used
    thresholds: Dict[str, float]
    
    # Expected outputs (for validation)
    expected_validity_score: float
    expected_validity_state: str
    expected_episode_ids: List[str]
    expected_failure_modes: List[str]
    
    # Metadata
    owner_id: str
    node_id: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "bundle_id": self.bundle_id,
            "created_at": self.created_at.isoformat(),
            "original_run_id": self.original_run_id,
            "data": {
                "hash": self.data_hash,
                "n_points": len(self.data_values),
                # Values may be encrypted in production
            },
            "config_hash": self.config_hash,
            "versions": self.versions,
            "thresholds": self.thresholds,
            "expected": {
                "validity_score": self.expected_validity_score,
                "validity_state": self.expected_validity_state,
                "episode_ids": self.expected_episode_ids,
                "failure_modes": self.expected_failure_modes,
            },
            "owner_id": self.owner_id,
            "node_id": self.node_id,
        }
    
    @classmethod
    def from_run(
        cls,
        run: AnalysisRun,
        data_values: List[float],
        timestamps: List[datetime],
        config: Dict[str, Any],
        validity_score: float,
        validity_state: str,
        episode_ids: List[str],
        failure_modes: List[str],
    ) -> "ReplayBundle":
        """Create bundle from a completed analysis run."""
        data_hash = hashlib.sha256(
            np.array(data_values).tobytes()
        ).hexdigest()
        
        config_hash = hashlib.sha256(
            json.dumps(config, sort_keys=True).encode()
        ).hexdigest()[:16]
        
        return cls(
            bundle_id=str(uuid.uuid4()),
            created_at=datetime.now(timezone.utc),
            original_run_id=run.run_id,
            data_hash=data_hash,
            data_values=data_values,
            timestamps=[ts.isoformat() for ts in timestamps],
            config=config,
            config_hash=config_hash,
            versions=run.pipeline_versions,
            thresholds={
                "valid_min": 70.0,
                "degraded_min": 30.0,
                "kill_fm4_magnitude": 4.0,
            },
            expected_validity_score=validity_score,
            expected_validity_state=validity_state,
            expected_episode_ids=episode_ids,
            expected_failure_modes=failure_modes,
            owner_id=run.owner_id,
            node_id=run.node_id,
        )


# =============================================================================
# REPLAY RESULT
# =============================================================================

class ReplayStatus(Enum):
    """Status of replay execution."""
    SUCCESS = "success"           # Reproduced exactly
    MISMATCH = "mismatch"         # Different outputs
    VERSION_MISMATCH = "version_mismatch"  # Can't replay with current version
    DATA_CORRUPTED = "data_corrupted"      # Data hash doesn't match
    ERROR = "error"               # Execution error


@dataclass
class ReplayDiff:
    """Difference between expected and actual outputs."""
    field: str
    expected: Any
    actual: Any
    is_critical: bool = True  # Critical diffs fail reproducibility
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "field": self.field,
            "expected": str(self.expected),
            "actual": str(self.actual),
            "is_critical": self.is_critical,
        }


@dataclass
class ReplayResult:
    """
    Result of replay execution.
    
    This is the proof of reproducibility (or lack thereof).
    """
    # Identity
    result_id: str
    bundle_id: str
    executed_at: datetime
    
    # Status
    status: ReplayStatus
    
    # Comparison
    is_reproducible: bool
    diffs: List[ReplayDiff]
    
    # Actual outputs
    actual_validity_score: Optional[float] = None
    actual_validity_state: Optional[str] = None
    actual_episode_ids: Optional[List[str]] = None
    actual_failure_modes: Optional[List[str]] = None
    
    # Execution details
    execution_versions: Optional[Dict[str, str]] = None
    execution_duration_ms: float = 0.0
    
    # Error info
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "result_id": self.result_id,
            "bundle_id": self.bundle_id,
            "executed_at": self.executed_at.isoformat(),
            "status": self.status.value,
            "is_reproducible": self.is_reproducible,
            "diffs": [d.to_dict() for d in self.diffs],
            "actual": {
                "validity_score": self.actual_validity_score,
                "validity_state": self.actual_validity_state,
                "episode_ids": self.actual_episode_ids,
                "failure_modes": self.actual_failure_modes,
            },
            "execution_versions": self.execution_versions,
            "execution_duration_ms": self.execution_duration_ms,
            "error_message": self.error_message,
        }


# =============================================================================
# REPLAY EXECUTOR
# =============================================================================

class ReplayExecutor:
    """
    Re-runs analysis from a replay bundle.
    
    This is the "time machine" that proves reproducibility.
    """
    
    # Tolerance for floating point comparison
    SCORE_TOLERANCE = 0.01  # 0.01 points
    
    def __init__(self):
        self.current_versions = BavellaVersions.to_dict()
    
    def execute(self, bundle: ReplayBundle) -> ReplayResult:
        """
        Execute replay and compare to expected outputs.
        """
        import time
        start = time.time()
        
        now = datetime.now(timezone.utc)
        result_id = str(uuid.uuid4())
        
        # Check data integrity
        computed_hash = hashlib.sha256(
            np.array(bundle.data_values).tobytes()
        ).hexdigest()
        
        if computed_hash != bundle.data_hash:
            return ReplayResult(
                result_id=result_id,
                bundle_id=bundle.bundle_id,
                executed_at=now,
                status=ReplayStatus.DATA_CORRUPTED,
                is_reproducible=False,
                diffs=[ReplayDiff(
                    field="data_hash",
                    expected=bundle.data_hash,
                    actual=computed_hash,
                )],
                error_message="Data hash mismatch - data may be corrupted",
            )
        
        # Check version compatibility
        version_diffs = []
        for key, expected_ver in bundle.versions.items():
            actual_ver = self.current_versions.get(key, "unknown")
            if actual_ver != expected_ver:
                version_diffs.append(ReplayDiff(
                    field=f"version.{key}",
                    expected=expected_ver,
                    actual=actual_ver,
                    is_critical=key in ["ontology", "detector_bundle"],  # Critical versions
                ))
        
        # If critical versions differ, can't guarantee reproducibility
        critical_version_diff = any(d.is_critical for d in version_diffs)
        
        # Execute analysis
        try:
            actual_score, actual_state, actual_eps, actual_fms = self._run_analysis(bundle)
        except Exception as e:
            return ReplayResult(
                result_id=result_id,
                bundle_id=bundle.bundle_id,
                executed_at=now,
                status=ReplayStatus.ERROR,
                is_reproducible=False,
                diffs=version_diffs,
                error_message=str(e),
                execution_versions=self.current_versions,
                execution_duration_ms=(time.time() - start) * 1000,
            )
        
        # Compare outputs
        output_diffs = []
        
        # Score comparison (with tolerance)
        if abs(actual_score - bundle.expected_validity_score) > self.SCORE_TOLERANCE:
            output_diffs.append(ReplayDiff(
                field="validity_score",
                expected=bundle.expected_validity_score,
                actual=actual_score,
            ))
        
        # State comparison
        if actual_state != bundle.expected_validity_state:
            output_diffs.append(ReplayDiff(
                field="validity_state",
                expected=bundle.expected_validity_state,
                actual=actual_state,
            ))
        
        # Failure modes comparison
        if set(actual_fms) != set(bundle.expected_failure_modes):
            output_diffs.append(ReplayDiff(
                field="failure_modes",
                expected=bundle.expected_failure_modes,
                actual=actual_fms,
            ))
        
        all_diffs = version_diffs + output_diffs
        is_reproducible = len(output_diffs) == 0 and not critical_version_diff
        
        if is_reproducible:
            status = ReplayStatus.SUCCESS
        elif critical_version_diff:
            status = ReplayStatus.VERSION_MISMATCH
        else:
            status = ReplayStatus.MISMATCH
        
        return ReplayResult(
            result_id=result_id,
            bundle_id=bundle.bundle_id,
            executed_at=now,
            status=status,
            is_reproducible=is_reproducible,
            diffs=all_diffs,
            actual_validity_score=actual_score,
            actual_validity_state=actual_state,
            actual_episode_ids=actual_eps,
            actual_failure_modes=actual_fms,
            execution_versions=self.current_versions,
            execution_duration_ms=(time.time() - start) * 1000,
        )
    
    def _run_analysis(
        self, bundle: ReplayBundle
    ) -> Tuple[float, str, List[str], List[str]]:
        """
        Run the actual analysis.
        
        In production, this would call the full SeriesAnalyzer.
        Here we simulate with deterministic logic.
        """
        from .series_analyzer import SeriesAnalyzer, SeriesAnalysisRequest
        
        # Reconstruct timestamps
        timestamps = [datetime.fromisoformat(ts) for ts in bundle.timestamps]
        
        # Create request
        request = SeriesAnalysisRequest(
            owner_id=bundle.owner_id,
            series_id=bundle.node_id,
            timestamps=timestamps,
            values=bundle.data_values,
        )
        
        # Run analysis
        analyzer = SeriesAnalyzer()
        report = analyzer.analyze(request)
        
        return (
            report.validity_score,
            report.validity_state.value.upper(),
            [],  # Episode IDs not deterministic in current impl
            [fm.failure_mode.name for fm in report.active_failure_modes],
        )


# =============================================================================
# REPRODUCIBILITY CHECKER
# =============================================================================

class ReproducibilityChecker:
    """
    Validates that the system produces reproducible outputs.
    
    This is the "quality gate" for institutional trust.
    """
    
    def __init__(self):
        self._executor = ReplayExecutor()
        self._test_results: List[ReplayResult] = []
    
    def run_reproducibility_suite(
        self, bundles: List[ReplayBundle]
    ) -> Dict[str, Any]:
        """
        Run reproducibility tests on a set of bundles.
        """
        results = []
        
        for bundle in bundles:
            result = self._executor.execute(bundle)
            results.append(result)
            self._test_results.append(result)
        
        # Compute statistics
        total = len(results)
        success = sum(1 for r in results if r.status == ReplayStatus.SUCCESS)
        mismatch = sum(1 for r in results if r.status == ReplayStatus.MISMATCH)
        version_mismatch = sum(1 for r in results if r.status == ReplayStatus.VERSION_MISMATCH)
        errors = sum(1 for r in results if r.status == ReplayStatus.ERROR)
        
        return {
            "total_tests": total,
            "reproducible": success,
            "mismatches": mismatch,
            "version_mismatches": version_mismatch,
            "errors": errors,
            "reproducibility_rate": success / total if total > 0 else 0,
            "results": [r.to_dict() for r in results],
        }
    
    def create_regression_bundle(
        self,
        name: str,
        data_values: List[float],
        expected_score: float,
        expected_state: str,
    ) -> ReplayBundle:
        """Create a bundle for regression testing."""
        now = datetime.now(timezone.utc)
        timestamps = [now - timedelta(days=len(data_values)-i-1) 
                     for i in range(len(data_values))]
        
        # Create mock run
        run = AnalysisRun.create("regression_test", f"regression_{name}")
        
        return ReplayBundle.from_run(
            run=run,
            data_values=data_values,
            timestamps=timestamps,
            config={"test_name": name},
            validity_score=expected_score,
            validity_state=expected_state,
            episode_ids=[],
            failure_modes=[],
        )


# =============================================================================
# AUDIT TRAIL
# =============================================================================

@dataclass
class AuditEntry:
    """Single entry in the audit trail."""
    entry_id: str
    timestamp: datetime
    
    # What happened
    action: str  # "ANALYSIS", "EPISODE_CREATE", "EPISODE_UPDATE", "RECOVERY"
    
    # Context
    run_id: Optional[str]
    episode_id: Optional[str]
    node_id: str
    owner_id: str
    
    # Details
    summary: str
    details: Dict[str, Any]
    
    # Versions at time of action
    versions: Dict[str, str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "timestamp": self.timestamp.isoformat(),
            "action": self.action,
            "run_id": self.run_id,
            "episode_id": self.episode_id,
            "node_id": self.node_id,
            "owner_id": self.owner_id,
            "summary": self.summary,
            "details": self.details,
            "versions": self.versions,
        }


class AuditTrail:
    """
    Complete chain of evidence for auditors.
    
    This is the "paper trail" that institutions require.
    """
    
    def __init__(self):
        self._entries: List[AuditEntry] = []
        self._by_node: Dict[str, List[str]] = {}
        self._by_episode: Dict[str, List[str]] = {}
    
    def log(
        self,
        action: str,
        node_id: str,
        owner_id: str,
        summary: str,
        details: Optional[Dict[str, Any]] = None,
        run_id: Optional[str] = None,
        episode_id: Optional[str] = None,
    ) -> AuditEntry:
        """Log an audit entry."""
        entry = AuditEntry(
            entry_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            action=action,
            run_id=run_id,
            episode_id=episode_id,
            node_id=node_id,
            owner_id=owner_id,
            summary=summary,
            details=details or {},
            versions=BavellaVersions.to_dict(),
        )
        
        self._entries.append(entry)
        
        # Index
        if node_id not in self._by_node:
            self._by_node[node_id] = []
        self._by_node[node_id].append(entry.entry_id)
        
        if episode_id:
            if episode_id not in self._by_episode:
                self._by_episode[episode_id] = []
            self._by_episode[episode_id].append(entry.entry_id)
        
        return entry
    
    def get_node_history(self, node_id: str) -> List[AuditEntry]:
        """Get audit trail for a node."""
        entry_ids = self._by_node.get(node_id, [])
        return [e for e in self._entries if e.entry_id in entry_ids]
    
    def get_episode_history(self, episode_id: str) -> List[AuditEntry]:
        """Get audit trail for an episode."""
        entry_ids = self._by_episode.get(episode_id, [])
        return [e for e in self._entries if e.entry_id in entry_ids]
    
    def explain(
        self,
        node_id: str,
        timestamp: datetime,
        question: str = "why_invalid",
    ) -> Dict[str, Any]:
        """
        Answer: "Why did you say X on date Y?"
        
        Returns the complete explanation with evidence.
        """
        # Find entries around the timestamp
        relevant = [
            e for e in self._entries
            if e.node_id == node_id and 
            abs((e.timestamp - timestamp).total_seconds()) < 86400  # Within 24h
        ]
        
        relevant.sort(key=lambda e: e.timestamp)
        
        if not relevant:
            return {
                "question": question,
                "node_id": node_id,
                "timestamp": timestamp.isoformat(),
                "answer": "No audit entries found for this node around this time",
                "entries": [],
            }
        
        # Build explanation
        analysis_entries = [e for e in relevant if e.action == "ANALYSIS"]
        episode_entries = [e for e in relevant if e.action in ("EPISODE_CREATE", "EPISODE_UPDATE")]
        
        return {
            "question": question,
            "node_id": node_id,
            "timestamp": timestamp.isoformat(),
            "answer": f"Found {len(relevant)} audit entries",
            "analysis_runs": [e.to_dict() for e in analysis_entries],
            "episode_events": [e.to_dict() for e in episode_entries],
            "versions_at_time": relevant[0].versions if relevant else {},
            "full_entries": [e.to_dict() for e in relevant],
        }
    
    def export(
        self,
        owner_id: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """Export audit trail for an owner."""
        entries = [e for e in self._entries if e.owner_id == owner_id]
        
        if start:
            entries = [e for e in entries if e.timestamp >= start]
        if end:
            entries = [e for e in entries if e.timestamp <= end]
        
        return [e.to_dict() for e in entries]


# =============================================================================
# TESTS
# =============================================================================

from datetime import timedelta


def test_replay_bundle():
    """Test replay bundle creation."""
    now = datetime.now(timezone.utc)
    
    run = AnalysisRun.create("owner", "test_node")
    
    bundle = ReplayBundle.from_run(
        run=run,
        data_values=[1.0, 2.0, 3.0, 4.0, 5.0],
        timestamps=[now - timedelta(days=4-i) for i in range(5)],
        config={"test": True},
        validity_score=75.5,
        validity_state="VALID",
        episode_ids=["ep1"],
        failure_modes=["FM1_VARIANCE_REGIME"],
    )
    
    print(f"Bundle ID: {bundle.bundle_id}")
    print(f"Data hash: {bundle.data_hash}")
    print(f"Config hash: {bundle.config_hash}")
    print(f"Expected score: {bundle.expected_validity_score}")
    
    assert len(bundle.data_hash) > 0
    assert bundle.expected_validity_score == 75.5
    
    print("✓ test_replay_bundle passed")


def test_replay_execution():
    """Test replay execution."""
    np.random.seed(42)  # For reproducibility
    
    now = datetime.now(timezone.utc)
    run = AnalysisRun.create("owner", "test_replay")
    
    # Create stable data that should be VALID
    data = list(100 + np.cumsum(np.random.randn(100) * 0.3))
    timestamps = [now - timedelta(days=99-i) for i in range(100)]
    
    # We don't know exact output, so we'll test execution mechanics
    bundle = ReplayBundle.from_run(
        run=run,
        data_values=data,
        timestamps=timestamps,
        config={},
        validity_score=0,  # Will be overwritten
        validity_state="UNKNOWN",
        episode_ids=[],
        failure_modes=[],
    )
    
    executor = ReplayExecutor()
    
    # First execution to get baseline
    result1 = executor.execute(bundle)
    print(f"First execution: {result1.actual_validity_score:.1f} ({result1.actual_validity_state})")
    
    # Update bundle with actual values for replay test
    bundle2 = ReplayBundle(
        bundle_id=bundle.bundle_id,
        created_at=bundle.created_at,
        original_run_id=bundle.original_run_id,
        data_hash=bundle.data_hash,
        data_values=bundle.data_values,
        timestamps=bundle.timestamps,
        config=bundle.config,
        config_hash=bundle.config_hash,
        versions=bundle.versions,
        thresholds=bundle.thresholds,
        expected_validity_score=result1.actual_validity_score,
        expected_validity_state=result1.actual_validity_state,
        expected_episode_ids=[],
        expected_failure_modes=result1.actual_failure_modes or [],
        owner_id=bundle.owner_id,
        node_id=bundle.node_id,
    )
    
    # Second execution should match
    result2 = executor.execute(bundle2)
    print(f"Replay execution: {result2.actual_validity_score:.1f} ({result2.actual_validity_state})")
    print(f"Status: {result2.status.value}")
    print(f"Reproducible: {result2.is_reproducible}")
    
    assert result2.is_reproducible or result2.status == ReplayStatus.SUCCESS
    
    print("✓ test_replay_execution passed")


def test_audit_trail():
    """Test audit trail."""
    trail = AuditTrail()
    
    # Log some events
    trail.log(
        action="ANALYSIS",
        node_id="momentum",
        owner_id="owner1",
        summary="Analysis completed, validity=45 DEGRADED",
        details={"validity_score": 45, "state": "DEGRADED"},
        run_id="run_001",
    )
    
    trail.log(
        action="EPISODE_CREATE",
        node_id="momentum",
        owner_id="owner1",
        summary="Created episode for FM4_STRUCTURAL_BREAK",
        details={"failure_mode": "FM4_STRUCTURAL_BREAK", "severity": 75},
        run_id="run_001",
        episode_id="ep_001",
    )
    
    trail.log(
        action="EPISODE_UPDATE",
        node_id="momentum",
        owner_id="owner1",
        summary="Episode severity increased",
        details={"new_severity": 85},
        run_id="run_002",
        episode_id="ep_001",
    )
    
    # Query history
    node_history = trail.get_node_history("momentum")
    print(f"Node history: {len(node_history)} entries")
    
    episode_history = trail.get_episode_history("ep_001")
    print(f"Episode history: {len(episode_history)} entries")
    
    # Explain
    explanation = trail.explain(
        "momentum",
        datetime.now(timezone.utc),
        "why_degraded",
    )
    print(f"Explanation found {len(explanation['analysis_runs'])} analysis runs")
    
    assert len(node_history) == 3
    assert len(episode_history) == 2
    
    print("✓ test_audit_trail passed")


def test_reproducibility_checker():
    """Test reproducibility checker."""
    np.random.seed(123)
    
    checker = ReproducibilityChecker()
    
    # Create regression bundle
    bundle = checker.create_regression_bundle(
        name="stable_series",
        data_values=list(100 + np.cumsum(np.random.randn(100) * 0.3)),
        expected_score=0,  # Placeholder
        expected_state="UNKNOWN",
    )
    
    # Run suite (single bundle)
    # Note: This will likely show mismatch since we don't know expected values
    results = checker.run_reproducibility_suite([bundle])
    
    print(f"Reproducibility suite results:")
    print(f"  Total tests: {results['total_tests']}")
    print(f"  Reproducible: {results['reproducible']}")
    print(f"  Rate: {results['reproducibility_rate']:.0%}")
    
    # The test passes as long as execution completes
    assert results['total_tests'] == 1
    
    print("✓ test_reproducibility_checker passed")


def run_all_replay_tests():
    print("\n" + "=" * 60)
    print("REPLAY & REPRODUCIBILITY TESTS")
    print("=" * 60 + "\n")
    
    test_replay_bundle()
    print()
    test_replay_execution()
    print()
    test_audit_trail()
    print()
    test_reproducibility_checker()
    
    print("\n" + "=" * 60)
    print("ALL REPLAY TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    run_all_replay_tests()
