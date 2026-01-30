"""
Bavella v2 — Build-Breaking Invariant Tests
=============================================

These tests enforce the FROZEN CANON.

If any of these fail, the build MUST fail.
These are not "nice to have" - they are the product.

Invariants tested:
    1. INVALID ⇒ no predictions/metrics in API serialization
    2. child_validity ≤ min(parent_validity) always
    3. confidence cannot INCREASE state in uncertainty zone
    4. episodes preserve first_seen_at across consecutive detections
    5. delayed ground truth cannot retroactively rewrite historical states
    6. attribution sums to 100% when score < 100
    7. kill switches override all scoring
    8. tenant isolation is absolute

Copyright 2024-2026 Bavella Technologies Sarl
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional

# Import modules
from .core import (
    ValidityState, ValidityVerdict, FailureMode, FailureSignal,
    GovernedValue, GovernorRefusal, NodeIdentity, Thresholds,
)
from .governor import (
    Governor, OutputLevel, OutputRequest, OutputDecision,
    GovernedResponseBuilder,
)
from .confidence_governance import (
    ConfidenceGovernanceRules, ConfidenceGovernedVerdict,
    apply_confidence_governance,
)
from .persistence_postgres import (
    InMemoryEpisodeStore, PersistenceConfig, EpisodeState,
)


# =============================================================================
# TEST RUNNER
# =============================================================================

class InvariantTestRunner:
    """
    Test runner that tracks pass/fail and exits with error on any failure.
    """
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.failures: List[str] = []
    
    def run(self, test_func):
        """Run a test function."""
        name = test_func.__name__
        try:
            test_func()
            self.passed += 1
            print(f"  ✓ {name}")
        except AssertionError as e:
            self.failed += 1
            self.failures.append(f"{name}: {e}")
            print(f"  ✗ {name}: {e}")
        except Exception as e:
            self.failed += 1
            self.failures.append(f"{name}: ERROR - {e}")
            print(f"  ✗ {name}: ERROR - {e}")
    
    def summary(self) -> bool:
        """Print summary and return True if all passed."""
        print(f"\nResults: {self.passed} passed, {self.failed} failed")
        
        if self.failures:
            print("\nFAILURES:")
            for f in self.failures:
                print(f"  - {f}")
        
        return self.failed == 0


# =============================================================================
# INVARIANT 1: INVALID ⇒ NO PREDICTIONS/METRICS IN SERIALIZATION
# =============================================================================

def test_invalid_suppresses_predictions():
    """
    INVARIANT: When validity state is INVALID, the Governor MUST refuse
    to emit any L1 (metrics) or L2 (predictions) outputs.
    
    This is not optional. This is the product.
    """
    governor = Governor()
    
    # Register an INVALID verdict
    verdict = ValidityVerdict(
        score=15.0,
        state=ValidityState.INVALID,
    )
    governor.register_verdict("node_1", verdict)
    
    # Attempt to emit a prediction (L2)
    decision = governor.request_emission(OutputRequest(
        value={"prediction": [1, 2, 3]},
        output_level=OutputLevel.L2_VALID_ONLY,
        description="Prediction",
        source_node_id="node_1",
    ))
    
    assert not decision.permitted, "INVALID must refuse L2 predictions"
    assert decision.value is None, "Refused emission must have no value"
    
    # Attempt to emit a metric (L1)
    decision = governor.request_emission(OutputRequest(
        value={"mean": 42.5},
        output_level=OutputLevel.L1_VALID_OR_DEGRADED,
        description="Metric",
        source_node_id="node_1",
    ))
    
    assert not decision.permitted, "INVALID must refuse L1 metrics"


def test_invalid_governed_value_raises():
    """
    INVARIANT: GovernedValue.get() MUST raise GovernorRefusal when INVALID.
    
    There is no .force() or .get_anyway(). INVALID means NO VALUE.
    """
    verdict = ValidityVerdict(score=10.0, state=ValidityState.INVALID)
    identity = NodeIdentity(
        node_id="test",
        node_type="test",
        created_at=datetime.now(timezone.utc),
        content_hash="abc123",
    )
    
    governed = GovernedValue(
        value={"secret": "data"},
        verdict=verdict,
        node_identity=identity,
    )
    
    # get() MUST raise
    raised = False
    try:
        governed.get()
    except GovernorRefusal:
        raised = True
    
    assert raised, "GovernedValue.get() must raise for INVALID"
    
    # Verify no backdoor methods exist
    assert not hasattr(governed, 'force'), "No force() method allowed"
    assert not hasattr(governed, 'get_anyway'), "No get_anyway() method allowed"
    assert not hasattr(governed, 'get_unsafe'), "No get_unsafe() method allowed"


def test_response_builder_suppresses_invalid():
    """
    INVARIANT: GovernedResponseBuilder must suppress fields when INVALID.
    """
    governor = Governor()
    
    verdict = ValidityVerdict(score=10.0, state=ValidityState.INVALID)
    governor.register_verdict("node_1", verdict)
    
    builder = GovernedResponseBuilder(governor)
    response = (
        builder
        .add_always("validity_score", "node_1", 10.0)  # L0 - should appear
        .add_metric("mean", "node_1", 42.5)            # L1 - should be suppressed
        .add_prediction("forecast", "node_1", [1,2,3]) # L2 - should be suppressed
        .build()
    )
    
    assert "validity_score" in response, "L0 must appear even when INVALID"
    assert "mean" not in response, "L1 must be suppressed when INVALID"
    assert "forecast" not in response, "L2 must be suppressed when INVALID"
    assert "_suppressed" in response, "Must indicate what was suppressed"


# =============================================================================
# INVARIANT 2: child_validity ≤ min(parent_validity)
# =============================================================================

def test_inheritance_hard_min():
    """
    INVARIANT: Derived validity ≤ min(upstream validity).
    
    This is a HARD constraint, not a soft influence.
    """
    from test_invariants import MockNode
    
    # Parent with degraded validity
    parent = MockNode("parent")
    parent._forced_signals = [
        FailureSignal(
            failure_mode=FailureMode.FM1_VARIANCE_REGIME,
            severity=100,
            confidence=1.0,
            explanation="Severe",
        ),
        FailureSignal(
            failure_mode=FailureMode.FM2_MEAN_DRIFT,
            severity=100,
            confidence=1.0,
            explanation="Severe",
        ),
    ]
    parent.compute_validity()
    parent_score = parent.verdict.score
    
    # Child with no issues
    child = MockNode("child", parent_nodes=[parent])
    child._forced_signals = []
    child.compute_validity()
    
    # Child MUST be capped at parent's score
    assert child.verdict.score <= parent_score, \
        f"Child ({child.verdict.score}) must be ≤ parent ({parent_score})"
    
    assert child.verdict.inherited_from is not None, \
        "Child must record inheritance source"


def test_invalid_parent_kills_child():
    """
    INVARIANT: If ANY parent is INVALID, child MUST be INVALID.
    """
    from test_invariants import MockNode
    
    # INVALID parent (via kill switch)
    parent = MockNode("parent")
    parent._forced_signals = [
        FailureSignal(
            failure_mode=FailureMode.FM4_STRUCTURAL_BREAK,
            severity=100,
            confidence=1.0,
            triggers_kill=True,
            kill_reason="Massive break",
            explanation="Kill",
        )
    ]
    parent.compute_validity()
    assert parent.verdict.state == ValidityState.INVALID
    
    # Child with no issues
    child = MockNode("child", parent_nodes=[parent])
    child._forced_signals = []
    child.compute_validity()
    
    # Child MUST be INVALID
    assert child.verdict.state == ValidityState.INVALID, \
        "Child must be INVALID when parent is INVALID"


# =============================================================================
# INVARIANT 3: confidence cannot INCREASE state
# =============================================================================

def test_confidence_cannot_improve_state():
    """
    INVARIANT: Low confidence can ONLY downgrade state, never upgrade.
    
    Specifically:
        - Low confidence + borderline VALID → DEGRADED (conservative)
        - High confidence + DEGRADED → still DEGRADED (cannot upgrade)
    """
    # Test 1: Low confidence downgrades borderline VALID
    verdict_borderline = ValidityVerdict(score=72.0, state=ValidityState.VALID)
    governed = apply_confidence_governance(verdict_borderline, validity_confidence=0.5)
    
    assert governed.governed_state == ValidityState.DEGRADED, \
        "Low confidence must downgrade borderline VALID to DEGRADED"
    
    # Test 2: High confidence CANNOT upgrade DEGRADED to VALID
    verdict_degraded = ValidityVerdict(score=50.0, state=ValidityState.DEGRADED)
    governed_high = apply_confidence_governance(verdict_degraded, validity_confidence=0.99)
    
    assert governed_high.governed_state == ValidityState.DEGRADED, \
        "High confidence must NOT upgrade DEGRADED to VALID"
    
    # Test 3: High confidence CANNOT upgrade INVALID
    verdict_invalid = ValidityVerdict(score=20.0, state=ValidityState.INVALID)
    governed_invalid = apply_confidence_governance(verdict_invalid, validity_confidence=0.99)
    
    assert governed_invalid.governed_state == ValidityState.INVALID, \
        "High confidence must NOT upgrade INVALID"


def test_confidence_governance_is_recorded():
    """
    INVARIANT: Confidence governance actions must be recorded in verdict.
    """
    verdict = ValidityVerdict(score=72.0, state=ValidityState.VALID)
    governed = apply_confidence_governance(verdict, validity_confidence=0.5)
    
    # Must record what happened
    assert governed.was_downgraded == True
    assert governed.downgrade_reason is not None
    assert "uncertainty" in governed.downgrade_reason.lower() or "confidence" in governed.downgrade_reason.lower()
    assert governed.rules_version == ConfidenceGovernanceRules.VERSION


# =============================================================================
# INVARIANT 4: episodes preserve first_seen_at
# =============================================================================

def test_episode_first_seen_immutable():
    """
    INVARIANT: first_seen_at MUST NOT change across consecutive detections.
    
    This is what makes causal ordering REAL.
    """
    import time
    
    store = InMemoryEpisodeStore()
    
    # First detection
    ep1 = store.record_detection("owner", "node", FailureSignal(
        failure_mode=FailureMode.FM1_VARIANCE_REGIME,
        severity=50,
        confidence=0.9,
        explanation="First",
    ))
    
    first_seen = ep1.first_seen_at
    episode_id = ep1.episode_id
    
    time.sleep(0.01)
    
    # Many updates
    for i in range(10):
        ep = store.record_detection("owner", "node", FailureSignal(
            failure_mode=FailureMode.FM1_VARIANCE_REGIME,
            severity=50 + i,
            confidence=0.9,
            explanation=f"Update {i}",
        ))
    
    # first_seen_at MUST be unchanged
    assert ep.first_seen_at == first_seen, \
        f"first_seen_at changed from {first_seen} to {ep.first_seen_at}"
    
    # Same episode
    assert ep.episode_id == episode_id, "Must be same episode"
    
    # last_seen_at should have changed
    assert ep.last_seen_at > first_seen, "last_seen_at should advance"


def test_episode_hysteresis_prevents_false_recovery():
    """
    INVARIANT: A single clean detection must NOT close an episode.
    
    Requires K consecutive clean detections (default K=3).
    """
    config = PersistenceConfig(
        consecutive_clean_to_close=3,
        severity_close_threshold=5.0,
    )
    store = InMemoryEpisodeStore(config)
    
    # Open episode
    ep1 = store.record_detection("owner", "node", FailureSignal(
        failure_mode=FailureMode.FM1_VARIANCE_REGIME,
        severity=50,
        confidence=0.9,
        explanation="Open",
    ))
    episode_id = ep1.episode_id
    
    # One clean detection
    ep2 = store.record_detection("owner", "node", FailureSignal(
        failure_mode=FailureMode.FM1_VARIANCE_REGIME,
        severity=2,  # Below threshold
        confidence=0.9,
        explanation="Clean 1",
    ))
    
    # Must NOT be closed
    assert ep2.state != EpisodeState.CLOSED, \
        "One clean detection must not close episode"
    assert ep2.state == EpisodeState.RECOVERING
    assert ep2.episode_id == episode_id
    
    # Severity spikes - should reset
    ep3 = store.record_detection("owner", "node", FailureSignal(
        failure_mode=FailureMode.FM1_VARIANCE_REGIME,
        severity=40,
        confidence=0.9,
        explanation="Spike",
    ))
    
    assert ep3.state == EpisodeState.ACTIVE
    assert ep3.consecutive_clean_count == 0
    assert ep3.episode_id == episode_id  # STILL same episode


# =============================================================================
# INVARIANT 5: append-only history (no retroactive rewrites)
# =============================================================================

def test_audit_trail_append_only():
    """
    INVARIANT: Audit trail must be append-only.
    
    We can add diagnostics, but we cannot rewrite historical states.
    """
    store = InMemoryEpisodeStore()
    
    # Create episode
    store.record_detection("owner", "node", FailureSignal(
        failure_mode=FailureMode.FM1_VARIANCE_REGIME,
        severity=50,
        confidence=0.9,
        explanation="Test",
    ))
    
    # Multiple updates
    for i in range(5):
        store.record_detection("owner", "node", FailureSignal(
            failure_mode=FailureMode.FM1_VARIANCE_REGIME,
            severity=50 + i,
            confidence=0.9,
            explanation=f"Update {i}",
        ))
    
    # Get audit trail
    audit = store.get_audit_trail("owner")
    
    # Must have all entries
    assert len(audit) >= 6, f"Audit trail must have all entries, got {len(audit)}"
    
    # Each entry has timestamp
    for entry in audit:
        assert "recorded_at" in entry


def test_validity_history_immutable():
    """
    INVARIANT: ValidityVerdict is frozen (immutable).
    """
    verdict = ValidityVerdict(score=75.0, state=ValidityState.VALID)
    
    # Attempting to modify must raise
    raised = False
    try:
        verdict.score = 50.0
    except (AttributeError, TypeError):
        raised = True
    
    assert raised, "ValidityVerdict must be immutable (frozen=True)"


# =============================================================================
# INVARIANT 6: attribution sums to 100%
# =============================================================================

def test_attribution_sums_to_100():
    """
    INVARIANT: When score < 100, attribution MUST sum to 100%.
    """
    from test_invariants import MockNode
    
    node = MockNode("test")
    node._forced_signals = [
        FailureSignal(
            failure_mode=FailureMode.FM1_VARIANCE_REGIME,
            severity=30,
            confidence=0.9,
            explanation="A",
        ),
        FailureSignal(
            failure_mode=FailureMode.FM2_MEAN_DRIFT,
            severity=40,
            confidence=0.8,
            explanation="B",
        ),
        FailureSignal(
            failure_mode=FailureMode.FM5_OUTLIER_CONTAMINATION,
            severity=20,
            confidence=0.7,
            explanation="C",
        ),
    ]
    node.compute_validity()
    
    if node.verdict.score < 100:
        total = sum(pct for _, pct in node.verdict.attributions)
        assert 99.0 <= total <= 101.0, \
            f"Attribution must sum to 100%, got {total}%"


def test_attribution_validation_method():
    """
    INVARIANT: validate_attribution() must detect invalid attributions.
    """
    # Invalid: doesn't sum to 100%
    bad_verdict = ValidityVerdict(
        score=70.0,
        state=ValidityState.VALID,
        attributions=(
            (FailureMode.FM1_VARIANCE_REGIME, 30.0),
            (FailureMode.FM2_MEAN_DRIFT, 20.0),
            # Missing 50%!
        ),
    )
    
    assert not bad_verdict.validate_attribution(), \
        "Must detect invalid attribution"
    
    # Valid: sums to 100%
    good_verdict = ValidityVerdict(
        score=70.0,
        state=ValidityState.VALID,
        attributions=(
            (FailureMode.FM1_VARIANCE_REGIME, 60.0),
            (FailureMode.FM2_MEAN_DRIFT, 40.0),
        ),
    )
    
    assert good_verdict.validate_attribution(), \
        "Must accept valid attribution"


# =============================================================================
# INVARIANT 7: kill switches override scoring
# =============================================================================

def test_kill_switch_forces_invalid():
    """
    INVARIANT: Kill switch must force INVALID regardless of computed score.
    """
    from test_invariants import MockNode
    
    # Mild failures except one kill switch
    node = MockNode("test")
    node._forced_signals = [
        FailureSignal(
            failure_mode=FailureMode.FM1_VARIANCE_REGIME,
            severity=10,
            confidence=0.9,
            explanation="Mild",
        ),
        FailureSignal(
            failure_mode=FailureMode.FM4_STRUCTURAL_BREAK,
            severity=40,  # Not even that severe
            confidence=0.9,
            triggers_kill=True,
            kill_reason="Break exceeds threshold",
            explanation="Kill",
        ),
    ]
    node.compute_validity()
    
    assert node.verdict.state == ValidityState.INVALID, \
        "Kill switch must force INVALID"
    assert node.verdict.is_killed, "Must be marked as killed"
    assert node.verdict.killed_by == FailureMode.FM4_STRUCTURAL_BREAK


# =============================================================================
# INVARIANT 8: tenant isolation
# =============================================================================

def test_tenant_isolation_absolute():
    """
    INVARIANT: Different tenants MUST NOT see each other's data.
    """
    store = InMemoryEpisodeStore()
    
    # Tenant A
    store.record_detection("tenant_a", "node_1", FailureSignal(
        failure_mode=FailureMode.FM1_VARIANCE_REGIME,
        severity=50,
        confidence=0.9,
        explanation="A",
    ))
    
    # Tenant B - same node_id
    store.record_detection("tenant_b", "node_1", FailureSignal(
        failure_mode=FailureMode.FM1_VARIANCE_REGIME,
        severity=60,
        confidence=0.9,
        explanation="B",
    ))
    
    # Each tenant sees only their own
    eps_a = store.get_active_episodes("tenant_a", "node_1")
    eps_b = store.get_active_episodes("tenant_b", "node_1")
    
    assert len(eps_a) == 1
    assert len(eps_b) == 1
    assert eps_a[0].episode_id != eps_b[0].episode_id
    assert eps_a[0].first_severity == 50
    assert eps_b[0].first_severity == 60
    
    # Audit trails are isolated
    audit_a = store.get_audit_trail("tenant_a")
    audit_b = store.get_audit_trail("tenant_b")
    
    for entry in audit_a:
        assert entry["owner_id"] == "tenant_a"
    for entry in audit_b:
        assert entry["owner_id"] == "tenant_b"


# =============================================================================
# RUN ALL INVARIANTS
# =============================================================================

def run_all_invariant_tests() -> bool:
    """
    Run all invariant tests.
    
    Returns True if all pass, False otherwise.
    This should be called in CI/CD and MUST fail the build on any failure.
    """
    print("\n" + "=" * 70)
    print("BAVELLA v2 — BUILD-BREAKING INVARIANT TESTS")
    print("=" * 70)
    print("These tests enforce FROZEN CANON. Failures must break the build.\n")
    
    runner = InvariantTestRunner()
    
    # Invariant 1: INVALID suppresses outputs
    print("INVARIANT 1: INVALID ⇒ No predictions/metrics")
    print("-" * 50)
    runner.run(test_invalid_suppresses_predictions)
    runner.run(test_invalid_governed_value_raises)
    runner.run(test_response_builder_suppresses_invalid)
    
    # Invariant 2: Inheritance is hard min
    print("\nINVARIANT 2: child_validity ≤ min(parent_validity)")
    print("-" * 50)
    runner.run(test_inheritance_hard_min)
    runner.run(test_invalid_parent_kills_child)
    
    # Invariant 3: Confidence cannot increase state
    print("\nINVARIANT 3: Confidence cannot INCREASE state")
    print("-" * 50)
    runner.run(test_confidence_cannot_improve_state)
    runner.run(test_confidence_governance_is_recorded)
    
    # Invariant 4: Episodes preserve first_seen_at
    print("\nINVARIANT 4: Episodes preserve first_seen_at")
    print("-" * 50)
    runner.run(test_episode_first_seen_immutable)
    runner.run(test_episode_hysteresis_prevents_false_recovery)
    
    # Invariant 5: Append-only history
    print("\nINVARIANT 5: Append-only history")
    print("-" * 50)
    runner.run(test_audit_trail_append_only)
    runner.run(test_validity_history_immutable)
    
    # Invariant 6: Attribution sums to 100%
    print("\nINVARIANT 6: Attribution sums to 100%")
    print("-" * 50)
    runner.run(test_attribution_sums_to_100)
    runner.run(test_attribution_validation_method)
    
    # Invariant 7: Kill switches override
    print("\nINVARIANT 7: Kill switches override scoring")
    print("-" * 50)
    runner.run(test_kill_switch_forces_invalid)
    
    # Invariant 8: Tenant isolation
    print("\nINVARIANT 8: Tenant isolation is absolute")
    print("-" * 50)
    runner.run(test_tenant_isolation_absolute)
    
    print("\n" + "=" * 70)
    success = runner.summary()
    
    if success:
        print("\n✓ ALL INVARIANTS ENFORCED - BUILD MAY PROCEED")
    else:
        print("\n✗ INVARIANT VIOLATIONS DETECTED - BUILD MUST FAIL")
    
    print("=" * 70)
    
    return success


if __name__ == "__main__":
    success = run_all_invariant_tests()
    sys.exit(0 if success else 1)
