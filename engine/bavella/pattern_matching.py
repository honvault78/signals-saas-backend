"""
Bavella v2 — Episode Pattern Matching
======================================

THE INSTITUTIONAL MEMORY: "This happened before"

When a user asks "Has this happened before?", Bavella answers with evidence:

    Similar episodes on this series:
    ● Nov 2021 — 84% similarity
      Duration: 45 days, Recovery: Partial
    ● Mar 2020 — 91% similarity  
      Duration: 67 days, Recovery: Full (rebaseline required)

This module implements:
    1. Episode fingerprinting (what makes an episode unique)
    2. Similarity computation (how alike are two episodes)
    3. Pattern matching across history
    4. Outcome tracking (what followed each episode)
    5. Recovery prediction (based on similar episodes)

The longer users run Bavella, the smarter this gets.
This is temporal lock-in, not just switching costs.

Copyright 2024-2026 Bavella Technologies Sarl
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Set
from collections import defaultdict

from .core import FailureMode, ValidityState
from .persistence_postgres import FailureEpisode, EpisodeState
from .epistemic_cost import Reversibility, EpistemicCostTable


# =============================================================================
# EPISODE FINGERPRINT
# =============================================================================

@dataclass(frozen=True)
class EpisodeFingerprint:
    """
    A fingerprint capturing the essential characteristics of an episode.
    
    Used for similarity matching.
    """
    # Which failure modes fired
    failure_modes: FrozenSet[FailureMode]
    
    # Severity profile
    max_severity: float
    avg_severity: float
    severity_trend: str  # "increasing", "decreasing", "stable", "volatile"
    
    # Duration bucket
    duration_bucket: str  # "flash" (<1d), "short" (1-7d), "medium" (7-30d), "long" (>30d)
    
    # Recovery outcome (for closed episodes)
    recovery_type: Optional[str] = None  # "full", "partial", "rebaseline", "none"
    recovery_days: Optional[int] = None
    
    # Context (optional)
    had_kill_switch: bool = False
    was_irreversible: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "failure_modes": [fm.name for fm in self.failure_modes],
            "max_severity": self.max_severity,
            "avg_severity": self.avg_severity,
            "severity_trend": self.severity_trend,
            "duration_bucket": self.duration_bucket,
            "recovery_type": self.recovery_type,
            "recovery_days": self.recovery_days,
            "had_kill_switch": self.had_kill_switch,
            "was_irreversible": self.was_irreversible,
        }


def compute_fingerprint(
    episodes: List[FailureEpisode],
    recovery_info: Optional[Dict[str, Any]] = None,
) -> EpisodeFingerprint:
    """
    Compute fingerprint from a set of concurrent episodes.
    
    Args:
        episodes: All active episodes at a point in time
        recovery_info: Optional recovery outcome data
    """
    if not episodes:
        return EpisodeFingerprint(
            failure_modes=frozenset(),
            max_severity=0,
            avg_severity=0,
            severity_trend="stable",
            duration_bucket="flash",
        )
    
    # Failure modes
    fms = frozenset(e.failure_mode for e in episodes)
    
    # Severity
    max_sev = max(e.max_severity for e in episodes)
    avg_sev = np.mean([e.last_severity for e in episodes])
    
    # Trend (based on first vs last severity)
    trends = []
    for e in episodes:
        if e.first_severity > 0:
            change = (e.last_severity - e.first_severity) / e.first_severity
            if change > 0.2:
                trends.append("increasing")
            elif change < -0.2:
                trends.append("decreasing")
            else:
                trends.append("stable")
    
    if not trends:
        severity_trend = "stable"
    elif trends.count("increasing") > len(trends) / 2:
        severity_trend = "increasing"
    elif trends.count("decreasing") > len(trends) / 2:
        severity_trend = "decreasing"
    elif len(set(trends)) > 1:
        severity_trend = "volatile"
    else:
        severity_trend = "stable"
    
    # Duration
    max_duration = max(e.duration for e in episodes)
    days = max_duration.total_seconds() / 86400
    if days < 1:
        duration_bucket = "flash"
    elif days < 7:
        duration_bucket = "short"
    elif days < 30:
        duration_bucket = "medium"
    else:
        duration_bucket = "long"
    
    # Kill switch / irreversibility
    had_kill = any(
        e.failure_mode in (FailureMode.FM4_STRUCTURAL_BREAK, FailureMode.FM7_DEPENDENCY_BREAK)
        and e.max_severity > 80
        for e in episodes
    )
    
    was_irreversible = any(
        EpistemicCostTable.is_irreversible(e.failure_mode)
        for e in episodes
    )
    
    # Recovery info
    recovery_type = None
    recovery_days = None
    if recovery_info:
        recovery_type = recovery_info.get("type")
        recovery_days = recovery_info.get("days")
    
    return EpisodeFingerprint(
        failure_modes=fms,
        max_severity=max_sev,
        avg_severity=avg_sev,
        severity_trend=severity_trend,
        duration_bucket=duration_bucket,
        recovery_type=recovery_type,
        recovery_days=recovery_days,
        had_kill_switch=had_kill,
        was_irreversible=was_irreversible,
    )


# =============================================================================
# SIMILARITY COMPUTATION
# =============================================================================

def compute_episode_similarity(
    fp1: EpisodeFingerprint,
    fp2: EpisodeFingerprint,
) -> float:
    """
    Compute similarity between two episode fingerprints.
    
    Returns: Similarity score 0-100%
    """
    scores = []
    weights = []
    
    # 1. Failure mode overlap (most important)
    if fp1.failure_modes and fp2.failure_modes:
        intersection = len(fp1.failure_modes & fp2.failure_modes)
        union = len(fp1.failure_modes | fp2.failure_modes)
        fm_sim = (intersection / union) * 100 if union > 0 else 0
    else:
        fm_sim = 0 if (fp1.failure_modes or fp2.failure_modes) else 100
    scores.append(fm_sim)
    weights.append(0.40)  # 40% weight
    
    # 2. Severity similarity
    max_diff = abs(fp1.max_severity - fp2.max_severity)
    sev_sim = max(0, 100 - max_diff)
    scores.append(sev_sim)
    weights.append(0.20)  # 20% weight
    
    # 3. Duration bucket match
    duration_sim = 100 if fp1.duration_bucket == fp2.duration_bucket else 50
    scores.append(duration_sim)
    weights.append(0.15)  # 15% weight
    
    # 4. Trend similarity
    trend_sim = 100 if fp1.severity_trend == fp2.severity_trend else 50
    scores.append(trend_sim)
    weights.append(0.10)  # 10% weight
    
    # 5. Kill switch match
    kill_sim = 100 if fp1.had_kill_switch == fp2.had_kill_switch else 50
    scores.append(kill_sim)
    weights.append(0.10)  # 10% weight
    
    # 6. Irreversibility match
    irrev_sim = 100 if fp1.was_irreversible == fp2.was_irreversible else 50
    scores.append(irrev_sim)
    weights.append(0.05)  # 5% weight
    
    # Weighted average
    total = sum(s * w for s, w in zip(scores, weights))
    return round(total, 1)


# =============================================================================
# EPISODE CLUSTER (a validity event with all concurrent FMs)
# =============================================================================

@dataclass
class EpisodeCluster:
    """
    A cluster of concurrent episodes forming a single validity event.
    
    When FM4 + FM1 + FM7 fire together, that's one "event" even though
    it's three episodes. This is what we match against history.
    """
    cluster_id: str
    owner_id: str
    node_id: str
    
    # Timing
    started_at: datetime
    ended_at: Optional[datetime] = None
    
    # Component episodes
    episode_ids: List[str] = field(default_factory=list)
    
    # Fingerprint
    fingerprint: Optional[EpisodeFingerprint] = None
    
    # Outcome (filled when cluster closes)
    recovery_type: Optional[str] = None
    recovery_days: Optional[int] = None
    post_validity_score: Optional[float] = None
    
    # Minimum validity during this cluster
    min_validity_score: float = 100.0
    min_validity_state: str = "VALID"
    
    @property
    def is_active(self) -> bool:
        return self.ended_at is None
    
    @property
    def duration_days(self) -> float:
        end = self.ended_at or datetime.now(timezone.utc)
        return (end - self.started_at).total_seconds() / 86400
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "node_id": self.node_id,
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "duration_days": round(self.duration_days, 1),
            "episode_ids": self.episode_ids,
            "fingerprint": self.fingerprint.to_dict() if self.fingerprint else None,
            "recovery_type": self.recovery_type,
            "recovery_days": self.recovery_days,
            "min_validity_score": self.min_validity_score,
            "min_validity_state": self.min_validity_state,
            "is_active": self.is_active,
        }


# =============================================================================
# SIMILAR EPISODE MATCH
# =============================================================================

@dataclass
class SimilarEpisodeMatch:
    """A historical episode that matches the current pattern."""
    cluster: EpisodeCluster
    similarity: float  # 0-100%
    
    # What happened
    failure_modes: List[str]
    duration_days: float
    recovery_type: Optional[str]
    recovery_days: Optional[int]
    
    # When it happened
    occurred_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "cluster_id": self.cluster.cluster_id,
            "similarity": self.similarity,
            "failure_modes": self.failure_modes,
            "duration_days": self.duration_days,
            "recovery_type": self.recovery_type,
            "recovery_days": self.recovery_days,
            "occurred_at": self.occurred_at.isoformat(),
        }


# =============================================================================
# PATTERN MATCHER
# =============================================================================

class EpisodePatternMatcher:
    """
    Matches current episodes against historical patterns.
    
    This is the "has this happened before?" engine.
    """
    
    def __init__(self):
        # In-memory cluster store (would be Postgres in production)
        self._clusters: Dict[str, List[EpisodeCluster]] = defaultdict(list)
    
    def record_cluster(
        self,
        owner_id: str,
        node_id: str,
        episodes: List[FailureEpisode],
        validity_score: float,
        validity_state: str,
    ) -> EpisodeCluster:
        """
        Record or update a cluster of concurrent episodes.
        """
        key = f"{owner_id}:{node_id}"
        
        # Check for active cluster
        active = None
        for cluster in self._clusters[key]:
            if cluster.is_active:
                active = cluster
                break
        
        if not episodes:
            # No active episodes - close any active cluster
            if active:
                active.ended_at = datetime.now(timezone.utc)
                active.recovery_days = int(active.duration_days)
                if validity_score >= 70:
                    active.recovery_type = "full"
                elif validity_score >= 50:
                    active.recovery_type = "partial"
                else:
                    active.recovery_type = "none"
                active.post_validity_score = validity_score
            return active
        
        # Update fingerprint
        fingerprint = compute_fingerprint(episodes)
        
        if active:
            # Update existing cluster
            active.fingerprint = fingerprint
            active.min_validity_score = min(active.min_validity_score, validity_score)
            if validity_score < 30:
                active.min_validity_state = "INVALID"
            elif validity_score < 70:
                active.min_validity_state = "DEGRADED"
            
            # Add any new episodes
            for e in episodes:
                if e.episode_id not in active.episode_ids:
                    active.episode_ids.append(e.episode_id)
            
            return active
        else:
            # Create new cluster
            import uuid
            cluster = EpisodeCluster(
                cluster_id=str(uuid.uuid4()),
                owner_id=owner_id,
                node_id=node_id,
                started_at=datetime.now(timezone.utc),
                episode_ids=[e.episode_id for e in episodes],
                fingerprint=fingerprint,
                min_validity_score=validity_score,
                min_validity_state=validity_state,
            )
            self._clusters[key].append(cluster)
            return cluster
    
    def find_similar(
        self,
        owner_id: str,
        node_id: str,
        current_fingerprint: EpisodeFingerprint,
        min_similarity: float = 50.0,
        limit: int = 5,
    ) -> List[SimilarEpisodeMatch]:
        """
        Find historical episodes similar to current pattern.
        """
        key = f"{owner_id}:{node_id}"
        matches = []
        
        for cluster in self._clusters[key]:
            # Skip active clusters
            if cluster.is_active:
                continue
            
            # Skip if no fingerprint
            if not cluster.fingerprint:
                continue
            
            # Compute similarity
            sim = compute_episode_similarity(current_fingerprint, cluster.fingerprint)
            
            if sim >= min_similarity:
                matches.append(SimilarEpisodeMatch(
                    cluster=cluster,
                    similarity=sim,
                    failure_modes=[fm.name for fm in cluster.fingerprint.failure_modes],
                    duration_days=cluster.duration_days,
                    recovery_type=cluster.recovery_type,
                    recovery_days=cluster.recovery_days,
                    occurred_at=cluster.started_at,
                ))
        
        # Sort by similarity
        matches.sort(key=lambda m: m.similarity, reverse=True)
        return matches[:limit]
    
    def get_cluster_history(
        self,
        owner_id: str,
        node_id: str,
        limit: int = 20,
    ) -> List[EpisodeCluster]:
        """Get historical clusters for a node."""
        key = f"{owner_id}:{node_id}"
        clusters = [c for c in self._clusters[key] if not c.is_active]
        clusters.sort(key=lambda c: c.started_at, reverse=True)
        return clusters[:limit]
    
    def get_recovery_statistics(
        self,
        owner_id: str,
        node_id: str,
        failure_modes: Optional[Set[FailureMode]] = None,
    ) -> Dict[str, Any]:
        """
        Get recovery statistics for a node.
        
        Answers: "When this happens, how long until recovery?"
        """
        key = f"{owner_id}:{node_id}"
        
        relevant = []
        for cluster in self._clusters[key]:
            if cluster.is_active:
                continue
            if failure_modes and cluster.fingerprint:
                if not (failure_modes & cluster.fingerprint.failure_modes):
                    continue
            relevant.append(cluster)
        
        if not relevant:
            return {
                "sample_size": 0,
                "avg_recovery_days": None,
                "median_recovery_days": None,
                "recovery_rate": None,
            }
        
        recovery_days = [c.recovery_days for c in relevant if c.recovery_days]
        full_recoveries = [c for c in relevant if c.recovery_type == "full"]
        
        return {
            "sample_size": len(relevant),
            "avg_recovery_days": round(np.mean(recovery_days), 1) if recovery_days else None,
            "median_recovery_days": round(np.median(recovery_days), 1) if recovery_days else None,
            "min_recovery_days": min(recovery_days) if recovery_days else None,
            "max_recovery_days": max(recovery_days) if recovery_days else None,
            "recovery_rate": round(len(full_recoveries) / len(relevant) * 100, 1),
            "rebaseline_rate": round(
                len([c for c in relevant if c.recovery_type == "rebaseline"]) / len(relevant) * 100, 1
            ),
        }


# =============================================================================
# "THIS HAPPENED BEFORE" RESPONSE
# =============================================================================

@dataclass
class ThisHappenedBeforeResponse:
    """
    Complete response to "Has this happened before?"
    """
    # Current situation
    current_failure_modes: List[str]
    current_severity: float
    current_duration_days: float
    
    # Similar episodes found
    similar_episodes: List[SimilarEpisodeMatch]
    
    # Statistics
    total_similar_count: int
    avg_recovery_days: Optional[float]
    recovery_rate: Optional[float]
    
    # Most relevant comparison
    most_similar: Optional[SimilarEpisodeMatch]
    
    # Narrative
    narrative: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "current": {
                "failure_modes": self.current_failure_modes,
                "severity": self.current_severity,
                "duration_days": self.current_duration_days,
            },
            "similar_episodes": [m.to_dict() for m in self.similar_episodes],
            "statistics": {
                "total_similar_count": self.total_similar_count,
                "avg_recovery_days": self.avg_recovery_days,
                "recovery_rate": self.recovery_rate,
            },
            "most_similar": self.most_similar.to_dict() if self.most_similar else None,
            "narrative": self.narrative,
        }


def build_this_happened_before(
    current_episodes: List[FailureEpisode],
    matcher: EpisodePatternMatcher,
    owner_id: str,
    node_id: str,
) -> ThisHappenedBeforeResponse:
    """
    Build complete "this happened before" response.
    """
    if not current_episodes:
        return ThisHappenedBeforeResponse(
            current_failure_modes=[],
            current_severity=0,
            current_duration_days=0,
            similar_episodes=[],
            total_similar_count=0,
            avg_recovery_days=None,
            recovery_rate=None,
            most_similar=None,
            narrative="No active failure episodes.",
        )
    
    # Current fingerprint
    fingerprint = compute_fingerprint(current_episodes)
    
    # Find similar
    similar = matcher.find_similar(owner_id, node_id, fingerprint)
    
    # Get stats
    fms = {e.failure_mode for e in current_episodes}
    stats = matcher.get_recovery_statistics(owner_id, node_id, fms)
    
    # Current info
    current_fms = [fm.name for fm in fingerprint.failure_modes]
    current_severity = fingerprint.max_severity
    current_duration = max(e.duration.total_seconds() / 86400 for e in current_episodes)
    
    # Build narrative
    if not similar:
        narrative = f"This is the first time this pattern ({', '.join(current_fms)}) has occurred on this series."
    else:
        most = similar[0]
        narrative = (
            f"Similar pattern found {len(similar)} times in history. "
            f"Most similar: {most.occurred_at.strftime('%b %Y')} ({most.similarity:.0f}% match). "
        )
        if most.recovery_days:
            narrative += f"That episode lasted {most.recovery_days} days"
            if most.recovery_type:
                narrative += f" with {most.recovery_type} recovery"
            narrative += ". "
        
        if stats["avg_recovery_days"]:
            narrative += f"Average recovery for this pattern: {stats['avg_recovery_days']:.0f} days."
    
    return ThisHappenedBeforeResponse(
        current_failure_modes=current_fms,
        current_severity=current_severity,
        current_duration_days=round(current_duration, 1),
        similar_episodes=similar,
        total_similar_count=len(similar),
        avg_recovery_days=stats.get("avg_recovery_days"),
        recovery_rate=stats.get("recovery_rate"),
        most_similar=similar[0] if similar else None,
        narrative=narrative,
    )


# =============================================================================
# TESTS
# =============================================================================

def test_fingerprint_computation():
    """Test episode fingerprinting."""
    from .persistence_postgres import FailureEpisode, EpisodeState
    from datetime import timedelta
    
    now = datetime.now(timezone.utc)
    
    episodes = [
        FailureEpisode(
            episode_id="e1", episode_key="k1", owner_id="owner", node_id="node",
            failure_mode=FailureMode.FM4_STRUCTURAL_BREAK,
            state=EpisodeState.ACTIVE,
            first_seen_at=now - timedelta(days=5),
            last_seen_at=now,
            first_severity=50, last_severity=70, max_severity=75,
        ),
        FailureEpisode(
            episode_id="e2", episode_key="k2", owner_id="owner", node_id="node",
            failure_mode=FailureMode.FM1_VARIANCE_REGIME,
            state=EpisodeState.ACTIVE,
            first_seen_at=now - timedelta(days=3),
            last_seen_at=now,
            first_severity=30, last_severity=45, max_severity=50,
        ),
    ]
    
    fp = compute_fingerprint(episodes)
    
    assert FailureMode.FM4_STRUCTURAL_BREAK in fp.failure_modes
    assert FailureMode.FM1_VARIANCE_REGIME in fp.failure_modes
    assert fp.max_severity == 75
    assert fp.duration_bucket == "short"
    assert fp.had_kill_switch == False  # severity not > 80
    assert fp.was_irreversible == True  # FM4 is irreversible
    
    print(f"Fingerprint: {fp.to_dict()}")
    print("✓ test_fingerprint_computation passed")


def test_similarity_computation():
    """Test similarity between fingerprints."""
    fp1 = EpisodeFingerprint(
        failure_modes=frozenset([FailureMode.FM4_STRUCTURAL_BREAK, FailureMode.FM1_VARIANCE_REGIME]),
        max_severity=75,
        avg_severity=60,
        severity_trend="increasing",
        duration_bucket="short",
        had_kill_switch=False,
        was_irreversible=True,
    )
    
    # Very similar
    fp2 = EpisodeFingerprint(
        failure_modes=frozenset([FailureMode.FM4_STRUCTURAL_BREAK, FailureMode.FM1_VARIANCE_REGIME]),
        max_severity=70,
        avg_severity=55,
        severity_trend="increasing",
        duration_bucket="short",
        had_kill_switch=False,
        was_irreversible=True,
    )
    
    sim_high = compute_episode_similarity(fp1, fp2)
    assert sim_high > 80, f"Similar fingerprints should have high similarity, got {sim_high}"
    
    # Different
    fp3 = EpisodeFingerprint(
        failure_modes=frozenset([FailureMode.FM2_MEAN_DRIFT]),
        max_severity=30,
        avg_severity=25,
        severity_trend="stable",
        duration_bucket="long",
        had_kill_switch=False,
        was_irreversible=False,
    )
    
    sim_low = compute_episode_similarity(fp1, fp3)
    assert sim_low < 50, f"Different fingerprints should have low similarity, got {sim_low}"
    
    print(f"High similarity: {sim_high}%")
    print(f"Low similarity: {sim_low}%")
    print("✓ test_similarity_computation passed")


def test_pattern_matching():
    """Test the full pattern matching flow."""
    from .persistence_postgres import FailureEpisode, EpisodeState
    from datetime import timedelta
    
    matcher = EpisodePatternMatcher()
    now = datetime.now(timezone.utc)
    
    # Record historical episode 1 (2 months ago)
    hist1_episodes = [
        FailureEpisode(
            episode_id="h1e1", episode_key="k1", owner_id="owner", node_id="node",
            failure_mode=FailureMode.FM4_STRUCTURAL_BREAK,
            state=EpisodeState.CLOSED,
            first_seen_at=now - timedelta(days=60),
            last_seen_at=now - timedelta(days=45),
            closed_at=now - timedelta(days=45),
            first_severity=60, last_severity=20, max_severity=70,
        ),
    ]
    cluster1 = matcher.record_cluster("owner", "node", hist1_episodes, 75, "VALID")
    cluster1.ended_at = now - timedelta(days=45)
    cluster1.recovery_type = "full"
    cluster1.recovery_days = 15
    
    # Record historical episode 2 (1 month ago)
    hist2_episodes = [
        FailureEpisode(
            episode_id="h2e1", episode_key="k2", owner_id="owner", node_id="node",
            failure_mode=FailureMode.FM4_STRUCTURAL_BREAK,
            state=EpisodeState.CLOSED,
            first_seen_at=now - timedelta(days=30),
            last_seen_at=now - timedelta(days=10),
            closed_at=now - timedelta(days=10),
            first_severity=70, last_severity=25, max_severity=80,
        ),
        FailureEpisode(
            episode_id="h2e2", episode_key="k3", owner_id="owner", node_id="node",
            failure_mode=FailureMode.FM1_VARIANCE_REGIME,
            state=EpisodeState.CLOSED,
            first_seen_at=now - timedelta(days=28),
            last_seen_at=now - timedelta(days=10),
            closed_at=now - timedelta(days=10),
            first_severity=40, last_severity=15, max_severity=50,
        ),
    ]
    cluster2 = matcher.record_cluster("owner", "node", hist2_episodes, 70, "VALID")
    cluster2.ended_at = now - timedelta(days=10)
    cluster2.recovery_type = "partial"
    cluster2.recovery_days = 20
    
    # Current episode (similar to hist2)
    current = [
        FailureEpisode(
            episode_id="c1", episode_key="kc1", owner_id="owner", node_id="node",
            failure_mode=FailureMode.FM4_STRUCTURAL_BREAK,
            state=EpisodeState.ACTIVE,
            first_seen_at=now - timedelta(days=2),
            last_seen_at=now,
            first_severity=65, last_severity=72, max_severity=72,
        ),
        FailureEpisode(
            episode_id="c2", episode_key="kc2", owner_id="owner", node_id="node",
            failure_mode=FailureMode.FM1_VARIANCE_REGIME,
            state=EpisodeState.ACTIVE,
            first_seen_at=now - timedelta(days=1),
            last_seen_at=now,
            first_severity=35, last_severity=42, max_severity=42,
        ),
    ]
    
    # Build response
    response = build_this_happened_before(current, matcher, "owner", "node")
    
    assert len(response.similar_episodes) > 0
    assert response.most_similar is not None
    print(f"Found {len(response.similar_episodes)} similar episodes")
    print(f"Most similar: {response.most_similar.similarity}%")
    print(f"Narrative: {response.narrative}")
    print("✓ test_pattern_matching passed")


def run_all_pattern_matching_tests():
    print("\n" + "=" * 60)
    print("EPISODE PATTERN MATCHING TESTS")
    print("=" * 60 + "\n")
    
    test_fingerprint_computation()
    print()
    test_similarity_computation()
    print()
    test_pattern_matching()
    
    print("\n" + "=" * 60)
    print("ALL PATTERN MATCHING TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    run_all_pattern_matching_tests()
