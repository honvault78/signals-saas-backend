"""
Bavella v2 — Cross-Node Contagion & Portfolio Intelligence
============================================================

THE "WHAT ELSE BROKE" LAYER

The real institutional memory moment is cross-node:
    "Momentum broke first, then value became invalid because FM7 dependency broke"
    "This pair's MR failure coincided with a volatility regime shift in the sector"
    "Your backtest is invalid because it spans a break that propagated into hedges"

This module defines:
    1. ContagionLink - explicit causal relationship between episodes
    2. NodeGraph - queryable graph of validity relationships
    3. PropagationTracker - detects and records cross-node causality
    4. PortfolioValidityRollup - aggregate validity across node sets

Episodes now support:
    - upstream/downstream links (caused_by_episode_id, propagated_to[])
    - node graph queries ("what else broke around this time?")
    - portfolio-level validity rollups

Copyright 2024-2026 Bavella Technologies Sarl
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Set, FrozenSet
import numpy as np

from .audit_infrastructure import (
    BavellaVersions, EvidenceType, EvidenceArtifact,
    EpisodeRecord, EpisodeStatus, NodeType,
)
from .core import FailureMode


# =============================================================================
# CONTAGION LINK TYPES
# =============================================================================

class ContagionType(Enum):
    """How validity failure propagated between nodes."""
    
    # Direct dependency break
    DEPENDENCY_BREAK = "dependency_break"  # FM7 in source → invalidity in dependent
    
    # Correlation/regime shift
    CORRELATION_SHIFT = "correlation_shift"  # Regime change affected related nodes
    
    # Volatility contagion
    VOLATILITY_CONTAGION = "volatility_contagion"  # Vol regime propagated
    
    # Structural break propagation
    STRUCTURAL_PROPAGATION = "structural_propagation"  # FM4 cascaded
    
    # Common factor exposure
    COMMON_FACTOR = "common_factor"  # Both exposed to same factor that broke
    
    # Temporal coincidence (suspicious but not causal)
    TEMPORAL_COINCIDENCE = "temporal_coincidence"
    
    # Manual link (analyst judgment)
    MANUAL = "manual"


class PropagationDirection(Enum):
    """Direction of causality."""
    UPSTREAM = "upstream"      # This episode was caused by another
    DOWNSTREAM = "downstream"  # This episode caused another
    BIDIRECTIONAL = "bidirectional"  # Mutual reinforcement


# =============================================================================
# CONTAGION LINK
# =============================================================================

@dataclass(frozen=True)
class ContagionLink:
    """
    Explicit causal relationship between episodes.
    
    This is what makes "portfolio intelligence" defensible.
    """
    # Identity
    link_id: str
    
    # Episodes linked
    source_episode_id: str  # The episode that propagated
    target_episode_id: str  # The episode that received
    
    # Nodes involved
    source_node_id: str
    target_node_id: str
    
    # Relationship
    contagion_type: ContagionType
    direction: PropagationDirection
    
    # Timing
    source_onset: datetime
    target_onset: datetime
    lag_hours: float  # How long after source did target start
    
    # Strength
    confidence: float  # 0-1, how sure are we this is causal
    
    # Evidence
    evidence_summary: str = ""
    evidence_ids: Tuple[str, ...] = ()
    
    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: str = "system"  # "system" or analyst ID for manual links
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "link_id": self.link_id,
            "source_episode_id": self.source_episode_id,
            "target_episode_id": self.target_episode_id,
            "source_node_id": self.source_node_id,
            "target_node_id": self.target_node_id,
            "contagion_type": self.contagion_type.value,
            "direction": self.direction.value,
            "source_onset": self.source_onset.isoformat(),
            "target_onset": self.target_onset.isoformat(),
            "lag_hours": self.lag_hours,
            "confidence": self.confidence,
            "evidence_summary": self.evidence_summary,
            "evidence_ids": list(self.evidence_ids),
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
        }


# =============================================================================
# EPISODE WITH CONTAGION
# =============================================================================

@dataclass
class EpisodeContagionInfo:
    """
    Contagion information attached to an episode.
    
    Extends EpisodeRecord with cross-node links.
    """
    episode_id: str
    
    # Upstream causality (what caused this)
    caused_by_episode_ids: List[str] = field(default_factory=list)
    caused_by_links: List[ContagionLink] = field(default_factory=list)
    
    # Downstream propagation (what this caused)
    propagated_to_episode_ids: List[str] = field(default_factory=list)
    propagated_to_links: List[ContagionLink] = field(default_factory=list)
    
    # Summary
    is_root_of_cascade: bool = False  # No upstream causes
    cascade_depth: int = 0  # How deep in propagation chain
    total_downstream_count: int = 0  # Total nodes affected
    
    def add_upstream_link(self, link: ContagionLink) -> None:
        if link.source_episode_id not in self.caused_by_episode_ids:
            self.caused_by_episode_ids.append(link.source_episode_id)
            self.caused_by_links.append(link)
    
    def add_downstream_link(self, link: ContagionLink) -> None:
        if link.target_episode_id not in self.propagated_to_episode_ids:
            self.propagated_to_episode_ids.append(link.target_episode_id)
            self.propagated_to_links.append(link)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "caused_by_episode_ids": self.caused_by_episode_ids,
            "propagated_to_episode_ids": self.propagated_to_episode_ids,
            "is_root_of_cascade": self.is_root_of_cascade,
            "cascade_depth": self.cascade_depth,
            "total_downstream_count": self.total_downstream_count,
            "upstream_links": [l.to_dict() for l in self.caused_by_links],
            "downstream_links": [l.to_dict() for l in self.propagated_to_links],
        }


# =============================================================================
# NODE GRAPH
# =============================================================================

@dataclass
class NodeRelationship:
    """Defined relationship between two nodes."""
    source_node_id: str
    target_node_id: str
    relationship_type: str  # "depends_on", "hedges", "correlated", "same_factor"
    strength: float = 1.0  # 0-1
    metadata: Dict[str, Any] = field(default_factory=dict)


class NodeGraph:
    """
    Queryable graph of validity relationships between nodes.
    
    This is the "portfolio intelligence" data structure.
    """
    
    def __init__(self):
        # Node registry
        self._nodes: Dict[str, Dict[str, Any]] = {}
        
        # Relationships (adjacency list)
        self._outgoing: Dict[str, List[NodeRelationship]] = defaultdict(list)
        self._incoming: Dict[str, List[NodeRelationship]] = defaultdict(list)
        
        # Episodes by node
        self._episodes_by_node: Dict[str, List[str]] = defaultdict(list)
        
        # Contagion links
        self._contagion_links: Dict[str, ContagionLink] = {}
        self._contagion_by_source: Dict[str, List[str]] = defaultdict(list)
        self._contagion_by_target: Dict[str, List[str]] = defaultdict(list)
        
        # Episode contagion info
        self._episode_contagion: Dict[str, EpisodeContagionInfo] = {}
    
    def register_node(
        self,
        node_id: str,
        node_type: NodeType,
        owner_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register a node in the graph."""
        self._nodes[node_id] = {
            "node_id": node_id,
            "node_type": node_type.value,
            "owner_id": owner_id,
            "metadata": metadata or {},
            "registered_at": datetime.now(timezone.utc).isoformat(),
        }
    
    def add_relationship(
        self,
        source_node_id: str,
        target_node_id: str,
        relationship_type: str,
        strength: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a relationship between nodes."""
        rel = NodeRelationship(
            source_node_id=source_node_id,
            target_node_id=target_node_id,
            relationship_type=relationship_type,
            strength=strength,
            metadata=metadata or {},
        )
        self._outgoing[source_node_id].append(rel)
        self._incoming[target_node_id].append(rel)
    
    def register_episode(self, episode_id: str, node_id: str) -> None:
        """Register an episode for a node."""
        if episode_id not in self._episodes_by_node[node_id]:
            self._episodes_by_node[node_id].append(episode_id)
        
        # Initialize contagion info
        if episode_id not in self._episode_contagion:
            self._episode_contagion[episode_id] = EpisodeContagionInfo(episode_id=episode_id)
    
    def add_contagion_link(self, link: ContagionLink) -> None:
        """Add a contagion link between episodes."""
        self._contagion_links[link.link_id] = link
        self._contagion_by_source[link.source_episode_id].append(link.link_id)
        self._contagion_by_target[link.target_episode_id].append(link.link_id)
        
        # Update episode contagion info
        if link.source_episode_id in self._episode_contagion:
            self._episode_contagion[link.source_episode_id].add_downstream_link(link)
        
        if link.target_episode_id in self._episode_contagion:
            self._episode_contagion[link.target_episode_id].add_upstream_link(link)
    
    def get_dependent_nodes(self, node_id: str) -> List[str]:
        """Get nodes that depend on this node."""
        return [rel.target_node_id for rel in self._outgoing[node_id]
                if rel.relationship_type == "depends_on"]
    
    def get_upstream_nodes(self, node_id: str) -> List[str]:
        """Get nodes that this node depends on."""
        return [rel.source_node_id for rel in self._incoming[node_id]
                if rel.relationship_type == "depends_on"]
    
    def get_related_nodes(
        self,
        node_id: str,
        relationship_types: Optional[List[str]] = None,
    ) -> List[Tuple[str, str, float]]:
        """Get all related nodes with relationship type and strength."""
        results = []
        
        for rel in self._outgoing[node_id]:
            if relationship_types is None or rel.relationship_type in relationship_types:
                results.append((rel.target_node_id, rel.relationship_type, rel.strength))
        
        for rel in self._incoming[node_id]:
            if relationship_types is None or rel.relationship_type in relationship_types:
                results.append((rel.source_node_id, rel.relationship_type, rel.strength))
        
        return results
    
    def get_episode_contagion(self, episode_id: str) -> Optional[EpisodeContagionInfo]:
        """Get contagion info for an episode."""
        return self._episode_contagion.get(episode_id)
    
    def get_episodes_in_window(
        self,
        start: datetime,
        end: datetime,
        node_ids: Optional[List[str]] = None,
    ) -> List[str]:
        """Get episodes that started within a time window."""
        # This would query the episode store in production
        # For now, return all registered episodes
        if node_ids:
            episodes = []
            for nid in node_ids:
                episodes.extend(self._episodes_by_node[nid])
            return list(set(episodes))
        else:
            all_episodes = []
            for eps in self._episodes_by_node.values():
                all_episodes.extend(eps)
            return list(set(all_episodes))
    
    def query_what_else_broke(
        self,
        node_id: str,
        episode_id: str,
        window_hours: float = 48,
    ) -> Dict[str, Any]:
        """
        Answer: "What else broke around this time?"
        
        Returns related episodes with contagion analysis.
        """
        related_nodes = self.get_related_nodes(node_id)
        
        result = {
            "query_node": node_id,
            "query_episode": episode_id,
            "window_hours": window_hours,
            "related_episodes": [],
            "contagion_links": [],
            "summary": {},
        }
        
        # Get upstream causes
        contagion_info = self.get_episode_contagion(episode_id)
        if contagion_info:
            result["upstream_causes"] = contagion_info.caused_by_episode_ids
            result["downstream_effects"] = contagion_info.propagated_to_episode_ids
            result["contagion_links"] = [l.to_dict() for l in contagion_info.caused_by_links]
        
        # Get related node episodes
        for related_node_id, rel_type, strength in related_nodes:
            for ep_id in self._episodes_by_node[related_node_id]:
                result["related_episodes"].append({
                    "episode_id": ep_id,
                    "node_id": related_node_id,
                    "relationship": rel_type,
                    "strength": strength,
                })
        
        return result


# =============================================================================
# PROPAGATION DETECTOR
# =============================================================================

class PropagationDetector:
    """
    Detects causal relationships between episodes.
    
    Uses timing, node relationships, and failure mode patterns
    to identify likely contagion.
    """
    
    # Configuration
    MAX_LAG_HOURS = 72  # Max time between source and target onset
    MIN_CONFIDENCE = 0.5  # Minimum confidence to create link
    
    def __init__(self, node_graph: NodeGraph):
        self._graph = node_graph
    
    def detect_propagation(
        self,
        source_episode: EpisodeRecord,
        target_episode: EpisodeRecord,
    ) -> Optional[ContagionLink]:
        """
        Detect if source episode caused target episode.
        
        Returns ContagionLink if causal relationship detected.
        """
        # Must be different nodes
        if source_episode.node_id == target_episode.node_id:
            return None
        
        # Check timing (source must start before or at same time as target)
        if source_episode.start_time > target_episode.start_time:
            return None
        
        lag = (target_episode.start_time - source_episode.start_time).total_seconds() / 3600
        if lag > self.MAX_LAG_HOURS:
            return None
        
        # Check if nodes are related
        related = self._graph.get_related_nodes(source_episode.node_id)
        relationship = None
        rel_strength = 0.0
        
        for node_id, rel_type, strength in related:
            if node_id == target_episode.node_id:
                relationship = rel_type
                rel_strength = strength
                break
        
        if not relationship:
            # Check for temporal coincidence (weaker)
            return self._create_temporal_link(source_episode, target_episode, lag)
        
        # Determine contagion type based on failure modes
        contagion_type, confidence = self._classify_contagion(
            source_episode, target_episode, relationship, rel_strength, lag
        )
        
        if confidence < self.MIN_CONFIDENCE:
            return None
        
        return ContagionLink(
            link_id=str(uuid.uuid4()),
            source_episode_id=source_episode.episode_id,
            target_episode_id=target_episode.episode_id,
            source_node_id=source_episode.node_id,
            target_node_id=target_episode.node_id,
            contagion_type=contagion_type,
            direction=PropagationDirection.DOWNSTREAM,
            source_onset=source_episode.start_time,
            target_onset=target_episode.start_time,
            lag_hours=lag,
            confidence=confidence,
            evidence_summary=f"Relationship: {relationship}, Lag: {lag:.1f}h",
        )
    
    def _classify_contagion(
        self,
        source: EpisodeRecord,
        target: EpisodeRecord,
        relationship: str,
        rel_strength: float,
        lag: float,
    ) -> Tuple[ContagionType, float]:
        """Classify the type of contagion and compute confidence."""
        
        source_fms = {fm.fm_code for fm in source.failure_modes}
        target_fms = {fm.fm_code for fm in target.failure_modes}
        
        # FM7 dependency break
        if "FM7_DEPENDENCY_BREAK" in source_fms:
            if relationship == "depends_on":
                return ContagionType.DEPENDENCY_BREAK, 0.9 * rel_strength
            else:
                return ContagionType.CORRELATION_SHIFT, 0.7 * rel_strength
        
        # FM4 structural break
        if "FM4_STRUCTURAL_BREAK" in source_fms:
            return ContagionType.STRUCTURAL_PROPAGATION, 0.8 * rel_strength
        
        # FM1 variance regime
        if "FM1_VARIANCE_REGIME" in source_fms:
            return ContagionType.VOLATILITY_CONTAGION, 0.6 * rel_strength
        
        # Common factor
        if relationship == "same_factor":
            return ContagionType.COMMON_FACTOR, 0.75 * rel_strength
        
        # Default: correlation shift
        return ContagionType.CORRELATION_SHIFT, 0.5 * rel_strength
    
    def _create_temporal_link(
        self,
        source: EpisodeRecord,
        target: EpisodeRecord,
        lag: float,
    ) -> Optional[ContagionLink]:
        """Create weak temporal coincidence link."""
        # Very short lag suggests possible causation
        if lag > 24:
            return None
        
        confidence = max(0.3, 0.5 - lag / 48)
        
        return ContagionLink(
            link_id=str(uuid.uuid4()),
            source_episode_id=source.episode_id,
            target_episode_id=target.episode_id,
            source_node_id=source.node_id,
            target_node_id=target.node_id,
            contagion_type=ContagionType.TEMPORAL_COINCIDENCE,
            direction=PropagationDirection.DOWNSTREAM,
            source_onset=source.start_time,
            target_onset=target.start_time,
            lag_hours=lag,
            confidence=confidence,
            evidence_summary=f"Temporal proximity: {lag:.1f}h (no known relationship)",
        )
    
    def analyze_cascade(
        self,
        root_episode_id: str,
    ) -> Dict[str, Any]:
        """
        Analyze full cascade starting from a root episode.
        """
        visited = set()
        cascade = []
        
        def dfs(ep_id: str, depth: int):
            if ep_id in visited:
                return
            visited.add(ep_id)
            
            contagion = self._graph.get_episode_contagion(ep_id)
            if contagion:
                cascade.append({
                    "episode_id": ep_id,
                    "depth": depth,
                    "downstream_count": len(contagion.propagated_to_episode_ids),
                })
                
                for downstream_id in contagion.propagated_to_episode_ids:
                    dfs(downstream_id, depth + 1)
        
        dfs(root_episode_id, 0)
        
        return {
            "root_episode_id": root_episode_id,
            "total_affected": len(cascade),
            "max_depth": max(c["depth"] for c in cascade) if cascade else 0,
            "cascade": cascade,
        }


# =============================================================================
# PORTFOLIO VALIDITY ROLLUP
# =============================================================================

@dataclass
class NodeValidityState:
    """Current validity state of a node."""
    node_id: str
    validity_score: float
    validity_state: str
    has_active_episode: bool
    active_episode_id: Optional[str] = None
    trust_penalty: float = 0.0


@dataclass
class PortfolioValidityRollup:
    """
    Aggregate validity across a set of nodes.
    
    This is where "budget" logic shows up.
    """
    # Identity
    rollup_id: str
    owner_id: str
    computed_at: datetime
    
    # Scope
    node_ids: List[str]
    node_count: int
    
    # Aggregate validity
    min_validity: float
    avg_validity: float
    weighted_validity: float  # Weighted by node importance
    
    # State counts
    valid_count: int
    degraded_count: int
    invalid_count: int
    
    # Episode summary
    active_episodes_count: int
    nodes_with_active_episodes: List[str]
    
    # Contagion summary
    contagion_links_count: int
    cascade_roots: List[str]  # Episodes that started cascades
    
    # Risk flags
    has_structural_break: bool
    has_dependency_break: bool
    requires_attention: bool
    
    # Warnings
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "rollup_id": self.rollup_id,
            "owner_id": self.owner_id,
            "computed_at": self.computed_at.isoformat(),
            "scope": {
                "node_ids": self.node_ids,
                "node_count": self.node_count,
            },
            "validity": {
                "min": self.min_validity,
                "avg": self.avg_validity,
                "weighted": self.weighted_validity,
            },
            "states": {
                "valid": self.valid_count,
                "degraded": self.degraded_count,
                "invalid": self.invalid_count,
            },
            "episodes": {
                "active_count": self.active_episodes_count,
                "affected_nodes": self.nodes_with_active_episodes,
            },
            "contagion": {
                "links_count": self.contagion_links_count,
                "cascade_roots": self.cascade_roots,
            },
            "risk_flags": {
                "has_structural_break": self.has_structural_break,
                "has_dependency_break": self.has_dependency_break,
                "requires_attention": self.requires_attention,
            },
            "warnings": self.warnings,
        }


def compute_portfolio_rollup(
    owner_id: str,
    node_states: List[NodeValidityState],
    node_graph: NodeGraph,
    weights: Optional[Dict[str, float]] = None,
) -> PortfolioValidityRollup:
    """
    Compute portfolio-level validity rollup.
    """
    now = datetime.now(timezone.utc)
    
    if not node_states:
        return PortfolioValidityRollup(
            rollup_id=str(uuid.uuid4()),
            owner_id=owner_id,
            computed_at=now,
            node_ids=[],
            node_count=0,
            min_validity=0,
            avg_validity=0,
            weighted_validity=0,
            valid_count=0,
            degraded_count=0,
            invalid_count=0,
            active_episodes_count=0,
            nodes_with_active_episodes=[],
            contagion_links_count=0,
            cascade_roots=[],
            has_structural_break=False,
            has_dependency_break=False,
            requires_attention=False,
        )
    
    # Basic stats
    scores = [n.validity_score for n in node_states]
    min_val = min(scores)
    avg_val = np.mean(scores)
    
    # Weighted average
    if weights:
        total_weight = sum(weights.get(n.node_id, 1.0) for n in node_states)
        weighted = sum(
            n.validity_score * weights.get(n.node_id, 1.0) / total_weight
            for n in node_states
        )
    else:
        weighted = avg_val
    
    # State counts
    valid_count = sum(1 for n in node_states if n.validity_state == "VALID")
    degraded_count = sum(1 for n in node_states if n.validity_state == "DEGRADED")
    invalid_count = sum(1 for n in node_states if n.validity_state == "INVALID")
    
    # Active episodes
    active_episodes = [n for n in node_states if n.has_active_episode]
    nodes_with_episodes = [n.node_id for n in active_episodes]
    
    # Contagion analysis
    cascade_roots = []
    contagion_count = 0
    
    for n in active_episodes:
        if n.active_episode_id:
            contagion = node_graph.get_episode_contagion(n.active_episode_id)
            if contagion:
                if contagion.is_root_of_cascade:
                    cascade_roots.append(n.active_episode_id)
                contagion_count += len(contagion.propagated_to_episode_ids)
    
    # Risk flags (would check episode FMs in production)
    has_structural = invalid_count > 0
    has_dependency = any(n.trust_penalty > 30 for n in node_states)
    requires_attention = invalid_count > 0 or degraded_count > len(node_states) * 0.3
    
    # Warnings
    warnings = []
    if invalid_count > 0:
        warnings.append(f"🔴 {invalid_count} node(s) INVALID")
    if cascade_roots:
        warnings.append(f"⚠️ {len(cascade_roots)} cascade root(s) detected")
    if min_val < 30:
        warnings.append(f"⚠️ Minimum validity critically low: {min_val:.0f}")
    
    return PortfolioValidityRollup(
        rollup_id=str(uuid.uuid4()),
        owner_id=owner_id,
        computed_at=now,
        node_ids=[n.node_id for n in node_states],
        node_count=len(node_states),
        min_validity=min_val,
        avg_validity=float(avg_val),
        weighted_validity=float(weighted),
        valid_count=valid_count,
        degraded_count=degraded_count,
        invalid_count=invalid_count,
        active_episodes_count=len(active_episodes),
        nodes_with_active_episodes=nodes_with_episodes,
        contagion_links_count=contagion_count,
        cascade_roots=cascade_roots,
        has_structural_break=has_structural,
        has_dependency_break=has_dependency,
        requires_attention=requires_attention,
        warnings=warnings,
    )


# =============================================================================
# TESTS
# =============================================================================

def test_node_graph():
    """Test node graph construction and queries."""
    graph = NodeGraph()
    
    # Register nodes
    graph.register_node("momentum", NodeType.FACTOR, "owner1")
    graph.register_node("value", NodeType.FACTOR, "owner1")
    graph.register_node("mv_pair", NodeType.PAIR, "owner1")
    
    # Add relationships
    graph.add_relationship("momentum", "mv_pair", "depends_on", 0.8)
    graph.add_relationship("value", "mv_pair", "depends_on", 0.8)
    graph.add_relationship("momentum", "value", "correlated", 0.5)
    
    # Test queries
    deps = graph.get_dependent_nodes("momentum")
    assert "mv_pair" in deps
    
    related = graph.get_related_nodes("momentum")
    assert len(related) == 2
    
    print(f"Nodes registered: {len(graph._nodes)}")
    print(f"Momentum dependents: {deps}")
    print(f"Momentum related: {related}")
    print("✓ test_node_graph passed")


def test_contagion_detection():
    """Test contagion link detection."""
    from .audit_infrastructure import FailureModeEntry, FailureModeRole
    
    graph = NodeGraph()
    graph.register_node("momentum", NodeType.FACTOR, "owner1")
    graph.register_node("value", NodeType.FACTOR, "owner1")
    graph.add_relationship("momentum", "value", "correlated", 0.7)
    
    now = datetime.now(timezone.utc)
    
    # Create source episode (momentum broke first)
    source_ep = EpisodeRecord.create("owner1", "momentum")
    source_ep.start_time = now - timedelta(hours=12)
    source_ep.failure_modes = [
        FailureModeEntry(
            fm_code="FM4_STRUCTURAL_BREAK",
            role=FailureModeRole.ROOT,
            onset_time=source_ep.start_time,
            peak_severity=75,
        )
    ]
    
    # Create target episode (value broke later)
    target_ep = EpisodeRecord.create("owner1", "value")
    target_ep.start_time = now - timedelta(hours=6)
    target_ep.failure_modes = [
        FailureModeEntry(
            fm_code="FM1_VARIANCE_REGIME",
            role=FailureModeRole.ROOT,
            onset_time=target_ep.start_time,
            peak_severity=55,
        )
    ]
    
    # Register episodes
    graph.register_episode(source_ep.episode_id, "momentum")
    graph.register_episode(target_ep.episode_id, "value")
    
    # Detect propagation
    detector = PropagationDetector(graph)
    link = detector.detect_propagation(source_ep, target_ep)
    
    assert link is not None
    print(f"Link detected: {link.contagion_type.value}")
    print(f"Lag: {link.lag_hours:.1f}h")
    print(f"Confidence: {link.confidence:.2f}")
    
    # Add to graph
    graph.add_contagion_link(link)
    
    # Query contagion
    contagion = graph.get_episode_contagion(target_ep.episode_id)
    assert len(contagion.caused_by_episode_ids) == 1
    print(f"Target caused by: {contagion.caused_by_episode_ids}")
    
    print("✓ test_contagion_detection passed")


def test_portfolio_rollup():
    """Test portfolio validity rollup."""
    graph = NodeGraph()
    
    # Create node states
    states = [
        NodeValidityState("momentum", 45, "DEGRADED", True, "ep1", 15),
        NodeValidityState("value", 72, "VALID", False, None, 0),
        NodeValidityState("quality", 25, "INVALID", True, "ep2", 40),
        NodeValidityState("size", 80, "VALID", False, None, 0),
    ]
    
    rollup = compute_portfolio_rollup("owner1", states, graph)
    
    print(f"Portfolio validity:")
    print(f"  Min: {rollup.min_validity:.0f}")
    print(f"  Avg: {rollup.avg_validity:.0f}")
    print(f"  Valid/Degraded/Invalid: {rollup.valid_count}/{rollup.degraded_count}/{rollup.invalid_count}")
    print(f"  Warnings: {rollup.warnings}")
    
    assert rollup.invalid_count == 1
    assert rollup.requires_attention == True
    assert len(rollup.warnings) > 0
    
    print("✓ test_portfolio_rollup passed")


def test_what_else_broke():
    """Test 'what else broke' query."""
    graph = NodeGraph()
    
    graph.register_node("momentum", NodeType.FACTOR, "owner1")
    graph.register_node("value", NodeType.FACTOR, "owner1")
    graph.add_relationship("momentum", "value", "correlated", 0.6)
    
    graph.register_episode("ep_mom", "momentum")
    graph.register_episode("ep_val", "value")
    
    # Create contagion link
    link = ContagionLink(
        link_id="link1",
        source_episode_id="ep_mom",
        target_episode_id="ep_val",
        source_node_id="momentum",
        target_node_id="value",
        contagion_type=ContagionType.CORRELATION_SHIFT,
        direction=PropagationDirection.DOWNSTREAM,
        source_onset=datetime.now(timezone.utc) - timedelta(hours=10),
        target_onset=datetime.now(timezone.utc) - timedelta(hours=5),
        lag_hours=5,
        confidence=0.75,
    )
    graph.add_contagion_link(link)
    
    # Query
    result = graph.query_what_else_broke("value", "ep_val")
    
    print(f"What else broke query result:")
    print(f"  Upstream causes: {result.get('upstream_causes', [])}")
    print(f"  Contagion links: {len(result.get('contagion_links', []))}")
    
    assert "ep_mom" in result.get("upstream_causes", [])
    
    print("✓ test_what_else_broke passed")


def run_all_contagion_tests():
    print("\n" + "=" * 60)
    print("CROSS-NODE CONTAGION TESTS")
    print("=" * 60 + "\n")
    
    test_node_graph()
    print()
    test_contagion_detection()
    print()
    test_portfolio_rollup()
    print()
    test_what_else_broke()
    
    print("\n" + "=" * 60)
    print("ALL CONTAGION TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    run_all_contagion_tests()
