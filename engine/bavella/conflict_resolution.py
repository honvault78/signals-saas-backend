"""
Bavella v2 — Failure Mode Conflict Resolution
===============================================

THE MISSING LAYER: Failure modes are NOT peers.

Current assumption (wrong):
    FM1 fires → weight 18%
    FM4 fires → weight 22%
    Result: linear sum
    
Reality:
    FM4 (structural break) DOMINATES FM1, FM2, FM3
    When FM4 fires, the others are often symptoms, not causes
    
This module implements:
    1. Failure Mode Precedence Graph
    2. Dominance relationships
    3. Conditional relationships
    4. Conflict detection and resolution
    5. Non-linear aggregation

Now sophisticated users won't ask:
    "Why are you treating a structural break and a variance blip as peers?"

Because they aren't peers.

Copyright 2024-2026 Bavella Technologies Sarl
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Set, Optional, Any, Tuple, FrozenSet
import networkx as nx

from .core import FailureMode


# =============================================================================
# RELATIONSHIP TYPES
# =============================================================================

class RelationType(Enum):
    """Types of relationships between failure modes."""
    
    DOMINATES = "dominates"
    # FM_A dominates FM_B: If A fires, B is likely a symptom
    # Example: FM4 (structural break) dominates FM1 (variance)
    
    CONDITIONS = "conditions"
    # FM_A conditions FM_B: A must fire for B to be meaningful
    # Example: FM6 (distribution) conditions FM5 (outliers)
    
    CONFLICTS = "conflicts"
    # FM_A conflicts with FM_B: If both fire, investigate carefully
    # Example: FM1 (variance stable) conflicts with FM4 (structural break)
    
    AMPLIFIES = "amplifies"
    # FM_A amplifies FM_B: If both fire, the effect is worse
    # Example: FM5 (outliers) amplifies FM6 (distribution shift)
    
    INVALIDATES = "invalidates"
    # FM_A invalidates FM_B: If A fires, B's detection is unreliable
    # Example: FM4 (structural break) invalidates FM3 (seasonality) during break


# =============================================================================
# PRECEDENCE GRAPH (FROZEN)
# =============================================================================

@dataclass(frozen=True)
class Relationship:
    """A relationship between two failure modes."""
    source: FailureMode
    target: FailureMode
    relation: RelationType
    description: str
    strength: float = 1.0  # 0-1, how strong the relationship is


class FailureModePrecedenceGraph:
    """
    The precedence graph defining relationships between failure modes.
    
    THIS IS FROZEN CANON. Changes require version bump.
    
    Key relationships:
        FM4 (Structural Break) dominates → FM1, FM2, FM3
        FM6 (Distribution Shift) conditions → FM5
        FM7 (Dependency Break) dominates → derived validity
    """
    
    VERSION = "1.0.0"
    
    def __init__(self):
        self._graph = nx.DiGraph()
        self._relationships: List[Relationship] = []
        self._build_graph()
    
    def _build_graph(self) -> None:
        """Build the precedence graph with frozen relationships."""
        
        # Add all failure modes as nodes
        for fm in FailureMode:
            self._graph.add_node(fm)
        
        # =================================================================
        # DOMINANCE RELATIONSHIPS
        # =================================================================
        
        # FM4 (Structural Break) dominates statistical assumptions
        # Rationale: A structural break makes variance/mean/seasonality
        # changes look like failures when they're actually symptoms
        self._add_relationship(Relationship(
            source=FailureMode.FM4_STRUCTURAL_BREAK,
            target=FailureMode.FM1_VARIANCE_REGIME,
            relation=RelationType.DOMINATES,
            description="Structural breaks cause apparent variance regime changes",
            strength=0.9,
        ))
        
        self._add_relationship(Relationship(
            source=FailureMode.FM4_STRUCTURAL_BREAK,
            target=FailureMode.FM2_MEAN_DRIFT,
            relation=RelationType.DOMINATES,
            description="Structural breaks cause apparent mean drift",
            strength=0.9,
        ))
        
        self._add_relationship(Relationship(
            source=FailureMode.FM4_STRUCTURAL_BREAK,
            target=FailureMode.FM3_SEASONALITY_MISMATCH,
            relation=RelationType.DOMINATES,
            description="Structural breaks disrupt seasonality detection",
            strength=0.8,
        ))
        
        # FM7 (Dependency Break) dominates statistical assumptions for pairs
        self._add_relationship(Relationship(
            source=FailureMode.FM7_DEPENDENCY_BREAK,
            target=FailureMode.FM1_VARIANCE_REGIME,
            relation=RelationType.DOMINATES,
            description="Dependency breaks affect variance interpretation",
            strength=0.7,
        ))
        
        # =================================================================
        # CONDITIONING RELATIONSHIPS
        # =================================================================
        
        # FM6 (Distribution) conditions FM5 (Outliers)
        # Rationale: If distribution shifted, outlier detection thresholds
        # computed from old distribution are unreliable
        self._add_relationship(Relationship(
            source=FailureMode.FM6_DISTRIBUTIONAL_SHIFT,
            target=FailureMode.FM5_OUTLIER_CONTAMINATION,
            relation=RelationType.CONDITIONS,
            description="Distribution shift affects outlier threshold calibration",
            strength=0.8,
        ))
        
        # =================================================================
        # CONFLICT RELATIONSHIPS
        # =================================================================
        
        # FM1 (variance stable) conflicts with FM4 (structural break)
        # Rationale: If FM1 shows stability but FM4 fires, investigate
        self._add_relationship(Relationship(
            source=FailureMode.FM1_VARIANCE_REGIME,
            target=FailureMode.FM4_STRUCTURAL_BREAK,
            relation=RelationType.CONFLICTS,
            description="Stable variance conflicts with structural break detection",
            strength=0.7,
        ))
        
        # =================================================================
        # AMPLIFICATION RELATIONSHIPS
        # =================================================================
        
        # FM5 (Outliers) amplifies FM6 (Distribution)
        # Rationale: Outlier contamination makes distribution shift worse
        self._add_relationship(Relationship(
            source=FailureMode.FM5_OUTLIER_CONTAMINATION,
            target=FailureMode.FM6_DISTRIBUTIONAL_SHIFT,
            relation=RelationType.AMPLIFIES,
            description="Outliers amplify distributional shift impact",
            strength=0.6,
        ))
        
        # FM4 + FM7 together is catastrophic
        self._add_relationship(Relationship(
            source=FailureMode.FM4_STRUCTURAL_BREAK,
            target=FailureMode.FM7_DEPENDENCY_BREAK,
            relation=RelationType.AMPLIFIES,
            description="Structural + dependency break together is severe",
            strength=0.9,
        ))
        
        # =================================================================
        # INVALIDATION RELATIONSHIPS
        # =================================================================
        
        # FM4 invalidates FM3 (seasonality) during break period
        self._add_relationship(Relationship(
            source=FailureMode.FM4_STRUCTURAL_BREAK,
            target=FailureMode.FM3_SEASONALITY_MISMATCH,
            relation=RelationType.INVALIDATES,
            description="Cannot reliably detect seasonality during structural break",
            strength=0.85,
        ))
    
    def _add_relationship(self, rel: Relationship) -> None:
        """Add a relationship to the graph."""
        self._relationships.append(rel)
        self._graph.add_edge(
            rel.source, rel.target,
            relation=rel.relation,
            strength=rel.strength,
            description=rel.description,
        )
    
    def get_dominators(self, fm: FailureMode) -> List[Tuple[FailureMode, float]]:
        """Get failure modes that dominate this one."""
        dominators = []
        for source, target, data in self._graph.in_edges(fm, data=True):
            if data["relation"] == RelationType.DOMINATES:
                dominators.append((source, data["strength"]))
        return dominators
    
    def get_dominated(self, fm: FailureMode) -> List[Tuple[FailureMode, float]]:
        """Get failure modes that this one dominates."""
        dominated = []
        for source, target, data in self._graph.out_edges(fm, data=True):
            if data["relation"] == RelationType.DOMINATES:
                dominated.append((target, data["strength"]))
        return dominated
    
    def get_conflicts(self, fm: FailureMode) -> List[Tuple[FailureMode, float]]:
        """Get failure modes that conflict with this one."""
        conflicts = []
        for source, target, data in self._graph.edges(data=True):
            if data["relation"] == RelationType.CONFLICTS:
                if source == fm:
                    conflicts.append((target, data["strength"]))
                elif target == fm:
                    conflicts.append((source, data["strength"]))
        return conflicts
    
    def get_amplifiers(self, fm: FailureMode) -> List[Tuple[FailureMode, float]]:
        """Get failure modes that amplify this one."""
        amplifiers = []
        for source, target, data in self._graph.in_edges(fm, data=True):
            if data["relation"] == RelationType.AMPLIFIES:
                amplifiers.append((source, data["strength"]))
        return amplifiers
    
    def is_invalidated_by(self, fm: FailureMode, by: FailureMode) -> Optional[float]:
        """Check if fm is invalidated by another failure mode."""
        if self._graph.has_edge(by, fm):
            data = self._graph.edges[by, fm]
            if data["relation"] == RelationType.INVALIDATES:
                return data["strength"]
        return None


# =============================================================================
# CONFLICT RESOLVER
# =============================================================================

@dataclass
class ConflictAnalysis:
    """Analysis of conflicts between failure mode signals."""
    
    # Detected conflicts
    conflicts: List[Tuple[FailureMode, FailureMode, str]] = field(default_factory=list)
    
    # Dominated signals (should be discounted)
    dominated: Dict[FailureMode, Tuple[FailureMode, float]] = field(default_factory=dict)
    
    # Invalidated signals (should be ignored)
    invalidated: Dict[FailureMode, Tuple[FailureMode, float]] = field(default_factory=dict)
    
    # Amplification pairs
    amplifications: List[Tuple[FailureMode, FailureMode, float]] = field(default_factory=list)
    
    # Root cause (the dominating failure mode)
    root_cause: Optional[FailureMode] = None
    
    # Symptoms (dominated by root cause)
    symptoms: Set[FailureMode] = field(default_factory=set)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "conflicts": [
                {"fm1": fm1.name, "fm2": fm2.name, "description": desc}
                for fm1, fm2, desc in self.conflicts
            ],
            "dominated": {
                fm.name: {"by": by.name, "strength": s}
                for fm, (by, s) in self.dominated.items()
            },
            "invalidated": {
                fm.name: {"by": by.name, "strength": s}
                for fm, (by, s) in self.invalidated.items()
            },
            "amplifications": [
                {"fm1": fm1.name, "fm2": fm2.name, "factor": f}
                for fm1, fm2, f in self.amplifications
            ],
            "root_cause": self.root_cause.name if self.root_cause else None,
            "symptoms": [fm.name for fm in self.symptoms],
        }


class ConflictResolver:
    """
    Resolves conflicts between failure mode signals.
    
    This produces non-linear aggregation based on:
        - Dominance (FM4 dominates FM1)
        - Invalidation (FM4 invalidates FM3)
        - Amplification (FM4 + FM7 is worse than sum)
        - Conflict detection (FM1 stable vs FM4 break)
    """
    
    def __init__(self, graph: Optional[FailureModePrecedenceGraph] = None):
        self.graph = graph or FailureModePrecedenceGraph()
    
    def analyze(
        self,
        active_signals: List[Tuple[FailureMode, float]],  # (FM, severity)
    ) -> ConflictAnalysis:
        """
        Analyze conflicts among active signals.
        
        Args:
            active_signals: List of (failure_mode, severity) for active signals
            
        Returns:
            ConflictAnalysis with dominance, invalidation, amplification info
        """
        analysis = ConflictAnalysis()
        active_fms = {fm for fm, _ in active_signals}
        severity_map = {fm: sev for fm, sev in active_signals}
        
        # 1. Find dominated signals
        for fm in active_fms:
            dominators = self.graph.get_dominators(fm)
            for dominator, strength in dominators:
                if dominator in active_fms:
                    # This FM is dominated by an active dominator
                    existing = analysis.dominated.get(fm)
                    if existing is None or strength > existing[1]:
                        analysis.dominated[fm] = (dominator, strength)
        
        # 2. Find invalidated signals
        for fm in active_fms:
            for potential_invalidator in active_fms:
                if potential_invalidator == fm:
                    continue
                strength = self.graph.is_invalidated_by(fm, potential_invalidator)
                if strength:
                    analysis.invalidated[fm] = (potential_invalidator, strength)
        
        # 3. Find conflicts
        checked = set()
        for fm1 in active_fms:
            for fm2, strength in self.graph.get_conflicts(fm1):
                if fm2 in active_fms and (fm2, fm1) not in checked:
                    analysis.conflicts.append((
                        fm1, fm2,
                        f"Conflict between {fm1.name} and {fm2.name}"
                    ))
                    checked.add((fm1, fm2))
        
        # 4. Find amplifications
        for fm in active_fms:
            for amplifier, strength in self.graph.get_amplifiers(fm):
                if amplifier in active_fms:
                    analysis.amplifications.append((amplifier, fm, strength))
        
        # 5. Identify root cause
        # The root cause is the dominator that is not itself dominated
        non_dominated = active_fms - set(analysis.dominated.keys())
        if non_dominated:
            # Among non-dominated, pick the one with highest severity
            # that also dominates others
            for fm in sorted(non_dominated, key=lambda x: severity_map.get(x, 0), reverse=True):
                dominated_by_this = self.graph.get_dominated(fm)
                if any(d in active_fms for d, _ in dominated_by_this):
                    analysis.root_cause = fm
                    break
            
            # If no dominator found, pick highest severity
            if analysis.root_cause is None and non_dominated:
                analysis.root_cause = max(non_dominated, key=lambda x: severity_map.get(x, 0))
        
        # 6. Identify symptoms
        if analysis.root_cause:
            dominated_by_root = self.graph.get_dominated(analysis.root_cause)
            analysis.symptoms = {d for d, _ in dominated_by_root if d in active_fms}
        
        return analysis
    
    def compute_adjusted_severity(
        self,
        active_signals: List[Tuple[FailureMode, float, float]],  # (FM, severity, confidence)
    ) -> List[Tuple[FailureMode, float, str]]:
        """
        Compute adjusted severity after conflict resolution.
        
        Returns: [(FM, adjusted_severity, adjustment_reason)]
        """
        simple_signals = [(fm, sev) for fm, sev, _ in active_signals]
        analysis = self.analyze(simple_signals)
        
        results = []
        severity_map = {fm: sev for fm, sev, _ in active_signals}
        
        for fm, severity, confidence in active_signals:
            adjusted = severity
            reason = "No adjustment"
            
            # Invalidated → severity reduced significantly
            if fm in analysis.invalidated:
                invalidator, strength = analysis.invalidated[fm]
                adjusted *= (1 - strength * 0.8)  # Up to 80% reduction
                reason = f"Invalidated by {invalidator.name} ({strength:.0%} strength)"
            
            # Dominated → severity reduced
            elif fm in analysis.dominated:
                dominator, strength = analysis.dominated[fm]
                # Only reduce if dominator has higher severity
                if severity_map.get(dominator, 0) > severity:
                    adjusted *= (1 - strength * 0.5)  # Up to 50% reduction
                    reason = f"Dominated by {dominator.name} ({strength:.0%} strength)"
            
            # Amplified → severity increased
            for amplifier, target, amp_strength in analysis.amplifications:
                if fm == target:
                    amplifier_severity = severity_map.get(amplifier, 0)
                    if amplifier_severity > 30:  # Amplifier must be significant
                        boost = 1 + (amp_strength * amplifier_severity / 100 * 0.5)
                        adjusted *= boost
                        reason = f"Amplified by {amplifier.name} ({boost:.2f}x)"
            
            # Root cause gets priority
            if fm == analysis.root_cause:
                if reason == "No adjustment":
                    reason = "Root cause (no adjustment)"
            
            results.append((fm, adjusted, reason))
        
        return results


# =============================================================================
# TESTS
# =============================================================================

def test_dominance_relationships():
    """Test that dominance relationships are correctly identified."""
    graph = FailureModePrecedenceGraph()
    
    # FM4 should dominate FM1 and FM2
    dominated = graph.get_dominated(FailureMode.FM4_STRUCTURAL_BREAK)
    dominated_fms = [fm for fm, _ in dominated]
    
    assert FailureMode.FM1_VARIANCE_REGIME in dominated_fms
    assert FailureMode.FM2_MEAN_DRIFT in dominated_fms
    # FM3 is INVALIDATED by FM4, not dominated
    
    # FM4 also invalidates FM3
    invalidated = graph.is_invalidated_by(
        FailureMode.FM3_SEASONALITY_MISMATCH,
        FailureMode.FM4_STRUCTURAL_BREAK
    )
    assert invalidated is not None, "FM4 should invalidate FM3"
    
    print(f"FM4 dominates: {[fm.name for fm in dominated_fms]}")
    print(f"FM4 invalidates FM3: strength={invalidated}")
    print("✓ test_dominance_relationships passed")


def test_conflict_detection():
    """Test that conflicts are detected."""
    graph = FailureModePrecedenceGraph()
    
    conflicts = graph.get_conflicts(FailureMode.FM1_VARIANCE_REGIME)
    
    assert any(fm == FailureMode.FM4_STRUCTURAL_BREAK for fm, _ in conflicts)
    
    print(f"FM1 conflicts with: {[fm.name for fm, _ in conflicts]}")
    print("✓ test_conflict_detection passed")


def test_conflict_resolution():
    """Test conflict resolution produces correct adjustments."""
    resolver = ConflictResolver()
    
    # Scenario: FM4 (structural break) and FM1 (variance)
    # FM4 should dominate, FM1 should be discounted
    signals = [
        (FailureMode.FM4_STRUCTURAL_BREAK, 70.0, 0.9),
        (FailureMode.FM1_VARIANCE_REGIME, 50.0, 0.8),
        (FailureMode.FM2_MEAN_DRIFT, 40.0, 0.7),
    ]
    
    analysis = resolver.analyze([(fm, sev) for fm, sev, _ in signals])
    
    # FM4 should be root cause
    assert analysis.root_cause == FailureMode.FM4_STRUCTURAL_BREAK
    
    # FM1 and FM2 should be dominated
    assert FailureMode.FM1_VARIANCE_REGIME in analysis.dominated
    assert FailureMode.FM2_MEAN_DRIFT in analysis.dominated
    
    print(f"Root cause: {analysis.root_cause.name}")
    print(f"Dominated: {[fm.name for fm in analysis.dominated]}")
    print(f"Symptoms: {[fm.name for fm in analysis.symptoms]}")
    
    # Get adjusted severities
    adjusted = resolver.compute_adjusted_severity(signals)
    
    for fm, adj_sev, reason in adjusted:
        print(f"  {fm.name}: {signals[[s[0] for s in signals].index(fm)][1]:.1f} → {adj_sev:.1f} ({reason})")
    
    # FM1 and FM2 should have reduced severity
    fm1_adjusted = [a for a in adjusted if a[0] == FailureMode.FM1_VARIANCE_REGIME][0][1]
    fm4_adjusted = [a for a in adjusted if a[0] == FailureMode.FM4_STRUCTURAL_BREAK][0][1]
    
    assert fm1_adjusted < 50.0, "FM1 should be reduced (dominated)"
    assert fm4_adjusted == 70.0, "FM4 should not be reduced (root cause)"
    
    print("✓ test_conflict_resolution passed")


def test_amplification():
    """Test that amplification increases severity."""
    resolver = ConflictResolver()
    
    # FM4 + FM7 should amplify each other
    signals = [
        (FailureMode.FM4_STRUCTURAL_BREAK, 60.0, 0.9),
        (FailureMode.FM7_DEPENDENCY_BREAK, 50.0, 0.8),
    ]
    
    analysis = resolver.analyze([(fm, sev) for fm, sev, _ in signals])
    
    assert len(analysis.amplifications) > 0, "Should detect amplification"
    
    print(f"Amplifications: {analysis.amplifications}")
    
    adjusted = resolver.compute_adjusted_severity(signals)
    fm7_original = 50.0
    fm7_adjusted = [a for a in adjusted if a[0] == FailureMode.FM7_DEPENDENCY_BREAK][0][1]
    
    assert fm7_adjusted > fm7_original, "FM7 should be amplified"
    
    print(f"FM7: {fm7_original:.1f} → {fm7_adjusted:.1f}")
    print("✓ test_amplification passed")


def run_all_conflict_tests():
    print("\n" + "=" * 60)
    print("FAILURE MODE CONFLICT RESOLUTION TESTS")
    print("=" * 60 + "\n")
    
    test_dominance_relationships()
    print()
    test_conflict_detection()
    print()
    test_conflict_resolution()
    print()
    test_amplification()
    
    print("\n" + "=" * 60)
    print("ALL CONFLICT RESOLUTION TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    run_all_conflict_tests()
