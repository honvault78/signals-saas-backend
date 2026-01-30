"""
Bavella v2 — Institutional Fingerprint & Match Results
=======================================================

THE "HERE'S WHY SIMILARITY IS 0.84" LAYER

Instead of: similarity=0.84
We get:     similarity=0.84 with breakdown:
            - fm_overlap: 0.92 (weight: 0.30)
            - root_match: 1.00 (weight: 0.15)
            - onset_order: 0.78 (weight: 0.10)
            - severity_shape: 0.82 (weight: 0.20)
            - duration: 0.90 (weight: 0.10)
            - context: 0.65 (weight: 0.15)

This module defines:
    1. InstitutionalFingerprint - with causal ordering + onset order
    2. MatchResult - immutable match with full breakdown
    3. MatchBreakdown - per-dimension similarity
    4. SimilarityMatcher - produces auditable matches

Copyright 2024-2026 Bavella Technologies Sarl
"""

from __future__ import annotations

import hashlib
import json
import uuid
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, FrozenSet

from .audit_infrastructure import (
    BavellaVersions, EvidenceType, EvidenceArtifact,
    FailureModeRole, CausalOrdering, RecoveryType,
)
from .severity_curve import SeverityCurveDescriptor, ShapeClass, compute_curve_similarity


# =============================================================================
# DURATION BUCKETS
# =============================================================================

class DurationBucket(Enum):
    """Canonical duration categories."""
    FLASH = "flash"      # < 1 day
    SHORT = "short"      # 1-7 days
    MEDIUM = "medium"    # 7-30 days
    LONG = "long"        # 30-90 days
    EXTENDED = "extended"  # > 90 days
    
    @classmethod
    def from_hours(cls, hours: float) -> "DurationBucket":
        days = hours / 24
        if days < 1:
            return cls.FLASH
        elif days < 7:
            return cls.SHORT
        elif days < 30:
            return cls.MEDIUM
        elif days < 90:
            return cls.LONG
        else:
            return cls.EXTENDED


# =============================================================================
# INSTITUTIONAL FINGERPRINT
# =============================================================================

@dataclass(frozen=True)
class FMSignature:
    """
    Failure mode signature with roles and ordering.
    """
    # Which FMs fired
    fm_set: FrozenSet[str]
    
    # Which is root cause
    root_fm: str
    
    # Roles for each FM
    fm_roles: Tuple[Tuple[str, FailureModeRole], ...]
    
    # Onset order (first FM first)
    onset_order: Tuple[str, ...]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "fm_set": sorted(list(self.fm_set)),
            "root_fm": self.root_fm,
            "fm_roles": {fm: role.value for fm, role in self.fm_roles},
            "onset_order": list(self.onset_order),
        }
    
    @property
    def symptom_fms(self) -> List[str]:
        return [fm for fm, role in self.fm_roles if role == FailureModeRole.SYMPTOM]


@dataclass(frozen=True)
class ContextFeatures:
    """
    Context at episode onset (huge moat potential).
    """
    # Volatility regime
    volatility_percentile: Optional[float] = None  # Where in historical vol distribution
    
    # Drawdown context
    drawdown_percentile: Optional[float] = None
    
    # Correlation context (for pairs/factors)
    correlation_shift: Optional[float] = None
    
    # Liquidity proxy
    liquidity_percentile: Optional[float] = None
    
    # Market regime
    market_regime: Optional[str] = None  # "risk_on", "risk_off", "transition"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "volatility_percentile": self.volatility_percentile,
            "drawdown_percentile": self.drawdown_percentile,
            "correlation_shift": self.correlation_shift,
            "liquidity_percentile": self.liquidity_percentile,
            "market_regime": self.market_regime,
        }
    
    def to_vector(self) -> np.ndarray:
        """Convert to vector for similarity (with defaults for None)."""
        return np.array([
            self.volatility_percentile if self.volatility_percentile is not None else 0.5,
            self.drawdown_percentile if self.drawdown_percentile is not None else 0.5,
            abs(self.correlation_shift) if self.correlation_shift is not None else 0.0,
            self.liquidity_percentile if self.liquidity_percentile is not None else 0.5,
        ])


@dataclass(frozen=True)
class InstitutionalFingerprint:
    """
    The fingerprint used for matching.
    
    This is what makes "similar episodes" defensible.
    Every dimension is explicit and versioned.
    """
    # Identity
    fingerprint_id: str
    fingerprint_version: str = BavellaVersions.FINGERPRINT_VERSION
    fingerprint_config_hash: str = ""
    
    # Source
    episode_id: Optional[str] = None  # None for "current hypothetical"
    
    # FM signature
    fm_signature: Optional[FMSignature] = None
    
    # Severity shape
    severity_curve_id: Optional[str] = None
    severity_shape_class: ShapeClass = ShapeClass.INDETERMINATE
    
    # Duration
    duration_hours: float = 0.0
    duration_bucket: DurationBucket = DurationBucket.SHORT
    
    # Irreversibility
    irreversibility_flag: bool = False
    
    # Recovery (if known)
    recovery_type: Optional[RecoveryType] = None
    
    # Context
    context_features: Optional[ContextFeatures] = None
    
    # Created
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "fingerprint_id": self.fingerprint_id,
            "fingerprint_version": self.fingerprint_version,
            "fingerprint_config_hash": self.fingerprint_config_hash,
            "episode_id": self.episode_id,
            "fm_signature": self.fm_signature.to_dict() if self.fm_signature else None,
            "severity_curve_id": self.severity_curve_id,
            "severity_shape_class": self.severity_shape_class.value,
            "duration_hours": self.duration_hours,
            "duration_bucket": self.duration_bucket.value,
            "irreversibility_flag": self.irreversibility_flag,
            "recovery_type": self.recovery_type.value if self.recovery_type else None,
            "context_features": self.context_features.to_dict() if self.context_features else None,
            "created_at": self.created_at.isoformat(),
        }
    
    @classmethod
    def create(
        cls,
        fm_set: List[str],
        root_fm: str,
        fm_roles: List[Tuple[str, FailureModeRole]],
        onset_order: List[str],
        duration_hours: float,
        severity_shape_class: ShapeClass = ShapeClass.INDETERMINATE,
        severity_curve_id: Optional[str] = None,
        irreversibility_flag: bool = False,
        recovery_type: Optional[RecoveryType] = None,
        context_features: Optional[ContextFeatures] = None,
        episode_id: Optional[str] = None,
    ) -> "InstitutionalFingerprint":
        """Create fingerprint with deterministic ID."""
        fm_sig = FMSignature(
            fm_set=frozenset(fm_set),
            root_fm=root_fm,
            fm_roles=tuple(fm_roles),
            onset_order=tuple(onset_order),
        )
        
        # Compute deterministic fingerprint ID
        content = {
            "fm_set": sorted(fm_set),
            "root_fm": root_fm,
            "onset_order": onset_order,
            "duration_hours": duration_hours,
            "shape_class": severity_shape_class.value,
            "irreversibility": irreversibility_flag,
        }
        fp_id = hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()[:16]
        
        return cls(
            fingerprint_id=fp_id,
            episode_id=episode_id,
            fm_signature=fm_sig,
            severity_curve_id=severity_curve_id,
            severity_shape_class=severity_shape_class,
            duration_hours=duration_hours,
            duration_bucket=DurationBucket.from_hours(duration_hours),
            irreversibility_flag=irreversibility_flag,
            recovery_type=recovery_type,
            context_features=context_features,
        )


# =============================================================================
# MATCH BREAKDOWN
# =============================================================================

@dataclass(frozen=True)
class MatchDimension:
    """One dimension of similarity."""
    name: str
    similarity: float  # 0-1
    weight: float
    weighted_contribution: float
    
    # Evidence
    current_value: Any = None
    candidate_value: Any = None
    comparison_method: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "similarity": self.similarity,
            "weight": self.weight,
            "weighted_contribution": self.weighted_contribution,
            "comparison": {
                "current": str(self.current_value) if self.current_value else None,
                "candidate": str(self.candidate_value) if self.candidate_value else None,
                "method": self.comparison_method,
            },
        }


@dataclass(frozen=True)
class MatchBreakdown:
    """
    Complete breakdown of why similarity is what it is.
    
    This is the "nobody can call it vibes" evidence.
    """
    dimensions: Tuple[MatchDimension, ...]
    overall_similarity: float
    
    # Weights used
    weights_version: str = "1.0.0"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "dimensions": [d.to_dict() for d in self.dimensions],
            "overall_similarity": self.overall_similarity,
            "weights_version": self.weights_version,
            "sum_of_weights": sum(d.weight for d in self.dimensions),
        }
    
    def get_top_contributors(self, n: int = 3) -> List[MatchDimension]:
        """Get top N contributing dimensions."""
        sorted_dims = sorted(self.dimensions, key=lambda d: d.weighted_contribution, reverse=True)
        return list(sorted_dims[:n])
    
    def get_weakest_dimensions(self, n: int = 3) -> List[MatchDimension]:
        """Get N weakest dimensions (lowest similarity)."""
        sorted_dims = sorted(self.dimensions, key=lambda d: d.similarity)
        return list(sorted_dims[:n])


# =============================================================================
# MATCH RESULT (immutable)
# =============================================================================

@dataclass(frozen=True)
class MatchResult:
    """
    Immutable match result.
    
    Every narrative response must cite match_result_id.
    """
    # Identity
    match_result_id: str
    
    # Context
    run_id: str
    created_at: datetime
    
    # What we matched
    current_fingerprint_id: str
    candidate_fingerprint_id: str
    candidate_episode_id: str
    
    # Result
    similarity_score: float  # 0-1
    
    # Full breakdown
    breakdown: MatchBreakdown
    
    # Evidence
    evidence_ids: Tuple[str, ...] = ()
    
    # Versions
    matcher_version: str = BavellaVersions.MATCHER_VERSION
    matcher_config_hash: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "match_result_id": self.match_result_id,
            "run_id": self.run_id,
            "created_at": self.created_at.isoformat(),
            "current_fingerprint_id": self.current_fingerprint_id,
            "candidate_fingerprint_id": self.candidate_fingerprint_id,
            "candidate_episode_id": self.candidate_episode_id,
            "similarity_score": self.similarity_score,
            "breakdown": self.breakdown.to_dict(),
            "evidence_ids": list(self.evidence_ids),
            "matcher_version": self.matcher_version,
            "matcher_config_hash": self.matcher_config_hash,
        }


# =============================================================================
# SIMILARITY MATCHER
# =============================================================================

class SimilarityWeights:
    """
    Weights for similarity dimensions. FROZEN - changes require version bump.
    """
    VERSION = "1.0.0"
    
    FM_OVERLAP = 0.25        # Jaccard of FM sets
    ROOT_MATCH = 0.15        # Root cause matches
    ONSET_ORDER = 0.10       # Onset order similarity (Kendall tau)
    SEVERITY_SHAPE = 0.20    # Severity curve similarity
    DURATION = 0.10          # Duration bucket match
    IRREVERSIBILITY = 0.05   # Irreversibility flag match
    RECOVERY = 0.05          # Recovery type match (if known)
    CONTEXT = 0.10           # Context features similarity
    
    @classmethod
    def to_dict(cls) -> Dict[str, float]:
        return {
            "fm_overlap": cls.FM_OVERLAP,
            "root_match": cls.ROOT_MATCH,
            "onset_order": cls.ONSET_ORDER,
            "severity_shape": cls.SEVERITY_SHAPE,
            "duration": cls.DURATION,
            "irreversibility": cls.IRREVERSIBILITY,
            "recovery": cls.RECOVERY,
            "context": cls.CONTEXT,
        }


class InstitutionalMatcher:
    """
    Produces auditable match results.
    """
    
    def __init__(self):
        self.weights_version = SimilarityWeights.VERSION
        self.matcher_version = BavellaVersions.MATCHER_VERSION
    
    def match(
        self,
        current: InstitutionalFingerprint,
        candidate: InstitutionalFingerprint,
        run_id: str,
        severity_curves: Optional[Tuple[SeverityCurveDescriptor, SeverityCurveDescriptor]] = None,
    ) -> MatchResult:
        """
        Compute similarity with full breakdown.
        """
        dimensions = []
        
        # 1. FM overlap (Jaccard)
        fm_sim = self._compute_fm_overlap(current, candidate)
        dimensions.append(MatchDimension(
            name="fm_overlap",
            similarity=fm_sim,
            weight=SimilarityWeights.FM_OVERLAP,
            weighted_contribution=fm_sim * SimilarityWeights.FM_OVERLAP,
            current_value=sorted(current.fm_signature.fm_set) if current.fm_signature else [],
            candidate_value=sorted(candidate.fm_signature.fm_set) if candidate.fm_signature else [],
            comparison_method="jaccard",
        ))
        
        # 2. Root match
        root_sim = self._compute_root_match(current, candidate)
        dimensions.append(MatchDimension(
            name="root_match",
            similarity=root_sim,
            weight=SimilarityWeights.ROOT_MATCH,
            weighted_contribution=root_sim * SimilarityWeights.ROOT_MATCH,
            current_value=current.fm_signature.root_fm if current.fm_signature else None,
            candidate_value=candidate.fm_signature.root_fm if candidate.fm_signature else None,
            comparison_method="exact_match",
        ))
        
        # 3. Onset order
        onset_sim = self._compute_onset_order_similarity(current, candidate)
        dimensions.append(MatchDimension(
            name="onset_order",
            similarity=onset_sim,
            weight=SimilarityWeights.ONSET_ORDER,
            weighted_contribution=onset_sim * SimilarityWeights.ONSET_ORDER,
            current_value=list(current.fm_signature.onset_order) if current.fm_signature else [],
            candidate_value=list(candidate.fm_signature.onset_order) if candidate.fm_signature else [],
            comparison_method="kendall_tau_normalized",
        ))
        
        # 4. Severity shape
        if severity_curves:
            shape_sim, _ = compute_curve_similarity(severity_curves[0], severity_curves[1])
        else:
            # Fall back to shape class match
            shape_sim = 1.0 if current.severity_shape_class == candidate.severity_shape_class else 0.5
        dimensions.append(MatchDimension(
            name="severity_shape",
            similarity=shape_sim,
            weight=SimilarityWeights.SEVERITY_SHAPE,
            weighted_contribution=shape_sim * SimilarityWeights.SEVERITY_SHAPE,
            current_value=current.severity_shape_class.value,
            candidate_value=candidate.severity_shape_class.value,
            comparison_method="curve_similarity" if severity_curves else "shape_class_match",
        ))
        
        # 5. Duration
        dur_sim = 1.0 if current.duration_bucket == candidate.duration_bucket else 0.5
        dimensions.append(MatchDimension(
            name="duration",
            similarity=dur_sim,
            weight=SimilarityWeights.DURATION,
            weighted_contribution=dur_sim * SimilarityWeights.DURATION,
            current_value=current.duration_bucket.value,
            candidate_value=candidate.duration_bucket.value,
            comparison_method="bucket_match",
        ))
        
        # 6. Irreversibility
        irrev_sim = 1.0 if current.irreversibility_flag == candidate.irreversibility_flag else 0.3
        dimensions.append(MatchDimension(
            name="irreversibility",
            similarity=irrev_sim,
            weight=SimilarityWeights.IRREVERSIBILITY,
            weighted_contribution=irrev_sim * SimilarityWeights.IRREVERSIBILITY,
            current_value=current.irreversibility_flag,
            candidate_value=candidate.irreversibility_flag,
            comparison_method="exact_match",
        ))
        
        # 7. Recovery (if both known)
        if current.recovery_type and candidate.recovery_type:
            rec_sim = 1.0 if current.recovery_type == candidate.recovery_type else 0.5
        else:
            rec_sim = 0.5  # Unknown
        dimensions.append(MatchDimension(
            name="recovery",
            similarity=rec_sim,
            weight=SimilarityWeights.RECOVERY,
            weighted_contribution=rec_sim * SimilarityWeights.RECOVERY,
            current_value=current.recovery_type.value if current.recovery_type else "unknown",
            candidate_value=candidate.recovery_type.value if candidate.recovery_type else "unknown",
            comparison_method="exact_match_or_unknown",
        ))
        
        # 8. Context
        ctx_sim = self._compute_context_similarity(current, candidate)
        dimensions.append(MatchDimension(
            name="context",
            similarity=ctx_sim,
            weight=SimilarityWeights.CONTEXT,
            weighted_contribution=ctx_sim * SimilarityWeights.CONTEXT,
            current_value=current.context_features.to_dict() if current.context_features else None,
            candidate_value=candidate.context_features.to_dict() if candidate.context_features else None,
            comparison_method="vector_cosine",
        ))
        
        # Overall
        overall = sum(d.weighted_contribution for d in dimensions)
        
        breakdown = MatchBreakdown(
            dimensions=tuple(dimensions),
            overall_similarity=overall,
            weights_version=self.weights_version,
        )
        
        return MatchResult(
            match_result_id=str(uuid.uuid4()),
            run_id=run_id,
            created_at=datetime.now(timezone.utc),
            current_fingerprint_id=current.fingerprint_id,
            candidate_fingerprint_id=candidate.fingerprint_id,
            candidate_episode_id=candidate.episode_id or "",
            similarity_score=overall,
            breakdown=breakdown,
            matcher_version=self.matcher_version,
        )
    
    def _compute_fm_overlap(
        self, fp1: InstitutionalFingerprint, fp2: InstitutionalFingerprint
    ) -> float:
        """Jaccard similarity of FM sets."""
        if not fp1.fm_signature or not fp2.fm_signature:
            return 0.0
        
        set1 = fp1.fm_signature.fm_set
        set2 = fp2.fm_signature.fm_set
        
        if not set1 and not set2:
            return 1.0
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def _compute_root_match(
        self, fp1: InstitutionalFingerprint, fp2: InstitutionalFingerprint
    ) -> float:
        """Check if root cause matches."""
        if not fp1.fm_signature or not fp2.fm_signature:
            return 0.5
        
        return 1.0 if fp1.fm_signature.root_fm == fp2.fm_signature.root_fm else 0.0
    
    def _compute_onset_order_similarity(
        self, fp1: InstitutionalFingerprint, fp2: InstitutionalFingerprint
    ) -> float:
        """Compute onset order similarity (normalized Kendall tau)."""
        if not fp1.fm_signature or not fp2.fm_signature:
            return 0.5
        
        order1 = fp1.fm_signature.onset_order
        order2 = fp2.fm_signature.onset_order
        
        if not order1 or not order2:
            return 0.5
        
        # Find common FMs
        common = set(order1) & set(order2)
        if len(common) < 2:
            return 0.5 if common else 0.0
        
        # Extract relative ordering of common elements
        def rank(order, common):
            filtered = [fm for fm in order if fm in common]
            return {fm: i for i, fm in enumerate(filtered)}
        
        rank1 = rank(order1, common)
        rank2 = rank(order2, common)
        
        # Count concordant and discordant pairs
        concordant = 0
        discordant = 0
        common_list = list(common)
        
        for i in range(len(common_list)):
            for j in range(i + 1, len(common_list)):
                a, b = common_list[i], common_list[j]
                diff1 = rank1[a] - rank1[b]
                diff2 = rank2[a] - rank2[b]
                
                if diff1 * diff2 > 0:
                    concordant += 1
                elif diff1 * diff2 < 0:
                    discordant += 1
        
        n_pairs = len(common_list) * (len(common_list) - 1) / 2
        if n_pairs == 0:
            return 0.5
        
        tau = (concordant - discordant) / n_pairs
        # Normalize from [-1, 1] to [0, 1]
        return (tau + 1) / 2
    
    def _compute_context_similarity(
        self, fp1: InstitutionalFingerprint, fp2: InstitutionalFingerprint
    ) -> float:
        """Compute context features similarity."""
        if not fp1.context_features or not fp2.context_features:
            return 0.5  # No context = neutral
        
        v1 = fp1.context_features.to_vector()
        v2 = fp2.context_features.to_vector()
        
        # Cosine similarity
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 < 1e-10 or norm2 < 1e-10:
            return 0.5
        
        cosine = np.dot(v1, v2) / (norm1 * norm2)
        # Normalize from [-1, 1] to [0, 1]
        return (cosine + 1) / 2


# =============================================================================
# TESTS
# =============================================================================

def test_fingerprint_creation():
    """Test fingerprint creation."""
    fp = InstitutionalFingerprint.create(
        fm_set=["FM4_STRUCTURAL_BREAK", "FM1_VARIANCE_REGIME"],
        root_fm="FM4_STRUCTURAL_BREAK",
        fm_roles=[
            ("FM4_STRUCTURAL_BREAK", FailureModeRole.ROOT),
            ("FM1_VARIANCE_REGIME", FailureModeRole.SYMPTOM),
        ],
        onset_order=["FM4_STRUCTURAL_BREAK", "FM1_VARIANCE_REGIME"],
        duration_hours=120,
        severity_shape_class=ShapeClass.SHOCK_PERSIST,
        irreversibility_flag=True,
    )
    
    print(f"Fingerprint ID: {fp.fingerprint_id}")
    print(f"FM set: {fp.fm_signature.fm_set}")
    print(f"Root: {fp.fm_signature.root_fm}")
    print(f"Onset order: {fp.fm_signature.onset_order}")
    print(f"Duration bucket: {fp.duration_bucket.value}")
    
    assert fp.fm_signature.root_fm == "FM4_STRUCTURAL_BREAK"
    assert fp.duration_bucket == DurationBucket.SHORT
    
    print("✓ test_fingerprint_creation passed")


def test_fingerprint_determinism():
    """Test fingerprint ID is deterministic."""
    kwargs = {
        "fm_set": ["FM4", "FM1"],
        "root_fm": "FM4",
        "fm_roles": [("FM4", FailureModeRole.ROOT), ("FM1", FailureModeRole.SYMPTOM)],
        "onset_order": ["FM4", "FM1"],
        "duration_hours": 100,
        "severity_shape_class": ShapeClass.SHOCK_REVERT,
    }
    
    fp1 = InstitutionalFingerprint.create(**kwargs)
    fp2 = InstitutionalFingerprint.create(**kwargs)
    
    assert fp1.fingerprint_id == fp2.fingerprint_id
    print(f"Deterministic ID: {fp1.fingerprint_id}")
    print("✓ test_fingerprint_determinism passed")


def test_match_with_breakdown():
    """Test matching with full breakdown."""
    # Current fingerprint
    current = InstitutionalFingerprint.create(
        fm_set=["FM4", "FM1", "FM2"],
        root_fm="FM4",
        fm_roles=[
            ("FM4", FailureModeRole.ROOT),
            ("FM1", FailureModeRole.SYMPTOM),
            ("FM2", FailureModeRole.SYMPTOM),
        ],
        onset_order=["FM4", "FM1", "FM2"],
        duration_hours=72,
        severity_shape_class=ShapeClass.SHOCK_PERSIST,
        irreversibility_flag=True,
        context_features=ContextFeatures(
            volatility_percentile=0.85,
            drawdown_percentile=0.72,
        ),
    )
    
    # Similar candidate
    similar = InstitutionalFingerprint.create(
        fm_set=["FM4", "FM1"],
        root_fm="FM4",
        fm_roles=[
            ("FM4", FailureModeRole.ROOT),
            ("FM1", FailureModeRole.SYMPTOM),
        ],
        onset_order=["FM4", "FM1"],
        duration_hours=96,
        severity_shape_class=ShapeClass.SHOCK_PERSIST,
        irreversibility_flag=True,
        recovery_type=RecoveryType.PARTIAL,
        episode_id="ep_similar",
        context_features=ContextFeatures(
            volatility_percentile=0.80,
            drawdown_percentile=0.65,
        ),
    )
    
    # Different candidate
    different = InstitutionalFingerprint.create(
        fm_set=["FM2", "FM6"],
        root_fm="FM6",
        fm_roles=[
            ("FM6", FailureModeRole.ROOT),
            ("FM2", FailureModeRole.SYMPTOM),
        ],
        onset_order=["FM6", "FM2"],
        duration_hours=480,
        severity_shape_class=ShapeClass.GRIND_UP,
        irreversibility_flag=False,
        recovery_type=RecoveryType.FULL,
        episode_id="ep_different",
    )
    
    matcher = InstitutionalMatcher()
    
    # Match against similar
    match_similar = matcher.match(current, similar, "run_001")
    print(f"\nSimilar match: {match_similar.similarity_score:.3f}")
    print("Breakdown:")
    for dim in match_similar.breakdown.dimensions:
        print(f"  {dim.name}: {dim.similarity:.2f} × {dim.weight:.2f} = {dim.weighted_contribution:.3f}")
    
    # Match against different
    match_different = matcher.match(current, different, "run_001")
    print(f"\nDifferent match: {match_different.similarity_score:.3f}")
    print("Breakdown:")
    for dim in match_different.breakdown.dimensions:
        print(f"  {dim.name}: {dim.similarity:.2f} × {dim.weight:.2f} = {dim.weighted_contribution:.3f}")
    
    # Verify
    assert match_similar.similarity_score > match_different.similarity_score
    assert match_similar.similarity_score > 0.7
    assert match_different.similarity_score < 0.5
    
    # Check top contributors
    top = match_similar.breakdown.get_top_contributors(3)
    print(f"\nTop contributors: {[d.name for d in top]}")
    
    print("\n✓ test_match_with_breakdown passed")


def test_onset_order_similarity():
    """Test onset order similarity computation."""
    matcher = InstitutionalMatcher()
    
    # Same order
    fp1 = InstitutionalFingerprint.create(
        fm_set=["A", "B", "C"],
        root_fm="A",
        fm_roles=[("A", FailureModeRole.ROOT), ("B", FailureModeRole.SYMPTOM), ("C", FailureModeRole.SYMPTOM)],
        onset_order=["A", "B", "C"],
        duration_hours=50,
    )
    
    fp2 = InstitutionalFingerprint.create(
        fm_set=["A", "B", "C"],
        root_fm="A",
        fm_roles=[("A", FailureModeRole.ROOT), ("B", FailureModeRole.SYMPTOM), ("C", FailureModeRole.SYMPTOM)],
        onset_order=["A", "B", "C"],
        duration_hours=50,
    )
    
    sim_same = matcher._compute_onset_order_similarity(fp1, fp2)
    print(f"Same order similarity: {sim_same:.2f}")
    assert sim_same == 1.0
    
    # Reversed order
    fp3 = InstitutionalFingerprint.create(
        fm_set=["A", "B", "C"],
        root_fm="C",
        fm_roles=[("C", FailureModeRole.ROOT), ("B", FailureModeRole.SYMPTOM), ("A", FailureModeRole.SYMPTOM)],
        onset_order=["C", "B", "A"],
        duration_hours=50,
    )
    
    sim_reversed = matcher._compute_onset_order_similarity(fp1, fp3)
    print(f"Reversed order similarity: {sim_reversed:.2f}")
    assert sim_reversed == 0.0
    
    print("✓ test_onset_order_similarity passed")


def test_match_result_immutability():
    """Test that match results are immutable."""
    fp1 = InstitutionalFingerprint.create(
        fm_set=["FM4"], root_fm="FM4",
        fm_roles=[("FM4", FailureModeRole.ROOT)],
        onset_order=["FM4"], duration_hours=50,
    )
    
    fp2 = InstitutionalFingerprint.create(
        fm_set=["FM4"], root_fm="FM4",
        fm_roles=[("FM4", FailureModeRole.ROOT)],
        onset_order=["FM4"], duration_hours=60,
        episode_id="ep_001",
    )
    
    matcher = InstitutionalMatcher()
    result = matcher.match(fp1, fp2, "run_001")
    
    # Try to modify (should fail)
    try:
        result.similarity_score = 0.5
        assert False, "Should have raised error"
    except AttributeError:
        pass
    
    print(f"Match result ID: {result.match_result_id}")
    print(f"Similarity: {result.similarity_score:.3f}")
    print("✓ test_match_result_immutability passed")


def run_all_fingerprint_tests():
    print("\n" + "=" * 60)
    print("INSTITUTIONAL FINGERPRINT & MATCH TESTS")
    print("=" * 60 + "\n")
    
    test_fingerprint_creation()
    print()
    test_fingerprint_determinism()
    print()
    test_match_with_breakdown()
    print()
    test_onset_order_similarity()
    print()
    test_match_result_immutability()
    
    print("\n" + "=" * 60)
    print("ALL FINGERPRINT & MATCH TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    run_all_fingerprint_tests()
