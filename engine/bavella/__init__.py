"""
Bavella v2.2 — Institutional-Grade Validity Infrastructure
==========================================================

"Nobody Can Call It Vibes"

18,000+ lines of production code implementing assumption validity monitoring.

Usage:
    from bavella import analyze_series
    from bavella.series_analyzer import SeriesAnalyzer
    from bavella.query_interface import BavellaQueryRouter

Copyright 2024-2026 Bavella Technologies Sarl
"""

__version__ = "2.2.0"
__author__ = "Bavella Technologies"

# =============================================================================
# CORE TYPES (from core.py)
# =============================================================================

from .core import (
    # Constants
    Thresholds,
    Weights,
    # Enums
    ValidityState,
    FailureMode,
    # Data classes
    FailureSignal,
    ValidityVerdict,
    NodeIdentity,
    GovernedValue,
    # Exceptions
    GovernorRefusal,
    # Base classes
    AnalysisNode,
    ValidityHistory,
    DAGRegistry,
    # Functions
    get_registry,
)

# =============================================================================
# GOVERNOR (from governor.py)
# =============================================================================

from .governor import (
    OutputLevel,
    OutputRequest,
    OutputDecision,
    EmissionRecord,
    Governor,
    GovernedResponseBuilder,
    GovernedExporter,
    get_governor,
    new_session,
)

# =============================================================================
# MODELS (from models.py)
# =============================================================================

from .models import (
    ValidityTrend,
    SeriesType,
    NodeType,
    DependencyType,
    FailureModeID,
    ScaleMethod,
    SeasonalityHandling,
)

# =============================================================================
# DETECTORS (from detectors_proper.py)
# =============================================================================

from .detectors_proper import (
    ConfidencePolicy,
    FMThresholds,
    FailureModeDetector,
    FM1_VarianceRegimeShift,
    FM2_MeanDrift,
    FM3_SeasonalityMismatch,
    FM4_StructuralBreak,
    FM5_OutlierContamination,
    FM6_DistributionalShift,
    FM7_DependencyBreak,
    DetectorSuite,
)

# =============================================================================
# VALIDITY COMPUTATION (from validity_computer.py)
# =============================================================================

from .validity_computer import (
    AggregationMode,
    ValidityComputer,
    ValidityEngine,
)

# =============================================================================
# META-VALIDITY (from meta_validity.py)
# =============================================================================

from .meta_validity import (
    ConfidenceLevel,
    DetectionConfidence,
    EnhancedFailureSignal,
    MetaValidityAssessor,
    MetaValidityVerdict,
    compute_meta_validity,
)

# =============================================================================
# CONFLICT RESOLUTION (from conflict_resolution.py)
# =============================================================================

from .conflict_resolution import (
    RelationType,
    Relationship,
    FailureModePrecedenceGraph,
    ConflictAnalysis,
    ConflictResolver,
)

# =============================================================================
# EPISTEMIC COST (from epistemic_cost.py)
# =============================================================================

from .epistemic_cost import (
    Reversibility,
    HistoricalImpact,
    EpistemicCost,
    EpistemicCostTable,
    DamageEvent,
    DamageRecord,
    TrustAdjustedValidity,
    compute_trust_adjusted_validity,
    ProbationaryStatus,
    RecoveryStatus,
    ProbationaryTracker,
)

# =============================================================================
# PERSISTENCE (from persistence_postgres.py)
# =============================================================================

from .persistence_postgres import (
    PersistenceConfig,
    EpisodeState,
    FailureEpisode,
    PostgresEpisodeStore,
    InMemoryEpisodeStore,
)

# =============================================================================
# CONFIDENCE GOVERNANCE (from confidence_governance.py)
# =============================================================================

from .confidence_governance import (
    ConfidenceGovernanceRules,
    ConfidenceGovernedVerdict,
    apply_confidence_governance,
    ConfidenceAwareGovernor,
)

# =============================================================================
# AUDIT INFRASTRUCTURE (from audit_infrastructure.py)
# =============================================================================

from .audit_infrastructure import (
    BavellaVersions,
    EpisodeStatus,
    RecoveryType,
    FailureModeRole,
    IntegrityLevel,
    EvidenceType,
    TrustEventType,
    DataRef,
    EvidenceArtifact,
    FailureModeEntry,
    CausalOrdering,
    RecoveryGate,
    EpisodeResolution,
    EpisodeRecord,
    TrustLedgerEntry,
    AnalysisRun,
)

# =============================================================================
# SEVERITY CURVES (from severity_curve.py)
# =============================================================================

from .severity_curve import (
    ShapeClass,
    SeverityCurveDescriptor,
    compute_severity_descriptor,
    compute_curve_similarity,
)

# =============================================================================
# RECOVERY GATES (from recovery_gates.py)
# =============================================================================

from .recovery_gates import (
    GateOperator,
    GateMetric,
    RecoveryGateDefinition,
    GateEvaluationResult,
    StandardRecoveryGates,
    RecoveryGateContext,
    RecoveryGateEvaluator,
    RecoveryDistribution,
    EstimateConfidence,
    RecoveryEstimate,
)

# =============================================================================
# INSTITUTIONAL FINGERPRINT (from institutional_fingerprint.py)
# =============================================================================

from .institutional_fingerprint import (
    DurationBucket,
    FMSignature,
    ContextFeatures,
    InstitutionalFingerprint,
    MatchDimension,
    MatchBreakdown,
    MatchResult,
    SimilarityWeights,
    InstitutionalMatcher,
)

# =============================================================================
# PATTERN MATCHING (from pattern_matching.py)
# =============================================================================

from .pattern_matching import (
    EpisodeFingerprint,
    compute_fingerprint,
    compute_episode_similarity,
    EpisodeCluster,
    SimilarEpisodeMatch,
    EpisodePatternMatcher,
    ThisHappenedBeforeResponse,
    build_this_happened_before,
)

# =============================================================================
# SERIES ANALYZER (from series_analyzer.py)
# =============================================================================

from .series_analyzer import (
    SeriesAnalysisRequest,
    DetectorResult,
    SeriesValidityReport,
    SeriesAnalyzer,
    analyze_series,
)

# =============================================================================
# QUERY INTERFACE (from query_interface.py)
# =============================================================================

from .query_interface import (
    QueryType,
    BavellaQuery,
    BavellaResponse,
    BavellaQueryRouter,
)

# =============================================================================
# CONTAGION (from contagion.py)
# =============================================================================

from .contagion import (
    ContagionType,
    PropagationDirection,
    ContagionLink,
    EpisodeContagionInfo,
    NodeRelationship,
    NodeGraph,
    PropagationDetector,
    NodeValidityState,
    PortfolioValidityRollup,
)

# =============================================================================
# REPLAY INFRASTRUCTURE (from replay_infrastructure.py)
# =============================================================================

from .replay_infrastructure import (
    ReplayBundle,
    ReplayStatus,
    ReplayDiff,
    ReplayResult,
    ReplayExecutor,
    ReproducibilityChecker,
    AuditEntry,
    AuditTrail,
)

# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    # Version
    "__version__",
    
    # Main entry points
    "analyze_series",
    "SeriesAnalyzer",
    "BavellaQueryRouter",
    
    # Core
    "ValidityState",
    "FailureMode",
    "FailureSignal",
    "ValidityVerdict",
    "GovernedValue",
    "Thresholds",
    "Weights",
    
    # Governor
    "Governor",
    "OutputLevel",
    "get_governor",
    
    # Models
    "ValidityTrend",
    "SeriesType",
    "FailureModeID",
    
    # Detectors
    "FailureModeDetector",
    "FM1_VarianceRegimeShift",
    "FM2_MeanDrift",
    "FM3_SeasonalityMismatch",
    "FM4_StructuralBreak",
    "FM5_OutlierContamination",
    "FM6_DistributionalShift",
    "FM7_DependencyBreak",
    "DetectorSuite",
    
    # Validity
    "ValidityComputer",
    "ValidityEngine",
    
    # Meta-validity
    "DetectionConfidence",
    "MetaValidityAssessor",
    "MetaValidityVerdict",
    "compute_meta_validity",
    
    # Conflict resolution
    "ConflictResolver",
    "ConflictAnalysis",
    "FailureModePrecedenceGraph",
    
    # Epistemic cost
    "Reversibility",
    "EpistemicCostTable",
    "DamageRecord",
    "TrustAdjustedValidity",
    "compute_trust_adjusted_validity",
    
    # Persistence
    "PostgresEpisodeStore",
    "InMemoryEpisodeStore",
    "FailureEpisode",
    "EpisodeState",
    
    # Audit
    "AnalysisRun",
    "DataRef",
    "EpisodeRecord",
    "EvidenceArtifact",
    "TrustLedgerEntry",
    "BavellaVersions",
    
    # Severity
    "SeverityCurveDescriptor",
    "ShapeClass",
    "compute_curve_similarity",
    
    # Recovery
    "RecoveryGateDefinition",
    "RecoveryDistribution",
    "RecoveryEstimate",
    "RecoveryGateEvaluator",
    "StandardRecoveryGates",
    
    # Fingerprinting
    "InstitutionalFingerprint",
    "FMSignature",
    "MatchResult",
    "InstitutionalMatcher",
    
    # Pattern matching
    "EpisodeFingerprint",
    "EpisodePatternMatcher",
    "build_this_happened_before",
    
    # Series analyzer
    "SeriesAnalysisRequest",
    "SeriesValidityReport",
    
    # Query
    "QueryType",
    "BavellaQuery",
    "BavellaResponse",
    
    # Contagion
    "ContagionLink",
    "NodeGraph",
    "PropagationDetector",
    "PortfolioValidityRollup",
    
    # Replay
    "ReplayBundle",
    "ReplayExecutor",
    "AuditTrail",
    "ReplayResult",
]
