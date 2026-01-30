"""
Bavella v2 — Core Data Models
==============================

Investment-grade data models implementing the Engineering Addendum v1.2.
All entities are immutable where specified, with full audit trail support.

Key Design Principles:
    - Validity is about INFERENCE, not data
    - Attribution is MANDATORY when score < 100
    - Suppression over fabrication
    - Immutability for audit compliance

Copyright 2024-2026 Bavella Technologies Sarl
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

import numpy as np


# =============================================================================
# ENUMERATIONS (FROZEN PER ONTOLOGY CHARTER)
# =============================================================================

class ValidityState(str, Enum):
    """
    Discretized validity status.
    
    Thresholds (frozen):
        VALID: score >= 70
        DEGRADED: 30 <= score < 70
        INVALID: score < 30
    """
    VALID = "valid"
    DEGRADED = "degraded"
    INVALID = "invalid"
    UNKNOWN = "unknown"
    
    @classmethod
    def from_score(cls, score: float) -> "ValidityState":
        if score is None or np.isnan(score):
            return cls.UNKNOWN
        if score >= 70:
            return cls.VALID
        if score >= 30:
            return cls.DEGRADED
        return cls.INVALID


class ValidityTrend(str, Enum):
    """Direction of validity score movement."""
    STABLE = "stable"
    DEGRADING = "degrading"
    RECOVERING = "recovering"


class SeriesType(str, Enum):
    """Behavioral type of time series - determines transform pipeline."""
    MULTIPLICATIVE_PRICE = "multiplicative_price"
    ADDITIVE_FLOW = "additive_flow"
    CUMULATIVE_BALANCE = "cumulative_balance"
    SEASONAL_BILL = "seasonal_bill"
    IRREGULAR_SIGNAL = "irregular_signal"
    UNKNOWN = "unknown"


class NodeType(str, Enum):
    """Type of analysis node in the epistemic DAG."""
    RAW_SERIES = "raw_series"
    STATISTICAL_TRANSFORM = "statistical_transform"
    DERIVED_SERIES = "derived_series"
    ML_MODEL = "ml_model"
    COMPOSITE_ANALYSIS = "composite_analysis"


class DependencyType(str, Enum):
    """Type of dependency between nodes."""
    INPUT = "input"
    FEATURE = "feature"
    RESIDUAL = "residual"
    REFERENCE = "reference"


class FailureModeID(str, Enum):
    """Enumerated failure modes (FM1-FM7 + reserved FM8)."""
    FM1_VARIANCE_REGIME_SHIFT = "fm1_variance_regime_shift"
    FM2_MEAN_DRIFT = "fm2_mean_drift"
    FM3_SEASONALITY_MISMATCH = "fm3_seasonality_mismatch"
    FM4_STRUCTURAL_BREAK = "fm4_structural_break"
    FM5_OUTLIER_CONTAMINATION = "fm5_outlier_contamination"
    FM6_DISTRIBUTIONAL_SHIFT = "fm6_distributional_shift"
    FM7_DEPENDENCY_BREAK = "fm7_dependency_break"
    FM8_PREDICTION_INSTABILITY = "fm8_prediction_instability"  # Reserved


class ScaleMethod(str, Enum):
    """Method for estimating local variability."""
    EWMA = "ewma"
    ROLLING_STD = "rolling_std"
    MAD = "mad"


class SeasonalityHandling(str, Enum):
    """How to handle seasonality in transforms."""
    NONE = "none"
    DETECT_AND_REMOVE = "detect_and_remove"
    USER_DEFINED = "user_defined"


class ModelFamily(str, Enum):
    """ML model family (metadata only)."""
    TREE = "tree"
    LINEAR = "linear"
    NEURAL = "neural"
    ENSEMBLE = "ensemble"
    UNKNOWN = "unknown"


class PredictionType(str, Enum):
    """Type of ML model output."""
    REGRESSION = "regression"
    CLASSIFICATION = "classification"
    SCORE = "score"
    RANKING = "ranking"


class ContractType(str, Enum):
    """Types of validity contracts users can declare."""
    HALF_LIFE_LT = "half_life_lt"
    HURST_LT = "hurst_lt"
    STATIONARITY_REQUIRED = "stationarity_required"
    CORRELATION_GT = "correlation_gt"
    VOLATILITY_LT = "volatility_lt"
    FEATURE_DISTRIBUTION_STABLE = "feature_distribution_stable"
    OUTPUT_VARIANCE_BOUNDED = "output_variance_bounded"
    RESIDUAL_MEAN_ZERO = "residual_mean_zero"
    PREDICTION_RANK_STABLE = "prediction_rank_stable"
    CUSTOM = "custom"


class ContractStatus(str, Enum):
    """Status of a validity contract."""
    HOLDING = "holding"
    BREACHED = "breached"
    PENDING = "pending"


# =============================================================================
# CORE DATA CLASSES
# =============================================================================

@dataclass
class SeriesMetadata:
    """Immutable metadata for a time series."""
    series_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    owner_id: Optional[str] = None
    display_name: str = "Unnamed Series"
    unit: Optional[str] = None
    timezone: str = "UTC"
    frequency: str = "daily"
    data_source: str = "upload"
    series_type_inferred: SeriesType = SeriesType.UNKNOWN
    series_type_override: Optional[SeriesType] = None
    type_confidence: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def series_type_effective(self) -> SeriesType:
        return self.series_type_override or self.series_type_inferred


@dataclass
class TransformSpec:
    """Immutable specification for the transform pipeline."""
    transform_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    series_id: str = ""
    seasonality_handling: SeasonalityHandling = SeasonalityHandling.DETECT_AND_REMOVE
    differencing_order: int = 1
    scale_method: ScaleMethod = ScaleMethod.EWMA
    scale_lookback: int = 20
    winsorization: str = "none"
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class AnalysisNode:
    """A node in the epistemic DAG."""
    node_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    owner_id: Optional[str] = None
    node_type: NodeType = NodeType.RAW_SERIES
    display_name: str = "Unnamed Node"
    description: str = ""
    series_id: Optional[str] = None
    transform_spec_id: Optional[str] = None
    ml_model_spec_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class AnalysisDependency:
    """Edge in the epistemic DAG."""
    parent_node_id: str
    child_node_id: str
    dependency_type: DependencyType = DependencyType.INPUT
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class MLModelSpec:
    """Metadata for ML models (NO weights/parameters stored)."""
    node_id: str
    model_family: ModelFamily = ModelFamily.UNKNOWN
    prediction_type: PredictionType = PredictionType.REGRESSION
    output_unit: Optional[str] = None
    training_window_desc: Optional[str] = None
    feature_count: int = 0
    has_ground_truth_stream: bool = False
    input_output_lag_days: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)


# =============================================================================
# FAILURE MODE & VALIDITY DATA STRUCTURES
# =============================================================================

@dataclass
class FailureModeSignal:
    """Signal for a specific failure mode detection."""
    signal_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    target_id: str = ""
    target_type: str = "series"
    failure_mode_id: FailureModeID = FailureModeID.FM1_VARIANCE_REGIME_SHIFT
    as_of_ts: datetime = field(default_factory=datetime.utcnow)
    lookback_days: int = 90
    
    # Core metrics
    severity: float = 0.0  # 0-100
    detection_confidence: float = 1.0  # 0-1
    
    # Thresholds
    threshold_degraded: float = 40.0
    threshold_invalid: float = 75.0
    
    # Evidence
    raw_metrics: Dict[str, Any] = field(default_factory=dict)
    evidence_ts: Optional[datetime] = None
    human_summary: str = ""
    
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def contributes_to_degradation(self) -> bool:
        return self.severity >= self.threshold_degraded
    
    @property
    def contributes_to_invalidation(self) -> bool:
        return self.severity >= self.threshold_invalid


@dataclass
class ValidityAttribution:
    """Explains why validity score is < 100."""
    failure_mode_id: FailureModeID
    contribution_pct: float
    severity: float
    key_evidence: Dict[str, Any] = field(default_factory=dict)
    human_summary: str = ""


@dataclass
class ValidityComputation:
    """Complete validity assessment for a target."""
    validity_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    target_id: str = ""
    target_type: str = "series"
    as_of_ts: datetime = field(default_factory=datetime.utcnow)
    lookback_days: int = 90
    
    # Core metrics
    validity_score: float = 100.0
    validity_state: ValidityState = ValidityState.VALID
    validity_trend: ValidityTrend = ValidityTrend.STABLE
    trend_slope: float = 0.0
    
    # History
    last_valid_ts: Optional[datetime] = None
    last_degraded_ts: Optional[datetime] = None
    last_invalid_ts: Optional[datetime] = None
    
    # Attribution (MANDATORY when score < 100)
    attributions: List[ValidityAttribution] = field(default_factory=list)
    
    # Inheritance
    inherited_from: List[Tuple[str, float, ValidityState]] = field(default_factory=list)
    upstream_invalid_count: int = 0
    downstream_impact_count: int = 0
    
    # Metadata
    analysis_kind: str = "statistical"
    suppression_reason: Optional[str] = None
    transform_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def validate_attribution(self) -> bool:
        """Verify attribution sums to 100% of validity loss."""
        if self.validity_score >= 100:
            return True
        if not self.attributions:
            raise ValueError("Attribution required when validity_score < 100")
        total = sum(a.contribution_pct for a in self.attributions)
        if abs(total - 100.0) > 0.5:
            raise ValueError(f"Attribution must sum to 100%, got {total}%")
        return True
    
    @property
    def top_attributions(self) -> List[ValidityAttribution]:
        return sorted(self.attributions, key=lambda a: a.contribution_pct, reverse=True)[:3]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "validity_id": self.validity_id,
            "target_id": self.target_id,
            "as_of_ts": self.as_of_ts.isoformat(),
            "validity_score": round(self.validity_score, 1),
            "validity_state": self.validity_state.value,
            "validity_trend": self.validity_trend.value,
            "trend_slope": round(self.trend_slope, 3),
            "attributions": [
                {
                    "failure_mode": a.failure_mode_id.value,
                    "contribution_pct": round(a.contribution_pct, 1),
                    "severity": round(a.severity, 1),
                    "summary": a.human_summary,
                }
                for a in self.attributions
            ],
            "upstream_invalid_count": self.upstream_invalid_count,
            "suppression_reason": self.suppression_reason,
        }


@dataclass
class ValidityContract:
    """User-declared assumption that Bavella monitors."""
    contract_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    owner_id: str = ""
    target_id: str = ""
    contract_type: ContractType = ContractType.CUSTOM
    parameters: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    status: ContractStatus = ContractStatus.PENDING
    last_evaluated_ts: Optional[datetime] = None
    margin: float = 1.0
    breach_ts: Optional[datetime] = None
    breach_magnitude: Optional[float] = None
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class DecisionStamp:
    """Immutable audit trail entry for user decisions."""
    stamp_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    owner_id: str = ""
    target_id: str = ""
    ts: datetime = field(default_factory=datetime.utcnow)
    decision_type: str = "note"
    note: str = ""
    validity_score_at_ts: float = 100.0
    validity_state_at_ts: ValidityState = ValidityState.VALID
    top_attribution_at_ts: List[Dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)


# =============================================================================
# SERIES PASSPORT
# =============================================================================

@dataclass
class SeriesPassport:
    """Auto-generated diagnostic summary for a series."""
    passport_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    series_id: str = ""
    generated_at: datetime = field(default_factory=datetime.utcnow)
    
    # Structure
    observation_count: int = 0
    date_range_start: Optional[datetime] = None
    date_range_end: Optional[datetime] = None
    frequency_detected: str = "unknown"
    
    # Classification
    series_type: SeriesType = SeriesType.UNKNOWN
    type_confidence: float = 0.0
    
    # Properties
    can_be_negative: bool = False
    crosses_zero: bool = False
    is_level_meaningful: bool = True
    variance_proportional_to_level: bool = False
    has_step_changes: bool = False
    
    # Seasonality
    has_seasonality: bool = False
    dominant_period_days: Optional[int] = None
    seasonality_strength: float = 0.0
    
    # Stationarity
    is_stationary: bool = False
    adf_pvalue: float = 1.0
    hurst_exponent: float = 0.5
    half_life_periods: Optional[float] = None
    
    # Risk shape
    skewness: float = 0.0
    kurtosis: float = 0.0
    tail_heaviness: str = "normal"
    jump_frequency: float = 0.0
    
    # Breaks
    break_count: int = 0
    break_dates: List[datetime] = field(default_factory=list)
    
    # Validity
    current_validity_score: float = 100.0
    current_validity_state: ValidityState = ValidityState.UNKNOWN
    
    # Warnings
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "passport_id": self.passport_id,
            "series_id": self.series_id,
            "generated_at": self.generated_at.isoformat(),
            "observation_count": self.observation_count,
            "date_range": {
                "start": self.date_range_start.isoformat() if self.date_range_start else None,
                "end": self.date_range_end.isoformat() if self.date_range_end else None,
            },
            "classification": {
                "type": self.series_type.value,
                "confidence_pct": round(self.type_confidence * 100, 1),
            },
            "properties": {
                "can_be_negative": self.can_be_negative,
                "crosses_zero": self.crosses_zero,
                "is_multiplicative": self.variance_proportional_to_level,
            },
            "seasonality": {
                "detected": self.has_seasonality,
                "period_days": self.dominant_period_days,
                "strength_pct": round(self.seasonality_strength * 100, 1),
            },
            "stationarity": {
                "is_stationary": self.is_stationary,
                "adf_pvalue": round(self.adf_pvalue, 4),
                "hurst": round(self.hurst_exponent, 3),
                "half_life": round(self.half_life_periods, 1) if self.half_life_periods else None,
            },
            "risk_shape": {
                "skewness": round(self.skewness, 3),
                "kurtosis": round(self.kurtosis, 3),
                "tail_heaviness": self.tail_heaviness,
            },
            "validity": {
                "score": round(self.current_validity_score, 1),
                "state": self.current_validity_state.value,
            },
            "warnings": self.warnings,
        }


# =============================================================================
# SUPPRESSION MATRIX
# =============================================================================

@dataclass
class SuppressionDecision:
    """Determines what to show/hide based on validity state."""
    validity_state: ValidityState
    
    # L0: Always visible
    show_raw_series: bool = True
    show_validity_score: bool = True
    show_attribution: bool = True
    show_break_timestamps: bool = True
    show_dependency_graph: bool = True
    
    # L1: Suppressed when INVALID
    show_statistical_metrics: bool = True
    show_regime_labels: bool = True
    show_relationship_metrics: bool = True
    show_model_predictions: bool = True
    
    # L2: Suppressed when DEGRADED
    show_comparison_rankings: bool = True
    allow_full_export: bool = True
    
    # Watermarks
    watermark_required: bool = False
    watermark_text: str = ""
    
    @classmethod
    def compute(cls, validity_state: ValidityState) -> "SuppressionDecision":
        decision = cls(validity_state=validity_state)
        
        if validity_state == ValidityState.INVALID:
            decision.show_statistical_metrics = False
            decision.show_regime_labels = False
            decision.show_relationship_metrics = False
            decision.show_model_predictions = False
            decision.show_comparison_rankings = False
            decision.allow_full_export = False
            
        elif validity_state == ValidityState.DEGRADED:
            decision.watermark_required = True
            decision.watermark_text = "DEGRADED — Use with caution"
            decision.show_comparison_rankings = False
            
        return decision
    
    def get_suppression_reason(self, field_name: str) -> Optional[str]:
        if self.validity_state == ValidityState.INVALID:
            return "Suppressed: inference assumptions violated"
        elif self.validity_state == ValidityState.DEGRADED:
            return "Caution: inference assumptions weakening"
        return None
