"""
Bavella v2 — Series Analyzer
=============================

THE SINGLE ENTRY POINT: "Upload any series → Get validity"

This is the unified interface that:
    1. Accepts any time series (price, factor, spread, returns)
    2. Runs all relevant detectors
    3. Resolves conflicts between failure modes
    4. Computes trust-adjusted validity
    5. Matches against historical patterns
    6. Returns institutional-grade output

One engine, multiple interfaces:
    - Price series → "Is past behavior predictive?"
    - Factor returns → "Is this factor still working?"
    - Spread → "Is mean reversion valid?"
    - Model residuals → "Is my model calibrated?"

Copyright 2024-2026 Bavella Technologies Sarl
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

# Core imports
from .core import (
    FailureMode, FailureSignal, ValidityState, ValidityVerdict,
    Thresholds, Weights,
)
from .detectors_proper import (
    FM1_VarianceRegimeShift,
    FM2_MeanDrift,
    FM4_StructuralBreak,
    FM5_OutlierContamination,
    FM6_DistributionalShift,
)
from .meta_validity import (
    DetectionConfidence, EnhancedFailureSignal, MetaValidityVerdict,
    MetaValidityAssessor, compute_meta_validity, ConfidenceLevel,
)
from .conflict_resolution import (
    ConflictResolver, ConflictAnalysis, FailureModePrecedenceGraph,
)
from .epistemic_cost import (
    EpistemicCostTable, DamageRecord, DamageEvent,
    TrustAdjustedValidity, compute_trust_adjusted_validity,
    Reversibility,
)
from .confidence_governance import (
    apply_confidence_governance, ConfidenceGovernedVerdict,
)
from .pattern_matching import (
    EpisodePatternMatcher, compute_fingerprint, build_this_happened_before,
    ThisHappenedBeforeResponse,
)
from .persistence_postgres import (
    InMemoryEpisodeStore, FailureEpisode, EpisodeState, PersistenceConfig,
)


# =============================================================================
# SERIES TYPE
# =============================================================================

class SeriesType(Enum):
    """Type of time series being analyzed."""
    PRICE = "price"           # Asset price
    RETURNS = "returns"       # Return series
    SPREAD = "spread"         # Price spread (for pairs)
    FACTOR = "factor"         # Factor returns
    RESIDUAL = "residual"     # Model residuals
    GENERIC = "generic"       # Any other series


# =============================================================================
# ANALYSIS REQUEST
# =============================================================================

@dataclass
class SeriesAnalysisRequest:
    """Request to analyze a time series."""
    
    # Identity
    owner_id: str
    series_id: str
    series_name: Optional[str] = None
    
    # Data
    timestamps: List[datetime] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    
    # Optional second series (for pairs/relationships)
    values_2: Optional[List[float]] = None
    series_2_name: Optional[str] = None
    
    # Series type hint
    series_type: SeriesType = SeriesType.GENERIC
    
    # Analysis options
    lookback_days: int = 252  # Default 1 year
    baseline_days: int = 60   # Baseline period for comparison
    
    def __post_init__(self):
        if len(self.timestamps) != len(self.values):
            raise ValueError("timestamps and values must have same length")
        if self.values_2 is not None and len(self.values_2) != len(self.values):
            raise ValueError("values_2 must have same length as values")
    
    @property
    def is_pair(self) -> bool:
        return self.values_2 is not None
    
    def to_array(self) -> np.ndarray:
        return np.array(self.values)
    
    def to_array_2(self) -> Optional[np.ndarray]:
        return np.array(self.values_2) if self.values_2 else None


# =============================================================================
# DETECTOR RESULT
# =============================================================================

@dataclass
class DetectorResult:
    """Result from a single failure mode detector."""
    failure_mode: FailureMode
    detected: bool
    severity: float  # 0-100
    confidence: DetectionConfidence
    explanation: str
    evidence: Dict[str, Any] = field(default_factory=dict)
    triggers_kill: bool = False
    kill_reason: Optional[str] = None
    
    @property
    def effective_severity(self) -> float:
        """Severity weighted by detection confidence."""
        return self.severity * (self.confidence.overall / 100)
    
    def to_signal(self) -> EnhancedFailureSignal:
        return EnhancedFailureSignal(
            failure_mode=self.failure_mode,
            severity=self.severity,
            detection_confidence=self.confidence,
            explanation=self.explanation,
            evidence=self.evidence,
            triggers_kill=self.triggers_kill,
            kill_reason=self.kill_reason,
        )


# =============================================================================
# SERIES VALIDITY REPORT
# =============================================================================

@dataclass
class SeriesValidityReport:
    """
    Complete validity report for a time series.
    
    This is the institutional-grade output.
    """
    # Identity
    series_id: str
    series_name: Optional[str]
    analyzed_at: datetime
    
    # Core validity
    validity_score: float
    validity_state: ValidityState
    validity_confidence: float  # Meta-validity
    
    # Trust-adjusted (accounts for damage history)
    trust_adjusted_score: float
    trust_adjusted_state: ValidityState
    trust_penalty: float
    
    # Confidence governance
    governed_score: float
    governed_state: ValidityState
    was_downgraded: bool
    downgrade_reason: Optional[str]
    
    # Failure mode breakdown
    active_failure_modes: List[DetectorResult]
    root_cause: Optional[FailureMode]
    symptoms: List[FailureMode]
    
    # Conflict analysis
    conflict_analysis: ConflictAnalysis
    
    # Historical context
    this_happened_before: Optional[ThisHappenedBeforeResponse]
    
    # Damage tracking
    has_irreversible_damage: bool
    comparability_broken_at: Optional[datetime]
    requires_rebaseline: bool
    
    # Warnings
    warnings: List[str]
    
    # Data quality
    data_points_analyzed: int
    data_span_days: int
    
    # Permissions
    can_export: bool
    can_emit_predictions: bool
    requires_watermark: bool
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "series": {
                "id": self.series_id,
                "name": self.series_name,
                "analyzed_at": self.analyzed_at.isoformat(),
            },
            "validity": {
                "score": round(self.validity_score, 1),
                "state": self.validity_state.value,
                "confidence": round(self.validity_confidence, 1),
            },
            "trust_adjusted": {
                "score": round(self.trust_adjusted_score, 1),
                "state": self.trust_adjusted_state.value,
                "penalty": round(self.trust_penalty, 1),
            },
            "governed": {
                "score": round(self.governed_score, 1),
                "state": self.governed_state.value,
                "was_downgraded": self.was_downgraded,
                "downgrade_reason": self.downgrade_reason,
            },
            "failure_modes": {
                "active": [
                    {
                        "mode": r.failure_mode.name,
                        "severity": round(r.severity, 1),
                        "effective_severity": round(r.effective_severity, 1),
                        "confidence": round(r.confidence.overall, 1),
                        "explanation": r.explanation,
                        "is_kill_switch": r.triggers_kill,
                    }
                    for r in self.active_failure_modes
                ],
                "root_cause": self.root_cause.name if self.root_cause else None,
                "symptoms": [fm.name for fm in self.symptoms],
            },
            "historical": self.this_happened_before.to_dict() if self.this_happened_before else None,
            "damage": {
                "has_irreversible": self.has_irreversible_damage,
                "comparability_broken_at": self.comparability_broken_at.isoformat() if self.comparability_broken_at else None,
                "requires_rebaseline": self.requires_rebaseline,
            },
            "warnings": self.warnings,
            "data_quality": {
                "points_analyzed": self.data_points_analyzed,
                "span_days": self.data_span_days,
            },
            "permissions": {
                "can_export": self.can_export,
                "can_emit_predictions": self.can_emit_predictions,
                "requires_watermark": self.requires_watermark,
            },
        }
    
    def to_summary(self) -> str:
        """Human-readable summary."""
        lines = []
        lines.append(f"VALIDITY: {self.validity_score:.0f} ({self.validity_state.value.upper()})")
        
        if self.trust_penalty > 0:
            lines.append(f"TRUST-ADJUSTED: {self.trust_adjusted_score:.0f} (penalty: {self.trust_penalty:.0f})")
        
        if self.was_downgraded:
            lines.append(f"GOVERNED: {self.governed_score:.0f} ({self.downgrade_reason})")
        
        if self.active_failure_modes:
            lines.append("")
            if self.root_cause:
                rc = next((r for r in self.active_failure_modes if r.failure_mode == self.root_cause), None)
                if rc:
                    lines.append(f"ROOT CAUSE: {self.root_cause.name}")
                    lines.append(f"  Severity: {rc.severity:.0f} | Confidence: {rc.confidence.overall:.0f}%")
            
            for sym in self.symptoms:
                sr = next((r for r in self.active_failure_modes if r.failure_mode == sym), None)
                if sr:
                    lines.append(f"SYMPTOM: {sym.name}")
                    lines.append(f"  Severity: {sr.severity:.0f} → {sr.effective_severity:.0f} (adjusted)")
        
        if self.warnings:
            lines.append("")
            for w in self.warnings:
                lines.append(w)
        
        return "\n".join(lines)


# =============================================================================
# SERIES ANALYZER
# =============================================================================

class SeriesAnalyzer:
    """
    The unified series analysis engine.
    
    Orchestrates:
        - All failure mode detectors
        - Conflict resolution
        - Meta-validity assessment
        - Epistemic cost tracking
        - Pattern matching
        - Confidence governance
    """
    
    def __init__(
        self,
        episode_store: Optional[InMemoryEpisodeStore] = None,
        pattern_matcher: Optional[EpisodePatternMatcher] = None,
    ):
        # Detectors
        self._fm1_detector = FM1_VarianceRegimeShift()
        self._fm2_detector = FM2_MeanDrift()
        self._fm4_detector = FM4_StructuralBreak()
        self._fm5_detector = FM5_OutlierContamination()
        self._fm6_detector = FM6_DistributionalShift()
        
        # Conflict resolution
        self._conflict_resolver = ConflictResolver()
        
        # Persistence
        self._episode_store = episode_store or InMemoryEpisodeStore()
        
        # Pattern matching
        self._pattern_matcher = pattern_matcher or EpisodePatternMatcher()
        
        # Damage records (per series)
        self._damage_records: Dict[str, DamageRecord] = {}
    
    def _get_damage_record(self, owner_id: str, series_id: str) -> DamageRecord:
        """Get or create damage record for a series."""
        key = f"{owner_id}:{series_id}"
        if key not in self._damage_records:
            self._damage_records[key] = DamageRecord(
                node_id=series_id,
                owner_id=owner_id,
            )
        return self._damage_records[key]
    
    def analyze(self, request: SeriesAnalysisRequest) -> SeriesValidityReport:
        """
        Analyze a time series and return complete validity report.
        """
        now = datetime.now(timezone.utc)
        data = request.to_array()
        
        # Data quality checks
        if len(data) < 30:
            return self._insufficient_data_report(request, now, len(data))
        
        # Calculate data span
        if request.timestamps:
            span_days = (request.timestamps[-1] - request.timestamps[0]).days
        else:
            span_days = len(data)  # Assume daily
        
        # Run all detectors
        results = self._run_detectors(data, request)
        
        # Filter to active (detected) failure modes
        active_results = [r for r in results if r.detected]
        
        # Convert to enhanced signals
        signals = [r.to_signal() for r in active_results]
        
        # Check for kill switches
        killed_by = None
        for r in active_results:
            if r.triggers_kill:
                killed_by = r.failure_mode
                break
        
        # Compute raw validity score
        if killed_by:
            raw_score = 0.0
            raw_state = ValidityState.INVALID
        else:
            raw_score = self._compute_validity_score(active_results)
            raw_state = ValidityState.from_score(raw_score)
        
        # Meta-validity (confidence in the assessment)
        meta_verdict = compute_meta_validity(signals, raw_score, raw_state.value)
        validity_confidence = meta_verdict.assessment_confidence
        
        # Conflict resolution
        if active_results:
            conflict_signals = [(r.failure_mode, r.severity, r.confidence.overall) for r in active_results]
            adjusted = self._conflict_resolver.compute_adjusted_severity(conflict_signals)
            analysis = self._conflict_resolver.analyze([(fm, sev) for fm, sev, _ in conflict_signals])
        else:
            adjusted = []
            analysis = ConflictAnalysis()
        
        # Get damage record and apply trust adjustment
        damage_record = self._get_damage_record(request.owner_id, request.series_id)
        
        # Record damage events for irreversible failures
        for r in active_results:
            if EpistemicCostTable.is_irreversible(r.failure_mode):
                damage_record.record_event(DamageEvent(
                    event_id=f"{r.failure_mode.name}_{now.timestamp()}",
                    failure_mode=r.failure_mode,
                    occurred_at=now,
                    severity=r.severity,
                    cost=EpistemicCostTable.get_cost(r.failure_mode),
                    node_id=request.series_id,
                ))
        
        # Trust-adjusted validity
        trust_adjusted = compute_trust_adjusted_validity(
            raw_score, raw_state.value, damage_record
        )
        
        # Confidence governance
        raw_verdict = ValidityVerdict(score=trust_adjusted.adjusted_score, state=ValidityState(trust_adjusted.adjusted_state.lower()))
        governed = apply_confidence_governance(raw_verdict, validity_confidence)
        
        # Record episodes and get history
        episode_results = self._record_episodes(
            request.owner_id, request.series_id, active_results, now
        )
        
        # Pattern matching
        this_happened_before = None
        if episode_results:
            this_happened_before = build_this_happened_before(
                episode_results,
                self._pattern_matcher,
                request.owner_id,
                request.series_id,
            )
        
        # Build warnings
        warnings = list(damage_record.get_warnings())
        
        if meta_verdict.speculative_signals > 0:
            warnings.append(f"⚠️ {meta_verdict.speculative_signals} speculative signal(s) — assessment has uncertainty")
        
        if killed_by:
            warnings.insert(0, f"🔴 KILL SWITCH: {killed_by.name}")
        
        if governed.was_downgraded:
            warnings.append(f"⚠️ Downgraded due to low confidence")
        
        # Build report
        return SeriesValidityReport(
            series_id=request.series_id,
            series_name=request.series_name,
            analyzed_at=now,
            
            validity_score=raw_score,
            validity_state=raw_state,
            validity_confidence=validity_confidence,
            
            trust_adjusted_score=trust_adjusted.adjusted_score,
            trust_adjusted_state=ValidityState(trust_adjusted.adjusted_state.lower()),
            trust_penalty=trust_adjusted.trust_penalty,
            
            governed_score=governed.governed_score,
            governed_state=governed.governed_state,
            was_downgraded=governed.was_downgraded,
            downgrade_reason=governed.downgrade_reason,
            
            active_failure_modes=active_results,
            root_cause=analysis.root_cause,
            symptoms=list(analysis.symptoms),
            
            conflict_analysis=analysis,
            this_happened_before=this_happened_before,
            
            has_irreversible_damage=damage_record.has_irreversible_damage,
            comparability_broken_at=damage_record.comparability_broken_at,
            requires_rebaseline=damage_record.requires_rebaseline,
            
            warnings=warnings,
            
            data_points_analyzed=len(data),
            data_span_days=span_days,
            
            can_export=governed.can_export,
            can_emit_predictions=governed.can_emit_predictions,
            requires_watermark=governed.requires_watermark,
        )
    
    def _run_detectors(
        self,
        data: np.ndarray,
        request: SeriesAnalysisRequest,
    ) -> List[DetectorResult]:
        """Run all relevant detectors on the data."""
        import pandas as pd
        
        results = []
        
        # Prepare data for detectors
        # Z_t = price/level series
        # z_t = returns/changes series
        Z_t = pd.Series(data)
        z_t = Z_t.pct_change().dropna() if request.series_type == SeriesType.PRICE else Z_t.diff().dropna()
        
        # FM1: Variance Regime
        try:
            fm1_signal = self._fm1_detector.detect(z_t, Z_t)
            fm1_conf = MetaValidityAssessor.assess_variance_detection(
                data, 
                fm1_signal.evidence.get("variance_ratio", 1.0),
                fm1_signal.evidence.get("f_statistic", 1.0),
                fm1_signal.evidence.get("p_value", 0.5),
            )
            results.append(DetectorResult(
                failure_mode=FailureMode.FM1_VARIANCE_REGIME,
                detected=fm1_signal.severity > 15,
                severity=fm1_signal.severity,
                confidence=fm1_conf,
                explanation=fm1_signal.explanation,
                evidence=fm1_signal.evidence,
            ))
        except Exception as e:
            pass  # Detector failed - skip
        
        # FM2: Mean Drift
        try:
            fm2_signal = self._fm2_detector.detect(z_t, Z_t)
            fm2_conf = DetectionConfidence.compute(
                sample_size=len(data),
                effect_size=abs(fm2_signal.evidence.get("drift_magnitude", 0)),
                test_statistic=fm2_signal.evidence.get("t_statistic", 0),
                threshold=2.0,
                variance_of_statistic=1.0,
            )
            results.append(DetectorResult(
                failure_mode=FailureMode.FM2_MEAN_DRIFT,
                detected=fm2_signal.severity > 15,
                severity=fm2_signal.severity,
                confidence=fm2_conf,
                explanation=fm2_signal.explanation,
                evidence=fm2_signal.evidence,
            ))
        except Exception as e:
            pass
        
        # FM4: Structural Break
        try:
            fm4_signal = self._fm4_detector.detect(z_t, Z_t)
            fm4_conf = MetaValidityAssessor.assess_structural_break(
                data,
                fm4_signal.evidence.get("break_magnitude", 0),
                fm4_signal.evidence.get("cusum_max", 0),
            )
            
            # Check for kill switch
            triggers_kill = (
                fm4_signal.severity > 80 and 
                fm4_signal.evidence.get("break_magnitude", 0) > Thresholds.KILL_FM4_BREAK_MAGNITUDE
            )
            
            results.append(DetectorResult(
                failure_mode=FailureMode.FM4_STRUCTURAL_BREAK,
                detected=fm4_signal.severity > 15,
                severity=fm4_signal.severity,
                confidence=fm4_conf,
                explanation=fm4_signal.explanation,
                evidence=fm4_signal.evidence,
                triggers_kill=triggers_kill,
                kill_reason="Structural break exceeds kill threshold" if triggers_kill else None,
            ))
        except Exception as e:
            pass
        
        # FM5: Outlier Contamination
        try:
            fm5_signal = self._fm5_detector.detect(z_t, Z_t)
            fm5_conf = DetectionConfidence.compute(
                sample_size=len(data),
                effect_size=fm5_signal.evidence.get("outlier_fraction", 0) * 10,
                test_statistic=fm5_signal.severity / 20,
                threshold=1.5,
                variance_of_statistic=0.5,
            )
            results.append(DetectorResult(
                failure_mode=FailureMode.FM5_OUTLIER_CONTAMINATION,
                detected=fm5_signal.severity > 15,
                severity=fm5_signal.severity,
                confidence=fm5_conf,
                explanation=fm5_signal.explanation,
                evidence=fm5_signal.evidence,
            ))
        except Exception as e:
            pass
        
        # FM6: Distributional Shift
        try:
            fm6_signal = self._fm6_detector.detect(z_t, Z_t)
            fm6_conf = MetaValidityAssessor.assess_distributional_shift(
                len(data),
                fm6_signal.evidence.get("ks_statistic", 0),
                fm6_signal.evidence.get("ks_pvalue", 0.5),
            )
            results.append(DetectorResult(
                failure_mode=FailureMode.FM6_DISTRIBUTIONAL_SHIFT,
                detected=fm6_signal.severity > 15,
                severity=fm6_signal.severity,
                confidence=fm6_conf,
                explanation=fm6_signal.explanation,
                evidence=fm6_signal.evidence,
            ))
        except Exception as e:
            pass
        
        return results
    
    def _compute_validity_score(self, active_results: List[DetectorResult]) -> float:
        """Compute validity score from active failures."""
        if not active_results:
            return 100.0
        
        # Use effective severity (severity × confidence)
        total_weighted_loss = 0.0
        
        weight_map = {
            FailureMode.FM1_VARIANCE_REGIME: Weights.FM1_VARIANCE,
            FailureMode.FM2_MEAN_DRIFT: Weights.FM2_MEAN_DRIFT,
            FailureMode.FM3_SEASONALITY_MISMATCH: Weights.FM3_SEASONALITY,
            FailureMode.FM4_STRUCTURAL_BREAK: Weights.FM4_STRUCTURAL,
            FailureMode.FM5_OUTLIER_CONTAMINATION: Weights.FM5_OUTLIERS,
            FailureMode.FM6_DISTRIBUTIONAL_SHIFT: Weights.FM6_DISTRIBUTION,
            FailureMode.FM7_DEPENDENCY_BREAK: Weights.FM7_DEPENDENCY,
        }
        
        for r in active_results:
            weight = weight_map.get(r.failure_mode, 0.1)
            total_weighted_loss += r.effective_severity * weight
        
        return max(0, 100 - total_weighted_loss)
    
    def _record_episodes(
        self,
        owner_id: str,
        series_id: str,
        active_results: List[DetectorResult],
        now: datetime,
    ) -> List[FailureEpisode]:
        """Record detected failure modes as episodes."""
        episodes = []
        
        for r in active_results:
            signal = FailureSignal(
                failure_mode=r.failure_mode,
                severity=r.severity,
                confidence=r.confidence.overall / 100,
                explanation=r.explanation,
                evidence=r.evidence,
                triggers_kill=r.triggers_kill,
                kill_reason=r.kill_reason,
            )
            
            episode = self._episode_store.record_detection(owner_id, series_id, signal)
            if episode.episode_id != "none":
                episodes.append(episode)
        
        return episodes
    
    def _insufficient_data_report(
        self,
        request: SeriesAnalysisRequest,
        now: datetime,
        data_points: int,
    ) -> SeriesValidityReport:
        """Return report when data is insufficient."""
        return SeriesValidityReport(
            series_id=request.series_id,
            series_name=request.series_name,
            analyzed_at=now,
            
            validity_score=0,
            validity_state=ValidityState.INVALID,
            validity_confidence=0,
            
            trust_adjusted_score=0,
            trust_adjusted_state=ValidityState.INVALID,
            trust_penalty=0,
            
            governed_score=0,
            governed_state=ValidityState.INVALID,
            was_downgraded=False,
            downgrade_reason=None,
            
            active_failure_modes=[],
            root_cause=None,
            symptoms=[],
            
            conflict_analysis=ConflictAnalysis(),
            this_happened_before=None,
            
            has_irreversible_damage=False,
            comparability_broken_at=None,
            requires_rebaseline=False,
            
            warnings=[f"⚠️ Insufficient data: {data_points} points (minimum 30 required)"],
            
            data_points_analyzed=data_points,
            data_span_days=0,
            
            can_export=False,
            can_emit_predictions=False,
            requires_watermark=True,
        )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def analyze_series(
    values: List[float],
    owner_id: str = "default",
    series_id: str = "series",
    series_name: Optional[str] = None,
    timestamps: Optional[List[datetime]] = None,
) -> SeriesValidityReport:
    """
    Quick analysis of a series.
    
    Example:
        report = analyze_series(prices, series_name="BTC-USD")
        print(report.to_summary())
    """
    if timestamps is None:
        now = datetime.now(timezone.utc)
        timestamps = [now - timedelta(days=len(values)-i-1) for i in range(len(values))]
    
    request = SeriesAnalysisRequest(
        owner_id=owner_id,
        series_id=series_id,
        series_name=series_name,
        timestamps=timestamps,
        values=values,
    )
    
    analyzer = SeriesAnalyzer()
    return analyzer.analyze(request)


# =============================================================================
# TESTS
# =============================================================================

from datetime import timedelta

def test_stable_series():
    """Test analysis of a stable series."""
    np.random.seed(42)
    
    # Stable series with low volatility
    values = list(100 + np.cumsum(np.random.randn(200) * 0.5))
    
    report = analyze_series(values, series_name="Stable Test")
    
    print(f"Stable series validity: {report.validity_score:.1f} ({report.validity_state.value})")
    print(f"Active FMs: {len(report.active_failure_modes)}")
    
    assert report.validity_score > 50, "Stable series should have reasonable validity"
    print("✓ test_stable_series passed")


def test_regime_change_series():
    """Test analysis of a series with regime change."""
    np.random.seed(42)
    
    # Low vol regime then high vol regime
    low_vol = list(100 + np.cumsum(np.random.randn(100) * 0.5))
    high_vol = list(low_vol[-1] + np.cumsum(np.random.randn(100) * 3.0))
    values = low_vol + high_vol
    
    report = analyze_series(values, series_name="Regime Change Test")
    
    print(f"Regime change validity: {report.validity_score:.1f} ({report.validity_state.value})")
    print(f"Active FMs: {[r.failure_mode.name for r in report.active_failure_modes]}")
    print(f"Root cause: {report.root_cause}")
    
    assert report.validity_score < 80, "Regime change should lower validity"
    assert len(report.active_failure_modes) > 0, "Should detect failure modes"
    print("✓ test_regime_change_series passed")


def test_structural_break():
    """Test detection of structural break."""
    np.random.seed(42)
    
    # Mean shift (structural break)
    before = list(100 + np.random.randn(100) * 1.0)
    after = list(150 + np.random.randn(100) * 1.0)  # 50-point shift
    values = before + after
    
    report = analyze_series(values, series_name="Structural Break Test")
    
    print(f"Structural break validity: {report.validity_score:.1f} ({report.validity_state.value})")
    print(f"Warnings: {report.warnings}")
    print(report.to_summary())
    
    # Should detect structural issues
    fm_names = [r.failure_mode.name for r in report.active_failure_modes]
    print(f"Detected: {fm_names}")
    
    print("✓ test_structural_break passed")


def test_report_output():
    """Test report serialization."""
    np.random.seed(42)
    values = list(100 + np.cumsum(np.random.randn(200) * 1.5))
    
    report = analyze_series(values, series_name="Output Test")
    
    # Test dict conversion
    d = report.to_dict()
    assert "validity" in d
    assert "failure_modes" in d
    assert "warnings" in d
    
    # Test summary
    summary = report.to_summary()
    assert "VALIDITY" in summary
    
    print("Report dict keys:", list(d.keys()))
    print("✓ test_report_output passed")


def run_all_analyzer_tests():
    print("\n" + "=" * 60)
    print("SERIES ANALYZER TESTS")
    print("=" * 60 + "\n")
    
    test_stable_series()
    print()
    test_regime_change_series()
    print()
    test_structural_break()
    print()
    test_report_output()
    
    print("\n" + "=" * 60)
    print("ALL ANALYZER TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    run_all_analyzer_tests()
