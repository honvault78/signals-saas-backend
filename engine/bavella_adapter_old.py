"""
Bavella Adapter — Bridge Bavella v2.2 to SaaS Frontend
======================================================

This adapter:
1. Takes inputs from existing /analyze endpoint
2. Runs Bavella v2.2 SeriesAnalyzer
3. Returns the FROZEN JSON contract for the frontend

The frozen contract is small and stable. The engine is big and can evolve.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional
from datetime import datetime

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# FROZEN OUTPUT CONTRACT (matches frontend expectations)
# =============================================================================

@dataclass
class RootCause:
    """Primary failure mode causing the validity issue."""
    code: str           # e.g., "FM4"
    label: str          # e.g., "Structural break"
    severity: int       # 0-100
    is_root: bool       # True if this is root cause, not symptom


@dataclass
class HistoricalAnalogue:
    """A past episode that matches current conditions."""
    date: str           # e.g., "Nov 2021"
    match_percent: int  # 0-100
    duration_days: int
    outcome: str        # "Full recovery", "Partial recovery", "Failed recovery"


@dataclass
class SecondaryFailure:
    """A failure mode that is a symptom, not root cause."""
    code: str
    label: str
    severity: int


@dataclass
class CompetingCause:
    """A potential explanation for validity degradation."""
    code: str           # FM1, FM2, etc.
    label: str          # Human-readable
    score: float        # 0.0-1.0 normalized contribution
    evidence: str       # One-line evidence summary


@dataclass
class CounterfactualCheck:
    """A robustness check on the diagnosis."""
    test: str           # What we tested
    result: str         # What we found
    changes_conclusion: bool  # Does this affect the verdict?


@dataclass
class ValidityOutput:
    """
    The FROZEN JSON contract for validity.
    
    This is what the frontend expects. Do not change without
    coordinating with frontend.
    """
    # Level 1: The Verdict (always visible)
    state: str              # "VALID" | "DEGRADED" | "BROKEN"
    score: int              # 0-100
    summary: str            # One-line explanation
    
    # Root cause (if degraded/broken)
    root_cause: Optional[RootCause]
    
    # Confidence in the diagnosis
    confidence: float       # 0.0-1.0
    
    # Historical context
    similar_episodes: int
    median_recovery_days: Optional[int]
    recovery_dispersion_days: Optional[int]
    
    # For "Why is this broken?" panel
    secondary_failures: List[SecondaryFailure]
    assessment_note: str    # e.g., "This is a root cause, not a symptom"
    
    # NEW: Competing explanations (makes multi-detector visible)
    competing_causes: List[CompetingCause] = field(default_factory=list)
    
    # NEW: Counterfactual checks (robustness)
    counterfactuals: List[CounterfactualCheck] = field(default_factory=list)
    
    # For "View history" panel
    analogues: List[HistoricalAnalogue] = field(default_factory=list)
    failure_probability: Optional[float] = None  # 0.0-1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "validity": {
                "state": self.state,
                "score": self.score,
                "summary": self.summary,
                "root_cause": asdict(self.root_cause) if self.root_cause else None,
                "confidence": round(self.confidence, 2),
            },
            "history": {
                "similar_episodes": self.similar_episodes,
                "median_recovery_days": self.median_recovery_days,
                "recovery_dispersion_days": self.recovery_dispersion_days,
            },
            "details": {
                "secondary_failures": [asdict(f) for f in self.secondary_failures],
                "assessment_note": self.assessment_note,
                "analogues": [asdict(a) for a in self.analogues],
                "failure_probability": round(self.failure_probability, 2) if self.failure_probability else None,
            },
            # NEW: Multi-detector attribution
            "attribution": {
                "competing_causes": [asdict(c) for c in self.competing_causes],
                "counterfactuals": [asdict(c) for c in self.counterfactuals],
            }
        }


# =============================================================================
# FAILURE MODE DEFINITIONS
# =============================================================================

# CONSISTENT TAXONOMY:
# FM1 = Variance/volatility regime change
# FM2 = Mean drift / parameter instability
# FM3 = Seasonality mismatch (rarely used)
# FM4 = Structural break (discrete changepoint)
# FM5 = Outlier contamination
# FM6 = Extreme positioning (z-score far from mean)
# FM7 = Dependency break (correlation/beta instability)

FM_LABELS = {
    "FM1": "Volatility regime shift",
    "FM2": "Parameter drift",
    "FM3": "Seasonality mismatch",
    "FM4": "Structural break",
    "FM5": "Outlier contamination",
    "FM6": "Extreme positioning",
    "FM7": "Dependency break",
}

FM_SUMMARIES = {
    "FM1": "Volatility regime has changed — recent vol differs significantly from historical",
    "FM2": "Parameter drift detected — mean or hedge ratio has shifted",
    "FM3": "Seasonal pattern mismatch — timing or amplitude changed",
    "FM4": "Structural break detected — discrete changepoint identified",
    "FM5": "Outlier contamination — extreme values affecting estimates",
    "FM6": "Extreme positioning — spread far from historical mean",
    "FM7": "Dependency break — correlation or beta instability detected",
}


# =============================================================================
# BAVELLA ANALYZER (Simplified integration)
# =============================================================================

class BavellaAnalyzer:
    """
    Runs Bavella v2.2 analysis and produces frozen output contract.
    
    This is the bridge between the 18k line engine and the tiny API surface.
    """
    
    def __init__(self):
        """Initialize analyzer with Bavella v2.2 components."""
        self._initialized = False
        self._series_analyzer = None
        self._pattern_matcher = None
        
    def _ensure_initialized(self):
        """Lazy initialization of Bavella components."""
        if self._initialized:
            return
            
        try:
            # Import Bavella v2.2 components from engine.bavella
            from engine.bavella.series_analyzer import SeriesAnalyzer
            from engine.bavella.pattern_matching import EpisodePatternMatcher
            from engine.bavella.detectors_proper import DetectorSuite
            
            self._series_analyzer = SeriesAnalyzer()
            self._pattern_matcher = EpisodePatternMatcher()
            self._initialized = True
            logger.info("Bavella v2.2 initialized successfully")
            
        except ImportError as e:
            logger.warning(f"Bavella v2.2 not available, using fallback: {e}")
            self._initialized = True  # Mark as initialized to avoid repeated attempts
    
    def analyze(
        self,
        returns: pd.Series,
        series_name: str = "series",
        owner_id: str = "default",
        adf_pvalue: Optional[float] = None,
        halflife: Optional[float] = None,
        z_score: Optional[float] = None,
        regime: str = "unknown",
    ) -> ValidityOutput:
        """
        Run validity analysis and return frozen contract.
        
        Args:
            returns: Daily returns series
            series_name: Identifier for the series
            owner_id: User/tenant ID
            adf_pvalue: ADF p-value from regime detection
            halflife: Mean reversion half-life
            z_score: Current Z-score
            regime: Current regime string
            
        Returns:
            ValidityOutput matching the frozen contract
        """
        self._ensure_initialized()
        
        # Use fallback if Bavella not available
        if self._series_analyzer is None:
            return self._analyze_fallback(
                returns=returns,
                adf_pvalue=adf_pvalue,
                halflife=halflife,
                z_score=z_score,
                regime=regime,
            )
        
        try:
            return self._analyze_with_bavella(
                returns=returns,
                series_name=series_name,
                owner_id=owner_id,
                adf_pvalue=adf_pvalue,
                halflife=halflife,
                z_score=z_score,
                regime=regime,
            )
        except Exception as e:
            logger.error(f"Bavella analysis failed, using fallback: {e}")
            return self._analyze_fallback(
                returns=returns,
                adf_pvalue=adf_pvalue,
                halflife=halflife,
                z_score=z_score,
                regime=regime,
            )
    
    def _analyze_with_bavella(
        self,
        returns: pd.Series,
        series_name: str,
        owner_id: str,
        adf_pvalue: Optional[float],
        halflife: Optional[float],
        z_score: Optional[float],
        regime: str,
    ) -> ValidityOutput:
        """Run full Bavella v2.2 analysis."""
        from datetime import datetime, timedelta
        from engine.bavella.series_analyzer import SeriesAnalysisRequest, SeriesType
        
        # Convert pandas Series to lists for the request
        if hasattr(returns.index, 'tolist'):
            timestamps = returns.index.tolist()
        else:
            # Create dummy timestamps if index isn't datetime
            base = datetime.now()
            timestamps = [base - timedelta(days=len(returns)-i) for i in range(len(returns))]
        
        # Convert timestamps to datetime if needed
        if timestamps and not isinstance(timestamps[0], datetime):
            timestamps = [datetime.fromisoformat(str(t)) if hasattr(t, '__str__') else datetime.now() for t in timestamps]
        
        values = returns.values.tolist()
        
        # Build request
        request = SeriesAnalysisRequest(
            owner_id=owner_id,
            series_id=series_name,
            series_name=series_name,
            timestamps=timestamps,
            values=values,
            series_type=SeriesType.RETURNS,
        )
        
        # Run analysis
        report = self._series_analyzer.analyze(request)
        
        # =====================================================================
        # COMPUTE CONFIDENCE FIRST (Bug A fix: must be defined before use)
        # =====================================================================
        raw_confidence = getattr(report, 'validity_confidence', 0.8)
        if raw_confidence > 1:
            confidence = raw_confidence / 100.0  # Convert from percentage
        else:
            confidence = raw_confidence
        confidence = max(0.0, min(1.0, confidence))  # Clamp to 0-1
        
        # Extract failure modes - active_failure_modes is List[DetectorResult]
        active_fms = report.active_failure_modes
        root_cause_fm = report.root_cause  # This is Optional[FailureMode] enum
        
        # Filter secondary failures (those that aren't the root cause)
        secondary_fms = []
        if root_cause_fm:
            for fm in active_fms:
                if fm.failure_mode != root_cause_fm:
                    secondary_fms.append(fm)
        
        # Determine state and score - use governed score if available
        validity_score = getattr(report, 'governed_score', report.validity_score)
        if validity_score >= 70:
            state = "VALID"
        elif validity_score >= 30:
            state = "DEGRADED"
        else:
            state = "BROKEN"
        
        # Map FailureMode enum to code/label
        FM_CODE_MAP = {
            "FM1_VARIANCE_REGIME": "FM1",
            "FM2_MEAN_DRIFT": "FM2",
            "FM3_SEASONALITY_MISMATCH": "FM3",
            "FM4_STRUCTURAL_BREAK": "FM4",
            "FM5_OUTLIER_CONTAMINATION": "FM5",
            "FM6_DISTRIBUTIONAL_SHIFT": "FM6",
            "FM7_DEPENDENCY_BREAK": "FM7",
        }
        
        # Build summary
        if root_cause_fm:
            fm_code = FM_CODE_MAP.get(root_cause_fm.name, root_cause_fm.name[:3])
            summary = FM_SUMMARIES.get(fm_code, f"{root_cause_fm.name.replace('_', ' ').title()} detected")
        else:
            summary = "Statistical structure is stable"
        
        # Pattern matching for historical analogues
        analogues = []
        similar_count = 0
        median_recovery = None
        recovery_dispersion = None
        failure_prob = None
        
        if hasattr(report, 'this_happened_before') and report.this_happened_before:
            thb = report.this_happened_before
            similar_count = thb.n_matches if hasattr(thb, 'n_matches') else 0
            
            if hasattr(thb, 'matches') and thb.matches:
                matches = thb.matches
                durations = [m.duration_days for m in matches if hasattr(m, 'duration_days')]
                if durations:
                    median_recovery = int(np.median(durations))
                    recovery_dispersion = int(np.percentile(durations, 75) - np.percentile(durations, 25)) if len(durations) > 1 else 12
                    
                    failed = sum(1 for m in matches if hasattr(m, 'outcome') and 'fail' in str(m.outcome).lower())
                    failure_prob = failed / len(matches) if matches else None
                    
                    for match in matches[:5]:
                        analogues.append(HistoricalAnalogue(
                            date=match.date.strftime("%b %Y") if hasattr(match, 'date') else "Unknown",
                            match_percent=int(match.similarity * 100) if hasattr(match, 'similarity') else 70,
                            duration_days=match.duration_days if hasattr(match, 'duration_days') else 30,
                            outcome=str(match.outcome).replace("_", " ").title() if hasattr(match, 'outcome') else "Unknown",
                        ))
        
        # Build root cause
        root_cause = None
        if root_cause_fm:
            fm_code = FM_CODE_MAP.get(root_cause_fm.name, root_cause_fm.name[:3])
            
            # =========================================================
            # BUG B FIX: Properly capture root FM's severity and confidence
            # =========================================================
            root_raw_severity = 50
            root_fm_confidence = 0.7  # Default
            
            for fm_obj in active_fms:
                if fm_obj.failure_mode == root_cause_fm:
                    root_raw_severity = int(fm_obj.severity)
                    root_fm_confidence = getattr(fm_obj, 'confidence', 0.7)
                    break
            
            # SEVERITY-CONFIDENCE COUPLING (belt-and-suspenders)
            if root_fm_confidence < 0.5:
                root_severity = min(root_raw_severity, 40)
            elif root_fm_confidence < 0.7:
                root_severity = min(root_raw_severity, 60)
            else:
                root_severity = root_raw_severity
            
            root_cause = RootCause(
                code=fm_code,
                label=FM_LABELS.get(fm_code, root_cause_fm.name.replace("_", " ").title()),
                severity=root_severity,
                is_root=True,
            )
        
        # Build secondary failures
        secondary_failures = []
        for fm in secondary_fms:
            fm_code = FM_CODE_MAP.get(fm.failure_mode.name, fm.failure_mode.name[:3])
            secondary_failures.append(SecondaryFailure(
                code=fm_code,
                label=FM_LABELS.get(fm_code, fm.failure_mode.name.replace("_", " ").title()),
                severity=int(fm.severity),
            ))
        
        # =====================================================================
        # BUILD COMPETING EXPLANATIONS (shows multi-detector attribution)
        # =====================================================================
        competing_causes = []
        for fm_obj in active_fms:
            fm_code = FM_CODE_MAP.get(fm_obj.failure_mode.name, fm_obj.failure_mode.name[:3])
            
            # Get raw severity and confidence
            raw_sev = fm_obj.severity
            fm_conf = getattr(fm_obj, 'confidence', 0.7)
            
            # Apply severity-confidence coupling
            if fm_conf < 0.5:
                capped_severity = min(raw_sev, 40)
            elif fm_conf < 0.7:
                capped_severity = min(raw_sev, 60)
            else:
                capped_severity = raw_sev
            
            # =========================================================
            # BUG D FIX: Score is attribution weight, not just severity
            # score = severity * confidence * persistence_factor
            # =========================================================
            persistence = getattr(fm_obj, 'persistence', 1.0)
            if persistence is None:
                persistence = 1.0
            
            # Attribution score: combines severity, confidence, and persistence
            # This makes the ordering feel "diagnostic" not "severity scoreboard"
            attribution_score = (capped_severity / 100.0) * fm_conf * min(1.5, max(0.5, persistence))
            attribution_score = min(1.0, attribution_score)  # Cap at 1.0
            
            # Get explanation from the signal
            evidence = getattr(fm_obj, 'explanation', '') or f"Severity: {int(capped_severity)}"
            if fm_conf < 0.5:
                evidence = f"[low conf] {evidence}"
            
            competing_causes.append(CompetingCause(
                code=fm_code,
                label=FM_LABELS.get(fm_code, fm_obj.failure_mode.name.replace("_", " ").title()),
                score=round(attribution_score, 2),
                evidence=evidence[:100],  # Truncate long explanations
            ))
        
        # Sort by score descending
        competing_causes.sort(key=lambda x: x.score, reverse=True)
        
        # If no active FMs, show all as zero
        if not competing_causes:
            for fm_code in ["FM1", "FM2", "FM4", "FM6", "FM7"]:
                competing_causes.append(CompetingCause(
                    code=fm_code,
                    label=FM_LABELS.get(fm_code, fm_code),
                    score=0.0,
                    evidence="No issues detected",
                ))
        
        # =====================================================================
        # COUNTERFACTUAL CHECKS (robustness indicators)
        # =====================================================================
        counterfactuals = []
        
        # Check if root cause confidence is high
        if root_cause and confidence > 0.7:
            counterfactuals.append(CounterfactualCheck(
                test="Root cause confidence",
                result=f"High confidence ({confidence:.0%}) in {root_cause.code} as primary explanation",
                changes_conclusion=False,
            ))
        elif root_cause and confidence < 0.5:
            counterfactuals.append(CounterfactualCheck(
                test="Root cause confidence",
                result=f"Low confidence ({confidence:.0%}) — multiple explanations viable",
                changes_conclusion=True,
            ))
        
        # Check if there are close competing explanations
        if len(competing_causes) >= 2:
            top_two = competing_causes[:2]
            if top_two[0].score > 0 and top_two[1].score > 0:
                gap = top_two[0].score - top_two[1].score
                if gap < 0.15:
                    counterfactuals.append(CounterfactualCheck(
                        test="Attribution certainty",
                        result=f"{top_two[0].code} ({top_two[0].score:.2f}) vs {top_two[1].code} ({top_two[1].score:.2f}) — close competition",
                        changes_conclusion=True,
                    ))
        
        # Assessment note
        if root_cause and secondary_failures:
            assessment_note = "This is a root cause, not a symptom. Secondary failures are downstream effects."
        elif root_cause:
            assessment_note = "This is the primary cause of validity degradation."
        else:
            assessment_note = "No structural issues detected."
        
        # Note: confidence was already computed at the top of this method
        
        return ValidityOutput(
            state=state,
            score=int(validity_score),
            summary=summary,
            root_cause=root_cause,
            confidence=confidence,
            similar_episodes=similar_count,
            median_recovery_days=median_recovery,
            recovery_dispersion_days=recovery_dispersion,
            secondary_failures=secondary_failures,
            assessment_note=assessment_note,
            competing_causes=competing_causes,
            counterfactuals=counterfactuals,
            analogues=analogues,
            failure_probability=failure_prob,
        )
    
    def _analyze_fallback(
        self,
        returns: pd.Series,
        adf_pvalue: Optional[float],
        halflife: Optional[float],
        z_score: Optional[float],
        regime: str,
    ) -> ValidityOutput:
        """
        Fallback analysis when Bavella v2.2 is not available.
        
        PRINCIPLE: Even in fallback, show the MULTI-DIMENSIONAL nature of validity.
        We compute ALL invariants and show competing explanations, not just the worst.
        """
        # Default values if not provided
        adf = adf_pvalue if adf_pvalue is not None else 0.5
        hl = halflife if halflife is not None else 30
        z = z_score if z_score is not None else 0
        
        # Calculate basic stats from returns
        n = len(returns)
        if n < 20:
            return ValidityOutput(
                state="VALID",
                score=75,
                summary="Insufficient data for full validity assessment",
                root_cause=None,
                confidence=0.4,
                similar_episodes=0,
                median_recovery_days=None,
                recovery_dispersion_days=None,
                secondary_failures=[],
                assessment_note="Fallback mode: limited data available",
                competing_causes=[],
                counterfactuals=[],
                analogues=[],
                failure_probability=None,
            )
        
        # =================================================================
        # COMPUTE ALL INVARIANT SCORES (not just failures)
        # =================================================================
        
        vol = returns.std() * np.sqrt(252)
        recent_vol = returns.tail(20).std() * np.sqrt(252)
        vol_ratio = recent_vol / vol if vol > 1e-10 else 1.0
        
        # Mean drift
        if n >= 60:
            past_mean = returns.iloc[:int(n*0.7)].mean()
            recent_mean = returns.iloc[int(n*0.7):].mean()
            overall_std = returns.std()
            mean_drift_sigma = abs(recent_mean - past_mean) / overall_std if overall_std > 1e-10 else 0
        else:
            mean_drift_sigma = 0
        
        # Stationarity strength (inverted ADF - lower p = stronger stationarity)
        stationarity_strength = max(0, 1 - adf)  # 0 = non-stationary, 1 = strongly stationary
        
        # Mean reversion practicality (based on half-life)
        if hl < 10:
            mr_practicality = 1.0  # Fast reversion
        elif hl < 30:
            mr_practicality = 0.7
        elif hl < 60:
            mr_practicality = 0.4
        elif hl < 90:
            mr_practicality = 0.2
        else:
            mr_practicality = 0.0  # Too slow to be practical
        
        # Position extremity
        position_extremity = min(1.0, abs(z) / 4.0)  # Normalized to 0-1
        
        # =================================================================
        # COMPUTE INVARIANT SCORES (0-1 scale, higher = more concern)
        # =================================================================
        
        # FM1: Variance regime shift
        fm1_score = min(1.0, max(0, (vol_ratio - 1.2) / 1.5)) if vol_ratio > 1.2 else 0
        fm1_evidence = f"Vol ratio: {vol_ratio:.2f}x (recent/historical)"
        
        # FM2: Parameter drift / mean instability
        fm2_score = min(1.0, max(0, (mean_drift_sigma - 1.0) / 2.0)) if mean_drift_sigma > 1.0 else 0
        fm2_evidence = f"Mean drift: {mean_drift_sigma:.2f}σ"
        
        # FM4: Structural break (CANNOT detect without proper tests)
        # We explicitly score this as 0 in fallback
        fm4_score = 0.0
        fm4_evidence = "Changepoint tests not available in fallback mode"
        
        # FM6: Distribution/positioning extremity
        fm6_score = min(1.0, max(0, (abs(z) - 2.5) / 2.0)) if abs(z) > 2.5 else 0
        fm6_evidence = f"Z-score: {z:+.2f}σ from mean"
        
        # FM7: Mean reversion breakdown (slow half-life + weak stationarity)
        fm7_base = max(0, (hl - 30) / 60) if hl > 30 else 0
        fm7_adf_boost = 0.3 if adf > 0.3 else 0
        fm7_score = min(1.0, fm7_base + fm7_adf_boost)
        fm7_evidence = f"Half-life: {hl:.1f}d, ADF p={adf:.3f}"
        
        # =================================================================
        # BUILD COMPETING CAUSES (all invariants with scores)
        # =================================================================
        
        all_invariants = [
            ("FM1", "Volatility regime shift", fm1_score, fm1_evidence),
            ("FM2", "Parameter drift", fm2_score, fm2_evidence),
            ("FM4", "Structural break", fm4_score, fm4_evidence),
            ("FM6", "Distributional stress", fm6_score, fm6_evidence),
            ("FM7", "Mean-reversion breakdown", fm7_score, fm7_evidence),
        ]
        
        # Sort by score descending
        all_invariants.sort(key=lambda x: x[2], reverse=True)
        
        competing_causes = [
            CompetingCause(
                code=inv[0],
                label=inv[1],
                score=round(inv[2], 2),
                evidence=inv[3],
            )
            for inv in all_invariants
        ]
        
        # =================================================================
        # COUNTERFACTUAL CHECKS (robustness)
        # =================================================================
        
        counterfactuals = []
        
        # Check: Does weak stationarity explain everything?
        if adf > 0.2 and fm1_score < 0.3 and fm2_score < 0.3:
            counterfactuals.append(CounterfactualCheck(
                test="Weak stationarity attribution",
                result=f"ADF p={adf:.3f} suggests weak mean-reversion, but no other invariants violated",
                changes_conclusion=False,
            ))
        
        # Check: Is the z-score just noise or meaningful?
        if abs(z) > 2.0:
            if stationarity_strength > 0.7:
                counterfactuals.append(CounterfactualCheck(
                    test="Z-score in stationary context",
                    result=f"Z={z:+.2f}σ with strong stationarity — likely a valid entry signal",
                    changes_conclusion=False,
                ))
            else:
                counterfactuals.append(CounterfactualCheck(
                    test="Z-score in weak stationarity context",
                    result=f"Z={z:+.2f}σ but stationarity is weak — signal less reliable",
                    changes_conclusion=True,
                ))
        
        # Check: Is volatility spike transient?
        if vol_ratio > 1.5:
            # Look at very recent vol vs 20-day vol
            very_recent_vol = returns.tail(5).std() * np.sqrt(252) if len(returns) > 5 else recent_vol
            if very_recent_vol < recent_vol * 0.8:
                counterfactuals.append(CounterfactualCheck(
                    test="Volatility mean-reversion",
                    result=f"Recent vol spike appears to be normalizing (5d vol < 20d vol)",
                    changes_conclusion=True,
                ))
        
        # =================================================================
        # DETERMINE STATE (based on worst non-FM4 invariant)
        # =================================================================
        
        # Get worst score (excluding FM4 which we can't measure)
        measurable_scores = [inv[2] for inv in all_invariants if inv[0] != "FM4"]
        max_score = max(measurable_scores) if measurable_scores else 0
        
        # Convert to validity score (invert: high concern = low validity)
        validity_score = int(100 - max_score * 70)  # Scale to leave room
        
        # Adjust for weak stationarity (not a failure, but a concern)
        weak_stationarity = adf > 0.20
        if weak_stationarity and validity_score > 70:
            validity_score = min(validity_score, 80)
        
        # Determine state and root cause
        if validity_score >= 70:
            state = "VALID"
            if weak_stationarity:
                summary = "Structure intact — weak stationarity noted"
            else:
                summary = "Statistical structure is stable"
            root_cause = None
        elif validity_score >= 30:
            state = "DEGRADED"
            # Root cause is the highest scoring invariant
            primary = all_invariants[0]
            if primary[2] > 0.1:  # Only if actually elevated
                summary = f"{primary[1]} detected — interpret with caution"
                root_cause = RootCause(
                    code=primary[0],
                    label=primary[1],
                    severity=int(primary[2] * 100),
                    is_root=True,
                )
            else:
                summary = "Multiple minor concerns — monitor closely"
                root_cause = None
        else:
            state = "BROKEN"
            primary = all_invariants[0]
            summary = f"{primary[1]} — analysis unreliable"
            root_cause = RootCause(
                code=primary[0],
                label=primary[1],
                severity=int(primary[2] * 100),
                is_root=True,
            )
        
        # =================================================================
        # SECONDARY FAILURES (other elevated invariants)
        # =================================================================
        
        secondary_failures = []
        if root_cause:
            for inv in all_invariants[1:]:  # Skip the root cause
                if inv[2] > 0.2:  # Only if meaningfully elevated
                    secondary_failures.append(SecondaryFailure(
                        code=inv[0],
                        label=inv[1],
                        severity=int(inv[2] * 100),
                    ))
        
        # =================================================================
        # CONFIDENCE (lower for fallback)
        # =================================================================
        
        base_confidence = 0.55
        if weak_stationarity:
            base_confidence -= 0.10
        if n < 60:
            base_confidence -= 0.10
        confidence = max(0.35, min(0.65, base_confidence))
        
        # =================================================================
        # ASSESSMENT NOTE (explain what we found)
        # =================================================================
        
        if state == "VALID":
            if weak_stationarity:
                assessment_note = (
                    f"Stationarity is weak (ADF p={adf:.3f}), meaning mean-reversion signals "
                    f"are less reliable. However, no invariants are violated — the relationship "
                    f"appears intact, just weakly mean-reverting. This is different from a structural break."
                )
            else:
                assessment_note = "All monitored invariants within normal bounds. Statistical assumptions intact."
        else:
            top_causes = [f"{inv[0]}:{inv[2]:.2f}" for inv in all_invariants[:3] if inv[2] > 0.1]
            assessment_note = (
                f"Competing explanations: {', '.join(top_causes) or 'none elevated'}. "
                f"Root cause determined by highest invariant score."
            )
        
        # =================================================================
        # PATTERN MATCHING (historical analogues)
        # =================================================================
        
        analogues = []
        similar_count = 0
        median_recovery = None
        recovery_dispersion = None
        failure_prob = None
        
        if state != "VALID" and len(returns) > 60:
            matches = self._find_similar_periods(returns, adf, vol_ratio)
            similar_count = len(matches)
            
            if matches:
                durations = [m["duration"] for m in matches]
                median_recovery = int(np.median(durations))
                recovery_dispersion = int(np.std(durations)) if len(durations) > 1 else 12
                
                failed = sum(1 for m in matches if m["outcome"] == "failed")
                failure_prob = failed / len(matches)
                
                for m in matches[:3]:
                    analogues.append(HistoricalAnalogue(
                        date=m["date"],
                        match_percent=m["match"],
                        duration_days=m["duration"],
                        outcome=m["outcome"].replace("_", " ").title(),
                    ))
        
        # NOTE: confidence was already calculated above
        
        return ValidityOutput(
            state=state,
            score=int(validity_score),
            summary=summary,
            root_cause=root_cause,
            confidence=confidence,
            similar_episodes=similar_count,
            median_recovery_days=median_recovery,
            recovery_dispersion_days=recovery_dispersion,
            secondary_failures=secondary_failures,
            assessment_note=assessment_note,
            competing_causes=competing_causes,
            counterfactuals=counterfactuals,
            analogues=analogues,
            failure_probability=failure_prob,
        )
    
    def _find_similar_periods(
        self,
        returns: pd.Series,
        current_adf: float,
        current_vol_ratio: float,
    ) -> List[Dict[str, Any]]:
        """Find similar periods in historical data."""
        matches = []
        window_size = 30
        
        if len(returns) < window_size * 3:
            return matches
        
        # Calculate rolling metrics
        rolling_vol = returns.rolling(window_size).std() * np.sqrt(252)
        overall_vol = returns.std() * np.sqrt(252)
        
        # Find periods with similar characteristics
        for i in range(window_size, len(returns) - window_size, window_size // 2):
            window_vol = rolling_vol.iloc[i] if i < len(rolling_vol) else overall_vol
            vol_ratio = window_vol / overall_vol if overall_vol > 0 else 1.0
            
            # Simple similarity score
            vol_sim = 1 - abs(vol_ratio - current_vol_ratio) / max(vol_ratio, current_vol_ratio)
            
            if vol_sim > 0.6:  # Similar enough
                # Check what happened after
                future_returns = returns.iloc[i:i+window_size]
                recovery = future_returns.sum()
                
                outcome = "full_recovery" if recovery > 0.02 else "partial_recovery" if recovery > -0.02 else "failed"
                
                date_idx = returns.index[i] if hasattr(returns.index, '__getitem__') else None
                date_str = date_idx.strftime("%b %Y") if hasattr(date_idx, 'strftime') else f"{i//252}y ago"
                
                matches.append({
                    "date": date_str,
                    "match": int(vol_sim * 100),
                    "duration": window_size + int(abs(recovery) * 100),
                    "outcome": outcome,
                })
        
        # Sort by match quality
        matches.sort(key=lambda x: x["match"], reverse=True)
        return matches[:5]


# =============================================================================
# GLOBAL ANALYZER INSTANCE
# =============================================================================

_analyzer: Optional[BavellaAnalyzer] = None


def get_analyzer() -> BavellaAnalyzer:
    """Get or create the global analyzer instance."""
    global _analyzer
    if _analyzer is None:
        _analyzer = BavellaAnalyzer()
    return _analyzer


# =============================================================================
# INTEGRATION FUNCTION (drop-in replacement)
# =============================================================================

def check_portfolio_validity(
    portfolio_returns: pd.Series,
    regime_summary: Dict[str, Any],
    portfolio_name: str = "Portfolio",
    owner_id: str = "default",
    z_score_override: Optional[float] = None,
) -> ValidityOutput:
    """
    Check portfolio validity — drop-in replacement for existing function.
    
    This function maintains the same signature as the old check_portfolio_validity
    but uses Bavella v2.2 internally and returns the frozen output contract.
    
    Integration in main.py:
    
        from bavella_adapter import check_portfolio_validity
        
        # In /analyze endpoint...
        validity_output = check_portfolio_validity(
            portfolio_returns=portfolio_returns.daily_returns,
            regime_summary=regime_summary.to_dict(),
            portfolio_name=request.portfolio_name,
            z_score_override=z_score,
        )
        
        # Include in response
        return {
            ...
            "validity": validity_output.to_dict(),
        }
    """
    analyzer = get_analyzer()
    
    # Extract values from regime_summary (handle nested structures)
    metrics = regime_summary.get("metrics", regime_summary)
    
    adf_pvalue = (
        regime_summary.get("adf_pvalue") or 
        metrics.get("adf_pvalue") or
        regime_summary.get("adf_p_value") or
        metrics.get("adf_p_value")
    )
    
    halflife = (
        regime_summary.get("halflife") or 
        metrics.get("halflife") or
        regime_summary.get("halflife_periods") or 
        metrics.get("halflife_periods") or
        regime_summary.get("mean_reversion_halflife") or
        metrics.get("mean_reversion_halflife")
    )
    
    z_score = z_score_override
    if z_score is None:
        z_score = (
            regime_summary.get("z_score") or 
            metrics.get("z_score") or
            regime_summary.get("zscore") or
            metrics.get("zscore")
        )
    
    regime = (
        regime_summary.get("current_regime") or
        regime_summary.get("regime") or
        regime_summary.get("primary_regime") or
        "unknown"
    )
    
    logger.info(f"Validity check - ADF: {adf_pvalue}, Halflife: {halflife}, Z-score: {z_score}, Regime: {regime}")
    
    return analyzer.analyze(
        returns=portfolio_returns,
        series_name=portfolio_name,
        owner_id=owner_id,
        adf_pvalue=adf_pvalue,
        halflife=halflife,
        z_score=z_score,
        regime=regime,
    )
