"""
Bavella Validity Engine — Main Analyzer
========================================

The main entry point for validity analysis.

Usage:
    analyzer = ValidityAnalyzer()
    report = analyzer.analyze(
        node_id="portfolio_1",
        returns=daily_returns,
        regime_analysis=regime_analysis,  # From your MarketRegimeDetector
    )
    
    print(report.to_dict())

Provides:
- Complete validity verdict
- Episode tracking
- Historical pattern matching
- Recovery estimation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional
import uuid
import numpy as np
import pandas as pd

from .core import (
    FailureMode,
    ValidityState,
    ValidityVerdict,
    FM_INFO,
)
from .detector import (
    ValidityDetector,
    DetectorResult,
    ActiveFailureMode,
)
from .episodes import (
    Episode,
    EpisodeState,
    EpisodeStore,
    EpisodeSummary,
    get_episode_store,
)


# =============================================================================
# HISTORICAL MATCH
# =============================================================================

@dataclass
class HistoricalMatch:
    """A historical episode that matches the current situation."""
    episode_id: str
    similarity: float  # 0-1
    
    # What happened
    root_cause_fm: str
    duration_days: float
    recovery_type: Optional[str]
    
    # Severity comparison
    initial_validity: float
    min_validity: float
    final_validity: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "similarity": round(self.similarity, 2),
            "root_cause_fm": self.root_cause_fm,
            "duration_days": round(self.duration_days, 1),
            "recovery_type": self.recovery_type,
            "severity": {
                "initial": round(self.initial_validity, 1),
                "min": round(self.min_validity, 1),
                "final": round(self.final_validity, 1),
            },
        }


@dataclass
class RecoveryEstimate:
    """Estimated recovery based on historical patterns."""
    estimated_days: float
    confidence: float  # 0-1
    
    # Distribution
    p25_days: float
    p50_days: float  # Median
    p75_days: float
    
    # Outcomes
    full_recovery_probability: float
    partial_recovery_probability: float
    rebaseline_probability: float
    
    # Basis
    n_precedents: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "estimated_days": round(self.estimated_days, 1),
            "confidence": round(self.confidence, 2),
            "distribution": {
                "p25": round(self.p25_days, 1),
                "p50": round(self.p50_days, 1),
                "p75": round(self.p75_days, 1),
            },
            "outcomes": {
                "full_recovery": round(self.full_recovery_probability, 2),
                "partial_recovery": round(self.partial_recovery_probability, 2),
                "rebaseline": round(self.rebaseline_probability, 2),
            },
            "n_precedents": self.n_precedents,
        }


# =============================================================================
# VALIDITY REPORT
# =============================================================================

@dataclass
class ValidityReport:
    """
    Complete validity analysis report.
    
    The main output of ValidityAnalyzer.analyze()
    """
    # Identity
    report_id: str
    node_id: str
    analyzed_at: datetime
    
    # Core verdict
    verdict: ValidityVerdict
    
    # Detailed detection results
    detection: DetectorResult
    
    # Episode information
    episode: Optional[Episode]
    episode_is_new: bool
    
    # Historical context
    historical_matches: List[HistoricalMatch]
    recovery_estimate: Optional[RecoveryEstimate]
    
    # Summary statistics
    episode_summary: EpisodeSummary
    
    # Actionable insights
    insights: List[str]
    warnings: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "node_id": self.node_id,
            "analyzed_at": self.analyzed_at.isoformat(),
            
            # Core verdict (flat for easy access)
            "validity_score": round(self.verdict.validity_score, 1),
            "validity_state": self.verdict.validity_state.value,
            "is_valid": self.verdict.is_valid,
            
            # Detection details
            "detection": self.detection.to_dict(),
            
            # Episode
            "episode": self.episode.to_dict() if self.episode else None,
            "episode_is_new": self.episode_is_new,
            "has_active_episode": self.episode is not None and self.episode.state == EpisodeState.ACTIVE,
            
            # Historical context
            "historical_matches": [m.to_dict() for m in self.historical_matches],
            "recovery_estimate": self.recovery_estimate.to_dict() if self.recovery_estimate else None,
            
            # Summary
            "episode_summary": self.episode_summary.to_dict(),
            
            # Actionable
            "insights": self.insights,
            "warnings": self.warnings,
        }
    
    def to_summary_dict(self) -> Dict[str, Any]:
        """Compact summary for UI display."""
        return {
            "validity_score": round(self.verdict.validity_score, 1),
            "validity_state": self.verdict.validity_state.value,
            "is_valid": self.verdict.is_valid,
            "fm_count": len(self.detection.active_fms),
            "primary_fm": self.detection.primary_fm.to_dict() if self.detection.primary_fm else None,
            "has_active_episode": self.episode is not None and self.episode.state == EpisodeState.ACTIVE,
            "episode_duration_days": self.episode.duration_days if self.episode else 0,
            "estimated_recovery_days": self.recovery_estimate.estimated_days if self.recovery_estimate else None,
            "regime": self.detection.regime,
            "insights": self.insights[:3],  # Top 3
        }


# =============================================================================
# VALIDITY ANALYZER
# =============================================================================

class ValidityAnalyzer:
    """
    Main validity analysis engine.
    
    Integrates:
    - Failure mode detection
    - Episode tracking
    - Historical pattern matching
    - Recovery estimation
    """
    
    def __init__(
        self,
        episode_store: Optional[EpisodeStore] = None,
    ):
        self.detector = ValidityDetector()
        self.episode_store = episode_store or get_episode_store()
    
    def analyze(
        self,
        node_id: str,
        returns: pd.Series,
        owner_id: str = "default",
        regime_analysis: Optional[Any] = None,
        adf_pvalue: Optional[float] = None,
        halflife: Optional[float] = None,
        z_score: Optional[float] = None,
        regime: str = "unknown",
        reference_returns: Optional[pd.Series] = None,
    ) -> ValidityReport:
        """
        Perform complete validity analysis.
        
        Args:
            node_id: Identifier for the series/relationship
            returns: Daily returns series
            owner_id: User/tenant identifier
            regime_analysis: Output from MarketRegimeDetector (optional)
            adf_pvalue: ADF p-value (if not in regime_analysis)
            halflife: Half-life (if not in regime_analysis)
            z_score: Current Z-score (if not in regime_analysis)
            regime: Current regime string
            reference_returns: Reference series for correlation analysis
        
        Returns:
            ValidityReport with complete analysis
        """
        now = datetime.now(timezone.utc)
        
        # Step 1: Detect failure modes
        detection = self.detector.detect(
            returns=returns,
            regime_analysis=regime_analysis,
            adf_pvalue=adf_pvalue,
            halflife=halflife,
            z_score=z_score,
            regime=regime,
            reference_returns=reference_returns,
        )
        
        # Step 2: Create verdict
        verdict = detection.to_verdict(node_id)
        
        # Step 3: Process episode
        episode = self.episode_store.process_verdict(
            node_id=node_id,
            owner_id=owner_id,
            verdict=verdict,
        )
        
        # Determine if episode is new
        episode_is_new = (
            episode is not None and 
            episode.verdict_count == 1 and
            episode.state == EpisodeState.ACTIVE
        )
        
        # Step 4: Find historical matches
        historical_matches = self._find_matches(verdict, episode)
        
        # Step 5: Estimate recovery
        recovery_estimate = None
        if episode and episode.state == EpisodeState.ACTIVE:
            recovery_estimate = self._estimate_recovery(
                verdict, episode, historical_matches
            )
        
        # Step 6: Get episode summary
        episode_summary = self.episode_store.get_summary(owner_id)
        
        # Step 7: Generate insights
        insights = self._generate_insights(
            detection, episode, historical_matches, recovery_estimate
        )
        
        # Step 8: Generate warnings
        warnings = self._generate_warnings(detection, episode)
        
        return ValidityReport(
            report_id=str(uuid.uuid4()),
            node_id=node_id,
            analyzed_at=now,
            verdict=verdict,
            detection=detection,
            episode=episode,
            episode_is_new=episode_is_new,
            historical_matches=historical_matches,
            recovery_estimate=recovery_estimate,
            episode_summary=episode_summary,
            insights=insights,
            warnings=warnings,
        )
    
    def _find_matches(
        self,
        verdict: ValidityVerdict,
        episode: Optional[Episode],
    ) -> List[HistoricalMatch]:
        """Find historical episodes that match current situation."""
        if not verdict.primary_fm:
            return []
        
        # Get similar episodes
        similar = self.episode_store.get_similar_episodes(
            root_fm=verdict.primary_fm,
            min_severity=100 - verdict.validity_score,
            limit=5,
        )
        
        # Exclude current episode
        if episode:
            similar = [e for e in similar if e.episode_id != episode.episode_id]
        
        # Convert to matches
        matches = []
        for ep in similar:
            similarity = self._calculate_similarity(verdict, ep)
            
            matches.append(HistoricalMatch(
                episode_id=ep.episode_id,
                similarity=similarity,
                root_cause_fm=ep.root_cause_fm.value if ep.root_cause_fm else "unknown",
                duration_days=ep.duration_days,
                recovery_type=ep.recovery_type,
                initial_validity=ep.initial_validity,
                min_validity=ep.min_validity,
                final_validity=ep.current_validity,
            ))
        
        return sorted(matches, key=lambda m: m.similarity, reverse=True)
    
    def _calculate_similarity(
        self,
        verdict: ValidityVerdict,
        episode: Episode,
    ) -> float:
        """Calculate similarity between current situation and historical episode."""
        score = 0.0
        
        # Same root cause FM (40% weight)
        if verdict.primary_fm and episode.root_cause_fm:
            if verdict.primary_fm == episode.root_cause_fm:
                score += 0.4
        
        # Similar initial severity (30% weight)
        severity_diff = abs(verdict.validity_score - episode.initial_validity)
        score += 0.3 * max(0, 1 - severity_diff / 50)
        
        # Similar FM count (15% weight)
        if episode.initial_fms:
            fm_count_diff = abs(len(verdict.active_fms) - len(episode.initial_fms))
            score += 0.15 * max(0, 1 - fm_count_diff / 3)
        
        # FM overlap (15% weight)
        if verdict.active_fms and episode.initial_fms:
            overlap = len(set(verdict.active_fms) & set(episode.initial_fms))
            total = len(set(verdict.active_fms) | set(episode.initial_fms))
            if total > 0:
                score += 0.15 * (overlap / total)
        
        return score
    
    def _estimate_recovery(
        self,
        verdict: ValidityVerdict,
        episode: Episode,
        matches: List[HistoricalMatch],
    ) -> RecoveryEstimate:
        """Estimate recovery time based on historical patterns."""
        # Get resolved matches
        resolved = [m for m in matches if m.recovery_type is not None]
        
        if not resolved:
            # No historical precedent - use FM-based heuristics
            return self._heuristic_recovery_estimate(verdict)
        
        # Calculate statistics from matches
        durations = [m.duration_days for m in resolved]
        
        p25 = np.percentile(durations, 25)
        p50 = np.percentile(durations, 50)
        p75 = np.percentile(durations, 75)
        
        # Weight by similarity
        weights = [m.similarity for m in resolved]
        total_weight = sum(weights)
        
        if total_weight > 0:
            weighted_estimate = sum(
                m.duration_days * m.similarity for m in resolved
            ) / total_weight
        else:
            weighted_estimate = p50
        
        # Outcome probabilities
        full_count = sum(1 for m in resolved if m.recovery_type == "full")
        partial_count = sum(1 for m in resolved if m.recovery_type == "partial")
        rebaseline_count = sum(1 for m in resolved if m.recovery_type == "rebaseline")
        total = len(resolved)
        
        # Confidence based on sample size and similarity
        base_confidence = min(0.9, 0.3 + 0.15 * len(resolved))
        avg_similarity = sum(weights) / len(weights) if weights else 0.5
        confidence = base_confidence * (0.5 + 0.5 * avg_similarity)
        
        return RecoveryEstimate(
            estimated_days=weighted_estimate,
            confidence=confidence,
            p25_days=p25,
            p50_days=p50,
            p75_days=p75,
            full_recovery_probability=full_count / total if total > 0 else 0.5,
            partial_recovery_probability=partial_count / total if total > 0 else 0.3,
            rebaseline_probability=rebaseline_count / total if total > 0 else 0.2,
            n_precedents=len(resolved),
        )
    
    def _heuristic_recovery_estimate(
        self,
        verdict: ValidityVerdict,
    ) -> RecoveryEstimate:
        """Heuristic recovery estimate when no historical precedent."""
        # Base estimate on severity and FM type
        base_days = 14  # Default
        
        if verdict.primary_fm:
            fm_info = FM_INFO[verdict.primary_fm]
            
            # Structural breaks take longer
            if verdict.primary_fm == FailureMode.FM4_STRUCTURAL_BREAK:
                base_days = 45
            elif not fm_info["typically_reversible"]:
                base_days = 30
            elif verdict.primary_fm == FailureMode.FM6_TAIL_EVENT:
                base_days = 5  # Tail events resolve quickly
        
        # Scale by severity
        severity = 100 - verdict.validity_score
        estimated = base_days * (0.5 + severity / 100)
        
        return RecoveryEstimate(
            estimated_days=estimated,
            confidence=0.3,  # Low confidence without precedent
            p25_days=estimated * 0.5,
            p50_days=estimated,
            p75_days=estimated * 2,
            full_recovery_probability=0.6,
            partial_recovery_probability=0.25,
            rebaseline_probability=0.15,
            n_precedents=0,
        )
    
    def _generate_insights(
        self,
        detection: DetectorResult,
        episode: Optional[Episode],
        matches: List[HistoricalMatch],
        recovery: Optional[RecoveryEstimate],
    ) -> List[str]:
        """Generate actionable insights."""
        insights = []
        
        # Validity state insight
        if detection.validity_state == ValidityState.VALID:
            insights.append("✓ Relationship is epistemically valid — safe to trade on")
        elif detection.validity_state == ValidityState.DEGRADED:
            insights.append("⚠️ Validity degraded — use caution with positions")
        else:
            insights.append("🔴 Relationship INVALID — avoid new positions")
        
        # Primary FM insight
        if detection.primary_fm:
            fm = detection.primary_fm.failure_mode
            fm_info = FM_INFO[fm]
            insights.append(
                f"Primary issue: {fm_info['name']} — {fm_info['description']}"
            )
        
        # Episode insight
        if episode and episode.state == EpisodeState.ACTIVE:
            insights.append(
                f"Episode active for {episode.duration_days:.1f} days "
                f"(worst validity: {episode.min_validity:.0f})"
            )
        
        # Recovery insight
        if recovery and recovery.n_precedents > 0:
            insights.append(
                f"Based on {recovery.n_precedents} similar episodes: "
                f"expected recovery in {recovery.p50_days:.0f} days "
                f"(confidence: {recovery.confidence:.0%})"
            )
        
        # Historical match insight
        if matches:
            best_match = matches[0]
            if best_match.similarity > 0.6:
                insights.append(
                    f"Similar to past episode — that one lasted {best_match.duration_days:.0f} days "
                    f"and ended with {best_match.recovery_type or 'unknown'} recovery"
                )
        
        return insights
    
    def _generate_warnings(
        self,
        detection: DetectorResult,
        episode: Optional[Episode],
    ) -> List[str]:
        """Generate warning messages."""
        warnings = []
        
        # FM4 structural break warning
        for fm in detection.active_fms:
            if fm.failure_mode == FailureMode.FM4_STRUCTURAL_BREAK:
                warnings.append(
                    "⚠️ STRUCTURAL BREAK detected — historical relationship may be permanently invalidated"
                )
        
        # Long episode warning
        if episode and episode.duration_days > 30:
            warnings.append(
                f"⚠️ Episode ongoing for {episode.duration_days:.0f} days — consider rebaseline"
            )
        
        # Multiple FM warning
        if len(detection.active_fms) >= 3:
            warnings.append(
                f"⚠️ Multiple failure modes active ({len(detection.active_fms)}) — "
                "compound invalidity risk"
            )
        
        # High ADF p-value warning
        if detection.adf_pvalue > 0.5:
            warnings.append(
                f"⚠️ Non-stationary behavior (ADF p={detection.adf_pvalue:.2f}) — "
                "mean reversion assumptions violated"
            )
        
        return warnings


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def analyze_validity(
    node_id: str,
    returns: pd.Series,
    regime_analysis: Optional[Any] = None,
    **kwargs,
) -> ValidityReport:
    """
    Convenience function for quick validity analysis.
    
    Usage:
        report = analyze_validity("portfolio_1", daily_returns, regime_analysis)
    """
    analyzer = ValidityAnalyzer()
    return analyzer.analyze(
        node_id=node_id,
        returns=returns,
        regime_analysis=regime_analysis,
        **kwargs,
    )
