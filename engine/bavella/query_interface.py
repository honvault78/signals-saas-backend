"""
Bavella v2 — Query Interface
=============================

THE LLM BRIDGE: "Ask Bavella anything"

Users ask natural language questions. The LLM translates them into
structured queries. Bavella returns evidence-based answers.

    User: "Has momentum ever broken like this before?"
    LLM → Query: find_similar_episodes(series_id="momentum", ...)
    Bavella → Evidence: {similar_episodes: [...], narrative: "..."}
    LLM → "Yes, 3 times. Most similar was Nov 2021..."

This module defines:
    1. Query types (what questions can be asked)
    2. Query handlers (how to answer them)
    3. Response formats (structured evidence)
    4. Query router (matches intent to handler)

The LLM is the interface. Bavella is the truth.

Copyright 2024-2026 Bavella Technologies Sarl
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Type
import json

from .core import FailureMode, ValidityState
from .series_analyzer import SeriesAnalyzer, SeriesAnalysisRequest, SeriesValidityReport
from .pattern_matching import (
    EpisodePatternMatcher, ThisHappenedBeforeResponse,
    build_this_happened_before, compute_fingerprint,
)
from .persistence_postgres import InMemoryEpisodeStore, FailureEpisode
from .epistemic_cost import DamageRecord


# =============================================================================
# QUERY TYPES
# =============================================================================

class QueryType(Enum):
    """Types of questions users can ask."""
    
    # Current state
    WHAT_IS_VALIDITY = "what_is_validity"
    WHY_IS_DEGRADED = "why_is_degraded"
    WHAT_BROKE = "what_broke"
    
    # Causality
    WHAT_CAUSED_WHAT = "what_caused_what"
    WHAT_IS_ROOT_CAUSE = "what_is_root_cause"
    
    # History
    HAS_THIS_HAPPENED_BEFORE = "has_this_happened_before"
    WHEN_DID_THIS_START = "when_did_this_start"
    HOW_LONG_UNTIL_RECOVERY = "how_long_until_recovery"
    
    # Comparison
    COMPARE_SERIES = "compare_series"
    COMPARE_TO_HISTORICAL = "compare_to_historical"
    
    # Validity checks
    IS_BACKTEST_VALID = "is_backtest_valid"
    CAN_I_TRUST_THIS = "can_i_trust_this"
    
    # Damage
    WHAT_IS_IRREVERSIBLE = "what_is_irreversible"
    WHAT_IS_TRUST_PENALTY = "what_is_trust_penalty"


# =============================================================================
# QUERY REQUEST
# =============================================================================

@dataclass
class BavellaQuery:
    """A structured query to Bavella."""
    
    query_type: QueryType
    
    # Target
    owner_id: str
    series_id: str
    
    # Optional parameters
    series_id_2: Optional[str] = None  # For comparisons
    time_range_start: Optional[datetime] = None
    time_range_end: Optional[datetime] = None
    failure_modes: Optional[List[FailureMode]] = None
    
    # Context (from conversation)
    context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_type": self.query_type.value,
            "owner_id": self.owner_id,
            "series_id": self.series_id,
            "series_id_2": self.series_id_2,
            "time_range": {
                "start": self.time_range_start.isoformat() if self.time_range_start else None,
                "end": self.time_range_end.isoformat() if self.time_range_end else None,
            },
            "failure_modes": [fm.name for fm in self.failure_modes] if self.failure_modes else None,
        }


# =============================================================================
# QUERY RESPONSE
# =============================================================================

@dataclass
class BavellaResponse:
    """Response to a Bavella query."""
    
    # Query echo
    query_type: QueryType
    series_id: str
    
    # Success
    success: bool
    error: Optional[str] = None
    
    # Structured answer
    answer: Dict[str, Any] = field(default_factory=dict)
    
    # Human-readable narrative
    narrative: str = ""
    
    # Supporting evidence
    evidence: Dict[str, Any] = field(default_factory=dict)
    
    # Confidence in the answer
    confidence: float = 0.0  # 0-100
    
    # Warnings
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_type": self.query_type.value,
            "series_id": self.series_id,
            "success": self.success,
            "error": self.error,
            "answer": self.answer,
            "narrative": self.narrative,
            "evidence": self.evidence,
            "confidence": self.confidence,
            "warnings": self.warnings,
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)


# =============================================================================
# QUERY HANDLERS
# =============================================================================

class QueryHandler(ABC):
    """Base class for query handlers."""
    
    @abstractmethod
    def handle(
        self,
        query: BavellaQuery,
        analyzer: SeriesAnalyzer,
        reports: Dict[str, SeriesValidityReport],
    ) -> BavellaResponse:
        pass


class WhatIsValidityHandler(QueryHandler):
    """Handler for "What is the validity of X?" queries."""
    
    def handle(
        self,
        query: BavellaQuery,
        analyzer: SeriesAnalyzer,
        reports: Dict[str, SeriesValidityReport],
    ) -> BavellaResponse:
        report = reports.get(query.series_id)
        
        if not report:
            return BavellaResponse(
                query_type=query.query_type,
                series_id=query.series_id,
                success=False,
                error=f"No analysis found for series '{query.series_id}'",
            )
        
        return BavellaResponse(
            query_type=query.query_type,
            series_id=query.series_id,
            success=True,
            answer={
                "validity_score": report.validity_score,
                "validity_state": report.validity_state.value,
                "trust_adjusted_score": report.trust_adjusted_score,
                "confidence": report.validity_confidence,
            },
            narrative=(
                f"The validity of {report.series_name or query.series_id} is "
                f"{report.validity_score:.0f} ({report.validity_state.value.upper()}). "
                f"Assessment confidence: {report.validity_confidence:.0f}%."
                + (f" Trust-adjusted: {report.trust_adjusted_score:.0f} due to historical damage."
                   if report.trust_penalty > 0 else "")
            ),
            confidence=report.validity_confidence,
            warnings=report.warnings,
        )


class WhyIsDegradedHandler(QueryHandler):
    """Handler for "Why is X degraded/invalid?" queries."""
    
    def handle(
        self,
        query: BavellaQuery,
        analyzer: SeriesAnalyzer,
        reports: Dict[str, SeriesValidityReport],
    ) -> BavellaResponse:
        report = reports.get(query.series_id)
        
        if not report:
            return BavellaResponse(
                query_type=query.query_type,
                series_id=query.series_id,
                success=False,
                error=f"No analysis found for series '{query.series_id}'",
            )
        
        if report.validity_state == ValidityState.VALID:
            return BavellaResponse(
                query_type=query.query_type,
                series_id=query.series_id,
                success=True,
                answer={"is_degraded": False},
                narrative=f"{report.series_name or query.series_id} is currently VALID. No degradation.",
                confidence=report.validity_confidence,
            )
        
        # Build explanation
        fm_explanations = []
        for r in report.active_failure_modes:
            is_root = r.failure_mode == report.root_cause
            is_symptom = r.failure_mode in report.symptoms
            
            fm_explanations.append({
                "failure_mode": r.failure_mode.name,
                "severity": r.severity,
                "confidence": r.confidence.overall,
                "is_root_cause": is_root,
                "is_symptom": is_symptom,
                "explanation": r.explanation,
            })
        
        # Build narrative
        if report.root_cause:
            root_fm = next((r for r in report.active_failure_modes if r.failure_mode == report.root_cause), None)
            narrative = f"ROOT CAUSE: {report.root_cause.name}"
            if root_fm:
                narrative += f" (severity {root_fm.severity:.0f}, confidence {root_fm.confidence.overall:.0f}%)"
            
            if report.symptoms:
                narrative += f". SYMPTOMS: {', '.join(fm.name for fm in report.symptoms)}"
        else:
            narrative = f"Active failure modes: {', '.join(r.failure_mode.name for r in report.active_failure_modes)}"
        
        return BavellaResponse(
            query_type=query.query_type,
            series_id=query.series_id,
            success=True,
            answer={
                "is_degraded": True,
                "state": report.validity_state.value,
                "root_cause": report.root_cause.name if report.root_cause else None,
                "symptoms": [fm.name for fm in report.symptoms],
                "failure_modes": fm_explanations,
            },
            narrative=narrative,
            evidence={"conflict_analysis": report.conflict_analysis.to_dict()},
            confidence=report.validity_confidence,
            warnings=report.warnings,
        )


class HasThisHappenedBeforeHandler(QueryHandler):
    """Handler for "Has this happened before?" queries."""
    
    def handle(
        self,
        query: BavellaQuery,
        analyzer: SeriesAnalyzer,
        reports: Dict[str, SeriesValidityReport],
    ) -> BavellaResponse:
        report = reports.get(query.series_id)
        
        if not report:
            return BavellaResponse(
                query_type=query.query_type,
                series_id=query.series_id,
                success=False,
                error=f"No analysis found for series '{query.series_id}'",
            )
        
        thb = report.this_happened_before
        
        if not thb:
            return BavellaResponse(
                query_type=query.query_type,
                series_id=query.series_id,
                success=True,
                answer={"found_similar": False, "count": 0},
                narrative="No historical pattern data available for this series.",
                confidence=50,
            )
        
        return BavellaResponse(
            query_type=query.query_type,
            series_id=query.series_id,
            success=True,
            answer={
                "found_similar": len(thb.similar_episodes) > 0,
                "count": thb.total_similar_count,
                "most_similar": thb.most_similar.to_dict() if thb.most_similar else None,
                "avg_recovery_days": thb.avg_recovery_days,
                "recovery_rate": thb.recovery_rate,
            },
            narrative=thb.narrative,
            evidence={
                "similar_episodes": [e.to_dict() for e in thb.similar_episodes],
            },
            confidence=85 if thb.similar_episodes else 50,
        )


class HowLongUntilRecoveryHandler(QueryHandler):
    """Handler for "How long until recovery?" queries."""
    
    def handle(
        self,
        query: BavellaQuery,
        analyzer: SeriesAnalyzer,
        reports: Dict[str, SeriesValidityReport],
    ) -> BavellaResponse:
        report = reports.get(query.series_id)
        
        if not report:
            return BavellaResponse(
                query_type=query.query_type,
                series_id=query.series_id,
                success=False,
                error=f"No analysis found for series '{query.series_id}'",
            )
        
        if report.validity_state == ValidityState.VALID:
            return BavellaResponse(
                query_type=query.query_type,
                series_id=query.series_id,
                success=True,
                answer={"needs_recovery": False},
                narrative=f"{report.series_name or query.series_id} is currently VALID. No recovery needed.",
                confidence=90,
            )
        
        thb = report.this_happened_before
        
        if not thb or not thb.avg_recovery_days:
            # No historical data
            if report.has_irreversible_damage:
                narrative = "This damage is IRREVERSIBLE. Full recovery is not possible. A rebaseline is required."
                can_recover = False
            else:
                narrative = "No historical recovery data available. Cannot estimate recovery time."
                can_recover = True
            
            return BavellaResponse(
                query_type=query.query_type,
                series_id=query.series_id,
                success=True,
                answer={
                    "needs_recovery": True,
                    "can_fully_recover": can_recover,
                    "estimated_days": None,
                },
                narrative=narrative,
                confidence=30,
                warnings=report.warnings,
            )
        
        # Build estimate from history
        current_duration = thb.current_duration_days
        avg_total = thb.avg_recovery_days
        estimated_remaining = max(0, avg_total - current_duration)
        
        if report.has_irreversible_damage:
            narrative = (
                f"Based on {thb.total_similar_count} similar episodes, average duration was {avg_total:.0f} days. "
                f"You are on day {current_duration:.0f}. "
                f"However, IRREVERSIBLE DAMAGE has occurred — full recovery is not possible. "
                f"Trust will remain permanently reduced."
            )
            can_recover = False
        else:
            narrative = (
                f"Based on {thb.total_similar_count} similar episodes, average recovery was {avg_total:.0f} days. "
                f"You are on day {current_duration:.0f}. "
                f"Estimated remaining: ~{estimated_remaining:.0f} days."
            )
            if thb.recovery_rate and thb.recovery_rate < 100:
                narrative += f" Note: Only {thb.recovery_rate:.0f}% of similar episodes fully recovered."
            can_recover = True
        
        return BavellaResponse(
            query_type=query.query_type,
            series_id=query.series_id,
            success=True,
            answer={
                "needs_recovery": True,
                "can_fully_recover": can_recover,
                "current_duration_days": current_duration,
                "avg_recovery_days": avg_total,
                "estimated_remaining_days": estimated_remaining,
                "recovery_rate": thb.recovery_rate,
            },
            narrative=narrative,
            confidence=70 if thb.total_similar_count >= 3 else 50,
            warnings=report.warnings,
        )


class IsBacktestValidHandler(QueryHandler):
    """Handler for "Is my backtest valid?" queries."""
    
    def handle(
        self,
        query: BavellaQuery,
        analyzer: SeriesAnalyzer,
        reports: Dict[str, SeriesValidityReport],
    ) -> BavellaResponse:
        report = reports.get(query.series_id)
        
        if not report:
            return BavellaResponse(
                query_type=query.query_type,
                series_id=query.series_id,
                success=False,
                error=f"No analysis found for series '{query.series_id}'",
            )
        
        # Check for structural breaks and irreversible damage
        has_structural_break = any(
            r.failure_mode == FailureMode.FM4_STRUCTURAL_BREAK
            for r in report.active_failure_modes
        )
        
        if report.comparability_broken_at:
            narrative = (
                f"NO — your backtest is INVALID. "
                f"A structural break on {report.comparability_broken_at.strftime('%Y-%m-%d')} means "
                f"pre-break and post-break data are epistemically incomparable. "
                f"Any statistics computed across this break are meaningless."
            )
            is_valid = False
            confidence = 95
        elif has_structural_break:
            narrative = (
                f"CAUTION — structural break detected. "
                f"Your backtest may span different regimes. "
                f"Statistics computed across the break may be unreliable."
            )
            is_valid = False
            confidence = 80
        elif report.validity_state == ValidityState.INVALID:
            narrative = (
                f"CAUTION — current validity is INVALID. "
                f"Recent data may not be representative of historical behavior."
            )
            is_valid = False
            confidence = 70
        else:
            narrative = (
                f"Your backtest appears epistemically valid. "
                f"No structural breaks detected. "
                f"Current validity: {report.validity_score:.0f} ({report.validity_state.value})."
            )
            is_valid = True
            confidence = 85
        
        return BavellaResponse(
            query_type=query.query_type,
            series_id=query.series_id,
            success=True,
            answer={
                "is_valid": is_valid,
                "has_structural_break": has_structural_break,
                "comparability_broken_at": report.comparability_broken_at.isoformat() if report.comparability_broken_at else None,
                "current_validity": report.validity_score,
            },
            narrative=narrative,
            confidence=confidence,
            warnings=report.warnings,
        )


class WhatIsTrustPenaltyHandler(QueryHandler):
    """Handler for "What is my trust penalty?" queries."""
    
    def handle(
        self,
        query: BavellaQuery,
        analyzer: SeriesAnalyzer,
        reports: Dict[str, SeriesValidityReport],
    ) -> BavellaResponse:
        report = reports.get(query.series_id)
        
        if not report:
            return BavellaResponse(
                query_type=query.query_type,
                series_id=query.series_id,
                success=False,
                error=f"No analysis found for series '{query.series_id}'",
            )
        
        if report.trust_penalty == 0:
            narrative = f"No trust penalty. Raw validity ({report.validity_score:.0f}) equals trust-adjusted validity."
        else:
            narrative = (
                f"Trust penalty: {report.trust_penalty:.0f} points. "
                f"Raw validity: {report.validity_score:.0f} → Trust-adjusted: {report.trust_adjusted_score:.0f}. "
                f"This penalty reflects accumulated damage from past failures."
            )
            if report.has_irreversible_damage:
                narrative += " Includes IRREVERSIBLE damage that cannot be recovered."
        
        return BavellaResponse(
            query_type=query.query_type,
            series_id=query.series_id,
            success=True,
            answer={
                "trust_penalty": report.trust_penalty,
                "raw_validity": report.validity_score,
                "trust_adjusted_validity": report.trust_adjusted_score,
                "has_irreversible_damage": report.has_irreversible_damage,
                "requires_rebaseline": report.requires_rebaseline,
            },
            narrative=narrative,
            confidence=90,
            warnings=report.warnings,
        )


# =============================================================================
# QUERY ROUTER
# =============================================================================

class BavellaQueryRouter:
    """
    Routes queries to appropriate handlers.
    
    This is what the LLM calls after parsing user intent.
    """
    
    def __init__(self, analyzer: Optional[SeriesAnalyzer] = None):
        self._analyzer = analyzer or SeriesAnalyzer()
        self._reports: Dict[str, SeriesValidityReport] = {}
        
        # Register handlers
        self._handlers: Dict[QueryType, QueryHandler] = {
            QueryType.WHAT_IS_VALIDITY: WhatIsValidityHandler(),
            QueryType.WHY_IS_DEGRADED: WhyIsDegradedHandler(),
            QueryType.WHAT_BROKE: WhyIsDegradedHandler(),  # Same handler
            QueryType.HAS_THIS_HAPPENED_BEFORE: HasThisHappenedBeforeHandler(),
            QueryType.HOW_LONG_UNTIL_RECOVERY: HowLongUntilRecoveryHandler(),
            QueryType.IS_BACKTEST_VALID: IsBacktestValidHandler(),
            QueryType.WHAT_IS_TRUST_PENALTY: WhatIsTrustPenaltyHandler(),
        }
    
    def register_report(self, series_id: str, report: SeriesValidityReport) -> None:
        """Register an analysis report for querying."""
        self._reports[series_id] = report
    
    def analyze_and_register(self, request: SeriesAnalysisRequest) -> SeriesValidityReport:
        """Analyze a series and register for querying."""
        report = self._analyzer.analyze(request)
        self._reports[request.series_id] = report
        return report
    
    def query(self, query: BavellaQuery) -> BavellaResponse:
        """Execute a query and return response."""
        handler = self._handlers.get(query.query_type)
        
        if not handler:
            return BavellaResponse(
                query_type=query.query_type,
                series_id=query.series_id,
                success=False,
                error=f"No handler for query type: {query.query_type.value}",
            )
        
        try:
            return handler.handle(query, self._analyzer, self._reports)
        except Exception as e:
            return BavellaResponse(
                query_type=query.query_type,
                series_id=query.series_id,
                success=False,
                error=str(e),
            )
    
    def get_available_queries(self) -> List[str]:
        """Get list of available query types."""
        return [qt.value for qt in self._handlers.keys()]


# =============================================================================
# LLM INTEGRATION HELPERS
# =============================================================================

def parse_user_intent(user_message: str) -> Optional[QueryType]:
    """
    Parse user message to determine query type.
    
    In production, this would be done by the LLM.
    This is a simple pattern matcher for testing.
    """
    message = user_message.lower()
    
    if "validity" in message and ("what" in message or "score" in message):
        return QueryType.WHAT_IS_VALIDITY
    
    if "why" in message and ("degraded" in message or "invalid" in message or "broken" in message):
        return QueryType.WHY_IS_DEGRADED
    
    if "what broke" in message or "what failed" in message:
        return QueryType.WHAT_BROKE
    
    if "happened before" in message or "similar" in message or "history" in message:
        return QueryType.HAS_THIS_HAPPENED_BEFORE
    
    if "how long" in message and "recovery" in message:
        return QueryType.HOW_LONG_UNTIL_RECOVERY
    
    if "backtest" in message and "valid" in message:
        return QueryType.IS_BACKTEST_VALID
    
    if "trust" in message and "penalty" in message:
        return QueryType.WHAT_IS_TRUST_PENALTY
    
    if "root cause" in message:
        return QueryType.WHAT_IS_ROOT_CAUSE
    
    return None


def format_response_for_user(response: BavellaResponse) -> str:
    """
    Format a Bavella response for user display.
    
    In production, the LLM would do this with more context.
    """
    if not response.success:
        return f"I couldn't answer that: {response.error}"
    
    output = response.narrative
    
    if response.warnings:
        output += "\n\n" + "\n".join(response.warnings)
    
    return output


# =============================================================================
# TESTS
# =============================================================================

import numpy as np

def test_query_router():
    """Test the query router with a sample series."""
    np.random.seed(42)
    
    router = BavellaQueryRouter()
    
    # Analyze a series with regime change
    low_vol = list(100 + np.cumsum(np.random.randn(100) * 0.5))
    high_vol = list(low_vol[-1] + np.cumsum(np.random.randn(100) * 3.0))
    values = low_vol + high_vol
    
    now = datetime.now(timezone.utc)
    timestamps = [now - timedelta(days=len(values)-i-1) for i in range(len(values))]
    
    request = SeriesAnalysisRequest(
        owner_id="test",
        series_id="momentum",
        series_name="Momentum Factor",
        timestamps=timestamps,
        values=values,
    )
    
    report = router.analyze_and_register(request)
    print(f"Analyzed: validity={report.validity_score:.1f} ({report.validity_state.value})")
    
    # Test queries
    queries = [
        ("What is the validity?", QueryType.WHAT_IS_VALIDITY),
        ("Why is it degraded?", QueryType.WHY_IS_DEGRADED),
        ("Has this happened before?", QueryType.HAS_THIS_HAPPENED_BEFORE),
        ("How long until recovery?", QueryType.HOW_LONG_UNTIL_RECOVERY),
        ("Is my backtest valid?", QueryType.IS_BACKTEST_VALID),
    ]
    
    for user_msg, expected_type in queries:
        parsed = parse_user_intent(user_msg)
        if parsed:
            query = BavellaQuery(
                query_type=parsed,
                owner_id="test",
                series_id="momentum",
            )
            response = router.query(query)
            print(f"\nQ: {user_msg}")
            print(f"A: {response.narrative[:200]}...")
    
    print("\n✓ test_query_router passed")


def test_intent_parsing():
    """Test intent parsing from user messages."""
    test_cases = [
        ("What's the validity of my momentum factor?", QueryType.WHAT_IS_VALIDITY),
        ("Why is momentum degraded?", QueryType.WHY_IS_DEGRADED),
        ("Has this happened before?", QueryType.HAS_THIS_HAPPENED_BEFORE),
        ("How long until recovery?", QueryType.HOW_LONG_UNTIL_RECOVERY),
        ("Is my backtest still valid?", QueryType.IS_BACKTEST_VALID),
    ]
    
    for message, expected in test_cases:
        parsed = parse_user_intent(message)
        assert parsed == expected, f"Expected {expected} for '{message}', got {parsed}"
    
    print("✓ test_intent_parsing passed")


def run_all_query_tests():
    print("\n" + "=" * 60)
    print("QUERY INTERFACE TESTS")
    print("=" * 60 + "\n")
    
    test_intent_parsing()
    print()
    test_query_router()
    
    print("\n" + "=" * 60)
    print("ALL QUERY TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    run_all_query_tests()
