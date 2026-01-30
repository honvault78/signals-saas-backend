"""
Bavella Validity Engine — Episode Tracking
============================================

Tracks validity episodes: periods when a relationship is INVALID or DEGRADED.

Episodes answer:
- "When did this break start?"
- "How long has it been invalid?"
- "What's the trajectory?"
- "Has it recovered before?"

For production: This in-memory store should be backed by Postgres.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional
from collections import defaultdict
import uuid

from .core import FailureMode, ValidityState, ValidityVerdict, FM_INFO


# =============================================================================
# EPISODE STATES
# =============================================================================

class EpisodeState(Enum):
    """State of a validity episode."""
    ACTIVE = "active"       # Episode ongoing
    RESOLVED = "resolved"   # Episode ended, relationship recovered
    REBASELINED = "rebaselined"  # Episode ended via rebaseline (FM4)


# =============================================================================
# EPISODE
# =============================================================================

@dataclass
class Episode:
    """
    A period of invalidity for a node.
    
    Tracks the full lifecycle of a validity break.
    """
    # Identity
    episode_id: str
    node_id: str
    owner_id: str
    
    # State
    state: EpisodeState
    
    # Timing
    started_at: datetime
    ended_at: Optional[datetime] = None
    duration_days: float = 0.0
    
    # Failure modes
    initial_fms: List[FailureMode] = field(default_factory=list)
    peak_fms: List[FailureMode] = field(default_factory=list)
    root_cause_fm: Optional[FailureMode] = None
    
    # Severity trajectory
    initial_validity: float = 100.0
    min_validity: float = 100.0  # Worst point
    current_validity: float = 100.0
    
    # Verdicts (history)
    verdict_count: int = 0
    
    # Recovery
    recovery_type: Optional[str] = None  # "full", "partial", "rebaseline"
    
    @classmethod
    def create(
        cls,
        node_id: str,
        owner_id: str,
        verdict: ValidityVerdict,
    ) -> "Episode":
        """Create new episode from initial verdict."""
        return cls(
            episode_id=str(uuid.uuid4()),
            node_id=node_id,
            owner_id=owner_id,
            state=EpisodeState.ACTIVE,
            started_at=verdict.timestamp,
            initial_fms=list(verdict.active_fms),
            peak_fms=list(verdict.active_fms),
            root_cause_fm=verdict.primary_fm,
            initial_validity=verdict.validity_score,
            min_validity=verdict.validity_score,
            current_validity=verdict.validity_score,
            verdict_count=1,
        )
    
    def update(self, verdict: ValidityVerdict) -> None:
        """Update episode with new verdict."""
        self.current_validity = verdict.validity_score
        self.verdict_count += 1
        
        # Track worst point
        if verdict.validity_score < self.min_validity:
            self.min_validity = verdict.validity_score
            self.peak_fms = list(verdict.active_fms)
        
        # Update duration
        self.duration_days = (
            verdict.timestamp - self.started_at
        ).total_seconds() / 86400
        
        # Check for resolution
        if verdict.validity_state == ValidityState.VALID:
            self.resolve(verdict.timestamp, "full")
    
    def resolve(
        self,
        ended_at: datetime,
        recovery_type: str = "full",
    ) -> None:
        """Mark episode as resolved."""
        self.state = EpisodeState.RESOLVED
        self.ended_at = ended_at
        self.recovery_type = recovery_type
        self.duration_days = (ended_at - self.started_at).total_seconds() / 86400
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "node_id": self.node_id,
            "owner_id": self.owner_id,
            "state": self.state.value,
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "duration_days": round(self.duration_days, 1),
            "initial_fms": [fm.value for fm in self.initial_fms],
            "peak_fms": [fm.value for fm in self.peak_fms],
            "root_cause_fm": self.root_cause_fm.value if self.root_cause_fm else None,
            "root_cause_name": FM_INFO[self.root_cause_fm]["name"] if self.root_cause_fm else None,
            "severity": {
                "initial": round(self.initial_validity, 1),
                "min": round(self.min_validity, 1),
                "current": round(self.current_validity, 1),
            },
            "verdict_count": self.verdict_count,
            "recovery_type": self.recovery_type,
        }


@dataclass
class EpisodeSummary:
    """Summary statistics for episodes."""
    total_episodes: int
    active_episodes: int
    resolved_episodes: int
    avg_duration_days: float
    most_common_fm: Optional[str]
    worst_severity: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_episodes": self.total_episodes,
            "active_episodes": self.active_episodes,
            "resolved_episodes": self.resolved_episodes,
            "avg_duration_days": round(self.avg_duration_days, 1),
            "most_common_fm": self.most_common_fm,
            "worst_severity": round(self.worst_severity, 1),
        }


# =============================================================================
# EPISODE STORE
# =============================================================================

class EpisodeStore:
    """
    In-memory episode storage.
    
    For production, back this with Postgres.
    """
    
    def __init__(self):
        self._episodes: Dict[str, Episode] = {}  # episode_id -> Episode
        self._by_node: Dict[str, List[str]] = defaultdict(list)  # node_id -> [episode_ids]
        self._active: Dict[str, str] = {}  # node_id -> active_episode_id
    
    def process_verdict(
        self,
        node_id: str,
        owner_id: str,
        verdict: ValidityVerdict,
    ) -> Optional[Episode]:
        """
        Process a new verdict, creating/updating episodes as needed.
        
        Returns the episode if one is active/created.
        """
        # Check for active episode
        active_episode_id = self._active.get(node_id)
        
        if verdict.validity_state == ValidityState.VALID:
            # Resolved
            if active_episode_id:
                episode = self._episodes[active_episode_id]
                episode.update(verdict)
                del self._active[node_id]
                return episode
            return None
        
        else:
            # DEGRADED or INVALID
            if active_episode_id:
                # Update existing
                episode = self._episodes[active_episode_id]
                episode.update(verdict)
                return episode
            else:
                # Create new episode
                episode = Episode.create(node_id, owner_id, verdict)
                self._episodes[episode.episode_id] = episode
                self._by_node[node_id].append(episode.episode_id)
                self._active[node_id] = episode.episode_id
                return episode
    
    def get_active_episode(self, node_id: str) -> Optional[Episode]:
        """Get the active episode for a node, if any."""
        episode_id = self._active.get(node_id)
        if episode_id:
            return self._episodes.get(episode_id)
        return None
    
    def get_node_history(
        self,
        node_id: str,
        limit: int = 10,
    ) -> List[Episode]:
        """Get episode history for a node."""
        episode_ids = self._by_node.get(node_id, [])
        episodes = [self._episodes[eid] for eid in episode_ids[-limit:]]
        return sorted(episodes, key=lambda e: e.started_at, reverse=True)
    
    def get_similar_episodes(
        self,
        root_fm: FailureMode,
        min_severity: float = 30,
        limit: int = 5,
    ) -> List[Episode]:
        """Find episodes with similar root cause."""
        similar = []
        
        for episode in self._episodes.values():
            if episode.root_cause_fm == root_fm:
                if episode.min_validity <= (100 - min_severity):
                    similar.append(episode)
        
        # Sort by recency
        similar.sort(key=lambda e: e.started_at, reverse=True)
        return similar[:limit]
    
    def get_summary(self, owner_id: Optional[str] = None) -> EpisodeSummary:
        """Get summary statistics."""
        episodes = list(self._episodes.values())
        
        if owner_id:
            episodes = [e for e in episodes if e.owner_id == owner_id]
        
        if not episodes:
            return EpisodeSummary(
                total_episodes=0,
                active_episodes=0,
                resolved_episodes=0,
                avg_duration_days=0,
                most_common_fm=None,
                worst_severity=100,
            )
        
        active = [e for e in episodes if e.state == EpisodeState.ACTIVE]
        resolved = [e for e in episodes if e.state == EpisodeState.RESOLVED]
        
        # Average duration (of resolved)
        if resolved:
            avg_duration = sum(e.duration_days for e in resolved) / len(resolved)
        else:
            avg_duration = 0
        
        # Most common FM
        fm_counts: Dict[str, int] = defaultdict(int)
        for e in episodes:
            if e.root_cause_fm:
                fm_counts[e.root_cause_fm.value] += 1
        
        most_common = None
        if fm_counts:
            most_common = max(fm_counts, key=fm_counts.get)
        
        # Worst severity
        worst = min(e.min_validity for e in episodes)
        
        return EpisodeSummary(
            total_episodes=len(episodes),
            active_episodes=len(active),
            resolved_episodes=len(resolved),
            avg_duration_days=avg_duration,
            most_common_fm=most_common,
            worst_severity=worst,
        )
    
    def clear(self) -> None:
        """Clear all episodes (for testing)."""
        self._episodes.clear()
        self._by_node.clear()
        self._active.clear()


# Global episode store (for simple usage)
_global_store: Optional[EpisodeStore] = None


def get_episode_store() -> EpisodeStore:
    """Get or create global episode store."""
    global _global_store
    if _global_store is None:
        _global_store = EpisodeStore()
    return _global_store
