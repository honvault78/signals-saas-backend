"""
Bavella v2 — Severity Curve Descriptor
=======================================

THE "NOT VIBES" SEVERITY REPRESENTATION

Instead of: max=75, avg=55, trend="increasing"
We get:    shape_class=SHOCK_PERSIST, peak_at=0.15, auc=42.3, half_life=None

This module defines:
    1. SeverityCurveDescriptor - compact, deterministic shape representation
    2. ShapeClassifier - algorithmic shape classification
    3. CurveComparator - similarity between curves

The descriptor is versioned and reproducible.

Copyright 2024-2026 Bavella Technologies Sarl
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
import hashlib
import json


# =============================================================================
# SHAPE CLASSES
# =============================================================================

class ShapeClass(Enum):
    """
    Canonical severity curve shapes.
    
    These are NOT arbitrary labels - they have operational meaning.
    """
    # Sudden spike that reverts
    SHOCK_REVERT = "shock_revert"
    
    # Sudden spike that persists
    SHOCK_PERSIST = "shock_persist"
    
    # Gradual increase
    GRIND_UP = "grind_up"
    
    # Gradual decrease (from elevated)
    GRIND_DOWN = "grind_down"
    
    # Multiple peaks
    MULTI_PEAK = "multi_peak"
    
    # Flat elevated
    PLATEAU = "plateau"
    
    # Oscillating
    VOLATILE = "volatile"
    
    # Insufficient data
    INDETERMINATE = "indeterminate"


# =============================================================================
# SEVERITY CURVE DESCRIPTOR
# =============================================================================

@dataclass(frozen=True)
class SeverityCurveDescriptor:
    """
    Compact, deterministic representation of how severity evolved.
    
    This is what makes similarity matching meaningful.
    "Two episodes with identical descriptors should be meaningfully 
    similar in how it felt."
    """
    # Identity
    descriptor_id: str
    descriptor_version: str = "1.0.0"
    
    # Raw stats
    n_points: int = 0
    duration_hours: float = 0.0
    
    # Key values
    start_severity: float = 0.0
    peak_severity: float = 0.0
    end_severity: float = 0.0
    
    # Timing (normalized 0-1)
    time_to_peak_normalized: float = 0.0  # When did peak occur? 0=start, 1=end
    
    # Quantiles (severity at 10/25/50/75/90% of episode time)
    q10: float = 0.0
    q25: float = 0.0
    q50: float = 0.0  # Median
    q75: float = 0.0
    q90: float = 0.0
    
    # Shape metrics
    auc: float = 0.0  # Area under curve (normalized by duration)
    auc_first_half: float = 0.0  # AUC in first half
    auc_second_half: float = 0.0  # AUC in second half
    
    # Peak analysis
    num_peaks: int = 0
    peak_prominence: float = 0.0  # How much peak exceeds mean
    
    # Decay analysis (from peak)
    half_life_normalized: Optional[float] = None  # Time to drop to 50% of peak
    decay_rate: Optional[float] = None  # Exponential decay rate estimate
    
    # Volatility
    severity_std: float = 0.0
    severity_range: float = 0.0  # max - min
    num_direction_changes: int = 0
    
    # Shape classification
    shape_class: ShapeClass = ShapeClass.INDETERMINATE
    shape_confidence: float = 0.0
    
    # Derivative features
    initial_slope: float = 0.0  # Slope in first 20%
    final_slope: float = 0.0    # Slope in last 20%
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "descriptor_id": self.descriptor_id,
            "descriptor_version": self.descriptor_version,
            "n_points": self.n_points,
            "duration_hours": self.duration_hours,
            "key_values": {
                "start": self.start_severity,
                "peak": self.peak_severity,
                "end": self.end_severity,
            },
            "timing": {
                "time_to_peak_normalized": self.time_to_peak_normalized,
            },
            "quantiles": {
                "q10": self.q10,
                "q25": self.q25,
                "q50": self.q50,
                "q75": self.q75,
                "q90": self.q90,
            },
            "auc": {
                "total": self.auc,
                "first_half": self.auc_first_half,
                "second_half": self.auc_second_half,
            },
            "peaks": {
                "count": self.num_peaks,
                "prominence": self.peak_prominence,
            },
            "decay": {
                "half_life_normalized": self.half_life_normalized,
                "decay_rate": self.decay_rate,
            },
            "volatility": {
                "std": self.severity_std,
                "range": self.severity_range,
                "direction_changes": self.num_direction_changes,
            },
            "shape": {
                "class": self.shape_class.value,
                "confidence": self.shape_confidence,
            },
            "slopes": {
                "initial": self.initial_slope,
                "final": self.final_slope,
            },
        }
    
    def to_vector(self) -> np.ndarray:
        """Convert to feature vector for similarity computation."""
        return np.array([
            self.start_severity / 100,
            self.peak_severity / 100,
            self.end_severity / 100,
            self.time_to_peak_normalized,
            self.q10 / 100,
            self.q25 / 100,
            self.q50 / 100,
            self.q75 / 100,
            self.q90 / 100,
            self.auc / 100,
            self.auc_first_half / 100,
            self.auc_second_half / 100,
            min(self.num_peaks, 5) / 5,  # Cap at 5
            self.peak_prominence / 100,
            self.half_life_normalized if self.half_life_normalized else 0.5,
            self.severity_std / 50,  # Normalize
            min(self.num_direction_changes, 10) / 10,
            (self.initial_slope + 1) / 2,  # Normalize -1 to 1 → 0 to 1
            (self.final_slope + 1) / 2,
        ])


# =============================================================================
# CURVE BUILDER
# =============================================================================

def compute_severity_descriptor(
    severity_values: List[float],
    timestamps: Optional[List[datetime]] = None,
    duration_hours: Optional[float] = None,
) -> SeverityCurveDescriptor:
    """
    Compute severity curve descriptor from time series.
    
    Args:
        severity_values: List of severity readings over time
        timestamps: Optional timestamps (otherwise assumes uniform sampling)
        duration_hours: Optional explicit duration
    """
    if not severity_values or len(severity_values) < 2:
        return SeverityCurveDescriptor(
            descriptor_id=_compute_descriptor_id([]),
            n_points=len(severity_values) if severity_values else 0,
            shape_class=ShapeClass.INDETERMINATE,
        )
    
    values = np.array(severity_values)
    n = len(values)
    
    # Duration
    if duration_hours:
        dur = duration_hours
    elif timestamps and len(timestamps) >= 2:
        dur = (timestamps[-1] - timestamps[0]).total_seconds() / 3600
    else:
        dur = float(n)  # Assume each point is 1 hour
    
    # Key values
    start = values[0]
    peak = np.max(values)
    end = values[-1]
    peak_idx = np.argmax(values)
    time_to_peak = peak_idx / (n - 1) if n > 1 else 0
    
    # Quantiles (temporal, not value-based)
    indices = [int(q * (n - 1)) for q in [0.1, 0.25, 0.5, 0.75, 0.9]]
    q10, q25, q50, q75, q90 = [values[i] for i in indices]
    
    # AUC (normalized) - handle numpy version compatibility
    try:
        auc = np.trapezoid(values) / n  # Average height
        mid = n // 2
        auc_first = np.trapezoid(values[:mid+1]) / (mid + 1) if mid > 0 else auc
        auc_second = np.trapezoid(values[mid:]) / (n - mid) if mid < n else auc
    except AttributeError:
        # Fallback for older numpy
        auc = np.trapz(values) / n
        mid = n // 2
        auc_first = np.trapz(values[:mid+1]) / (mid + 1) if mid > 0 else auc
        auc_second = np.trapz(values[mid:]) / (n - mid) if mid < n else auc
    
    # Peak analysis
    peaks = _find_peaks(values)
    num_peaks = len(peaks)
    mean_sev = np.mean(values)
    peak_prominence = peak - mean_sev
    
    # Decay analysis
    half_life = None
    decay_rate = None
    if peak_idx < n - 1 and peak > 0:
        half_target = peak * 0.5
        for i in range(peak_idx + 1, n):
            if values[i] <= half_target:
                half_life = (i - peak_idx) / (n - 1)  # Normalized
                break
        
        # Estimate decay rate
        post_peak = values[peak_idx:]
        if len(post_peak) > 2 and post_peak[0] > 0:
            try:
                log_vals = np.log(np.clip(post_peak / post_peak[0], 1e-10, 1))
                decay_rate = -np.mean(np.diff(log_vals))
            except:
                decay_rate = None
    
    # Volatility
    std = np.std(values)
    range_val = np.max(values) - np.min(values)
    diffs = np.diff(values)
    direction_changes = np.sum(diffs[:-1] * diffs[1:] < 0) if len(diffs) > 1 else 0
    
    # Slopes
    first_20_idx = max(1, int(0.2 * n))
    last_20_idx = min(n - 1, int(0.8 * n))
    
    initial_slope = (values[first_20_idx] - values[0]) / (first_20_idx + 1) if first_20_idx > 0 else 0
    final_slope = (values[-1] - values[last_20_idx]) / (n - last_20_idx) if last_20_idx < n - 1 else 0
    
    # Normalize slopes to -1 to 1 range
    max_slope = np.max(np.abs(diffs)) if len(diffs) > 0 else 1
    if max_slope > 0:
        initial_slope = np.clip(initial_slope / max_slope, -1, 1)
        final_slope = np.clip(final_slope / max_slope, -1, 1)
    
    # Shape classification
    shape_class, shape_conf = _classify_shape(
        values, time_to_peak, half_life, num_peaks, 
        direction_changes, initial_slope, final_slope, std
    )
    
    # Compute ID
    desc_id = _compute_descriptor_id(severity_values)
    
    return SeverityCurveDescriptor(
        descriptor_id=desc_id,
        n_points=n,
        duration_hours=dur,
        start_severity=float(start),
        peak_severity=float(peak),
        end_severity=float(end),
        time_to_peak_normalized=float(time_to_peak),
        q10=float(q10),
        q25=float(q25),
        q50=float(q50),
        q75=float(q75),
        q90=float(q90),
        auc=float(auc),
        auc_first_half=float(auc_first),
        auc_second_half=float(auc_second),
        num_peaks=num_peaks,
        peak_prominence=float(peak_prominence),
        half_life_normalized=half_life,
        decay_rate=decay_rate,
        severity_std=float(std),
        severity_range=float(range_val),
        num_direction_changes=int(direction_changes),
        shape_class=shape_class,
        shape_confidence=shape_conf,
        initial_slope=float(initial_slope),
        final_slope=float(final_slope),
    )


def _find_peaks(values: np.ndarray, min_prominence: float = 5.0) -> List[int]:
    """Find peaks in severity curve."""
    peaks = []
    for i in range(1, len(values) - 1):
        if values[i] > values[i-1] and values[i] > values[i+1]:
            # Check prominence
            left_base = np.min(values[:i+1])
            right_base = np.min(values[i:])
            prominence = values[i] - max(left_base, right_base)
            if prominence >= min_prominence:
                peaks.append(i)
    return peaks


def _classify_shape(
    values: np.ndarray,
    time_to_peak: float,
    half_life: Optional[float],
    num_peaks: int,
    direction_changes: int,
    initial_slope: float,
    final_slope: float,
    std: float,
) -> Tuple[ShapeClass, float]:
    """Classify the shape of the severity curve."""
    n = len(values)
    mean = np.mean(values)
    
    # Check for insufficient data
    if n < 5:
        return ShapeClass.INDETERMINATE, 0.3
    
    # Multi-peak
    if num_peaks >= 2:
        return ShapeClass.MULTI_PEAK, 0.8
    
    # High volatility / oscillating
    cv = std / mean if mean > 0 else 0
    if cv > 0.4 and direction_changes > n * 0.3:
        return ShapeClass.VOLATILE, 0.7
    
    # Shock patterns
    if time_to_peak < 0.3:  # Peak in first 30%
        if half_life and half_life < 0.4:
            return ShapeClass.SHOCK_REVERT, 0.85
        else:
            return ShapeClass.SHOCK_PERSIST, 0.8
    
    # Grind patterns
    if initial_slope > 0.3 and time_to_peak > 0.6:
        return ShapeClass.GRIND_UP, 0.75
    
    if initial_slope < -0.3 and time_to_peak < 0.3:
        return ShapeClass.GRIND_DOWN, 0.75
    
    # Plateau
    if cv < 0.15 and direction_changes < n * 0.15:
        return ShapeClass.PLATEAU, 0.7
    
    # Default
    return ShapeClass.INDETERMINATE, 0.4


def _compute_descriptor_id(values: List[float]) -> str:
    """Compute deterministic ID for descriptor."""
    content = json.dumps(values, sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


# =============================================================================
# CURVE SIMILARITY
# =============================================================================

def compute_curve_similarity(
    desc1: SeverityCurveDescriptor,
    desc2: SeverityCurveDescriptor,
) -> Tuple[float, Dict[str, float]]:
    """
    Compute similarity between two severity curve descriptors.
    
    Returns: (overall_similarity, breakdown_by_dimension)
    """
    v1 = desc1.to_vector()
    v2 = desc2.to_vector()
    
    # Dimension names for breakdown
    dim_names = [
        "start_severity", "peak_severity", "end_severity",
        "time_to_peak", "q10", "q25", "q50", "q75", "q90",
        "auc_total", "auc_first", "auc_second",
        "num_peaks", "peak_prominence", "half_life",
        "volatility", "direction_changes",
        "initial_slope", "final_slope",
    ]
    
    # Weights (higher for more important dimensions)
    weights = np.array([
        0.05, 0.12, 0.08,  # Key values: peak most important
        0.10, 0.03, 0.05, 0.07, 0.05, 0.03,  # Quantiles: median most important
        0.08, 0.04, 0.04,  # AUC
        0.06, 0.05, 0.06,  # Peaks
        0.04, 0.03,  # Volatility
        0.04, 0.04,  # Slopes
    ])
    weights = weights / weights.sum()  # Normalize
    
    # Per-dimension similarity (1 - |diff|)
    diffs = np.abs(v1 - v2)
    dim_similarities = 1 - np.clip(diffs, 0, 1)
    
    # Shape class match bonus
    shape_bonus = 0.05 if desc1.shape_class == desc2.shape_class else 0
    
    # Weighted overall
    overall = float(np.dot(dim_similarities, weights)) + shape_bonus
    overall = min(1.0, overall)
    
    # Breakdown
    breakdown = {name: float(sim) for name, sim in zip(dim_names, dim_similarities)}
    breakdown["shape_class_match"] = 1.0 if desc1.shape_class == desc2.shape_class else 0.0
    
    return overall, breakdown


# =============================================================================
# TESTS
# =============================================================================

def test_shock_revert():
    """Test shock-revert pattern detection."""
    # Sharp spike then decay
    values = [10, 80, 70, 50, 30, 20, 15, 12, 10, 10]
    desc = compute_severity_descriptor(values)
    
    print(f"Shape: {desc.shape_class.value} (conf={desc.shape_confidence})")
    print(f"Peak: {desc.peak_severity} at t={desc.time_to_peak_normalized:.2f}")
    print(f"Half-life: {desc.half_life_normalized}")
    print(f"AUC: {desc.auc:.1f}")
    
    assert desc.shape_class == ShapeClass.SHOCK_REVERT
    assert desc.time_to_peak_normalized < 0.3
    assert desc.half_life_normalized is not None and desc.half_life_normalized < 0.5
    
    print("✓ test_shock_revert passed")


def test_grind_up():
    """Test grind-up pattern detection."""
    # Gradual increase
    values = [10, 15, 22, 30, 40, 52, 65, 75, 82, 85]
    desc = compute_severity_descriptor(values)
    
    print(f"Shape: {desc.shape_class.value} (conf={desc.shape_confidence})")
    print(f"Initial slope: {desc.initial_slope:.2f}")
    print(f"Time to peak: {desc.time_to_peak_normalized:.2f}")
    
    assert desc.shape_class == ShapeClass.GRIND_UP
    assert desc.initial_slope > 0
    
    print("✓ test_grind_up passed")


def test_multi_peak():
    """Test multi-peak pattern detection."""
    # Two peaks
    values = [10, 60, 30, 20, 70, 40, 25, 20, 15, 10]
    desc = compute_severity_descriptor(values)
    
    print(f"Shape: {desc.shape_class.value} (conf={desc.shape_confidence})")
    print(f"Num peaks: {desc.num_peaks}")
    
    assert desc.shape_class == ShapeClass.MULTI_PEAK
    assert desc.num_peaks >= 2
    
    print("✓ test_multi_peak passed")


def test_curve_similarity():
    """Test similarity between curves."""
    # Two similar shock-revert patterns
    values1 = [10, 80, 70, 50, 30, 20, 15, 12, 10, 10]
    values2 = [12, 75, 65, 48, 32, 22, 18, 14, 11, 10]
    
    desc1 = compute_severity_descriptor(values1)
    desc2 = compute_severity_descriptor(values2)
    
    sim_high, breakdown_high = compute_curve_similarity(desc1, desc2)
    
    # One different pattern (grind up)
    values3 = [10, 15, 22, 30, 40, 52, 65, 75, 82, 85]
    desc3 = compute_severity_descriptor(values3)
    
    sim_low, breakdown_low = compute_curve_similarity(desc1, desc3)
    
    print(f"Similar curves: {sim_high:.2f}")
    print(f"Different curves: {sim_low:.2f}")
    print(f"Key dims (high): peak={breakdown_high['peak_severity']:.2f}, shape={breakdown_high['shape_class_match']}")
    print(f"Key dims (low): peak={breakdown_low['peak_severity']:.2f}, shape={breakdown_low['shape_class_match']}")
    
    assert sim_high > sim_low
    assert sim_high > 0.8
    assert sim_low < 0.75  # Different patterns should be noticeably less similar
    
    print("✓ test_curve_similarity passed")


def test_descriptor_determinism():
    """Test that descriptor computation is deterministic."""
    values = [10, 50, 80, 60, 40, 30, 25, 20, 15, 12]
    
    desc1 = compute_severity_descriptor(values)
    desc2 = compute_severity_descriptor(values)
    
    assert desc1.descriptor_id == desc2.descriptor_id
    assert desc1.auc == desc2.auc
    assert desc1.shape_class == desc2.shape_class
    
    print(f"ID: {desc1.descriptor_id}")
    print("✓ test_descriptor_determinism passed")


def run_all_severity_curve_tests():
    print("\n" + "=" * 60)
    print("SEVERITY CURVE DESCRIPTOR TESTS")
    print("=" * 60 + "\n")
    
    test_shock_revert()
    print()
    test_grind_up()
    print()
    test_multi_peak()
    print()
    test_curve_similarity()
    print()
    test_descriptor_determinism()
    
    print("\n" + "=" * 60)
    print("ALL SEVERITY CURVE TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    run_all_severity_curve_tests()
