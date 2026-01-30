"""
Bavella v2 — Visualization
===========================

Professional-grade visualization components for validity analysis.

Key visualizations:
    - Validity Dashboard: Multi-panel validity overview
    - Transform Panel: NIS transformation visualization
    - Failure Mode Panel: FM severity breakdown
    - Regime Panel: Regime detection visualization
    - Timeline Panel: Validity score over time

Copyright 2024-2026 Bavella Technologies Sarl
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import io
import base64

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import Rectangle, FancyBboxPatch
    from matplotlib.colors import LinearSegmentedColormap
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from .models import ValidityState, ValidityComputation, FailureModeID
from .transforms import NISTransformResult
from .validity_engine import CompleteAnalysisResult
from .passport import SeriesPassport


# =============================================================================
# COLOR SCHEMES
# =============================================================================

COLORS = {
    # Validity states
    "valid": "#22c55e",
    "degraded": "#f59e0b", 
    "invalid": "#ef4444",
    "unknown": "#6b7280",
    
    # Bavella brand
    "primary": "#2563eb",
    "secondary": "#64748b",
    "background": "#f8fafc",
    "surface": "#ffffff",
    "text": "#1e293b",
    "text_secondary": "#64748b",
    
    # Chart colors
    "series": "#2563eb",
    "z_t": "#8b5cf6",
    "Z_t": "#06b6d4",
    "trend": "#f97316",
    "seasonal": "#84cc16",
    "residual": "#ec4899",
    
    # FM colors
    "fm1": "#ef4444",
    "fm2": "#f97316",
    "fm3": "#eab308",
    "fm4": "#84cc16",
    "fm5": "#22c55e",
    "fm6": "#06b6d4",
    "fm7": "#8b5cf6",
}

def get_validity_color(state: ValidityState) -> str:
    """Get color for validity state."""
    return {
        ValidityState.VALID: COLORS["valid"],
        ValidityState.DEGRADED: COLORS["degraded"],
        ValidityState.INVALID: COLORS["invalid"],
        ValidityState.UNKNOWN: COLORS["unknown"],
    }.get(state, COLORS["unknown"])


# =============================================================================
# VALIDITY DASHBOARD
# =============================================================================

class ValidityDashboard:
    """
    Creates a comprehensive validity dashboard.
    
    Layout:
        ┌─────────────────────────────────────────────┐
        │  Header: Score, State, Attribution Summary  │
        ├─────────────────────┬───────────────────────┤
        │  Original Series    │  Normalized (z_t)     │
        ├─────────────────────┼───────────────────────┤
        │  Cumulative (Z_t)   │  FM Severity Radar    │
        ├─────────────────────┴───────────────────────┤
        │  Attribution Breakdown                       │
        └─────────────────────────────────────────────┘
    """
    
    def __init__(self, figsize: Tuple[int, int] = (14, 16)):
        self.figsize = figsize
        
    def create(
        self,
        result: CompleteAnalysisResult,
        passport: Optional[SeriesPassport] = None,
        title: Optional[str] = None,
    ) -> "plt.Figure":
        """
        Create the validity dashboard figure.
        
        Args:
            result: Complete analysis result
            passport: Optional Series Passport for additional info
            title: Optional title override
            
        Returns:
            matplotlib Figure object
        """
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib required for visualization")
            
        fig = plt.figure(figsize=self.figsize, facecolor=COLORS["background"])
        
        # Create grid
        gs = gridspec.GridSpec(
            5, 2,
            figure=fig,
            height_ratios=[0.8, 1.5, 1.5, 1.5, 1],
            hspace=0.3,
            wspace=0.25,
            left=0.08,
            right=0.95,
            top=0.95,
            bottom=0.05,
        )
        
        # Header panel
        ax_header = fig.add_subplot(gs[0, :])
        self._draw_header(ax_header, result, title)
        
        # Original series
        ax_original = fig.add_subplot(gs[1, 0])
        self._draw_original_series(ax_original, result)
        
        # Normalized z_t
        ax_z = fig.add_subplot(gs[1, 1])
        self._draw_normalized(ax_z, result)
        
        # Cumulative Z_t
        ax_Z = fig.add_subplot(gs[2, 0])
        self._draw_cumulative(ax_Z, result)
        
        # FM Radar
        ax_radar = fig.add_subplot(gs[2, 1], projection='polar')
        self._draw_fm_radar(ax_radar, result)
        
        # Scale series and structural info
        ax_scale = fig.add_subplot(gs[3, 0])
        self._draw_scale(ax_scale, result)
        
        # Validity gauge
        ax_gauge = fig.add_subplot(gs[3, 1])
        self._draw_validity_gauge(ax_gauge, result)
        
        # Attribution breakdown
        ax_attr = fig.add_subplot(gs[4, :])
        self._draw_attribution(ax_attr, result)
        
        return fig
    
    def _draw_header(
        self,
        ax: "plt.Axes",
        result: CompleteAnalysisResult,
        title: Optional[str]
    ):
        """Draw the header panel."""
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        validity = result.validity
        state_color = get_validity_color(validity.validity_state)
        
        # Title
        display_title = title or f"Validity Analysis: {result.target_id}"
        ax.text(0, 0.7, display_title, fontsize=16, fontweight='bold', color=COLORS["text"])
        
        # Score badge
        score_box = FancyBboxPatch(
            (0, 0.1), 1.2, 0.5,
            boxstyle="round,pad=0.02,rounding_size=0.1",
            facecolor=state_color,
            edgecolor='none',
        )
        ax.add_patch(score_box)
        ax.text(0.6, 0.35, f"{validity.validity_score:.0f}", fontsize=24, fontweight='bold',
                color='white', ha='center', va='center')
        
        # State label
        ax.text(1.5, 0.35, validity.validity_state.value.upper(),
                fontsize=14, fontweight='bold', color=state_color, va='center')
        
        # Trend indicator
        trend_symbol = {"stable": "→", "degrading": "↓", "recovering": "↑"}.get(
            validity.validity_trend.value, "→"
        )
        ax.text(3, 0.35, f"Trend: {trend_symbol} ({validity.trend_slope:+.2f}/day)",
                fontsize=10, color=COLORS["text_secondary"], va='center')
        
        # Timestamp
        ax.text(10, 0.7, f"As of: {validity.as_of_ts.strftime('%Y-%m-%d %H:%M UTC')}",
                fontsize=9, color=COLORS["text_secondary"], ha='right')
        
        # Series type
        ax.text(10, 0.35, f"Type: {result.transform_result.series_type_detected.value}",
                fontsize=9, color=COLORS["text_secondary"], ha='right')
    
    def _draw_original_series(self, ax: "plt.Axes", result: CompleteAnalysisResult):
        """Draw original series panel."""
        series = result.transform_result.original_series
        if series is None or len(series) == 0:
            ax.text(0.5, 0.5, "No data", ha='center', va='center')
            return
            
        ax.plot(series.index, series.values, color=COLORS["series"], linewidth=1)
        ax.fill_between(series.index, series.values, alpha=0.1, color=COLORS["series"])
        
        ax.set_title("Original Series", fontsize=11, fontweight='bold', color=COLORS["text"])
        ax.set_xlabel("")
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3)
        
        # Add suppression watermark if needed
        if result.validity.validity_state == ValidityState.INVALID:
            ax.text(0.5, 0.5, "INFERENCE SUPPRESSED", transform=ax.transAxes,
                    fontsize=14, color=COLORS["invalid"], alpha=0.5,
                    ha='center', va='center', rotation=30, fontweight='bold')
    
    def _draw_normalized(self, ax: "plt.Axes", result: CompleteAnalysisResult):
        """Draw normalized innovations (z_t)."""
        z_t = result.transform_result.z_t
        if z_t is None or len(z_t) == 0:
            ax.text(0.5, 0.5, "No data", ha='center', va='center')
            return
            
        ax.plot(z_t.index, z_t.values, color=COLORS["z_t"], linewidth=0.8)
        
        # Add ±2σ bands
        ax.axhline(2, color=COLORS["degraded"], linestyle='--', alpha=0.5, linewidth=0.8)
        ax.axhline(-2, color=COLORS["degraded"], linestyle='--', alpha=0.5, linewidth=0.8)
        ax.axhline(0, color=COLORS["text_secondary"], linestyle='-', alpha=0.3, linewidth=0.5)
        
        ax.set_title("Normalized Innovations (z_t)", fontsize=11, fontweight='bold', color=COLORS["text"])
        ax.set_ylabel("σ", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3)
    
    def _draw_cumulative(self, ax: "plt.Axes", result: CompleteAnalysisResult):
        """Draw cumulative process (Z_t)."""
        Z_t = result.transform_result.Z_t
        if Z_t is None or len(Z_t) == 0:
            ax.text(0.5, 0.5, "No data", ha='center', va='center')
            return
            
        ax.plot(Z_t.index, Z_t.values, color=COLORS["Z_t"], linewidth=1)
        ax.fill_between(Z_t.index, Z_t.values, alpha=0.1, color=COLORS["Z_t"])
        ax.axhline(0, color=COLORS["text_secondary"], linestyle='-', alpha=0.3, linewidth=0.5)
        
        ax.set_title("Cumulative Process (Z_t)", fontsize=11, fontweight='bold', color=COLORS["text"])
        ax.set_ylabel("Σz", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3)
    
    def _draw_fm_radar(self, ax: "plt.Axes", result: CompleteAnalysisResult):
        """Draw failure mode severity radar."""
        # Get severities
        fm_severities = {}
        for signal in result.signals:
            fm_severities[signal.failure_mode_id] = signal.severity
            
        # Order and labels
        fm_order = [
            FailureModeID.FM1_VARIANCE_REGIME_SHIFT,
            FailureModeID.FM2_MEAN_DRIFT,
            FailureModeID.FM3_SEASONALITY_MISMATCH,
            FailureModeID.FM4_STRUCTURAL_BREAK,
            FailureModeID.FM5_OUTLIER_CONTAMINATION,
            FailureModeID.FM6_DISTRIBUTIONAL_SHIFT,
        ]
        labels = ["Variance", "Mean Drift", "Seasonality", "Breaks", "Outliers", "Distribution"]
        
        values = [fm_severities.get(fm, 0) for fm in fm_order]
        values.append(values[0])  # Close the polygon
        
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        angles.append(angles[0])
        
        # Plot
        ax.plot(angles, values, color=COLORS["invalid"], linewidth=2)
        ax.fill(angles, values, color=COLORS["invalid"], alpha=0.25)
        
        # Threshold lines
        ax.plot(angles, [40] * len(angles), color=COLORS["degraded"], linestyle='--', alpha=0.5)
        ax.plot(angles, [75] * len(angles), color=COLORS["invalid"], linestyle='--', alpha=0.5)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, size=8)
        ax.set_ylim(0, 100)
        ax.set_title("FM Severity", fontsize=11, fontweight='bold', color=COLORS["text"], pad=10)
    
    def _draw_scale(self, ax: "plt.Axes", result: CompleteAnalysisResult):
        """Draw scale series and structural info."""
        scale = result.transform_result.scale_series
        if scale is None or len(scale) == 0:
            ax.text(0.5, 0.5, "No scale data", ha='center', va='center')
            return
            
        ax.plot(scale.index, scale.values, color=COLORS["trend"], linewidth=1)
        ax.fill_between(scale.index, scale.values, alpha=0.2, color=COLORS["trend"])
        
        ax.set_title("Local Volatility Estimate", fontsize=11, fontweight='bold', color=COLORS["text"])
        ax.set_ylabel("Scale", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3)
    
    def _draw_validity_gauge(self, ax: "plt.Axes", result: CompleteAnalysisResult):
        """Draw validity score gauge."""
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-0.5, 1.2)
        ax.axis('off')
        
        score = result.validity.validity_score
        state = result.validity.validity_state
        
        # Draw arc background
        theta = np.linspace(np.pi, 0, 100)
        x = np.cos(theta)
        y = np.sin(theta)
        
        # Color segments
        for i, (start, end, color) in enumerate([
            (0, 30, COLORS["invalid"]),
            (30, 70, COLORS["degraded"]),
            (70, 100, COLORS["valid"]),
        ]):
            start_angle = np.pi * (1 - start/100)
            end_angle = np.pi * (1 - end/100)
            segment_theta = np.linspace(start_angle, end_angle, 30)
            segment_x = np.cos(segment_theta)
            segment_y = np.sin(segment_theta)
            ax.plot(segment_x, segment_y, color=color, linewidth=20, solid_capstyle='round')
            
        # Needle
        needle_angle = np.pi * (1 - score/100)
        needle_x = 0.8 * np.cos(needle_angle)
        needle_y = 0.8 * np.sin(needle_angle)
        ax.annotate('', xy=(needle_x, needle_y), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color=COLORS["text"], lw=2))
        
        # Center circle
        circle = plt.Circle((0, 0), 0.15, color=COLORS["text"])
        ax.add_patch(circle)
        
        # Score text
        ax.text(0, -0.3, f"{score:.0f}", fontsize=28, fontweight='bold',
                color=get_validity_color(state), ha='center')
        ax.text(0, -0.45, state.value.upper(), fontsize=12, fontweight='bold',
                color=get_validity_color(state), ha='center')
        
        # Labels
        ax.text(-1.3, 0, "0", fontsize=9, color=COLORS["text_secondary"], ha='center')
        ax.text(0, 1.1, "50", fontsize=9, color=COLORS["text_secondary"], ha='center')
        ax.text(1.3, 0, "100", fontsize=9, color=COLORS["text_secondary"], ha='center')
        
        ax.set_title("Validity Score", fontsize=11, fontweight='bold', color=COLORS["text"])
    
    def _draw_attribution(self, ax: "plt.Axes", result: CompleteAnalysisResult):
        """Draw attribution breakdown."""
        ax.axis('off')
        
        attributions = result.validity.attributions
        if not attributions:
            ax.text(0.5, 0.5, "Score = 100: No validity loss to attribute",
                    ha='center', va='center', fontsize=11, color=COLORS["text_secondary"])
            return
            
        # Sort by contribution
        sorted_attr = sorted(attributions, key=lambda a: a.contribution_pct, reverse=True)
        
        # Draw bars
        y_positions = np.linspace(0.8, 0.2, min(5, len(sorted_attr)))
        
        fm_colors = {
            FailureModeID.FM1_VARIANCE_REGIME_SHIFT: COLORS["fm1"],
            FailureModeID.FM2_MEAN_DRIFT: COLORS["fm2"],
            FailureModeID.FM3_SEASONALITY_MISMATCH: COLORS["fm3"],
            FailureModeID.FM4_STRUCTURAL_BREAK: COLORS["fm4"],
            FailureModeID.FM5_OUTLIER_CONTAMINATION: COLORS["fm5"],
            FailureModeID.FM6_DISTRIBUTIONAL_SHIFT: COLORS["fm6"],
            FailureModeID.FM7_DEPENDENCY_BREAK: COLORS["fm7"],
        }
        
        ax.text(0.05, 0.95, "Attribution of Validity Loss", fontsize=11, fontweight='bold',
                color=COLORS["text"], transform=ax.transAxes)
        
        for i, attr in enumerate(sorted_attr[:5]):
            y = y_positions[i]
            pct = attr.contribution_pct
            color = fm_colors.get(attr.failure_mode_id, COLORS["secondary"])
            
            # Bar
            bar_width = pct / 100 * 0.6
            bar = FancyBboxPatch(
                (0.25, y - 0.05), bar_width, 0.08,
                boxstyle="round,pad=0.01,rounding_size=0.02",
                facecolor=color, edgecolor='none', transform=ax.transAxes
            )
            ax.add_patch(bar)
            
            # Label
            fm_name = attr.failure_mode_id.value.replace("_", " ").replace("fm", "FM").title()
            ax.text(0.05, y, fm_name[:25], fontsize=9, color=COLORS["text"],
                    va='center', transform=ax.transAxes)
            
            # Percentage
            ax.text(0.88, y, f"{pct:.1f}%", fontsize=9, color=COLORS["text"],
                    va='center', ha='right', transform=ax.transAxes)
            
            # Summary (truncated)
            summary = attr.human_summary[:60] + "..." if len(attr.human_summary) > 60 else attr.human_summary
            ax.text(0.92, y, summary, fontsize=7, color=COLORS["text_secondary"],
                    va='center', transform=ax.transAxes)
    
    def save(
        self,
        result: CompleteAnalysisResult,
        filepath: str,
        **kwargs
    ):
        """Save dashboard to file."""
        fig = self.create(result, **kwargs)
        fig.savefig(filepath, dpi=150, bbox_inches='tight', facecolor=COLORS["background"])
        plt.close(fig)
        
    def to_base64(self, result: CompleteAnalysisResult, **kwargs) -> str:
        """Generate dashboard as base64 PNG."""
        fig = self.create(result, **kwargs)
        
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor=COLORS["background"])
        plt.close(fig)
        
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def visualize_analysis(
    result: CompleteAnalysisResult,
    save_path: Optional[str] = None,
    show: bool = True,
) -> Optional["plt.Figure"]:
    """
    Convenience function to visualize analysis results.
    
    Args:
        result: CompleteAnalysisResult from analyze_series()
        save_path: Optional path to save figure
        show: Whether to display the figure
        
    Returns:
        Figure object if show=False, else None
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required. Install with: pip install matplotlib")
        
    dashboard = ValidityDashboard()
    fig = dashboard.create(result)
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=COLORS["background"])
        
    if show:
        plt.show()
        return None
    else:
        return fig


def plot_validity_timeline(
    history: List[ValidityComputation],
    title: str = "Validity Score Timeline",
    figsize: Tuple[int, int] = (12, 4),
) -> "plt.Figure":
    """
    Plot validity score over time.
    
    Args:
        history: List of ValidityComputation objects
        title: Plot title
        figsize: Figure size
        
    Returns:
        Figure object
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required")
        
    fig, ax = plt.subplots(figsize=figsize, facecolor=COLORS["background"])
    ax.set_facecolor(COLORS["surface"])
    
    timestamps = [h.as_of_ts for h in history]
    scores = [h.validity_score for h in history]
    states = [h.validity_state for h in history]
    
    # Color by state
    colors = [get_validity_color(s) for s in states]
    
    ax.plot(timestamps, scores, color=COLORS["primary"], linewidth=1.5, zorder=2)
    ax.scatter(timestamps, scores, c=colors, s=30, zorder=3)
    
    # Threshold lines
    ax.axhline(70, color=COLORS["valid"], linestyle='--', alpha=0.5, label="VALID threshold")
    ax.axhline(30, color=COLORS["invalid"], linestyle='--', alpha=0.5, label="INVALID threshold")
    
    # Fill regions
    ax.fill_between(timestamps, 70, 100, alpha=0.1, color=COLORS["valid"])
    ax.fill_between(timestamps, 30, 70, alpha=0.1, color=COLORS["degraded"])
    ax.fill_between(timestamps, 0, 30, alpha=0.1, color=COLORS["invalid"])
    
    ax.set_ylim(0, 100)
    ax.set_xlabel("Time")
    ax.set_ylabel("Validity Score")
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower left', fontsize=8)
    
    plt.tight_layout()
    return fig
