"""
Report Module - Professional HTML Investment Memo Report Generator

WITH BAVELLA v2.2 VALIDITY INTEGRATION

This module generates investment memo reports with:
- VALIDITY ANALYSIS AT THE TOP (governs everything)
- Dark theme professional styling
- Signal badge parsing from AI memo
- Risk bar visualization
- Two-column layout for charts
- Proper metric formatting

Production-grade implementation.
Copyright 2024-2026 Bavella Technologies Sarl
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def _parse_signal_from_memo(memo_text: str) -> tuple[str, str]:
    """
    Parse the trade signal from AI memo text.
    
    Looks for One-Line Summary section and extracts ENTER/EXIT/WAIT signal.
    Matches notebook logic exactly.
    
    Returns
    -------
    tuple[str, str]
        (signal_class, signal_text) e.g., ("signal-buy", "ENTER")
    """
    signal_class = "signal-hold"
    signal_text = "HOLD / MONITOR"
    
    if not memo_text:
        return signal_class, signal_text
    
    memo_lower = memo_text.lower()
    
    # Look for "### One-Line Summary" or "One-Line Summary" section
    one_line_start = -1
    for marker in ['### one-line summary', '### one‑line summary', 'one-line summary', 'one‑line summary']:
        pos = memo_lower.find(marker)
        if pos != -1:
            one_line_start = pos
            break
    
    if one_line_start != -1:
        # Extract text after the marker
        summary_text = memo_lower[one_line_start:one_line_start + 500]
        
        # Check for ENTER/BUY signals (GREEN)
        if any(word in summary_text for word in ['enter:', 'enter ', 'enter,', 'buy:', 'buy ', 'buy,']):
            signal_class = "signal-buy"
            signal_text = "ENTER"
        
        # Check for EXIT/SELL/REDUCE signals (RED)
        elif any(word in summary_text for word in ['exit:', 'exit ', 'sell:', 'sell ', 'reduce:', 'reduce ', 'close:']):
            signal_class = "signal-sell"
            signal_text = "EXIT"
        
        # Check for WAIT/HOLD signals (ORANGE)
        elif any(word in summary_text for word in ['wait:', 'wait ', 'wait,', 'hold:', 'hold ', 'hold,']):
            signal_class = "signal-hold"
            signal_text = "WAIT"
    
    # Fallback: if no one-line summary, check the recommendation section
    else:
        if 'recommendation: enter' in memo_lower or 'verdict: enter' in memo_lower:
            signal_class = "signal-buy"
            signal_text = "ENTER"
        elif 'recommendation: exit' in memo_lower or 'recommendation: sell' in memo_lower or 'verdict: exit' in memo_lower:
            signal_class = "signal-sell"
            signal_text = "EXIT"
        elif 'recommendation: wait' in memo_lower or 'verdict: wait' in memo_lower:
            signal_class = "signal-hold"
            signal_text = "WAIT"
    
    return signal_class, signal_text


def _generate_validity_section(validity_data: Optional[Dict[str, Any]]) -> str:
    """
    Generate the validity analysis HTML section.
    
    This goes at the TOP of the report - it governs everything else.
    """
    if not validity_data:
        return ""
    
    # Handle nested structure from to_dict() - validity data may be under "validity" key
    if "validity" in validity_data and isinstance(validity_data["validity"], dict):
        inner = validity_data["validity"]
        state = inner.get("state", "VALID")
        score = inner.get("score", 100)
        summary = inner.get("summary", "Statistical structure is stable")
        raw_confidence = inner.get("confidence", 0.8)
        root_cause = inner.get("root_cause")
    else:
        # Flat structure
        state = validity_data.get("state", "VALID")
        score = validity_data.get("score", 100)
        summary = validity_data.get("summary", "Statistical structure is stable")
        raw_confidence = validity_data.get("confidence", 0.8)
        root_cause = validity_data.get("root_cause")
    
    # Confidence - normalize to 0-1 range
    if raw_confidence > 1:
        confidence = raw_confidence / 100.0
    else:
        confidence = raw_confidence
    confidence = max(0.0, min(1.0, confidence))
    confidence_pct = int(confidence * 100)
    confidence_label = "High" if confidence >= 0.8 else "Medium" if confidence >= 0.6 else "Low"
    
    # Root cause
    root_cause_html = ""
    if root_cause:
        root_cause_html = f"""
        <div style="margin-top: 20px;">
            <div style="color: #888; font-size: 11px; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px;">Root Cause</div>
            <div style="display: flex; align-items: center; gap: 12px;">
                <span style="background: rgba(231, 76, 60, 0.2); border: 1px solid rgba(231, 76, 60, 0.4); color: #e74c3c; padding: 4px 10px; border-radius: 6px; font-size: 12px; font-weight: 700;">{root_cause.get('code', 'FM?')}</span>
                <span style="color: #f39c12; font-weight: 600;">{root_cause.get('label', 'Unknown')}</span>
                <span style="background: rgba(231, 76, 60, 0.15); color: #e74c3c; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600;">PRIMARY</span>
                <span style="color: #e74c3c; margin-left: auto; font-weight: 600;">{root_cause.get('severity', 0)} severity</span>
            </div>
        </div>
        """
    
    # Secondary failures
    secondary_failures = validity_data.get("details", {}).get("secondary_failures", [])
    if not secondary_failures:
        secondary_failures = validity_data.get("secondary_failures", [])
    
    secondary_html = ""
    if secondary_failures:
        secondary_items = ""
        for sf in secondary_failures:
            secondary_items += f"""
            <div style="display: flex; align-items: center; gap: 12px; padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.05);">
                <span style="background: rgba(136, 136, 136, 0.2); border: 1px solid rgba(136, 136, 136, 0.3); color: #888; padding: 3px 8px; border-radius: 4px; font-size: 11px; font-weight: 600;">{sf.get('code', 'FM?')}</span>
                <span style="color: #aaa;">{sf.get('label', 'Unknown')}</span>
                <span style="color: #888; margin-left: auto;">{sf.get('severity', 0)}</span>
            </div>
            """
        secondary_html = f"""
        <div style="margin-top: 16px;">
            <div style="color: #666; font-size: 11px; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px;">Secondary Symptoms</div>
            {secondary_items}
        </div>
        """
    
    # Assessment note
    assessment_note = validity_data.get("details", {}).get("assessment_note", "")
    if not assessment_note:
        assessment_note = validity_data.get("assessment_note", "")
    
    assessment_html = ""
    if assessment_note:
        assessment_html = f"""
        <div style="margin-top: 16px; padding: 12px; background: rgba(0,0,0,0.2); border-radius: 8px; font-style: italic; color: #888; font-size: 13px;">
            "{assessment_note}"
        </div>
        """
    
    # Historical context
    history = validity_data.get("history", {})
    similar_episodes = history.get("similar_episodes", validity_data.get("similar_episodes", 0))
    median_recovery = history.get("median_recovery_days", validity_data.get("median_recovery_days"))
    
    history_html = ""
    if similar_episodes and similar_episodes > 0:
        recovery_text = f", typical recovery: {median_recovery} days" if median_recovery else ""
        history_html = f"""
        <div style="margin-top: 16px; padding: 12px; background: rgba(79, 195, 247, 0.1); border: 1px solid rgba(79, 195, 247, 0.2); border-radius: 8px;">
            <div style="color: #4fc3f7; font-size: 13px;">
                <strong>Historical Pattern:</strong> {similar_episodes} similar episodes found{recovery_text}
            </div>
        </div>
        """
    
    # NEW: Competing explanations panel (makes multi-detector visible)
    attribution = validity_data.get("attribution", {})
    competing_causes = attribution.get("competing_causes", validity_data.get("competing_causes", []))
    counterfactuals = attribution.get("counterfactuals", validity_data.get("counterfactuals", []))
    
    competing_html = ""
    if competing_causes and len(competing_causes) > 0:
        # Check if all scores are zero (clean bill of health)
        all_clear = all(cause.get("score", 0) == 0 for cause in competing_causes)
        
        # Build bar chart for competing causes
        causes_items = ""
        for cause in competing_causes[:5]:  # Show top 5
            cause_score = cause.get("score", 0)
            bar_width = max(2, int(cause_score * 100))  # Min 2px for visibility when zero
            code = cause.get("code", "?")
            label = cause.get("label", "Unknown")
            evidence = cause.get("evidence", "")
            
            # Color based on score
            if cause_score >= 0.5:
                bar_color = "#e74c3c"
                text_color = "#e74c3c"
            elif cause_score >= 0.3:
                bar_color = "#f39c12"
                text_color = "#f39c12"
            elif cause_score > 0:
                bar_color = "#888"
                text_color = "#888"
            else:
                # Zero score - show as green checkmark
                bar_color = "#00b894"
                text_color = "#00b894"
            
            # For zero scores, show checkmark
            score_display = f"✓" if cause_score == 0 else f"{cause_score:.2f}"
            
            causes_items += f"""
            <div style="margin-bottom: 12px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <span style="color: {text_color}; font-size: 11px; font-weight: 700; width: 35px;">{code}</span>
                        <span style="color: #ccc; font-size: 13px;">{label}</span>
                    </div>
                    <span style="color: {text_color}; font-weight: 600; font-size: 14px;">{score_display}</span>
                </div>
                <div style="background: rgba(255,255,255,0.1); border-radius: 4px; height: 8px; overflow: hidden;">
                    <div style="background: {bar_color}; height: 100%; width: {bar_width}%; border-radius: 4px; transition: width 0.3s;"></div>
                </div>
                <div style="color: #666; font-size: 11px; margin-top: 2px;">{evidence}</div>
            </div>
            """
        
        # Different header for all-clear vs issues
        section_title = "Invariants Checked" if all_clear else "Competing Explanations"
        footer_text = "All monitored invariants within normal bounds" if all_clear else "Root cause = highest scoring explanation given evidence consistency"
        
        competing_html = f"""
        <div style="margin-top: 20px; padding-top: 16px; border-top: 1px solid rgba(255,255,255,0.1);">
            <div style="color: #888; font-size: 11px; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 12px;">{section_title}</div>
            {causes_items}
            <div style="color: #555; font-size: 10px; margin-top: 8px; font-style: italic;">
                {footer_text}
            </div>
        </div>
        """
    
    # Counterfactual checks
    counterfactual_html = ""
    if counterfactuals and len(counterfactuals) > 0:
        cf_items = ""
        for cf in counterfactuals[:3]:  # Show top 3
            test = cf.get("test", "")
            result = cf.get("result", "")
            changes = cf.get("changes_conclusion", False)
            
            icon = "→" if changes else "—"
            icon_color = "#f39c12" if changes else "#666"
            
            cf_items += f"""
            <div style="display: flex; gap: 8px; padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.05);">
                <span style="color: {icon_color}; font-size: 14px;">{icon}</span>
                <div>
                    <div style="color: #aaa; font-size: 12px;">{test}</div>
                    <div style="color: #888; font-size: 11px;">{result}</div>
                </div>
            </div>
            """
        
        counterfactual_html = f"""
        <div style="margin-top: 16px;">
            <div style="color: #666; font-size: 11px; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px;">Robustness Checks</div>
            {cf_items}
        </div>
        """
    
    # State-based styling
    if state == "VALID":
        state_color = "#00b894"
        state_bg = "rgba(0, 184, 148, 0.15)"
        state_border = "rgba(0, 184, 148, 0.3)"
        state_icon = "✓"
    elif state == "DEGRADED":
        state_color = "#f39c12"
        state_bg = "rgba(243, 156, 18, 0.15)"
        state_border = "rgba(243, 156, 18, 0.3)"
        state_icon = "⚠"
    else:  # BROKEN
        state_color = "#e74c3c"
        state_bg = "rgba(231, 76, 60, 0.15)"
        state_border = "rgba(231, 76, 60, 0.3)"
        state_icon = "✕"
    
    html = f"""
    <div style="background: linear-gradient(135deg, #1a2a3a 0%, #1a1a2e 100%); border-radius: 15px; padding: 25px; margin-bottom: 25px; border: 2px solid {state_border}; box-shadow: 0 10px 30px rgba(0,0,0,0.3);">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 20px;">
            <!-- Left: State and Score -->
            <div style="display: flex; align-items: center; gap: 20px;">
                <div style="background: {state_bg}; border: 2px solid {state_border}; border-radius: 12px; padding: 16px 24px; text-align: center; min-width: 100px;">
                    <div style="color: {state_color}; font-size: 14px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px;">{state}</div>
                    <div style="display: flex; align-items: center; justify-content: center; gap: 6px; margin-top: 4px;">
                        <span style="color: {state_color}; font-size: 12px;">{state_icon}</span>
                        <span style="color: {state_color}; font-size: 28px; font-weight: 700;">{score}</span>
                    </div>
                </div>
                <div>
                    <div style="color: #fff; font-size: 18px; font-weight: 600;">{summary}</div>
                    <div style="color: #888; font-size: 13px; margin-top: 6px;">
                        Confidence: <span style="color: #fff; font-weight: 500;">{confidence_label}</span> ({confidence_pct}%)
                    </div>
                </div>
            </div>
        </div>
        
        {root_cause_html}
        {secondary_html}
        {competing_html}
        {counterfactual_html}
        {assessment_html}
        {history_html}
        
        <div style="margin-top: 20px; padding-top: 16px; border-top: 1px solid rgba(255,255,255,0.1); color: #666; font-size: 11px;">
            <div>Assessment based on multi-detector invariant monitoring with attribution.</div>
            <div style="margin-top: 4px;">Scope: structural integrity of the relationship (not strategy suitability). Bavella v2.2</div>
        </div>
    </div>
    """
    
    return html


def generate_html_report(
    enhanced_stats: Dict[str, Any],
    regime_summary: Dict[str, Any],
    memo_text: Optional[str] = None,
    long_positions: Optional[Dict[str, float]] = None,
    short_positions: Optional[Dict[str, float]] = None,
    portfolio_name: str = "Long/Short Equity Portfolio",
    regime_chart_base64: Optional[str] = None,
    performance_chart_base64: Optional[str] = None,
    distribution_chart_base64: Optional[str] = None,
    validity_data: Optional[Dict[str, Any]] = None,
    analysis_period_days: int = 180,
) -> str:
    """
    Generate a professional HTML investment memo report with embedded charts.
    
    NOW WITH BAVELLA v2.2 VALIDITY ANALYSIS AT THE TOP.
    
    Parameters
    ----------
    enhanced_stats : Dict[str, Any]
        Enhanced statistics dictionary
    regime_summary : Dict[str, Any]
        Regime summary dictionary
    memo_text : Optional[str]
        AI-generated memo text
    long_positions : Optional[Dict[str, float]]
        Long position weights
    short_positions : Optional[Dict[str, float]]
        Short position weights
    portfolio_name : str
        Portfolio name for title
    regime_chart_base64 : Optional[str]
        Base64-encoded regime chart PNG
    performance_chart_base64 : Optional[str]
        Base64-encoded performance chart PNG
    distribution_chart_base64 : Optional[str]
        Base64-encoded distribution chart PNG
    validity_data : Optional[Dict[str, Any]]
        Bavella v2.2 validity analysis results
    analysis_period_days : int
        Actual analysis period for display
        
    Returns
    -------
    str
        Complete HTML document
    """
    report_date = datetime.now().strftime('%B %d, %Y')
    
    # Parse signal from memo
    signal_class, signal_text = _parse_signal_from_memo(memo_text)
    
    # Extract values for display
    rsi = regime_summary.get('rsi', 50)
    z_score = regime_summary.get('z_score', 0)
    regime = regime_summary.get('current_regime', 'unknown').upper()
    
    # Determine value classes
    ret_class = "positive" if enhanced_stats.get("annualized_return", 0) > 0 else "negative"
    rsi_class = "positive" if rsi < 30 else "negative" if rsi > 70 else "neutral"
    z_class = "positive" if z_score < -1 else "negative" if z_score > 1 else "neutral"
    skew_class = "value-positive" if enhanced_stats.get("skewness", 0) > 0 else "value-negative"
    total_ret_class = "positive" if enhanced_stats.get("total_return", 0) > 0 else "negative"
    rsi_text = "(Oversold)" if rsi < 30 else "(Overbought)" if rsi > 70 else ""
    
    # Regime badge class
    regime_lower = regime.lower().replace(' ', '-').replace('_', '-')
    regime_class = f"regime-{regime_lower}"
    
    # Risk level
    risk_level = regime_summary.get('risk_level', 'medium').lower()
    if risk_level == 'low':
        risk_class = 'risk-low'
        risk_width = '33%'
    elif risk_level == 'high':
        risk_class = 'risk-high'
        risk_width = '100%'
    else:
        risk_class = 'risk-medium'
        risk_width = '66%'
    
    # RSI/Z table classes
    rsi_td_class = "value-positive" if rsi < 30 else "value-negative" if rsi > 70 else ""
    z_td_class = "value-positive" if z_score < -1 else "value-negative" if z_score > 1 else ""
    
    # CSS
    css = """
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); color: #e0e0e0; line-height: 1.6; min-height: 100vh; }
        .container { max-width: 1400px; margin: 0 auto; padding: 20px; }
        .header { background: linear-gradient(135deg, #0f3460 0%, #1a1a2e 100%); border-radius: 15px; padding: 30px; margin-bottom: 25px; border: 1px solid #3a3a5a; box-shadow: 0 10px 30px rgba(0,0,0,0.3); }
        .header-top { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 20px; }
        .logo { font-size: 14px; color: #888; text-transform: uppercase; letter-spacing: 2px; }
        .report-date { color: #888; font-size: 14px; }
        .header h1 { font-size: 28px; color: #fff; margin-bottom: 10px; font-weight: 600; }
        .header-subtitle { color: #aaa; font-size: 16px; }
        .signal-container { display: flex; justify-content: center; margin: 25px 0; }
        .signal-badge { padding: 15px 40px; border-radius: 50px; font-size: 20px; font-weight: 700; letter-spacing: 2px; text-transform: uppercase; }
        .signal-buy { background: linear-gradient(135deg, #00b894 0%, #00cec9 100%); color: #fff; box-shadow: 0 5px 20px rgba(0, 184, 148, 0.4); }
        .signal-sell { background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%); color: #fff; box-shadow: 0 5px 20px rgba(231, 76, 60, 0.4); }
        .signal-hold { background: linear-gradient(135deg, #f39c12 0%, #e67e22 100%); color: #fff; box-shadow: 0 5px 20px rgba(243, 156, 18, 0.4); }
        .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 15px; margin-bottom: 25px; }
        .metric-card { background: linear-gradient(135deg, #1e3a5f 0%, #1a1a2e 100%); border-radius: 12px; padding: 20px; text-align: center; border: 1px solid #3a3a5a; transition: transform 0.2s, box-shadow 0.2s; }
        .metric-card:hover { transform: translateY(-3px); box-shadow: 0 8px 25px rgba(0,0,0,0.3); }
        .metric-value { font-size: 28px; font-weight: 700; color: #4fc3f7; margin-bottom: 5px; }
        .metric-value.positive { color: #00b894; }
        .metric-value.negative { color: #e74c3c; }
        .metric-value.neutral { color: #f39c12; }
        .metric-label { font-size: 12px; color: #888; text-transform: uppercase; letter-spacing: 1px; }
        .section { background: linear-gradient(135deg, #1e2a3a 0%, #1a1a2e 100%); border-radius: 15px; padding: 25px; margin-bottom: 25px; border: 1px solid #3a3a5a; }
        .section-title { font-size: 18px; color: #4fc3f7; margin-bottom: 20px; padding-bottom: 10px; border-bottom: 2px solid #3a3a5a; text-transform: uppercase; letter-spacing: 1px; }
        .data-table { width: 100%; border-collapse: collapse; margin-top: 15px; }
        .data-table th { background: #0f3460; color: #4fc3f7; padding: 12px 15px; text-align: left; font-size: 12px; text-transform: uppercase; letter-spacing: 1px; }
        .data-table td { padding: 12px 15px; border-bottom: 1px solid #3a3a5a; font-size: 14px; }
        .data-table tr:hover { background: rgba(79, 195, 247, 0.1); }
        .value-positive { color: #00b894; font-weight: 600; }
        .value-negative { color: #e74c3c; font-weight: 600; }
        .chart-container { background: linear-gradient(135deg, #e8f4fc 0%, #d6eaf8 100%); border-radius: 12px; padding: 15px; margin-top: 20px; box-shadow: 0 5px 20px rgba(0,0,0,0.2); }
        .chart-container img { width: 100%; height: auto; border-radius: 8px; }
        .chart-title { color: #1a5276; font-size: 14px; font-weight: 600; margin-bottom: 10px; text-transform: uppercase; letter-spacing: 1px; }
        .two-column { display: grid; grid-template-columns: 1fr 1fr; gap: 25px; }
        .three-column { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; }
        @media (max-width: 1200px) { .three-column { grid-template-columns: 1fr 1fr; } }
        @media (max-width: 900px) { .two-column, .three-column { grid-template-columns: 1fr; } }
        .memo-content { background: #1a1a2e; border-radius: 10px; padding: 25px; border-left: 4px solid #4fc3f7; white-space: pre-wrap; font-size: 14px; line-height: 1.8; }
        .regime-badge { display: inline-block; padding: 8px 20px; border-radius: 25px; font-size: 14px; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; }
        .regime-distribution { background: #e67e22; color: #fff; }
        .regime-accumulation { background: #3498db; color: #fff; }
        .regime-ranging { background: #2ecc71; color: #fff; }
        .regime-trending-up { background: #27ae60; color: #fff; }
        .regime-trending-down { background: #c0392b; color: #fff; }
        .regime-breakdown { background: #e74c3c; color: #fff; }
        .regime-breakout { background: #1abc9c; color: #fff; }
        .regime-volatile-expansion { background: #9b59b6; color: #fff; }
        .regime-uncertain { background: #7f8c8d; color: #fff; }
        .regime-pure-trend { background: #27ae60; color: #fff; }
        .regime-trend-following { background: #2ecc71; color: #fff; }
        .regime-mixed-mode { background: #f39c12; color: #fff; }
        .regime-mean-reversion { background: #3498db; color: #fff; }
        .regime-strong-mean-reversion { background: #2980b9; color: #fff; }
        .footer { text-align: center; padding: 30px; color: #666; font-size: 12px; margin-top: 20px; }
        .risk-indicator { display: flex; align-items: center; gap: 10px; margin-top: 10px; }
        .risk-bar { flex: 1; height: 10px; background: #3a3a5a; border-radius: 5px; overflow: hidden; }
        .risk-fill { height: 100%; border-radius: 5px; }
        .risk-low { background: linear-gradient(90deg, #00b894, #00cec9); }
        .risk-medium { background: linear-gradient(90deg, #f39c12, #e67e22); }
        .risk-high { background: linear-gradient(90deg, #e74c3c, #c0392b); }
        .full-width { grid-column: 1 / -1; }
        .conditional-warning { background: rgba(243, 156, 18, 0.1); border: 1px solid rgba(243, 156, 18, 0.3); border-radius: 8px; padding: 12px 16px; margin-bottom: 20px; color: #f39c12; font-size: 13px; }
    """
    
    # Build HTML
    html = '<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">'
    html += f'<title>Investment Memo - {portfolio_name}</title>'
    html += f'<style>{css}</style></head><body><div class="container">'
    
    # Header
    html += '<div class="header"><div class="header-top">'
    html += '<div class="logo">Tactical Trade Analysis</div>'
    html += f'<div class="report-date">{report_date}</div></div>'
    html += f'<h1>{portfolio_name}</h1>'
    html += f'<div class="header-subtitle">{analysis_period_days}-Day Analysis with Regime Detection</div>'
    
    # Portfolio composition
    if long_positions or short_positions:
        html += '<div style="margin-top: 20px; padding-top: 15px; border-top: 1px solid #3a3a5a;">'
        if long_positions:
            long_str = ", ".join([f"{t} ({w:.0%})" for t, w in long_positions.items()])
            html += f'<div style="color: #00b894; font-size: 13px; margin-bottom: 5px;"><strong>LONG:</strong> {long_str}</div>'
        if short_positions:
            short_str = ", ".join([f"{t} ({w:.0%})" for t, w in short_positions.items()])
            html += f'<div style="color: #e74c3c; font-size: 13px;"><strong>SHORT:</strong> {short_str}</div>'
        html += '</div>'
    
    html += '</div>'
    
    # ==========================================================================
    # VALIDITY ANALYSIS SECTION (GOVERNS EVERYTHING - AT THE TOP)
    # ==========================================================================
    validity_html = _generate_validity_section(validity_data)
    if validity_html:
        html += '<div class="section" style="border: none; padding: 0; background: transparent;">'
        html += '<div class="section-title" style="border-bottom: none; margin-bottom: 15px;">Structural Validity Assessment</div>'
        html += validity_html
        html += '</div>'
    
    # Conditional warning if degraded or broken
    if validity_data:
        state = validity_data.get("state", "VALID")
        if state == "DEGRADED":
            html += '<div class="conditional-warning">⚠️ <strong>Conditional Analysis:</strong> The metrics and signals below should be interpreted with caution due to detected structural instability.</div>'
        elif state == "BROKEN":
            html += '<div class="conditional-warning" style="background: rgba(231, 76, 60, 0.1); border-color: rgba(231, 76, 60, 0.3); color: #e74c3c;">⛔ <strong>Historical Reference Only:</strong> The analysis below describes past behavior that may no longer apply under current conditions.</div>'
    
    # Key Metrics
    html += '<div class="metrics-grid">'
    html += f'<div class="metric-card"><div class="metric-value {ret_class}">{enhanced_stats.get("annualized_return", 0):.1%}</div><div class="metric-label">Ann. Return</div></div>'
    html += f'<div class="metric-card"><div class="metric-value">{enhanced_stats.get("annualized_volatility", 0):.1%}</div><div class="metric-label">Volatility</div></div>'
    html += f'<div class="metric-card"><div class="metric-value">{enhanced_stats.get("win_rate", 0):.0%}</div><div class="metric-label">Win Rate</div></div>'
    html += f'<div class="metric-card"><div class="metric-value negative">{enhanced_stats.get("max_drawdown", 0):.1%}</div><div class="metric-label">Max Drawdown</div></div>'
    html += f'<div class="metric-card"><div class="metric-value {rsi_class}">{rsi:.0f}</div><div class="metric-label">RSI (14)</div></div>'
    html += f'<div class="metric-card"><div class="metric-value {z_class}">{z_score:+.2f}σ</div><div class="metric-label">Z-Score</div></div>'
    html += '</div>'
    
    # Current Regime
    html += '<div class="section"><div class="section-title">Current Market Regime</div>'
    html += '<div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 20px;"><div>'
    html += f'<span class="regime-badge {regime_class}">{regime}</span>'
    html += f'<p style="margin-top: 15px; color: #aaa;">{regime_summary.get("strategy", "")}</p></div>'
    html += f'<div style="min-width: 250px;"><div style="color: #888; font-size: 12px; margin-bottom: 5px;">RISK LEVEL: {risk_level.upper()}</div>'
    html += f'<div class="risk-indicator"><div class="risk-bar"><div class="risk-fill {risk_class}" style="width: {risk_width};"></div></div></div></div></div></div>'
    
    # Charts Section - Regime Analysis (full width)
    if regime_chart_base64:
        html += '<div class="section"><div class="section-title">Technical Analysis & Regime Detection</div>'
        html += f'<div class="chart-container"><img src="data:image/png;base64,{regime_chart_base64}" alt="Regime Analysis Chart"></div></div>'
    
    # Two charts side by side
    if performance_chart_base64 or distribution_chart_base64:
        html += '<div class="two-column">'
        
        if performance_chart_base64:
            html += '<div class="section"><div class="section-title">Performance & Drawdown</div>'
            html += f'<div class="chart-container"><img src="data:image/png;base64,{performance_chart_base64}" alt="Performance Chart"></div></div>'
        
        if distribution_chart_base64:
            html += '<div class="section"><div class="section-title">Returns Distribution</div>'
            html += f'<div class="chart-container"><img src="data:image/png;base64,{distribution_chart_base64}" alt="Distribution Chart"></div></div>'
        
        html += '</div>'
    
    # Two column tables
    html += '<div class="two-column">'
    
    # Risk Metrics Table
    html += '<div class="section"><div class="section-title">Risk Metrics</div><table class="data-table">'
    html += f'<tr><td>Max Drawdown</td><td class="value-negative">{enhanced_stats.get("max_drawdown", 0):.2%}</td></tr>'
    html += f'<tr><td>Worst Day</td><td class="value-negative">{enhanced_stats.get("worst_day", 0):.2%}</td></tr>'
    html += f'<tr><td>VaR (95%)</td><td class="value-negative">{enhanced_stats.get("var_95", 0):.2%}</td></tr>'
    html += f'<tr><td>CVaR (95%)</td><td class="value-negative">{enhanced_stats.get("cvar_95", 0):.2%}</td></tr>'
    html += f'<tr><td>Skewness</td><td class="{skew_class}">{enhanced_stats.get("skewness", 0):.3f}</td></tr>'
    html += f'<tr><td>Kurtosis</td><td>{enhanced_stats.get("kurtosis", 0):.3f}</td></tr>'
    html += '</table></div>'
    
    # Entry Point Table
    html += '<div class="section"><div class="section-title">Entry Point Analysis</div><table class="data-table">'
    html += f'<tr><td>RSI (14)</td><td class="{rsi_td_class}">{rsi:.0f} {rsi_text}</td></tr>'
    html += f'<tr><td>Z-Score</td><td class="{z_td_class}">{z_score:+.2f}σ</td></tr>'
    html += f'<tr><td>ADF p-value</td><td>{regime_summary.get("adf_pvalue", 0):.4f}</td></tr>'
    html += f'<tr><td>Hurst Exponent</td><td>{regime_summary.get("hurst_exponent", regime_summary.get("hurst", 0)):.4f}</td></tr>'
    html += f'<tr><td>Half-Life</td><td>{regime_summary.get("halflife", 0):.1f} days</td></tr>'
    html += f'<tr><td>Confidence</td><td>{regime_summary.get("confidence", 0):.0f}%</td></tr>'
    html += '</table></div>'
    html += '</div>'
    
    # Performance Summary
    html += f'<div class="section"><div class="section-title">Performance Summary ({analysis_period_days} Days)</div>'
    html += '<div class="metrics-grid" style="grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));">'
    html += f'<div class="metric-card"><div class="metric-value {total_ret_class}">{enhanced_stats.get("total_return", 0):.2%}</div><div class="metric-label">Total Return</div></div>'
    html += f'<div class="metric-card"><div class="metric-value">{enhanced_stats.get("win_rate", 0):.0%}</div><div class="metric-label">Win Rate</div></div>'
    html += f'<div class="metric-card"><div class="metric-value positive">{enhanced_stats.get("best_day", 0):.2%}</div><div class="metric-label">Best Day</div></div>'
    html += '</div></div>'
    
    # Add memo if provided
    if memo_text:
        html += '<div class="section"><div class="section-title">AI Trade Analysis</div>'
        html += f'<div class="memo-content">{memo_text}</div></div>'
    
    # Disclaimer
    html += '''
    <div class="section" style="margin-top: 40px; background: rgba(15, 15, 26, 0.8); border: 1px solid rgba(255,255,255,0.1);">
        <div class="section-title" style="color: #888; font-size: 14px;">Disclaimer</div>
        <div style="color: #666; font-size: 11px; line-height: 1.7;">
            <p style="margin-bottom: 10px;">Bavella provides analytical tools designed to assess the structural validity of statistical relationships, models, and analytical assumptions. Bavella does not provide investment advice, investment recommendations, research, portfolio management services, or execution services.</p>
            <p style="margin-bottom: 10px;">Any information, analysis, or commentary generated by Bavella is of a general and informational nature and does not constitute a personal recommendation as defined under the EU Markets in Financial Instruments Directive (Directive 2014/65/EU) ("MiFID II"). Outputs are not tailored to the individual circumstances, objectives, or financial situation of any user.</p>
            <p style="margin-bottom: 10px;">Bavella does not express discretionary opinions, forecasts, or views on the future performance of any financial instrument. All assessments are based on predefined, auditable analytical logic applied to historical and current data.</p>
            <p style="margin-bottom: 10px;">Users remain solely responsible for their investment decisions and for assessing the suitability of any strategy, instrument, or position. Users should seek independent professional advice before making any investment decision.</p>
            <p style="margin-bottom: 10px;">Past performance, statistical properties, or historical relationships are not indicative of future results. Financial instruments involve risk, including the possible loss of capital.</p>
            <p>Information may be derived from third-party data sources believed to be reliable; however, Bavella does not guarantee the accuracy, completeness, or timeliness of such data. All outputs are provided "as is" without warranty of any kind.</p>
        </div>
    </div>
    '''
    
    # Footer
    html += '<div class="footer"><p>© 2026 Bavella Technologies Sarl • Validity Engine v2.2</p></div>'
    html += '</div></body></html>'
    
    return html
