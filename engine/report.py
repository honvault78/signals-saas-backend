"""
Report Module - Professional HTML Investment Memo Report Generator

This module generates investment memo reports that EXACTLY match 
the Jupyter notebook's output with:
- Dark theme professional styling
- Signal badge parsing from AI memo
- Risk bar visualization
- Two-column layout for charts
- Proper metric formatting

Production-grade implementation.
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
) -> str:
    """
    Generate a professional HTML investment memo report with embedded charts.
    
    MATCHES THE JUPYTER NOTEBOOK OUTPUT EXACTLY.
    
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
    rsi_td_class = "value-positive" if rsi < 30 else "value-negative" if rsi > 70 else ""
    z_td_class = "value-positive" if z_score < -1 else "value-negative" if z_score > 1 else ""
    
    # Regime and risk
    regime_class = "regime-" + regime_summary.get('current_regime', 'unknown').lower().replace('_', '-')
    risk_level = regime_summary.get('risk_level', 'medium').lower()
    risk_width = "30%" if risk_level == "low" else "60%" if risk_level == "medium" else "90%"
    risk_class = "risk-" + risk_level
    
    # CSS - EXACT MATCH TO NOTEBOOK
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
        .footer { text-align: center; padding: 30px; color: #666; font-size: 12px; margin-top: 20px; }
        .risk-indicator { display: flex; align-items: center; gap: 10px; margin-top: 10px; }
        .risk-bar { flex: 1; height: 10px; background: #3a3a5a; border-radius: 5px; overflow: hidden; }
        .risk-fill { height: 100%; border-radius: 5px; }
        .risk-low { background: linear-gradient(90deg, #00b894, #00cec9); }
        .risk-medium { background: linear-gradient(90deg, #f39c12, #e67e22); }
        .risk-high { background: linear-gradient(90deg, #e74c3c, #c0392b); }
        .full-width { grid-column: 1 / -1; }
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
    html += '<div class="header-subtitle">180-Day Analysis with Regime Detection</div>'
    
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
    
    # Signal Badge
    html += f'<div class="signal-container"><div class="signal-badge {signal_class}">{signal_text}</div></div>'
    
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
    html += '<div class="section"><div class="section-title">Performance Summary (180 Days)</div>'
    html += '<div class="metrics-grid" style="grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));">'
    html += f'<div class="metric-card"><div class="metric-value {total_ret_class}">{enhanced_stats.get("total_return", 0):.2%}</div><div class="metric-label">Total Return</div></div>'
    html += f'<div class="metric-card"><div class="metric-value">{enhanced_stats.get("win_rate", 0):.0%}</div><div class="metric-label">Win Rate</div></div>'
    html += f'<div class="metric-card"><div class="metric-value positive">{enhanced_stats.get("best_day", 0):.2%}</div><div class="metric-label">Best Day</div></div>'
    html += '</div></div>'
    
    # Add memo if provided
    if memo_text:
        html += '<div class="section"><div class="section-title">AI Trade Analysis</div>'
        html += f'<div class="memo-content">{memo_text}</div></div>'
    
    # Footer
    html += '<div class="footer"><p>This report is for informational purposes only and does not constitute investment advice.</p>'
    html += '<p>Generated by Bavella Technologies Sarl</p></div>'
    html += '</div></body></html>'
    
    return html
