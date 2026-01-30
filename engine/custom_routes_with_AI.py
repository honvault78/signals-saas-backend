"""
Custom Data Routes - Upload and analyze confidential time series

Integrates with existing Bavella engine for identical output to Public Equities.
Supports both dated and sequential (no-date) data formats.
"""

import os
import json
import logging
from typing import Optional, Dict
from io import BytesIO

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Header
from fastapi.responses import PlainTextResponse, HTMLResponse, Response

from .custom_data import (
    parse_uploaded_file,
    build_synthetic_series,
    get_analysis_metadata,
    generate_template_csv,
    ValidationResult,
    ParsedData,
    DateStatus,
)

# Import existing engine modules
from .regime import (
    detect_regime,
    generate_trading_signals,
    calculate_technical_indicators,
)
from .charts import create_all_charts
from .stats_analysis import calculate_memo_stats, calculate_enhanced_statistics
from .memo import generate_memo
from .report import generate_html_report

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

router = APIRouter()


# =============================================================================
# TEMPLATE ENDPOINT
# =============================================================================

@router.get("/template/csv")
async def download_template():
    """
    Download a CSV template for data upload.
    
    Template includes date column by default.
    Users can delete the date column if they prefer sequential-only data.
    """
    content = generate_template_csv()
    
    return PlainTextResponse(
        content=content,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=bavella_template.csv"}
    )


# =============================================================================
# VALIDATION ENDPOINT
# =============================================================================

@router.post("/validate/upload")
async def validate_upload(file: UploadFile = File(...)):
    """
    Validate uploaded file before running analysis.
    
    Returns validation status, detected series, and any warnings.
    """
    content = await file.read()
    
    result = parse_uploaded_file(content, file.filename or "upload.csv")
    
    if not result.valid:
        return {
            "valid": False,
            "errors": result.errors,
            "warnings": result.warnings,
            "data": None
        }
    
    parsed = result.parsed_data
    
    return {
        "valid": True,
        "errors": [],
        "warnings": result.warnings,
        "data": {
            "series": parsed.series_names,
            "row_count": parsed.row_count,
            "date_status": parsed.date_status.value,
            "date_range": parsed.date_range,
            "frequency": parsed.frequency,
        }
    }


# =============================================================================
# MAIN ANALYSIS ENDPOINT
# =============================================================================

@router.post("/analyze/custom", response_class=HTMLResponse)
async def analyze_custom(
    file: UploadFile = File(...),
    weights: str = Form(...),
    analysis_name: str = Form(default="Custom Analysis"),
    context: str = Form(default=""),
    include_ai_memo: bool = Form(default=True),
    x_openai_key: Optional[str] = Header(None, alias="X-OpenAI-Key"),
):
    """
    Run full analysis on uploaded custom data.
    
    Uses the same engine as Public Equities for identical output.
    
    Parameters:
    - file: CSV or Excel file with time series data
    - weights: JSON object mapping series names to weights (e.g., {"Series_A": 1.0, "Series_B": -0.5})
    - analysis_name: Name for the analysis report
    - context: Optional context about what the data represents
    - include_ai_memo: Whether to generate AI trading memo
    """
    # Parse weights
    try:
        weights_dict = json.loads(weights)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid weights JSON")
    
    # Read and validate file
    content = await file.read()
    result = parse_uploaded_file(content, file.filename or "upload.csv")
    
    if not result.valid:
        raise HTTPException(status_code=400, detail="; ".join(result.errors))
    
    parsed = result.parsed_data
    has_dates = parsed.date_status == DateStatus.DETECTED
    
    # Build synthetic returns series
    try:
        daily_returns = build_synthetic_series(parsed, weights_dict)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    if len(daily_returns) < 50:
        raise HTTPException(
            status_code=400, 
            detail=f"Insufficient data after processing: {len(daily_returns)} observations. Need at least 50."
        )
    
    # Build cumulative returns (starting at 1.0)
    cumulative = (1 + daily_returns).cumprod()
    
    logger.info(f"Running analysis on {len(daily_returns)} observations, has_dates={has_dates}")
    
    # ==========================================================================
    # RUN FULL ENGINE (same as Public Equities)
    # ==========================================================================
    
    try:
        # Step 1: Regime detection
        regime_df, regime_summary, indicators, z_score = detect_regime(
            cumulative=cumulative,
            daily_returns=daily_returns,
            window=min(60, len(daily_returns) - 20)  # Adjust window for shorter series
        )
        
        # Step 2: Generate trading signals
        signals = generate_trading_signals(regime_df, indicators, cumulative, z_score)
        
        # Step 3: Calculate statistics
        enhanced_stats = calculate_enhanced_statistics(daily_returns)
        memo_stats = calculate_memo_stats(daily_returns, cumulative)
        
        # Step 4: Generate charts
        charts = create_all_charts(
            cumulative=cumulative,
            daily_returns=daily_returns,
            regime_df=regime_df,
            indicators=indicators,
            z_score=z_score,
            signals=signals,
            portfolio_name=analysis_name
        )
        
    except Exception as e:
        logger.error(f"Analysis engine error: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    
    # ==========================================================================
    # ADJUST STATS FOR NO-DATE MODE
    # ==========================================================================
    
    if not has_dates:
        # Remove annualized metrics - they're meaningless without dates
        memo_stats_adjusted = memo_stats.copy()
        memo_stats_adjusted["annualized_return"] = None
        memo_stats_adjusted["annualized_volatility"] = None
        memo_stats_adjusted["sharpe_ratio"] = None
        memo_stats_adjusted["sortino_ratio"] = None
        memo_stats_adjusted["calmar_ratio"] = None
    else:
        memo_stats_adjusted = memo_stats
    
    # ==========================================================================
    # GENERATE AI MEMO (if requested)
    # ==========================================================================
    
    memo = None
    if include_ai_memo:
        openai_key = x_openai_key or os.environ.get("OPENAI_API_KEY")
        if openai_key:
            try:
                # Build position descriptions from weights
                long_positions = {k: v for k, v in weights_dict.items() if v > 0}
                short_positions = {k: abs(v) for k, v in weights_dict.items() if v < 0}
                
                # Normalize weights for display
                total_long = sum(long_positions.values()) if long_positions else 1
                total_short = sum(short_positions.values()) if short_positions else 1
                
                long_weights = {k: v/total_long for k, v in long_positions.items()}
                short_weights = {k: v/total_short for k, v in short_positions.items()}
                
                # Add context to portfolio name if provided
                memo_name = f"{analysis_name}"
                if context:
                    memo_name += f" ({context})"
                
                memo = await generate_memo(
                    enhanced_stats=memo_stats_adjusted,
                    regime_summary=regime_summary.to_dict(),
                    portfolio_name=memo_name,
                    long_positions=long_weights if long_weights else None,
                    short_positions=short_weights if short_weights else None,
                    api_key=openai_key,
                    model="gpt-4o"
                )
            except Exception as e:
                logger.warning(f"AI memo generation failed: {e}")
                memo = _generate_custom_fallback_memo(memo_stats_adjusted, regime_summary.to_dict(), has_dates)
        else:
            memo = _generate_custom_fallback_memo(memo_stats_adjusted, regime_summary.to_dict(), has_dates)
    
    # ==========================================================================
    # GENERATE HTML REPORT (same template as Public Equities)
    # ==========================================================================
    
    # Build position display
    long_positions_display = {k: v/sum(v for v in weights_dict.values() if v > 0) 
                             for k, v in weights_dict.items() if v > 0} if any(v > 0 for v in weights_dict.values()) else None
    short_positions_display = {k: abs(v)/sum(abs(v) for v in weights_dict.values() if v < 0) 
                              for k, v in weights_dict.items() if v < 0} if any(v < 0 for v in weights_dict.values()) else None
    
    # Use the existing report generator
    html_report = generate_html_report(
        enhanced_stats=memo_stats_adjusted,
        regime_summary=regime_summary.to_dict(),
        memo_text=memo,
        long_positions=long_positions_display,
        short_positions=short_positions_display,
        portfolio_name=analysis_name,
        regime_chart_base64=charts.get("regime"),
        performance_chart_base64=charts.get("performance"),
        distribution_chart_base64=charts.get("distribution"),
    )
    
    # Add confidentiality badge to report
    html_report = _add_confidentiality_badge(html_report, has_dates, parsed)
    
    return HTMLResponse(content=html_report)


# =============================================================================
# PDF EXPORT ENDPOINT
# =============================================================================

@router.post("/analyze/custom/pdf")
async def analyze_custom_pdf(
    file: UploadFile = File(...),
    weights: str = Form(...),
    analysis_name: str = Form(default="Custom Analysis"),
    context: str = Form(default=""),
    include_ai_memo: bool = Form(default=True),
    x_openai_key: Optional[str] = Header(None, alias="X-OpenAI-Key"),
):
    """
    Run analysis and return as PDF download.
    
    Uses weasyprint to convert HTML to PDF.
    """
    # First generate the HTML
    html_response = await analyze_custom(
        file=file,
        weights=weights,
        analysis_name=analysis_name,
        context=context,
        include_ai_memo=include_ai_memo,
        x_openai_key=x_openai_key,
    )
    
    html_content = html_response.body.decode('utf-8')
    
    # Convert to PDF
    try:
        from weasyprint import HTML
        
        pdf_bytes = HTML(string=html_content).write_pdf()
        
        # Sanitize filename
        safe_name = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in analysis_name)
        
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="{safe_name}_analysis.pdf"'
            }
        )
    except ImportError:
        raise HTTPException(
            status_code=501, 
            detail="PDF generation requires weasyprint. Install with: pip install weasyprint"
        )
    except Exception as e:
        logger.error(f"PDF generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(e)}")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _generate_custom_fallback_memo(stats: Dict, regime: Dict, has_dates: bool) -> str:
    """Generate a basic memo when AI is unavailable."""
    
    total_ret = stats.get('total_return', 0) or 0
    max_dd = stats.get('max_drawdown', 0) or 0
    current_regime = regime.get('current_regime', 'unknown')
    z_score = regime.get('z_score', 0)
    rsi = regime.get('rsi', 50)
    
    # Simple rule-based recommendation
    if total_ret > 0.02 and max_dd > -0.10:
        recommendation = "ENTER"
        reasoning = "Strategy is profitable with contained risk."
    elif total_ret > 0 and max_dd > -0.15:
        recommendation = "WAIT"
        reasoning = "Positive but monitor for better entry."
    else:
        recommendation = "PASS"
        reasoning = "Risk/reward not attractive at current levels."
    
    # Build memo
    memo = f"""## Trade Analysis Summary

**Current Regime:** {current_regime.upper()}
**Recommendation:** {recommendation}

### Key Metrics
- Total Return: {total_ret:.2%}
- Max Drawdown: {max_dd:.2%}"""
    
    if has_dates and stats.get('annualized_return') is not None:
        memo += f"""
- Annualized Return: {stats.get('annualized_return', 0):.2%}
- Sharpe Ratio: {stats.get('sharpe_ratio', 0):.2f}"""
    
    memo += f"""
- Z-Score: {z_score:+.2f}
- RSI: {rsi:.0f}

### Assessment
{reasoning}

### One-Line Summary
{recommendation}: {reasoning}

*Note: AI analysis unavailable. This is a simplified summary based on key metrics.*
"""
    return memo


def _add_confidentiality_badge(html: str, has_dates: bool, parsed: ParsedData) -> str:
    """Add confidentiality notice and data summary to the report."""
    
    # Build data summary
    if has_dates:
        date_info = f"{parsed.date_range[0]} to {parsed.date_range[1]} • {parsed.frequency.capitalize()} data"
    else:
        date_info = "Sequential data (no dates)"
    
    summary = f"{parsed.row_count} observations • {date_info}"
    
    # Fix the hardcoded "180-Day Analysis" subtitle
    html = html.replace(
        '180-Day Analysis with Regime Detection',
        f'{parsed.row_count} Observations • Regime Detection'
    )
    
    # Also fix "Performance Summary (180 Days)" section title
    html = html.replace(
        'Performance Summary (180 Days)',
        f'Performance Summary ({parsed.row_count} Observations)'
    )
    
    # Confidentiality badge HTML
    badge_html = f'''
    <div style="background: linear-gradient(135deg, #1a3a1a 0%, #0f2a0f 100%); 
                border: 1px solid #2d5a2d; 
                border-radius: 10px; 
                padding: 15px 20px; 
                margin-bottom: 20px;
                display: flex;
                align-items: center;
                gap: 12px;">
        <svg width="24" height="24" fill="none" stroke="#4ade80" stroke-width="2" viewBox="0 0 24 24">
            <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
            <path d="M9 12l2 2 4-4"/>
        </svg>
        <div>
            <div style="color: #4ade80; font-weight: 600; font-size: 14px;">✓ Confidential Analysis - Data Not Stored</div>
            <div style="color: #888; font-size: 12px; margin-top: 4px;">{summary}</div>
        </div>
    </div>
    <div style="background: rgba(79, 195, 247, 0.1); 
                border: 1px solid rgba(79, 195, 247, 0.3); 
                border-radius: 8px; 
                padding: 12px 16px; 
                margin-bottom: 20px;
                color: #4fc3f7;
                font-size: 13px;">
        <strong>ℹ</strong> Your data was processed in memory and has been deleted. Only the analysis results remain.
    </div>
    '''
    
    # Insert after the header section
    # Find the signal-container div and insert before it
    insert_point = html.find('<div class="signal-container">')
    if insert_point > 0:
        html = html[:insert_point] + badge_html + html[insert_point:]
    
    return html
