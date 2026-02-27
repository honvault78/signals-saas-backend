"""
Validity-Governed Memo Generator
================================

Generates professional tactical trade memos using OpenAI GPT-4.
The memo is GOVERNED by validity state — the epistemic status of the
analysis is reflected in the memo's tone, structure, and recommendations.

Three modes:
- VALID: Confident, actionable recommendations
- DEGRADED: Cautious, conditional recommendations with explicit warnings
- BROKEN: Historical framing — analysis describes past behavior, not current

Key principle: The memo is no longer a conclusion. It is a contextualized interpretation.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


# Try to load dotenv for local development
def _load_env():
    """Load environment variables from .env file if available."""
    try:
        from dotenv import load_dotenv
        env_paths = [Path(".env"), Path("../.env"), Path(__file__).parent / ".env"]
        for env_path in env_paths:
            if env_path.exists():
                load_dotenv(env_path)
                return True
        load_dotenv()
        return True
    except ImportError:
        return False

_load_env()


# =============================================================================
# VALIDITY-GOVERNED SYSTEM PROMPTS
# =============================================================================

SYSTEM_PROMPT_VALID = """You are a senior trader evaluating tactical trades. The structural integrity of this analysis has been verified as STABLE. You can be confident in your assessment, though always acknowledge risk factors."""

SYSTEM_PROMPT_DEGRADED = """You are a senior trader evaluating tactical trades. IMPORTANT: The structural integrity of this analysis has been flagged as DEGRADED. A validity issue has been detected. You must lead with the structural assessment and frame all conclusions as CONDITIONAL on structural recovery."""

SYSTEM_PROMPT_BROKEN = """You are a senior trader evaluating tactical trades. CRITICAL: The structural integrity of this analysis has FAILED. The statistical structure is no longer reliable. You must clearly state that this analysis describes HISTORICAL behavior that may no longer apply. Any positioning should be considered historical in nature."""


# =============================================================================
# VALIDITY-GOVERNED PROMPT BUILDER
# =============================================================================

def build_validity_governed_prompt(
    enhanced_stats: Dict,
    regime_summary: Dict,
    validity_data: Dict,
    portfolio_name: str = "Long/Short Portfolio",
    long_positions: Optional[Dict[str, float]] = None,
    short_positions: Optional[Dict[str, float]] = None
) -> tuple[str, str]:
    """
    Build validity-governed prompt for memo generation.
    
    Returns (system_prompt, user_prompt) tuple.
    The prompt structure changes based on validity state.
    """
    # Extract validity info
    validity = validity_data.get("validity", {})
    history = validity_data.get("history", {})
    details = validity_data.get("details", {})
    
    state = validity.get("state", "VALID")
    score = validity.get("score", 100)
    summary = validity.get("summary", "Statistical structure is stable")
    root_cause = validity.get("root_cause")
    confidence = validity.get("confidence", 0.8)
    
    similar_episodes = history.get("similar_episodes", 0)
    median_recovery = history.get("median_recovery_days")
    
    # Select system prompt based on state
    if state == "VALID":
        system_prompt = SYSTEM_PROMPT_VALID
    elif state == "DEGRADED":
        system_prompt = SYSTEM_PROMPT_DEGRADED
    else:
        system_prompt = SYSTEM_PROMPT_BROKEN
    
    # Portfolio composition
    portfolio_comp = ""
    if long_positions:
        portfolio_comp += "LONG: " + ", ".join([f"{t} ({w:.0%})" for t, w in long_positions.items()]) + "\n"
    if short_positions:
        portfolio_comp += "SHORT: " + ", ".join([f"{t} ({w:.0%})" for t, w in short_positions.items()])
    
    # Build the prompt based on validity state
    prompt = f"""
## Portfolio: {portfolio_name}
{portfolio_comp if portfolio_comp else "Long/Short Equity Spread"}

---

## REQUIRED MEMO STRUCTURE (Follow this order exactly)

### 1. Structural Assessment (MANDATORY FIRST SECTION)
"""

    # Structural assessment content varies by state
    if state == "VALID":
        prompt += f"""
Write exactly:
"**Structural assessment: Stable.** The statistical structure underlying this analysis remains intact."

Then optionally add one sentence about detection confidence ({confidence:.0%}).
"""
    elif state == "DEGRADED":
        prompt += f"""
Write exactly:
"**Structural assessment: Degraded.** {summary} Confidence: {confidence:.0%}."

Then add: "**The following analysis should be interpreted with caution.**"
"""
        if root_cause:
            prompt += f"""
Mention the root cause: {root_cause['code']} — {root_cause['label']} (severity {root_cause['severity']})
"""
        if similar_episodes > 0 and median_recovery:
            prompt += f"""
Add: "Similar structural conditions were identified on {similar_episodes} prior occasion(s). Median recovery time: approximately {median_recovery} days."
"""
    else:  # BROKEN
        prompt += f"""
Write exactly:
"**Structural assessment: Failed.** The statistical structure underlying this analysis is no longer reliable."

Then add: "**This memo describes historical behavior that may no longer apply under current conditions.**"
"""
        if root_cause:
            prompt += f"""
Mention: "A severe {root_cause['label'].lower()} has been identified as the root cause of failure."
"""

    # Historical context section
    prompt += """
### 2. Historical Context
"""
    if state == "VALID":
        prompt += """
Write: "No prior structural breakdowns with comparable characteristics were identified in the historical record."
"""
    elif similar_episodes > 0:
        prompt += f"""
Write about the {similar_episodes} similar episode(s) found. Median recovery was {median_recovery or 'unknown'} days.
"""
    else:
        prompt += """
Note if no historical analogues were identified.
"""

    # Analysis section
    prompt += """
### 3. Analysis
"""
    
    # Add stats
    if enhanced_stats:
        total_ret = enhanced_stats.get('total_return', 0)
        ann_ret = enhanced_stats.get('annualized_return', 0)
        max_dd = enhanced_stats.get('max_drawdown', 0)
        var_95 = enhanced_stats.get('var_95', 0)
        skewness = enhanced_stats.get('skewness', 0)
        
        prompt += f"""
Key metrics to discuss:
| Metric | Value |
|--------|-------|
| Total Return | {total_ret:.2%} |
| Annualized Return | {ann_ret:.2%} |
| Max Drawdown | {max_dd:.2%} |
| VaR (95%) | {var_95:.2%} |
| Skewness | {skewness:.3f} |
"""

    if state == "VALID":
        prompt += """
Present the analysis confidently. If performance is positive, acknowledge it clearly.
"""
    elif state == "DEGRADED":
        prompt += """
Use hedging language: "Historically, the spread exhibited..." and "However, current regime stability is impaired."
"""
    else:
        prompt += """
Frame everything in past tense: "Prior to the breakdown, the spread exhibited..."
Make clear these characteristics may not hold under current conditions.
"""

    # Regime context
    if regime_summary:
        z = regime_summary.get('z_score', 0)
        rsi = regime_summary.get('rsi', 50)
        adf_p = regime_summary.get('adf_pvalue', 0.5)
        current_regime = regime_summary.get('current_regime', 'unknown')
        
        prompt += f"""
### 4. Current Entry Point
| Metric | Value |
|--------|-------|
| Current Regime | {current_regime.upper()} |
| Z-Score | {z:+.2f} |
| RSI | {rsi:.0f} |
| ADF p-value | {adf_p:.4f} |
"""

    # Recommendation framing
    prompt += """
### 5. Recommendation Framing
"""
    
    if state == "VALID":
        prompt += """
Provide clear recommendation: ENTER, WAIT, or PASS based on the evidence.
Be confident but acknowledge risk factors.
"""
    elif state == "DEGRADED":
        prompt += """
Frame recommendation as CONDITIONAL:
"Current positioning suggests [X] **contingent on structural stabilization** (currently: DEGRADED)."
Explicitly state the analysis should be interpreted with caution.
"""
    else:
        prompt += """
Frame as historical:
"Any positioning based on this analysis should be considered historical in nature, pending structural recovery."
Do NOT recommend entry while structure is broken.
"""

    # One-line summary
    prompt += """
### 6. One-Line Summary
A single sentence verdict that includes the structural status.
"""

    # Style guidelines
    prompt += """
---

## CRITICAL STYLE RULES

1. **Never ignore the structural assessment** — it must be the first section
2. **Match your confidence to validity state:**
   - VALID: Confident, direct
   - DEGRADED: Cautious, conditional, hedged
   - BROKEN: Historical framing, no actionable recommendations
3. **Use precise language:**
   - "Detected" not "AI thinks"
   - "Identified" not "model believes"
   - "Based on historical episodes" not "predicted"
4. **Keep memo under 600 words**
5. **Do NOT use Sharpe ratio** as a decision factor
"""

    return system_prompt, prompt


# =============================================================================
# MAIN GENERATION FUNCTION
# =============================================================================

async def generate_memo(
    enhanced_stats: Dict,
    regime_summary: Dict,
    portfolio_name: str = "Long/Short Portfolio",
    long_positions: Optional[Dict[str, float]] = None,
    short_positions: Optional[Dict[str, float]] = None,
    api_key: Optional[str] = None,
    model: str = "gpt-4o",
    validity_data: Optional[Dict] = None,
) -> str:
    """
    Generate validity-governed investment memo using OpenAI API.
    
    Parameters
    ----------
    enhanced_stats : Dict
        Performance and risk statistics.
    regime_summary : Dict
        Current regime analysis.
    portfolio_name : str
        Name for the portfolio.
    long_positions : Dict
        Long position weights.
    short_positions : Dict
        Short position weights.
    api_key : str, optional
        OpenAI API key.
    model : str
        OpenAI model to use.
    validity_data : Dict, optional
        Validity output from Bavella. If not provided, assumes VALID.
        
    Returns
    -------
    str
        Generated memo text.
    """
    # Default validity if not provided
    if validity_data is None:
        validity_data = {
            "validity": {"state": "VALID", "score": 85, "summary": "Statistical structure is stable", "confidence": 0.8, "root_cause": None},
            "history": {"similar_episodes": 0, "median_recovery_days": None},
            "details": {"secondary_failures": [], "assessment_note": "No structural issues detected."},
        }
    
    # Get API key
    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        logger.warning("No OpenAI API key provided, returning placeholder memo")
        return _generate_fallback_memo(enhanced_stats, regime_summary, validity_data)
    
    try:
        from openai import AsyncOpenAI
        
        client = AsyncOpenAI(api_key=api_key)
        
        system_prompt, user_prompt = build_validity_governed_prompt(
            enhanced_stats=enhanced_stats,
            regime_summary=regime_summary,
            validity_data=validity_data,
            portfolio_name=portfolio_name,
            long_positions=long_positions,
            short_positions=short_positions
        )
        
        logger.info(f"Generating validity-governed memo ({validity_data.get('validity', {}).get('state', 'VALID')})...")
        
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=2000
        )
        
        memo = response.choices[0].message.content
        logger.info("Validity-governed memo generated successfully")
        
        return memo
        
    except ImportError:
        logger.error("openai package not installed")
        return _generate_fallback_memo(enhanced_stats, regime_summary, validity_data)
    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        return _generate_fallback_memo(enhanced_stats, regime_summary, validity_data)


def _generate_fallback_memo(
    enhanced_stats: Dict, 
    regime_summary: Dict,
    validity_data: Dict
) -> str:
    """Generate a basic validity-governed memo without AI."""
    
    validity = validity_data.get("validity", {})
    history = validity_data.get("history", {})
    
    state = validity.get("state", "VALID")
    score = validity.get("score", 85)
    summary = validity.get("summary", "Statistical structure is stable")
    root_cause = validity.get("root_cause")
    confidence = validity.get("confidence", 0.8)
    
    total_ret = enhanced_stats.get('total_return', 0)
    max_dd = enhanced_stats.get('max_drawdown', 0)
    regime = regime_summary.get('current_regime', 'unknown')
    z_score = regime_summary.get('z_score', 0)
    
    similar_episodes = history.get("similar_episodes", 0)
    median_recovery = history.get("median_recovery_days")
    
    # Build memo based on state
    if state == "VALID":
        structural = f"**Structural assessment: Stable.** The statistical structure underlying this analysis remains intact."
        historical = "No prior structural breakdowns with comparable characteristics were identified."
        framing = ""
        
        if total_ret > 0.02 and max_dd > -0.10:
            recommendation = "ENTER"
            reasoning = "Strategy is profitable with contained risk."
        elif total_ret > 0 and max_dd > -0.15:
            recommendation = "WAIT"
            reasoning = "Positive but monitor for better entry."
        else:
            recommendation = "PASS"
            reasoning = "Risk/reward not attractive at current levels."
            
    elif state == "DEGRADED":
        structural = f"**Structural assessment: Degraded.** {summary} Confidence: {confidence:.0%}."
        
        if root_cause:
            structural += f"\n\nRoot cause: {root_cause['code']} — {root_cause['label']} (severity {root_cause['severity']})"
        
        structural += "\n\n**The following analysis should be interpreted with caution.**"
        
        if similar_episodes > 0 and median_recovery:
            historical = f"Similar structural conditions were identified on {similar_episodes} prior occasion(s). Median recovery time: approximately {median_recovery} days."
        else:
            historical = "Limited historical precedent available."
        
        framing = "**contingent on structural stabilization** (currently: DEGRADED)"
        recommendation = "WAIT"
        reasoning = f"Setup not ready. {framing}"
        
    else:  # BROKEN
        structural = f"**Structural assessment: Failed.** The statistical structure underlying this analysis is no longer reliable."
        
        if root_cause:
            structural += f"\n\nA severe {root_cause['label'].lower()} has been identified as the root cause of failure."
        
        structural += "\n\n**This memo describes historical behavior that may no longer apply under current conditions.**"
        
        if similar_episodes > 0:
            historical = f"Comparable breakdowns occurred on {similar_episodes} prior occasion(s)."
        else:
            historical = "No comparable historical episodes available."
        
        framing = "Any positioning based on this analysis should be considered historical in nature, pending structural recovery."
        recommendation = "PASS"
        reasoning = framing
    
    return f"""## Trade Analysis Summary

### Structural Assessment
{structural}

### Historical Context
{historical}

### Analysis
**Current Regime:** {regime.upper()}
- Total Return: {total_ret:.2%}
- Max Drawdown: {max_dd:.2%}
- Z-Score: {z_score:+.2f}

### Recommendation
**{recommendation}**: {reasoning}

### One-Line Summary
{recommendation}: {reasoning}

*Assessment based on multi-detector analysis with historical pattern matching.*
"""


# =============================================================================
# LEGACY COMPATIBILITY
# =============================================================================

def build_tactical_memo_prompt(
    enhanced_stats: Dict,
    regime_summary: Dict,
    portfolio_name: str = "Long/Short Portfolio",
    long_positions: Optional[Dict[str, float]] = None,
    short_positions: Optional[Dict[str, float]] = None
) -> str:
    """
    Build the prompt for tactical trade memo generation.
    LEGACY FUNCTION - kept for backward compatibility.
    """
    # Portfolio composition
    portfolio_comp = ""
    if long_positions:
        portfolio_comp += "LONG: " + ", ".join([f"{t} ({w:.0%})" for t, w in long_positions.items()]) + "\n"
    if short_positions:
        portfolio_comp += "SHORT: " + ", ".join([f"{t} ({w:.0%})" for t, w in short_positions.items()])
    
    prompt = f"""
You are a senior investment analyst evaluating a TACTICAL TRADE opportunity. Be BALANCED and HONEST - look at what the data actually shows, not just worst-case scenarios.

## Critical Framing
- Evaluate as a TRADE with defined entry/exit rules
- Be HONEST but BALANCED - if performance is positive, acknowledge it
- If the trade has been working, say so
- Conclusion can be ENTER, WAIT, or PASS based on actual evidence
- Do NOT be reflexively negative or overly cautious

## Portfolio Composition
{portfolio_comp if portfolio_comp else "Long/Short Equity Spread"}

## Portfolio: {portfolio_name}

### RISK METRICS
"""

    if enhanced_stats:
        total_ret = enhanced_stats.get('total_return', 0)
        ann_ret = enhanced_stats.get('annualized_return', 0)
        prompt += f"""
| Metric | Value |
|--------|-------|
| Max Drawdown | {enhanced_stats.get('max_drawdown', 0):.2%} |
| Worst Day | {enhanced_stats.get('worst_day', 0):.2%} |
| VaR (95%) | {enhanced_stats.get('var_95', 0):.2%} |
| CVaR (95%) | {enhanced_stats.get('cvar_95', 0):.2%} |
| Volatility (ann.) | {enhanced_stats.get('annualized_volatility', 0):.2%} |
| Skewness | {enhanced_stats.get('skewness', 0):.3f} |

### PERFORMANCE
| Metric | Value |
|--------|-------|
| Total Return | {total_ret:.2%} |
| Annualized Return | {ann_ret:.2%} |
| Win Rate | {enhanced_stats.get('win_rate', 0):.1%} |
| Best Day | {enhanced_stats.get('best_day', 0):.2%} |
"""
    
    if regime_summary:
        z = regime_summary.get('z_score', 0)
        rsi = regime_summary.get('rsi', 50)
        adf_p = regime_summary.get('adf_pvalue', 0.5)
        
        prompt += f"""
### CURRENT REGIME
| Metric | Value |
|--------|-------|
| Current Regime | {regime_summary.get('current_regime', 'N/A').upper()} |
| Z-Score | {z:+.2f} |
| RSI | {rsi:.0f} |
| ADF p-value | {adf_p:.4f} |
| Half-Life | {regime_summary.get('halflife', 999):.1f} days |
| Risk Level | {regime_summary.get('risk_level', 'N/A').upper()} |
"""
    
    prompt += """
---

Provide a balanced analysis with:
1. Trade Thesis
2. Risk Assessment  
3. Current Entry Point
4. Trade Recommendation (ENTER/WAIT/PASS)
5. One-Line Summary

Keep memo under 500 words. Do NOT use Sharpe ratio.
"""
    
    return prompt


# Keep old function signature working
async def generate_memo_legacy(
    enhanced_stats: Dict,
    regime_summary: Dict,
    portfolio_name: str = "Long/Short Portfolio",
    long_positions: Optional[Dict[str, float]] = None,
    short_positions: Optional[Dict[str, float]] = None,
    api_key: Optional[str] = None,
    model: str = "gpt-4o"
) -> str:
    """Legacy function without validity governance (assumes VALID)."""
    return await generate_memo(
        enhanced_stats=enhanced_stats,
        regime_summary=regime_summary,
        portfolio_name=portfolio_name,
        long_positions=long_positions,
        short_positions=short_positions,
        api_key=api_key,
        model=model,
        validity_data=None,  # Assumes VALID
    )
