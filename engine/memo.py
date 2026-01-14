"""
Memo Generator Module - AI-Powered Trade Analysis

Generates professional tactical trade memos using OpenAI GPT-4.
The prompt is carefully crafted for balanced, investment-grade analysis.

Supports loading API key from:
1. Direct parameter
2. Environment variable OPENAI_API_KEY
3. .env file in project root
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# Try to load dotenv for local development
def _load_env():
    """Load environment variables from .env file if available."""
    try:
        from dotenv import load_dotenv
        
        # Look for .env in current directory and parent directories
        env_paths = [
            Path(".env"),
            Path("../.env"),
            Path(__file__).parent / ".env",
            Path(__file__).parent.parent / ".env",
        ]
        
        for env_path in env_paths:
            if env_path.exists():
                load_dotenv(env_path)
                logger.info(f"Loaded environment from {env_path}")
                return True
        
        # Try default dotenv loading
        load_dotenv()
        return True
        
    except ImportError:
        logger.debug("python-dotenv not installed, using system environment only")
        return False


# Load environment at module import
_load_env()


SYSTEM_PROMPT = """You are a senior trader evaluating tactical trades. Be balanced and honest - acknowledge both positives and negatives. If a strategy has been profitable, say so. Do not be reflexively negative."""


def build_tactical_memo_prompt(
    enhanced_stats: Dict,
    regime_summary: Dict,
    portfolio_name: str = "Long/Short Portfolio",
    long_positions: Optional[Dict[str, float]] = None,
    short_positions: Optional[Dict[str, float]] = None
) -> str:
    """
    Build the prompt for tactical trade memo generation.
    
    This prompt is carefully designed to produce balanced, professional analysis.
    MATCHES THE JUPYTER NOTEBOOK PROMPT EXACTLY.
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

## Target Audience
- Investment professionals who understand tactical trading
- Decision makers evaluating "should we put this trade on NOW?"

## Required Sections

### Trade Thesis
What is this trade? Look at ACTUAL recent performance - has it been working or not?

### Risk Assessment
Can this trade blow up? Focus on:
- Downside containment: Max drawdown, VaR, worst day
- Tail risk: Skewness and kurtosis
- Use $10M position for concrete examples
- Be factual, not alarmist

### Current Entry Point
Based on regime data - is this a good entry?
- Consider the TREND: is fair value rising or falling?
- RSI and Z-score context
- If recently pulled back from highs after a run-up, that can be GOOD entry
- If at all-time highs with overbought signals, that's POOR entry

### What the Data Actually Shows
Look at the numbers honestly:
- If total return is POSITIVE, the strategy is working
- If max drawdown is contained, risk is managed
- If skewness is positive, no hidden tail risk
- Don't dismiss good results

### Trade Recommendation
Clear verdict based on EVIDENCE:
- **ENTER** - Good setup, strategy working, acceptable risk
- **WAIT** - Setup not ready, specify what would improve it
- **PASS** - Risk/reward not attractive OR strategy clearly broken

### One-Line Summary
Honest verdict.

## Style Guidelines

DO:
- Be balanced - acknowledge positives AND negatives
- If the strategy made money, say "the strategy has been profitable"
- If drawdowns are small, say "risk has been well-contained"
- Look at the actual trend in fair value
- Consider that pullbacks after gains can be good entries

DON'T:
- Be reflexively negative or overly cautious
- Dismiss positive performance
- Focus only on what could go wrong
- Recommend PASS just because you're being conservative
- Use Sharpe ratio as a decision factor (irrelevant for tactical trades)

## Key Context

This is a long/short equity spread:
- Returns come from spread convergence/divergence
- Low market correlation by design
- Regime signals dictate entry/exit

---

# TRADE DATA

## Portfolio: {portfolio_name}

### RISK METRICS (Can it blow up?)
"""

    if enhanced_stats:
        total_ret = enhanced_stats.get('total_return', 0)
        ann_ret = enhanced_stats.get('annualized_return', 0)
        prompt += f"""
| Metric | Value | Interpretation |
|--------|-------|----------------|
| Max Drawdown | {enhanced_stats.get('max_drawdown', 0):.2%} | Worst peak-to-trough |
| Worst Day | {enhanced_stats.get('worst_day', 0):.2%} | Single day max pain |
| VaR (95%) | {enhanced_stats.get('var_95', 0):.2%} | 19/20 days within this |
| CVaR (95%) | {enhanced_stats.get('cvar_95', 0):.2%} | Average bad day loss |
| Volatility (ann.) | {enhanced_stats.get('annualized_volatility', 0):.2%} | Risk level |
| Skewness | {enhanced_stats.get('skewness', 0):.3f} | Positive = no hidden tail risk |
| Kurtosis | {enhanced_stats.get('kurtosis', 0):.3f} | <3 = fewer surprises |

### ACTUAL PERFORMANCE (Has it been working?)
| Metric | Value | Reading |
|--------|-------|---------|
| Total Return (180d) | {total_ret:.2%} | {"POSITIVE - strategy working" if total_ret > 0 else "NEGATIVE - strategy struggling"} |
| Annualized Return | {ann_ret:.2%} | {"Profitable" if ann_ret > 0 else "Unprofitable"} |
| Win Rate | {enhanced_stats.get('win_rate', 0):.1%} | {"Above 50% - edge exists" if enhanced_stats.get('win_rate', 0) > 0.5 else "Below 50%"} |
| Best Day | {enhanced_stats.get('best_day', 0):.2%} | Upside captured |
"""
    
    if regime_summary:
        z = regime_summary.get('z_score', 0)
        rsi = regime_summary.get('rsi', 50)
        adf_p = regime_summary.get('adf_pvalue', 0.5)
        
        z_reading = "Oversold - potential buy" if z < -1 else "Overbought - caution" if z > 1.5 else "Near fair value"
        rsi_reading = "Oversold" if rsi < 30 else "Overbought" if rsi > 70 else "Neutral"
        adf_reading = "Can trend" if adf_p > 0.3 else "Mean-reverting"
        
        prompt += f"""
### CURRENT ENTRY POINT
| Metric | Value | Reading |
|--------|-------|---------|
| Current Regime | {regime_summary.get('current_regime', 'N/A').upper()} | Market state |
| Z-Score | {z:+.2f} | {z_reading} |
| RSI | {rsi:.0f} | {rsi_reading} |
| ADF p-value | {adf_p:.4f} | {adf_reading} |
| Half-Life | {regime_summary.get('halflife', 999):.1f} days | Trade duration |
| Risk Level | {regime_summary.get('risk_level', 'N/A').upper()} | Current risk |

**Strategy Suggestion:** {regime_summary.get('strategy', 'N/A')}
"""
    
    prompt += """
---

**IMPORTANT:** 
- Be BALANCED - if the strategy made money, acknowledge it
- Look at what the data ACTUALLY shows
- Pullbacks after gains can be good entries, not red flags
- Do NOT use Sharpe ratio in your analysis
- Keep memo under 500 words
"""
    
    return prompt


async def generate_memo(
    enhanced_stats: Dict,
    regime_summary: Dict,
    portfolio_name: str = "Long/Short Portfolio",
    long_positions: Optional[Dict[str, float]] = None,
    short_positions: Optional[Dict[str, float]] = None,
    api_key: Optional[str] = None,
    model: str = "gpt-4o"
) -> str:
    """
    Generate investment memo using OpenAI API.
    
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
        OpenAI API key. If not provided, uses OPENAI_API_KEY env var.
    model : str
        OpenAI model to use. Valid options: gpt-4o, gpt-4o-mini, gpt-4-turbo
        
    Returns
    -------
    str
        Generated memo text.
    """
    # Try to get API key from multiple sources
    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        logger.warning("No OpenAI API key provided, returning placeholder memo")
        return _generate_fallback_memo(enhanced_stats, regime_summary)
    
    # Validate model string
    valid_models = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"]
    if model not in valid_models:
        logger.warning(f"Invalid model '{model}', using 'gpt-4o'")
        model = "gpt-4o"
    
    try:
        from openai import AsyncOpenAI
        
        client = AsyncOpenAI(api_key=api_key)
        
        prompt = build_tactical_memo_prompt(
            enhanced_stats=enhanced_stats,
            regime_summary=regime_summary,
            portfolio_name=portfolio_name,
            long_positions=long_positions,
            short_positions=short_positions
        )
        
        logger.info(f"Calling OpenAI API ({model})...")
        
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2000
        )
        
        memo = response.choices[0].message.content
        logger.info("Memo generated successfully")
        
        return memo
        
    except ImportError:
        logger.error("openai package not installed. Install with: pip install openai")
        return _generate_fallback_memo(enhanced_stats, regime_summary)
    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        return _generate_fallback_memo(enhanced_stats, regime_summary)


def _generate_fallback_memo(enhanced_stats: Dict, regime_summary: Dict) -> str:
    """Generate a basic memo without AI when API is unavailable."""
    
    total_ret = enhanced_stats.get('total_return', 0)
    max_dd = enhanced_stats.get('max_drawdown', 0)
    regime = regime_summary.get('current_regime', 'unknown')
    z_score = regime_summary.get('z_score', 0)
    rsi = regime_summary.get('rsi', 50)
    
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
    
    return f"""## Trade Analysis Summary

**Current Regime:** {regime.upper()}
**Recommendation:** {recommendation}

### Key Metrics
- Total Return: {total_ret:.2%}
- Max Drawdown: {max_dd:.2%}
- Z-Score: {z_score:+.2f}
- RSI: {rsi:.0f}

### Assessment
{reasoning}

### One-Line Summary
{recommendation}: {reasoning}

*Note: AI analysis unavailable. This is a simplified summary based on key metrics.*
"""
