"""
Bavella — Decision Brief Memo Generator (v2)
==============================================

PHILOSOPHY: The memo answers ONE question: "What should I do with this trade?"

v2 CHANGES from v1:
- REVERSE added to recommendation taxonomy (pairs can be flipped, not just exited)
- RISK PROFILE forces math-based derivation — monthly VaR = daily x sqrt(21)
- BIDIRECTIONAL P&L framing: always shows what the inverse trade would have done
- FUNDAMENTAL DRIVER HYPOTHESES: maps statistical patterns to likely fundamental causes
  (valuation gap, analyst re-rating, sector rotation, management change, etc.)
- WHAT WOULD CHANGE uses hard negative examples to prevent GPT laziness
- Validity score vs confidence explicitly disambiguated in prompt
- Regime-vs-drawdown consistency check catches misclassification
- Optional fundamental_context param for user-provided thesis
- Max tokens bumped to 3500 for multi-FM briefs
- System prompt sharpened with anti-patterns list

Copyright 2024-2026 Bavella Technologies Sarl
"""

from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


FM_DESCRIPTIONS = {
    "FM1": {
        "name": "Volatility Regime Shift",
        "plain_english": "The spread's volatility has changed significantly compared to its historical baseline. This means the risk you're actually taking is different from what backtests suggest.",
        "so_what": "Your position sizing is probably wrong — it was calibrated to old volatility. Stop-losses set on historical vol may be too tight (whipsawed out on noise) or too loose (absorbing larger losses than expected).",
        "what_to_do": "Resize the position to reflect current vol. If vol doubled, halve the position.",
        "regime_note": {"trend": "Less concerning — trending markets naturally exhibit changing volatility.", "mr": "Serious concern — mean-reversion strategies assume stable variance, and that assumption just broke."},
    },
    "FM2": {
        "name": "Mean Drift",
        "plain_english": "The 'fair value' of the spread is moving — the equilibrium level you're mean-reverting to is shifting over time rather than staying fixed.",
        "so_what": "If you're trading mean reversion, you're reverting to a target that's no longer there. It's like aiming at where the goal was 5 minutes ago.",
        "what_to_do": "Re-estimate the equilibrium level. If the drift is persistent and large, the fundamental relationship between the two legs may be changing.",
        "regime_note": {"trend": "Expected in trending regimes — the spread is supposed to be moving.", "mr": "Red flag — the core assumption of your trade (stable mean) is violated."},
    },
    "FM3": {
        "name": "Seasonality Mismatch",
        "plain_english": "The spread's behavior follows seasonal or cyclical patterns that the analysis window doesn't fully capture. The 'signal' you're seeing may be a recurring calendar effect.",
        "so_what": "You may be entering at a seasonally adverse time, or the apparent opportunity is just a pattern that repeats every year (e.g., sector rotation, earnings cycles).",
        "what_to_do": "Check if the entry aligns with or fights the seasonal pattern. Consider timing differently.",
        "regime_note": {"trend": "May explain part of the trend — check if it's a seasonal move masquerading as a trend.", "mr": "Could create false entry signals if the seasonal pattern is strong."},
    },
    "FM4": {
        "name": "Structural Break",
        "plain_english": "A fundamental, discrete shift has occurred in the statistical relationship. Think of it as a 'before and after' moment — the pair's properties permanently changed.",
        "so_what": "THIS IS THE MOST SERIOUS FAILURE MODE. Everything before the break point is irrelevant for forecasting. Your backtests are showing you a relationship that no longer exists.",
        "what_to_do": "Strong signal to exit or avoid entirely. Only re-enter after enough post-break data confirms a new stable relationship has formed (typically 40-60 trading days minimum).",
        "regime_note": {"trend": "Still dangerous — structural breaks invalidate the relationship regardless of regime.", "mr": "Critical — the mean-reversion equilibrium has been destroyed."},
    },
    "FM5": {
        "name": "Outlier Contamination",
        "plain_english": "A few extreme observations (outlier days) are distorting the statistics. The 'signal' may be driven by 2-3 anomalous days rather than genuine market dynamics.",
        "so_what": "Risk metrics (VaR, max drawdown) are unreliable — they're being pulled by outliers. The trade may look better or worse than it actually is.",
        "what_to_do": "Verify the signal holds after removing outliers. If the thesis disappears without those extreme days, it wasn't a real signal. Size conservatively.",
        "regime_note": {"trend": "Outliers during trends often mark regime transitions — watch for follow-through.", "mr": "Can create false entry signals if extreme days distort Z-scores."},
    },
    "FM6": {
        "name": "Extreme Positioning / Crowding",
        "plain_english": "The spread is at extreme levels relative to its historical range, suggesting positioning may be heavily one-sided.",
        "so_what": "Crowded trades unwind violently. Even if your thesis is correct, a positioning squeeze can cause significant drawdown before convergence.",
        "what_to_do": "Scale in gradually rather than taking full position. Set wider stops to survive a potential squeeze.",
        "regime_note": {"trend": "Extreme positioning in a trend may persist longer than expected.", "mr": "Could be a legitimate mean-reversion entry — but only if other FMs are clean."},
    },
    "FM7": {
        "name": "Correlation / Dependency Breakdown",
        "plain_english": "The correlation between the two legs of your trade is deteriorating. They're becoming less related to each other over time.",
        "so_what": "A pairs trade IS the correlation. If correlation is breaking down, you don't have a hedge — you have two separate directional bets.",
        "what_to_do": "In mean-reversion regimes, this is very dangerous — reduce or exit. In trending regimes, some decorrelation is expected and less alarming.",
        "regime_note": {"trend": "Some decorrelation is EXPECTED in trending markets — severity is attenuated.", "mr": "MOST DANGEROUS HERE — the entire trade thesis depends on stable correlation."},
    },
    "COMPOSITE": {
        "name": "Multiple Factors",
        "plain_english": "The engine detected degradation from multiple sources that couldn't be attributed to a single failure mode.",
        "so_what": "When the engine can't pinpoint one cause, it typically means several mild issues are compounding.",
        "what_to_do": "Review the validity score and consider reducing exposure until a clearer signal emerges.",
        "regime_note": {"trend": "Ambiguity is common in transitional markets.", "mr": "Multiple mild issues in MR regimes can compound into real problems."},
    },
}


REGIME_TRADING_CONTEXT = {
    "strong_mean_reversion": {"label": "Strong Mean Reversion", "suitable_for": "Classic pairs trading and statistical arbitrage — this is the ideal regime", "not_suitable_for": "Momentum or trend-following strategies", "fm_note": "All failure modes are at full relevance. FM7 (correlation breakdown) is especially dangerous here.", "entry_guidance": "Z-score extremes are meaningful entry signals. Half-life gives your expected trade duration."},
    "mean_reversion": {"label": "Mean Reversion", "suitable_for": "Pairs trading with moderate conviction", "not_suitable_for": "Aggressive sizing — stationary but less strongly so", "fm_note": "Most failure modes are relevant. Some tolerance for mild FM2 if magnitude is small.", "entry_guidance": "Z-score entries valid but use wider bands. Expect some drift around the mean."},
    "mixed_mode": {"label": "Mixed Mode (Ambiguous)", "suitable_for": "Flexible strategies — but punishes strong conviction either way", "not_suitable_for": "Pure mean reversion OR pure trend following", "fm_note": "Failure modes at ~70% relevance. Both signals will appear and contradict.", "entry_guidance": "REDUCE SIZE. Wait for regime clarity if possible."},
    "trend_following": {"label": "Trend Following", "suitable_for": "Momentum strategies on the spread direction", "not_suitable_for": "Mean reversion — Z-score extremes may NOT revert", "fm_note": "FM7 is EXPECTED. FM4 still dangerous. FM1 partially expected.", "entry_guidance": "Trade with the trend. Z-score extremes may be the new equilibrium."},
    "pure_trend": {"label": "Pure Trend (Non-Stationary)", "suitable_for": "Strong directional bets only", "not_suitable_for": "ANY mean-reversion strategy", "fm_note": "Most FMs attenuated except FM4. This regime IS the signal.", "entry_guidance": "If in a MR trade, this regime says GET OUT."},
    "ranging": {"label": "Ranging / Low Conviction", "suitable_for": "Small positions with tight risk limits", "not_suitable_for": "Large conviction bets", "fm_note": "FMs should be taken at face value. Ranging + active FMs = stay away.", "entry_guidance": "Small MR entries possible if Z extended and FMs clean. Can transition without warning."},
}


SPREAD_MOVE_PATTERNS = {
    "slow_divergence": {
        "signature": "Steady, low-vol directional move over weeks/months. Spread drifts persistently.",
        "likely_drivers": {
            "equity": [
                "Valuation re-rating — one leg being repriced (P/E expansion or compression)",
                "Margin improvement / deterioration — one company's fundamentals gradually improving vs the other",
                "Analyst consensus shift — sell-side slowly upgrading one leg and/or downgrading the other",
                "Structural sector rotation — capital flows shifting between sub-sectors",
            ],
            "crypto": [
                "Narrative rotation — market shifting allocation between store-of-value (BTC) and smart-contract platforms (ETH/SOL/etc)",
                "Institutional flow divergence — ETF inflows favoring one asset, retail speculation favoring the other",
                "Ecosystem growth differential — one chain gaining TVL, developers, or dApp activity faster",
                "Risk-on/risk-off rotation — BTC outperforms in risk-off, alts outperform in risk-on",
                "Token unlock or inflation schedule — one asset diluting holders faster than the other",
            ],
        },
        "verification": {
            "equity": "Check: relative P/E trend, earnings revision momentum, margin forecasts, broker notes.",
            "crypto": "Check: ETF flow data (BTC/ETH), DeFiLlama TVL trends, developer activity (Electric Capital), token unlock calendars, funding rates.",
        },
        "implication": "If fundamental, the move is likely PERSISTENT. Mean reversion is fighting a real repricing — dangerous.",
    },
    "sharp_break": {
        "signature": "Sudden, large move over 1-5 days. High kurtosis. Outlier days visible.",
        "likely_drivers": {
            "equity": [
                "Earnings surprise — one leg beat/missed significantly",
                "Management change — CEO/CFO departure, activist involvement",
                "M&A event — takeover bid, merger announcement, deal break",
                "Analyst re-rating — major broker initiates/upgrades/downgrades with PT change",
                "Regulatory / legal event — fine, lawsuit, licensing change on one leg",
            ],
            "crypto": [
                "Regulatory event — SEC action, ETF approval/rejection, exchange enforcement affecting one asset",
                "Protocol incident — exploit, bridge hack, network outage, consensus failure on one chain",
                "Major listing/delisting — Coinbase/Binance adding or removing one asset",
                "Whale liquidation or accumulation — large on-chain movements triggering cascade",
                "Ecosystem shock — major protocol collapse (FTX-type event), stablecoin depeg affecting one chain",
            ],
        },
        "verification": {
            "equity": "Check: recent earnings dates, 8-K filings, analyst action dates, news headlines around the break date.",
            "crypto": "Check: CoinDesk/The Block headlines around break date, on-chain whale alerts, exchange announcement feeds, DeFi exploit trackers.",
        },
        "implication": "Assess if event is PRICED IN or has legs. One-time events often mean-revert; structural changes don't.",
    },
    "seasonal_cyclical": {
        "signature": "FM3 active. Spread follows recurring calendar patterns.",
        "likely_drivers": {
            "equity": [
                "Earnings cycle mismatch — legs report in different quarters or have different seasonal revenue",
                "Sector rotation — periodic capital flows (sell in May, tax-loss selling, January effect)",
                "Index rebalancing — one leg added/removed from major indices",
                "Dividend calendar — ex-div dates creating temporary spread distortions",
            ],
            "crypto": [
                "Bitcoin halving cycle effects — BTC dominance tends to rise pre-halving, alts outperform post-halving",
                "Quarterly futures/options expiry — basis trade unwinds creating temporary dislocations",
                "Tax-loss harvesting season — year-end selling pressure differs between established and newer assets",
                "Conference/event calendar — major chain-specific events (Solana Breakpoint, ETH Devcon) creating temporary narrative pumps",
            ],
        },
        "verification": {
            "equity": "Check: earnings calendars, index reconstitution dates, ex-dividend dates, YoY spread overlay.",
            "crypto": "Check: CME/Deribit expiry calendar, halving cycle position, major conference dates, YoY spread overlay.",
        },
        "implication": "Seasonal moves are PREDICTABLE but timing-dependent. Entering against the pattern bets this time is different.",
    },
    "correlation_breakdown": {
        "signature": "FM7 active. Two legs becoming less correlated. Spread vol increasing.",
        "likely_drivers": {
            "equity": [
                "Fundamental decoupling — companies diverging in business mix, geography, or strategy",
                "Sector reclassification — one leg re-categorized by GICS/ICB, attracting different investors",
                "M&A speculation — one leg trading on takeover premium, disconnecting from sector",
                "Idiosyncratic risk — company-specific event (fraud, product failure, regulatory action) on one leg",
            ],
            "crypto": [
                "Narrative divergence — one asset gaining a distinct use case (BTC=digital gold, SOL=memecoins/DeFi)",
                "Institutional vs retail split — one asset dominated by ETF/institutional flows, the other by retail speculation",
                "Technology divergence — chain upgrades or performance issues creating genuinely different risk profiles",
                "Regulatory bifurcation — one asset classified differently (commodity vs security) changing investor base",
            ],
        },
        "verification": {
            "equity": "Check: revenue mix changes, geographic exposure shifts, GICS reclassifications, short interest divergence.",
            "crypto": "Check: on-chain holder composition (institutional vs retail wallets), DEX vs CEX volume ratio, regulatory classification status.",
        },
        "implication": "If correlation breaking for fundamental reasons, pair thesis is INVALID. Now two separate directional bets.",
    },
    "vol_regime_shift": {
        "signature": "FM1 active. Spread volatility changed significantly from baseline.",
        "likely_drivers": {
            "equity": [
                "New information regime — catalyst creating ongoing uncertainty (pending regulatory decision, earnings)",
                "Liquidity change — one leg less liquid (delisting risk, small-cap neglect, MM withdrawal)",
                "Options/derivatives activity — unusual flow on one leg changing realized vol",
                "Macro regime shift — rates/credit spread changes affecting legs differently",
            ],
            "crypto": [
                "Leverage cycle — one asset seeing funding rate spikes or liquidation cascades on perps",
                "Liquidity migration — market makers moving between venues, widening spreads on one asset",
                "Macro regime shift — Fed pivot, DXY move, or risk event driving BTC (macro asset) differently from alts",
                "Network congestion / fee spikes — one chain experiencing capacity issues changing trading dynamics",
                "Derivatives market structure — new perpetual or options listings changing the vol surface for one asset",
            ],
        },
        "verification": {
            "equity": "Check: implied vol (options skew), bid-ask spread changes, ADV trends, macro factor exposure.",
            "crypto": "Check: funding rates (Coinglass), liquidation data, DEX liquidity depth, on-chain fee trends, derivatives OI changes.",
        },
        "implication": "Vol shifts invalidate sizing. Also signal SOMETHING is happening — find out what.",
    },
    "structural_break": {
        "signature": "FM4 active. Discrete regime change — clear before/after in the data.",
        "likely_drivers": {
            "equity": [
                "Corporate action — spin-off, rights issue, major divestiture changing company profile",
                "Regulatory change — new regulation/deregulation affecting one leg's business model",
                "Management / strategy pivot — new CEO with different capital allocation",
                "Credit event — restructuring, covenant breach, rating downgrade on one leg",
                "Sector disruption — tech or competitive shift permanently altering one leg's position",
            ],
            "crypto": [
                "Regulatory classification — SEC ruling one asset is a security, ETF approval/rejection",
                "Protocol governance crisis — contentious hard fork, foundation controversy, leadership exodus",
                "Major ecosystem collapse — largest protocol on one chain fails (Luna/UST-type event)",
                "Token economics change — supply schedule alteration, fee burn mechanism, major unlock event",
                "Exchange catastrophe — major exchange supporting one ecosystem collapses (FTX-type, but chain-specific)",
            ],
        },
        "verification": {
            "equity": "Check: corporate actions calendar, regulatory filings, credit ratings, strategic announcements around break date.",
            "crypto": "Check: governance proposals, SEC/CFTC filings, protocol upgrade timelines, token unlock schedules, exchange status.",
        },
        "implication": "Structural breaks are PERMANENT until proven otherwise. Pre-break data unreliable. Pair may need to be retired.",
    },
}


# =============================================================================
# ASSET CLASS DETECTION (adapts prompt language automatically)
# =============================================================================

_CRYPTO_SUFFIXES = {"-USD", "-USDT", "-BTC", "-ETH", "-EUR", "-GBP", "-BUSD", "-USDC"}
_CRYPTO_TICKERS = {
    "BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE", "AVAX", "DOT", "MATIC",
    "LINK", "UNI", "ATOM", "LTC", "FIL", "APT", "ARB", "OP", "NEAR", "ICP",
    "SUI", "SEI", "TIA", "RENDER", "INJ", "FET", "TAO", "WIF", "PEPE", "BONK",
    "JUP", "PYTH", "JTO", "TRX", "SHIB", "TON", "HBAR",
}


def _detect_asset_class(tickers: list) -> str:
    """Detect whether tickers are equity, crypto, or mixed."""
    crypto_count = 0
    for t in tickers:
        t_upper = t.upper()
        # Check suffix pattern (BTC-USD, SOL-USDT, etc.)
        if any(t_upper.endswith(s) for s in _CRYPTO_SUFFIXES):
            crypto_count += 1
            continue
        # Check known crypto tickers
        base = t_upper.split("-")[0].split(".")[0]
        if base in _CRYPTO_TICKERS:
            crypto_count += 1
    
    if crypto_count == len(tickers):
        return "crypto"
    elif crypto_count == 0:
        return "equity"
    return "mixed"


def _infer_spread_driver(enhanced_stats, failure_modes=None, asset_class="equity"):
    """Infer most likely spread driver from statistical signatures.
    
    asset_class: 'equity', 'crypto', or 'mixed' — selects the right driver descriptions.
    Returns (pattern_key, flattened_pattern_dict) with lists/strings, not nested dicts.
    """
    active_fm_keys = set()
    if failure_modes:
        for fm in failure_modes:
            if fm.get('detected', False):
                fm_id = fm.get('failure_mode_id', '')
                fm_key = fm_id.split('_')[0].upper() if '_' in str(fm_id) else str(fm_id).upper()
                active_fm_keys.add(fm_key)
    
    kurt = enhanced_stats.get('kurtosis', 3)
    max_dd = abs(enhanced_stats.get('max_drawdown', 0))
    best_day = abs(enhanced_stats.get('best_day', 0))
    worst_day = abs(enhanced_stats.get('worst_day', 0))
    
    pattern_key = "none"
    if "FM4" in active_fm_keys:
        pattern_key = "structural_break"
    elif "FM7" in active_fm_keys:
        pattern_key = "correlation_breakdown"
    elif "FM3" in active_fm_keys:
        pattern_key = "seasonal_cyclical"
    elif kurt > 5 or (max(best_day, worst_day) > 0.03 and max_dd > 0.10):
        pattern_key = "sharp_break"
    elif "FM1" in active_fm_keys:
        pattern_key = "vol_regime_shift"
    elif abs(enhanced_stats.get('total_return', 0)) > 0.05:
        pattern_key = "slow_divergence"
    
    if pattern_key == "none":
        return "none", {}
    
    raw = SPREAD_MOVE_PATTERNS[pattern_key]
    # Determine which variant to use
    ac = asset_class if asset_class in ("equity", "crypto") else "equity"
    
    # Flatten: extract the right variant from nested dicts
    drivers = raw.get("likely_drivers", {})
    if isinstance(drivers, dict):
        # New format with equity/crypto variants
        drivers_list = drivers.get(ac, drivers.get("equity", []))
    else:
        # Legacy flat list format (shouldn't happen but defensive)
        drivers_list = drivers
    
    verification = raw.get("verification", "")
    if isinstance(verification, dict):
        verification_str = verification.get(ac, verification.get("equity", ""))
    else:
        verification_str = verification
    
    return pattern_key, {
        "signature": raw.get("signature", ""),
        "likely_drivers": drivers_list,
        "verification": verification_str,
        "implication": raw.get("implication", ""),
    }


# =============================================================================
# FUNDAMENTAL DATA FETCHER (real valuation + margin data from yfinance)
# =============================================================================

_FUNDAMENTAL_METRICS = {
    # VALUATION
    "trailingPE": {"label": "P/E (trailing)", "fmt": ".1f", "category": "valuation"},
    "forwardPE": {"label": "P/E (forward)", "fmt": ".1f", "category": "valuation"},
    "enterpriseToEbitda": {"label": "EV/EBITDA", "fmt": ".1f", "category": "valuation"},
    "priceToBook": {"label": "P/Book", "fmt": ".2f", "category": "valuation"},
    "enterpriseToRevenue": {"label": "EV/Revenue", "fmt": ".2f", "category": "valuation"},
    "pegRatio": {"label": "PEG Ratio", "fmt": ".2f", "category": "valuation"},
    # MARGINS
    "grossMargins": {"label": "Gross Margin", "fmt": ".1%", "category": "margins"},
    "operatingMargins": {"label": "Operating Margin", "fmt": ".1%", "category": "margins"},
    "profitMargins": {"label": "Net Margin", "fmt": ".1%", "category": "margins"},
    "ebitdaMargins": {"label": "EBITDA Margin", "fmt": ".1%", "category": "margins"},
    # GROWTH
    "revenueGrowth": {"label": "Revenue Growth (YoY)", "fmt": ".1%", "category": "growth"},
    "earningsGrowth": {"label": "Earnings Growth (YoY)", "fmt": ".1%", "category": "growth"},
    "earningsQuarterlyGrowth": {"label": "Earnings Growth (QoQ)", "fmt": ".1%", "category": "growth"},
    # CAPITAL EFFICIENCY
    "returnOnEquity": {"label": "ROE", "fmt": ".1%", "category": "returns"},
    "returnOnAssets": {"label": "ROA", "fmt": ".1%", "category": "returns"},
    # BALANCE SHEET STRENGTH
    "debtToEquity": {"label": "Debt/Equity", "fmt": ".1f", "category": "balance_sheet"},
    "currentRatio": {"label": "Current Ratio", "fmt": ".2f", "category": "balance_sheet"},
    "freeCashflowYield": {"label": "FCF Yield", "fmt": ".1%", "category": "balance_sheet"},
    # ANALYST SENTIMENT (the Street's view)
    "recommendationMean": {"label": "Analyst Rating (1=Strong Buy, 5=Sell)", "fmt": ".2f", "category": "analyst"},
    "targetUpside": {"label": "Price Target Upside", "fmt": ".1%", "category": "analyst"},
    "numberOfAnalystOpinions": {"label": "Analyst Coverage", "fmt": ".0f", "category": "analyst"},
    # MOMENTUM (relative performance)
    "beta": {"label": "Beta (market sensitivity)", "fmt": ".2f", "category": "momentum"},
    "fiftyTwoWeekChange": {"label": "52-Week Price Change", "fmt": ".1%", "category": "momentum"},
    "distFromHigh": {"label": "Distance from 52W High", "fmt": ".1%", "category": "momentum"},
    # SIZE & INCOME
    "marketCap": {"label": "Market Cap", "fmt": "cap", "category": "size"},
    "dividendYield": {"label": "Dividend Yield", "fmt": ".1%", "category": "yield"},
}


def _format_metric(value, fmt: str) -> str:
    if value is None:
        return "N/A"
    try:
        if fmt == "cap":
            v = float(value)
            if v >= 1e12: return f"${v/1e12:.1f}T"
            if v >= 1e9: return f"${v/1e9:.1f}B"
            if v >= 1e6: return f"${v/1e6:.0f}M"
            return f"${v:,.0f}"
        return f"{float(value):{fmt}}"
    except (ValueError, TypeError):
        return str(value)


def _safe_divide(a, b, min_denominator=0.01):
    """Safe division — returns None if denominator too small or inputs invalid."""
    try:
        a, b = float(a), float(b)
        if abs(b) < min_denominator:
            return None
        return a / b
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def fetch_pair_fundamentals(tickers: list) -> dict:
    """
    Fetch valuation, margins, growth from yfinance with SELF-COMPUTED RATIOS.
    
    Yahoo's pre-computed ratios (trailingPE, forwardPE, enterpriseToEbitda) are 
    frequently stale or wrong. We compute from raw components (price, EPS, EV, 
    EBITDA) which update with market price, then cross-validate.
    """
    try:
        import yfinance as yf
    except ImportError:
        logger.warning("yfinance not installed — skipping fundamental data fetch")
        return {}
    result = {}
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info or {}
            td = {"name": info.get("shortName") or info.get("longName") or ticker,
                  "sector": info.get("sector", "Unknown"), "industry": info.get("industry", "Unknown"),
                  "currency": info.get("currency", ""), "metrics": {}, "warnings": [], "error": None}
            
            # ── Step 1: Extract raw components ──
            price = info.get("currentPrice") or info.get("regularMarketPrice")
            trailing_eps = info.get("trailingEps")
            forward_eps = info.get("forwardEps")
            book_value = info.get("bookValue")
            ev = info.get("enterpriseValue")
            ebitda = info.get("ebitda")
            total_revenue = info.get("totalRevenue")
            
            # ── Step 2: COMPUTE ratios ourselves (more reliable than pre-computed) ──
            calc_trailing_pe = _safe_divide(price, trailing_eps)
            calc_forward_pe = _safe_divide(price, forward_eps)
            calc_pb = _safe_divide(price, book_value)
            calc_ev_ebitda = _safe_divide(ev, ebitda, min_denominator=1e6)
            calc_ev_revenue = _safe_divide(ev, total_revenue, min_denominator=1e6)
            
            # Map: metric_key -> (calculated_value, yahoo_pre_computed, label)
            valuation_ratios = {
                "trailingPE": (calc_trailing_pe, info.get("trailingPE"), "P/E (trailing)"),
                "forwardPE": (calc_forward_pe, info.get("forwardPE"), "P/E (forward)"),
                "priceToBook": (calc_pb, info.get("priceToBook"), "P/Book"),
                "enterpriseToEbitda": (calc_ev_ebitda, info.get("enterpriseToEbitda"), "EV/EBITDA"),
                "enterpriseToRevenue": (calc_ev_revenue, info.get("enterpriseToRevenue"), "EV/Revenue"),
            }
            
            for key, (calc_val, yahoo_val, label) in valuation_ratios.items():
                # Prefer calculated value; fall back to Yahoo if we can't compute
                val = calc_val if calc_val is not None else yahoo_val
                if val is None or val == 0:
                    continue
                
                # Cross-validate: warn if calculated and Yahoo diverge significantly
                if calc_val is not None and yahoo_val is not None and yahoo_val != 0:
                    pct_diff = abs(calc_val - yahoo_val) / abs(yahoo_val)
                    if pct_diff > 0.20:  # >20% divergence
                        td["warnings"].append(
                            f"{label}: calculated={calc_val:.1f} vs Yahoo pre-computed={yahoo_val:.1f} "
                            f"({pct_diff:.0%} divergence). Using calculated value from current price."
                        )
                        val = calc_val  # always prefer our calculation
                
                # Outlier guards (same as before but applied to our calculated value)
                if key in ("trailingPE", "forwardPE") and (val > 80 or val < 0):
                    td["warnings"].append(
                        f"{label} = {val:.1f} — DISTORTED "
                        f"(likely one-off charges/gains). Use alternative metrics."
                    )
                    td["metrics"][f"_{key}_raw"] = val
                    continue
                if key == "enterpriseToEbitda" and (val > 50 or val < 0):
                    td["warnings"].append(
                        f"EV/EBITDA = {val:.1f} — DISTORTED. Likely EBITDA near zero or negative."
                    )
                    td["metrics"][f"_{key}_raw"] = val
                    continue
                
                td["metrics"][key] = val
            
            # ── Step 2b: Cross-validate P/E vs EV/EBITDA consistency ──
            # If both exist, they should be directionally consistent. P/E-to-EV/EBITDA  
            # ratio is normally 0.3x to 3.5x. Outside this = one number is wrong.
            best_pe = td["metrics"].get("forwardPE") or td["metrics"].get("trailingPE")
            ev_ebitda_val = td["metrics"].get("enterpriseToEbitda")
            if best_pe and ev_ebitda_val and ev_ebitda_val > 0:
                pe_to_ev = best_pe / ev_ebitda_val
                if pe_to_ev > 4.0 or pe_to_ev < 0.2:
                    td["warnings"].append(
                        f"EV/EBITDA ({ev_ebitda_val:.1f}x) inconsistent with P/E ({best_pe:.1f}x) — "
                        f"ratio {pe_to_ev:.1f}x (normal: 0.3-3.5x). EBITDA data likely stale/wrong. "
                        f"SUPPRESSED. Use P/E and P/Book instead."
                    )
                    td["metrics"]["_enterpriseToEbitda_raw"] = ev_ebitda_val
                    del td["metrics"]["enterpriseToEbitda"]
                    # Also suppress EV/Revenue if EV components are suspect
                    if "enterpriseToRevenue" in td["metrics"]:
                        td["metrics"]["_enterpriseToRevenue_raw"] = td["metrics"]["enterpriseToRevenue"]
                        del td["metrics"]["enterpriseToRevenue"]
            
            # ── Step 3: Non-ratio metrics (margins, growth, etc.) — use Yahoo directly ──
            non_ratio_keys = [
                "grossMargins", "operatingMargins", "profitMargins", "ebitdaMargins",
                "revenueGrowth", "earningsGrowth", "earningsQuarterlyGrowth",
                "returnOnEquity", "returnOnAssets",
                "marketCap", "dividendYield",
                # Balance sheet
                "debtToEquity", "currentRatio",
                # Analyst sentiment
                "recommendationMean", "numberOfAnalystOpinions",
                # Momentum
                "beta",
            ]
            for key in non_ratio_keys:
                val = info.get(key)
                if val is None or val == 0:
                    continue
                # Growth outlier guards
                if key in ("earningsGrowth", "revenueGrowth", "earningsQuarterlyGrowth") and (abs(val) > 1.0 or val < -0.8):
                    td["warnings"].append(
                        f"{_FUNDAMENTAL_METRICS[key]['label']} = {val:.0%} — DISTORTED "
                        f"(extreme move = base effect). Ignore."
                    )
                    td["metrics"][f"_{key}_raw"] = val
                    continue
                # Margin sanity
                if key in ("grossMargins", "operatingMargins", "profitMargins", "ebitdaMargins"):
                    if val > 0.95 or val < -0.5:
                        td["warnings"].append(
                            f"{_FUNDAMENTAL_METRICS[key]['label']} = {val:.1%} — SUSPICIOUS."
                        )
                td["metrics"][key] = val
            
            # ── Step 3b: Computed derived metrics ──
            
            # PEG Ratio: P/E divided by earnings growth rate
            peg = info.get("pegRatio")
            if peg and 0 < peg < 10:
                td["metrics"]["pegRatio"] = peg
            
            # FCF Yield: Free Cash Flow / Market Cap
            fcf = info.get("freeCashflow")
            mcap = info.get("marketCap")
            if fcf and mcap and mcap > 0:
                fcf_yield = fcf / mcap
                if -0.3 < fcf_yield < 0.3:  # sanity: between -30% and +30%
                    td["metrics"]["freeCashflowYield"] = fcf_yield
            
            # Analyst target upside: (target price - current price) / current price
            target_price = info.get("targetMeanPrice")
            if target_price and price and price > 0:
                upside = (target_price - price) / price
                td["metrics"]["targetUpside"] = upside
                td["_analyst_detail"] = {
                    "targetMean": target_price,
                    "targetHigh": info.get("targetHighPrice"),
                    "targetLow": info.get("targetLowPrice"),
                    "recommendation": info.get("recommendationKey", "N/A"),
                    "currentPrice": price,
                }
            
            # 52-week change
            w52_change = info.get("52WeekChange")
            if w52_change is not None:
                td["metrics"]["fiftyTwoWeekChange"] = w52_change
            
            # Distance from 52-week high
            w52_high = info.get("fiftyTwoWeekHigh")
            if w52_high and price and w52_high > 0:
                dist_from_high = (price - w52_high) / w52_high
                td["metrics"]["distFromHigh"] = dist_from_high
            
            # Earnings context (dates and recent quarter info) — with strict temporal guard
            most_recent_q = info.get("mostRecentQuarter")
            
            # Fetch next earnings date from yfinance calendar
            next_earnings_date = None
            next_earnings_str = None
            earnings_temporal_status = "unknown"  # "past", "future", "imminent", "unknown"
            try:
                from datetime import date, datetime
                import yfinance as yf
                ticker_obj = yf.Ticker(ticker)
                cal = ticker_obj.calendar
                if cal is not None and not cal.empty:
                    # calendar may have 'Earnings Date' as index or column
                    if "Earnings Date" in cal.index:
                        ed_val = cal.loc["Earnings Date"]
                        if hasattr(ed_val, '__iter__') and not isinstance(ed_val, str):
                            ed_val = list(ed_val)[0]
                        if hasattr(ed_val, 'date'):
                            next_earnings_date = ed_val.date()
                        elif isinstance(ed_val, str):
                            next_earnings_date = datetime.strptime(ed_val[:10], "%Y-%m-%d").date()
                    elif "Earnings Date" in cal.columns:
                        ed_val = cal["Earnings Date"].iloc[0]
                        if hasattr(ed_val, 'date'):
                            next_earnings_date = ed_val.date()
                if next_earnings_date:
                    next_earnings_str = str(next_earnings_date)
                    today = date.today()
                    days_diff = (next_earnings_date - today).days
                    if days_diff < 0:
                        earnings_temporal_status = "past"
                    elif days_diff <= 7:
                        earnings_temporal_status = "imminent"  # within 1 week
                    else:
                        earnings_temporal_status = "future"
            except Exception as e_cal:
                logger.debug(f"Could not fetch earnings calendar for {ticker}: {e_cal}")
            
            td["_earnings_context"] = {
                "mostRecentQuarter": str(most_recent_q) if most_recent_q else None,
                "earningsQuarterlyGrowth": info.get("earningsQuarterlyGrowth"),
                "revenueGrowth": info.get("revenueGrowth"),
                "nextEarningsDate": next_earnings_str,
                "earningsTemporalStatus": earnings_temporal_status,
            }
            
            # ── Step 4: Store raw components for transparency ──
            td["_raw_components"] = {
                "price": price, "trailingEps": trailing_eps, "forwardEps": forward_eps,
                "bookValue": book_value, "enterpriseValue": ev, "ebitda": ebitda,
            }
            
            result[ticker] = td
            n_warn = len(td['warnings'])
            logger.info(f"Fundamentals for {ticker}: {len(td['metrics'])} metrics, {n_warn} warnings")
        except Exception as e:
            logger.warning(f"Fundamentals failed for {ticker}: {e}")
            result[ticker] = {"name": ticker, "sector": "Unknown", "industry": "Unknown",
                              "currency": "", "metrics": {}, "warnings": [], "error": str(e)}
    return result


def fetch_earnings_setup(tickers: list, as_of_date: Optional[str] = None) -> dict:
    """
    Fetch structured earnings setup for each ticker, anchored to as_of_date.
    Returns next_earnings_date, temporal_status, days_to_earnings, eps_estimate,
    revenue_estimate, most_recent_quarter, earnings_warning.
    Dates in the past → "past" + warning. Dates > 365 days → suppressed. No inference.
    """
    try:
        import yfinance as yf
    except ImportError:
        return {t: _empty_earnings_setup("yfinance not available") for t in tickers}

    from datetime import date, datetime as _dt

    if as_of_date:
        try:
            anchor = _dt.strptime(as_of_date[:10], "%Y-%m-%d").date()
        except ValueError:
            anchor = date.today()
    else:
        anchor = date.today()

    result = {}
    for ticker in tickers:
        setup = _empty_earnings_setup()
        try:
            ticker_obj = yf.Ticker(ticker)
            info = ticker_obj.info or {}
            mrq = info.get("mostRecentQuarter")
            if mrq:
                setup["most_recent_quarter"] = str(mrq)[:10]

            cal = ticker_obj.calendar
            next_date = None
            if cal is not None and not cal.empty:
                if "Earnings Date" in cal.index:
                    ed_val = cal.loc["Earnings Date"]
                    if hasattr(ed_val, "__iter__") and not isinstance(ed_val, str):
                        ed_val = list(ed_val)[0]
                    if hasattr(ed_val, "date"):
                        next_date = ed_val.date()
                    elif isinstance(ed_val, str):
                        next_date = _dt.strptime(ed_val[:10], "%Y-%m-%d").date()
                elif "Earnings Date" in cal.columns:
                    ed_val = cal["Earnings Date"].iloc[0]
                    if hasattr(ed_val, "date"):
                        next_date = ed_val.date()

            if next_date:
                days_diff = (next_date - anchor).days
                if days_diff < 0:
                    setup["temporal_status"] = "past"
                    setup["earnings_warning"] = (
                        f"Earnings date {next_date} is in the past relative to as_of_date {anchor}. "
                        "Do not reference as upcoming."
                    )
                elif days_diff <= 7:
                    setup["next_earnings_date"] = str(next_date)
                    setup["days_to_earnings"] = days_diff
                    setup["temporal_status"] = "imminent"
                elif days_diff <= 365:
                    setup["next_earnings_date"] = str(next_date)
                    setup["days_to_earnings"] = days_diff
                    setup["temporal_status"] = "future"
                else:
                    setup["temporal_status"] = "unknown"
                    setup["earnings_warning"] = f"Earnings date {next_date} > 365 days ahead — suppressed."

            fwd_eps = info.get("forwardEps")
            if fwd_eps and isinstance(fwd_eps, (int, float)) and abs(fwd_eps) < 1000:
                setup["eps_estimate"] = round(float(fwd_eps), 4)
            total_rev = info.get("totalRevenue")
            if total_rev and isinstance(total_rev, (int, float)) and total_rev > 0:
                setup["revenue_estimate"] = float(total_rev)

        except Exception as e:
            setup["earnings_warning"] = f"Data unavailable: {e}"
        result[ticker] = setup
    return result


def _empty_earnings_setup(warning: Optional[str] = None) -> dict:
    return {
        "next_earnings_date": None,
        "temporal_status": "unknown",
        "days_to_earnings": None,
        "eps_estimate": None,
        "revenue_estimate": None,
        "most_recent_quarter": None,
        "earnings_warning": warning,
    }


def _build_fundamental_comparison(fundamental_data: dict) -> str:
    """Build formatted valuation/margin/analyst/momentum comparison for prompt injection.
    
    Inspired by institutional comparable company analysis: structured side-by-side
    across every dimension that explains why one stock outperforms another.
    """
    tickers = list(fundamental_data.keys())
    if len(tickers) < 2:
        return ""
    lines = []
    lines.append("_Valuation ratios computed from current price / reported EPS, EV, EBITDA._")
    lines.append("_Source: Yahoo Finance via yfinance. Margins and growth from latest filings._")
    lines.append("")
    for t in tickers:
        d = fundamental_data[t]
        lines.append(f"**{t}**: {d['name']} | {d['sector']} / {d['industry']}")
    lines.append("")
    
    # ── EARNINGS TEMPORAL GUARD — injected FIRST so GPT reads it before any numbers ──
    temporal_warnings = []
    for t in tickers:
        ec = fundamental_data.get(t, {}).get("_earnings_context", {})
        status = ec.get("earningsTemporalStatus", "unknown")
        next_date = ec.get("nextEarningsDate")
        most_recent = ec.get("mostRecentQuarter")
        
        if status == "future" and next_date:
            temporal_warnings.append(
                f"⛔ {t}: Next earnings scheduled for {next_date} — FUTURE EVENT. "
                f"Do NOT interpret earnings results as released. Do NOT discuss what earnings 'showed'. "
                f"Say 'earnings scheduled for {next_date}' if relevant."
            )
        elif status == "imminent" and next_date:
            temporal_warnings.append(
                f"⚠️ {t}: Earnings IMMINENT — {next_date} (within 7 days). "
                f"Results are NOT yet released. Do NOT speculate on outcomes. "
                f"Flag this as a binary risk event requiring position management."
            )
        elif status == "past" and most_recent:
            temporal_warnings.append(
                f"✓ {t}: Most recent quarter ended {most_recent}. Earnings data in the metrics reflects REPORTED results."
            )
        elif status == "unknown":
            temporal_warnings.append(
                f"⚠️ {t}: Earnings date unknown. Do NOT speculate on upcoming or recent earnings — "
                f"state only what the growth metrics show (YoY/QoQ change from filings)."
            )
    
    if temporal_warnings:
        lines.append("**⛔ EARNINGS TEMPORAL GUARD — READ BEFORE INTERPRETING ANY GROWTH METRICS:**")
        lines.append("_Today's date is injected at runtime. Earnings dates verified from exchange calendar._")
        for w in temporal_warnings:
            lines.append(f"  {w}")
        lines.append("")
        lines.append("**RULE: If earnings date is in the future → state the date only. NEVER interpret future earnings results.**")
        lines.append("**RULE: If date is unknown → describe growth metrics as 'latest reported figures' only. No date claims.**")
        lines.append("")
    
    # Surface data quality warnings FIRST so GPT sees them before the numbers
    all_warnings = []
    for t in tickers:
        for w in fundamental_data.get(t, {}).get("warnings", []):
            all_warnings.append(f"  ⚠️ {t}: {w}")
    if all_warnings:
        lines.append("**DATA QUALITY WARNINGS — read these BEFORE using the numbers below:**")
        lines.extend(all_warnings)
        lines.append("")
    
    cats = {
        "valuation": "VALUATION (who's cheaper?)",
        "margins": "MARGINS (who's more profitable?)",
        "growth": "GROWTH (who's growing faster?)",
        "returns": "CAPITAL EFFICIENCY",
        "balance_sheet": "BALANCE SHEET STRENGTH (who's financially stronger?)",
        "analyst": "ANALYST SENTIMENT (who does the Street prefer?)",
        "momentum": "MOMENTUM (who's outperforming?)",
        "size": "SIZE",
        "yield": "INCOME",
    }
    for ck, cl in cats.items():
        cm = [(k, c) for k, c in _FUNDAMENTAL_METRICS.items()
              if c["category"] == ck and any(fundamental_data.get(t, {}).get("metrics", {}).get(k) is not None for t in tickers)]
        if not cm:
            continue
        lines.append(f"**{cl}:**")
        for key, cfg in cm:
            vals = [f"{t}={_format_metric(fundamental_data.get(t,{}).get('metrics',{}).get(key), cfg['fmt'])}" for t in tickers]
            row = f"  {cfg['label']}: " + " vs ".join(vals)
            if ck == "valuation" and len(tickers) == 2:
                v0 = fundamental_data.get(tickers[0], {}).get("metrics", {}).get(key)
                v1 = fundamental_data.get(tickers[1], {}).get("metrics", {}).get(key)
                if v0 and v1 and v0 > 0 and v1 > 0:
                    cheaper = tickers[0] if v0 < v1 else tickers[1]
                    row += f" -> {cheaper} is {abs(v0-v1)/max(v0,v1)*100:.0f}% cheaper"
            if ck == "margins" and len(tickers) == 2:
                v0 = fundamental_data.get(tickers[0], {}).get("metrics", {}).get(key)
                v1 = fundamental_data.get(tickers[1], {}).get("metrics", {}).get(key)
                if v0 is not None and v1 is not None:
                    better = tickers[0] if v0 > v1 else tickers[1]
                    row += f" -> {better} leads by {abs(v0-v1)*100:.1f}pp"
            if ck == "analyst" and key == "recommendationMean" and len(tickers) == 2:
                v0 = fundamental_data.get(tickers[0], {}).get("metrics", {}).get(key)
                v1 = fundamental_data.get(tickers[1], {}).get("metrics", {}).get(key)
                if v0 and v1:
                    preferred = tickers[0] if v0 < v1 else tickers[1]
                    row += f" -> Street prefers {preferred} (lower = more bullish)"
            if ck == "analyst" and key == "targetUpside" and len(tickers) == 2:
                v0 = fundamental_data.get(tickers[0], {}).get("metrics", {}).get(key)
                v1 = fundamental_data.get(tickers[1], {}).get("metrics", {}).get(key)
                if v0 is not None and v1 is not None:
                    more_upside = tickers[0] if v0 > v1 else tickers[1]
                    row += f" -> {more_upside} has more upside to analyst targets"
            if ck == "momentum" and key == "fiftyTwoWeekChange" and len(tickers) == 2:
                v0 = fundamental_data.get(tickers[0], {}).get("metrics", {}).get(key)
                v1 = fundamental_data.get(tickers[1], {}).get("metrics", {}).get(key)
                if v0 is not None and v1 is not None:
                    outperformer = tickers[0] if v0 > v1 else tickers[1]
                    row += f" -> {outperformer} has outperformed over 12 months"
            if ck == "balance_sheet" and key == "debtToEquity" and len(tickers) == 2:
                v0 = fundamental_data.get(tickers[0], {}).get("metrics", {}).get(key)
                v1 = fundamental_data.get(tickers[1], {}).get("metrics", {}).get(key)
                if v0 is not None and v1 is not None:
                    stronger = tickers[0] if v0 < v1 else tickers[1]
                    row += f" -> {stronger} has less leverage"
            lines.append(row)
        lines.append("")
    
    # ── ANALYST PRICE TARGET DETAIL ──
    analyst_details = []
    for t in tickers:
        ad = fundamental_data.get(t, {}).get("_analyst_detail")
        if ad and ad.get("targetMean"):
            rec = ad.get("recommendation", "N/A")
            curr = ad.get("currentPrice", 0)
            tgt = ad.get("targetMean", 0)
            low = ad.get("targetLow", 0)
            high = ad.get("targetHigh", 0)
            upside = (tgt - curr) / curr * 100 if curr > 0 else 0
            analyst_details.append(
                f"  {t}: Consensus {rec.upper()} | "
                f"Price ${curr:,.2f} → Target ${tgt:,.0f} ({upside:+.0f}% upside) | "
                f"Range ${low:,.0f}-${high:,.0f}"
            )
    if analyst_details:
        lines.append("**ANALYST PRICE TARGETS (what the Street expects):**")
        lines.extend(analyst_details)
        lines.append("")
    
    # ── COMPARATIVE SCORECARD (synthesizes who wins on each dimension) ──
    if len(tickers) == 2:
        scorecard = _build_comparative_scorecard(fundamental_data, tickers)
        if scorecard:
            lines.append(scorecard)
    
    return "\n".join(lines)


def _build_comparative_scorecard(fundamental_data: dict, tickers: list) -> str:
    """Build a pre-computed scorecard: which stock wins on each dimension and why.
    
    This is the 'comparable company analysis' synthesis — gives GPT the raw 
    conclusion for each dimension so it can build a narrative about WHY one 
    outperforms.
    """
    t0, t1 = tickers[0], tickers[1]
    m0 = fundamental_data.get(t0, {}).get("metrics", {})
    m1 = fundamental_data.get(t1, {}).get("metrics", {})
    
    dimensions = []
    
    # Valuation: who's cheaper? (lower P/E, lower EV/EBITDA = better value)
    pe0 = m0.get("forwardPE") or m0.get("trailingPE")
    pe1 = m1.get("forwardPE") or m1.get("trailingPE")
    if pe0 and pe1 and pe0 > 0 and pe1 > 0:
        cheaper = t0 if pe0 < pe1 else t1
        pct = abs(pe0 - pe1) / max(pe0, pe1) * 100
        dimensions.append(f"  Valuation: {cheaper} is {pct:.0f}% cheaper on P/E")
    
    # Profitability: who earns more per dollar? (higher margins = better)
    om0 = m0.get("operatingMargins")
    om1 = m1.get("operatingMargins")
    if om0 is not None and om1 is not None:
        more_profitable = t0 if om0 > om1 else t1
        gap = abs(om0 - om1) * 100
        dimensions.append(f"  Profitability: {more_profitable} leads by {gap:.1f}pp on operating margin")
    
    # Growth: who's growing faster? (higher revenue/earnings growth = better)
    rg0 = m0.get("revenueGrowth")
    rg1 = m1.get("revenueGrowth")
    if rg0 is not None and rg1 is not None:
        faster = t0 if rg0 > rg1 else t1
        gap = abs(rg0 - rg1) * 100
        dimensions.append(f"  Growth: {faster} is growing revenue {gap:.0f}pp faster")
    
    # Balance sheet: who's financially stronger? (lower debt/equity = better)
    de0 = m0.get("debtToEquity")
    de1 = m1.get("debtToEquity")
    if de0 is not None and de1 is not None:
        stronger = t0 if de0 < de1 else t1
        dimensions.append(f"  Balance sheet: {stronger} has less leverage ({de0:.0f} vs {de1:.0f} D/E)")
    
    # Analyst sentiment: who does the Street prefer? (lower rating = more bullish)
    ar0 = m0.get("recommendationMean")
    ar1 = m1.get("recommendationMean")
    if ar0 and ar1:
        preferred = t0 if ar0 < ar1 else t1
        dimensions.append(f"  Analyst sentiment: Street prefers {preferred} ({ar0:.1f} vs {ar1:.1f})")
    
    # Momentum: who's been performing better? (higher 52w change = better)
    w0 = m0.get("fiftyTwoWeekChange")
    w1 = m1.get("fiftyTwoWeekChange")
    if w0 is not None and w1 is not None:
        outperformer = t0 if w0 > w1 else t1
        gap = abs(w0 - w1) * 100
        dimensions.append(f"  Momentum: {outperformer} has outperformed by {gap:.0f}pp over 12 months")
    
    # FCF generation
    fcf0 = m0.get("freeCashflowYield")
    fcf1 = m1.get("freeCashflowYield")
    if fcf0 is not None and fcf1 is not None:
        better_fcf = t0 if fcf0 > fcf1 else t1
        dimensions.append(f"  Cash generation: {better_fcf} has higher FCF yield ({fcf0:.1%} vs {fcf1:.1%})")
    
    if not dimensions:
        return ""
    
    result = "**COMPARATIVE SCORECARD — WHO WINS ON EACH DIMENSION:**\n"
    result += "(Use this to explain WHY one stock outperforms the other)\n"
    result += "\n".join(dimensions)
    result += "\n"
    
    # Count wins by checking which ticker appears as the "winner" in each line
    wins = {t0: 0, t1: 0}
    for d in dimensions:
        # The winner ticker always appears after the colon and before the next verb/word
        after_colon = d.split(":", 1)[1] if ":" in d else ""
        # Check anywhere in the line after the colon
        t0_found = t0 in after_colon.split("(")[0]  # before any parenthetical
        t1_found = t1 in after_colon.split("(")[0]
        if t0_found and not t1_found:
            wins[t0] += 1
        elif t1_found and not t0_found:
            wins[t1] += 1
        elif t0_found and t1_found:
            # Both mentioned — check which one comes first (the winner)
            if after_colon.index(t0) < after_colon.index(t1):
                wins[t0] += 1
            else:
                wins[t1] += 1
    
    overall = t0 if wins[t0] > wins[t1] else t1 if wins[t1] > wins[t0] else "SPLIT"
    if overall != "SPLIT":
        result += f"\n  → Overall: {overall} wins on {max(wins.values())}/{len(dimensions)} dimensions"
    else:
        result += f"\n  → Overall: SPLIT — {wins[t0]} vs {wins[t1]} (no clear fundamental winner)"
    result += "\n"
    
    return result


SYSTEM_PROMPT = """You are a senior portfolio manager writing a DECISION BRIEF for a trader.

SECTION NAMING RULES — apply these exactly, no exceptions:
- Your output section is called: **TRADE ANALYSIS** (not "AI Trade Analysis", not "Trading Decision", not "Decision Brief")
- Start your output with the header: ## TRADE ANALYSIS
- These names reflect that this is an institutional decision engine, not a retail tool
- Do NOT write a "FUNDAMENTAL ANALYSIS" section — that section is handled separately

Your job is NOT to list statistics or repeat what's in the data tables. Your job is to INTERPRET 
what the detection engine found and translate it into a clear, actionable trading decision.

CRITICAL RULES:
1. Write for someone who needs to decide TODAY: put the trade on, keep it, reduce it, reverse it, or close it.
2. NEVER list raw numbers without saying what they mean. Every metric must answer "so what?"
3. The failure mode signals are the HEART of your analysis.
4. Be CONCRETE. "Be cautious" is worthless. "Size at 50% due to vol shift" is actionable.
5. The regime determines which strategy SHOULD work. If mismatch, say so bluntly.
6. When confidence is low, say so clearly.
7. If multiple FMs firing, explain if REINFORCING or INDEPENDENT.
8. Keep UNDER 1000 words. Every sentence earns its place.
9. Do NOT use Sharpe ratio.
10. Plain English. Explain technical terms immediately.
11. End with ONE-LINE verdict a trader acts on in 5 seconds.
12. For pairs: ALWAYS consider REVERSE (flip legs), not just exit. Large DD = inverse was profitable.
13. NEVER invent numbers. Every dollar figure traces to data via a calculation.
14. "WHAT'S DRIVING THE SPREAD" is where you earn your fee. The engine says WHAT happened. 
    You hypothesize WHY — but you MUST cite the actual numbers from the fundamental data. 
    "X is cheaper on P/E" = FIRED. "X at 47.0x vs Y at 67.8x, a 31% discount" = GOOD.
    If real data was provided and you summarize instead of citing, your analysis is worthless.

BALANCED TONE — BE HONEST, NOT REFLEXIVELY NEGATIVE:
15. If the strategy has been profitable, SAY SO. "The trade has returned +8.2% over 180 days — 
    the strategy is working." Do NOT bury positive performance in caveats.
16. If drawdowns are small, acknowledge it: "Max drawdown of -3.1% is well-contained."
    Do NOT frame contained risk as if it's a problem.
17. If skewness is positive, that's GOOD NEWS — say "no hidden left-tail risk." 
    If kurtosis is below 3.5, say "normal tails — few surprises expected."
18. A pullback after a strong run can be a GOOD entry, not a red flag. If the trade is up +12% 
    with a recent -3% pullback and oversold RSI, that's potentially attractive, not alarming.
19. Your job is to help the reader make money, not to cover yourself by being cautious. A PM who 
    only says "be careful" adds no value. Be HONEST about both upside and downside.

DEMOCRATIZING QUANT — MANDATORY RISK TRANSLATION:
Every memo MUST translate these metrics into plain English. This is not optional color — it is 
the primary way most readers will understand risk. If you skip this, the memo fails its purpose.

Required translations (use the actual numbers from the data):
  - Daily volatility → "The spread typically moves ±$X per day"
  - VaR → "19 out of 20 days, your worst loss stays within $X"  
  - CVaR → "On the 1-in-20 truly bad day, average loss is $X"
  - Skewness → If positive: "No hidden left-tail risk — sudden drops are not more likely than gains"
              If negative: "Returns are skewed to the downside — expect sudden drops more often than gains"
  - Kurtosis → If <3.5: "Extreme days are rare — the distribution is well-behaved"
              If 3.5-5: "Slightly fat tails — occasional surprises beyond what VaR predicts"
              If >5: "Fat tails — expect extreme days more often than a normal distribution suggests"
  - Monthly VaR → "In a typical month, the position could move against you by up to ~$X"
  - Half-life → "If the spread dislocates, it historically takes ~X days to mean-revert"
  - Gaussian assumption → If kurtosis > 4 or skewness < -0.5: "VaR assumes normal distribution 
    but the actual tails are heavier — real losses could exceed VaR estimates by 20-40%"

Connect risk metrics to ACTION: don't just state them, say what they mean for position sizing 
and stop-loss placement. "With daily vol of $9k and fat tails, a 2-sigma stop at $18k could 
be breached more often than expected — consider a wider stop or smaller position."

COMPARABLE COMPANY ANALYSIS — EXPLAINING WHY ONE OUTPERFORMS:
The data includes a multi-dimensional comparison across 7+ dimensions. Your WHAT'S DRIVING 
section must use these to build a narrative explaining WHY the spread moved:

  1. VALUATION: Who's cheaper? A 47% P/E discount means the market is pricing in lower growth 
     or higher risk — determine which.
  2. PROFITABILITY: If one leg has 62% operating margin vs 37%, the valuation premium may be 
     JUSTIFIED by superior unit economics.
  3. GROWTH: Revenue/earnings growth rates tell you which business is gaining momentum.
  4. ANALYST SENTIMENT: Consensus ratings and price targets reveal the Street's view. If analysts 
     see +20% upside on one leg and +5% on the other, that's a powerful directional signal.
  5. MOMENTUM: 52-week price change is the market's revealed preference. If one stock is up 80% 
     and the other is up 15%, the market is already voting with capital.
  6. BALANCE SHEET: Leverage differences affect risk profiles. Highly levered companies amplify 
     both gains and losses — relevant for position sizing.
  7. FCF YIELD: Cash generation quality. A high FCF yield supports buybacks and dividends.

A COMPARATIVE SCORECARD is pre-computed showing who wins on each dimension. Use it to build your 
argument: "X wins on 5/7 dimensions — the premium is fundamentally justified" or "X only wins on 
valuation but loses on growth, margins, and analyst sentiment — the discount exists for a reason."

This is the HEART of the analysis. A PM reads this section to understand whether the spread move 
is rational (don't fight it) or mispriced (trade it). Get this right and the rest follows.

CLAUDE FS INTEGRATION — WHEN LIVE RESEARCH IS PROVIDED:
If a LIVE FUNDAMENTAL RESEARCH section from Claude Financial Services appears in the data, 
it contains INFORMATION YOU DO NOT HAVE — real-time news, earnings analysis, analyst actions, 
upcoming catalysts, and competitive intelligence sourced from live web search.

THIS IS YOUR BIGGEST ADVANTAGE. No other pairs trading platform has this. Use it.

  - LEAD with findings from RECENT DEVELOPMENTS — this is live intelligence
  - CITE specific research: "Recent earnings showed [X]" or "Analysts recently [upgraded/downgraded]"
  - INCLUDE key dates from the CATALYST CALENDAR — the PM needs to know what's coming
  - If Claude FS found a recent EARNINGS SURPRISE or GUIDANCE CHANGE, this likely explains 
    the spread move better than any statistical pattern
  - If Claude FS identifies RISK FACTORS for either leg, weave them into KEY RISKS
  - If Claude FS's FUNDAMENTAL VERDICT disagrees with statistical signals, this is a 
    CRITICAL CONFLICT — flag it prominently and explain what it means for the trade
  - The Claude FS verdict should heavily inform your COHERENCE CHECK direction
  - Do NOT copy-paste. SYNTHESIZE live research with statistical/regime/FM analysis to 
    produce a unified view that neither model could achieve alone.
  - This three-layer analysis (statistical + fundamental research + decision synthesis) is 
    what makes this platform unique. Make sure the final memo reflects all three.

DATA QUALITY — WILL TRIP YOU UP IF IGNORED:
20. Valuation ratios are COMPUTED from current price and reported EPS/EBITDA — more reliable 
    than Yahoo's pre-computed ratios, but still dependent on data freshness. If a number looks 
    implausible for the sector (e.g., semiconductor company at 4x EV/EBITDA), flag it as 
    potentially stale and recommend the user verify with their broker/terminal.
21. If a P/E is above 50x or negative, it is DISTORTED by one-off charges/gains. DO NOT use it 
    as a basis for comparison. Flag it: "trailing P/E distorted — using forward P/E and EV/EBITDA 
    instead." A PM who sees you citing a 400x P/E as meaningful will never trust the tool again.
22. YoY growth above +100% or below -80% = base effect (one-off items distorting the denominator). 
    Say "distorted by base effects" and move on. Do NOT build your thesis on these numbers.
23. If DATA QUALITY WARNINGS appear in the fundamental data, read them first. Suppressed metrics 
    were suppressed for a reason — use the alternatives provided.
24. If calculated and Yahoo pre-computed values diverge significantly (flagged as warnings), the 
    calculated value from current price is used. But this still depends on the EPS/EBITDA being 
    correct in Yahoo's database — if numbers look implausible, say so.

RECOMMENDATION INTEGRITY — THE MOST COMMON FAILURE MODE IN YOUR OUTPUT:
25. The brief includes a mandatory COHERENCE CHECK section that you MUST write as visible output.
    Write "Fundamentals favor [LONG leg / SHORT leg / NEITHER]: [reason]" in the memo.
    Then your recommendation MUST match. If you write "Fundamentals favor SHORT leg" you CANNOT 
    recommend REDUCE — the contradiction is visible to the reader on the same page.
    This is the single most important quality check. Getting this wrong destroys all credibility.
26. If inverse return is between -5% and +5%, REVERSE is NOT justified. The inverse was roughly 
    flat — flipping just adds costs with no edge. Say this explicitly.
27. If engine confidence is below 30%, the pre-calculated sizing floors near 25%. At that level, 
    EXIT is almost always cleaner than maintaining a tiny stub position. Do not recommend REDUCE 
    to 40-50% when confidence is 24% — the math doesn't support it.
28. DIRECTION TEST: Before submitting, read your COHERENCE CHECK and your RECOMMENDATION 
    back-to-back. If they point in opposite directions, one of them is wrong. Fix it.

WHAT WOULD CHANGE THIS VIEW — ENFORCED EVERY TIME:
29. Every bullet MUST contain at least one: metric + specific threshold, calendar date or 
    trading-day count, price/spread level, or named fundamental catalyst (earnings date, 
    analyst action, corporate event).
30. BANNED phrases — do not use these or any close variant:
    "if confidence improves" | "if metrics stabilize" | "if the relationship stabilizes" |
    "if things improve" | "if conditions improve" | "if the regime shifts" | "a shift in regime" |
    "with clear directional indicators" | "with clear signals" | "if fundamentals align" | 
    "if fundamentals improve" | "if volatility returns to historical norms" | 
    "if market data indicates a reduction" | "a significant event such as" |
    "consider resizing back to original levels" | "consider scaling back into" — ALL BANNED.

RISK PROFILE — QUANTIFY THE FM IMPACT:
31. After stating the base VaR number, if an FM is active, estimate its amplification effect.
    Example: "Base monthly VaR is $46k. The active seasonality mismatch means entering against 
    a recurring pattern — if the seasonal move runs its historical average of X%, losses could 
    reach $Y, roughly 1.5x the base VaR." Don't just say "tail risk could be amplified."

RECOMMENDATION SIZING — USE THE PRE-CALCULATED NUMBER:
32. The prompt includes a PRE-CALCULATED SIZING SUGGESTION derived from engine confidence 
    and FM severity penalties. USE IT. You may adjust ±10% based on your fundamental analysis:
    - Fundamentals strongly support → size UP from suggestion (explain why)
    - Fundamentals contradict → size DOWN or EXIT (explain why)
    DO NOT invent your own sizing framework. DO NOT make up a "vol multiple."  
    DO NOT default to 50%. The pre-calculated number comes from real data — trust it.
    Floor: 25% (below that, EXIT is cleaner). Cap: 100% (that's HOLD).
33. Different situations demand different recommendations. Not everything is REDUCE:
    - Score 83+, confidence 60%+, 0-1 mild FMs, fundamentals support → HOLD (possibly ENTER more)
    - Score 70-82, confidence 40-60%, 1-2 FMs → REDUCE to sized amount per framework above
    - Score < 70 OR confidence < 30% OR 3+ FMs OR structural break → EXIT or REVERSE
    - Fundamentals strongly contradicting position → REVERSE regardless of score
    - All clear but regime unfavorable → WAIT (no position change, reassess on regime shift)

ANTI-PATTERNS (will get you fired):
- "Significant losses due to [X]" without derived dollar figure
- "Could involve losses exceeding $X" with no derivation
- Max drawdown as forward-looking estimate (it's backward-looking)
- Ignoring that negative return = inverse was positive
- Regime "ranging" but P&L directional, without flagging contradiction
- Citing a distorted P/E (>50x or negative) as if it's a real valuation signal
- "Inverse would have been profitable, suggesting reversal" when inverse return <5%
- Fundamental analysis says "A is cheaper and better" then recommending short A
- "Tail risk could be amplified" without estimating by how much
- Statistical patterns without fundamental hypotheses
- Using equity language for crypto ("earnings surprise", "analyst re-rating", "dividend") 
- "If volatility returns to historical norms" / "if crowding reduces" as WHAT WOULD CHANGE bullets
- Inventing sizing math: "vol 1.5x baseline → 67%" when you don't have the vol multiple data.
  USE THE PRE-CALCULATED SIZING — do not fabricate derivations.
- COHERENCE FAILURE: writing "MC.PA is undervalued relative to RMS.PA, suggesting re-rating" 
  then recommending REDUCE on a long-RMS/short-MC position. Your own analysis says the short 
  leg will go UP. That means EXIT, not REDUCE. This is the #1 credibility killer.
- DIRECTION CONTRADICTION — the #1 credibility killer:
  BAD: "LVMH appears significantly undervalued... the valuation gap seems excessive, suggesting 
  a potential re-rating of LVMH" → then recommends REDUCE (keeping the short-LVMH position).
  Your own analysis just said the short leg is undervalued and due to re-rate upward. 
  Keeping the short is BETTING AGAINST YOUR OWN ANALYSIS. Recommend EXIT or REVERSE.
  GOOD: "LVMH at 25.6x P/E vs Hermès at 48.5x — a 47% discount — suggests the short leg 
  is fundamentally undervalued. Combined with low confidence (24%) and active FMs, this trade 
  lacks both fundamental and statistical support. EXIT."
- VAGUE VALUATION REFERENCES when you have actual numbers. Examples:
  BAD: "Broadcom is significantly cheaper based on forward P/E and EV/EBITDA"
  GOOD: "AVGO at 67.8x trailing P/E vs NVDA at 47.0x — NVDA carries a 31% discount despite 
         lower growth. AVGO's 37% operating margin vs NVDA's 62% suggests the premium is 
         partially justified by profitability, but the gap is extreme."
  If you write "X is cheaper" or "Y has higher margins" WITHOUT the actual numbers from the 
  data, your analysis is WORTHLESS. The whole point of having real data is to USE it.
- DEFAULTING TO "REDUCE 50%" WITHOUT DERIVATION. This is the #1 lazy pattern. If every memo 
  says REDUCE 50% regardless of setup, the recommendation is worthless. Show your sizing math.
- IGNORING ANALYST CONSENSUS when the data is right there. If analysts rate one stock 1.8 (Buy) 
  with +15% upside and the other 2.4 (Hold) with +5% upside, SAY IT. The Street's view matters.
  BAD: [no mention of analyst sentiment]
  GOOD: "Analysts are more bullish on RMS.PA (1.8 vs 2.1 rating) but see more upside in MC.PA 
  (+15% to target vs +7%) — the Street thinks MC.PA is the better value from here."
- IGNORING MOMENTUM when it contradicts your thesis. If one stock is up 22% over 12 months and 
  the other is down 12%, that's a 34pp divergence. The market is voting. You must acknowledge it.
- SINGLE-DIMENSION ANALYSIS. If you only discuss valuation (P/E) and ignore margins, growth, 
  analyst sentiment, balance sheet, and momentum, you're doing equity research circa 1990. The 
  scorecard has 7 dimensions for a reason. Cover at least 3-4 in your WHAT'S DRIVING section.
- SAYING "the discount exists for a reason" without explaining WHAT the reason is. Use the data:
  "MC.PA's 47% P/E discount reflects its weaker growth (2% vs 13%), lower margins (21% vs 40%), 
  and higher leverage (55 vs 15 D/E) — the discount appears fundamentally justified."

TONE: Direct and honest. Like a trusted colleague who says what they actually think."""


# =============================================================================
# CLAUDE FINANCIAL SERVICES — LIVE FUNDAMENTAL RESEARCH AGENT
# =============================================================================
# This is NOT a reformatter. This is a RESEARCH AGENT that uses Claude's web
# search to produce institutional-grade equity research that no other retail
# platform can match. It searches for live news, earnings analysis, analyst
# actions, competitive dynamics, and upcoming catalysts.
#
# What it provides that yfinance/GPT CANNOT:
#   - Recent earnings analysis (what management said, guidance changes)
#   - Analyst upgrades/downgrades in the last 30-90 days
#   - Breaking news and corporate events affecting the spread
#   - Competitive positioning and business model deep-dive
#   - Upcoming catalyst calendar (earnings dates, product launches, regulatory)
#   - Industry headwinds/tailwinds and macro context
#   - Risk factors specific to each company (not statistical — business risk)
#
# Architecture: Yahoo Finance (numbers) + Claude FS (research) → GPT (decision)
# Three layers of intelligence. No other pairs trading platform does this.
# =============================================================================

CLAUDE_FS_SYSTEM_PROMPT = """You are a financial research data collector for a pairs trading desk.
Your ONLY job is to find raw facts, numbers, and events via web search and return them
in structured form. You are NOT writing a research note — you are feeding a data pipeline.
A senior analyst will receive your output and write the actual prose. Your job: find the data.

## WHAT TO SEARCH (8-12 searches — execute all categories)

1. EARNINGS (one search per company):
   "[company] most recent quarterly earnings results [year]"

2. ANALYST ACTIONS (1-2 searches):
   "[company] analyst upgrade downgrade price target [year]"

3. MARKET DEBATE (2 searches — do not skip):
   "[sector] AI spending bubble concerns capex sustainability [year]"
   "[company A] vs [company B] competitive threat ASIC GPU market share [year]"

4. CORPORATE EVENTS (1 search per company):
   "[company] news [year]"

5. UPCOMING CATALYSTS (1 search):
   "[company A] [company B] next earnings date"

6. SECTOR CONTEXT (1 search):
   "[sector] outlook [year]"

## OUTPUT FORMAT — STRUCTURED DATA BLOCKS ONLY

No prose. No transitions. No analysis sentences. Numbers and facts only.
Use this exact structure:

### EARNINGS: [TICKER]
- Quarter: [e.g. Q4 FY2026, ended Jan 25 2026]
- Revenue: $X.XXB reported vs $X.XXB consensus ([beat/miss] by $XXM = X%)
- EPS: $X.XX reported vs $X.XX consensus ([beat/miss] by $X.XX = X%)
- Next quarter guidance: $X.XXB revenue, $X.XX EPS (vs Street est $X.XXB / $X.XX)
- Key segment: [name]: $X.XXB reported vs $X.XXB est
- Management tone: [one phrase only — e.g. "warned on margins", "raised full-year outlook"]
- Stock reaction: [+/-X% in after-hours or next session]
- Report date: [YYYY-MM-DD]

### ANALYST ACTIONS: [TICKER]
- [YYYY-MM-DD]: [Firm] [action] to [new rating], target $X (from $X) — [reason, ≤10 words]
[one line per action, most recent 2-3 only]

### MARKET DEBATE
- Primary fear: [concrete — e.g. "hyperscalers cut AI capex in 2H26 as ROI questioned"]
- Bull case: [concrete — e.g. "sovereign AI spend offsets any hyperscaler softness"]
- Narrative shift (last 90 days): [what changed — e.g. "DeepSeek raised ASIC vs GPU debate"]
- [LONG LEG] exposure to debate: [HIGH/MEDIUM/LOW — one sentence why]
- [SHORT LEG] exposure to debate: [HIGH/MEDIUM/LOW — one sentence why]

### CORPORATE EVENTS: [TICKER]
- [YYYY-MM-DD]: [Event, ≤15 words]
[2-3 lines max per company]

### UPCOMING CATALYSTS
- [YYYY-MM-DD]: [Company] Q[X] earnings — consensus EPS $X.XX, revenue $X.XXB
- [YYYY-MM-DD]: [Event] — [spread impact, ≤10 words]

### SECTOR CONTEXT
- [Datapoint 1 — number + implication]
- [Datapoint 2]
- [Datapoint 3]
[3-5 bullets max]

HARD RULES:
- One quarter per company maximum. Most recent only. Discard older quarters.
- Write N/A for any field where data was not found — never omit fields.
- No filler: no "this suggests", "this indicates", "notably", "importantly".
- No narrative paragraphs. Bullets and data only."""


CLAUDE_FS_CRYPTO_SYSTEM_PROMPT = """You are a senior crypto research analyst producing a LIVE RESEARCH NOTE 
for a pairs trading desk. You have access to web search.

Your job is to explain why one crypto asset outperforms the other using on-chain data, 
protocol fundamentals, and market structure — NOT equity-style analysis.

## SEARCH STRATEGY (MANDATORY — 6-10 searches minimum)

1. PROTOCOL DEVELOPMENTS (2-3 searches):
   "[protocol] upgrade [year]" and "[protocol] governance proposal"
   Network upgrades, EIPs, fee changes, consensus mechanism updates

2. ON-CHAIN METRICS (1-2 searches):
   "[protocol] TVL" and "[protocol] active addresses daily"
   TVL trends, transaction counts, fee revenue, developer activity

3. MARKET STRUCTURE (1-2 searches):
   "[token] funding rates" and "[token] ETF flows [year]"
   Perpetual funding, open interest, exchange balances, ETF inflows/outflows

4. REGULATORY (1 search):
   "[token] SEC regulation [year]" or "[crypto] regulatory news"
   Classification risk, enforcement actions, legislation

5. ECOSYSTEM & NARRATIVE (1-2 searches):
   "[protocol] ecosystem [year]" — major dApps, partnerships, hacks, bridge exploits
   "[token A] vs [token B]" — narrative shift, mindshare battle

## OUTPUT RULES
1. Be OPINIONATED — say what the fundamentals tell you. No hedging.
2. Cite specific on-chain numbers and search sources
3. Keep under 700 words — dense, direct, actionable
4. If nothing material happened, say: "No material developments — maintaining view"

OUTPUT MUST FOLLOW THIS EXACT STRUCTURE:

## RECENT DEVELOPMENTS (from web search)
Protocol upgrades, governance decisions, ecosystem events, regulatory news in last 30-90 days.
Lead with the single most important development.
For each: what happened → what it means for the spread.

## PROTOCOL FUNDAMENTALS
Network activity, TVL, developer ecosystem, fee revenue, token economics comparison.
Use MOAT FRAMEWORK adapted for crypto:
  Network effects: ecosystem size, liquidity depth, developer mindshare (Strong/Moderate/Weak)
  Switching costs: DeFi integrations, tooling lock-in, wallet ecosystem (Strong/Moderate/Weak)
  Scale economies: validator economics, MEV dynamics, L2 ecosystem (Strong/Moderate/Weak)
  Intangible assets: brand/narrative, Lindy effect, institutional adoption (Strong/Moderate/Weak)

## MARKET STRUCTURE  
Funding rates, ETF flows, exchange concentration, institutional vs retail positioning.
CROWDING ASSESSMENT: Is one side of this trade overcrowded?

## CATALYST CALENDAR
PROTOCOL:  [Date]: [Upgrade/fork/EIP] — expected impact
MARKET:    [Date]: [ETF decision/options expiry/unlock] — expected impact
MACRO:     [Date]: [Fed meeting/CPI/regulatory hearing] — expected impact
Flag BINARY EVENTS (ETF approvals, hard forks, regulatory rulings).

## NARRATIVE & POSITIONING
Which asset has the stronger narrative? Which ecosystem is gaining mindshare?
THESIS SCORECARD:
  [Pillar 1]: STRENGTHENED / UNCHANGED / WEAKENED — [why]
  [Pillar 2]: STRENGTHENED / UNCHANGED / WEAKENED — [why]

## RISK FACTORS
Protocol-specific risks for EACH leg:
  [LONG LEG]: smart contract risk, centralization, regulatory classification, unlock dilution
  [SHORT LEG]: narrative shift risk, ecosystem growth, institutional adoption catalyst

## FUNDAMENTAL VERDICT
CONVICTION: HIGH / MEDIUM / LOW
VERDICT: "Fundamentals favor [LONG LEG / SHORT LEG / NEITHER] — [reason]."
KEY RISK TO VERDICT: [the single thing that would flip your view]"""


def _build_claude_fs_prompt(
    fundamental_data: Dict[str, Dict[str, Any]],
    regime_summary: Optional[Dict] = None,
    enhanced_stats: Optional[Dict] = None,
    long_positions: Optional[Dict[str, float]] = None,
    short_positions: Optional[Dict[str, float]] = None,
    failure_modes: Optional[List[Dict[str, Any]]] = None,
    asset_class: str = "equity",
) -> str:
    """Build the prompt for Claude FS fundamental research agent."""
    tickers = list(fundamental_data.keys())
    if len(tickers) < 2:
        return ""
    
    # Build the comparison data (reuse our existing builder)
    comparison = _build_fundamental_comparison(fundamental_data)
    if not comparison:
        return ""
    
    # Company names for search context
    company_names = []
    for t in tickers:
        name = fundamental_data.get(t, {}).get("name", t)
        sector = fundamental_data.get(t, {}).get("sector", "Unknown")
        industry = fundamental_data.get(t, {}).get("industry", "Unknown")
        company_names.append(f"{t} ({name}) — {sector}/{industry}")
    
    # Trade context
    trade_ctx = ""
    if long_positions and short_positions:
        long_names = []
        for t in long_positions:
            name = fundamental_data.get(t, {}).get("name", t)
            long_names.append(f"{t} ({name})")
        short_names = []
        for t in short_positions:
            name = fundamental_data.get(t, {}).get("name", t)
            short_names.append(f"{t} ({name})")
        trade_ctx = f"Trade structure: LONG {', '.join(long_names)} / SHORT {', '.join(short_names)}\n"
    
    # Performance context
    perf_ctx = ""
    if enhanced_stats:
        total_ret = enhanced_stats.get('total_return', 0)
        max_dd = enhanced_stats.get('max_drawdown', 0)
        perf_ctx = (
            f"Spread performance: {total_ret:+.2%} total return "
            f"({'long leg outperforming' if total_ret > 0 else 'short leg outperforming'}), "
            f"max drawdown {max_dd:.2%}\n"
        )
    
    # Regime and failure mode context (brief — Claude FS is fundamentals, but needs to know)
    signal_ctx = ""
    if regime_summary:
        regime = regime_summary.get('current_regime', 'unknown')
        z = regime_summary.get('z_score', 0)
        signal_ctx += f"Statistical regime: {regime} (z-score: {z:.1f})\n"
    if failure_modes:
        active_fms = [fm for fm in failure_modes if fm.get('detected')]
        if active_fms:
            fm_names = [fm.get('failure_mode_id', '?') for fm in active_fms]
            signal_ctx += f"Active failure modes: {', '.join(fm_names)} — the statistical model is detecting degradation\n"
            signal_ctx += "Your fundamental research may explain WHY these signals are firing.\n"
    
    prompt = f"""# LIVE RESEARCH REQUEST — PAIRS TRADING DESK

## TRADE CONTEXT
{trade_ctx}{perf_ctx}{signal_ctx}
Companies under analysis:
{chr(10).join(f'  - {cn}' for cn in company_names)}

## FUNDAMENTAL DATA (from market data feeds)
{comparison}

## RESEARCH MANDATE

You have the quantitative data above (P/E, margins, growth, analyst ratings).
Your job is to find the LIVE intelligence the data table cannot provide.
Execute ALL searches. Every required field must be populated or marked N/A.

1. **EARNINGS — one search per company, most recent quarter only**:
   "[company name] most recent quarterly earnings results [current year]"
   REQUIRED fields per company (mark N/A if not found — do NOT infer):
   - Revenue reported vs consensus: exact $ beat/miss and %
   - EPS reported vs consensus: exact $ beat/miss and %
   - Most important segment result vs estimate
   - Next quarter guidance vs Street consensus
   - CFO/CEO key statement (exact quote ≤20 words if possible)
   - Stock price reaction (% change after-hours or next session)
   - Report date (YYYY-MM-DD)

2. **ANALYST ACTIONS — 1-2 searches per company**:
   "[company name] analyst upgrade downgrade price target [current year]"
   REQUIRED fields per action (mark N/A if not found):
   - Date, firm name, action (upgrade/downgrade/initiation/target change)
   - New rating, new price target, prior price target
   - Stated reason (≤15 words)
   Collect 2-3 most recent actions per company. If none found: mark N/A.

3. **MARKET DEBATE — 2 searches, do not skip**:
   "[sector] AI capex sustainability DeepSeek bubble [current year]"
   "[company A] vs [company B] ASIC GPU competitive threat [current year]"
   REQUIRED fields (mark N/A if not found):
   - Primary market fear (specific — name the concern and the number behind it)
   - Bull counter-argument
   - Which leg more exposed to bear case and why (one sentence each)
   - Any narrative shift in last 90 days

4. **UPCOMING CATALYSTS — 1 search**:
   "[company A] [company B] next earnings date [current year]"
   Include: exact date, consensus EPS and revenue, key metric to watch.

5. **SECTOR CONTEXT — 1 search**:
   "[sector] outlook [current year]"
   3-5 specific datapoints. No generic statements.

Return structured data in the blocks defined in your system instructions.
Every field populated or explicitly N/A. No omissions. No inference.
"""
    return prompt


async def fetch_claude_fs_analysis(
    fundamental_data: Dict[str, Dict[str, Any]],
    anthropic_api_key: str,
    regime_summary: Optional[Dict] = None,
    enhanced_stats: Optional[Dict] = None,
    long_positions: Optional[Dict[str, float]] = None,
    short_positions: Optional[Dict[str, float]] = None,
    failure_modes: Optional[List[Dict[str, Any]]] = None,
    model: str = "claude-sonnet-4-5",
    mcp_connectors: Optional[List[Dict[str, str]]] = None,
) -> Optional[str]:
    """
    Call Claude API with MCP financial connectors + web search for live research.
    
    This IS the Claude Financial Services agent — Claude + MCP connectors to 
    institutional data providers (LSEG, Moody's, FactSet, S&P, Aiera) + web search.
    
    Architecture:
      Claude API (with mcp-client beta)
        → MCP connectors: LSEG (live market data), Moody's (credit ratings), 
          FactSet (fundamentals), S&P (Capital IQ), Aiera (earnings transcripts)
        → Web search: news, analyst actions, catalysts, competitive intelligence
        → System prompt: senior equity research analyst
        → Output: institutional-grade comparable company research note
    
    MCP connectors are configured via mcp_connectors parameter or CLAUDE_FS_CONNECTORS 
    env var (JSON array). Each connector needs:
      {"url": "https://api.analytics.lseg.com/lfa/mcp", "name": "lseg", "token": "..."}
    
    Without connectors, falls back to web search only (still powerful).
    With connectors, this is the same stack Goldman Sachs and BCI use.
    
    Returns the analysis text, or None if the call fails.
    """
    # Detect asset class
    all_tickers = list((long_positions or {}).keys()) + list((short_positions or {}).keys())
    if not all_tickers:
        all_tickers = list(fundamental_data.keys())
    asset_class = _detect_asset_class(all_tickers)
    
    system_prompt = (
        CLAUDE_FS_CRYPTO_SYSTEM_PROMPT if asset_class == "crypto" 
        else CLAUDE_FS_SYSTEM_PROMPT
    )
    
    prompt = _build_claude_fs_prompt(
        fundamental_data=fundamental_data,
        regime_summary=regime_summary,
        enhanced_stats=enhanced_stats,
        long_positions=long_positions,
        short_positions=short_positions,
        failure_modes=failure_modes,
        asset_class=asset_class,
    )
    
    if not prompt:
        logger.warning("Claude FS: insufficient fundamental data to analyze")
        return None
    
    if not anthropic_api_key:
        logger.warning("Claude FS: no Anthropic API key — skipping fundamental research")
        return None
    
    try:
        import httpx
    except ImportError:
        try:
            import subprocess
            subprocess.check_call(["pip", "install", "httpx", "-q", "--break-system-packages"])
            import httpx
        except Exception:
            logger.warning("Claude FS: httpx not available — skipping")
            return None
    
    # ── Build MCP server connections ──
    # Load from parameter, env var, or defaults
    connectors = mcp_connectors or _load_mcp_connectors()
    
    mcp_servers = []
    tools = []
    
    for conn in connectors:
        server_def = {
            "type": "url",
            "url": conn["url"],
            "name": conn["name"],
        }
        if conn.get("token"):
            server_def["authorization_token"] = conn["token"]
        mcp_servers.append(server_def)
        # Enable all tools from each MCP server
        tools.append({
            "type": "mcp_toolset",
            "mcp_server_name": conn["name"],
        })
    
    # Always include web search
    tools.append({
        "type": "web_search_20250305",
        "name": "web_search",
        "max_uses": 10,
    })
    
    has_mcp = len(mcp_servers) > 0
    connector_names = [c["name"] for c in connectors]
    
    # Add connector context to the prompt so Claude knows what data it can pull
    if has_mcp:
        connector_hint = "\n## CONNECTED DATA SOURCES\n"
        connector_hint += "You have LIVE connections to these institutional data providers via MCP:\n"
        for conn in connectors:
            desc = _MCP_CONNECTOR_DESCRIPTIONS.get(conn["name"], conn["name"])
            connector_hint += f"  - **{conn['name'].upper()}**: {desc}\n"
        
        # Add LSEG-specific tool chaining from the LSEG equity research plugin
        if any(c["name"] == "lseg" for c in connectors):
            connector_hint += """
## LSEG MCP TOOL CHAIN (execute in this order for each company)

1. **qa_ibes_consensus** — Pull FY1 and FY2 consensus estimates (EPS, Revenue, EBITDA, DPS).
   Note analyst count, dispersion (high dispersion = contested thesis), and high/low range.
   
2. **qa_company_fundamentals** — Pull last 3-5 fiscal years of reported financials.
   Extract: revenue growth trajectory, margin trends, ROE/ROIC, leverage (Net Debt/EBITDA).
   
3. **qa_historical_equity_price** — Pull 1Y price history. 
   Compute: YTD return, 1Y return, 52-week range position, beta for each leg.
   
4. **qa_macroeconomic** — Pull GDP, CPI, and policy rate for each company's primary market.
   Assess: is macro a tailwind or headwind for each leg? Differential impact on spread?

SYNTHESIZE: Forward P/E from price / consensus EPS. Compare legs. Where might consensus 
be WRONG? That's where alpha lives. Present consensus tables in your output.
"""
        connector_hint += "\nUSE THESE AGGRESSIVELY. Pull live data for each company — this is your edge.\n"
        connector_hint += "After using connectors, also use web search for recent news and catalysts.\n"
        prompt = prompt.replace("## RESEARCH MANDATE", connector_hint + "\n## RESEARCH MANDATE")
    
    # ── Build request ──
    request_body = {
        "model": model,
        "max_tokens": 4000,
        "temperature": 0.2,
        "system": system_prompt,
        "messages": [{"role": "user", "content": prompt}],
        "tools": tools,
    }
    if mcp_servers:
        request_body["mcp_servers"] = mcp_servers
    
    headers = {
        "x-api-key": anthropic_api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    # MCP connector requires beta header
    if has_mcp:
        headers["anthropic-beta"] = "mcp-client-2025-11-20"
    
    try:
        # Longer timeout: MCP calls + web search + generation can take a while
        timeout = 180.0 if has_mcp else 120.0
        logger.info(
            f"Claude FS: calling {model} with "
            f"{len(mcp_servers)} MCP connectors ({', '.join(connector_names) if connector_names else 'none'}) "
            f"+ web search..."
        )
        
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=request_body,
            )
            
            if response.status_code != 200:
                error_text = response.text[:400]
                logger.error(f"Claude FS API error: {response.status_code} — {error_text}")
                
                # If MCP failed, retry without MCP but keep web search
                if has_mcp and ("mcp" in error_text.lower() or response.status_code == 400):
                    logger.info("Claude FS: MCP error — retrying with web search only...")
                    return await _claude_fs_fallback_websearch(
                        prompt, system_prompt, anthropic_api_key, model
                    )
                # If web search failed, retry without any tools
                if "tool" in error_text.lower() or "web_search" in error_text.lower():
                    logger.info("Claude FS: tool error — retrying without tools...")
                    return await _claude_fs_fallback_plain(
                        prompt, system_prompt, anthropic_api_key, model
                    )
                return None
            
            data = response.json()
            content = data.get("content", [])
            
            # Extract text, count tool uses
            analysis_parts = []
            search_count = 0
            mcp_tool_count = 0
            for block in content:
                if block.get("type") == "text":
                    text = block.get("text", "").strip()
                    if text:
                        analysis_parts.append(text)
                elif block.get("type") == "web_search_tool_result":
                    search_count += 1
                elif block.get("type") == "mcp_tool_use":
                    mcp_tool_count += 1
                elif block.get("type") == "mcp_tool_result":
                    # MCP results may contain useful text we should capture
                    mcp_content = block.get("content", [])
                    for mc in mcp_content:
                        if mc.get("type") == "text" and mc.get("text", "").strip():
                            # Don't add raw MCP data — Claude will synthesize it
                            pass
            
            analysis = "\n\n".join(analysis_parts)
            
            if not analysis or len(analysis) < 100:
                logger.warning("Claude FS: response too short — falling back")
                return await _claude_fs_fallback_websearch(
                    prompt, system_prompt, anthropic_api_key, model
                )
            
            # Track usage
            usage = data.get("usage", {})
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
            logger.info(
                f"Claude FS: research complete ({len(analysis)} chars, "
                f"{mcp_tool_count} MCP calls, {search_count} web searches, "
                f"{input_tokens}+{output_tokens} tokens)"
            )
            
            return analysis
            
    except httpx.TimeoutException:
        logger.error(f"Claude FS: API call timed out ({timeout}s)")
        return None
    except Exception as e:
        logger.error(f"Claude FS: unexpected error — {e}")
        return None


# =============================================================================
# MCP CONNECTOR CONFIGURATION
# =============================================================================
# Known Claude FS MCP connector URLs and descriptions.
# Users configure tokens via env vars or parameter.
# =============================================================================

_MCP_CONNECTOR_REGISTRY = {
    "lseg": {
        "url": "https://api.analytics.lseg.com/lfa/mcp",
        "description": "Live market data: equities, FX, fixed income, yields, macro indicators, analyst estimates",
        "env_token": "LSEG_MCP_TOKEN",
    },
    "moodys": {
        "url": "https://api.moodys.com/genai-ready-data/m1/mcp",
        "description": "Credit ratings, company intelligence, ownership data, financials on 600M+ entities",
        "env_token": "MOODYS_MCP_TOKEN",
    },
    "factset": {
        "url": None,  # User must provide — varies by FactSet deployment
        "description": "Real-time fundamentals, earnings estimates, research, ownership data",
        "env_token": "FACTSET_MCP_TOKEN",
    },
    "sp_global": {
        "url": None,  # User must provide — via connector directory
        "description": "Capital IQ financials, earnings transcripts, deal intelligence",
        "env_token": "SP_GLOBAL_MCP_TOKEN",
    },
    "aiera": {
        "url": None,  # User must provide
        "description": "Real-time earnings call transcripts, investor event summaries, management commentary",
        "env_token": "AIERA_MCP_TOKEN",
    },
    "morningstar": {
        "url": None,  # User must provide
        "description": "Fund research, equity ratings, investment analysis, fair value estimates",
        "env_token": "MORNINGSTAR_MCP_TOKEN",
    },
    "pitchbook": {
        "url": None,  # User must provide
        "description": "Private market data, VC/PE deal flow, company valuations",
        "env_token": "PITCHBOOK_MCP_TOKEN",
    },
    "mt_newswires": {
        "url": None,  # User must provide
        "description": "Real-time global financial market news across all asset classes",
        "env_token": "MT_NEWSWIRES_MCP_TOKEN",
    },
}

_MCP_CONNECTOR_DESCRIPTIONS = {
    name: info["description"] for name, info in _MCP_CONNECTOR_REGISTRY.items()
}


def _load_mcp_connectors() -> List[Dict[str, str]]:
    """
    Load MCP connector configuration from environment.
    
    Two ways to configure:
    
    1. CLAUDE_FS_CONNECTORS env var (JSON array):
       [{"url": "https://api.analytics.lseg.com/lfa/mcp", "name": "lseg", "token": "..."}]
    
    2. Individual env vars for known connectors:
       LSEG_MCP_TOKEN=your-oauth-token  → auto-connects to LSEG
       MOODYS_MCP_TOKEN=your-token      → auto-connects to Moody's
       FACTSET_MCP_URL=https://...       → need URL + FACTSET_MCP_TOKEN
    """
    connectors = []
    
    # Method 1: Explicit JSON config (highest priority)
    env_config = os.environ.get("CLAUDE_FS_CONNECTORS")
    if env_config:
        try:
            parsed = json.loads(env_config)
            if isinstance(parsed, list):
                for c in parsed:
                    if c.get("url") and c.get("name"):
                        connectors.append({
                            "url": c["url"],
                            "name": c["name"],
                            "token": c.get("token", ""),
                        })
                logger.info(f"Claude FS: loaded {len(connectors)} connectors from CLAUDE_FS_CONNECTORS")
                return connectors
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"Claude FS: invalid CLAUDE_FS_CONNECTORS JSON: {e}")
    
    # Method 2: Auto-detect from individual env vars
    for name, info in _MCP_CONNECTOR_REGISTRY.items():
        token = os.environ.get(info["env_token"], "")
        url = info["url"] or os.environ.get(f"{name.upper()}_MCP_URL", "")
        
        if token and url:
            connectors.append({"url": url, "name": name, "token": token})
            logger.info(f"Claude FS: auto-detected {name} connector")
    
    if connectors:
        logger.info(f"Claude FS: {len(connectors)} connectors configured: {[c['name'] for c in connectors]}")
    else:
        logger.info("Claude FS: no MCP connectors configured — using web search only")
    
    return connectors


async def _claude_fs_fallback_websearch(
    prompt: str,
    system_prompt: str,
    anthropic_api_key: str,
    model: str,
) -> Optional[str]:
    """Fallback: Claude FS with web search but no MCP connectors."""
    import httpx
    
    try:
        logger.info("Claude FS fallback: web search only...")
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": anthropic_api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": model,
                    "max_tokens": 3000,
                    "temperature": 0.2,
                    "system": system_prompt,
                    "tools": [{"type": "web_search_20250305", "name": "web_search", "max_uses": 10}],
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            
            if response.status_code != 200:
                logger.error(f"Claude FS websearch fallback error: {response.status_code}")
                return await _claude_fs_fallback_plain(prompt, system_prompt, anthropic_api_key, model)
            
            data = response.json()
            analysis = "\n\n".join(
                block.get("text", "").strip() for block in data.get("content", [])
                if block.get("type") == "text" and block.get("text", "").strip()
            )
            
            if analysis and len(analysis) > 100:
                logger.info(f"Claude FS fallback (web search): {len(analysis)} chars")
                return analysis
            return None
    except Exception as e:
        logger.error(f"Claude FS websearch fallback error: {e}")
        return await _claude_fs_fallback_plain(prompt, system_prompt, anthropic_api_key, model)


async def _claude_fs_fallback_plain(
    prompt: str,
    system_prompt: str,
    anthropic_api_key: str,
    model: str,
) -> Optional[str]:
    """Last resort fallback: Claude FS without any tools."""
    import httpx
    
    try:
        logger.info("Claude FS fallback: plain (no tools)...")
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": anthropic_api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": model,
                    "max_tokens": 2000,
                    "temperature": 0.3,
                    "system": system_prompt,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            
            if response.status_code != 200:
                logger.error(f"Claude FS plain fallback error: {response.status_code}")
                return None
            
            data = response.json()
            analysis = "\n".join(
                block.get("text", "") for block in data.get("content", [])
                if block.get("type") == "text"
            )
            
            if analysis and len(analysis) > 100:
                logger.info(f"Claude FS fallback (plain): {len(analysis)} chars")
                return analysis
            return None
    except Exception as e:
        logger.error(f"Claude FS plain fallback error: {e}")
        return None


def build_decision_brief_prompt(
    enhanced_stats: Dict,
    regime_summary: Dict,
    portfolio_name: str = "Long/Short Portfolio",
    long_positions: Optional[Dict[str, float]] = None,
    short_positions: Optional[Dict[str, float]] = None,
    validity_data: Optional[Dict[str, Any]] = None,
    failure_modes: Optional[List[Dict[str, Any]]] = None,
    pair_state: Optional[str] = None,
    fundamental_context: Optional[str] = None,
    fundamental_data: Optional[Dict[str, Dict[str, Any]]] = None,
    claude_fs_analysis: Optional[str] = None,
    deterministic_decision: Optional[Dict[str, Any]] = None,
) -> str:
    from datetime import date as _date
    today_str = _date.today().strftime("%Y-%m-%d")
    
    portfolio_comp = ""
    if long_positions:
        portfolio_comp += "LONG: " + ", ".join([f"{t} ({w:.0%})" for t, w in long_positions.items()]) + "\n"
    if short_positions:
        portfolio_comp += "SHORT: " + ", ".join([f"{t} ({w:.0%})" for t, w in short_positions.items()])
    
    # Detect asset class to adapt language (equity vs crypto vs mixed)
    all_tickers = list((long_positions or {}).keys()) + list((short_positions or {}).keys())
    asset_class = _detect_asset_class(all_tickers)
    
    prompt = f"# DECISION BRIEF: {portfolio_name}\n\n"
    prompt += f"**TODAY'S DATE: {today_str}** — All temporal reasoning must use this date.\n"
    prompt += "**SECTION NAMING: Write EXACTLY TWO top-level sections in this order: '## TRADE ANALYSIS' then '## FUNDAMENTAL ANALYSIS'. Both are required.**\n\n"
    prompt += f"## TRADE IDENTITY\n{portfolio_comp if portfolio_comp else 'Long/Short Equity Spread'}\n"
    
    # ── DETERMINISTIC DECISION LOCK (PROBLEM 4 FIX) ──
    # The decision is pre-computed by deterministic logic. The LLM EXPLAINS it — never overrides it.
    if deterministic_decision:
        dd = deterministic_decision
        prompt += f"\n## ⚡ LOCKED DECISION: {dd['decision']} ({dd['size_pct']}% of full position) ⚡\n"
        prompt += f"This decision was computed deterministically from: validity score, failure modes, regime, and P&L.\n"
        prompt += f"Engine rationale: {dd['rationale']}\n"
        prompt += f"Your task: EXPLAIN this decision using the data below. You MUST NOT change the decision or the size.\n"
        prompt += f"You may say: 'This decision is supported by...' or 'The key factors driving this are...' — never 'I recommend...'\n"
        prompt += f"Context: {dd['override_reason']}\n"
        prompt += f"\n## ══════════════════════════════════════════════════════════════\n\n"
    
    # Validity
    if validity_data:
        v = validity_data.get("validity", {})
        score = v.get("score")
        state = v.get("state", "UNKNOWN")
        confidence = v.get("confidence", 0)
        summary = v.get("summary", "")
        
        score_conf_note = ""
        if score is not None and score >= 70 and confidence < 0.4:
            score_conf_note = "IMPORTANT: High score but low confidence = structure looks OK on paper but engine doesn't trust its read. Treat as UNCERTAIN, not confirmed-valid."
        elif score is not None and score < 50 and confidence > 0.7:
            score_conf_note = "IMPORTANT: Low score + high confidence = engine is SURE relationship has broken down. Strong exit signal."
        elif score is not None and score < 50 and confidence < 0.4:
            score_conf_note = "Low score + low confidence = may be broken but unsure. Size small."
        
        state_meaning = {"VALID": "Relationship intact. Green light.", "DEGRADED": "Weakening. Reduce exposure.", "INVALID": "Broken down. Exit or avoid."}
        conf_label = "high" if confidence > 0.7 else "moderate" if confidence > 0.5 else "LOW — engine uncertain"
        
        prompt += f"\n## ENGINE VERDICT: {state} (Score: {score}/100)\n{state_meaning.get(state, 'Unknown')}\nConfidence: {confidence:.0%} ({conf_label})\n"
        if summary:
            prompt += f"Summary: {summary}\n"
        if score_conf_note:
            prompt += f"{score_conf_note}\n"
    
    # Regime
    is_trend_regime = False
    if regime_summary:
        raw_regime = regime_summary.get('current_regime', 'unknown')
        regime_key = raw_regime.lower().replace(' ', '_').replace('-', '_')
        regime_ctx = REGIME_TRADING_CONTEXT.get(regime_key, {})
        if not regime_ctx:
            for key in REGIME_TRADING_CONTEXT:
                if key in regime_key or regime_key in key:
                    regime_ctx = REGIME_TRADING_CONTEXT[key]
                    break
        if not regime_ctx:
            regime_ctx = REGIME_TRADING_CONTEXT.get("ranging", {})
        
        z = regime_summary.get('z_score', 0)
        if hasattr(z, 'iloc'):
            z = float(z.iloc[-1])
        rsi = regime_summary.get('rsi', 50)
        adf_p = regime_summary.get('adf_pvalue', 0.5)
        halflife = regime_summary.get('halflife', 999)
        risk_level = regime_summary.get('risk_level', 'N/A')
        is_trend_regime = adf_p > 0.5 or regime_key in ('pure_trend', 'trend_following')
        
        z_interp = "Deeply oversold" if z < -1.5 else "Oversold" if z < -1 else "Near fair value" if abs(z) < 1 else "Overbought" if z < 1.5 else "Deeply overbought"
        adf_interp = "Strongly stationary (good for MR)" if adf_p < 0.05 else "Stationary" if adf_p < 0.2 else "Ambiguous" if adf_p < 0.5 else "Non-stationary — MR NOT supported"
        
        prompt += f"\n## MARKET REGIME: {regime_ctx.get('label', raw_regime.upper())}\n"
        prompt += f"Good for: {regime_ctx.get('suitable_for', 'N/A')}\nBad for: {regime_ctx.get('not_suitable_for', 'N/A')}\n"
        prompt += f"FM context: {regime_ctx.get('fm_note', 'N/A')}\nEntry guidance: {regime_ctx.get('entry_guidance', 'N/A')}\n"
        prompt += f"Z-Score: {z:+.2f} ({z_interp}) | RSI: {rsi:.0f} | ADF: {adf_p:.4f} ({adf_interp}) | Half-life: {halflife:.0f}d | Risk: {risk_level}\n"
    
    # Failure modes
    prompt += "\n## FAILURE MODE SIGNALS\n"
    if failure_modes and len(failure_modes) > 0:
        active_fms = [fm for fm in failure_modes if fm.get('detected', False)]
        if active_fms:
            prompt += f"{len(active_fms)} active:\n"
            for fm in active_fms:
                fm_id = fm.get('failure_mode_id', 'UNKNOWN')
                fm_key = fm_id.split('_')[0].upper() if '_' in str(fm_id) else str(fm_id).upper()
                fm_desc = FM_DESCRIPTIONS.get(fm_key, FM_DESCRIPTIONS.get("COMPOSITE", {}))
                severity = fm.get('severity', 0)
                confidence = fm.get('confidence', {})
                conf_level = confidence.get('level', 'unknown') if isinstance(confidence, dict) else str(confidence)
                conf_overall = confidence.get('overall', 0) if isinstance(confidence, dict) else 0
                engine_detail = fm.get('explanation', '')
                triggers_kill = fm.get('triggers_kill', False)
                is_root = fm.get('is_root_cause', False)
                regime_note = fm_desc.get("regime_note", {}).get("trend" if is_trend_regime else "mr", "")
                
                prompt += f"\n### {fm_desc.get('name', fm_id)} [{fm_key}] — Severity: {severity}/100 | Confidence: {conf_level} ({conf_overall}%)\n"
                if is_root:
                    prompt += "ROOT CAUSE\n"
                if triggers_kill:
                    prompt += "KILL SWITCH\n"
                prompt += f"What: {fm_desc.get('plain_english', '')}\nImpact: {fm_desc.get('so_what', '')}\nRegime: {regime_note}\nAction: {fm_desc.get('what_to_do', '')}\n"
                if engine_detail:
                    prompt += f"Detail: {engine_detail}\n"
        else:
            prompt += "All clean — positive signal.\n"
    else:
        prompt += "No FM data available.\n"
    
    # Performance (bidirectional)
    if enhanced_stats:
        total_ret = enhanced_stats.get('total_return', 0)
        max_dd = enhanced_stats.get('max_drawdown', 0)
        ann_vol = enhanced_stats.get('annualized_volatility', 0)
        win_rate = enhanced_stats.get('win_rate', 0)
        skew = enhanced_stats.get('skewness', 0)
        kurt = enhanced_stats.get('kurtosis', 0)
        var95 = enhanced_stats.get('var_95', 0)
        best_day = enhanced_stats.get('best_day', 0)
        worst_day = enhanced_stats.get('worst_day', 0)
        
        daily_var_dollars = abs(var95 * 1_000_000) if var95 else 0
        monthly_var_dollars = daily_var_dollars * (21 ** 0.5) if daily_var_dollars else 0
        daily_vol_dollars = abs(ann_vol * 1_000_000 / 16) if ann_vol else 0
        inverse_ret = -total_ret
        cvar95 = enhanced_stats.get('cvar_95', 0)
        daily_cvar_dollars = abs(cvar95 * 1_000_000) if cvar95 else 0
        
        inverse_note = "OPPOSITE highly profitable — evaluate REVERSE" if inverse_ret > 0.10 else "inverse also losing — may not be tradeable either way" if inverse_ret < -0.05 else "roughly flat"
        
        # Plain English interpretations (democratize quant for non-quant readers)
        ret_read = "POSITIVE — strategy is working" if total_ret > 0.02 else "ROUGHLY FLAT — no clear edge yet" if total_ret > -0.02 else "NEGATIVE — strategy is struggling"
        dd_read = "well-contained" if abs(max_dd) < 0.08 else "moderate" if abs(max_dd) < 0.15 else "significant — watch closely"
        skew_read = "positive — no hidden left-tail risk" if skew > 0.2 else "roughly symmetric" if skew > -0.2 else "NEGATIVE — prone to sudden drops"
        kurt_read = "normal tails — few surprises" if kurt < 3.5 else "slightly fat tails" if kurt < 5 else "FAT TAILS — expect occasional extreme days"
        wr_read = "edge exists" if win_rate > 0.52 else "coin-flip — no consistent edge" if win_rate > 0.48 else "below 50% — losing more often than winning"
        var_read = f"19 out of 20 days, daily loss stays within ${daily_var_dollars:,.0f}"
        cvar_read = f"on the 1-in-20 bad day, average loss is ${daily_cvar_dollars:,.0f}" if daily_cvar_dollars > 0 else ""
        
        # Distribution health assessment (Gaussian normality caveat)
        is_heavy_tailed = kurt > 4 or skew < -0.5
        if is_heavy_tailed:
            gaussian_caveat = (
                "⚠️ NON-GAUSSIAN: "
                + ("Negative skewness means sudden drops are more likely than gains. " if skew < -0.5 else "")
                + (f"Kurtosis of {kurt:.1f} means extreme days occur more often than a normal distribution predicts. " if kurt > 4 else "")
                + "VaR and monthly estimates assume normality — actual tail losses could exceed these by 20-40%. "
                + "Position sizing and stop-losses should account for heavier tails."
            )
        else:
            gaussian_caveat = (
                "✓ DISTRIBUTION WELL-BEHAVED: Skewness and kurtosis are within normal bounds. "
                "VaR estimates are reasonably reliable. Standard position sizing applies."
            )
        
        regime_dd_flag = ""
        if regime_summary:
            raw_regime = regime_summary.get('current_regime', 'unknown').lower()
            if ('rang' in raw_regime or 'mean' in raw_regime) and abs(max_dd) > 0.15:
                regime_dd_flag = f"REGIME-DRAWDOWN MISMATCH: '{raw_regime}' but DD={max_dd:.1%} suggests directional move."
        
        prompt += f"\n## PERFORMANCE & RISK (on $1M position)\n"
        prompt += f"Return: {total_ret:.2%} ({ret_read})\n"
        
        # Expected daily return
        n_days = enhanced_stats.get('n_trading_days', 180)
        if n_days and n_days > 0 and total_ret != 0:
            daily_ret_avg = total_ret / n_days
            daily_ret_dollars = daily_ret_avg * 1_000_000
            prompt += f"Expected daily return: {daily_ret_avg:.4%} = ~${daily_ret_dollars:+,.0f}/day on $1M "
            prompt += f"({'positive drift — trade has been generating value' if daily_ret_dollars > 0 else 'negative drift — trade is bleeding money daily'})\n"
        
        prompt += f"Max DD: {max_dd:.2%} ({dd_read})\n"
        prompt += f"Best day: {best_day:.2%} | Worst day: {worst_day:.2%}\n"
        prompt += f"Inverse: ~{inverse_ret:+.2%} ({inverse_note})\n"
        prompt += f"\n### CAN IT BLOW UP? (risk metrics with plain English)\n"
        prompt += f"Volatility: {ann_vol:.1%} annualized = ~${daily_vol_dollars:,.0f}/day on $1M\n"
        prompt += f"VaR 95%: {var95:.2%} daily = ${daily_var_dollars:,.0f}/day — {var_read}\n"
        if cvar95:
            prompt += f"CVaR 95%: {cvar95:.2%} daily = ${daily_cvar_dollars:,.0f}/day — {cvar_read}\n"
        prompt += f"Monthly VaR (daily × √21): ~${monthly_var_dollars:,.0f}\n"
        prompt += f"Win rate: {win_rate:.0%} ({wr_read})\n"
        prompt += f"Skewness: {skew:+.2f} ({skew_read})\n"
        prompt += f"Kurtosis: {kurt:.1f} ({kurt_read})\n"
        prompt += f"\n### DISTRIBUTION HEALTH (Gaussian normality assessment)\n"
        prompt += f"{gaussian_caveat}\n"
        if regime_dd_flag:
            prompt += f"{regime_dd_flag}\n"
    
    # Pre-computed sizing inputs for recommendation
    if enhanced_stats or validity_data or failure_modes:
        prompt += "\n## SIZING INPUTS (use these to derive your recommendation — do NOT ignore)\n"
        
        # Confidence
        confidence_pct = 50
        if validity_data:
            confidence_pct = validity_data.get('confidence', {}).get('overall', 50) if isinstance(validity_data.get('confidence'), dict) else validity_data.get('confidence', 50)
        prompt += f"Engine confidence: {confidence_pct}%\n"
        
        # Active FMs summary
        active_fm_count = 0
        total_severity = 0
        max_severity = 0
        if failure_modes:
            for fm in failure_modes:
                if fm.get('detected', False):
                    active_fm_count += 1
                    sev = fm.get('severity', 0)
                    total_severity += sev
                    max_severity = max(max_severity, sev)
        avg_severity = total_severity / active_fm_count if active_fm_count > 0 else 0
        prompt += f"Active failure modes: {active_fm_count} | Avg severity: {avg_severity:.0f}/100 | Max severity: {max_severity}/100\n"
        
        # Validity score
        v_score = validity_data.get('validity_score', 50) if validity_data else 50
        prompt += f"Validity score: {v_score}/100\n"
        
        # Suggested sizing calculation (for GPT to verify/adjust, not blindly copy)
        # Base from confidence
        conf_factor = confidence_pct / 100
        # FM penalty: -10% per FM with severity > 50, -5% per FM with severity 30-50
        fm_penalty = 0
        if failure_modes:
            for fm in failure_modes:
                if fm.get('detected', False):
                    sev = fm.get('severity', 0)
                    if sev > 50:
                        fm_penalty += 0.10
                    elif sev > 30:
                        fm_penalty += 0.05
        
        suggested_size = max(0.25, min(1.0, conf_factor - fm_penalty))
        
        prompt += f"\n⚡ PRE-CALCULATED SIZING: {suggested_size:.0%} of full position ⚡\n"
        prompt += f"  Derivation: confidence {confidence_pct}% → {conf_factor:.2f}"
        if fm_penalty > 0:
            prompt += f", FM penalty -{fm_penalty:.0%}"
        prompt += f" → {suggested_size:.0%}\n"
        prompt += "  USE THIS NUMBER in your recommendation. You may adjust ±10% based on \n"
        prompt += "  fundamentals, but you MUST cite this number and explain any deviation.\n"
        prompt += "  DO NOT invent your own sizing math or make up a 'vol multiple.'\n"
        
        # Decision tree hint
        if v_score < 70 or confidence_pct < 25 or max_severity > 80:
            prompt += "  ⚠️⚠️ LOW SCORE/CONFIDENCE/HIGH SEVERITY → EXIT is almost certainly correct.\n"
            prompt += "  At this confidence level, REDUCE to a tiny position is worse than a clean EXIT.\n"
            prompt += "  Only deviate from EXIT if fundamentals STRONGLY support the trade direction.\n"
        elif suggested_size < 0.30:
            prompt += "  ⚠️ SUGGESTED SIZE BELOW 30% → EXIT is cleaner than maintaining a stub.\n"
            prompt += "  Only keep if fundamentals provide a clear reason to stay.\n"
        elif v_score >= 83 and confidence_pct >= 60 and active_fm_count <= 1 and max_severity <= 40:
            prompt += "  ✓ STRONG SETUP → HOLD may be appropriate if fundamentals confirm. Don't REDUCE just to be safe.\n"
        elif suggested_size < 0.40:
            prompt += "  ⚠️ SMALL SIZE → consider whether the position is worth the operational overhead.\n"
    
    # Fundamental drivers
    prompt += "\n## WHAT'S LIKELY DRIVING THE SPREAD\n"
    
    # Asset class context
    if asset_class == "crypto":
        prompt += """**ASSET CLASS: CRYPTOCURRENCY**
These are crypto assets — they do NOT have earnings, P/E ratios, margins, or dividends.
Do NOT use equity language (earnings surprise, analyst re-rating, margin improvement).
Think in terms of: narrative rotation, on-chain activity, institutional flows (ETF), 
funding rates, token unlocks, protocol upgrades, regulatory events, liquidity shifts,
DeFi TVL, developer activity, and market structure (perps, options OI).
"""
    elif asset_class == "mixed":
        prompt += "**ASSET CLASS: MIXED** — This trade crosses asset classes. Adapt your language accordingly.\n"
    
    if fundamental_context:
        prompt += f"User thesis: {fundamental_context}\nUse as PRIMARY lens.\n"
    
    # Inject real fundamental data comparison if available
    has_real_fundamentals = False
    if fundamental_data and len(fundamental_data) >= 2:
        comparison = _build_fundamental_comparison(fundamental_data)
        if comparison:
            has_real_fundamentals = True
            prompt += f"\n### REAL-TIME FUNDAMENTAL DATA (from market data)\n\n{comparison}\n"
            prompt += "THESE ARE REAL, CURRENT NUMBERS. You MUST cite them by value in your "
            prompt += "WHAT'S DRIVING section. Writing 'X is cheaper on P/E' without the actual "
            prompt += "multiples is a fireable offense. Say '47.0x vs 67.8x' — the numbers are "
            prompt += "right here, there is no excuse for not using them.\n"
    
    if enhanced_stats:
        pattern_key, pattern_data = _infer_spread_driver(enhanced_stats, failure_modes, asset_class)
        if pattern_key != "none" and pattern_data:
            prompt += f"Pattern: {pattern_key.replace('_', ' ').title()} — {pattern_data.get('signature', '')}\n"
            prompt += "Likely drivers:\n"
            for d in pattern_data.get('likely_drivers', []):
                prompt += f"- {d}\n"
            prompt += f"Verify: {pattern_data.get('verification', '')}\n"
            prompt += f"Implication: {pattern_data.get('implication', '')}\n"
        else:
            prompt += "No strong pattern — range-bound.\n"
    
    # ── CLAUDE FS RAW RESEARCH DATA (if available) ──
    if claude_fs_analysis:
        prompt += "\n## ══════════════════════════════════════════════════════════════\n"
        prompt += "## RAW RESEARCH DATA (collected by web search agent)\n"
        prompt += "## ══════════════════════════════════════════════════════════════\n"
        prompt += "The following is structured raw data collected from live web searches.\n"
        prompt += "It contains earnings results, analyst actions, market debate context,\n"
        prompt += "corporate events, and upcoming catalysts.\n\n"
        prompt += "YOUR JOB: Use this data to write the ## FUNDAMENTAL ANALYSIS section\n"
        prompt += "(see output template below). Write it in exactly the same sell-side\n"
        prompt += "voice as the ## TRADE ANALYSIS section — same analyst, same note.\n\n"
        prompt += claude_fs_analysis
        prompt += "\n## ══════════════════════════════════════════════════════════════\n\n"
    
    # Output template — adapt WHAT'S DRIVING section by asset class
    if asset_class == "crypto":
        driving_section = """### WHAT'S DRIVING THE SPREAD (3-5 sentences — EARN YOUR FEE)
Hypothesize WHY the spread moved. For crypto pairs, think about:
- NARRATIVE: Which asset is gaining/losing market attention? ETF flows vs retail speculation?
- ON-CHAIN: TVL changes, developer activity, transaction volumes, active addresses
- MARKET STRUCTURE: Funding rates, liquidation cascades, perp OI, exchange flows
- CATALYSTS: Protocol upgrades, token unlocks, regulatory events, ecosystem developments
- MACRO: BTC as macro asset (correlated with risk-on/off) vs alt-specific narratives
Name the most probable driver and state whether temporary (supports MR) or structural (exit/reverse).
Do NOT use equity language — crypto has no earnings, no P/E, no analyst ratings."""
    else:
        driving_section = """### WHAT'S DRIVING THE SPREAD (5-8 sentences — EARN YOUR FEE)
Hypothesize WHY one stock outperforms the other. Use ALL the fundamental data dimensions.

MANDATORY FORMAT — cite specific numbers with gaps for EACH dimension you discuss:
  "AVGO trades at 67.8x trailing P/E vs NVDA at 47.0x — a 44% premium."
  "NVDA's operating margin of 62% vs AVGO's 37% partially justifies the valuation gap."
Do NOT write "X is cheaper based on P/E" without the actual numbers. The numbers are above — USE THEM.

COVER THESE DIMENSIONS (when data is available — skip any without data):
1. VALUATION GAP: Who's cheaper? By how much? Is the discount justified or excessive?
2. PROFITABILITY: Who earns more per dollar? Does the margin gap explain the valuation gap?
3. GROWTH: Who's growing faster? Does faster growth justify a premium?
4. ANALYST SENTIMENT: Who does the Street prefer? Cite the consensus rating and target upside.
   "Analysts rate X at 1.8 (Buy) with +15% upside to target vs Y at 2.4 (Hold) with +5% upside."
5. MOMENTUM: Who's been outperforming? 52-week performance tells you the market's revealed preference.
6. BALANCE SHEET: Who's financially stronger? Leverage differences affect risk profiles.
7. FCF YIELD: Who generates more cash? Cash generation supports buybacks, dividends, debt reduction.

USE THE COMPARATIVE SCORECARD if provided — it pre-computes who wins on each dimension.
Then SYNTHESIZE: "X wins on 5/7 dimensions — the market is paying a premium for fundamentally 
stronger business, and that premium appears [justified / excessive / insufficient]."

After the dimensional analysis:
- Name the most probable CATALYST: valuation re-rating, earnings divergence, management change, 
  sector rotation, analyst action
- State whether temporary (supports MR) or structural (supports exit/reverse)
- If LIVE FUNDAMENTAL RESEARCH from Claude FS is available above, LEAD WITH IT.
  It contains earnings results, analyst actions, news, and catalysts that the numbers cannot show.
  "Claude FS research found that [company] beat Q4 earnings by 12% and raised FY guidance — 
  this explains the 34pp 12-month outperformance and suggests the premium is earned."
  If Claude FS's verdict DISAGREES with the statistical signal, say so clearly.
  INCLUDE upcoming catalysts: "Next earnings on [date] — a potential inflection point."
DATA QUALITY: If any metric was flagged as distorted, DO NOT use it. Say why.
If no fundamental data provided, use your knowledge but flag uncertainty."""
    
    prompt += f"""
## WRITE THE DECISION BRIEF

### SITUATION (2-3 sentences)
Trade, regime, engine verdict. If score/confidence contradictory, explain plainly.
BE HONEST about performance: if the strategy has been profitable, lead with that.
"The trade has returned +X% — the strategy is working, but [risks]." Don't bury good news.

{driving_section}

### KEY RISKS — CAN IT BLOW UP? (200-250 words)

Structure this section with these mandatory elements:

**1. Failure Mode Risks** (per active FM):
What's wrong, regime relevance, P&L impact in dollars on $1M. If multiple: reinforcing or independent?

**2. Daily Risk in Plain English:**
"On a normal day, the spread moves about ±$X (daily volatility). On a bad day — the kind that 
happens about once a month — losses reach ~$X (daily VaR). On a truly bad day — the 1-in-20 
worst — the average loss is ~$X (CVaR)."
USE THE ACTUAL NUMBERS from the data. This paragraph is MANDATORY.

**3. Tail Risk Assessment:**
Translate skewness and kurtosis into plain language the reader can act on:
- Skewness: "Returns are [symmetric / skewed to the downside — sudden drops are more likely 
  than sudden gains / skewed to the upside — the occasional big win offsets smaller losses]"
- Kurtosis: "Extreme days are [rare — the distribution is well-behaved / about as frequent as 
  you'd expect / MORE frequent than a normal distribution suggests — fat tails mean occasional 
  surprises]"
- CONNECT these to the trade: "This means [the risk is well-contained and largely predictable / 
  you should expect occasional sharp moves beyond what the VaR suggests / the left tail is heavy, 
  so position sizing should account for worse-than-expected drawdowns]"

**4. Drawdown Context:**
Is the max drawdown modest (<8%), moderate (8-15%), or significant (>15%)? Say it plainly.
Be FACTUAL, not alarmist. If risk is well-contained, say so.

### RISK PROFILE (3-4 sentences)
SHOW MATH: monthly = daily VaR × √21. Give ONE base number on $1M.
Then explain what this means: "In a typical month, you should expect the position to move against 
you by up to ~$X. This assumes returns are roughly normally distributed — but [skewness/kurtosis 
suggest they are / suggest fatter tails, so actual monthly losses could exceed this by 20-40%]."
If CVaR data is available, cite it: "On the 1-in-20 bad day, average loss is ~$X."
Then: if an FM is active, QUANTIFY its amplification — e.g., "seasonality adds ~30% to base VaR 
during adverse windows, taking monthly risk from $46k to ~$60k." Do NOT just say "tail risk 
could be amplified" — estimate by how much and explain the mechanism.

### COHERENCE CHECK (mandatory — write this section BEFORE the recommendation)
Answer these explicitly in the memo. These are VISIBLE to the reader.

**Direction:** Does your WHAT'S DRIVING analysis favor the LONG leg, the SHORT leg, or neither?
  Write: "Fundamentals favor [LONG/SHORT/NEITHER]: [one sentence why]"
  
**Implication:** What does this mean for the trade?
  - If fundamentals favor the LONG leg → trade has fundamental support → lean HOLD/ENTER
  - If fundamentals favor the SHORT leg → your own analysis says the trade is WRONG → EXIT or REVERSE
  - If neither → no fundamental edge → decision rests on statistical signal and confidence

**Confidence gate:** Engine confidence is [X]%. At this level:
  - Above 60%: normal sizing applies
  - 30-60%: reduced sizing, use pre-calculated number
  - Below 30%: near-minimum position — seriously consider EXIT over a stub

### RECOMMENDATION
ENTER / HOLD / WAIT / REDUCE X% / REVERSE / EXIT

Your recommendation MUST be consistent with the COHERENCE CHECK above. If you wrote 
"Fundamentals favor SHORT leg" you CANNOT recommend REDUCE — only EXIT or REVERSE.
If confidence is below 30%, you CANNOT recommend REDUCE to 40%+ without explanation.

IF REDUCE: Use the pre-calculated sizing from SIZING INPUTS above.
  - Cite the number: "Pre-calculated sizing: X%"
  - If you deviate: explain why (fundamentals support → size up; fundamentals contradict → size down)
  - DO NOT invent sizing math. DO NOT make up a "vol multiple."

### UPCOMING CATALYSTS (if Claude FS research is available — 3-5 bullets)
If the LIVE FUNDAMENTAL RESEARCH section contains catalyst information, include key dates
organized by type:

EARNINGS & FINANCIAL:
- [Date]: [Company] Q[X] earnings — what to watch for

CORPORATE:
- [Date]: [Event] — expected impact on spread

BINARY EVENTS (if any — these require position management):
- [Date]: [Event] — HIGH IMPACT, consider reducing ahead

If NO research is available, SKIP THIS SECTION entirely.

### THESIS SCORECARD (if Claude FS provided one)
If the LIVE FUNDAMENTAL RESEARCH contains a thesis scorecard with 
STRENGTHENED/UNCHANGED/WEAKENED ratings, reproduce it here.
If ANY pillar is WEAKENED, flag it prominently and discuss in WHAT WOULD CHANGE.
If overall CONVICTION from Claude FS is LOW, this should weight heavily toward EXIT/REDUCE.

### WHAT WOULD CHANGE THIS VIEW (2-3 bullets)
Each bullet MUST contain at least one of:
- Metric + threshold: "ADF p-value below 0.05 for 20+ consecutive days"
- Time condition: "after 40 trading days of post-break data"  
- Price/level: "spread returns within 1σ of 60d mean"
- Named catalyst: "post Q2 earnings release on [date]" or for crypto: "after SOL token unlock on [date]", 
  "BTC ETF daily inflows exceed $500M for 5 consecutive days", "SOL funding rate normalizes below 0.01%"
- KEY RISK TO VERDICT from Claude FS: "[specific event/development that would flip fundamental view]"
BANNED (system-level rule — these will be rejected every time):
"if confidence improves" / "if metrics stabilize" / "if things improve" / 
"if the regime shifts" / "with clear indicators" / "if fundamentals align" /
"if volatility returns to historical norms" / "if market data indicates a reduction in crowding"

### BOTTOM LINE (one sentence)
Action + sizing. 5-second decision.

## ─────────────────────────────────────────────────────────────
## FUNDAMENTAL ANALYSIS
## ─────────────────────────────────────────────────────────────

Write this section immediately after ## TRADE ANALYSIS. Same document, same voice.
This is the section a PM reads to understand the fundamental picture behind the spread.

## ANTI-HALLUCINATION RULE — READ BEFORE WRITING

You have two factual sources:
  (A) RAW RESEARCH DATA block — live web search results collected by Claude FS
  (B) REAL-TIME FUNDAMENTAL DATA table — yfinance metrics injected above

RULE 1: Every number, analyst action, earnings figure, management quote, and company
        event MUST come from source (A) or (B). Do NOT use training knowledge for facts.
RULE 2: If a required field is absent from both (A) and (B), write exactly:
        "[data unavailable]" — do NOT substitute a generic claim or estimate.
RULE 3: Do NOT write "analysts are bullish on NVDA" — write the specific firm, action,
        target, and date from (A), or "[analyst actions: data unavailable]".
RULE 4: Do NOT write "NVIDIA has strong growth" — write "NVIDIA grew revenue 62.5% YoY"
        citing the specific figure from (B), or the figure from (A) if more recent.
RULE 5: If (A) and (B) conflict on a number, use (A) as more recent and note discrepancy.

Violating these rules produces false information that could cause real financial harm.

## STYLE — NON-NEGOTIABLE

Flowing prose. Every paragraph minimum 4 sentences. No bullet points inside paragraphs.
Numbers woven naturally into sentences:
  CORRECT: "NVIDIA posted $68.1B in Q4 revenue, beating the $65.8B consensus by $2.3B
            (3.5%), with data center contributing $62.3B — up 75% YoY and well ahead of
            the $60.7B estimate — while Q1 guidance of $78.0B came in $5.2B above Street."
  BANNED: "Revenue $68.1B. Beat $2.3B. Data center $62.3B. Guidance $78B."
No source annotations. No "(per Bloomberg)". No "(from research block)". No "[Source: X]".
Be opinionated — state what the data MEANS, not just what it says.

## OUTPUT — 4 SECTIONS IN EXACT ORDER

### The Setup

One dense paragraph (5-7 sentences). Both companies' most recent quarter woven into
a single narrative that explains WHY the spread is where it is today.

YOU MUST INCLUDE (from source A — or write [data unavailable]):
✓ [LONG LEG] revenue reported vs consensus: $X beat/miss = X%
✓ [LONG LEG] EPS reported vs consensus: $X beat/miss = X%
✓ [LONG LEG] next quarter guidance vs Street estimate
✓ [LONG LEG] stock reaction % and the reason the market reacted that way
✓ [SHORT LEG] same four fields
✓ At least 2 analyst actions (firm + direction + target) for either leg
✓ The single sentence that explains the spread divergence from earnings

### Valuation & Analyst Positioning

One paragraph (4-5 sentences). The valuation gap and what the Street thinks.

YOU MUST INCLUDE (from source B — or write [data unavailable]):
✓ Forward P/E for both legs: "[LONG] Xx vs [SHORT] Xx — a X% [premium/discount]"
✓ Operating margin gap: "[LONG] X% vs [SHORT] X% — a Xpp gap"
✓ Revenue growth differential: "[LONG] +X% YoY vs [SHORT] +X% YoY"
✓ Analyst consensus rating and price target upside for each leg
✓ Your view: is the valuation gap justified, stretched, or at an inflection point?
  State which specific metric drives your view.

### Market Narrative & Positioning

Two paragraphs (3-4 sentences each).

Paragraph 1 — The live debate (use MARKET DEBATE from source A):
Name the specific fear or thesis the market is pricing right now. Concrete numbers:
  CORRECT: "The central debate is whether $602B in projected 2026 hyperscaler AI capex
            — up 36% YoY — is sustainable when AI-generated revenue of ~$25B represents
            only 4% of that spend, with OpenAI's 2H 2026 IPO threatening to expose the
            unit economics publicly."
  BANNED: "The market has concerns about AI spending sustainability."
If MARKET DEBATE data unavailable from source (A): write one sentence stating this
and skip to paragraph 2.

Paragraph 2 — How each leg is positioned:
Which leg is more exposed to the dominant fear? Which benefits from the bull case?
Use business model differences. Anchor with ≥2 specific numbers from source (B).

### What to Watch

4-6 clean bullets. Each bullet MUST have a date AND a specific spread implication.
  CORRECT: "- March 4, 2026: AVGO Q1 FY2026 earnings (consensus $19.1B / $2.02 EPS)
             — a gross margin guide above 78% would neutralise the margin compression
             thesis and likely compress the spread"
  BANNED: "- March 2026: AVGO earnings — important for the spread"
Use dates from source (A). If no date available: write [date unavailable].
Flag binary events (events that can gap the spread violently) with ⚠️.

BANNED bullets — never include these or any variant:
- Index rebalancing, index inclusion/exclusion
- Sector rotation or capital flows
- "Watch for market reaction" or "could impact the spread" without specifics
- Any bullet without a named company, specific date, and concrete spread implication
- Any bullet sourced from general knowledge rather than source (A)
If you cannot find 4 real catalyst bullets from source (A), write fewer bullets.
3 real bullets beat 5 bullets where 2 are fabricated.

### Fundamental Verdict

Exactly 3 lines. No more.

CONVICTION: [HIGH / MEDIUM / LOW] — [one sentence: the specific data point driving
this conviction level — e.g. "LOW — earnings beat on both sides but opposite market
reactions suggest the spread driver is sentiment, not fundamentals"]
VERDICT: "Fundamentals favor [LONG ticker / SHORT ticker / NEITHER] — [one sentence
citing a specific number or event from sources A or B that drives this view]."
KEY RISK: [One specific development with a name, date, or threshold that would flip
this verdict — e.g. "AVGO March 4 gross margin guidance above 78% would eliminate
the margin compression thesis and shift fundamental support to the long leg"]

BANNED in Fundamental Verdict:
- Copy-pasted boilerplate that doesn't match the actual data (e.g. writing "earnings
  beat on both sides" when one company missed — verify against source (A) first)
- Generic KEY RISK like "if results deviate from expectations" — name the specific
  metric, threshold, and date

## FINAL CONSISTENCY SCAN (before submitting)

1. DATA CITATION: Did WHAT'S DRIVING cite ≥2 specific numbers with gaps? If not, rewrite.
2. COHERENCE CHECK vs RECOMMENDATION: Does the direction in COHERENCE CHECK match the 
   recommendation? "Fundamentals favor SHORT leg" + "REDUCE" = INCOHERENT → fix it.
3. SIZING: Does the % match the pre-calculated suggestion (±10%)? If confidence was 24%, 
   pre-calc is ~25%. Recommending 45% is fabrication.
4. WHAT WOULD CHANGE: Do all bullets have specific metrics/thresholds/dates? Any banned phrases?
5. CLAUDE FS INTEGRATION: If live research was provided:
   a) Did you cite at least 3 findings from it? (earnings data, analyst actions, catalysts)
   b) Did you include the catalyst calendar with dates?
   c) Did you reproduce the thesis scorecard if provided?
   d) If any thesis pillar was WEAKENED, did you flag it?
   e) If Claude FS conviction was LOW but your statistical signals are positive, did you flag the conflict?
   f) Did you cite the MOAT assessment to explain competitive positioning?
   If not, rewrite.

If ANY contradiction found: FIX IT before submitting.
"""
    return prompt


def compute_deterministic_decision(
    validity_data: Optional[Dict[str, Any]] = None,
    failure_modes: Optional[List[Dict[str, Any]]] = None,
    regime_summary: Optional[Dict] = None,
    enhanced_stats: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    PROBLEM 4 FIX: Deterministic decision logic that runs BEFORE the LLM.

    The LLM must EXPLAIN this decision — it may not override it.
    This eliminates entropy from having regime + FMs + fundamentals + narrative
    all simultaneously influencing the recommendation.

    Returns:
        {
            "decision":  "EXIT" | "REDUCE" | "HOLD" | "WAIT" | "REVERSE" | "ENTER",
            "size_pct":  0-100  (target position size as % of full),
            "rationale": str    (one-sentence machine-generated reason),
            "override_reason": str (human-readable explanation for the LLM to elaborate on),
        }
    """
    from datetime import date as _date

    # ── Extract key inputs ──
    v_score, v_state, v_confidence = 50, "UNKNOWN", 0.5
    if validity_data:
        v = validity_data.get("validity", {})
        v_score = v.get("score", 50) or 50
        v_state = v.get("state", "UNKNOWN")
        v_confidence = v.get("confidence", 0.5) or 0.5

    kill_switch = False
    active_fm_count = 0
    max_fm_severity = 0
    if failure_modes:
        for fm in failure_modes:
            if fm.get("detected", False):
                active_fm_count += 1
                sev = fm.get("severity", 0)
                max_fm_severity = max(max_fm_severity, sev)
                if fm.get("triggers_kill", False):
                    kill_switch = True

    total_ret = enhanced_stats.get("total_return", 0) if enhanced_stats else 0
    max_dd = enhanced_stats.get("max_drawdown", 0) if enhanced_stats else 0
    inverse_ret = -total_ret

    adf_p = 0.5
    if regime_summary:
        adf_p = regime_summary.get("adf_pvalue", 0.5) or 0.5
    is_non_stationary = adf_p > 0.5

    # ── Decision tree (order matters — most severe first) ──
    if kill_switch:
        return {
            "decision": "EXIT",
            "size_pct": 0,
            "rationale": f"Kill switch active — critical failure mode (severity {max_fm_severity}/100).",
            "override_reason": "A kill-switch failure mode was detected. Exit is mandatory regardless of other signals.",
        }

    if v_state == "INVALID" or v_score < 40:
        return {
            "decision": "EXIT",
            "size_pct": 0,
            "rationale": f"Validity INVALID (score {v_score}/100) — relationship broken.",
            "override_reason": "The statistical relationship has broken down. Maintaining the position bets against the engine's core assessment.",
        }

    # High-severity FM → EXIT (no partial drip-risk)
    if max_fm_severity >= 70:
        return {
            "decision": "EXIT",
            "size_pct": 0,
            "rationale": f"High-severity failure mode (severity {max_fm_severity}/100) — full exit.",
            "override_reason": "A single high-severity failure mode is sufficient to exit. Partial reduction at this severity level is not appropriate.",
        }

    # Multiple FMs co-firing → EXIT
    if active_fm_count >= 3:
        return {
            "decision": "EXIT",
            "size_pct": 0,
            "rationale": f"{active_fm_count} failure modes co-firing — structural thesis broken, exit.",
            "override_reason": "Multiple simultaneous failure modes indicate compounded structural risk. Exit is cleaner than partial exposure.",
        }

    if total_ret < -0.10 and inverse_ret > 0.10 and is_non_stationary:
        return {
            "decision": "REVERSE",
            "size_pct": 50,
            "rationale": f"Spread lost {total_ret:.1%}; inverse returned {inverse_ret:+.1%}. Non-stationary regime. Flip legs at 50%.",
            "override_reason": "The trade is directionally wrong and the spread is trending, not mean-reverting. Reversing the legs captures the actual observed direction.",
        }

    if v_state == "DEGRADED" or (v_score is not None and v_score < 60):
        conf_label = f"{v_confidence:.0%}" if v_confidence else "unknown"
        if v_confidence and v_confidence >= 0.55 and max_fm_severity < 70:
            return {
                "decision": "REDUCE",
                "size_pct": 50,
                "rationale": f"Validity DEGRADED ({v_score}/100, confidence {conf_label}) — reduce to 50%.",
                "override_reason": "Validity is weakening but confidence is sufficient to maintain a reduced position while the relationship clarifies.",
            }
        else:
            return {
                "decision": "EXIT",
                "size_pct": 0,
                "rationale": f"Validity DEGRADED ({v_score}/100, confidence {conf_label}) — low confidence, exit.",
                "override_reason": "Validity is degraded and confidence is too low to justify any exposure. Exit until structure stabilises.",
            }

    if is_non_stationary and adf_p > 0.8:
        return {
            "decision": "WAIT",
            "size_pct": 0,
            "rationale": "Non-stationary regime (ADF p > 0.8) — mean reversion not supported.",
            "override_reason": "The spread is trending strongly. Mean-reversion entry is not statistically supported until stationarity returns.",
        }

    if active_fm_count == 0 and v_score >= 70 and v_confidence >= 0.5 and total_ret >= 0:
        return {
            "decision": "HOLD",
            "size_pct": 100,
            "rationale": f"All clear: validity {v_score}/100, 0 FMs, return {total_ret:+.1%}.",
            "override_reason": "Statistical relationship is intact, no failure modes active, performance positive. Full position is supported.",
        }

    if active_fm_count <= 1 and v_score >= 60:
        return {
            "decision": "HOLD",
            "size_pct": 75,
            "rationale": f"Mild stress: {active_fm_count} FM, validity {v_score}/100 — maintain at 75%.",
            "override_reason": "One mild failure mode does not justify full exit. 75% sizing reflects modest caution.",
        }

    # Default: mixed signals
    return {
        "decision": "WAIT",
        "size_pct": 0,
        "rationale": "Mixed signals — insufficient conviction to enter or hold.",
        "override_reason": "The engine cannot form a clear view. Waiting preserves capital until the setup clarifies.",
    }


async def generate_memo(
    enhanced_stats: Dict,
    regime_summary: Dict,
    portfolio_name: str = "Long/Short Portfolio",
    long_positions: Optional[Dict[str, float]] = None,
    short_positions: Optional[Dict[str, float]] = None,
    api_key: Optional[str] = None,
    model: str = "gpt-4o",
    validity_data: Optional[Dict[str, Any]] = None,
    failure_modes: Optional[List[Dict[str, Any]]] = None,
    pair_state: Optional[str] = None,
    fundamental_context: Optional[str] = None,
    fundamental_data: Optional[Dict[str, Dict[str, Any]]] = None,
    claude_fs_analysis: Optional[str] = None,
    deterministic_decision: Optional[Dict[str, Any]] = None,
) -> str:
    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.warning("No OpenAI API key — structured fallback")
        return _generate_structured_fallback(enhanced_stats, regime_summary, failure_modes, validity_data, pair_state, fundamental_context)
    
    valid_models = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"]
    if model not in valid_models:
        model = "gpt-4o"
    
    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=api_key)
        prompt = build_decision_brief_prompt(
            enhanced_stats=enhanced_stats, regime_summary=regime_summary,
            portfolio_name=portfolio_name, long_positions=long_positions,
            short_positions=short_positions, validity_data=validity_data,
            failure_modes=failure_modes, pair_state=pair_state,
            fundamental_context=fundamental_context,
            fundamental_data=fundamental_data,
            claude_fs_analysis=claude_fs_analysis,
            deterministic_decision=deterministic_decision,
        )
        logger.info(f"Generating decision brief via {model}...")
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=3500,
        )
        memo = response.choices[0].message.content
        logger.info("Decision brief generated successfully")
        
        # ── TRADE ANALYSIS: rename only, change nothing else ──
        import re as _re
        memo = _re.sub(r'(?i)^#{1,3}\s*(AI\s+Trade\s+Analysis|Trade\s+Analysis|Decision\s+Brief|Trading\s+Decision)\b', '## TRADE ANALYSIS', memo, count=1, flags=_re.MULTILINE)
        if not _re.search(r'^##\s+TRADE ANALYSIS', memo, flags=_re.MULTILINE):
            memo = '## TRADE ANALYSIS\n\n' + memo
        
        # NOTE: Claude FS fundamental HTML is rendered and placed by report.py
        # as a first-class section (PROBLEM 1 FIX). No string injection here.
        
        return memo
    except ImportError:
        logger.error("openai package not installed")
        return _generate_structured_fallback(enhanced_stats, regime_summary, failure_modes, validity_data, pair_state, fundamental_context)
    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        return _generate_structured_fallback(enhanced_stats, regime_summary, failure_modes, validity_data, pair_state, fundamental_context)


# =============================================================================
# CLAUDE FS — HTML SECTION RENDERER
# =============================================================================

def render_claude_fs_html(analysis_text: str, as_of_date: Optional[str] = None) -> str:
    """
    Render Claude FS fundamental research as plain HTML content.

    Returns ONLY the converted markdown content — no wrapper div, no custom colours.
    report.py wraps it in .section + .memo-content identical to TRADE ANALYSIS.
    Strips narration leakage before rendering.
    """
    if not analysis_text:
        return ""

    import re
    from datetime import date as _date

    narration_patterns = [
        r"I'll execute[^.]*\.",
        r"I will (?:now )?search[^.]*\.",
        r"I'm (?:now )?searching[^.]*\.",
        r"Now I need to[^.]*\.",
        r"Let me (?:search|synthesize|now)[^.]*\.",
        r"Perfect\.\s*(?:I now have[^.]*\.)?",
        r"Great\.\s*(?:I (?:found|have)[^.]*\.)?",
        r"Starting with the mandatory[^.]*\.",
        r"I now have comprehensive[^.]*\.",
        r"Based on my research[^.]*\.",
        r"Having searched[^.]*\.",
        r"After searching[^.]*\.",
        r"I'll (?:now )?analyze[^.]*\.",
    ]
    for pat in narration_patterns:
        analysis_text = re.sub(pat, "", analysis_text, flags=re.IGNORECASE)
    analysis_text = re.sub(r'\n{3,}', '\n\n', analysis_text).strip()

    # ── Citation artefact cleanup ──
    # The Anthropic API strips  tags but leaves orphan punctuation
    # (commas, periods, semicolons) on their own lines. Merge them upward.
    lines_raw = analysis_text.split('\n')
    merged = []
    for line in lines_raw:
        stripped_line = line.strip()
        # A line is a dangling punctuation fragment if it consists ONLY of
        # punctuation characters (possibly followed by whitespace)
        if stripped_line and re.match(r'^[,\.;:]+$', stripped_line) and merged:
            # Append the punctuation directly to the previous line (no space)
            merged[-1] = merged[-1].rstrip() + stripped_line
        else:
            merged.append(line)
    # Also collapse any line that starts with punctuation and is very short
    # (e.g. ", and" left over from a split citation) into the previous line
    cleaned = []
    for line in merged:
        stripped_line = line.strip()
        if stripped_line and re.match(r'^[,\.;:]\s+', stripped_line) and cleaned:
            cleaned[-1] = cleaned[-1].rstrip() + ' ' + stripped_line.lstrip(',. ;:').lstrip()
        else:
            cleaned.append(line)
    analysis_text = '\n'.join(cleaned)
    # Final pass: collapse 3+ blank lines again after merging
    analysis_text = re.sub(r'\n{3,}', '\n\n', analysis_text).strip()

    as_of = as_of_date or _date.today().strftime("%Y-%m-%d")

    lines = analysis_text.split('\n')
    html_parts = []
    paragraph_buffer = []

    def flush_paragraph():
        """Flush buffered lines into a single <p>. Never adds spacing — caller decides."""
        if paragraph_buffer:
            text = ' '.join(paragraph_buffer).strip()
            if text:
                text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
                text = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'<em>\1</em>', text)
                html_parts.append(f'<p style="margin: 0; line-height: 1.7;">{text}</p>')
            paragraph_buffer.clear()

    def add_paragraph_spacer():
        """Add spacing between paragraphs — only when last item was actual content."""
        if html_parts and html_parts[-1] not in (
            '<div style="margin-bottom:12px;"></div>',
        ):
            html_parts.append('<div style="margin-bottom:12px;"></div>')

    for line in lines:
        stripped = line.rstrip()

        if re.match(r'^#{1,3}\s+', stripped):
            flush_paragraph()
            # No spacer before section headers — they have their own margin-top
            title = re.sub(r'^#{1,3}\s+', '', stripped)
            title = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', title)
            html_parts.append(
                f'<div style="font-weight:700; font-size:13px; text-transform:uppercase; '
                f'letter-spacing:0.8px; margin:18px 0 6px 0; color:#bbb;">{title}</div>'
            )
        elif re.match(r'^[-•]\s+', stripped):
            flush_paragraph()
            content = re.sub(r'^[-•]\s+', '', stripped)
            content = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', content)
            content = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'<em>\1</em>', content)
            html_parts.append(
                f'<div style="padding-left:16px; margin:2px 0; position:relative;">'
                f'<span style="position:absolute; left:0;">–</span>{content}</div>'
            )
        elif re.match(r'^\d+\.\s+', stripped):
            flush_paragraph()
            m = re.match(r'^(\d+)\.\s+(.+)$', stripped)
            if m:
                num, content = m.group(1), m.group(2)
                content = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', content)
                content = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'<em>\1</em>', content)
                html_parts.append(
                    f'<div style="padding-left:20px; margin:2px 0; position:relative;">'
                    f'<span style="position:absolute; left:0; font-weight:600;">{num}.</span>{content}</div>'
                )
        elif re.match(r'^---+$', stripped):
            flush_paragraph()
            html_parts.append('<hr style="border:none; border-top:1px solid rgba(255,255,255,0.1); margin:10px 0;">')
        elif stripped == '':
            # Genuine blank line = paragraph break: flush buffer then add spacer
            flush_paragraph()
            add_paragraph_spacer()
        else:
            paragraph_buffer.append(stripped)

    flush_paragraph()

    as_of_stamp = (
        f'<div style="font-size:11px; color:#888; margin-bottom:14px; '
        f'letter-spacing:0.5px;">AS OF {as_of}</div>'
    )
    return as_of_stamp + '\n'.join(html_parts)

def _generate_structured_fallback(enhanced_stats, regime_summary, failure_modes=None, validity_data=None, pair_state=None, fundamental_context=None):
    total_ret = enhanced_stats.get('total_return', 0)
    max_dd = enhanced_stats.get('max_drawdown', 0)
    ann_vol = enhanced_stats.get('annualized_volatility', 0)
    var95 = enhanced_stats.get('var_95', 0)
    regime = regime_summary.get('current_regime', 'unknown').upper()
    z_score = regime_summary.get('z_score', 0)
    if hasattr(z_score, 'iloc'):
        z_score = float(z_score.iloc[-1])
    rsi = regime_summary.get('rsi', 50)
    adf_p = regime_summary.get('adf_pvalue', 0.5)
    
    is_trend = adf_p > 0.5
    daily_var_dollars = abs(var95 * 1_000_000) if var95 else 0
    monthly_var_dollars = daily_var_dollars * (21 ** 0.5) if daily_var_dollars else 0
    inverse_ret = -total_ret
    
    active_fms = []
    kill_switch = False
    if failure_modes:
        active_fms = [fm for fm in failure_modes if fm.get('detected', False)]
        kill_switch = any(fm.get('triggers_kill', False) for fm in active_fms)
    
    validity_score, validity_state, validity_confidence = None, pair_state or "UNKNOWN", 0.5
    if validity_data:
        v = validity_data.get("validity", {})
        validity_score = v.get("score")
        validity_state = v.get("state", validity_state)
        validity_confidence = v.get("confidence", 0.5)
    
    if kill_switch:
        rec, reason = "EXIT (IMMEDIATE)", "Kill switch — critical FM."
    elif validity_state == "INVALID" or (validity_score is not None and validity_score < 40):
        rec, reason = "EXIT (ORDERLY)", f"Validity {validity_score}/100 — broken."
    elif total_ret < -0.10 and inverse_ret > 0.10 and (is_trend or adf_p > 0.3):
        rec = "REVERSE"
        reason = f"Lost {total_ret:.1%}, inverse {inverse_ret:+.1%}. Flip legs at 50%."
    elif len(active_fms) >= 3:
        rec, reason = "REDUCE to 25%", f"{len(active_fms)} FMs co-firing."
    elif validity_state == "DEGRADED":
        rec, reason = "REDUCE to 50%", f"Validity degraded ({validity_score}/100)."
    elif is_trend and adf_p > 0.8:
        rec, reason = "WAIT", "Pure trend — MR not supported."
    elif total_ret > 0.02 and max_dd > -0.10 and len(active_fms) <= 1:
        rec = "ENTER" if abs(z_score) > 1 else "HOLD"
        reason = "Profitable, risk contained."
    elif total_ret > 0 and len(active_fms) <= 1:
        rec, reason = "HOLD", "Modestly positive."
    else:
        rec, reason = "WAIT", "Mixed signals."
    
    brief = f"### SITUATION\n{regime} regime. ADF p={adf_p:.4f}. Z={z_score:+.2f}, RSI {rsi:.0f}."
    if validity_score is not None:
        brief += f" Verdict: {validity_state} ({validity_score}/100)"
        if validity_confidence < 0.4:
            brief += f", low confidence ({validity_confidence:.0%})"
    brief += ".\n"
    
    brief += "\n### WHAT'S DRIVING THE SPREAD\n"
    if fundamental_context:
        brief += f"User thesis: {fundamental_context}\n"
    pk, pd = _infer_spread_driver(enhanced_stats, failure_modes)
    if pk != "none" and pd:
        brief += f"Pattern: **{pk.replace('_',' ').title()}**\n"
        for d in pd.get('likely_drivers', [])[:3]:
            brief += f"- {d}\n"
        brief += f"Verify: {pd.get('verification','')}\n"
    
    brief += "\n### KEY RISKS\n"
    if active_fms:
        for fm in active_fms:
            fk = fm.get('failure_mode_id','').split('_')[0].upper()
            fd = FM_DESCRIPTIONS.get(fk, FM_DESCRIPTIONS.get("COMPOSITE",{}))
            brief += f"**{fd.get('name',fk)}** ({fm.get('severity',0)}/100): {fd.get('plain_english','')} Action: {fd.get('what_to_do','')}\n"
    else:
        brief += "All clean.\n"
    
    brief += f"\n### RISK PROFILE\nVol {ann_vol:.1%}. VaR daily {var95:.2%}."
    if monthly_var_dollars > 0:
        brief += f" Monthly $1M risk: ~${monthly_var_dollars:,.0f}."
    if abs(inverse_ret) > 0.05:
        brief += f" Inverse: {inverse_ret:+.1%}."
    brief += f" Historical max DD: {max_dd:.2%} (backward).\n"
    
    brief += f"\n### RECOMMENDATION\n**{rec}**: {reason}\n"
    brief += f"\n### BOTTOM LINE\n{rec} — {reason}\n"
    brief += "\n*Structured fallback — AI unavailable.*\n"
    return brief


build_tactical_memo_prompt = build_decision_brief_prompt
build_validity_governed_prompt = build_decision_brief_prompt

async def generate_memo_legacy(enhanced_stats, regime_summary, portfolio_name="Long/Short Portfolio", long_positions=None, short_positions=None, api_key=None, model="gpt-4o"):
    return await generate_memo(enhanced_stats=enhanced_stats, regime_summary=regime_summary, portfolio_name=portfolio_name, long_positions=long_positions, short_positions=short_positions, api_key=api_key, model=model)
