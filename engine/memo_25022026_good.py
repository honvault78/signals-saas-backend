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
    "trailingPE": {"label": "P/E (trailing)", "fmt": ".1f", "category": "valuation"},
    "forwardPE": {"label": "P/E (forward)", "fmt": ".1f", "category": "valuation"},
    "enterpriseToEbitda": {"label": "EV/EBITDA", "fmt": ".1f", "category": "valuation"},
    "priceToBook": {"label": "P/Book", "fmt": ".2f", "category": "valuation"},
    "enterpriseToRevenue": {"label": "EV/Revenue", "fmt": ".2f", "category": "valuation"},
    "grossMargins": {"label": "Gross Margin", "fmt": ".1%", "category": "margins"},
    "operatingMargins": {"label": "Operating Margin", "fmt": ".1%", "category": "margins"},
    "profitMargins": {"label": "Net Margin", "fmt": ".1%", "category": "margins"},
    "ebitdaMargins": {"label": "EBITDA Margin", "fmt": ".1%", "category": "margins"},
    "revenueGrowth": {"label": "Revenue Growth (YoY)", "fmt": ".1%", "category": "growth"},
    "earningsGrowth": {"label": "Earnings Growth (YoY)", "fmt": ".1%", "category": "growth"},
    "returnOnEquity": {"label": "ROE", "fmt": ".1%", "category": "returns"},
    "returnOnAssets": {"label": "ROA", "fmt": ".1%", "category": "returns"},
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
                "revenueGrowth", "earningsGrowth",
                "returnOnEquity", "returnOnAssets",
                "marketCap", "dividendYield",
            ]
            for key in non_ratio_keys:
                val = info.get(key)
                if val is None or val == 0:
                    continue
                # Growth outlier guards
                if key in ("earningsGrowth", "revenueGrowth") and (abs(val) > 1.0 or val < -0.8):
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


def _build_fundamental_comparison(fundamental_data: dict) -> str:
    """Build formatted valuation/margin comparison for prompt injection."""
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
    
    # Surface data quality warnings FIRST so GPT sees them before the numbers
    all_warnings = []
    for t in tickers:
        for w in fundamental_data.get(t, {}).get("warnings", []):
            all_warnings.append(f"  ⚠️ {t}: {w}")
    if all_warnings:
        lines.append("**DATA QUALITY WARNINGS — read these BEFORE using the numbers below:**")
        lines.extend(all_warnings)
        lines.append("")
    
    cats = {"valuation": "VALUATION (who's cheaper?)", "margins": "MARGINS (who's more profitable?)",
            "growth": "GROWTH (who's growing faster?)", "returns": "CAPITAL EFFICIENCY",
            "size": "SIZE", "yield": "INCOME"}
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
            lines.append(row)
        lines.append("")
    return "\n".join(lines)


SYSTEM_PROMPT = """You are a senior portfolio manager writing a DECISION BRIEF for a trader.

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
8. Keep UNDER 750 words. Every sentence earns its place.
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

DEMOCRATIZING QUANT — MAKE EVERY METRIC ACCESSIBLE:
The data section includes plain English interpretations next to each metric (e.g., "19 out of 20 
days, daily loss stays within $X"). USE these interpretations in your writing. Your reader may be 
a sophisticated investor who understands portfolio theory but is NOT a quant. Translate:
  - VaR → "On a normal bad day, you'd lose about $X"  
  - CVaR → "On a truly bad day (the 1-in-20), average loss is $X"
  - Skewness → "Returns are [symmetric / skewed to the downside / skewed to the upside]"
  - Kurtosis → "Extreme days are [rare / about normal / more frequent than you'd expect]"
  - Half-life → "If the spread dislocates, it historically takes ~X days to mean-revert"
Do NOT just repeat the plain English from the data — weave it into your analysis naturally.

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

TONE: Direct and honest. Like a trusted colleague who says what they actually think."""


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
) -> str:
    portfolio_comp = ""
    if long_positions:
        portfolio_comp += "LONG: " + ", ".join([f"{t} ({w:.0%})" for t, w in long_positions.items()]) + "\n"
    if short_positions:
        portfolio_comp += "SHORT: " + ", ".join([f"{t} ({w:.0%})" for t, w in short_positions.items()])
    
    # Detect asset class to adapt language (equity vs crypto vs mixed)
    all_tickers = list((long_positions or {}).keys()) + list((short_positions or {}).keys())
    asset_class = _detect_asset_class(all_tickers)
    
    prompt = f"# DECISION BRIEF: {portfolio_name}\n\n## TRADE IDENTITY\n{portfolio_comp if portfolio_comp else 'Long/Short Equity Spread'}\n"
    
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
        
        regime_dd_flag = ""
        if regime_summary:
            raw_regime = regime_summary.get('current_regime', 'unknown').lower()
            if ('rang' in raw_regime or 'mean' in raw_regime) and abs(max_dd) > 0.15:
                regime_dd_flag = f"REGIME-DRAWDOWN MISMATCH: '{raw_regime}' but DD={max_dd:.1%} suggests directional move."
        
        prompt += f"\n## PERFORMANCE & RISK (on $1M position)\n"
        prompt += f"Return: {total_ret:.2%} ({ret_read})\n"
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
        driving_section = """### WHAT'S DRIVING THE SPREAD (3-5 sentences — EARN YOUR FEE)
Hypothesize WHY the spread moved. Use the real fundamental data when provided.

MANDATORY FORMAT — you MUST include these exact structures when fundamental data is available:
  "AVGO trades at 67.8x trailing P/E vs NVDA at 47.0x — a 44% premium."
  "NVDA's operating margin of 62% vs AVGO's 37% partially justifies the valuation gap."
Do NOT write "X is cheaper based on P/E" without the actual numbers. That is the SAME as 
writing nothing. The numbers are in the data above — USE THEM.

After citing the numbers:
- Assess whether the valuation gap is JUSTIFIED by better fundamentals or EXCESSIVE
- Name the most probable driver: valuation re-rating, earnings divergence, management change, 
  sector rotation, analyst action
- State whether temporary (supports MR) or structural (supports exit/reverse)
DATA QUALITY: If any metric was flagged as distorted (extreme P/E, base-effect growth), DO NOT 
use it. Say "trailing P/E is distorted — using forward P/E and EV/EBITDA instead."
If no fundamental data provided, use your knowledge but flag uncertainty."""
    
    prompt += f"""
## WRITE THE DECISION BRIEF

### SITUATION (2-3 sentences)
Trade, regime, engine verdict. If score/confidence contradictory, explain plainly.
BE HONEST about performance: if the strategy has been profitable, lead with that.
"The trade has returned +X% — the strategy is working, but [risks]." Don't bury good news.

{driving_section}

### KEY RISKS — CAN IT BLOW UP? (150-200 words)
Per active FM: what's wrong, regime relevance, P&L impact in dollars on $1M.
Multiple FMs: reinforcing or independent?
Use the plain English interpretations from the data: translate VaR into "on a normal bad day, 
you'd lose about $X" and skewness/kurtosis into whether extreme days are likely.
Be FACTUAL, not alarmist. If risk is well-contained, say so: "max drawdown of -3.1% is modest."

### RISK PROFILE (2-3 sentences)
SHOW MATH: monthly = daily VaR × √21. Give ONE base number on $1M.
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

### WHAT WOULD CHANGE THIS VIEW (2-3 bullets)
Each bullet MUST contain at least one of:
- Metric + threshold: "ADF p-value below 0.05 for 20+ consecutive days"
- Time condition: "after 40 trading days of post-break data"  
- Price/level: "spread returns within 1σ of 60d mean"
- Named catalyst: "post Q2 earnings release on [date]" or for crypto: "after SOL token unlock on [date]", 
  "BTC ETF daily inflows exceed $500M for 5 consecutive days", "SOL funding rate normalizes below 0.01%"
BANNED (system-level rule — these will be rejected every time):
"if confidence improves" / "if metrics stabilize" / "if things improve" / 
"if the regime shifts" / "with clear indicators" / "if fundamentals align" /
"if volatility returns to historical norms" / "if market data indicates a reduction in crowding"

### BOTTOM LINE (one sentence)
Action + sizing. 5-second decision.

## FINAL CONSISTENCY SCAN (before submitting)

1. DATA CITATION: Did WHAT'S DRIVING cite ≥2 specific numbers with gaps? If not, rewrite.
2. COHERENCE CHECK vs RECOMMENDATION: Does the direction in COHERENCE CHECK match the 
   recommendation? "Fundamentals favor SHORT leg" + "REDUCE" = INCOHERENT → fix it.
3. SIZING: Does the % match the pre-calculated suggestion (±10%)? If confidence was 24%, 
   pre-calc is ~25%. Recommending 45% is fabrication.
4. WHAT WOULD CHANGE: Do all bullets have specific metrics/thresholds/dates? Any banned phrases?

If ANY contradiction found: FIX IT before submitting.
"""
    return prompt


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
        return memo
    except ImportError:
        logger.error("openai package not installed")
        return _generate_structured_fallback(enhanced_stats, regime_summary, failure_modes, validity_data, pair_state, fundamental_context)
    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        return _generate_structured_fallback(enhanced_stats, regime_summary, failure_modes, validity_data, pair_state, fundamental_context)


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
