"""
Portfolio-Level Analysis Module for Bavella

Integrates methodologies from Claude FS Wealth Management & Equity Research plugins:
  - Portfolio Rebalance: drift analysis, tax-aware rebalancing trades
  - Tax-Loss Harvesting: scan pairs for TLH opportunities
  - Client Report: portfolio performance and allocation reporting
  - Idea Generation: systematic screening for new pair candidates
  - Thesis Tracker: pillar-based thesis validation across portfolio

Architecture:
  portfolio_analysis.py (this file)
    → Called from main.py API endpoints
    → Uses memo.py for individual pair analysis when needed
    → Uses Claude FS for live research on portfolio-level themes
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import math

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. PORTFOLIO DRIFT ANALYSIS
#    (Adapted from: wealth-management/skills/portfolio-rebalance)
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_portfolio_drift(
    pairs: List[Dict[str, Any]],
    target_allocations: Optional[Dict[str, float]] = None,
    rebalance_band: float = 0.05,
) -> Dict[str, Any]:
    """
    Analyze allocation drift across a portfolio of pairs.
    
    Each pair dict should contain:
      - pair_id: str (e.g., "NVDA_AVGO")
      - long_ticker: str
      - short_ticker: str
      - current_value: float (current notional)
      - initial_value: float (initial allocation)
      - pnl: float (unrealized P&L)
      - regime: str (current regime mode)
      - validity_score: float (0-100)
      - failure_modes_active: int (count of active FMs)
      - sector: str (sector classification)
    
    target_allocations: Dict mapping pair_id → target weight (0-1).
        If None, assumes equal weight.
    
    rebalance_band: float, threshold for flagging drift (default ±5%).
    
    Returns portfolio health analysis with drift, risk concentration,
    and rebalancing recommendations.
    """
    if not pairs:
        return {"error": "No pairs provided"}
    
    # ── Calculate total portfolio value ──
    total_value = sum(p.get("current_value", 0) for p in pairs)
    if total_value <= 0:
        return {"error": "Total portfolio value is zero or negative"}
    
    # ── Default to equal weight if no targets ──
    n = len(pairs)
    if target_allocations is None:
        target_allocations = {
            p["pair_id"]: 1.0 / n for p in pairs
        }
    
    # ── Drift Analysis ──
    drift_report = []
    total_abs_drift = 0.0
    pairs_outside_band = 0
    
    for pair in pairs:
        pid = pair["pair_id"]
        current_val = pair.get("current_value", 0)
        current_weight = current_val / total_value if total_value > 0 else 0
        target_weight = target_allocations.get(pid, 1.0 / n)
        drift = current_weight - target_weight
        drift_pct = drift / target_weight if target_weight > 0 else 0
        abs_drift = abs(drift)
        total_abs_drift += abs_drift
        outside_band = abs_drift > rebalance_band
        if outside_band:
            pairs_outside_band += 1
        
        # Dollar amount over/under target
        target_value = target_weight * total_value
        dollar_diff = current_val - target_value
        
        drift_report.append({
            "pair_id": pid,
            "long_ticker": pair.get("long_ticker", "?"),
            "short_ticker": pair.get("short_ticker", "?"),
            "current_value": round(current_val, 2),
            "current_weight": round(current_weight * 100, 2),
            "target_weight": round(target_weight * 100, 2),
            "drift_pct": round(drift * 100, 2),
            "drift_relative_pct": round(drift_pct * 100, 1),
            "dollar_over_under": round(dollar_diff, 2),
            "outside_band": outside_band,
            "regime": pair.get("regime", "unknown"),
            "validity_score": pair.get("validity_score", 0),
            "failure_modes_active": pair.get("failure_modes_active", 0),
            "sector": pair.get("sector", "Unknown"),
        })
    
    # ── Sector Concentration ──
    sector_weights = {}
    for entry in drift_report:
        sector = entry["sector"]
        sector_weights[sector] = sector_weights.get(sector, 0) + entry["current_weight"]
    
    max_sector_weight = max(sector_weights.values()) if sector_weights else 0
    sector_concentration_warning = max_sector_weight > 40  # >40% in one sector
    
    # ── Regime Risk ──
    broken_pairs = [d for d in drift_report if d["regime"] in ("Broken", "Fragile")]
    broken_weight = sum(d["current_weight"] for d in broken_pairs)
    
    # ── Validity Risk ──
    low_validity = [d for d in drift_report if d["validity_score"] < 50]
    low_validity_weight = sum(d["current_weight"] for d in low_validity)
    
    # ── FM Concentration ──
    high_fm = [d for d in drift_report if d["failure_modes_active"] >= 3]
    high_fm_weight = sum(d["current_weight"] for d in high_fm)
    
    # ── Portfolio Health Score (0-100) ──
    health = 100
    health -= min(30, pairs_outside_band * 10)  # -10 per drifted pair, max -30
    health -= min(20, broken_weight * 0.5)  # Broken/fragile weight penalty
    health -= min(20, low_validity_weight * 0.4)  # Low validity penalty
    health -= min(15, high_fm_weight * 0.3)  # High FM count penalty
    health -= min(15, max(0, max_sector_weight - 30) * 0.5)  # Sector concentration
    health = max(0, round(health))
    
    return {
        "portfolio_value": round(total_value, 2),
        "pair_count": n,
        "health_score": health,
        "drift_analysis": sorted(drift_report, key=lambda x: abs(x["drift_pct"]), reverse=True),
        "pairs_outside_band": pairs_outside_band,
        "rebalance_band_pct": round(rebalance_band * 100, 1),
        "total_absolute_drift": round(total_abs_drift * 100, 2),
        "sector_weights": {k: round(v, 2) for k, v in sorted(sector_weights.items(), key=lambda x: -x[1])},
        "sector_concentration_warning": sector_concentration_warning,
        "risk_summary": {
            "broken_fragile_weight_pct": round(broken_weight, 2),
            "broken_fragile_pairs": [d["pair_id"] for d in broken_pairs],
            "low_validity_weight_pct": round(low_validity_weight, 2),
            "low_validity_pairs": [d["pair_id"] for d in low_validity],
            "high_fm_weight_pct": round(high_fm_weight, 2),
            "high_fm_pairs": [d["pair_id"] for d in high_fm],
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 2. REBALANCING RECOMMENDATIONS
#    (Adapted from: wealth-management/skills/portfolio-rebalance)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_rebalancing_trades(
    drift_analysis: Dict[str, Any],
    account_type: str = "taxable",
    min_trade_size: float = 500.0,
) -> Dict[str, Any]:
    """
    Generate rebalancing trade recommendations from drift analysis.
    
    Tax-Aware Rebalancing Rules (from wealth management plugin):
    - In taxable accounts, prefer trimming winners (favorable tax treatment for LT gains)
    - Avoid selling positions with large short-term gains
    - Direct new capital to underweight positions before selling overweight ones
    - Flag wash sale risk if harvesting losses
    
    Args:
        drift_analysis: Output from analyze_portfolio_drift()
        account_type: "taxable", "ira", "roth" — affects tax logic
        min_trade_size: Minimum trade notional to bother with
    
    Returns trade list with rationale.
    """
    if "error" in drift_analysis:
        return drift_analysis
    
    entries = drift_analysis.get("drift_analysis", [])
    total_value = drift_analysis.get("portfolio_value", 0)
    
    trades = []
    total_buys = 0.0
    total_sells = 0.0
    
    for entry in entries:
        if not entry["outside_band"]:
            continue
        
        dollar_diff = entry["dollar_over_under"]
        
        # Skip trivially small adjustments
        if abs(dollar_diff) < min_trade_size:
            continue
        
        pid = entry["pair_id"]
        regime = entry["regime"]
        validity = entry["validity_score"]
        fms = entry["failure_modes_active"]
        
        # ── Decision Logic ──
        # Regime-aware: don't add to Broken/Fragile pairs, trim them
        if regime in ("Broken", "Fragile"):
            if dollar_diff > 0:
                # Overweight + broken → trim more aggressively
                action = "REDUCE"
                reason = f"Overweight by ${abs(dollar_diff):,.0f} AND regime is {regime} — trim to target or below"
                trade_amount = -abs(dollar_diff)
            else:
                # Underweight + broken → do NOT add, close instead
                action = "DO NOT ADD"
                reason = f"Underweight but regime is {regime} — do not add capital to degrading relationship"
                trade_amount = 0
        elif validity < 40:
            if dollar_diff > 0:
                action = "REDUCE"
                reason = f"Overweight by ${abs(dollar_diff):,.0f} with low validity ({validity}) — trim exposure"
                trade_amount = -abs(dollar_diff)
            else:
                action = "HOLD"
                reason = f"Underweight but validity only {validity} — wait for improvement before adding"
                trade_amount = 0
        elif fms >= 3:
            if dollar_diff > 0:
                action = "TRIM"
                reason = f"Overweight by ${abs(dollar_diff):,.0f} with {fms} active failure modes — reduce concentration"
                trade_amount = -abs(dollar_diff) * 0.5  # Trim half
            else:
                action = "PARTIAL ADD"
                reason = f"Underweight by ${abs(dollar_diff):,.0f} but {fms} failure modes active — add cautiously"
                trade_amount = abs(dollar_diff) * 0.5
        else:
            # Normal regime, good validity
            if dollar_diff > 0:
                action = "TRIM"
                reason = f"Overweight by ${abs(dollar_diff):,.0f} vs target — rebalance to target"
                trade_amount = -abs(dollar_diff)
            else:
                action = "ADD"
                reason = f"Underweight by ${abs(dollar_diff):,.0f} vs target — increase to target"
                trade_amount = abs(dollar_diff)
        
        # Tax impact note
        tax_note = ""
        if account_type == "taxable":
            if trade_amount < 0:
                tax_note = "Check holding period: prefer closing lots >1yr for LT capital gains rate"
            elif trade_amount > 0:
                tax_note = "New position — starts short-term holding period clock"
        elif account_type in ("ira", "roth"):
            tax_note = "Tax-advantaged account — no tax impact from rebalancing"
        
        trades.append({
            "pair_id": pid,
            "long_ticker": entry["long_ticker"],
            "short_ticker": entry["short_ticker"],
            "action": action,
            "trade_amount": round(trade_amount, 2),
            "current_weight": entry["current_weight"],
            "target_weight": entry["target_weight"],
            "reason": reason,
            "tax_note": tax_note,
            "regime": regime,
            "validity_score": validity,
            "priority": "HIGH" if regime in ("Broken", "Fragile") or fms >= 3 else "MEDIUM" if entry["outside_band"] else "LOW",
        })
        
        if trade_amount > 0:
            total_buys += trade_amount
        else:
            total_sells += abs(trade_amount)
    
    # ── Net Capital Required ──
    net_capital = total_buys - total_sells
    
    return {
        "trades": sorted(trades, key=lambda t: {"HIGH": 0, "MEDIUM": 1, "LOW": 2}.get(t["priority"], 3)),
        "trade_count": len(trades),
        "total_buys": round(total_buys, 2),
        "total_sells": round(total_sells, 2),
        "net_capital_required": round(net_capital, 2),
        "account_type": account_type,
        "note": (
            "Regime-aware rebalancing: Broken/Fragile pairs are trimmed regardless of target. "
            "Low-validity pairs are held, not added to. High failure-mode pairs are half-sized."
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 3. TAX-LOSS HARVESTING SCANNER
#    (Adapted from: wealth-management/skills/tax-loss-harvesting)
# ═══════════════════════════════════════════════════════════════════════════════

def scan_tax_loss_opportunities(
    pairs: List[Dict[str, Any]],
    min_loss_threshold: float = 500.0,
    marginal_tax_rate: float = 0.37,
    ltcg_rate: float = 0.20,
) -> Dict[str, Any]:
    """
    Scan portfolio for tax-loss harvesting opportunities.
    
    Each pair dict should additionally contain:
      - unrealized_pnl: float (negative = loss)
      - entry_date: str (ISO date of entry)
      - cost_basis: float
      - replacement_candidates: Optional[List[str]] (similar pairs to maintain exposure)
    
    Methodology from wealth management plugin:
    1. Identify positions with unrealized losses
    2. Prioritize: largest absolute loss first, then short-term losses (higher tax savings)
    3. Suggest replacement securities to maintain market exposure
    4. Flag wash sale risk (30-day window)
    5. Estimate tax savings
    """
    if not pairs:
        return {"error": "No pairs provided"}
    
    now = datetime.now()
    opportunities = []
    total_harvestable_loss = 0.0
    total_estimated_savings = 0.0
    
    for pair in pairs:
        pnl = pair.get("unrealized_pnl", 0)
        if pnl >= 0:
            continue  # No loss to harvest
        
        loss = abs(pnl)
        if loss < min_loss_threshold:
            continue  # Below threshold
        
        # Determine holding period
        entry_str = pair.get("entry_date", "")
        try:
            entry_date = datetime.fromisoformat(entry_str)
            holding_days = (now - entry_date).days
            is_long_term = holding_days > 365
            holding_period = "Long-term" if is_long_term else "Short-term"
        except (ValueError, TypeError):
            holding_days = 0
            is_long_term = False
            holding_period = "Unknown"
        
        # Tax savings estimate
        tax_rate = ltcg_rate if is_long_term else marginal_tax_rate
        estimated_savings = loss * tax_rate
        
        # Replacement candidates
        replacements = pair.get("replacement_candidates", [])
        
        total_harvestable_loss += loss
        total_estimated_savings += estimated_savings
        
        opportunities.append({
            "pair_id": pair["pair_id"],
            "long_ticker": pair.get("long_ticker", "?"),
            "short_ticker": pair.get("short_ticker", "?"),
            "unrealized_loss": round(-loss, 2),
            "cost_basis": round(pair.get("cost_basis", 0), 2),
            "current_value": round(pair.get("current_value", 0), 2),
            "loss_pct": round(loss / pair.get("cost_basis", 1) * 100, 1) if pair.get("cost_basis") else 0,
            "holding_period": holding_period,
            "holding_days": holding_days,
            "tax_rate_applicable": round(tax_rate * 100, 1),
            "estimated_tax_savings": round(estimated_savings, 2),
            "regime": pair.get("regime", "unknown"),
            "validity_score": pair.get("validity_score", 0),
            "replacement_candidates": replacements,
            "wash_sale_note": (
                "30-day wash sale window: do not repurchase substantially identical "
                "position within 30 days before or after the sale. Similar but non-identical "
                "pairs (e.g., different sector ETFs in same industry) are generally acceptable."
            ),
            "recommendation": _tlh_recommendation(pair, loss, is_long_term),
        })
    
    # Sort by estimated savings descending
    opportunities.sort(key=lambda x: x["estimated_tax_savings"], reverse=True)
    
    return {
        "opportunities": opportunities,
        "opportunity_count": len(opportunities),
        "total_harvestable_loss": round(total_harvestable_loss, 2),
        "total_estimated_tax_savings": round(total_estimated_savings, 2),
        "marginal_tax_rate_used": round(marginal_tax_rate * 100, 1),
        "ltcg_rate_used": round(ltcg_rate * 100, 1),
        "note": (
            "Tax-loss harvesting resets cost basis. Future gains on replacement position "
            "will be higher (basis step-down). Consider whether the immediate tax savings "
            "outweigh the future tax cost. Also coordinate across all accounts to avoid "
            "wash sale violations."
        ),
    }


def _tlh_recommendation(pair: Dict, loss: float, is_long_term: bool) -> str:
    """Generate a TLH recommendation based on pair state."""
    regime = pair.get("regime", "unknown")
    validity = pair.get("validity_score", 0)
    
    if regime in ("Broken", "Fragile"):
        return (
            f"HARVEST — Pair is {regime}. Close the position for the ${loss:,.0f} tax loss "
            f"and DO NOT replace. Statistical relationship has degraded."
        )
    elif validity < 40:
        return (
            f"HARVEST & WAIT — Validity score {validity} suggests pair may be breaking down. "
            f"Take the ${loss:,.0f} loss. Monitor for re-entry if validity recovers above 60."
        )
    elif is_long_term:
        return (
            f"CONSIDER — Long-term loss of ${loss:,.0f} saves {loss * 0.20:,.0f} at LTCG rate. "
            f"If pair is still fundamentally valid, harvest and replace with similar pair "
            f"to maintain exposure."
        )
    else:
        return (
            f"HARVEST — Short-term loss of ${loss:,.0f} saves {loss * 0.37:,.0f} at ordinary "
            f"income rate. Replace with similar pair to maintain market exposure."
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 4. PORTFOLIO PERFORMANCE SUMMARY
#    (Adapted from: wealth-management/skills/client-report)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_portfolio_summary(
    pairs: List[Dict[str, Any]],
    benchmark_return: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Generate portfolio performance summary.
    
    Each pair should contain:
      - pair_id: str
      - current_value: float
      - initial_value: float
      - pnl: float (realized + unrealized)
      - return_pct: float
      - sharpe_ratio: float
      - max_drawdown: float
      - regime: str
      - validity_score: float
      - entry_date: str
    
    Returns structured performance data for reporting.
    """
    if not pairs:
        return {"error": "No pairs provided"}
    
    total_initial = sum(p.get("initial_value", 0) for p in pairs)
    total_current = sum(p.get("current_value", 0) for p in pairs)
    total_pnl = sum(p.get("pnl", 0) for p in pairs)
    
    portfolio_return = (total_current / total_initial - 1) if total_initial > 0 else 0
    
    # ── By-pair performance ──
    pair_performance = []
    winners = 0
    losers = 0
    
    for pair in pairs:
        ret = pair.get("return_pct", 0)
        if ret > 0:
            winners += 1
        elif ret < 0:
            losers += 1
        
        pair_performance.append({
            "pair_id": pair["pair_id"],
            "return_pct": round(ret * 100, 2) if isinstance(ret, float) and ret < 1 else round(ret, 2),
            "pnl": round(pair.get("pnl", 0), 2),
            "sharpe": round(pair.get("sharpe_ratio", 0), 2),
            "max_drawdown_pct": round(pair.get("max_drawdown", 0) * 100, 2),
            "regime": pair.get("regime", "unknown"),
            "validity": pair.get("validity_score", 0),
            "weight_pct": round(pair.get("current_value", 0) / total_current * 100, 1) if total_current > 0 else 0,
        })
    
    # Sort by return
    pair_performance.sort(key=lambda x: x["return_pct"], reverse=True)
    
    # ── Portfolio-level metrics ──
    sharpes = [p.get("sharpe_ratio", 0) for p in pairs if p.get("sharpe_ratio")]
    avg_sharpe = sum(sharpes) / len(sharpes) if sharpes else 0
    
    drawdowns = [p.get("max_drawdown", 0) for p in pairs if p.get("max_drawdown")]
    worst_dd = min(drawdowns) if drawdowns else 0
    
    validities = [p.get("validity_score", 0) for p in pairs]
    avg_validity = sum(validities) / len(validities) if validities else 0
    
    # ── Regime Distribution ──
    regime_dist = {}
    for pair in pairs:
        r = pair.get("regime", "unknown")
        regime_dist[r] = regime_dist.get(r, 0) + 1
    
    return {
        "portfolio_value": round(total_current, 2),
        "initial_value": round(total_initial, 2),
        "total_pnl": round(total_pnl, 2),
        "portfolio_return_pct": round(portfolio_return * 100, 2),
        "benchmark_return_pct": round(benchmark_return * 100, 2) if benchmark_return is not None else None,
        "alpha_pct": round((portfolio_return - benchmark_return) * 100, 2) if benchmark_return is not None else None,
        "pair_count": len(pairs),
        "winners": winners,
        "losers": losers,
        "win_rate_pct": round(winners / len(pairs) * 100, 1) if pairs else 0,
        "avg_sharpe": round(avg_sharpe, 2),
        "worst_drawdown_pct": round(worst_dd * 100, 2),
        "avg_validity_score": round(avg_validity, 1),
        "regime_distribution": regime_dist,
        "pair_performance": pair_performance,
        "best_pair": pair_performance[0] if pair_performance else None,
        "worst_pair": pair_performance[-1] if pair_performance else None,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 5. THESIS TRACKER — PORTFOLIO-WIDE
#    (Adapted from: equity-research/skills/thesis-tracker)
# ═══════════════════════════════════════════════════════════════════════════════

def portfolio_thesis_review(
    pairs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Run a portfolio-wide thesis review.
    
    For each pair, assess whether the original trade thesis is still intact
    using the Thesis Tracker methodology:
    
    THESIS PILLARS:
    1. Statistical validity (regime + ADF)
    2. Fundamental quality (validity score)
    3. Regime health (failure modes)
    
    Status: INTACT / MONITORING / DEGRADED / BROKEN
    
    Each pair dict should contain:
      - pair_id: str
      - regime: str
      - validity_score: float
      - failure_modes_active: int
      - adf_pvalue: float (optional)
      - days_in_current_regime: int (optional)
      - spread_zscore: float (optional)
      - pnl: float
    """
    if not pairs:
        return {"error": "No pairs provided"}
    
    reviews = []
    status_counts = {"INTACT": 0, "MONITORING": 0, "DEGRADED": 0, "BROKEN": 0}
    
    for pair in pairs:
        pid = pair["pair_id"]
        regime = pair.get("regime", "unknown")
        validity = pair.get("validity_score", 0)
        fms = pair.get("failure_modes_active", 0)
        adf = pair.get("adf_pvalue", None)
        days_regime = pair.get("days_in_current_regime", 0)
        z = pair.get("spread_zscore", 0)
        pnl = pair.get("pnl", 0)
        
        # ── Assess each pillar ──
        pillars = []
        
        # Pillar 1: Statistical validity
        if regime in ("Actionable", "Watch"):
            p1_status = "STRENGTHENED" if (adf and adf < 0.05) else "UNCHANGED"
            p1_note = f"Regime: {regime}, ADF: {adf:.3f}" if adf else f"Regime: {regime}"
        elif regime == "Dormant":
            p1_status = "UNCHANGED"
            p1_note = f"Dormant — waiting for activation signal"
        elif regime == "Fragile":
            p1_status = "WEAKENED"
            p1_note = f"Fragile for {days_regime} days — approaching breakdown"
        elif regime == "Broken":
            p1_status = "BROKEN"
            p1_note = f"Broken — cointegration relationship dissolved"
        else:
            p1_status = "UNCHANGED"
            p1_note = f"Regime: {regime}"
        pillars.append({"name": "Statistical Validity", "status": p1_status, "note": p1_note})
        
        # Pillar 2: Fundamental quality
        if validity >= 70:
            p2_status = "STRENGTHENED"
            p2_note = f"Validity score {validity} — strong fundamental support"
        elif validity >= 50:
            p2_status = "UNCHANGED"
            p2_note = f"Validity score {validity} — adequate"
        elif validity >= 30:
            p2_status = "WEAKENED"
            p2_note = f"Validity score {validity} — fundamental support declining"
        else:
            p2_status = "BROKEN"
            p2_note = f"Validity score {validity} — insufficient fundamental basis"
        pillars.append({"name": "Fundamental Quality", "status": p2_status, "note": p2_note})
        
        # Pillar 3: Regime health / failure modes
        if fms == 0:
            p3_status = "STRENGTHENED"
            p3_note = "No failure modes active — clean"
        elif fms <= 2:
            p3_status = "UNCHANGED"
            p3_note = f"{fms} failure mode(s) active — within normal range"
        elif fms <= 4:
            p3_status = "WEAKENED"
            p3_note = f"{fms} failure modes active — degradation signals present"
        else:
            p3_status = "BROKEN"
            p3_note = f"{fms} failure modes active — relationship severely impaired"
        pillars.append({"name": "Regime Health", "status": p3_status, "note": p3_note})
        
        # ── Overall thesis status ──
        statuses = [p["status"] for p in pillars]
        if "BROKEN" in statuses:
            overall = "BROKEN"
            action = "EXIT — one or more thesis pillars have broken. Relationship is no longer valid."
        elif statuses.count("WEAKENED") >= 2:
            overall = "DEGRADED"
            action = "REDUCE — multiple thesis pillars weakening. Begin unwinding position."
        elif "WEAKENED" in statuses:
            overall = "MONITORING"
            action = "WATCH — one pillar weakening. Tighten stops, prepare exit plan."
        else:
            overall = "INTACT"
            action = "HOLD — thesis pillars intact. Continue position."
        
        status_counts[overall] += 1
        
        reviews.append({
            "pair_id": pid,
            "overall_status": overall,
            "action": action,
            "pillars": pillars,
            "pnl": round(pnl, 2),
            "current_regime": regime,
            "validity_score": validity,
            "failure_modes_active": fms,
        })
    
    # Sort: BROKEN first, then DEGRADED, MONITORING, INTACT
    priority = {"BROKEN": 0, "DEGRADED": 1, "MONITORING": 2, "INTACT": 3}
    reviews.sort(key=lambda r: priority.get(r["overall_status"], 4))
    
    return {
        "review_date": datetime.now().isoformat(),
        "pair_count": len(pairs),
        "status_distribution": status_counts,
        "action_required": status_counts["BROKEN"] + status_counts["DEGRADED"],
        "monitoring": status_counts["MONITORING"],
        "healthy": status_counts["INTACT"],
        "reviews": reviews,
        "portfolio_thesis_health": (
            "CRITICAL" if status_counts["BROKEN"] > 0 else
            "WARNING" if status_counts["DEGRADED"] > 0 else
            "CAUTION" if status_counts["MONITORING"] > 0 else
            "HEALTHY"
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 6. PAIR IDEA GENERATION / SCREENING
#    (Adapted from: equity-research/skills/idea-generation)
# ═══════════════════════════════════════════════════════════════════════════════

def screen_pair_candidates(
    candidates: List[Dict[str, Any]],
    screen_type: str = "quality",
    min_validity: float = 60.0,
    max_failure_modes: int = 2,
    min_half_life: int = 5,
    max_half_life: int = 120,
) -> Dict[str, Any]:
    """
    Screen potential pair candidates using quantitative filters.
    
    Screen types (from idea generation plugin, adapted for pairs):
      - "quality": High validity, low FMs, good regime — best all-around
      - "value": Actionable regime with extreme z-scores — mean reversion opportunity
      - "momentum": Strong trend-following signals, positive recent returns
      - "contrarian": Recently Broken pairs showing signs of recovery — higher risk/reward
    
    Each candidate dict should contain:
      - pair_id: str
      - long_ticker: str
      - short_ticker: str
      - validity_score: float
      - regime: str
      - failure_modes_active: int
      - adf_pvalue: float
      - half_life: float (days)
      - spread_zscore: float
      - recent_return: float (30-day return)
      - sharpe_ratio: float
      - correlation: float
      - sector: str
    """
    if not candidates:
        return {"error": "No candidates provided"}
    
    passed = []
    
    for c in candidates:
        validity = c.get("validity_score", 0)
        regime = c.get("regime", "unknown")
        fms = c.get("failure_modes_active", 99)
        adf = c.get("adf_pvalue", 1.0)
        hl = c.get("half_life", 0)
        z = c.get("spread_zscore", 0)
        ret = c.get("recent_return", 0)
        sharpe = c.get("sharpe_ratio", 0)
        corr = c.get("correlation", 0)
        
        # ── Common filters ──
        if hl < min_half_life or hl > max_half_life:
            continue
        
        # ── Screen-specific filters ──
        if screen_type == "quality":
            if validity < min_validity:
                continue
            if fms > max_failure_modes:
                continue
            if regime in ("Broken",):
                continue
            if adf > 0.10:  # Want p-value below 0.10
                continue
            score = validity * 0.4 + (1 - adf) * 100 * 0.3 + max(0, sharpe) * 20 * 0.3
            
        elif screen_type == "value":
            # Looking for extreme z-scores in valid pairs
            if validity < 40:
                continue
            if abs(z) < 1.5:  # Need z-score beyond ±1.5
                continue
            if regime not in ("Actionable", "Watch"):
                continue
            score = abs(z) * 30 + validity * 0.3 + (1 - adf) * 50
            
        elif screen_type == "momentum":
            # Strong recent performance in trending regimes
            if validity < 50:
                continue
            if ret <= 0:
                continue
            if regime in ("Broken", "Fragile"):
                continue
            score = ret * 100 * 0.4 + sharpe * 20 * 0.3 + validity * 0.3
            
        elif screen_type == "contrarian":
            # Recently broken but showing recovery signs
            if regime not in ("Fragile", "Watch"):
                continue
            if fms <= 1:  # Need some failure modes (shows recent stress)
                continue
            if validity < 30:
                continue
            # Higher score for pairs recovering from stress
            score = validity * 0.5 + (max_failure_modes - fms + 1) * 10 * 0.3 + (1 - adf) * 50 * 0.2
        
        else:
            score = validity  # Default
        
        passed.append({
            **c,
            "screen_score": round(score, 1),
            "screen_type": screen_type,
        })
    
    # Sort by score descending
    passed.sort(key=lambda x: x.get("screen_score", 0), reverse=True)
    
    # Top 10
    top = passed[:10]
    
    return {
        "screen_type": screen_type,
        "candidates_screened": len(candidates),
        "candidates_passed": len(passed),
        "top_ideas": top,
        "screen_criteria": {
            "quality": "High validity, low failure modes, cointegrated, good Sharpe",
            "value": "Extreme z-scores in actionable/watch regime — mean reversion plays",
            "momentum": "Positive recent returns, trending regime — follow the winner",
            "contrarian": "Recovering from Broken/Fragile — higher risk/reward",
        }.get(screen_type, screen_type),
        "note": (
            "Screens surface candidates, not conclusions. Each passed idea needs fundamental "
            "research (via Claude FS) before trading. The best ideas come from intersections: "
            "statistically valid pairs with a fundamental catalyst."
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 7. PORTFOLIO ANALYSIS PROMPT FOR CLAUDE FS
#    (Combines equity research + wealth management methodologies)
# ═══════════════════════════════════════════════════════════════════════════════

PORTFOLIO_ANALYSIS_SYSTEM_PROMPT = """You are a senior portfolio strategist at an institutional asset manager.
You are producing a PORTFOLIO HEALTH REPORT for a pairs trading portfolio.

You have access to web search. Use it to research sector themes, macro factors,
and any events affecting portfolio positions.

## YOUR ROLE
1. ASSESS portfolio-level risk: concentration, correlation, regime distribution
2. RESEARCH macro themes affecting the portfolio's sector exposures
3. IDENTIFY which pairs need immediate attention and why
4. RECOMMEND portfolio-level actions: rebalancing, hedging, new ideas, exits

## SEARCH STRATEGY (4-6 searches)
1. Search for macro themes: "[sector] outlook [year]" for top sector exposures
2. Search for market regime: "market volatility VIX [year]" or "correlation regime"
3. Search for sector rotation: "sector rotation [year]" — which sectors in/out of favor
4. Search for any breaking news on positions flagged as CRITICAL

## OUTPUT STRUCTURE

## PORTFOLIO CONTEXT
Brief assessment of overall portfolio health based on the data provided.
Note: total pairs, regime distribution, average validity, sector concentration.

## MACRO ENVIRONMENT (from web search)
How does the current macro environment affect this portfolio?
Rate sensitivity, sector rotation, volatility regime, correlation trends.
Which sector exposures are tailwinds vs headwinds?

## CRITICAL POSITIONS
Pairs that need immediate attention: Broken/Fragile regime, low validity, 
high failure mode count. For each: what happened and what to do.

## REBALANCING ASSESSMENT
Is the portfolio well-balanced? Sector concentration risks?
Any pairs that have drifted significantly from targets?
Regime-aware rebalancing: don't add to broken pairs.

## UPCOMING PORTFOLIO CATALYSTS
Earnings dates, events, macro releases that affect multiple positions.
Next 2 weeks: what events could move which pairs?

## RECOMMENDATIONS (3-5 bullets)
Specific, actionable portfolio-level recommendations:
- Which pairs to trim, add, close, or replace
- New pair ideas based on current themes (if relevant)
- Hedging suggestions for concentrated exposures
- Timing considerations based on catalyst calendar

Keep under 500 words. Be opinionated and specific."""


def build_portfolio_claude_fs_prompt(
    drift_analysis: Dict[str, Any],
    thesis_review: Dict[str, Any],
    performance: Dict[str, Any],
    tlh_opportunities: Optional[Dict[str, Any]] = None,
) -> str:
    """Build a portfolio-level prompt for Claude FS analysis."""
    
    lines = ["# PORTFOLIO HEALTH CHECK — PAIRS TRADING PORTFOLIO\n"]
    
    # Summary stats
    lines.append(f"## PORTFOLIO OVERVIEW")
    lines.append(f"Total value: ${drift_analysis.get('portfolio_value', 0):,.0f}")
    lines.append(f"Active pairs: {drift_analysis.get('pair_count', 0)}")
    lines.append(f"Health score: {drift_analysis.get('health_score', 0)}/100")
    lines.append(f"Portfolio return: {performance.get('portfolio_return_pct', 0):+.1f}%")
    lines.append(f"Win rate: {performance.get('win_rate_pct', 0):.0f}%")
    lines.append(f"Avg Sharpe: {performance.get('avg_sharpe', 0):.2f}")
    lines.append("")
    
    # Sector concentration
    sectors = drift_analysis.get("sector_weights", {})
    if sectors:
        lines.append("## SECTOR EXPOSURE")
        for sector, weight in sectors.items():
            lines.append(f"  {sector}: {weight:.1f}%")
        if drift_analysis.get("sector_concentration_warning"):
            lines.append("  ⚠️ CONCENTRATION WARNING: >40% in single sector")
        lines.append("")
    
    # Thesis review summary
    lines.append(f"## THESIS HEALTH")
    lines.append(f"Portfolio thesis: {thesis_review.get('portfolio_thesis_health', 'unknown')}")
    sd = thesis_review.get("status_distribution", {})
    lines.append(f"  INTACT: {sd.get('INTACT', 0)} | MONITORING: {sd.get('MONITORING', 0)} | "
                 f"DEGRADED: {sd.get('DEGRADED', 0)} | BROKEN: {sd.get('BROKEN', 0)}")
    lines.append(f"  Action required on {thesis_review.get('action_required', 0)} pair(s)")
    lines.append("")
    
    # Critical positions
    critical = [r for r in thesis_review.get("reviews", []) 
                if r["overall_status"] in ("BROKEN", "DEGRADED")]
    if critical:
        lines.append("## CRITICAL POSITIONS (need immediate attention)")
        for r in critical:
            lines.append(f"  {r['pair_id']}: {r['overall_status']} — {r['action']}")
            for p in r.get("pillars", []):
                lines.append(f"    {p['name']}: {p['status']} — {p['note']}")
        lines.append("")
    
    # Drift summary
    drifted = [d for d in drift_analysis.get("drift_analysis", []) if d["outside_band"]]
    if drifted:
        lines.append("## DRIFTED POSITIONS")
        for d in drifted:
            lines.append(f"  {d['pair_id']}: {d['current_weight']:.1f}% (target {d['target_weight']:.1f}%) "
                         f"→ drift {d['drift_pct']:+.1f}% | regime: {d['regime']}")
        lines.append("")
    
    # TLH
    if tlh_opportunities and tlh_opportunities.get("opportunity_count", 0) > 0:
        lines.append(f"## TAX-LOSS HARVESTING OPPORTUNITIES")
        lines.append(f"Total harvestable: ${tlh_opportunities['total_harvestable_loss']:,.0f}")
        lines.append(f"Estimated tax savings: ${tlh_opportunities['total_estimated_tax_savings']:,.0f}")
        for opp in tlh_opportunities.get("opportunities", [])[:5]:
            lines.append(f"  {opp['pair_id']}: ${abs(opp['unrealized_loss']):,.0f} loss "
                         f"({opp['holding_period']}) → saves ${opp['estimated_tax_savings']:,.0f}")
        lines.append("")
    
    # Performance by pair
    lines.append("## PAIR PERFORMANCE")
    for pp in performance.get("pair_performance", []):
        lines.append(f"  {pp['pair_id']}: {pp['return_pct']:+.1f}% | "
                     f"Sharpe {pp['sharpe']:.2f} | {pp['regime']} | validity {pp['validity']}")
    
    return "\n".join(lines)
