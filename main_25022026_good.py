"""
Signals SaaS - Main FastAPI Application

Investment-grade analysis API for portfolio analysis.
WITH BAVELLA v2.2 VALIDITY ENGINE INTEGRATION
"""

from __future__ import annotations
from contextlib import asynccontextmanager
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Header, Depends, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import httpx
from rapidfuzz import fuzz
import numpy as np

from engine.auth import ClerkUser, check_rate_limit
from database import init_database, close_database
from database.routes import router as db_router
from engine.custom_routes import router as custom_router
from engine.validity.routes import router as validity_router

# =============================================================================
# BAVELLA v2.2 INTEGRATION
# =============================================================================
# Using bavella_adapter instead of old validity routes for check_portfolio_validity
from engine.bavella_adapter import check_portfolio_validity

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# CUSTOM JSON ENCODER FOR NUMPY TYPES
# =============================================================================

class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if hasattr(obj, 'isoformat'):
            return obj.isoformat()
        return super().default(obj)


def json_response(data: Any, status_code: int = 200) -> Response:
    """Create a JSON response with numpy-safe encoding."""
    content = json.dumps(data, cls=NumpyEncoder)
    return Response(content=content, status_code=status_code, media_type="application/json")


# =============================================================================
# APP SETUP
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        await init_database()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
    yield
    await close_database()
    logger.info("Database connections closed")


app = FastAPI(
    title="Signals SaaS API",
    description="Investment-grade portfolio analysis and regime detection",
    version="2.2.0",  # Updated version for Bavella v2.2 integration
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(db_router, prefix="/api", tags=["database"])
app.include_router(custom_router, tags=["custom-data"])
app.include_router(validity_router, prefix="/api/validity", tags=["validity"])

YAHOO_SEARCH_URL = "https://query1.finance.yahoo.com/v1/finance/search"


# =============================================================================
# REQUEST MODELS
# =============================================================================

class PositionInput(BaseModel):
    ticker: str
    amount: float


class AnalysisRequest(BaseModel):
    positions: List[PositionInput] = Field(..., min_items=1, max_items=20)
    analysis_period_days: int = Field(default=180, ge=30, le=365)
    include_ai_memo: bool = Field(default=True)
    portfolio_name: Optional[str] = Field(default="Portfolio Analysis")
    analysis_end_date: Optional[str] = Field(default=None, description="Override end date (YYYY-MM-DD). Defaults to today.")


class QuickSignalRequest(BaseModel):
    canonical_id: str


# =============================================================================
# HEALTH CHECK
# =============================================================================

@app.get("/health")
async def health():
    from database import check_database_connection
    db_healthy = await check_database_connection()
    return {"ok": True, "timestamp": datetime.now().isoformat(), "database": "connected" if db_healthy else "disconnected"}


# =============================================================================
# TICKER RESOLUTION
# =============================================================================

def _normalize_query(q: str) -> str:
    return " ".join((q or "").strip().split())


def _strip_bbg_noise(q: str) -> str:
    q = q.upper().replace(" EQUITY", "").replace(" INDEX", "").replace(" CORP", "").strip()
    parts = q.split()
    if len(parts) >= 2 and parts[-1] in {"US", "UW", "UN", "FP", "LN", "GR", "GY", "NA", "SW", "IM"}:
        return parts[0]
    return q


def _rank_equities(q_raw: str, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    q = _normalize_query(q_raw)
    q_upper = q.upper()
    q_simpl = _strip_bbg_noise(q_upper)

    ranked = []
    for it in items:
        symbol = (it.get("symbol") or "").strip()
        name = (it.get("shortname") or it.get("longname") or it.get("name") or "").strip()
        exch = (it.get("exchDisp") or it.get("exchange") or "").strip()
        curr = (it.get("currency") or "").strip()

        score = 0
        if symbol.upper() == q_upper or symbol.upper() == q_simpl:
            score += 100
        if symbol.upper().startswith(q_simpl):
            score += 25
        if name:
            score += int(0.45 * fuzz.partial_ratio(q_upper, name.upper()))
        if exch:
            score += 1
        if curr:
            score += 1

        ranked.append({
            "name": name or symbol,
            "provider": "yahoo",
            "provider_symbol": symbol,
            "type": "EQUITY",
            "exchange": exch,
            "currency": curr,
            "confidence": min(score / 140.0, 0.999),
            "_score": score,
        })

    ranked.sort(key=lambda x: x["_score"], reverse=True)
    return ranked


def _auto_select(ranked: List[Dict[str, Any]]) -> bool:
    if not ranked:
        return False
    if ranked[0]["confidence"] < 0.72:
        return False
    if len(ranked) == 1:
        return True
    return (ranked[0]["_score"] - ranked[1]["_score"]) >= 18


@app.get("/resolve")
async def resolve(q: str):
    q = _normalize_query(q)
    if not q:
        return {"auto_selected": False, "candidates": []}

    q_for_yahoo = _strip_bbg_noise(q)
    params = {"q": q_for_yahoo, "quotesCount": 20, "newsCount": 0, "enableFuzzyQuery": True}

    try:
        async with httpx.AsyncClient(timeout=10.0, headers={"User-Agent": "Mozilla/5.0"}) as client:
            r = await client.get(YAHOO_SEARCH_URL, params=params)
            r.raise_for_status()
            data = r.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Yahoo search failed: {type(e).__name__}: {e}")

    quotes = data.get("quotes") or []
    # Include: EQUITY, ETF, INDEX, MUTUALFUND, FUTURE, CURRENCY, CRYPTOCURRENCY
    ALLOWED_TYPES = {"EQUITY", "ETF", "INDEX", "MUTUALFUND", "FUTURE", "CURRENCY", "CRYPTOCURRENCY"}
    filtered = [x for x in quotes if x.get("symbol") and (x.get("quoteType") or "").upper() in ALLOWED_TYPES]
    ranked = _rank_equities(q, filtered)
    top5 = [{k: v for k, v in c.items() if k != "_score"} for c in ranked[:5]]

    return {"auto_selected": _auto_select(ranked), "candidates": top5}


# =============================================================================
# FAILURE MODE EXTRACTION FOR MEMO
# =============================================================================

def _extract_failure_modes(validity_output) -> list:
    """
    Extract failure mode signals from ValidityOutput into the format
    expected by the decision brief memo generator.
    
    The validity output has root_cause and secondary_failures — we need
    to reconstruct a flat list of FM detections for the memo prompt.
    """
    failure_modes = []
    vdict = validity_output.to_dict()
    
    # Extract root cause as primary FM
    root = vdict.get("validity", {}).get("root_cause")
    if root and isinstance(root, dict):
        failure_modes.append({
            "failure_mode_id": root.get("code", "UNKNOWN"),
            "detected": True,
            "severity": root.get("severity", 0),
            "confidence": {"level": "high", "overall": 80},
            "explanation": root.get("label", "") + " — " + root.get("summary", ""),
            "triggers_kill": root.get("severity", 0) >= 80,
            "evidence": {},
            "is_root_cause": True,
        })
    
    # Extract secondary failures
    secondaries = vdict.get("details", {}).get("secondary_failures", [])
    for sf in secondaries:
        if isinstance(sf, dict):
            failure_modes.append({
                "failure_mode_id": sf.get("code", "UNKNOWN"),
                "detected": True,
                "severity": sf.get("severity", 0),
                "confidence": {"level": "medium", "overall": 60},
                "explanation": sf.get("label", "") + " — " + sf.get("summary", ""),
                "triggers_kill": False,
                "evidence": {},
                "is_root_cause": False,
            })
    
    # Extract competing causes if available
    competing = vdict.get("details", {}).get("competing_causes", [])
    for cc in competing:
        if isinstance(cc, dict):
            # Only add if not already in the list
            code = cc.get("code", "")
            if code and not any(fm.get("failure_mode_id") == code for fm in failure_modes):
                failure_modes.append({
                    "failure_mode_id": code,
                    "detected": True,
                    "severity": cc.get("severity", 0),
                    "confidence": {"level": "low", "overall": 40},
                    "explanation": cc.get("label", "") + " — " + cc.get("summary", ""),
                    "triggers_kill": False,
                    "evidence": {},
                    "is_root_cause": False,
                })
    
    # If no FMs found at all but validity is degraded/invalid, create a generic signal
    if not failure_modes and validity_output.state != "VALID":
        summary = vdict.get("validity", {}).get("summary", "Validity degraded")
        failure_modes.append({
            "failure_mode_id": "COMPOSITE",
            "detected": True,
            "severity": max(0, 100 - (validity_output.score or 0)),
            "confidence": {"level": "medium", "overall": int((validity_output.confidence or 0.5) * 100)},
            "explanation": summary,
            "triggers_kill": validity_output.state == "INVALID",
            "evidence": {},
            "is_root_cause": True,
        })
    
    return failure_modes


# =============================================================================
# FULL ANALYSIS ENDPOINT
# =============================================================================

@app.post("/analyze")
async def analyze(
    request: AnalysisRequest,
    user: ClerkUser = Depends(check_rate_limit),
    x_openai_key: Optional[str] = Header(None, alias="X-OpenAI-Key"),
):
    logger.info(f"User {user.user_id} requested analysis")
    
    from engine import (
        fetch_multiple_tickers, calculate_returns, PortfolioDefinition,
        build_portfolio_returns, run_full_backtest, detect_regime,
        generate_trading_signals, create_all_charts,
        DataFetchError, InsufficientDataError,
    )
    from engine.stats_analysis import calculate_enhanced_statistics, calculate_memo_stats
    from engine.memo import generate_memo, fetch_pair_fundamentals
    from engine.report import generate_html_report
    
    logger.info(f"Starting analysis for {len(request.positions)} positions")
    
    positions_dict = {p.ticker: p.amount for p in request.positions}
    portfolio = PortfolioDefinition.from_dict(positions_dict)
    
    logger.info(f"Portfolio type: {portfolio.portfolio_type}")
    logger.info(f"Gross exposure: ${portfolio.gross_exposure:,.0f}")
    
    # Step 1: Fetch data
    try:
        logger.info("Fetching market data...")
        # Fetch extra data for indicator calculations
        fetch_days = max(400, request.analysis_period_days + 100)
        # Minimum required is 70% of requested period or 60 days, whichever is higher
        min_required = max(60, int(request.analysis_period_days * 0.7))
        price_data = fetch_multiple_tickers(tickers=portfolio.tickers, days=fetch_days, min_required_days=min_required)
        
        # If analysis_end_date specified, truncate data to that date
        # This enables historical backtesting without changing the data fetcher
        if request.analysis_end_date:
            import pandas as pd
            end_dt = pd.Timestamp(request.analysis_end_date)
            logger.info(f"Truncating data to end date: {end_dt.date()}")
            price_data = {
                ticker: df[df.index <= end_dt]
                for ticker, df in price_data.items()
            }
            # Verify we still have enough data after truncation
            for ticker, df in price_data.items():
                if len(df) < min_required:
                    raise InsufficientDataError(
                        f"{ticker}: only {len(df)} days before {end_dt.date()}, need {min_required}"
                    )
    except (DataFetchError, InsufficientDataError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    # Step 2: Calculate returns
    logger.info("Calculating returns...")
    stock_returns = calculate_returns(price_data)
    
    # Step 3: Run backtest
    logger.info("Running backtest...")
    backtest_result = run_full_backtest(
        stock_returns=stock_returns,
        portfolio=portfolio,
        primary_period=request.analysis_period_days,
        additional_periods=[14, 30],
    )
    
    # Step 4: Build returns for regime analysis
    logger.info("Building portfolio returns for regime analysis...")
    portfolio_returns = build_portfolio_returns(
        stock_returns=stock_returns,
        portfolio=portfolio,
        analysis_days=request.analysis_period_days
    )
    
    # Step 5: Regime detection
    logger.info("Running regime detection...")
    regime_df, regime_summary, indicators, z_score = detect_regime(
        cumulative=portfolio_returns.cumulative,
        daily_returns=portfolio_returns.daily_returns,
        window=60
    )
    
    # DEBUG: Log regime_summary keys to understand structure
    regime_dict = regime_summary.to_dict()
    logger.info(f"Regime summary keys: {list(regime_dict.keys())}")
    if 'metrics' in regime_dict:
        logger.info(f"Metrics keys: {list(regime_dict['metrics'].keys()) if isinstance(regime_dict['metrics'], dict) else 'not a dict'}")
    logger.info(f"Z-score from detect_regime: {z_score}")
    
    # =========================================================================
    # Step 6: Validity analysis (BAVELLA v2.2)
    # =========================================================================
    logger.info("Running validity analysis (Bavella v2.2)...")
    
    # Extract scalar z_score (it may be a Series)
    z_score_scalar = None
    if z_score is not None:
        if hasattr(z_score, 'iloc'):
            z_score_scalar = float(z_score.iloc[-1])
        else:
            z_score_scalar = float(z_score)
    
    validity_output = check_portfolio_validity(
        portfolio_returns=portfolio_returns.daily_returns,
        regime_summary=regime_summary.to_dict(),
        portfolio_name=request.portfolio_name or "Portfolio",
        owner_id=user.user_id,
        z_score_override=z_score_scalar,
    )
    logger.info(f"Validity: {validity_output.score} ({validity_output.state})")
    
    # Step 7: Generate trading signals
    logger.info("Generating trading signals...")
    signals = generate_trading_signals(regime_df, indicators, portfolio_returns.cumulative, z_score)
    
    # Step 8: Calculate statistics
    logger.info("Calculating statistics...")
    enhanced_stats = calculate_enhanced_statistics(portfolio_returns.daily_returns)
    memo_stats = calculate_memo_stats(portfolio_returns.daily_returns, portfolio_returns.cumulative)
    
    # Add CVaR (Conditional VaR) if not already present — average loss beyond VaR threshold
    if 'cvar_95' not in memo_stats:
        try:
            dr = portfolio_returns.daily_returns.dropna()
            var_threshold = dr.quantile(0.05)
            tail_losses = dr[dr <= var_threshold]
            memo_stats['cvar_95'] = float(tail_losses.mean()) if len(tail_losses) > 0 else None
        except Exception:
            memo_stats['cvar_95'] = None
    
    # Step 9: Generate charts
    logger.info("Generating charts...")
    charts = create_all_charts(
        cumulative=portfolio_returns.cumulative,
        daily_returns=portfolio_returns.daily_returns,
        regime_df=regime_df,
        indicators=indicators,
        z_score=z_score,
        signals=signals,
        portfolio_name=request.portfolio_name or "Portfolio"
    )
    
    # =========================================================================
    # Step 9b: Fetch fundamental data for memo context (valuation, margins)
    # =========================================================================
    fundamental_data = {}
    if request.include_ai_memo:
        try:
            logger.info("Fetching fundamental data for memo...")
            fundamental_data = fetch_pair_fundamentals(portfolio.tickers)
            logger.info(f"Fundamentals fetched for {len(fundamental_data)} tickers")
        except Exception as e:
            logger.warning(f"Fundamental data fetch failed (non-fatal): {e}")
            fundamental_data = {}
    
    # =========================================================================
    # Step 10: Generate AI memo (DECISION BRIEF with FM signals)
    # =========================================================================
    memo = None
    if request.include_ai_memo:
        logger.info("Generating decision brief memo...")
        openai_key = x_openai_key or os.environ.get("OPENAI_API_KEY")
        if openai_key:
            # Extract failure mode signals from validity output for the memo
            validity_dict = validity_output.to_dict()
            failure_modes_for_memo = _extract_failure_modes(validity_output)
            
            memo = await generate_memo(
                enhanced_stats=memo_stats,
                regime_summary=regime_summary.to_dict(),
                portfolio_name=request.portfolio_name or "Portfolio",
                long_positions=portfolio.long_weights,
                short_positions=portfolio.short_weights,
                api_key=openai_key,
                model="gpt-4o",
                validity_data=validity_dict,
                failure_modes=failure_modes_for_memo,
                pair_state=validity_output.state,  # VALID/DEGRADED/INVALID
                fundamental_data=fundamental_data,
            )
    
    # Step 11: Generate HTML report
    logger.info("Generating HTML report...")
    html_report = generate_html_report(
        enhanced_stats=memo_stats,
        regime_summary=regime_summary.to_dict(),
        memo_text=memo,
        long_positions=portfolio.long_weights,
        short_positions=portfolio.short_weights,
        portfolio_name=request.portfolio_name or "Portfolio Analysis",
        regime_chart_base64=charts.get("regime"),
        performance_chart_base64=charts.get("performance"),
        distribution_chart_base64=charts.get("distribution"),
        validity_data=validity_output.to_dict(),
        analysis_period_days=request.analysis_period_days,
    )
    
    logger.info("Analysis complete!")
    
    # =========================================================================
    # Build response (using FROZEN JSON CONTRACT for validity)
    # =========================================================================
    response = {
        "portfolio": {
            "type": portfolio.portfolio_type,
            "gross_exposure": float(portfolio.gross_exposure),
            "net_exposure": float(portfolio.net_exposure),
            "positions": [
                {"ticker": p.ticker, "amount": float(p.amount), "type": "LONG" if p.amount > 0 else "SHORT"}
                for p in portfolio.positions
            ],
        },
        "backtest": backtest_result.to_dict(),
        "regime": regime_summary.to_dict(),
        "validity": validity_output.to_dict(),  # Frozen JSON contract (includes all details)
        "statistics": enhanced_stats.to_dict() if hasattr(enhanced_stats, 'to_dict') else enhanced_stats,
        "signals": {
            "buy_signals": int(len(signals[signals == "BUY"])),
            "sell_signals": int(len(signals[signals == "SELL"])),
            "recent_signal": str(signals.iloc[-1]) if len(signals) > 0 else "HOLD",
        },
        "memo": memo,
        "html_report": html_report,
        "charts": {
            "regime": (charts.get("regime", "")[:100] + "...") if charts.get("regime") else "",
            "performance": (charts.get("performance", "")[:100] + "...") if charts.get("performance") else "",
            "distribution": (charts.get("distribution", "")[:100] + "...") if charts.get("distribution") else "",
        },
    }
    
    # Return with custom encoder to handle numpy types
    return json_response(response)


@app.post("/analyze/report", response_class=HTMLResponse)
async def analyze_report(
    request: AnalysisRequest,
    user: ClerkUser = Depends(check_rate_limit),
    x_openai_key: Optional[str] = Header(None, alias="X-OpenAI-Key"),
):
    result = await analyze(request, user, x_openai_key)
    content = json.loads(result.body)
    return HTMLResponse(content=content["html_report"])


# =============================================================================
# BACKTEST ENDPOINT (no auth — local development only)
# =============================================================================

@app.post("/analyze/backtest")
async def analyze_backtest(
    request: AnalysisRequest,
    x_openai_key: Optional[str] = Header(None, alias="X-OpenAI-Key"),
):
    """
    Same as /analyze but without Clerk auth.
    For local backtesting scripts only.
    """
    try:
        dummy_user = ClerkUser(user_id="backtest", session_id="backtest")
    except TypeError:
        dummy_user = ClerkUser(user_id="backtest")
    return await analyze(request, dummy_user, x_openai_key)


# =============================================================================
# QUICK SIGNAL ENDPOINT
# =============================================================================

@app.post("/signal")
async def quick_signal(payload: QuickSignalRequest):
    canonical_id = payload.canonical_id.strip()
    
    if not canonical_id or ":" not in canonical_id:
        raise HTTPException(status_code=400, detail="canonical_id required, e.g. 'yahoo:AAPL'")
    
    provider, symbol = canonical_id.split(":", 1)
    if provider.strip().lower() != "yahoo":
        raise HTTPException(status_code=400, detail="Only 'yahoo' provider supported")
    
    from data_fetcher import fetch_ticker_data, InsufficientDataError, DataFetchError
    
    try:
        df = fetch_ticker_data(symbol.strip(), days=365, min_required_days=220)
    except (DataFetchError, InsufficientDataError) as e:
        raise HTTPException(status_code=502, detail=str(e))
    
    closes = df["Close"].values
    
    last = float(closes[-1])
    sma20 = float(closes[-20:].mean())
    sma50 = float(closes[-50:].mean())
    sma200 = float(closes[-200:].mean())
    
    if last > sma50 > sma200:
        sig, regime, confidence = "LONG", "TREND_UP", 70 + (6 if last > sma20 else 0)
    elif last < sma50 < sma200:
        sig, regime, confidence = "SHORT", "TREND_DOWN", 70 + (6 if last < sma20 else 0)
    else:
        sig, regime, confidence = "NEUTRAL", "RANGE", 55
    
    dist20 = (last / sma20 - 1) * 100
    risk_notes = []
    vol = "HIGH" if abs(dist20) > 6 else ("MEDIUM" if abs(dist20) > 3 else "LOW")
    if abs(dist20) > 6:
        risk_notes.append("Extended vs 20D mean (mean reversion risk)")
    risk_notes.append(f"Price vs 20D: {dist20:+.1f}%; 50D={sma50:.2f}, 200D={sma200:.2f}")
    
    return {
        "canonical_id": canonical_id,
        "signal": sig,
        "confidence": int(min(max(confidence, 1), 99)),
        "regime": regime,
        "volatility": vol,
        "risk_notes": risk_notes[:6],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
