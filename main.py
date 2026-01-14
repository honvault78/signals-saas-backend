"""
Signals SaaS - Main FastAPI Application

Investment-grade analysis API for portfolio analysis.
Produces output identical to the Jupyter notebook implementation.
"""

from __future__ import annotations
from contextlib import asynccontextmanager

from engine.auth import ClerkUser, get_current_user, check_rate_limit, get_optional_user
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import httpx
from rapidfuzz import fuzz

# Database imports
from database import init_database, close_database, get_db, crud
from database.routes import router as db_router
from sqlalchemy.ext.asyncio import AsyncSession

# Try to load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# Database Lifespan Handler
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database on startup, close on shutdown."""
    # Startup
    try:
        await init_database()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        # Continue anyway - app can work without DB for basic endpoints
    yield
    # Shutdown
    await close_database()
    logger.info("Database connections closed")


app = FastAPI(
    title="Signals SaaS API",
    description="Investment-grade portfolio analysis and regime detection",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS Configuration
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://192.168.1.16:3000",
    "*",  # Allow all for development - restrict in production
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# Database Routes
# =============================================================================

app.include_router(db_router, prefix="/api", tags=["database"])

# Yahoo Finance URLs
YAHOO_SEARCH_URL = "https://query1.finance.yahoo.com/v1/finance/search"


# =============================================================================
# Request/Response Models
# =============================================================================

class PositionInput(BaseModel):
    """Single position in the portfolio."""
    ticker: str = Field(..., description="Yahoo Finance ticker symbol")
    amount: float = Field(..., description="Dollar amount (positive=long, negative=short)")


class AnalysisRequest(BaseModel):
    """Request for full portfolio analysis."""
    positions: List[PositionInput] = Field(..., min_items=1, max_items=20)
    analysis_period_days: int = Field(default=180, ge=30, le=365)
    include_ai_memo: bool = Field(default=True)
    portfolio_name: Optional[str] = Field(default="Portfolio Analysis")


class QuickSignalRequest(BaseModel):
    """Request for quick signal on single ticker."""
    canonical_id: str = Field(..., description="Provider:Symbol format, e.g., 'yahoo:AAPL'")


# =============================================================================
# Health Check
# =============================================================================

@app.get("/health")
async def health():
    """Health check endpoint with database status."""
    from database import check_database_connection
    
    db_healthy = await check_database_connection()
    
    return {
        "ok": True,
        "timestamp": datetime.now().isoformat(),
        "database": "connected" if db_healthy else "disconnected",
    }


# =============================================================================
# Ticker Resolution (from your existing code)
# =============================================================================

def _normalize_query(q: str) -> str:
    return " ".join((q or "").strip().split())


def _strip_bbg_noise(q: str) -> str:
    """Handle Bloomberg-style ticker inputs."""
    q = q.upper().replace(" EQUITY", "").replace(" INDEX", "").replace(" CORP", "").strip()
    parts = q.split()
    # Map common Bloomberg exchange suffixes to nothing (we'll search Yahoo)
    if len(parts) >= 2 and parts[-1] in {"US", "UW", "UN", "FP", "LN", "GR", "GY", "NA", "SW", "IM"}:
        return parts[0]
    return q


def _rank_equities(q_raw: str, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rank search results by relevance."""
    q = _normalize_query(q_raw)
    q_upper = q.upper()
    q_simpl = _strip_bbg_noise(q_upper)

    ranked: List[Dict[str, Any]] = []
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
    """Determine if top result should be auto-selected."""
    if not ranked:
        return False
    if ranked[0]["confidence"] < 0.72:
        return False
    if len(ranked) == 1:
        return True
    return (ranked[0]["_score"] - ranked[1]["_score"]) >= 18


@app.get("/resolve")
async def resolve(q: str):
    """
    Resolve ticker search query to Yahoo Finance symbols.
    
    Supports:
    - Company names (e.g., "Apple", "Airbus")
    - Ticker symbols (e.g., "AAPL", "AIR.PA")
    - Bloomberg-style tickers (e.g., "AAPL US Equity", "AIR FP Equity")
    """
    q = _normalize_query(q)
    if not q:
        return {"auto_selected": False, "candidates": []}

    q_for_yahoo = _strip_bbg_noise(q)

    params = {
        "q": q_for_yahoo,
        "quotesCount": 20,
        "newsCount": 0,
        "enableFuzzyQuery": True,
    }

    try:
        async with httpx.AsyncClient(timeout=10.0, headers={"User-Agent": "Mozilla/5.0"}) as client:
            r = await client.get(YAHOO_SEARCH_URL, params=params)
            r.raise_for_status()
            data = r.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Yahoo search failed: {type(e).__name__}: {e}")

    quotes = data.get("quotes") or []
    
    # Filter to equities only
    equities = [
        x for x in quotes
        if x.get("symbol") and (x.get("quoteType") or "").upper() == "EQUITY"
    ]

    ranked = _rank_equities(q, equities)
    top5 = [{k: v for k, v in c.items() if k != "_score"} for c in ranked[:5]]

    return {"auto_selected": _auto_select(ranked), "candidates": top5}


# =============================================================================
# Full Analysis Endpoint
# =============================================================================

@app.post("/analyze")
async def analyze(
    request: AnalysisRequest,
    user: ClerkUser,  # Add this parameter
    x_openai_key: Optional[str] = None,
):
    """
    Run complete portfolio analysis.
    
    This is the main endpoint that:
    1. Fetches historical data for all tickers
    2. Constructs portfolio and calculates returns
    3. Runs backtest for multiple periods
    4. Performs regime detection
    5. Calculates statistical metrics
    6. Generates charts
    7. Optionally generates AI memo
    8. Produces HTML report (matching Jupyter notebook exactly)
    
    Pass OpenAI API key via X-OpenAI-Key header or set OPENAI_API_KEY env var.
    """
    logger.info(f"User {user.user_id} requested analysis")
    # Import from engine module
    from engine import (
        fetch_multiple_tickers,
        calculate_returns,
        PortfolioDefinition,
        build_portfolio_returns,
        run_full_backtest,
        detect_regime,
        generate_trading_signals,
        create_all_charts,
        DataFetchError,
        InsufficientDataError,
    )
    from engine.stats_analysis import calculate_enhanced_statistics, calculate_memo_stats
    from engine.memo import generate_memo
    from engine.report import generate_html_report
    
    logger.info(f"Starting analysis for {len(request.positions)} positions")
    
    # Build portfolio definition
    positions_dict = {p.ticker: p.amount for p in request.positions}
    portfolio = PortfolioDefinition.from_dict(positions_dict)
    
    logger.info(f"Portfolio type: {portfolio.portfolio_type}")
    logger.info(f"Gross exposure: ${portfolio.gross_exposure:,.0f}")
    
    # Step 1: Fetch data
    try:
        logger.info("Fetching market data...")
        price_data = fetch_multiple_tickers(
            tickers=portfolio.tickers,
            days=400,  # Extra buffer for indicators
            min_required_days=220
        )
    except DataFetchError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except InsufficientDataError as e:
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
    
    # Step 6: Generate trading signals
    logger.info("Generating trading signals...")
    signals = generate_trading_signals(regime_df, indicators, portfolio_returns.cumulative, z_score)
    
    # Step 7: Calculate enhanced statistics
    logger.info("Calculating statistics...")
    enhanced_stats = calculate_enhanced_statistics(portfolio_returns.daily_returns)
    memo_stats = calculate_memo_stats(portfolio_returns.daily_returns, portfolio_returns.cumulative)
    
    # Step 8: Generate charts
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
    
    # Step 9: Generate AI memo (optional)
    memo = None
    if request.include_ai_memo:
        logger.info("Generating AI memo...")
        # Use header key if provided, otherwise fall back to env var
        openai_key = x_openai_key or os.environ.get("OPENAI_API_KEY")
        if openai_key:
            memo = await generate_memo(
                enhanced_stats=memo_stats,
                regime_summary=regime_summary.to_dict(),
                portfolio_name=request.portfolio_name or "Portfolio",
                long_positions=portfolio.long_weights,
                short_positions=portfolio.short_weights,
                api_key=openai_key,
                model="gpt-4o"
            )
        else:
            logger.warning("No OpenAI API key provided, skipping AI memo")
    
    # Step 10: Generate HTML report (MATCHING NOTEBOOK EXACTLY)
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
    )
    
    logger.info("Analysis complete!")
    
    # Build response
    return {
        "portfolio": {
            "type": portfolio.portfolio_type,
            "gross_exposure": portfolio.gross_exposure,
            "net_exposure": portfolio.net_exposure,
            "positions": [
                {"ticker": p.ticker, "amount": p.amount, "type": "LONG" if p.amount > 0 else "SHORT"}
                for p in portfolio.positions
            ],
        },
        "backtest": backtest_result.to_dict(),
        "regime": regime_summary.to_dict(),
        "statistics": enhanced_stats.to_dict() if hasattr(enhanced_stats, 'to_dict') else enhanced_stats,
        "signals": {
            "buy_signals": len(signals[signals == "BUY"]),
            "sell_signals": len(signals[signals == "SELL"]),
            "recent_signal": signals.iloc[-1] if len(signals) > 0 else "HOLD",
        },
        "memo": memo,
        "html_report": html_report,
        "charts": {
            "regime": charts.get("regime", "")[:100] + "..." if charts.get("regime") else "",
            "performance": charts.get("performance", "")[:100] + "..." if charts.get("performance") else "",
            "distribution": charts.get("distribution", "")[:100] + "..." if charts.get("distribution") else "",
        },
    }


@app.post("/analyze/report", response_class=HTMLResponse)
async def analyze_report(
    request: AnalysisRequest,
    user: ClerkUser = Depends(check_rate_limit),
    x_openai_key: Optional[str] = Header(None, alias="X-OpenAI-Key"),
):
    """
    Run analysis and return HTML report directly.
    
    This endpoint is optimized for displaying the report in an iframe.
    """
    result = await analyze(request, user, x_openai_key)  # Pass user here
    return HTMLResponse(content=result["html_report"])


# =============================================================================
# Quick Signal Endpoint (simplified, for backwards compatibility)
# =============================================================================

@app.post("/signal")
async def quick_signal(payload: QuickSignalRequest):
    """
    Quick signal for a single instrument.
    
    This is a simplified endpoint for quick checks.
    For full analysis, use /analyze.
    """
    canonical_id = payload.canonical_id.strip()
    
    if not canonical_id or ":" not in canonical_id:
        raise HTTPException(status_code=400, detail="canonical_id required, e.g. 'yahoo:AAPL'")
    
    provider, symbol = canonical_id.split(":", 1)
    provider = provider.strip().lower()
    symbol = symbol.strip()
    
    if provider != "yahoo":
        raise HTTPException(status_code=400, detail="Only 'yahoo' provider supported")
    
    # Simplified analysis
    from data_fetcher import fetch_ticker_data, InsufficientDataError, DataFetchError
    
    try:
        df = fetch_ticker_data(symbol, days=365, min_required_days=220)
    except (DataFetchError, InsufficientDataError) as e:
        raise HTTPException(status_code=502, detail=str(e))
    
    closes = df["Close"].values
    
    # Simple SMA-based signal
    def sma(arr, n):
        return arr[-n:].mean()
    
    last = closes[-1]
    sma20 = sma(closes, 20)
    sma50 = sma(closes, 50)
    sma200 = sma(closes, 200)
    
    trend_up = last > sma50 > sma200
    trend_down = last < sma50 < sma200
    
    if trend_up:
        sig = "LONG"
        regime = "TREND_UP"
        confidence = 70 + (6 if last > sma20 else 0)
    elif trend_down:
        sig = "SHORT"
        regime = "TREND_DOWN"
        confidence = 70 + (6 if last < sma20 else 0)
    else:
        sig = "NEUTRAL"
        regime = "RANGE"
        confidence = 55
    
    dist20 = (last / sma20 - 1) * 100
    
    risk_notes = []
    vol = "MEDIUM"
    if abs(dist20) > 6:
        vol = "HIGH"
        risk_notes.append("Extended vs 20D mean (mean reversion risk)")
    elif abs(dist20) > 3:
        risk_notes.append("Moderately extended vs 20D mean")
    else:
        vol = "LOW"
    
    risk_notes.append(f"Price vs 20D: {dist20:+.1f}%; 50D={sma50:.2f}, 200D={sma200:.2f}")
    
    return {
        "canonical_id": canonical_id,
        "signal": sig,
        "confidence": int(min(max(confidence, 1), 99)),
        "regime": regime,
        "volatility": vol,
        "risk_notes": risk_notes[:6],
    }


# =============================================================================
# Run with uvicorn
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
