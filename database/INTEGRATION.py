"""
Database Integration Guide

This file shows exactly how to integrate the database into your existing main.py.
Follow these steps to add persistent storage to your application.
"""

# =============================================================================
# STEP 1: Add imports to main.py (at the top)
# =============================================================================

# Add these imports:
"""
from contextlib import asynccontextmanager

from database import (
    init_database,
    close_database,
    get_db,
    crud,
)
from database.schemas import (
    AnalysisCreate,
    PositionBase,
)
from database.routes import router as db_router
"""


# =============================================================================
# STEP 2: Add lifespan handler for database (before app = FastAPI())
# =============================================================================

"""
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await init_database()
    logger.info("Database initialized")
    yield
    # Shutdown
    await close_database()
    logger.info("Database closed")

# Update your FastAPI app:
app = FastAPI(
    title="Signals SaaS API",
    lifespan=lifespan,  # Add this
)
"""


# =============================================================================
# STEP 3: Include the database routes (after app = FastAPI())
# =============================================================================

"""
# Include database routes
app.include_router(db_router, prefix="/api", tags=["database"])
"""


# =============================================================================
# STEP 4: Update the analyze endpoint to save results
# =============================================================================

# Here's the updated analyze function with auto-save:

"""
@app.post("/analyze")
async def analyze(
    request: AnalysisRequest,
    user: ClerkUser = Depends(check_rate_limit),
    x_openai_key: Optional[str] = Header(None, alias="X-OpenAI-Key"),
    db: AsyncSession = Depends(get_db),  # ADD THIS
):
    \"\"\"
    Run complete portfolio analysis.
    Results are automatically saved to history.
    \"\"\"
    import time
    start_time = time.time()
    
    logger.info(f"User {user.user_id} requested analysis")
    
    # Sync user to database
    db_user = await crud.sync_clerk_user(
        db,
        clerk_id=user.user_id,
        email=user.email or "",
        first_name=user.first_name,
        last_name=user.last_name,
    )
    
    # Check usage limits
    usage_count, usage_limit = await crud.increment_usage(db, db_user.id)
    if usage_count > usage_limit:
        raise HTTPException(
            status_code=429,
            detail=f"Monthly analysis limit reached ({usage_limit}). Upgrade your plan for more."
        )
    
    # ... your existing analysis code ...
    # (all the portfolio building, regime detection, chart generation, etc.)
    
    # After generating the report, save to database:
    duration_ms = int((time.time() - start_time) * 1000)
    
    # Build result summary for quick display
    result_summary = {
        "regime": regime_summary.current_regime,
        "signal": latest_signal,
        "trend_score": regime_summary.trend_score,
        "z_score": regime_summary.z_score,
        "rsi": regime_summary.rsi,
        "adf_pvalue": regime_summary.adf_pvalue,
        "cumulative_return": float(portfolio_returns.cumulative.iloc[-1] - 1) if len(portfolio_returns.cumulative) > 0 else 0,
        "volatility": float(enhanced_stats.volatility) if hasattr(enhanced_stats, 'volatility') else 0,
        "max_drawdown": float(enhanced_stats.max_drawdown) if hasattr(enhanced_stats, 'max_drawdown') else 0,
    }
    
    # Create analysis record
    analysis_data = AnalysisCreate(
        portfolio_name=request.portfolio_name or "Portfolio",
        positions=[PositionBase(ticker=p.ticker, amount=p.amount) for p in request.positions],
        analysis_period_days=request.analysis_period_days,
        result_summary=result_summary,
        html_report=html_report,  # The generated HTML
        ai_memo=memo,  # AI memo if generated
        duration_ms=duration_ms,
        portfolio_id=None,  # Link to saved portfolio if applicable
    )
    
    analysis = await crud.create_analysis(db, db_user.id, analysis_data)
    logger.info(f"Saved analysis {analysis.id} for user {db_user.id}")
    
    # Return your normal response
    return {
        "analysis_id": str(analysis.id),  # Include ID for reference
        "portfolio_name": request.portfolio_name,
        # ... rest of your response
    }
"""


# =============================================================================
# STEP 5: Full example of updated analyze endpoint
# =============================================================================

async def analyze_with_storage(
    request,  # AnalysisRequest
    user,     # ClerkUser
    x_openai_key,
    db,       # AsyncSession
):
    """
    Complete example showing where to add database calls.
    Copy the relevant parts into your existing analyze function.
    """
    import time
    start_time = time.time()
    
    # 1. Sync user to database (ADD THIS AT START)
    db_user = await crud.sync_clerk_user(
        db,
        clerk_id=user.user_id,
        email=user.email or "",
        first_name=user.first_name,
        last_name=user.last_name,
    )
    
    # 2. Check usage limits (ADD THIS)
    usage_count, usage_limit = await crud.increment_usage(db, db_user.id)
    if usage_count > usage_limit:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=429,
            detail=f"Monthly analysis limit reached ({usage_limit}). Upgrade your plan."
        )
    
    # ============================================
    # YOUR EXISTING ANALYSIS CODE GOES HERE
    # - Fetch prices
    # - Build portfolio
    # - Run regime detection
    # - Generate signals
    # - Create charts
    # - Generate HTML report
    # - Generate AI memo (optional)
    # ============================================
    
    # Placeholder for your existing variables:
    regime_summary = None  # Your regime summary
    latest_signal = "HOLD"  # Your signal
    enhanced_stats = None  # Your stats
    html_report = ""  # Your HTML report
    memo = None  # Your AI memo
    
    # 3. Calculate duration (ADD AFTER ANALYSIS)
    duration_ms = int((time.time() - start_time) * 1000)
    
    # 4. Build result summary (ADD THIS)
    result_summary = {
        "regime": regime_summary.current_regime if regime_summary else "unknown",
        "signal": latest_signal,
        "trend_score": regime_summary.trend_score if regime_summary else 0,
        "z_score": regime_summary.z_score if regime_summary else 0,
        "rsi": regime_summary.rsi if regime_summary else 50,
        "adf_pvalue": regime_summary.adf_pvalue if regime_summary else 0.5,
    }
    
    # 5. Save to database (ADD THIS)
    from database.schemas import AnalysisCreate, PositionBase
    
    analysis_data = AnalysisCreate(
        portfolio_name=request.portfolio_name or "Portfolio",
        positions=[PositionBase(ticker=p.ticker, amount=p.amount) for p in request.positions],
        analysis_period_days=request.analysis_period_days,
        result_summary=result_summary,
        html_report=html_report,
        ai_memo=memo,
        duration_ms=duration_ms,
    )
    
    analysis = await crud.create_analysis(db, db_user.id, analysis_data)
    
    # 6. Return response with analysis ID (ADD analysis_id TO RESPONSE)
    return {
        "analysis_id": str(analysis.id),
        # ... your other response fields
    }


# =============================================================================
# STEP 6: Update requirements.txt
# =============================================================================

"""
# Add these dependencies:
sqlalchemy>=2.0.0
asyncpg>=0.29.0
pydantic-settings>=2.0.0
"""


# =============================================================================
# STEP 7: Environment variable
# =============================================================================

"""
# Add to your .env file:
DATABASE_URL=postgresql+asyncpg://neondb_owner:YOUR_PASSWORD@ep-wild-snow-abaqlszu-pooler.eu-west-2.aws.neon.tech/neondb?sslmode=require
"""


# =============================================================================
# CHECKLIST
# =============================================================================

"""
[ ] 1. Run the migration SQL in Neon console
[ ] 2. Add DATABASE_URL to .env
[ ] 3. Install dependencies: pip install sqlalchemy asyncpg pydantic-settings
[ ] 4. Copy database/ folder to backend/
[ ] 5. Update main.py with imports and lifespan
[ ] 6. Add db_router to app
[ ] 7. Update analyze endpoint to save results
[ ] 8. Test with a real analysis
"""
