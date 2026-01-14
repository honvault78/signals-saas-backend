-- ============================================================================
-- Signals SaaS Database Schema
-- Initial Migration (v2 - with fixes)
-- ============================================================================
-- Run this SQL against your Neon database to create all tables.
-- 
-- How to run:
--   1. Go to Neon console -> SQL Editor
--   2. Paste this entire file
--   3. Click "Run"
-- ============================================================================

-- Enable UUID extension (usually already enabled on Neon)
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================================
-- ENUMS
-- ============================================================================

-- User subscription plans
CREATE TYPE user_plan AS ENUM ('free', 'pro', 'enterprise');

-- Alert types
CREATE TYPE alert_type AS ENUM ('buy', 'sell', 'regime_change', 'drawdown', 'zscore_extreme');

-- Alert severity levels
CREATE TYPE alert_severity AS ENUM ('info', 'warning', 'critical');

-- ============================================================================
-- USERS TABLE
-- ============================================================================

CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    clerk_id VARCHAR(255) NOT NULL UNIQUE,
    email VARCHAR(255) NOT NULL,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    plan user_plan NOT NULL DEFAULT 'free',
    usage_count INTEGER NOT NULL DEFAULT 0,
    usage_reset_at TIMESTAMP WITH TIME ZONE NOT NULL,  -- No default, set by Python
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    last_active_at TIMESTAMP WITH TIME ZONE
);

-- Indexes for users
CREATE INDEX ix_users_clerk_id ON users(clerk_id);
CREATE INDEX ix_users_email ON users(email);

-- ============================================================================
-- PORTFOLIOS TABLE
-- ============================================================================

CREATE TABLE portfolios (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    positions JSONB NOT NULL DEFAULT '[]'::jsonb,
    is_default BOOLEAN NOT NULL DEFAULT FALSE,
    is_tracked BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Indexes for portfolios
CREATE INDEX ix_portfolios_user_id ON portfolios(user_id);

-- CRITICAL: Enforce only ONE default portfolio per user at DB level
-- This prevents race conditions from creating multiple defaults
CREATE UNIQUE INDEX ux_one_default_portfolio_per_user 
ON portfolios(user_id) WHERE is_default = TRUE;

-- ============================================================================
-- ANALYSES TABLE (Saved on explicit user action only)
-- ============================================================================

CREATE TABLE analyses (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    portfolio_id UUID REFERENCES portfolios(id) ON DELETE SET NULL,
    portfolio_name VARCHAR(255) NOT NULL,
    positions JSONB NOT NULL,
    analysis_period_days INTEGER NOT NULL,
    result_summary JSONB NOT NULL DEFAULT '{}'::jsonb,
    html_report TEXT,
    ai_memo TEXT,
    duration_ms INTEGER,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Indexes for analyses
CREATE INDEX ix_analyses_user_id ON analyses(user_id);
CREATE INDEX ix_analyses_portfolio_id ON analyses(portfolio_id);
CREATE INDEX ix_analyses_created_at ON analyses(created_at);
CREATE INDEX ix_analyses_user_created ON analyses(user_id, created_at);

-- ============================================================================
-- PORTFOLIO SNAPSHOTS TABLE (Daily Tracking - automated)
-- ============================================================================

CREATE TABLE portfolio_snapshots (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    portfolio_id UUID NOT NULL REFERENCES portfolios(id) ON DELETE CASCADE,
    snapshot_date TIMESTAMP WITH TIME ZONE NOT NULL,
    cumulative_return DOUBLE PRECISION NOT NULL,
    daily_return DOUBLE PRECISION NOT NULL,
    portfolio_value DOUBLE PRECISION NOT NULL,
    regime VARCHAR(50) NOT NULL,
    signal VARCHAR(20) NOT NULL,
    z_score DOUBLE PRECISION NOT NULL,
    rsi DOUBLE PRECISION NOT NULL,
    adf_pvalue DOUBLE PRECISION NOT NULL,
    trend_score DOUBLE PRECISION NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Indexes for snapshots
CREATE INDEX ix_snapshots_portfolio_id ON portfolio_snapshots(portfolio_id);
CREATE INDEX ix_snapshots_snapshot_date ON portfolio_snapshots(snapshot_date);
CREATE INDEX ix_snapshots_portfolio_date ON portfolio_snapshots(portfolio_id, snapshot_date);

-- ============================================================================
-- ALERTS TABLE
-- ============================================================================

CREATE TABLE alerts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    portfolio_id UUID NOT NULL REFERENCES portfolios(id) ON DELETE CASCADE,
    alert_type alert_type NOT NULL,
    severity alert_severity NOT NULL DEFAULT 'info',
    signal_date TIMESTAMP WITH TIME ZONE NOT NULL,
    message TEXT NOT NULL,
    portfolio_value DOUBLE PRECISION,
    regime VARCHAR(50),
    z_score DOUBLE PRECISION,
    rsi DOUBLE PRECISION,
    is_read BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    read_at TIMESTAMP WITH TIME ZONE
);

-- Indexes for alerts
CREATE INDEX ix_alerts_user_id ON alerts(user_id);
CREATE INDEX ix_alerts_portfolio_id ON alerts(portfolio_id);
CREATE INDEX ix_alerts_alert_type ON alerts(alert_type);
CREATE INDEX ix_alerts_created_at ON alerts(created_at);
CREATE INDEX ix_alerts_user_unread ON alerts(user_id, is_read);
CREATE INDEX ix_alerts_user_created ON alerts(user_id, created_at);

-- ============================================================================
-- TRIGGER: Auto-update updated_at on portfolios
-- ============================================================================

CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_portfolios_updated_at
    BEFORE UPDATE ON portfolios
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- VERIFICATION QUERIES (run these to verify tables were created)
-- ============================================================================

-- Uncomment and run these to verify:
-- SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';
-- SELECT indexname FROM pg_indexes WHERE tablename = 'portfolios';

-- ============================================================================
-- DONE!
-- ============================================================================
-- Tables created:
--   - users (with usage tracking)
--   - portfolios (with unique default constraint)
--   - analyses (explicit save only)
--   - portfolio_snapshots (daily tracking)
--   - alerts (BUY/SELL signals)
-- ============================================================================
