"""
Statistics Module - Comprehensive Statistical Analysis

Investment-grade statistical analysis including:
- Descriptive statistics
- Hypothesis testing
- Normality tests
- Risk metrics (VaR, CVaR)
- Tail risk analysis
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import jarque_bera, shapiro, kstest, anderson, skewtest, kurtosistest

logger = logging.getLogger(__name__)


@dataclass
class DescriptiveStats:
    """Descriptive statistics."""
    observations: int
    mean_daily: float
    std_daily: float
    min_return: float
    max_return: float
    skewness: float
    kurtosis: float
    percentiles: Dict[str, float]


@dataclass
class HypothesisTests:
    """Hypothesis test results."""
    t_statistic: float
    p_value: float
    ci_95_lower: float
    ci_95_upper: float
    ci_99_lower: float
    ci_99_upper: float
    mean_significantly_different_from_zero: bool


@dataclass
class NormalityTests:
    """Normality test results."""
    jarque_bera_stat: float
    jarque_bera_pval: float
    shapiro_stat: float
    shapiro_pval: float
    ks_stat: float
    ks_pval: float
    is_normal_jb: bool
    is_normal_shapiro: bool
    is_normal_ks: bool


@dataclass
class RiskMetrics:
    """Risk metrics."""
    var_95: float  # Value at Risk (5th percentile)
    var_99: float  # Value at Risk (1st percentile)
    cvar_95: float  # Conditional VaR (Expected Shortfall)
    cvar_99: float
    var_95_parametric: float  # Assuming normal distribution
    var_99_parametric: float


@dataclass
class TailRisk:
    """Tail risk analysis."""
    positive_extremes: int  # Returns > 2 std
    negative_extremes: int  # Returns < -2 std
    total_extremes: int
    extreme_ratio: float
    tail_ratio: float  # 95th / 5th percentile ratio


@dataclass
class EnhancedStatistics:
    """Complete statistical analysis."""
    descriptive: DescriptiveStats
    hypothesis: HypothesisTests
    normality: NormalityTests
    risk: RiskMetrics
    tail: TailRisk
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "descriptive": {
                "observations": self.descriptive.observations,
                "mean_daily_pct": round(self.descriptive.mean_daily * 100, 4),
                "std_daily_pct": round(self.descriptive.std_daily * 100, 4),
                "min_return_pct": round(self.descriptive.min_return * 100, 2),
                "max_return_pct": round(self.descriptive.max_return * 100, 2),
                "skewness": round(self.descriptive.skewness, 4),
                "kurtosis": round(self.descriptive.kurtosis, 4),
                "percentiles": {
                    k: round(v * 100, 3) for k, v in self.descriptive.percentiles.items()
                },
            },
            "hypothesis": {
                "t_statistic": round(self.hypothesis.t_statistic, 4),
                "p_value": round(self.hypothesis.p_value, 4),
                "ci_95_lower_pct": round(self.hypothesis.ci_95_lower * 100, 4),
                "ci_95_upper_pct": round(self.hypothesis.ci_95_upper * 100, 4),
                "ci_99_lower_pct": round(self.hypothesis.ci_99_lower * 100, 4),
                "ci_99_upper_pct": round(self.hypothesis.ci_99_upper * 100, 4),
                "significant": self.hypothesis.mean_significantly_different_from_zero,
            },
            "normality": {
                "jarque_bera": {
                    "statistic": round(self.normality.jarque_bera_stat, 4),
                    "p_value": round(self.normality.jarque_bera_pval, 4),
                    "is_normal": self.normality.is_normal_jb,
                },
                "shapiro_wilk": {
                    "statistic": round(self.normality.shapiro_stat, 4),
                    "p_value": round(self.normality.shapiro_pval, 4),
                    "is_normal": self.normality.is_normal_shapiro,
                },
                "kolmogorov_smirnov": {
                    "statistic": round(self.normality.ks_stat, 4),
                    "p_value": round(self.normality.ks_pval, 4),
                    "is_normal": self.normality.is_normal_ks,
                },
            },
            "risk": {
                "var_95_pct": round(self.risk.var_95 * 100, 4),
                "var_99_pct": round(self.risk.var_99 * 100, 4),
                "cvar_95_pct": round(self.risk.cvar_95 * 100, 4),
                "cvar_99_pct": round(self.risk.cvar_99 * 100, 4),
                "var_95_parametric_pct": round(self.risk.var_95_parametric * 100, 4),
                "var_99_parametric_pct": round(self.risk.var_99_parametric * 100, 4),
            },
            "tail": {
                "positive_extremes": self.tail.positive_extremes,
                "negative_extremes": self.tail.negative_extremes,
                "total_extremes": self.tail.total_extremes,
                "extreme_ratio_pct": round(self.tail.extreme_ratio * 100, 2),
                "tail_ratio": round(self.tail.tail_ratio, 3),
            },
        }


def calculate_enhanced_statistics(daily_returns: pd.Series) -> EnhancedStatistics:
    """
    Calculate comprehensive statistics on portfolio returns.
    
    Parameters
    ----------
    daily_returns : pd.Series
        Daily returns in decimal form (0.01 = 1%)
        
    Returns
    -------
    EnhancedStatistics
        Complete statistical analysis.
    """
    returns = daily_returns.dropna().values
    n = len(returns)
    
    if n < 20:
        raise ValueError(f"Insufficient data for statistics: {n} observations (need at least 20)")
    
    # Descriptive Statistics
    mean = np.mean(returns)
    std = np.std(returns, ddof=1)
    skew = stats.skew(returns)
    kurt = stats.kurtosis(returns)
    
    percentiles = {
        "p1": np.percentile(returns, 1),
        "p5": np.percentile(returns, 5),
        "p10": np.percentile(returns, 10),
        "p25": np.percentile(returns, 25),
        "p50": np.percentile(returns, 50),
        "p75": np.percentile(returns, 75),
        "p90": np.percentile(returns, 90),
        "p95": np.percentile(returns, 95),
        "p99": np.percentile(returns, 99),
    }
    
    descriptive = DescriptiveStats(
        observations=n,
        mean_daily=mean,
        std_daily=std,
        min_return=np.min(returns),
        max_return=np.max(returns),
        skewness=skew,
        kurtosis=kurt,
        percentiles=percentiles,
    )
    
    # Hypothesis Testing
    t_stat, p_val = stats.ttest_1samp(returns, 0)
    sem = stats.sem(returns)
    ci_95 = stats.t.interval(0.95, n-1, loc=mean, scale=sem)
    ci_99 = stats.t.interval(0.99, n-1, loc=mean, scale=sem)
    
    hypothesis = HypothesisTests(
        t_statistic=t_stat,
        p_value=p_val,
        ci_95_lower=ci_95[0],
        ci_95_upper=ci_95[1],
        ci_99_lower=ci_99[0],
        ci_99_upper=ci_99[1],
        mean_significantly_different_from_zero=p_val < 0.05,
    )
    
    # Normality Tests
    jb_stat, jb_pval = jarque_bera(returns)
    sw_stat, sw_pval = shapiro(returns[:5000])  # Shapiro limited to 5000
    ks_stat, ks_pval = kstest(returns, 'norm', args=(mean, std))
    
    normality = NormalityTests(
        jarque_bera_stat=jb_stat,
        jarque_bera_pval=jb_pval,
        shapiro_stat=sw_stat,
        shapiro_pval=sw_pval,
        ks_stat=ks_stat,
        ks_pval=ks_pval,
        is_normal_jb=jb_pval >= 0.05,
        is_normal_shapiro=sw_pval >= 0.05,
        is_normal_ks=ks_pval >= 0.05,
    )
    
    # Risk Metrics
    var_95 = np.percentile(returns, 5)
    var_99 = np.percentile(returns, 1)
    
    # Conditional VaR (Expected Shortfall)
    cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else var_95
    cvar_99 = returns[returns <= var_99].mean() if len(returns[returns <= var_99]) > 0 else var_99
    
    # Parametric VaR (assuming normal)
    var_95_param = mean - 1.645 * std
    var_99_param = mean - 2.326 * std
    
    risk = RiskMetrics(
        var_95=var_95,
        var_99=var_99,
        cvar_95=cvar_95,
        cvar_99=cvar_99,
        var_95_parametric=var_95_param,
        var_99_parametric=var_99_param,
    )
    
    # Tail Risk
    positive_extreme = np.sum(returns > mean + 2 * std)
    negative_extreme = np.sum(returns < mean - 2 * std)
    total_extreme = positive_extreme + negative_extreme
    
    p95 = np.percentile(returns, 95)
    p5 = np.percentile(returns, 5)
    tail_ratio = abs(p95 / p5) if p5 != 0 else 1.0
    
    tail = TailRisk(
        positive_extremes=int(positive_extreme),
        negative_extremes=int(negative_extreme),
        total_extremes=int(total_extreme),
        extreme_ratio=total_extreme / n,
        tail_ratio=tail_ratio,
    )
    
    return EnhancedStatistics(
        descriptive=descriptive,
        hypothesis=hypothesis,
        normality=normality,
        risk=risk,
        tail=tail,
    )


def calculate_memo_stats(
    daily_returns: pd.Series,
    cumulative: pd.Series
) -> Dict:
    """
    Calculate statistics formatted for the AI memo prompt.
    
    Returns stats in the exact format expected by the memo generator.
    """
    returns = daily_returns.dropna()
    n = len(returns)
    
    total_return = cumulative.iloc[-1] - 1
    ann_return = (cumulative.iloc[-1]) ** (252 / n) - 1
    ann_vol = returns.std() * np.sqrt(252)
    
    # Sharpe
    sharpe = (ann_return) / ann_vol if ann_vol > 0 else 0
    
    # Sortino
    downside = returns[returns < 0]
    downside_std = downside.std() * np.sqrt(252) if len(downside) > 0 else ann_vol
    sortino = ann_return / downside_std if downside_std > 0 else 0
    
    # Max drawdown
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_dd = drawdown.min()
    
    # Win rate
    win_rate = (returns > 0).sum() / n
    
    # Best/worst
    best_day = returns.max()
    worst_day = returns.min()
    
    # VaR/CVaR
    var_95 = np.percentile(returns, 5)
    cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else var_95
    
    # Skew/Kurt
    skew = stats.skew(returns)
    kurt = stats.kurtosis(returns)
    
    # T-test
    t_stat, p_val = stats.ttest_1samp(returns, 0)
    
    # Calmar
    calmar = ann_return / abs(max_dd) if max_dd != 0 else 0
    
    return {
        "total_return": total_return,
        "annualized_return": ann_return,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "best_day": best_day,
        "worst_day": worst_day,
        "var_95": var_95,
        "cvar_95": cvar_95,
        "skewness": skew,
        "kurtosis": kurt,
        "t_statistic": t_stat,
        "p_value": p_val,
        "calmar_ratio": calmar,
    }
