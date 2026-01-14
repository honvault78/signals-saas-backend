"""
Charts Module - Matplotlib Visualization Generation

Generates professional charts for the analysis report:
- 6-panel regime analysis chart
- Performance chart (cumulative + daily + drawdown)
- Distribution chart

All charts are converted to base64 for embedding in HTML.
"""

from __future__ import annotations

import base64
import logging
from io import BytesIO
from typing import Dict, Optional

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from .regime import TechnicalIndicators, MarketRegime

logger = logging.getLogger(__name__)

# Color scheme matching the notebook
BACKGROUND_COLOR = '#e8f4fc'  # Pale blue
REGIME_COLORS = {
    'ranging': 'lightgreen',
    'trending_up': 'green',
    'trending_down': 'red',
    'breakout': 'lime',
    'breakdown': 'salmon',
    'accumulation': 'lightblue',
    'distribution': 'orange',
    'volatile_expansion': 'yellow',
    'volatile_contraction': 'lightyellow',
    'uncertain': 'lightgray',
}


def fig_to_base64(fig: plt.Figure) -> str:
    """Convert matplotlib figure to base64 string."""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', 
                facecolor=BACKGROUND_COLOR, edgecolor='none')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close(fig)
    return img_base64


def add_regime_shading(ax, regime_series: pd.Series):
    """Add colored shading for different regimes."""
    prev_regime = None
    start_date = None
    
    for date, regime in regime_series.items():
        if regime != prev_regime:
            if prev_regime is not None and start_date is not None:
                color = REGIME_COLORS.get(prev_regime, 'white')
                ax.axvspan(start_date, date, alpha=0.3, color=color)
            prev_regime = regime
            start_date = date
    
    # Final segment
    if prev_regime is not None and start_date is not None:
        color = REGIME_COLORS.get(prev_regime, 'white')
        ax.axvspan(start_date, regime_series.index[-1], alpha=0.3, color=color)


def create_regime_chart(
    cumulative: pd.Series,
    daily_returns: pd.Series,
    regime_df: pd.DataFrame,
    indicators: TechnicalIndicators,
    z_score: pd.Series,
    signals: pd.Series,
    portfolio_name: str = "Portfolio"
) -> str:
    """
    Create the 6-panel regime analysis chart.
    
    Panels:
    1. Price with EMAs, Bollinger Bands, signals, regime shading
    2. RSI with overbought/oversold zones
    3. MACD with histogram
    4. Trend Score vs Mean Reversion Score
    5. ADF p-value and Hurst Exponent
    6. Z-Score from Fair Value
    
    Returns base64 encoded PNG.
    """
    fig, axes = plt.subplots(6, 1, figsize=(16, 22))
    fig.patch.set_facecolor(BACKGROUND_COLOR)
    for ax in axes:
        ax.set_facecolor(BACKGROUND_COLOR)
    plt.subplots_adjust(hspace=0.35)
    
    # Align data
    common_idx = regime_df.index
    
    # Panel 1: Price with indicators and regime shading
    ax1 = axes[0]
    add_regime_shading(ax1, regime_df['regime'])
    
    # Bollinger Bands
    bb_upper = indicators.bb_upper.reindex(common_idx)
    bb_lower = indicators.bb_lower.reindex(common_idx)
    bb_middle = indicators.bb_middle.reindex(common_idx)
    ax1.fill_between(common_idx, bb_lower, bb_upper, alpha=0.15, color='blue', label='BB (20,2)')
    ax1.plot(bb_middle, 'b--', alpha=0.4, linewidth=1)
    
    # Price and EMAs
    cum_aligned = cumulative.reindex(common_idx)
    ax1.plot(cum_aligned.index, cum_aligned.values, 'k-', linewidth=1.5, label='Portfolio')
    ax1.plot(indicators.ema50.reindex(common_idx), 'orange', linewidth=1.2, label='EMA50', alpha=0.8)
    ax1.plot(indicators.ema100.reindex(common_idx), 'purple', linewidth=1.2, label='EMA100', alpha=0.8)
    
    # Signals
    buy_signals = signals[signals == 'BUY'].index
    sell_signals = signals[signals == 'SELL'].index
    
    if len(buy_signals) > 0:
        buy_prices = cumulative.reindex(buy_signals).dropna()
        ax1.scatter(buy_prices.index, buy_prices.values, marker='^', 
                   color='lime', s=120, label='BUY', zorder=5, 
                   edgecolors='darkgreen', linewidths=1.5)
    
    if len(sell_signals) > 0:
        sell_prices = cumulative.reindex(sell_signals).dropna()
        ax1.scatter(sell_prices.index, sell_prices.values, marker='v',
                   color='red', s=120, label='SELL', zorder=5,
                   edgecolors='darkred', linewidths=1.5)
    
    ax1.set_ylabel('Value', fontsize=11)
    ax1.set_title(f'{portfolio_name}: Price, Indicators & Regime', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=8, ncol=4)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=1, color='black', linestyle='--', alpha=0.3)
    
    # Regime legend
    regime_legend = [
        Patch(facecolor='lightgreen', alpha=0.5, edgecolor='darkgreen', linewidth=1, label='RANGING'),
        Patch(facecolor='green', alpha=0.5, edgecolor='darkgreen', linewidth=1, label='TRENDING UP'),
        Patch(facecolor='red', alpha=0.5, edgecolor='darkred', linewidth=1, label='TRENDING DOWN'),
        Patch(facecolor='lime', alpha=0.5, edgecolor='green', linewidth=1, label='BREAKOUT'),
        Patch(facecolor='salmon', alpha=0.5, edgecolor='red', linewidth=1, label='BREAKDOWN'),
        Patch(facecolor='lightblue', alpha=0.5, edgecolor='blue', linewidth=1, label='ACCUMULATION'),
        Patch(facecolor='orange', alpha=0.5, edgecolor='darkorange', linewidth=1, label='DISTRIBUTION'),
    ]
    ax1.legend(handles=regime_legend, loc='lower left', fontsize=7,
               title='Regime Legend', title_fontsize=9, ncol=2,
               framealpha=0.9, fancybox=True)
    
    # Panel 2: RSI
    ax2 = axes[1]
    rsi_aligned = indicators.rsi.reindex(common_idx)
    ax2.plot(rsi_aligned, 'purple', linewidth=1.2)
    ax2.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Overbought (70)')
    ax2.axhline(y=30, color='green', linestyle='--', alpha=0.7, label='Oversold (30)')
    ax2.axhline(y=50, color='gray', linestyle='-', alpha=0.3)
    ax2.fill_between(rsi_aligned.index, 30, 
                     rsi_aligned.where(rsi_aligned < 30), color='green', alpha=0.3)
    ax2.fill_between(rsi_aligned.index, 70,
                     rsi_aligned.where(rsi_aligned > 70), color='red', alpha=0.3)
    ax2.set_ylabel('RSI', fontsize=11)
    ax2.set_title('RSI (14) - Oversold/Overbought', fontsize=11)
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)
    
    # Panel 3: MACD
    ax3 = axes[2]
    macd_line = indicators.macd_line.reindex(common_idx)
    macd_signal = indicators.macd_signal.reindex(common_idx)
    macd_hist = indicators.macd_histogram.reindex(common_idx)
    
    ax3.plot(macd_line, 'blue', linewidth=1.2, label='MACD')
    ax3.plot(macd_signal, 'red', linewidth=1.2, label='Signal')
    colors_hist = ['green' if v >= 0 else 'red' for v in macd_hist.fillna(0)]
    ax3.bar(macd_hist.index, macd_hist.values, color=colors_hist, alpha=0.5, width=1)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
    ax3.set_ylabel('MACD', fontsize=11)
    ax3.set_title('MACD (12, 26, 9)', fontsize=11)
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Trend Score vs Mean Reversion Score
    ax4 = axes[3]
    ax4.plot(regime_df.index, regime_df['trend_score'], 'blue', linewidth=1.5, label='Trend Score')
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax4.axhline(y=2, color='green', linestyle='--', alpha=0.5, label='Trend threshold')
    ax4.axhline(y=-2, color='red', linestyle='--', alpha=0.5)
    ax4.fill_between(regime_df.index, 0, regime_df['trend_score'],
                     where=regime_df['trend_score'] > 0, alpha=0.3, color='green')
    ax4.fill_between(regime_df.index, 0, regime_df['trend_score'],
                     where=regime_df['trend_score'] < 0, alpha=0.3, color='red')
    ax4.set_ylabel('Trend Score', fontsize=11, color='blue')
    ax4.tick_params(axis='y', labelcolor='blue')
    
    ax4_twin = ax4.twinx()
    ax4_twin.plot(regime_df.index, regime_df['mean_reversion_score'], 'orange',
                  linewidth=1.5, label='MR Score', alpha=0.8)
    ax4_twin.set_ylabel('Mean Reversion Score', fontsize=11, color='orange')
    ax4_twin.tick_params(axis='y', labelcolor='orange')
    ax4_twin.set_ylim(0, 100)
    
    ax4.set_title('Trend Score vs Mean Reversion Score', fontsize=11)
    ax4.legend(loc='upper left', fontsize=8)
    ax4_twin.legend(loc='upper right', fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    # Panel 5: ADF p-value and Hurst
    ax5 = axes[4]
    ax5.plot(regime_df.index, regime_df['adf_pvalue'], 'lightblue', linewidth=1, 
             label='ADF p-value (raw)', alpha=0.6)
    ax5.plot(regime_df.index, regime_df['adf_smooth'], 'darkblue', linewidth=2,
             label='ADF Smoothed')
    ax5.axhline(y=0.05, color='green', linestyle='--', linewidth=1.5, label='p=0.05 (Strong MR)')
    ax5.axhline(y=0.20, color='orange', linestyle=':', alpha=0.7, label='p=0.20')
    ax5.axhline(y=0.50, color='red', linestyle=':', alpha=0.7, label='p=0.50')
    ax5.fill_between(regime_df.index, 0, 0.05, alpha=0.2, color='green')
    ax5.set_ylabel('ADF p-value', fontsize=11, color='darkblue')
    ax5.tick_params(axis='y', labelcolor='darkblue')
    ax5.set_ylim(0, 1)
    
    ax5_twin = ax5.twinx()
    ax5_twin.plot(regime_df.index, regime_df['hurst'], 'purple', linewidth=1.5,
                  label='Hurst', alpha=0.8)
    ax5_twin.axhline(y=0.5, color='red', linestyle='--', alpha=0.7)
    ax5_twin.set_ylabel('Hurst Exponent', fontsize=11, color='purple')
    ax5_twin.tick_params(axis='y', labelcolor='purple')
    ax5_twin.set_ylim(0, 1)
    
    ax5.set_title('ADF p-value & Hurst Exponent', fontsize=11)
    ax5.legend(loc='upper left', fontsize=8)
    ax5_twin.legend(loc='upper right', fontsize=8)
    ax5.grid(True, alpha=0.3)
    
    # Panel 6: Z-Score
    ax6 = axes[5]
    z_aligned = z_score.reindex(common_idx).fillna(0)
    colors_z = ['green' if z < 0 else 'red' for z in z_aligned]
    ax6.bar(z_aligned.index, z_aligned.values, color=colors_z, alpha=0.6, width=1)
    ax6.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax6.axhline(y=1.5, color='red', linestyle='--', alpha=0.7, label='±1.5σ')
    ax6.axhline(y=-1.5, color='green', linestyle='--', alpha=0.7)
    ax6.axhline(y=2, color='darkred', linestyle=':', alpha=0.5, label='±2σ')
    ax6.axhline(y=-2, color='darkgreen', linestyle=':', alpha=0.5)
    ax6.set_ylabel('Z-Score', fontsize=11)
    ax6.set_title('Distance from Fair Value (Z-Score)', fontsize=11)
    ax6.legend(loc='upper right', fontsize=8)
    ax6.grid(True, alpha=0.3)
    ax6.set_ylim(-3, 3)
    ax6.set_xlabel('Date', fontsize=11)
    
    return fig_to_base64(fig)


def create_performance_chart(
    cumulative: pd.Series,
    daily_returns: pd.Series,
    portfolio_name: str = "Portfolio"
) -> str:
    """
    Create the 3-panel performance chart.
    
    Panels:
    1. Cumulative returns with running max and drawdown shading
    2. Daily returns bar chart
    3. Drawdown from peak
    
    Returns base64 encoded PNG.
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [2, 1, 1]})
    fig.patch.set_facecolor(BACKGROUND_COLOR)
    for ax in axes:
        ax.set_facecolor(BACKGROUND_COLOR)
    plt.subplots_adjust(hspace=0.3)
    
    # Panel 1: Cumulative Performance
    ax1 = axes[0]
    running_max = cumulative.expanding().max()
    
    # Drawdown shading
    ax1.fill_between(cumulative.index, cumulative, running_max,
                     where=cumulative < running_max, alpha=0.3, color='red')
    ax1.plot(cumulative.index, cumulative.values, 'b-', linewidth=1.5, label='Portfolio Value')
    ax1.plot(running_max.index, running_max.values, 'g--', linewidth=1, alpha=0.7, label='Running Maximum')
    ax1.axhline(y=1.0, color='black', linestyle='--', alpha=0.3)
    ax1.set_ylabel('Cumulative Value', fontsize=11)
    ax1.set_title(f'{portfolio_name} Performance', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Daily Returns
    ax2 = axes[1]
    colors = ['green' if r > 0 else 'red' for r in daily_returns]
    ax2.bar(daily_returns.index, daily_returns.values * 100, color=colors, alpha=0.7, width=1)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax2.set_ylabel('Daily Return %', fontsize=11)
    ax2.set_title('Daily Returns', fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Drawdown
    ax3 = axes[2]
    drawdown = (cumulative - running_max) / running_max
    ax3.fill_between(drawdown.index, 0, drawdown.values * 100, color='red', alpha=0.3)
    ax3.plot(drawdown.index, drawdown.values * 100, 'r-', linewidth=1.5)
    max_dd = drawdown.min()
    ax3.axhline(y=max_dd * 100, color='darkred', linestyle='--', alpha=0.7,
                label=f'Max DD: {max_dd*100:.2f}%')
    ax3.set_ylabel('Drawdown %', fontsize=11)
    ax3.set_xlabel('Date', fontsize=11)
    ax3.set_title('Drawdown from Peak', fontsize=11)
    ax3.legend(loc='lower left', fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    return fig_to_base64(fig)


def create_distribution_chart(
    daily_returns: pd.Series,
    portfolio_name: str = "Portfolio"
) -> str:
    """
    Create the returns distribution chart.
    
    Shows histogram with normal distribution overlay and statistics.
    
    Returns base64 encoded PNG.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(BACKGROUND_COLOR)
    ax.set_facecolor(BACKGROUND_COLOR)
    
    returns_pct = daily_returns * 100
    
    # Histogram
    n, bins, patches = ax.hist(returns_pct, bins=50, alpha=0.7, color='blue', 
                                edgecolor='black', density=True)
    
    # Normal distribution overlay
    from scipy import stats
    mu = returns_pct.mean()
    sigma = returns_pct.std()
    x = np.linspace(returns_pct.min(), returns_pct.max(), 100)
    ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal Distribution')
    
    # Vertical lines
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.5, linewidth=2)
    ax.axvline(x=mu, color='green', linestyle='--', alpha=0.7, linewidth=1.5, label=f'Mean: {mu:.3f}%')
    
    # VaR lines
    var_95 = np.percentile(returns_pct, 5)
    ax.axvline(x=var_95, color='red', linestyle=':', alpha=0.7, linewidth=1.5, 
               label=f'VaR 95%: {var_95:.2f}%')
    
    # Statistics box
    stats_text = (
        f'Mean: {mu:.3f}%\n'
        f'Std Dev: {sigma:.3f}%\n'
        f'Skewness: {returns_pct.skew():.2f}\n'
        f'Kurtosis: {returns_pct.kurtosis():.2f}\n'
        f'Min: {returns_pct.min():.2f}%\n'
        f'Max: {returns_pct.max():.2f}%'
    )
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, verticalalignment='top',
            fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.set_xlabel('Daily Return %', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title(f'{portfolio_name} Daily Returns Distribution', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    return fig_to_base64(fig)


def create_all_charts(
    cumulative: pd.Series,
    daily_returns: pd.Series,
    regime_df: pd.DataFrame,
    indicators: TechnicalIndicators,
    z_score: pd.Series,
    signals: pd.Series,
    portfolio_name: str = "Portfolio"
) -> Dict[str, str]:
    """
    Create all charts and return as dictionary of base64 strings.
    """
    return {
        "regime": create_regime_chart(
            cumulative, daily_returns, regime_df, indicators, z_score, signals, portfolio_name
        ),
        "performance": create_performance_chart(
            cumulative, daily_returns, portfolio_name
        ),
        "distribution": create_distribution_chart(
            daily_returns, portfolio_name
        ),
    }
