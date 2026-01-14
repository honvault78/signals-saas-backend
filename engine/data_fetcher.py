"""
Data Fetcher Module - Yahoo Finance Data Retrieval
Replaces Bloomberg (blpapi/xbbg) with yfinance for production SaaS.

Investment-grade implementation with proper error handling and validation.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class DataFetchError(Exception):
    """Raised when data fetching fails."""
    pass


class InsufficientDataError(Exception):
    """Raised when there isn't enough historical data."""
    pass


def fetch_ticker_data(
    ticker: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    days: int = 400,
    min_required_days: int = 220,
    auto_adjust: bool = False,
) -> pd.DataFrame:
    """
    Fetch historical daily data for a single ticker from Yahoo Finance.
    
    Parameters
    ----------
    ticker : str
        Yahoo Finance ticker symbol (e.g., 'AAPL', 'DGE.L', 'CPR.MI')
    start_date : str, optional
        Start date in 'YYYY-MM-DD' format. If None, calculated from days.
    end_date : str, optional
        End date in 'YYYY-MM-DD' format. If None, uses today.
    days : int
        Number of calendar days to fetch if start_date not specified.
    min_required_days : int
        Minimum number of trading days required.
    auto_adjust : bool
        If True, Yahoo prices are adjusted (splits/dividends) and returned in the 'Close' column.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: Open, High, Low, Close, Volume, Adj Close
        Index is DatetimeIndex.
        
    Raises
    ------
    DataFetchError
        If the ticker cannot be fetched or is invalid.
    InsufficientDataError
        If there isn't enough historical data.
    """
    # Calculate date range
    if end_date is None:
        end_dt = datetime.now()
    else:
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    
    if start_date is None:
        start_dt = end_dt - timedelta(days=days)
    else:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    
    start_str = start_dt.strftime("%Y-%m-%d")
    end_str = end_dt.strftime("%Y-%m-%d")
    
    logger.info(f"Fetching {ticker} from {start_str} to {end_str}")
    
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_str, end=end_str, auto_adjust=auto_adjust)
    except Exception as e:
        raise DataFetchError(f"Failed to fetch {ticker}: {type(e).__name__}: {e}")
    
    if df is None or df.empty:
        raise DataFetchError(f"No data returned for ticker '{ticker}'. Check if the symbol is valid.")
    
    # Validate we have enough data
    if len(df) < min_required_days:
        raise InsufficientDataError(
            f"Insufficient data for {ticker}: got {len(df)} days, need at least {min_required_days}. "
            f"This ticker may be too new or have limited trading history."
        )
    
    # Ensure datetime index
    df.index = pd.to_datetime(df.index)
    
    # Remove timezone info if present (for consistency)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    
    logger.info(f"Fetched {len(df)} days for {ticker}")
    
    return df


def fetch_multiple_tickers(
    tickers: List[str],
    days: int = 400,
    min_required_days: int = 220,
    auto_adjust: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    Fetch historical data for multiple tickers.
    
    If ANY ticker fails, raises an exception (fail-fast behavior).
    
    Parameters
    ----------
    tickers : List[str]
        List of Yahoo Finance ticker symbols.
    days : int
        Number of calendar days to fetch.
    min_required_days : int
        Minimum trading days required per ticker.
        
    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary mapping ticker to its DataFrame.
        
    Raises
    ------
    DataFetchError
        If any ticker fails to fetch.
    """
    results = {}
    
    for ticker in tickers:
        # This will raise on failure (fail-fast)
        df = fetch_ticker_data(
            ticker=ticker,
            days=days,
            min_required_days=min_required_days,
            auto_adjust=auto_adjust,
        )
        results[ticker] = df
    
    return results


def calculate_returns(
    price_data: Dict[str, pd.DataFrame],
    price_column: str = "Close",
    prefer_adj_close: bool = True,
    align_prices_ffill: bool = True,
) -> pd.DataFrame:
    """
    Calculate daily percentage returns from price data.

    Returns are in PERCENTAGE form (1.5 means 1.5%, not 0.015).

    IMPORTANT (to match Bloomberg-style economics / notebook outputs):
    - Prefer "Adj Close" when available (splits/dividends adjusted) unless prefer_adj_close=False
    - Align tickers on a common calendar by forward-filling PRICES (not returns) when align_prices_ffill=True

    Parameters
    ----------
    price_data : Dict[str, pd.DataFrame]
        Dictionary mapping ticker to price DataFrame.
    price_column : str
        Fallback column to use for returns calculation when Adj Close is unavailable.
    prefer_adj_close : bool
        If True, use "Adj Close" when present, otherwise use price_column.
    align_prices_ffill : bool
        If True, aligns all price series on the union of dates and forward-fills prices before computing returns.

    Returns
    -------
    pd.DataFrame
        DataFrame with tickers as columns, returns in percentage form.
        NaN values are filled with 0.
    """
    price_series = {}

    for ticker, df in price_data.items():
        if df is None or df.empty:
            raise DataFetchError(f"Empty data for {ticker}")

        # Choose the most appropriate column
        chosen_col = None
        if prefer_adj_close and "Adj Close" in df.columns:
            chosen_col = "Adj Close"
        elif price_column in df.columns:
            chosen_col = price_column
        elif "Close" in df.columns:
            chosen_col = "Close"
        else:
            raise DataFetchError(f"No suitable price column found for {ticker}. Available: {list(df.columns)}")

        s = df[chosen_col].astype(float).copy()
        s = s[~s.index.duplicated(keep="last")]
        s.index = pd.to_datetime(s.index)
        if getattr(s.index, "tz", None) is not None:
            s.index = s.index.tz_localize(None)
        s = s.sort_index()
        price_series[ticker] = s

    prices_df = pd.DataFrame(price_series).sort_index()

    if align_prices_ffill:
        # Align on union of dates; forward-fill prices to handle non-overlapping holiday calendars.
        prices_df = prices_df.ffill()

    returns_df = (prices_df / prices_df.shift(1) - 1) * 100.0
    returns_df = returns_df.dropna(how="all").fillna(0.0)

    return returns_df


def get_ticker_info(ticker: str) -> Dict:
    """
    Get basic information about a ticker.
    
    Parameters
    ----------
    ticker : str
        Yahoo Finance ticker symbol.
        
    Returns
    -------
    Dict
        Dictionary with ticker info (name, currency, exchange, etc.)
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        return {
            "symbol": ticker,
            "name": info.get("shortName") or info.get("longName") or ticker,
            "currency": info.get("currency", "USD"),
            "exchange": info.get("exchange", ""),
            "sector": info.get("sector", ""),
            "industry": info.get("industry", ""),
        }
    except Exception as e:
        logger.warning(f"Could not fetch info for {ticker}: {e}")
        return {
            "symbol": ticker,
            "name": ticker,
            "currency": "USD",
            "exchange": "",
            "sector": "",
            "industry": "",
        }


def validate_tickers(tickers: List[str]) -> Tuple[List[str], List[str]]:
    """
    Validate a list of tickers by attempting to fetch minimal data.
    
    Parameters
    ----------
    tickers : List[str]
        List of ticker symbols to validate.
        
    Returns
    -------
    Tuple[List[str], List[str]]
        (valid_tickers, invalid_tickers)
    """
    valid = []
    invalid = []
    
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            # Try to get just 5 days of data as a validation check
            df = stock.history(period="5d")
            if df is not None and not df.empty:
                valid.append(ticker)
            else:
                invalid.append(ticker)
        except Exception:
            invalid.append(ticker)
    
    return valid, invalid
