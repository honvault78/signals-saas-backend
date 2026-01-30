"""
Custom Data Parser for Upload Data Feature

Handles CSV/Excel parsing, validation, and series extraction for user-uploaded data.
Supports both dated and sequential (no-date) formats.
"""

import pandas as pd
import numpy as np
from io import BytesIO, StringIO
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

# Constants
MAX_ROWS = 5000
MIN_ROWS = 100  # Minimum for reliable ADF test, half-life calculation, and regime detection
MAX_SERIES = 20


class DateStatus(str, Enum):
    DETECTED = "detected"
    SEQUENTIAL = "sequential"


@dataclass
class ParsedData:
    """Result of parsing uploaded data"""
    df: pd.DataFrame  # Index is either DatetimeIndex or RangeIndex, columns are series
    series_names: List[str]
    row_count: int
    date_status: DateStatus
    date_range: Optional[Tuple[str, str]]  # (start, end) if dates detected
    frequency: Optional[str]  # 'daily', 'weekly', 'monthly', 'irregular', or None
    warnings: List[str]
    

@dataclass
class ValidationResult:
    """Result of validating uploaded data"""
    valid: bool
    errors: List[str]
    warnings: List[str]
    parsed_data: Optional[ParsedData]


def detect_date_column(df: pd.DataFrame) -> Optional[str]:
    """
    Detect if the first column is a datetime column.
    Returns column name if detected, None otherwise.
    """
    if df.empty or len(df.columns) == 0:
        return None
    
    first_col = df.columns[0]
    col_data = df[first_col]
    
    # Already datetime
    if pd.api.types.is_datetime64_any_dtype(col_data):
        return first_col
    
    # Try to parse as dates
    if col_data.dtype == object or col_data.dtype == str:
        try:
            parsed = pd.to_datetime(col_data, errors='coerce')
            valid_ratio = parsed.notna().sum() / len(parsed)
            if valid_ratio > 0.9:  # 90%+ parseable as dates
                return first_col
        except Exception:
            pass
    
    # Check if it's sequential integers (1, 2, 3...)
    if pd.api.types.is_numeric_dtype(col_data):
        if col_data.is_monotonic_increasing:
            # Could be index, not dates - treat as sequential
            return None
    
    return None


def infer_frequency(dates: pd.DatetimeIndex) -> str:
    """Infer the frequency of a datetime index"""
    if len(dates) < 2:
        return "unknown"
    
    diffs = dates.to_series().diff().dropna()
    median_diff = diffs.median()
    
    if median_diff <= pd.Timedelta(hours=2):
        return "hourly"
    elif median_diff <= pd.Timedelta(days=1.5):
        return "daily"
    elif median_diff <= pd.Timedelta(days=8):
        return "weekly"
    elif median_diff <= pd.Timedelta(days=35):
        return "monthly"
    else:
        return "irregular"


def parse_uploaded_file(
    file_content: bytes,
    filename: str
) -> ValidationResult:
    """
    Parse uploaded CSV or Excel file.
    
    Expected format (wide):
    - Optional first column: dates (YYYY-MM-DD) or sequential index
    - Remaining columns: series values with headers as names
    
    Example:
        Series_A,Series_B,Series_C
        100.00,50.00,75.00
        101.25,51.10,74.20
    
    Or with dates:
        datetime,Series_A,Series_B
        2024-01-02,100.00,50.00
        2024-01-03,101.25,51.10
    """
    errors = []
    warnings = []
    
    # Determine file type and read
    try:
        if filename.lower().endswith('.xlsx') or filename.lower().endswith('.xls'):
            df = pd.read_excel(BytesIO(file_content))
        elif filename.lower().endswith('.csv'):
            # Try different encodings
            try:
                df = pd.read_csv(BytesIO(file_content), encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(BytesIO(file_content), encoding='latin-1')
        else:
            # Try CSV as default
            try:
                df = pd.read_csv(BytesIO(file_content))
            except Exception:
                errors.append(f"Unsupported file format. Please upload CSV or Excel (.xlsx)")
                return ValidationResult(valid=False, errors=errors, warnings=[], parsed_data=None)
    except Exception as e:
        errors.append(f"Failed to read file: {str(e)}")
        return ValidationResult(valid=False, errors=errors, warnings=[], parsed_data=None)
    
    # Basic validation
    if df.empty:
        errors.append("File is empty")
        return ValidationResult(valid=False, errors=errors, warnings=[], parsed_data=None)
    
    if len(df) > MAX_ROWS:
        errors.append(f"Too many rows ({len(df):,}). Maximum is {MAX_ROWS:,}.")
        return ValidationResult(valid=False, errors=errors, warnings=[], parsed_data=None)
    
    if len(df) < MIN_ROWS:
        if len(df) >= 80:
            errors.append(
                f"Almost there — {len(df)} rows provided, need {MIN_ROWS}. "
                f"Add ~{MIN_ROWS - len(df)} more observations for reliable analysis."
            )
        else:
            errors.append(
                f"Insufficient data ({len(df)} rows). Minimum {MIN_ROWS} required. "
                f"Regime detection, ADF stationarity tests, and half-life calculations need sufficient history to produce meaningful results."
            )
        return ValidationResult(valid=False, errors=errors, warnings=[], parsed_data=None)
    
    # Detect date column
    date_col = detect_date_column(df)
    date_status = DateStatus.DETECTED if date_col else DateStatus.SEQUENTIAL
    date_range = None
    frequency = None
    
    if date_col:
        # Parse dates and set as index
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        
        # Check for parsing failures
        null_dates = df[date_col].isna().sum()
        if null_dates > 0:
            warnings.append(f"{null_dates} rows had unparseable dates and will be excluded")
            df = df.dropna(subset=[date_col])
        
        df = df.set_index(date_col).sort_index()
        
        if isinstance(df.index, pd.DatetimeIndex):
            date_range = (df.index.min().strftime('%Y-%m-%d'), df.index.max().strftime('%Y-%m-%d'))
            frequency = infer_frequency(df.index)
    else:
        # Sequential index
        df = df.reset_index(drop=True)
    
    # Get series columns (all remaining columns should be numeric)
    series_names = list(df.columns)
    
    if len(series_names) == 0:
        errors.append("No data series found. Ensure your file has numeric columns.")
        return ValidationResult(valid=False, errors=errors, warnings=[], parsed_data=None)
    
    if len(series_names) > MAX_SERIES:
        errors.append(f"Too many series ({len(series_names)}). Maximum is {MAX_SERIES}.")
        return ValidationResult(valid=False, errors=errors, warnings=[], parsed_data=None)
    
    # Validate numeric columns
    non_numeric = []
    for col in series_names:
        if not pd.api.types.is_numeric_dtype(df[col]):
            # Try to convert
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except Exception:
                non_numeric.append(col)
    
    if non_numeric:
        errors.append(f"Non-numeric values in columns: {', '.join(non_numeric)}")
        return ValidationResult(valid=False, errors=errors, warnings=[], parsed_data=None)
    
    # Check for NaN values
    for col in series_names:
        nan_count = df[col].isna().sum()
        if nan_count > 0:
            nan_pct = (nan_count / len(df)) * 100
            if nan_pct > 10:
                errors.append(f"Column '{col}' has {nan_pct:.1f}% missing values. Please fill gaps or remove column.")
            else:
                warnings.append(f"Column '{col}' has {nan_count} missing values ({nan_pct:.1f}%). Will forward-fill.")
                df[col] = df[col].ffill().bfill()
    
    if errors:
        return ValidationResult(valid=False, errors=errors, warnings=warnings, parsed_data=None)
    
    # Final row count after cleaning
    row_count = len(df)
    if row_count < MIN_ROWS:
        if row_count >= 80:
            errors.append(
                f"After cleaning, {row_count} valid rows remain — need {MIN_ROWS}. "
                f"Add ~{MIN_ROWS - row_count} more observations or fix data quality issues."
            )
        else:
            errors.append(
                f"After cleaning, only {row_count} valid rows remain. "
                f"Minimum {MIN_ROWS} required for reliable regime analysis."
            )
        return ValidationResult(valid=False, errors=errors, warnings=[], parsed_data=None)
    
    parsed_data = ParsedData(
        df=df,
        series_names=series_names,
        row_count=row_count,
        date_status=date_status,
        date_range=date_range,
        frequency=frequency,
        warnings=warnings
    )
    
    return ValidationResult(
        valid=True,
        errors=[],
        warnings=warnings,
        parsed_data=parsed_data
    )


def build_synthetic_series(
    parsed_data: ParsedData,
    weights: Dict[str, float]
) -> pd.Series:
    """
    Build a single synthetic series from weighted combination of input series.
    
    Args:
        parsed_data: Parsed upload data
        weights: Dict mapping series name to weight (e.g., {"Series_A": 1.0, "Series_B": -0.5})
    
    Returns:
        Combined series (weighted sum of returns, not levels)
    """
    df = parsed_data.df
    
    # Filter to only series with non-zero weights
    active_weights = {k: v for k, v in weights.items() if v != 0 and k in df.columns}
    
    if not active_weights:
        raise ValueError("No valid series with non-zero weights")
    
    # Calculate returns for each series
    returns_df = df[list(active_weights.keys())].pct_change().dropna()
    
    # Weight and sum
    weighted_returns = pd.Series(0.0, index=returns_df.index)
    total_abs_weight = sum(abs(w) for w in active_weights.values())
    
    for series_name, weight in active_weights.items():
        # Normalize weight by total absolute weight for proper scaling
        normalized_weight = weight / total_abs_weight
        weighted_returns += returns_df[series_name] * normalized_weight
    
    return weighted_returns


def get_analysis_metadata(
    parsed_data: ParsedData,
    weights: Dict[str, float],
    context: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate metadata for the analysis report.
    """
    active_series = [k for k, v in weights.items() if v != 0 and k in parsed_data.series_names]
    
    metadata = {
        "has_dates": parsed_data.date_status == DateStatus.DETECTED,
        "row_count": parsed_data.row_count,
        "series_count": len(active_series),
        "series_names": active_series,
        "weights": {k: weights[k] for k in active_series},
        "context": context,
    }
    
    if parsed_data.date_status == DateStatus.DETECTED:
        metadata["date_range"] = parsed_data.date_range
        metadata["frequency"] = parsed_data.frequency
    
    return metadata


def generate_template_csv() -> str:
    """
    Generate a sample template CSV for users to download.
    
    Note: This is just a format example. Actual uploads require minimum 100 rows
    for reliable regime analysis (ADF test, half-life, etc.).
    
    Includes date column by default - users can delete it for sequential-only data.
    """
    template = """date,Series_A,Series_B,Series_C
2024-01-02,100.00,50.00,200.00
2024-01-03,101.25,51.10,198.50
2024-01-04,99.80,50.50,201.25
2024-01-05,102.30,52.00,199.00
2024-01-08,100.50,51.75,202.50
2024-01-09,103.10,52.50,197.75
2024-01-10,101.90,51.25,203.00
2024-01-11,104.00,53.00,196.50
2024-01-12,102.50,52.75,204.25
2024-01-15,105.20,53.50,195.00
2024-01-16,103.80,52.90,197.25
2024-01-17,106.50,54.00,193.50
2024-01-18,104.90,53.25,199.75
2024-01-19,107.30,54.50,191.00
2024-01-22,105.60,53.75,196.00
2024-01-23,108.20,55.00,188.50
2024-01-24,106.40,54.25,194.25
2024-01-25,109.00,55.50,186.00
2024-01-26,107.10,54.75,192.50
2024-01-29,110.50,56.00,183.75
"""
    return template.strip()
