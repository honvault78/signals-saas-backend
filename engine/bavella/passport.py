"""
Bavella v2 — Series Passport
==============================

The "instant trust" diagnostic summary.

Copyright 2024-2026 Bavella Technologies Sarl
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime

from .models import SeriesPassport, SeriesType, ValidityState
from .transforms import NISTransformer, NISTransformResult, TransformedSeriesAnalyzer, SeriesTypeDetector
from .validity_engine import ValidityEngine


class PassportGenerator:
    """Generates comprehensive Series Passports."""
    
    def __init__(self):
        self.transformer = NISTransformer()
        self.engine = ValidityEngine()
        
    def generate(self, series: pd.Series, series_id: str = "") -> SeriesPassport:
        """Generate a complete Series Passport."""
        series = series.dropna()
        n = len(series)
        
        passport = SeriesPassport(
            series_id=series_id or f"series_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            generated_at=datetime.utcnow(),
            observation_count=n,
        )
        
        if n < 30:
            passport.warnings.append(f"Insufficient observations ({n}). Minimum 30 required.")
            passport.current_validity_state = ValidityState.UNKNOWN
            return passport
            
        # Date range
        try:
            if hasattr(series.index, 'min'):
                min_date = series.index.min()
                max_date = series.index.max()
                if hasattr(min_date, 'to_pydatetime'):
                    passport.date_range = (min_date.to_pydatetime(), max_date.to_pydatetime())
                elif isinstance(min_date, datetime):
                    passport.date_range = (min_date, max_date)
        except:
            pass
            
        # Frequency detection
        passport.frequency_detected = self._detect_frequency(series)
        
        # Series type detection
        detector = SeriesTypeDetector(series)
        series_type, type_confidence, evidence = detector.detect()
        passport.series_type = series_type
        passport.type_confidence = type_confidence
        
        # Structural properties
        passport.can_be_negative = evidence.get("can_be_negative", False)
        passport.crosses_zero = evidence.get("crosses_zero", False)
        passport.variance_proportional_to_level = evidence.get("variance_level_correlation", 0) > 0.5
        
        # Transform
        transform_result = self.transformer.transform(series)
        
        if not transform_result.is_valid:
            passport.warnings.extend(transform_result.warnings)
            passport.current_validity_state = ValidityState.UNKNOWN
            return passport
            
        # Seasonality
        passport.has_seasonality = transform_result.seasonality_detected
        passport.dominant_period_days = transform_result.dominant_period
        passport.seasonality_strength = transform_result.seasonality_strength
        
        # Statistical analysis
        analyzer = TransformedSeriesAnalyzer(transform_result)
        stats = analyzer.compute_all_statistics()
        
        # Stationarity
        stationarity = stats.get("stationarity", {})
        passport.is_stationary = stationarity.get("is_stationary", False)
        passport.adf_pvalue = stationarity.get("adf_pvalue", 1.0)
        
        # Mean reversion
        mean_rev = stats.get("mean_reversion", {})
        passport.hurst_exponent = mean_rev.get("hurst_exponent", 0.5)
        passport.half_life_periods = mean_rev.get("half_life")
        
        # Risk shape
        z_stats = stats.get("z_t_stats", {})
        passport.skewness = z_stats.get("skewness", 0)
        passport.kurtosis = z_stats.get("kurtosis", 0)
        
        tail = stats.get("tail_risk", {})
        passport.tail_heaviness = tail.get("tail_classification", "normal")
        
        # Jumps
        jumps = stats.get("jumps", {})
        passport.jump_frequency = jumps.get("jump_frequency", 0)
        
        # Validity assessment
        validity = self.engine.assess_validity(
            transform_result=transform_result,
            target_id=passport.series_id,
        )
        passport.current_validity_score = validity.validity_score
        passport.current_validity_state = validity.validity_state
        
        # Warnings
        passport.warnings.extend(transform_result.warnings)
        passport.warnings.extend(self._generate_warnings(passport))
        
        return passport
    
    def _detect_frequency(self, series: pd.Series) -> str:
        try:
            if not hasattr(series.index, 'to_series'):
                return "unknown"
            diffs = pd.Series(series.index).diff().dropna()
            if len(diffs) == 0:
                return "unknown"
            median_diff = diffs.median()
            if hasattr(median_diff, 'days'):
                days = median_diff.days
                if days <= 1:
                    return "daily"
                elif days <= 7:
                    return "weekly"
                elif days <= 31:
                    return "monthly"
                else:
                    return "irregular"
            return "sequential"
        except:
            return "unknown"
    
    def _generate_warnings(self, passport: SeriesPassport) -> List[str]:
        warnings = []
        
        if not passport.is_stationary and passport.adf_pvalue > 0.1:
            warnings.append(f"Series appears non-stationary (ADF p={passport.adf_pvalue:.3f}).")
            
        if passport.hurst_exponent > 0.6:
            warnings.append(f"High Hurst ({passport.hurst_exponent:.2f}) indicates trending behavior.")
            
        if passport.tail_heaviness in ["heavy", "extreme"]:
            warnings.append(f"Heavy tails detected. Normal distribution may understate risk.")
            
        if passport.jump_frequency > 0.05:
            warnings.append(f"Elevated jump frequency ({passport.jump_frequency:.1%}).")
            
        return warnings


class PassportFormatter:
    """Formats passports for different outputs."""
    
    @staticmethod
    def to_text(passport: SeriesPassport) -> str:
        lines = [
            "=" * 60,
            "SERIES PASSPORT",
            "=" * 60,
            f"Series ID: {passport.series_id}",
            f"Generated: {passport.generated_at.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            f"Observations: {passport.observation_count}",
            f"Frequency: {passport.frequency_detected}",
            f"Type: {passport.series_type.value} ({passport.type_confidence:.0%} confidence)",
            "",
            f"Stationary: {'Yes' if passport.is_stationary else 'No'} (ADF p={passport.adf_pvalue:.4f})",
            f"Hurst: {passport.hurst_exponent:.3f}",
            f"Half-life: {f'{passport.half_life_periods:.1f} periods' if passport.half_life_periods else 'N/A'}",
            "",
            f"Skewness: {passport.skewness:.3f}",
            f"Kurtosis: {passport.kurtosis:.3f}",
            f"Tail: {passport.tail_heaviness}",
            "",
            f"Validity Score: {passport.current_validity_score:.1f}",
            f"Validity State: {passport.current_validity_state.value.upper()}",
            "",
        ]
        
        if passport.warnings:
            lines.append("WARNINGS:")
            for w in passport.warnings:
                lines.append(f"  - {w}")
                
        lines.append("=" * 60)
        return "\n".join(lines)


def generate_passport(series: pd.Series, series_id: str = "", format: str = "dict") -> Any:
    """Generate a Series Passport in the requested format."""
    generator = PassportGenerator()
    passport = generator.generate(series, series_id)
    
    if format == "dict":
        return passport.to_dict()
    elif format == "text":
        return PassportFormatter.to_text(passport)
    elif format == "html":
        return f"<pre>{PassportFormatter.to_text(passport)}</pre>"
    else:
        return passport.to_dict()
