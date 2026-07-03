"""OHLCV candle resampling helpers."""

from __future__ import annotations

import pandas as pd


def resample_ohlcv(df: pd.DataFrame, interval_minutes: int) -> pd.DataFrame:
    """Resample a UTC OHLCV frame to a larger minute interval."""
    if df.empty:
        return df.copy()
    frame = df.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame.sort_values("timestamp", inplace=True)
    resampled = (
        frame.set_index("timestamp")
        .resample(f"{interval_minutes}min", label="left", closed="left")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna()
        .reset_index()
    )
    return resampled


def interval_to_minutes(interval: str) -> int:
    """Convert Bybit minute interval notation to integer minutes."""
    return int(interval)
