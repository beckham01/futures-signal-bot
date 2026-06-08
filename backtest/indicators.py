"""Pure pandas/numpy indicator calculations."""

from __future__ import annotations

import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    """Return an exponential moving average."""
    return series.astype("float64").ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Return RSI using Wilder smoothing."""
    values = series.astype("float64")
    delta = values.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss
    result = 100 - (100 / (1 + rs))
    result = result.mask((avg_loss == 0) & (avg_gain > 0), 100.0)
    result = result.mask((avg_gain == 0) & (avg_loss > 0), 0.0)
    result = result.mask((avg_gain == 0) & (avg_loss == 0), 50.0)
    return result


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Return Average True Range using Wilder smoothing."""
    high = df["high"].astype("float64")
    low = df["low"].astype("float64")
    close = df["close"].astype("float64")
    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def volume_sma(series: pd.Series, period: int = 20) -> pd.Series:
    """Return simple moving average of volume."""
    return series.astype("float64").rolling(window=period, min_periods=period).mean()


def ema_slope(series: pd.Series, lookback: int = 3) -> pd.Series:
    """Return relative EMA slope over a lookback window."""
    values = series.astype("float64")
    previous = values.shift(lookback)
    return (values - previous) / previous
