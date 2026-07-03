"""BTC weekly market regime classifier (net-trending vs net-chop).

This gate is used by Strategy B ONLY (strategy_b_fvg_breaker_15m). Strategy C
(strategy_c_fvg_breaker_4h) must never depend on this module — regime
filtering was tested against historical Strategy C trades and it reduced both
win rate and total R there, because it strips out DOGEUSDT's wins, which
occur independent of market regime. See config/strategy_filters.py for the
full rationale.

The classification threshold (avg ADX14 daily >= 25 -> "net-trending", else
"net-chop") reproduces the method already used to produce
market_regime_breakdown.csv: see tests/test_regime.py, which validates this
threshold against that CSV as ground truth.
"""

from __future__ import annotations

import logging
from typing import Callable

import pandas as pd

from backtest.indicators import adx
from backtest.resample import resample_ohlcv

LOGGER = logging.getLogger(__name__)

ADX_TRENDING_THRESHOLD = 25.0
REGIME_WINDOW_DAYS = 7
REGIME_SYMBOL = "BTCUSDT"
REGIME_SOURCE_INTERVAL = "60"  # daily candles are derived by resampling 1h data

# Keyed by ISO week-start (Monday) date string so a signal's regime stays
# constant for the whole week it falls in, instead of drifting bar-to-bar.
# Also means repeated calls within the same scan cycle (across the 5
# whitelisted symbols) reuse one computation instead of recomputing.
_regime_cache: dict[str, bool] = {}


def classify_regime(avg_adx14: float) -> str:
    """Classify a weekly average ADX14 reading as 'net-trending' or 'net-chop'."""
    return "net-trending" if avg_adx14 >= ADX_TRENDING_THRESHOLD else "net-chop"


def week_start_utc(timestamp: pd.Timestamp) -> pd.Timestamp:
    """Return the Monday 00:00 UTC that starts the ISO week containing `timestamp`."""
    ts = pd.Timestamp(timestamp)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return ts.normalize() - pd.Timedelta(days=ts.dayofweek)


def compute_trailing_avg_adx14(
    df_hourly: pd.DataFrame,
    as_of: pd.Timestamp,
    window_days: int = REGIME_WINDOW_DAYS,
) -> float:
    """Return the trailing average daily ADX14 over `window_days` ending at `as_of`.

    `df_hourly` (1h OHLCV candles) is resampled to daily candles internally,
    mirroring the method used to produce market_regime_breakdown.csv.
    """
    as_of = pd.Timestamp(as_of)
    as_of = as_of.tz_localize("UTC") if as_of.tzinfo is None else as_of.tz_convert("UTC")
    daily = resample_ohlcv(df_hourly, interval_minutes=1440)
    if daily.empty:
        return float("nan")
    daily["adx14"] = adx(daily, 14)
    window_start = as_of - pd.Timedelta(days=window_days)
    window = daily[(daily["timestamp"] > window_start) & (daily["timestamp"] <= as_of)]
    if window.empty or window["adx14"].isna().all():
        return float("nan")
    return float(window["adx14"].mean())


def clear_regime_cache() -> None:
    """Clear the per-week BTC regime cache (new scan cycle / test isolation)."""
    _regime_cache.clear()


def is_btc_regime_trending(
    current_time: pd.Timestamp,
    df_hourly: pd.DataFrame | None = None,
    fetch_hourly: Callable[[str, str, int, int], pd.DataFrame] | None = None,
) -> bool:
    """Return True if BTCUSDT is currently classified 'net-trending'.

    Classification is computed once per ISO week (Monday-anchored) using the
    prior *completed* week's trailing 7-day average ADX14 - never the
    in-progress current week, which would be lookahead bias in a live/backtest
    gate - and held constant for every bar in the following week. Cached
    in-process per week, so repeated calls within the same scan cycle (across
    the 5 whitelisted Strategy B symbols) reuse the same result.
    """
    current_time = pd.Timestamp(current_time)
    as_of = current_time.tz_localize("UTC") if current_time.tzinfo is None else current_time.tz_convert("UTC")
    week_start = week_start_utc(as_of)
    cache_key = week_start.strftime("%Y-%m-%d")
    if cache_key in _regime_cache:
        return _regime_cache[cache_key]

    calc_as_of = week_start - pd.Timedelta(seconds=1)  # end of the prior completed week

    if df_hourly is None:
        if fetch_hourly is None:
            from backtest.data_fetcher import fetch_klines

            fetch_hourly = fetch_klines
        start_ms = int((calc_as_of - pd.Timedelta(days=REGIME_WINDOW_DAYS + 5)).timestamp() * 1000)
        end_ms = int(calc_as_of.timestamp() * 1000)
        df_hourly = fetch_hourly(REGIME_SYMBOL, REGIME_SOURCE_INTERVAL, start_ms, end_ms)

    avg_adx14 = compute_trailing_avg_adx14(df_hourly, calc_as_of)
    trending = avg_adx14 == avg_adx14 and classify_regime(avg_adx14) == "net-trending"  # NaN-safe check
    _regime_cache[cache_key] = trending
    return trending


def current_btc_regime_label(current_time: pd.Timestamp, **kwargs) -> str:
    """Return 'net-trending' or 'net-chop' for `current_time` (for telemetry)."""
    return "net-trending" if is_btc_regime_trending(current_time, **kwargs) else "net-chop"
