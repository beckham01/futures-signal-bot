from pathlib import Path

import pandas as pd
import pytest

from backtest.regime import (
    ADX_TRENDING_THRESHOLD,
    classify_regime,
    clear_regime_cache,
    compute_trailing_avg_adx14,
    is_btc_regime_trending,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
REGIME_CSV = REPO_ROOT / "market_regime_breakdown.csv"
BTC_HOURLY_CACHE = REPO_ROOT / "data" / "cache" / "BTCUSDT_60.csv"


def _load_ground_truth_rows() -> list[tuple[float, str]]:
    df = pd.read_csv(REGIME_CSV)
    return list(zip(df["avg_adx14_daily"].astype(float), df["classification"]))


@pytest.fixture(autouse=True)
def _clear_cache():
    clear_regime_cache()
    yield
    clear_regime_cache()


@pytest.mark.parametrize("avg_adx14,expected", _load_ground_truth_rows())
def test_classify_regime_matches_market_regime_breakdown_csv(avg_adx14, expected):
    assert classify_regime(avg_adx14) == expected


def test_threshold_boundary_matches_csv_gap():
    # market_regime_breakdown.csv has a clean gap between its highest "net-chop"
    # sample (24.81) and its lowest "net-trending" sample (28.19); the threshold
    # must fall inside that gap.
    rows = _load_ground_truth_rows()
    chop_values = [value for value, label in rows if label == "net-chop"]
    trending_values = [value for value, label in rows if label == "net-trending"]
    assert max(chop_values) < ADX_TRENDING_THRESHOLD <= min(trending_values)


@pytest.mark.skipif(not BTC_HOURLY_CACHE.exists(), reason="requires cached BTCUSDT 1h data")
def test_compute_trailing_avg_adx14_classifies_known_trending_week():
    # CSV: BTCUSDT week 2026-02-17..2026-02-23 -> avg_adx14_daily=62.01, net-trending
    df_hourly = pd.read_csv(BTC_HOURLY_CACHE)
    as_of = pd.Timestamp("2026-02-23T23:59:59Z")
    avg_adx14 = compute_trailing_avg_adx14(df_hourly, as_of)
    assert classify_regime(avg_adx14) == "net-trending"


@pytest.mark.skipif(not BTC_HOURLY_CACHE.exists(), reason="requires cached BTCUSDT 1h data")
def test_compute_trailing_avg_adx14_classifies_known_chop_week():
    # CSV: BTCUSDT week 2026-04-07..2026-04-13 -> avg_adx14_daily=15.31, net-chop
    df_hourly = pd.read_csv(BTC_HOURLY_CACHE)
    as_of = pd.Timestamp("2026-04-13T23:59:59Z")
    avg_adx14 = compute_trailing_avg_adx14(df_hourly, as_of)
    assert classify_regime(avg_adx14) == "net-chop"


def _sample_hourly_frame(start: str, periods: int, adx_like_high_low_spread: float) -> pd.DataFrame:
    ts = pd.date_range(start, periods=periods, freq="1h", tz="UTC")
    trend = pd.Series(range(periods), dtype="float64") * adx_like_high_low_spread
    close = 100.0 + trend
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": close - 0.1,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": [100.0] * periods,
        }
    )


def test_is_btc_regime_trending_uses_cache_within_same_day(monkeypatch):
    calls = {"count": 0}

    def fake_fetch(symbol, interval, start_ms, end_ms):
        calls["count"] += 1
        return _sample_hourly_frame("2026-01-01", 24 * 60, 0.5)

    current_time = pd.Timestamp("2026-03-01T00:15:00Z")
    result_1 = is_btc_regime_trending(current_time, fetch_hourly=fake_fetch)
    result_2 = is_btc_regime_trending(current_time + pd.Timedelta(hours=1), fetch_hourly=fake_fetch)

    assert result_1 == result_2
    assert calls["count"] == 1


def test_is_btc_regime_trending_recomputes_on_new_day(monkeypatch):
    calls = {"count": 0}

    def fake_fetch(symbol, interval, start_ms, end_ms):
        calls["count"] += 1
        return _sample_hourly_frame("2026-01-01", 24 * 60, 0.5)

    is_btc_regime_trending(pd.Timestamp("2026-03-01T00:15:00Z"), fetch_hourly=fake_fetch)
    is_btc_regime_trending(pd.Timestamp("2026-03-02T00:15:00Z"), fetch_hourly=fake_fetch)

    assert calls["count"] == 2


def test_is_btc_regime_trending_uses_cache_within_same_week(monkeypatch):
    # 2026-03-02 is a Monday (week start); 2026-03-04 is the Wednesday of the
    # same ISO week - both should hit the same cached classification.
    calls = {"count": 0}

    def fake_fetch(symbol, interval, start_ms, end_ms):
        calls["count"] += 1
        return _sample_hourly_frame("2026-01-01", 24 * 60, 0.5)

    result_1 = is_btc_regime_trending(pd.Timestamp("2026-03-02T00:15:00Z"), fetch_hourly=fake_fetch)
    result_2 = is_btc_regime_trending(pd.Timestamp("2026-03-04T18:00:00Z"), fetch_hourly=fake_fetch)

    assert result_1 == result_2
    assert calls["count"] == 1


def test_week_start_utc_returns_monday():
    from backtest.regime import week_start_utc

    assert week_start_utc(pd.Timestamp("2026-03-04T18:00:00Z")) == pd.Timestamp("2026-03-02T00:00:00Z")
    assert week_start_utc(pd.Timestamp("2026-03-01T00:15:00Z")) == pd.Timestamp("2026-02-23T00:00:00Z")
