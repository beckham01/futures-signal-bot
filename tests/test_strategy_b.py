import logging

import pandas as pd
import pytest

from backtest.strategy_b import STRATEGY_NAME, compute_confidence_b, evaluate_signals_b


@pytest.fixture(autouse=True)
def _default_regime_trending(monkeypatch):
    # Isolate existing signal-generation tests from the BTC regime gate (and
    # from any network/cache access it would otherwise trigger). Dedicated
    # regime-gate tests override this per-test via monkeypatch.
    monkeypatch.setattr("backtest.strategy_b.is_btc_regime_trending", lambda *args, **kwargs: True)


def cfg(**overrides):
    base = {
        "timeframe_entry": "15",
        "timeframe_trend": "60",
        "ema_fast": 21,
        "ema_slow": 55,
        "rsi_period": 14,
        "atr_period": 14,
        "volume_sma_period": 20,
        "fvg_min_gap_atr": 0.2,
        "fvg_max_age_candles": 48,
        "breaker_confirm_candles": 3,
        "breaker_max_age_candles": 96,
        "sl_atr_buffer": 0.3,
        "tp1_risk_multiplier": 1.5,
        "tp2_risk_multiplier": 2.5,
        "min_risk_reward": 2.0,
        "min_confidence": 50,
        "candle_body_min_pct": 0.40,
        "atr_max_pct": 0.04,
        "atr_min_pct": 0.003,
        "cooldown_hours": 4,
        "fast_scan": False,
    }
    base.update(overrides)
    return base


def frames(direction="LONG", **mutations):
    periods = 12
    ts = pd.date_range("2026-01-01", periods=periods, freq="15min", tz="UTC")
    close = [101.0] * periods
    open_ = [100.5] * periods
    high = [102.5] * periods
    low = [100.5] * periods
    rsi = [50.0] * periods
    idx = 9
    if direction == "LONG":
        open_[idx], high[idx], low[idx], close[idx] = 100.0, 103.0, 100.5, 102.0
        rsi[idx - 1], rsi[idx] = 50.0, 58.0
    else:
        open_[idx], high[idx], low[idx], close[idx] = 102.0, 101.5, 98.0, 98.8
        rsi[idx - 1], rsi[idx] = 50.0, 42.0
    df_15m = pd.DataFrame(
        {
            "timestamp": ts,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": [150.0] * periods,
            "ema21": [100.0] * periods,
            "ema55": [99.0] * periods,
            "rsi14": rsi,
            "atr14": [1.0] * periods,
            "volume_sma20": [100.0] * periods,
        }
    )
    trend = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-12-31T23:00:00Z", periods=3, freq="1h"),
            "open": [100.0] * 3,
            "high": [101.0] * 3,
            "low": [99.0] * 3,
            "close": [100.0] * 3,
            "volume": [100.0] * 3,
            "ema21": [101.0 if direction == "LONG" else 99.0] * 3,
            "ema55": [100.0] * 3,
            "rsi14": [58.0 if direction == "LONG" else 42.0] * 3,
        }
    )
    for column, value in mutations.items():
        if column.startswith("trend_"):
            trend.loc[:, column.replace("trend_", "")] = value
        else:
            df_15m.loc[idx, column] = value
    return df_15m, trend


def patch_confluence(monkeypatch, direction="bullish", gap_size=0.8, formed_at=5):
    monkeypatch.setattr(
        "backtest.strategy_b.detect_fvgs",
        lambda *args, **kwargs: [{"type": direction, "zone": (99.0, 101.0), "formed_at": 5, "filled": False, "gap_size": gap_size}],
    )
    monkeypatch.setattr(
        "backtest.strategy_b.detect_breaker_block",
        lambda *args, **kwargs: {"type": args[3], "zone": (98.5, 101.0), "formed_at": formed_at, "mitigated": False},
    )


def test_long_signal_all_conditions_met(monkeypatch):
    # Core pattern-detection logic is tested independent of the business-level
    # SHORT-only restriction, which is covered separately by the filter tests below.
    monkeypatch.setattr("config.strategy_filters.STRATEGY_B_DIRECTIONS_ALLOWED", ["LONG", "SHORT"])
    patch_confluence(monkeypatch)
    df, trend = frames("LONG")
    signals = evaluate_signals_b("BTCUSDT", df, trend, cfg=cfg())
    assert len(signals) == 1
    assert signals[0].direction == "LONG"


def test_short_signal_all_conditions_met(monkeypatch):
    patch_confluence(monkeypatch, "bearish")
    df, trend = frames("SHORT")
    signals = evaluate_signals_b("BTCUSDT", df, trend, cfg=cfg())
    assert len(signals) == 1
    assert signals[0].direction == "SHORT"


def test_no_signal_bearish_1h_trend_on_long_setup(monkeypatch):
    patch_confluence(monkeypatch)
    df, trend = frames("LONG", trend_ema21=99.0, trend_ema55=100.0)
    assert evaluate_signals_b("BTCUSDT", df, trend, cfg=cfg()) == []


def test_no_signal_no_confluence_zone(monkeypatch):
    monkeypatch.setattr("backtest.strategy_b.detect_fvgs", lambda *args, **kwargs: [])
    monkeypatch.setattr("backtest.strategy_b.detect_breaker_block", lambda *args, **kwargs: None)
    df, trend = frames("LONG")
    assert evaluate_signals_b("BTCUSDT", df, trend, cfg=cfg()) == []


def test_no_signal_price_did_not_wick_into_zone(monkeypatch):
    patch_confluence(monkeypatch)
    df, trend = frames("LONG", low=102.0)
    assert evaluate_signals_b("BTCUSDT", df, trend, cfg=cfg()) == []


def test_no_signal_candle_closed_below_zone_low(monkeypatch):
    patch_confluence(monkeypatch)
    df, trend = frames("LONG", close=98.8)
    assert evaluate_signals_b("BTCUSDT", df, trend, cfg=cfg()) == []


def test_no_signal_rsi_falling_on_long(monkeypatch):
    patch_confluence(monkeypatch)
    df, trend = frames("LONG", rsi14=47.0)
    df.loc[8, "rsi14"] = 55.0
    assert evaluate_signals_b("BTCUSDT", df, trend, cfg=cfg()) == []


def test_no_signal_weak_candle_body(monkeypatch):
    patch_confluence(monkeypatch)
    df, trend = frames("LONG", open=101.8)
    assert evaluate_signals_b("BTCUSDT", df, trend, cfg=cfg()) == []


def test_no_signal_bad_atr(monkeypatch):
    patch_confluence(monkeypatch)
    df, trend = frames("LONG", atr14=10.0)
    assert evaluate_signals_b("BTCUSDT", df, trend, cfg=cfg()) == []


def test_no_signal_rr_below_minimum(monkeypatch):
    patch_confluence(monkeypatch)
    df, trend = frames("LONG")
    assert evaluate_signals_b("BTCUSDT", df, trend, cfg=cfg(tp2_risk_multiplier=1.0)) == []


def test_cooldown_blocks_same_direction(monkeypatch):
    monkeypatch.setattr("config.strategy_filters.STRATEGY_B_DIRECTIONS_ALLOWED", ["LONG", "SHORT"])
    patch_confluence(monkeypatch)
    df, trend = frames("LONG")
    df.loc[10, ["open", "high", "low", "close", "rsi14"]] = [100.0, 103.0, 100.5, 102.0, 58.0]
    signals = evaluate_signals_b("BTCUSDT", df, trend, cfg=cfg())
    assert len(signals) == 1


def test_confidence_all_conditions_true():
    score, label = compute_confidence_b(
        {
            "trend_rsi_strong": True,
            "large_fvg": True,
            "fresh_breaker": True,
            "momentum_aligned": True,
            "strong_rejection_body": True,
            "volume_above_average": True,
        }
    )
    assert score == 100
    assert label == "STRONG"


def test_confidence_partial_score():
    score, label = compute_confidence_b({"trend_rsi_strong": True, "large_fvg": True})
    assert score == 40
    assert label == "LOW"


def test_strategy_name_is_strategy_b(monkeypatch):
    monkeypatch.setattr("config.strategy_filters.STRATEGY_B_DIRECTIONS_ALLOWED", ["LONG", "SHORT"])
    patch_confluence(monkeypatch)
    df, trend = frames("LONG")
    assert evaluate_signals_b("BTCUSDT", df, trend, cfg=cfg())[0].strategy_name == STRATEGY_NAME


def test_signal_rejected_symbol_not_whitelisted(monkeypatch, caplog):
    patch_confluence(monkeypatch, "bearish")
    df, trend = frames("SHORT")
    caplog.set_level(logging.DEBUG)
    signals = evaluate_signals_b("DOGEUSDT", df, trend, cfg=cfg())
    assert signals == []
    assert any(getattr(record, "reason", None) == "symbol_not_whitelisted" for record in caplog.records)


def test_signal_rejected_direction_filtered(monkeypatch, caplog):
    # Strategy B's default config only allows SHORT; a LONG setup on a whitelisted
    # symbol must still be rejected by the direction gate.
    patch_confluence(monkeypatch)
    df, trend = frames("LONG")
    caplog.set_level(logging.DEBUG)
    signals = evaluate_signals_b("BTCUSDT", df, trend, cfg=cfg())
    assert signals == []
    assert any(getattr(record, "reason", None) == "direction_filtered" for record in caplog.records)


def test_filter_whitelist_config_override(monkeypatch):
    patch_confluence(monkeypatch, "bearish")
    df, trend = frames("SHORT")
    monkeypatch.setattr("config.strategy_filters.STRATEGY_B_WHITELIST", ["DOGEUSDT"])
    assert evaluate_signals_b("BTCUSDT", df, trend, cfg=cfg()) == []
    monkeypatch.setattr("config.strategy_filters.STRATEGY_B_WHITELIST", ["BTCUSDT"])
    assert len(evaluate_signals_b("BTCUSDT", df, trend, cfg=cfg())) == 1


def test_filter_direction_config_override(monkeypatch):
    patch_confluence(monkeypatch)
    df, trend = frames("LONG")
    monkeypatch.setattr("config.strategy_filters.STRATEGY_B_DIRECTIONS_ALLOWED", ["LONG"])
    assert len(evaluate_signals_b("BTCUSDT", df, trend, cfg=cfg())) == 1
    monkeypatch.setattr("config.strategy_filters.STRATEGY_B_DIRECTIONS_ALLOWED", ["SHORT"])
    assert evaluate_signals_b("BTCUSDT", df, trend, cfg=cfg()) == []


def test_signal_rejected_regime_chop(monkeypatch, caplog):
    monkeypatch.setattr("config.strategy_filters.STRATEGY_B_DIRECTIONS_ALLOWED", ["LONG", "SHORT"])
    monkeypatch.setattr("backtest.strategy_b.is_btc_regime_trending", lambda *args, **kwargs: False)
    patch_confluence(monkeypatch)
    df, trend = frames("LONG")
    caplog.set_level(logging.DEBUG)
    signals = evaluate_signals_b("BTCUSDT", df, trend, cfg=cfg())
    assert signals == []
    assert any(getattr(record, "reason", None) == "regime_chop" for record in caplog.records)


def test_signal_allowed_when_regime_trending(monkeypatch):
    monkeypatch.setattr("config.strategy_filters.STRATEGY_B_DIRECTIONS_ALLOWED", ["LONG", "SHORT"])
    monkeypatch.setattr("backtest.strategy_b.is_btc_regime_trending", lambda *args, **kwargs: True)
    patch_confluence(monkeypatch)
    df, trend = frames("LONG")
    assert len(evaluate_signals_b("BTCUSDT", df, trend, cfg=cfg())) == 1
