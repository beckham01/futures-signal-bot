import pandas as pd

from backtest.strategy import SignalEvent, compute_confidence, evaluate_signals


def test_confidence_score_max():
    score, label = compute_confidence(
        {
            "trend_rsi_strong": True,
            "rsi_clean_cross": True,
            "strong_volume": True,
            "ema55_slope": True,
            "candle_close_quality": True,
            "atr_expanding": True,
            "no_opposing_signal": True,
        }
    )
    assert score == 100
    assert label == "STRONG"


def test_confidence_score_partial():
    score, label = compute_confidence(
        {
            "trend_rsi_strong": True,
            "rsi_clean_cross": True,
            "strong_volume": True,
            "no_opposing_signal": True,
        }
    )
    assert score == 65
    assert label == "MODERATE"


def test_no_signal_in_sideways_market():
    times_15m = pd.date_range("2026-01-01", periods=80, freq="15min", tz="UTC")
    times_1h = pd.date_range("2025-12-31", periods=40, freq="1h", tz="UTC")
    df_15m = pd.DataFrame(
        {
            "timestamp": times_15m,
            "open": [100.0] * 80,
            "high": [100.2] * 80,
            "low": [99.8] * 80,
            "close": [100.0] * 80,
            "volume": [100.0] * 80,
        }
    )
    df_1h = pd.DataFrame(
        {
            "timestamp": times_1h,
            "open": [100.0] * 40,
            "high": [100.2] * 40,
            "low": [99.8] * 40,
            "close": [100.0] * 40,
            "volume": [100.0] * 40,
        }
    )
    assert evaluate_signals("BTCUSDT", df_15m, df_1h) == []


def prepared_frames(direction="LONG", duplicate=False, low_rr=False):
    timestamps = pd.date_range("2026-01-01T00:00:00Z", periods=8, freq="15min")
    close = [99.4, 99.6, 99.7, 100.2, 100.1, 99.7, 100.4, 100.5]
    rsi_values = [50, 50, 50, 44, 47, 44, 47, 48]
    if direction == "SHORT":
        close = [100.6, 100.4, 100.3, 99.8, 99.9, 100.3, 99.6, 99.5]
        rsi_values = [50, 50, 50, 56, 53, 56, 53, 52]
    if not duplicate:
        close[5:] = [close[4]] * 3
        rsi_values[5:] = [rsi_values[4]] * 3

    df_15m = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": close,
            "high": [value + 0.5 for value in close],
            "low": [value - 0.5 for value in close],
            "close": close,
            "volume": [100, 100, 100, 160, 160, 100, 170, 100],
            "ema21": [100.0] * 8,
            "ema55": [99.0] * 8,
            "rsi14": rsi_values,
            "atr14": [40.0 if low_rr else 1.0] * 8,
            "volume_sma20": [100.0] * 8,
            "atr_expanding": [True] * 8,
        }
    )
    df_1h = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-12-31T21:00:00Z", periods=4, freq="1h"),
            "open": [100.0] * 4,
            "high": [102.0] * 4,
            "low": [98.0] * 4,
            "close": [105.0 if direction == "LONG" else 95.0] * 4,
            "volume": [100.0] * 4,
            "ema21": [102.0 if direction == "LONG" else 98.0] * 4,
            "ema55": [100.0 if direction == "LONG" else 100.0] * 4,
            "rsi14": [55.0 if direction == "LONG" else 45.0] * 4,
            "atr14": [1.0] * 4,
            "ema55_slope": [0.01 if direction == "LONG" else -0.01] * 4,
        }
    )
    if direction == "SHORT":
        df_15m["ema55"] = 101.0
    return df_15m, df_1h


def test_long_signal_all_conditions_met():
    df_15m, df_1h = prepared_frames("LONG")
    signals = evaluate_signals("BTCUSDT", df_15m, df_1h)
    assert len(signals) == 1
    assert signals[0].direction == "LONG"
    assert signals[0].confidence_conditions["strong_volume"] is True
    assert "trend_rsi_strong" in signals[0].confidence_conditions


def test_high_confidence_signal_extends_target_to_5r():
    df_15m, df_1h = prepared_frames("LONG")
    df_15m.loc[4, "high"] = 100.2
    df_15m.loc[4, "low"] = 99.2
    cfg = {
        "ema_fast": 21,
        "ema_slow": 55,
        "rsi_period": 14,
        "atr_period": 14,
        "volume_sma_period": 20,
        "volume_spike_threshold": 1.5,
        "volume_strong_threshold": 1.5,
        "pullback_atr_tolerance": 0.35,
        "rsi_long_threshold": 45,
        "rsi_short_threshold": 55,
        "tp1_atr_multiplier": 2.0,
        "tp2_atr_multiplier": 4.5,
        "sl_atr_multiplier": 1.5,
        "min_risk_reward": 3.0,
        "min_confidence": 55,
        "atr_max_pct": 0.03,
        "atr_min_pct": 0.003,
        "extended_targets": {
            "enabled": True,
            "base_rr": 3.0,
            "high_confidence_rr": 5.0,
            "high_confidence_min_score": 95,
        },
    }

    signal = evaluate_signals("BTCUSDT", df_15m, df_1h, cfg=cfg)[0]

    assert signal.confidence == 100
    assert signal.risk_reward == 5.0
    assert signal.target_rr == 5.0
    assert signal.tp2 == signal.entry + abs(signal.entry - signal.stop_loss) * 5.0
    assert "Extended target" in signal.target_note


def test_lower_confidence_signal_keeps_base_target():
    df_15m, df_1h = prepared_frames("LONG")
    df_15m.loc[4, "high"] = 100.2
    df_15m.loc[4, "low"] = 99.2
    cfg = {
        "ema_fast": 21,
        "ema_slow": 55,
        "rsi_period": 14,
        "atr_period": 14,
        "volume_sma_period": 20,
        "volume_spike_threshold": 1.5,
        "volume_strong_threshold": 2.0,
        "pullback_atr_tolerance": 0.35,
        "rsi_long_threshold": 45,
        "rsi_short_threshold": 55,
        "tp1_atr_multiplier": 2.0,
        "tp2_atr_multiplier": 4.5,
        "sl_atr_multiplier": 1.5,
        "min_risk_reward": 3.0,
        "min_confidence": 55,
        "atr_max_pct": 0.03,
        "atr_min_pct": 0.003,
        "extended_targets": {
            "enabled": True,
            "base_rr": 3.0,
            "high_confidence_rr": 5.0,
            "high_confidence_min_score": 95,
        },
    }

    signal = evaluate_signals("BTCUSDT", df_15m, df_1h, cfg=cfg)[0]

    assert signal.confidence == 85
    assert signal.risk_reward == 3.0
    assert signal.target_rr == 3.0
    assert "Base target" in signal.target_note


def test_short_signal_all_conditions_met():
    df_15m, df_1h = prepared_frames("SHORT")
    signals = evaluate_signals("BTCUSDT", df_15m, df_1h)
    assert len(signals) == 1
    assert signals[0].direction == "SHORT"


def test_cooldown_suppresses_duplicate():
    df_15m, df_1h = prepared_frames("LONG", duplicate=True)
    signals = evaluate_signals("BTCUSDT", df_15m, df_1h, cooldown_hours=4)
    assert len(signals) == 1


def test_volatility_filter_rejects_wide_atr():
    df_15m, df_1h = prepared_frames("LONG", low_rr=True)
    signals = evaluate_signals("BTCUSDT", df_15m, df_1h)
    assert signals == []


def test_signal_event_dataclass_importable():
    assert SignalEvent.__name__ == "SignalEvent"
