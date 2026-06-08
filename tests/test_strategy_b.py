import pandas as pd

from backtest.strategy_b import (
    STRATEGY_NAME,
    compute_confidence_b,
    evaluate_signals_b,
    get_swing_high,
    get_swing_low,
)


def cfg(**overrides):
    base = {
        "timeframe": "15",
        "ema_slow": 55,
        "rsi_period": 14,
        "atr_period": 14,
        "volume_sma_period": 20,
        "volume_spike_threshold": 1.3,
        "volume_strong_threshold": 1.6,
        "swing_lookback": 8,
        "ema_slope_lookback": 5,
        "candle_body_min_pct": 0.5,
        "candle_body_strong_pct": 0.65,
        "breakout_margin_atr": 0.3,
        "tp1_risk_multiplier": 1.5,
        "tp2_risk_multiplier": 2.5,
        "sl_atr_buffer": 0.5,
        "min_risk_reward": 2.0,
        "min_confidence": 50,
        "atr_max_pct": 0.04,
        "atr_min_pct": 0.003,
        "cooldown_hours": 4,
    }
    base.update(overrides)
    return base


def frame(direction="LONG", duplicate=False, opposite=False, **mutations):
    periods = 24 if duplicate or opposite else 14
    timestamps = pd.date_range("2026-01-01T00:00:00Z", periods=periods, freq="15min")
    close = [100.0 + i * 0.04 for i in range(periods)]
    open_ = [value - 0.1 for value in close]
    high = [value + 0.2 for value in close]
    low = [value - 0.2 for value in close]
    rsi = [52.0] * periods
    ema55 = [99.0 + i * 0.02 for i in range(periods)]
    volume = [100.0] * periods
    atr = [1.0] * periods

    idx = 10
    close[idx] = max(high[idx - 8 : idx]) + 0.6
    open_[idx] = close[idx] - 0.8
    high[idx] = close[idx] + 0.1
    low[idx] = close[idx] - 1.0
    rsi[idx - 1] = 56.0
    rsi[idx] = 62.0
    volume[idx] = 180.0

    if duplicate:
        idx2 = 14
        close[idx2] = max(high[idx2 - 8 : idx2]) + 0.6
        open_[idx2] = close[idx2] - 0.8
        high[idx2] = close[idx2] + 0.1
        low[idx2] = close[idx2] - 1.0
        rsi[idx2 - 1] = 56.0
        rsi[idx2] = 62.0
        volume[idx2] = 180.0

    if opposite:
        idx2 = 14
        close[idx2] = min(low[idx2 - 8 : idx2]) - 0.6
        open_[idx2] = close[idx2] + 0.8
        high[idx2] = close[idx2] + 1.0
        low[idx2] = close[idx2] - 0.1
        rsi[idx2 - 1] = 44.0
        rsi[idx2] = 38.0
        volume[idx2] = 180.0
        ema55[idx2 - 5] = 103.0
        ema55[idx2] = 102.0

    if direction == "SHORT":
        close = [100.0 - i * 0.04 for i in range(periods)]
        open_ = [value + 0.1 for value in close]
        high = [value + 0.2 for value in close]
        low = [value - 0.2 for value in close]
        ema55 = [101.0 - i * 0.02 for i in range(periods)]
        close[idx] = min(low[idx - 8 : idx]) - 0.6
        open_[idx] = close[idx] + 0.8
        high[idx] = close[idx] + 1.0
        low[idx] = close[idx] - 0.1
        rsi[idx - 1] = 44.0
        rsi[idx] = 38.0
        volume[idx] = 180.0

    data = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "ema55": ema55,
            "ema55_slope": [0.02 if direction == "LONG" else -0.02] * periods,
            "rsi14": rsi,
            "atr14": atr,
            "volume_sma20": [100.0] * periods,
            "atr_expanding": [True] * periods,
        }
    )
    for column, value in mutations.items():
        data.loc[idx, column] = value
    return data


def test_long_signal_all_conditions_met():
    signals = evaluate_signals_b("BTCUSDT", frame("LONG"), cfg=cfg())
    assert len(signals) == 1
    assert signals[0].direction == "LONG"


def test_short_signal_all_conditions_met():
    signals = evaluate_signals_b("BTCUSDT", frame("SHORT"), cfg=cfg())
    assert len(signals) == 1
    assert signals[0].direction == "SHORT"


def test_no_signal_price_below_ema55():
    data = frame("LONG")
    data.loc[10, "ema55"] = data.loc[10, "close"] + 1
    assert evaluate_signals_b("BTCUSDT", data, cfg=cfg()) == []


def test_no_signal_no_swing_breakout():
    data = frame("LONG")
    data.loc[10, "close"] = data.loc[9, "high"] - 0.1
    assert evaluate_signals_b("BTCUSDT", data, cfg=cfg()) == []


def test_no_signal_weak_volume():
    assert evaluate_signals_b("BTCUSDT", frame("LONG", volume=110.0), cfg=cfg()) == []


def test_no_signal_rsi_falling():
    data = frame("LONG")
    data.loc[9, "rsi14"] = 65
    data.loc[10, "rsi14"] = 60
    assert evaluate_signals_b("BTCUSDT", data, cfg=cfg()) == []


def test_no_signal_weak_candle_body():
    data = frame("LONG")
    data.loc[10, "open"] = data.loc[10, "close"] - 0.2
    assert evaluate_signals_b("BTCUSDT", data, cfg=cfg()) == []


def test_no_signal_bad_atr():
    assert evaluate_signals_b("BTCUSDT", frame("LONG", atr14=5.0), cfg=cfg()) == []


def test_cooldown_blocks_second_signal():
    signals = evaluate_signals_b("BTCUSDT", frame("LONG", duplicate=True), cfg=cfg())
    assert len(signals) == 1


def test_cooldown_allows_opposite_direction():
    signals = evaluate_signals_b("BTCUSDT", frame("LONG", opposite=True), cfg=cfg())
    assert [signal.direction for signal in signals] == ["LONG", "SHORT"]


def test_confidence_all_conditions_true():
    score, label = compute_confidence_b(
        {
            "ema55_slope_strong": True,
            "strong_volume": True,
            "strong_rsi": True,
            "strong_candle_body": True,
            "clean_breakout": True,
            "atr_expanding": True,
        }
    )
    assert score == 100
    assert label == "STRONG"


def test_confidence_partial():
    score, label = compute_confidence_b({"ema55_slope_strong": True, "strong_volume": True})
    assert score == 40
    assert label == "LOW"


def test_low_confidence_signal_rejected():
    data = frame("LONG")
    data.loc[10, "volume"] = 140.0
    data.loc[10, "rsi14"] = 56.5
    data.loc[10, "open"] = data.loc[10, "close"] - 0.55
    data.loc[10, "atr_expanding"] = False
    assert evaluate_signals_b("BTCUSDT", data, cfg=cfg(min_confidence=50)) == []


def test_rr_filter_rejects_low_rr():
    assert evaluate_signals_b("BTCUSDT", frame("LONG"), cfg=cfg(tp2_risk_multiplier=1.5)) == []


def test_strategy_name_in_signal():
    signal = evaluate_signals_b("BTCUSDT", frame("LONG"), cfg=cfg())[0]
    assert signal.strategy_name == STRATEGY_NAME


def test_swing_high_correct():
    data = pd.DataFrame({"high": [1, 4, 3, 2, 9, 5, 6, 7, 8], "low": [0] * 9})
    assert get_swing_high(data, 8, 4) == 9


def test_swing_low_correct():
    data = pd.DataFrame({"high": [1] * 9, "low": [9, 4, 3, 2, 8, 5, 6, 7, 1]})
    assert get_swing_low(data, 8, 4) == 5
