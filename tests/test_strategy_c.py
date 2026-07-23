import logging

import pandas as pd

from backtest.strategy_c import STRATEGY_NAME, compute_confidence_c, evaluate_signals_c


def cfg(**overrides):
    base = {
        "timeframe_entry": "240",
        "ema_fast": 21,
        "ema_slow": 55,
        "rsi_period": 14,
        "atr_period": 14,
        "fvg_min_gap_atr": 0.3,
        "fvg_max_age_candles": 48,
        "breaker_confirm_candles": 3,
        "breaker_max_age_candles": 96,
        "sl_atr_buffer": 0.5,
        "tp1_risk_multiplier": 1.0,
        "tp2_risk_multiplier": 2.0,
        "tp1_position_pct": 0.40,
        "tp2_position_pct": 0.60,
        "min_risk_reward": 2.0,
        "min_confidence": 60,
        "candle_body_min_pct": 0.45,
        "atr_max_pct": 0.06,
        "atr_min_pct": 0.005,
        "cooldown_hours": 12,
    }
    base.update(overrides)
    return base


def frame(direction="LONG", **mutations):
    periods = 16
    idx = 9
    ts = pd.date_range("2026-01-01", periods=periods, freq="4h", tz="UTC")
    close = [102.0] * periods
    open_ = [101.0] * periods
    high = [103.0] * periods
    low = [100.0] * periods
    rsi = [56.0] * periods
    ema21 = [101.0 + i * 0.1 for i in range(periods)]
    ema55 = [99.0 + i * 0.05 for i in range(periods)]
    ema21_slope = [0.01] * periods
    ema55_slope = [0.01] * periods
    if direction == "SHORT":
        close = [98.0] * periods
        open_ = [99.0] * periods
        high = [100.0] * periods
        low = [97.0] * periods
        rsi = [44.0] * periods
        ema21 = [99.0 - i * 0.1 for i in range(periods)]
        ema55 = [101.0 - i * 0.05 for i in range(periods)]
        ema21_slope = [-0.01] * periods
        ema55_slope = [-0.01] * periods
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": [100.0] * periods,
            "ema21": ema21,
            "ema55": ema55,
            "ema21_slope": ema21_slope,
            "ema55_slope": ema55_slope,
            "rsi14": rsi,
            "atr14": [1.0] * periods,
        }
    )
    if direction == "LONG":
        df.loc[idx, ["open", "high", "low", "close", "rsi14"]] = [100.0, 103.0, 100.5, 102.0, 58.0]
    else:
        df.loc[idx, ["open", "high", "low", "close", "rsi14"]] = [101.2, 101.5, 97.0, 99.0, 42.0]
    for column, value in mutations.items():
        df.loc[idx, column] = value
    return df


def patch_confluence(monkeypatch, direction="bullish", gap_size=1.2, formed_at=5):
    monkeypatch.setattr(
        "backtest.strategy_c.detect_fvgs",
        lambda *args, **kwargs: [{"type": direction, "zone": (99.0, 101.0), "formed_at": 5, "filled": False, "gap_size": gap_size}],
    )
    monkeypatch.setattr(
        "backtest.strategy_c.detect_breaker_block",
        lambda *args, **kwargs: {"type": args[3], "zone": (98.5, 101.0), "formed_at": formed_at, "mitigated": False},
    )


def test_long_signal_all_conditions_met(monkeypatch):
    patch_confluence(monkeypatch)
    signals = evaluate_signals_c("BTCUSDT", frame("LONG"), cfg=cfg())
    assert len(signals) == 1
    assert signals[0].direction == "LONG"


def test_short_signal_all_conditions_met(monkeypatch):
    patch_confluence(monkeypatch, "bearish")
    signals = evaluate_signals_c("BTCUSDT", frame("SHORT"), cfg=cfg())
    assert len(signals) == 1
    assert signals[0].direction == "SHORT"


def test_no_signal_flat_4h_ema(monkeypatch):
    patch_confluence(monkeypatch)
    assert evaluate_signals_c("BTCUSDT", frame("LONG", ema21=100.0, ema55=100.0), cfg=cfg()) == []


def test_no_signal_no_confluence_zone(monkeypatch):
    monkeypatch.setattr("backtest.strategy_c.detect_fvgs", lambda *args, **kwargs: [])
    monkeypatch.setattr("backtest.strategy_c.detect_breaker_block", lambda *args, **kwargs: None)
    assert evaluate_signals_c("BTCUSDT", frame("LONG"), cfg=cfg()) == []


def test_no_signal_rsi_overbought(monkeypatch):
    patch_confluence(monkeypatch)
    assert evaluate_signals_c("BTCUSDT", frame("LONG", rsi14=75.0), cfg=cfg()) == []


def test_no_signal_rr_below_2(monkeypatch):
    patch_confluence(monkeypatch)
    assert evaluate_signals_c("BTCUSDT", frame("LONG"), cfg=cfg(tp2_risk_multiplier=1.0)) == []


def test_no_signal_weak_candle_body(monkeypatch):
    patch_confluence(monkeypatch)
    assert evaluate_signals_c("BTCUSDT", frame("LONG", open=101.8), cfg=cfg()) == []


def test_cooldown_12h_blocks_second_signal(monkeypatch):
    patch_confluence(monkeypatch)
    data = frame("LONG")
    data.loc[10, ["open", "high", "low", "close", "rsi14"]] = [100.0, 103.0, 100.5, 102.0, 58.0]
    assert len(evaluate_signals_c("BTCUSDT", data, cfg=cfg())) == 1


def test_cooldown_12h_allows_after_expiry(monkeypatch):
    patch_confluence(monkeypatch)
    data = frame("LONG")
    data.loc[13, ["open", "high", "low", "close", "rsi14"]] = [100.0, 103.0, 100.5, 102.0, 58.0]
    assert len(evaluate_signals_c("BTCUSDT", data, cfg=cfg())) == 2


def test_confidence_all_conditions_true():
    score, label = compute_confidence_c(
        {
            "ema_slopes_aligned": True,
            "large_fvg": True,
            "fresh_breaker": True,
            "healthy_rsi": True,
            "strong_rejection_body": True,
        }
    )
    assert score == 100
    assert label == "STRONG"


def test_tp2_carries_60_pct_position(monkeypatch):
    patch_confluence(monkeypatch)
    signal = evaluate_signals_c("BTCUSDT", frame("LONG"), cfg=cfg())[0]
    assert signal.tp1_position_pct == 0.40
    assert signal.tp2_position_pct == 0.60


def test_strategy_name_is_strategy_c(monkeypatch):
    patch_confluence(monkeypatch)
    signal = evaluate_signals_c("BTCUSDT", frame("LONG"), cfg=cfg())[0]
    assert signal.strategy_name == STRATEGY_NAME

def test_signal_rejected_symbol_not_whitelisted(monkeypatch, caplog):
    # XRPUSDT is not in STRATEGY_C_WHITELIST (BTC/ETH/DOGE/SOL only).
    patch_confluence(monkeypatch)
    caplog.set_level(logging.DEBUG)
    signals = evaluate_signals_c("XRPUSDT", frame("LONG"), cfg=cfg())
    assert signals == []
    assert any(getattr(record, "reason", None) == "symbol_not_whitelisted" for record in caplog.records)


def test_signal_rejected_direction_filtered(monkeypatch, caplog):
    patch_confluence(monkeypatch)
    monkeypatch.setattr("config.strategy_filters.STRATEGY_C_DIRECTIONS_ALLOWED", ["SHORT"])
    caplog.set_level(logging.DEBUG)
    signals = evaluate_signals_c("BTCUSDT", frame("LONG"), cfg=cfg())
    assert signals == []
    assert any(getattr(record, "reason", None) == "direction_filtered" for record in caplog.records)


def test_filter_whitelist_config_override(monkeypatch):
    patch_confluence(monkeypatch)
    monkeypatch.setattr("config.strategy_filters.STRATEGY_C_WHITELIST", ["XRPUSDT"])
    assert evaluate_signals_c("BTCUSDT", frame("LONG"), cfg=cfg()) == []
    monkeypatch.setattr("config.strategy_filters.STRATEGY_C_WHITELIST", ["BTCUSDT"])
    assert len(evaluate_signals_c("BTCUSDT", frame("LONG"), cfg=cfg())) == 1


def test_filter_direction_config_override(monkeypatch):
    patch_confluence(monkeypatch)
    monkeypatch.setattr("config.strategy_filters.STRATEGY_C_DIRECTIONS_ALLOWED", ["SHORT"])
    assert evaluate_signals_c("BTCUSDT", frame("LONG"), cfg=cfg()) == []
    monkeypatch.setattr("config.strategy_filters.STRATEGY_C_DIRECTIONS_ALLOWED", ["LONG", "SHORT"])
    assert len(evaluate_signals_c("BTCUSDT", frame("LONG"), cfg=cfg())) == 1


def test_strategy_c_never_calls_regime_gate(monkeypatch):
    # Strategy C must remain ungated by the BTC regime filter (regime filtering
    # hurts C because it strips out DOGEUSDT's regime-independent wins). If
    # strategy_c ever started calling is_btc_regime_trending, this would blow up.
    def _boom(*args, **kwargs):
        raise AssertionError("Strategy C must never call the BTC regime gate")

    monkeypatch.setattr("backtest.regime.is_btc_regime_trending", _boom)
    patch_confluence(monkeypatch)
    signals = evaluate_signals_c("BTCUSDT", frame("LONG"), cfg=cfg())
    assert len(signals) == 1


def test_strategy_c_module_does_not_import_regime():
    import backtest.strategy_c as strategy_c_module

    assert not hasattr(strategy_c_module, "is_btc_regime_trending")
    assert "regime" not in vars(strategy_c_module)
