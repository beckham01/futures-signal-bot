import pandas as pd

from backtest.simulator import simulate_trade
from backtest.strategy import SignalEvent


def make_signal(direction="LONG"):
    return SignalEvent(
        symbol="BTCUSDT",
        direction=direction,
        timestamp=pd.Timestamp("2026-01-01T00:00:00Z"),
        entry=100.0,
        stop_loss=98.5 if direction == "LONG" else 101.5,
        tp1=102.0 if direction == "LONG" else 98.0,
        tp2=104.5 if direction == "LONG" else 95.5,
        risk_reward=3.0,
        confidence=85,
        confidence_label="STRONG",
        atr_15m=1.0,
        reasons=[],
    )


def candles(rows):
    base = pd.Timestamp("2026-01-01T00:15:00Z")
    return pd.DataFrame(
        [
            {
                "timestamp": base + pd.Timedelta(minutes=15 * index),
                "open": row[0],
                "high": row[1],
                "low": row[2],
                "close": row[3],
                "volume": 1.0,
            }
            for index, row in enumerate(rows)
        ]
    )


def test_tp2_hit_returns_correct_r():
    result = simulate_trade(make_signal(), candles([(100, 105, 99, 104)]))
    assert result.outcome == "TP2_HIT"
    assert result.pnl_r == 2.167


def test_stop_hit_returns_minus_one_r():
    result = simulate_trade(make_signal(), candles([(100, 101, 98, 98.2)]))
    assert result.outcome == "STOP_HIT"
    assert result.pnl_r == -1.0


def test_tp1_then_stop_returns_breakeven():
    result = simulate_trade(make_signal(), candles([(100, 102.2, 99.5, 102), (102, 103, 100, 100.1)]))
    assert result.outcome == "TP1_ONLY"
    assert result.pnl_r == 0.667


def test_tp2_r_scales_with_larger_target():
    signal = make_signal()
    signal.tp2 = 115.0
    signal.risk_reward = 10.0

    result = simulate_trade(signal, candles([(100, 116, 99, 115)]))

    assert result.outcome == "TP2_HIT"
    assert result.pnl_r == 5.667


def test_tp2_uses_configured_position_split():
    signal = make_signal()
    signal.tp1_position_pct = 0.40
    signal.tp2_position_pct = 0.60

    result = simulate_trade(signal, candles([(100, 105, 99, 104)]))

    assert result.outcome == "TP2_HIT"
    assert result.pnl_r == 2.333


def test_open_trade_after_96_bars():
    result = simulate_trade(make_signal(), candles([(100, 101, 99, 100.5)] * 100))
    assert result.outcome == "OPEN"
    assert result.bars_held == 96
