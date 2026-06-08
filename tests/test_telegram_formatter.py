import pandas as pd

from backtest.strategy import SignalEvent
from bot.formatter import format_signal, format_status, format_watchlist
from bot.state_manager import BotState


def make_signal():
    return SignalEvent(
        symbol="BTCUSDT",
        direction="LONG",
        timestamp=pd.Timestamp("2026-01-01T00:00:00Z"),
        entry=100.0,
        stop_loss=98.5,
        tp1=102.0,
        tp2=104.5,
        risk_reward=3.0,
        confidence=90,
        confidence_label="STRONG",
        atr_15m=1.0,
        reasons=["EMA trend confirmed", "Volume spike"],
        target_rr=5.0,
        target_note="Extended target 1:5.00 because confidence >= 95",
    )


def test_format_signal_contains_key_levels():
    message = format_signal(make_signal())
    assert "LONG - BTCUSDT" in message
    assert "Entry:" in message
    assert "TP2:" in message
    assert "Extended target 1:5.00" in message
    assert "Strategy:  A+ Trend Pullback" in message
    assert "https://www.tradingview.com/chart/?symbol=BYBIT:BTCUSDT.P" in message
    assert "Not financial advice" in message


def test_format_signal_labels_strategy_b():
    signal = make_signal()
    signal.strategy_name = "strategy_b_daily_momentum"
    message = format_signal(signal)
    assert "Strategy:  Daily Momentum Continuation" in message
    assert "Timeframes: 15m momentum breakout" in message


def test_format_status_and_watchlist():
    state = BotState(cooldowns={}, last_signal=None, signals_today=2, last_scan_time=0.0)
    status = format_status(state, {"watchlist": ["BTCUSDT", "ETHUSDT"]})
    assert "Watching 2 symbols" in status
    assert "BTCUSDT: Bullish" in format_watchlist({"BTCUSDT": "Bullish"})
