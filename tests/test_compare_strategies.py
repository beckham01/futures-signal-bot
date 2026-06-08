import pandas as pd

from backtest.compare_strategies import combine_results
from backtest.simulator import TradeResult
from backtest.strategy import SignalEvent


def result(symbol, timestamp, strategy_name):
    signal = SignalEvent(
        symbol=symbol,
        direction="LONG",
        timestamp=pd.Timestamp(timestamp),
        entry=100.0,
        stop_loss=98.0,
        tp1=103.0,
        tp2=105.0,
        risk_reward=2.5,
        confidence=80,
        confidence_label="STRONG",
        atr_15m=1.0,
        reasons=[],
        strategy_name=strategy_name,
    )
    return TradeResult(signal, "TP1_ONLY", 0.75, 4, 103.0, signal.timestamp + pd.Timedelta(hours=1))


def test_combine_results_drops_strategy_b_conflict_with_strategy_a():
    a = [result("BTCUSDT", "2026-01-01T00:00:00Z", "strategy_a_trend_pullback")]
    b = [
        result("BTCUSDT", "2026-01-01T00:30:00Z", "strategy_b_daily_momentum"),
        result("ETHUSDT", "2026-01-01T00:30:00Z", "strategy_b_daily_momentum"),
    ]

    combined, conflicts = combine_results(a, b)

    assert len(combined) == 2
    assert len(conflicts) == 1
    assert {item.signal.symbol for item in combined} == {"BTCUSDT", "ETHUSDT"}
