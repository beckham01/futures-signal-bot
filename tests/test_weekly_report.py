import csv

import pandas as pd
import pytest

from backtest.simulator import TradeResult
from backtest.strategy import SignalEvent
from backtest.weekly_report import (
    ALERT_MIN_TRADES,
    STRATEGY_WIN_RATE_TARGETS,
    aggregate_by_symbol_direction,
    check_win_rate_alerts,
    compare_live_vs_backtest,
    load_backtest_trade_log,
    load_telemetry_signals,
    resolve_trade_outcomes,
    rolling_win_rate,
)


def _telemetry_row(**overrides):
    base = {
        "logged_at": "2026-03-01T00:00:00+00:00",
        "signal_timestamp": "2026-03-01T00:00:00+00:00",
        "strategy_name": "strategy_b_fvg_breaker_15m",
        "symbol": "BTCUSDT",
        "direction": "SHORT",
        "whitelist_passed": "True",
        "direction_passed": "True",
        "regime_classification": "net-trending",
        "mode": "live",
        "entry": "100.0",
        "stop_loss": "102.0",
        "tp1": "97.0",
        "tp2": "94.0",
        "tp1_position_pct": "0.5",
        "tp2_position_pct": "0.5",
        "execution_timeframe": "15",
    }
    base.update(overrides)
    return base


def _write_telemetry_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _result(symbol, direction, outcome, pnl_r, strategy_name="strategy_b_fvg_breaker_15m", timestamp="2026-03-01"):
    signal = SignalEvent(
        symbol=symbol,
        direction=direction,
        timestamp=pd.Timestamp(timestamp),
        entry=100.0,
        stop_loss=102.0,
        tp1=97.0,
        tp2=94.0,
        risk_reward=3.0,
        confidence=80,
        confidence_label="HIGH",
        atr_15m=1.0,
        reasons=[],
        strategy_name=strategy_name,
    )
    return TradeResult(signal, outcome, pnl_r, 5, 97.0, pd.Timestamp(timestamp) + pd.Timedelta(hours=1))


def test_load_telemetry_signals_reconstructs_from_csv(tmp_path):
    path = tmp_path / "telemetry.csv"
    _write_telemetry_csv(path, [_telemetry_row()])

    signals = load_telemetry_signals(path)

    assert len(signals) == 1
    signal = signals[0]
    assert signal.symbol == "BTCUSDT"
    assert signal.direction == "SHORT"
    assert signal.entry == 100.0
    assert signal.stop_loss == 102.0
    assert signal.tp1 == 97.0
    assert signal.tp2 == 94.0
    assert signal.execution_timeframe == "15"
    assert signal.strategy_name == "strategy_b_fvg_breaker_15m"


def test_load_telemetry_signals_missing_file_returns_empty(tmp_path):
    assert load_telemetry_signals(tmp_path / "does_not_exist.csv") == []


def test_resolve_trade_outcomes_uses_simulator(tmp_path, monkeypatch):
    path = tmp_path / "telemetry.csv"
    _write_telemetry_csv(path, [_telemetry_row()])
    signals = load_telemetry_signals(path)

    ts = pd.date_range("2026-03-01", periods=10, freq="15min", tz="UTC")
    future = pd.DataFrame(
        {
            "timestamp": ts,
            "open": [100.0] * 10,
            "high": [100.5] * 10,
            "low": [96.0] * 10,
            "close": [96.5] * 10,
            "volume": [1.0] * 10,
        }
    )

    monkeypatch.setattr("backtest.weekly_report.fetch_klines", lambda symbol, interval, start_ms, end_ms: future)

    results = resolve_trade_outcomes(signals)

    assert len(results) == 1
    assert results[0].outcome in {"TP1_ONLY", "TP2_HIT", "STOP_HIT", "OPEN"}


def test_aggregate_by_symbol_direction_groups_correctly():
    results = [
        _result("BTCUSDT", "SHORT", "TP2_HIT", 2.0),
        _result("BTCUSDT", "SHORT", "STOP_HIT", -1.0),
        _result("ETHUSDT", "SHORT", "TP2_HIT", 2.0, strategy_name="strategy_c_fvg_breaker_4h"),
    ]

    table = aggregate_by_symbol_direction(results)

    assert len(table) == 2
    btc_row = table[table["symbol"] == "BTCUSDT"].iloc[0]
    assert btc_row["count"] == 2
    assert btc_row["total_r"] == 1.0
    eth_row = table[table["symbol"] == "ETHUSDT"].iloc[0]
    assert eth_row["strategy_name"] == "strategy_c_fvg_breaker_4h"


def test_rolling_win_rate_returns_none_below_min_trades():
    results = [_result("BTCUSDT", "SHORT", "TP2_HIT", 2.0)] * 5
    assert rolling_win_rate(results, window=ALERT_MIN_TRADES) is None


def test_rolling_win_rate_computes_over_window():
    wins = [_result("BTCUSDT", "SHORT", "TP2_HIT", 2.0, timestamp=f"2026-03-{i:02d}") for i in range(1, 11)]
    losses = [_result("BTCUSDT", "SHORT", "STOP_HIT", -1.0, timestamp=f"2026-03-{i:02d}") for i in range(11, 16)]
    results = wins + losses
    win_rate = rolling_win_rate(results, window=15)
    assert win_rate == pytest.approx(10 / 15 * 100)


def test_check_win_rate_alerts_flags_significant_drop():
    target = STRATEGY_WIN_RATE_TARGETS["strategy_b_fvg_breaker_15m"]
    assert target - 20 > 0
    losing_results = [
        _result("BTCUSDT", "SHORT", "STOP_HIT", -1.0, timestamp=f"2026-03-{i:02d}") for i in range(1, 16)
    ]
    warnings = check_win_rate_alerts({"strategy_b_fvg_breaker_15m": losing_results})
    assert any("strategy_b_fvg_breaker_15m" in warning for warning in warnings)


def test_check_win_rate_alerts_silent_when_on_target():
    winning_results = [
        _result("BTCUSDT", "SHORT", "TP2_HIT", 2.0, timestamp=f"2026-03-{i:02d}") for i in range(1, 16)
    ]
    warnings = check_win_rate_alerts({"strategy_b_fvg_breaker_15m": winning_results})
    assert warnings == []


def test_check_win_rate_alerts_skips_when_too_few_trades():
    few_results = [_result("BTCUSDT", "SHORT", "STOP_HIT", -1.0)] * 5
    warnings = check_win_rate_alerts({"strategy_b_fvg_breaker_15m": few_results})
    assert warnings == []


def test_load_backtest_trade_log_missing_file_returns_empty(tmp_path):
    table = load_backtest_trade_log(tmp_path / "does_not_exist.csv")
    assert table.empty


def test_compare_live_vs_backtest_joins_and_computes_delta(tmp_path):
    log_path = tmp_path / "strategy_b_trade_log.csv"
    with log_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["symbol", "entry_time", "entry_price", "direction", "exit_time", "exit_price", "exit_reason", "R_outcome"])
        writer.writerow(["BTCUSDT", "2026-01-01T00:00:00+00:00", 100.0, "SHORT", "2026-01-01T01:00:00+00:00", 98.0, "TP2", 2.0])
        writer.writerow(["BTCUSDT", "2026-01-02T00:00:00+00:00", 100.0, "SHORT", "2026-01-02T01:00:00+00:00", 102.0, "SL", -1.0])

    live_results = [
        _result("BTCUSDT", "SHORT", "TP2_HIT", 2.0),
        _result("BTCUSDT", "SHORT", "TP2_HIT", 2.0),
    ]

    comparison = compare_live_vs_backtest(live_results, log_path)

    assert len(comparison) == 1
    row = comparison.iloc[0]
    assert row["backtest_count"] == 2
    assert row["backtest_win_rate"] == 50.0
    assert row["win_rate"] == 100.0
    assert row["win_rate_delta"] == 50.0


def test_compare_live_vs_backtest_empty_live_returns_empty(tmp_path):
    comparison = compare_live_vs_backtest([], tmp_path / "missing.csv")
    assert comparison.empty
