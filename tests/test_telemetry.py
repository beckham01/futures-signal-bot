import csv

import pandas as pd
import pytest

from backtest.strategy import SignalEvent
from bot.telemetry import (
    LIVE_TELEMETRY_PATH,
    PAPER_TELEMETRY_PATH,
    is_paper_trading_mode,
    record_signal_telemetry,
    telemetry_path,
)


def _signal(**overrides) -> SignalEvent:
    base = dict(
        symbol="BTCUSDT",
        direction="SHORT",
        timestamp=pd.Timestamp("2026-03-01T00:00:00Z"),
        entry=100.0,
        stop_loss=102.0,
        tp1=97.0,
        tp2=94.0,
        risk_reward=3.0,
        confidence=80,
        confidence_label="HIGH",
        atr_15m=1.0,
        reasons=["test"],
        strategy_name="strategy_b_fvg_breaker_15m",
        execution_timeframe="15",
        tp1_position_pct=0.5,
        tp2_position_pct=0.5,
    )
    base.update(overrides)
    return SignalEvent(**base)


def test_record_signal_telemetry_writes_row_with_header(tmp_path):
    path = tmp_path / "telemetry.csv"
    record_signal_telemetry(_signal(), regime_classification="net-trending", path=path)

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 1
    row = rows[0]
    assert row["symbol"] == "BTCUSDT"
    assert row["direction"] == "SHORT"
    assert row["strategy_name"] == "strategy_b_fvg_breaker_15m"
    assert row["whitelist_passed"] == "True"
    assert row["direction_passed"] == "True"
    assert row["regime_classification"] == "net-trending"
    assert row["entry"] == "100.0"
    assert row["stop_loss"] == "102.0"
    assert row["tp1"] == "97.0"
    assert row["tp2"] == "94.0"
    assert row["execution_timeframe"] == "15"


def test_record_signal_telemetry_appends_without_duplicate_header(tmp_path):
    path = tmp_path / "telemetry.csv"
    record_signal_telemetry(_signal(), path=path)
    record_signal_telemetry(_signal(symbol="ETHUSDT"), path=path)

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 2
    assert [row["symbol"] for row in rows] == ["BTCUSDT", "ETHUSDT"]


def test_record_signal_telemetry_defaults_regime_to_blank(tmp_path):
    path = tmp_path / "telemetry.csv"
    record_signal_telemetry(_signal(strategy_name="strategy_c_fvg_breaker_4h"), path=path)

    with path.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))

    assert row["regime_classification"] == ""


@pytest.mark.parametrize("value,expected", [("1", True), ("true", True), ("TRUE", True), ("yes", True), ("on", True), ("0", False), ("false", False), ("", False)])
def test_is_paper_trading_mode_reads_env(monkeypatch, value, expected):
    monkeypatch.setenv("PAPER_TRADING_MODE", value)
    assert is_paper_trading_mode() is expected


def test_is_paper_trading_mode_defaults_false_when_unset(monkeypatch):
    monkeypatch.delenv("PAPER_TRADING_MODE", raising=False)
    assert is_paper_trading_mode() is False


def test_telemetry_path_switches_with_paper_mode(monkeypatch):
    monkeypatch.delenv("PAPER_TRADING_MODE", raising=False)
    assert telemetry_path() == LIVE_TELEMETRY_PATH
    monkeypatch.setenv("PAPER_TRADING_MODE", "true")
    assert telemetry_path() == PAPER_TELEMETRY_PATH


def test_record_signal_telemetry_mode_column_matches_paper_flag(tmp_path, monkeypatch):
    monkeypatch.setenv("PAPER_TRADING_MODE", "true")
    path = tmp_path / "telemetry.csv"
    record_signal_telemetry(_signal(), path=path)

    with path.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))

    assert row["mode"] == "paper"
