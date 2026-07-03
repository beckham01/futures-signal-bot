import asyncio
from datetime import datetime, timezone

import pandas as pd
import pytest

from bot.scanner import next_scan_delay_seconds, scan_once, strategy_conflicts_with_a


def test_next_scan_delay_seconds():
    now = datetime(2026, 1, 1, 14, 7, 30, tzinfo=timezone.utc)
    assert next_scan_delay_seconds(now) == 450


@pytest.mark.asyncio
async def test_scan_once_enqueues_signals(monkeypatch):
    async_queue = asyncio.Queue()

    class Signal:
        symbol = "BTCUSDT"
        direction = "LONG"
        timestamp = pd.Timestamp("2026-01-01T00:30:00Z")

    def fake_fetch_recent(symbol, interval, limit=100):
        return pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-01-01", periods=3, freq="15min", tz="UTC"),
                "open": [1.0, 1.0, 1.0],
                "high": [1.0, 1.0, 1.0],
                "low": [1.0, 1.0, 1.0],
                "close": [1.0, 1.0, 1.0],
                "volume": [1.0, 1.0, 1.0],
            }
        )

    monkeypatch.setattr("bot.scanner.fetch_recent_klines", fake_fetch_recent)
    monkeypatch.setattr("bot.scanner.compute_indicators", lambda df_15m, df_1h, cfg: (df_15m, df_1h))
    monkeypatch.setattr("bot.scanner.evaluate_signals", lambda *args, **kwargs: [Signal()])
    monkeypatch.setattr("bot.scanner.compute_indicators_b", lambda df_15m, cfg: df_15m)
    monkeypatch.setattr("bot.scanner.evaluate_signals_b", lambda *args, **kwargs: [])
    config = {
        "watchlist": ["BTCUSDT"],
        "strategies": {
            "strategy_a_trend_pullback": {"enabled": True},
            "strategy_b_fvg_breaker_15m": {"enabled": False},
        },
        "strategy": {"entry_timeframe": "15", "trend_timeframe": "60"},
        "strategy_b": {"timeframe_entry": "15", "timeframe_trend": "60", "cooldown_hours": 4},
        "bot": {"cooldown_hours": 4, "symbol_scan_pause_seconds": 0},
        "backtest": {"bybit_api_base_url": "https://example.test", "cache_dir": "data/cache"},
    }

    emitted = await scan_once(async_queue, config)
    assert emitted == 1
    assert (await async_queue.get()).symbol == "BTCUSDT"


@pytest.mark.asyncio
async def test_scan_once_paces_between_recent_fetches(monkeypatch):
    async_queue = asyncio.Queue()
    sleeps = []

    def fake_fetch_recent(symbol, interval, limit=100):
        return pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-01-01", periods=3, freq="15min", tz="UTC"),
                "open": [1.0, 1.0, 1.0],
                "high": [1.0, 1.0, 1.0],
                "low": [1.0, 1.0, 1.0],
                "close": [1.0, 1.0, 1.0],
                "volume": [1.0, 1.0, 1.0],
            }
        )

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr("bot.scanner.fetch_recent_klines", fake_fetch_recent)
    monkeypatch.setattr("bot.scanner.compute_indicators", lambda df_15m, df_1h, cfg: (df_15m, df_1h))
    monkeypatch.setattr("bot.scanner.evaluate_signals", lambda *args, **kwargs: [])
    monkeypatch.setattr("bot.scanner.compute_indicators_b", lambda df_15m, cfg: df_15m)
    monkeypatch.setattr("bot.scanner.evaluate_signals_b", lambda *args, **kwargs: [])
    monkeypatch.setattr("bot.scanner.asyncio.sleep", fake_sleep)
    config = {
        "watchlist": ["BTCUSDT", "ETHUSDT"],
        "strategies": {
            "strategy_a_trend_pullback": {"enabled": True},
            "strategy_b_fvg_breaker_15m": {"enabled": False},
        },
        "strategy": {"entry_timeframe": "15", "trend_timeframe": "60"},
        "strategy_b": {"timeframe_entry": "15", "timeframe_trend": "60", "cooldown_hours": 4},
        "bot": {"cooldown_hours": 4, "symbol_scan_pause_seconds": 1.5},
        "backtest": {"bybit_api_base_url": "https://example.test", "cache_dir": "data/cache"},
    }

    assert await scan_once(async_queue, config) == 0
    assert sleeps == [1.5, 1.5, 1.5]


@pytest.mark.asyncio
async def test_scan_once_enqueues_strategy_b_when_enabled(monkeypatch):
    async_queue = asyncio.Queue()

    class Signal:
        symbol = "BTCUSDT"
        direction = "LONG"
        timestamp = pd.Timestamp("2026-01-01T00:30:00Z")
        strategy_name = "strategy_b_fvg_breaker_15m"

    def fake_fetch_recent(symbol, interval, limit=100):
        return pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-01-01", periods=3, freq="15min", tz="UTC"),
                "open": [1.0, 1.0, 1.0],
                "high": [1.0, 1.0, 1.0],
                "low": [1.0, 1.0, 1.0],
                "close": [1.0, 1.0, 1.0],
                "volume": [1.0, 1.0, 1.0],
            }
        )

    monkeypatch.setattr("bot.scanner.fetch_recent_klines", fake_fetch_recent)
    monkeypatch.setattr("bot.scanner.compute_indicators_b", lambda df_15m, df_1h, cfg: (df_15m, df_1h))
    monkeypatch.setattr("bot.scanner.evaluate_signals_b", lambda *args, **kwargs: [Signal()])
    config = {
        "watchlist": ["BTCUSDT"],
        "strategies": {
            "strategy_a_trend_pullback": {"enabled": False},
            "strategy_b_fvg_breaker_15m": {"enabled": True},
        },
        "strategy": {"entry_timeframe": "15", "trend_timeframe": "60"},
        "strategy_b": {"timeframe_entry": "15", "timeframe_trend": "60", "cooldown_hours": 4},
        "bot": {"cooldown_hours": 4, "symbol_scan_pause_seconds": 0},
        "backtest": {"bybit_api_base_url": "https://example.test", "cache_dir": "data/cache"},
    }

    emitted = await scan_once(async_queue, config)

    assert emitted == 1
    assert (await async_queue.get()).strategy_name == "strategy_b_fvg_breaker_15m"


def test_strategy_b_conflicts_with_nearby_strategy_a_signal():
    class Signal:
        symbol = "BTCUSDT"

        def __init__(self, strategy_name, timestamp):
            self.strategy_name = strategy_name
            self.timestamp = pd.Timestamp(timestamp)

    strategy_a = [Signal("strategy_a_trend_pullback", "2026-01-01T00:00:00Z")]
    strategy_b = Signal("strategy_b_fvg_breaker_15m", "2026-01-01T00:30:00Z")
    assert strategy_conflicts_with_a(strategy_b, strategy_a) is True
