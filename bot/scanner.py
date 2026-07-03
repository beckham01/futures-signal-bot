"""Live market scanner loop."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

import pandas as pd

from backtest.data_fetcher import BybitDataError, fetch_recent_klines, set_data_source
from backtest.regime import current_btc_regime_label
from backtest.strategy import compute_indicators, evaluate_signals
from backtest.strategy_b import compute_indicators_b, evaluate_signals_b
from backtest.strategy_c import compute_indicators_c, evaluate_signals_c
from bot.telemetry import record_signal_telemetry

LOGGER = logging.getLogger(__name__)
DEFAULT_SYMBOL_SCAN_PAUSE_SECONDS = 1.5
STRATEGY_CONFLICT_WINDOW = pd.Timedelta(hours=1)


def strategy_conflicts_with_a(signal, strategy_a_signals: list) -> bool:
    """Return True when Strategy B should be suppressed by a nearby Strategy A signal."""
    if signal.strategy_name != "strategy_b_fvg_breaker_15m":
        return False
    return any(
        signal.symbol == a_signal.symbol
        and abs(signal.timestamp - a_signal.timestamp) <= STRATEGY_CONFLICT_WINDOW
        for a_signal in strategy_a_signals
    )


_TELEMETRY_STRATEGIES = {"strategy_b_fvg_breaker_15m", "strategy_c_fvg_breaker_4h"}


def _emit_telemetry(signal) -> None:
    """Record telemetry for Strategy B/C live signals (see bot/telemetry.py).

    Telemetry recording must never crash the live scan loop, so any failure
    here (network, disk, or a malformed signal) is logged and swallowed.
    """
    if getattr(signal, "strategy_name", None) not in _TELEMETRY_STRATEGIES:
        return
    try:
        regime_classification = None
        if signal.strategy_name == "strategy_b_fvg_breaker_15m":
            try:
                regime_classification = current_btc_regime_label(signal.timestamp)
            except BybitDataError as exc:
                LOGGER.warning("Could not classify BTC regime for telemetry: %s", exc)
        record_signal_telemetry(signal, regime_classification=regime_classification)
    except Exception as exc:  # noqa: BLE001 - telemetry must not break live scanning
        LOGGER.warning("Failed to record signal telemetry: %s", exc)


def next_scan_delay_seconds(now: datetime | None = None) -> float:
    """Return seconds until the next 15-minute boundary."""
    now = now or datetime.now(timezone.utc)
    seconds_into_hour = now.minute * 60 + now.second + now.microsecond / 1_000_000
    next_boundary = ((int(seconds_into_hour // 900) + 1) * 900)
    return next_boundary - seconds_into_hour


def current_biases(config: dict) -> dict[str, str]:
    """Fetch recent trend candles and return each symbol's coarse 1h bias."""
    set_data_source(config["backtest"].get("bybit_api_base_url"), config["backtest"].get("cache_dir"))
    biases = {}
    for symbol in config["watchlist"]:
        try:
            df_1h = fetch_recent_klines(symbol, config["strategy"]["trend_timeframe"], limit=100)
            df_15m = fetch_recent_klines(symbol, config["strategy"]["entry_timeframe"], limit=100)
            _, df_1h = compute_indicators(df_15m, df_1h, config["strategy"])
            latest = df_1h.dropna().iloc[-1]
            if latest["close"] > latest["ema21"] > latest["ema55"]:
                biases[symbol] = "Bullish"
            elif latest["close"] < latest["ema21"] < latest["ema55"]:
                biases[symbol] = "Bearish"
            else:
                biases[symbol] = "Neutral"
        except (BybitDataError, IndexError, KeyError) as exc:
            LOGGER.warning("Could not compute bias for %s: %s", symbol, exc)
            biases[symbol] = "Unknown"
    return biases


async def scan_once(signal_queue: asyncio.Queue, config: dict) -> int:
    """Scan all symbols once and enqueue generated SignalEvents."""
    set_data_source(config["backtest"].get("bybit_api_base_url"), config["backtest"].get("cache_dir"))
    emitted = 0
    strategy_cfg = config["strategy"]
    strategy_b_cfg = config.get("strategy_b", {})
    strategy_c_cfg = config.get("strategy_c", {})
    enabled_strategies = config.get("strategies", {})
    strategy_a_enabled = enabled_strategies.get("strategy_a_trend_pullback", {}).get("enabled", True)
    strategy_b_enabled = enabled_strategies.get("strategy_b_fvg_breaker_15m", {}).get(
        "enabled",
        strategy_b_cfg.get("enabled", False),
    )
    strategy_c_enabled = enabled_strategies.get("strategy_c_fvg_breaker_4h", {}).get(
        "enabled",
        strategy_c_cfg.get("enabled", False),
    )
    scan_pause = float(config.get("bot", {}).get("symbol_scan_pause_seconds", DEFAULT_SYMBOL_SCAN_PAUSE_SECONDS))
    watchlist = config["watchlist"]
    for index, symbol in enumerate(watchlist):
        LOGGER.info("Scanning %s (%s/%s)", symbol, index + 1, len(watchlist))
        try:
            df_15m_raw = fetch_recent_klines(symbol, strategy_cfg["entry_timeframe"], limit=150)
            await asyncio.sleep(scan_pause)
            strategy_a_signals = []
            strategy_b_signals = []
            strategy_c_signals = []
            latest_timestamp_c = None
            df_15m = df_15m_raw
            df_1h = None
            if strategy_a_enabled:
                df_1h = fetch_recent_klines(symbol, strategy_cfg["trend_timeframe"], limit=150)
                df_15m, df_1h = compute_indicators(df_15m_raw.copy(), df_1h, strategy_cfg)
                strategy_a_signals = evaluate_signals(
                    symbol,
                    df_15m,
                    df_1h,
                    config["bot"]["cooldown_hours"],
                    strategy_cfg,
                )
            if strategy_b_enabled:
                if df_1h is None:
                    df_1h = fetch_recent_klines(symbol, strategy_b_cfg["timeframe_trend"], limit=150)
                df_15m_b, df_1h_b = compute_indicators_b(df_15m_raw.copy(), df_1h.copy(), strategy_b_cfg)
                strategy_b_signals = evaluate_signals_b(
                    symbol,
                    df_15m_b,
                    df_1h_b,
                    strategy_b_cfg["cooldown_hours"],
                    strategy_b_cfg,
                )
            if strategy_c_enabled:
                df_240_raw = fetch_recent_klines(symbol, strategy_c_cfg["timeframe_entry"], limit=150)
                latest_timestamp_c = pd.to_datetime(df_240_raw["timestamp"], utc=True).max()
                df_240_c = compute_indicators_c(df_240_raw.copy(), strategy_c_cfg)
                strategy_c_signals = evaluate_signals_c(
                    symbol,
                    df_240_c,
                    strategy_c_cfg["cooldown_hours"],
                    strategy_c_cfg,
                )
        except BybitDataError as exc:
            LOGGER.warning("Scan failed for %s: %s", symbol, exc)
            continue
        latest_timestamp = pd.to_datetime(df_15m_raw["timestamp"], utc=True).max()
        signals = strategy_a_signals + [
            signal for signal in strategy_b_signals if not strategy_conflicts_with_a(signal, strategy_a_signals)
        ]
        for signal in signals:
            if signal.timestamp != latest_timestamp:
                continue
            _emit_telemetry(signal)
            await signal_queue.put(signal)
            emitted += 1
        for signal in strategy_c_signals:
            if latest_timestamp_c is None or signal.timestamp != latest_timestamp_c:
                continue
            _emit_telemetry(signal)
            await signal_queue.put(signal)
            emitted += 1
        if index < len(watchlist) - 1:
            await asyncio.sleep(scan_pause)
    return emitted


async def scan_loop(signal_queue: asyncio.Queue, config: dict):
    """Infinite scan loop aligned to 15-minute candle closes."""
    while True:
        delay = next_scan_delay_seconds()
        LOGGER.info("Next scan in %.0f seconds", delay)
        await asyncio.sleep(delay)
        count = await scan_once(signal_queue, config)
        LOGGER.info("Scan completed; emitted %s signals", count)
