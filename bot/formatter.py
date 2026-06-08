"""Telegram message formatting for live signals and commands."""

from __future__ import annotations

from datetime import datetime, timezone

from backtest.strategy import SignalEvent
from bot.state_manager import BotState


def _pct(level: float, entry: float, direction: str) -> float:
    raw = (level - entry) / entry * 100
    return raw if direction == "LONG" else -raw


def tradingview_link(symbol: str) -> str:
    return f"https://www.tradingview.com/chart/?symbol=BYBIT:{symbol}.P"


def strategy_display_name(strategy_name: str) -> str:
    names = {
        "strategy_a_trend_pullback": "A+ Trend Pullback",
        "strategy_b_daily_momentum": "Daily Momentum Continuation",
    }
    return names.get(strategy_name, strategy_name)


def strategy_timeframes(strategy_name: str) -> str:
    if strategy_name == "strategy_b_daily_momentum":
        return "15m momentum breakout"
    return "1h trend + 15m entry"


def format_signal(signal: SignalEvent) -> str:
    icon = "[LONG]" if signal.direction == "LONG" else "[SHORT]"
    lines = [
        f"{icon} {signal.direction} - {signal.symbol}",
        "--------------------",
        f"Entry:    ${signal.entry:.4f} zone",
        f"SL:       ${signal.stop_loss:.4f} ({_pct(signal.stop_loss, signal.entry, signal.direction):+.2f}%)",
        f"TP1:      ${signal.tp1:.4f} ({_pct(signal.tp1, signal.entry, signal.direction):+.2f}%) - exit 50%",
        f"TP2:      ${signal.tp2:.4f} ({_pct(signal.tp2, signal.entry, signal.direction):+.2f}%) - exit 50%",
        f"R:R:      1 : {signal.risk_reward:.2f}",
        f"Target:   {signal.target_note or 'Base target'}",
        "--------------------",
        f"Strategy:  {strategy_display_name(signal.strategy_name)}",
        f"Confidence: {signal.confidence} / 100 [{signal.confidence_label}]",
        f"Timeframes: {strategy_timeframes(signal.strategy_name)}",
        "Leverage:   5x context (not advice)",
        f"Chart:      {tradingview_link(signal.symbol)}",
        "--------------------",
        "Reasons:",
    ]
    lines.extend(f"- {reason}" for reason in signal.reasons)
    lines.extend(
        [
            "--------------------",
            f"{signal.timestamp.strftime('%Y-%m-%d %H:%M UTC')}",
            "Signal only. Not financial advice.",
        ]
    )
    return "\n".join(lines)


def format_status(state: BotState, config: dict) -> str:
    last_scan = "never"
    if state.last_scan_time:
        last_scan = datetime.fromtimestamp(state.last_scan_time, timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return (
        f"Bot online. Last scan: {last_scan}. Mode: live.\n"
        f"Watching {len(config.get('watchlist', []))} symbols. Signals today: {state.signals_today}."
    )


def format_watchlist(biases: dict[str, str]) -> str:
    if not biases:
        return "Watchlist is empty."
    return "\n".join(f"{symbol}: {bias}" for symbol, bias in sorted(biases.items()))
