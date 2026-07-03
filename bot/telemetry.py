"""Per-signal telemetry logging for live/paper trading.

Every signal emitted by the live scanner (bot/scanner.py `scan_once`) is
recorded here: symbol, direction, whitelist/direction filter status, and -
for Strategy B only - the BTC regime classification at signal time. This
happens regardless of whether Telegram delivery succeeds.

PAPER_TRADING_MODE (env var) does not change the signal pipeline itself
(filters run identically either way) - it only selects which telemetry file
rows are appended to, so paper and live trades are never mixed together when
computing rolling win rate/R per symbol later.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import pandas as pd

from backtest.strategy import SignalEvent

TELEMETRY_FIELDS = [
    "logged_at",
    "signal_timestamp",
    "strategy_name",
    "symbol",
    "direction",
    "whitelist_passed",
    "direction_passed",
    "regime_classification",
    "mode",
    "entry",
    "stop_loss",
    "tp1",
    "tp2",
    "tp1_position_pct",
    "tp2_position_pct",
    "execution_timeframe",
]

LIVE_TELEMETRY_PATH = Path("logs/live_trade_telemetry.csv")
PAPER_TELEMETRY_PATH = Path("logs/paper_trade_telemetry.csv")


def is_paper_trading_mode() -> bool:
    """Return True when PAPER_TRADING_MODE is enabled via environment variable."""
    return os.environ.get("PAPER_TRADING_MODE", "").strip().lower() in {"1", "true", "yes", "on"}


def telemetry_path() -> Path:
    """Return the telemetry CSV path for the currently active mode (live/paper)."""
    return PAPER_TELEMETRY_PATH if is_paper_trading_mode() else LIVE_TELEMETRY_PATH


def record_signal_telemetry(
    signal: SignalEvent,
    regime_classification: str | None = None,
    path: Path | None = None,
) -> None:
    """Append one telemetry row for an emitted (already-filtered) live/paper signal.

    Signals reaching here have, by construction, already passed the strategy's
    whitelist and direction filters - evaluate_signals_b/evaluate_signals_c
    reject non-matching signals before they ever become SignalEvents. Those two
    columns are therefore always True today; they are still recorded
    explicitly so the telemetry log stays self-describing and auditable if
    filter logic changes later.
    """
    target_path = path or telemetry_path()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    is_new_file = not target_path.exists()
    with target_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TELEMETRY_FIELDS)
        if is_new_file:
            writer.writeheader()
        writer.writerow(
            {
                "logged_at": pd.Timestamp.now(tz="UTC").isoformat(),
                "signal_timestamp": pd.Timestamp(signal.timestamp).isoformat(),
                "strategy_name": signal.strategy_name,
                "symbol": signal.symbol,
                "direction": signal.direction,
                "whitelist_passed": True,
                "direction_passed": True,
                "regime_classification": regime_classification or "",
                "mode": "paper" if is_paper_trading_mode() else "live",
                "entry": signal.entry,
                "stop_loss": signal.stop_loss,
                "tp1": signal.tp1,
                "tp2": signal.tp2,
                "tp1_position_pct": signal.tp1_position_pct,
                "tp2_position_pct": signal.tp2_position_pct,
                "execution_timeframe": signal.execution_timeframe,
            }
        )
