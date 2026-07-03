"""Weekly rollup of live/paper trade telemetry.

Aggregates recorded Strategy B / Strategy C signals (bot/telemetry.py) by
symbol and direction, resolves each signal's outcome against fresh candle
data (reusing the same simulator used by backtests), and reports rolling
win rate / total R per symbol per strategy. Also raises a warning when a
strategy's rolling win rate drops too far below its backtested target.

CLI usage:
    python -m backtest.weekly_report
    python -m backtest.weekly_report --telemetry-path logs/paper_trade_telemetry.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from backtest.data_fetcher import fetch_klines
from backtest.report import _group_metrics, _total_r
from backtest.simulator import TradeResult, simulate_all
from backtest.strategy import SignalEvent

LOGGER = logging.getLogger(__name__)

# Backtested targets these live/paper strategies are expected to track.
STRATEGY_WIN_RATE_TARGETS = {
    "strategy_b_fvg_breaker_15m": 62.0,
    "strategy_c_fvg_breaker_4h": 68.0,
}
ALERT_DROP_THRESHOLD_PP = 15.0
ALERT_MIN_TRADES = 15


def load_telemetry_signals(path: str | Path) -> list[SignalEvent]:
    """Load recorded telemetry rows back into SignalEvent objects for simulation."""
    path = Path(path)
    if not path.exists():
        return []
    df = pd.read_csv(path)
    signals: list[SignalEvent] = []
    for _, row in df.iterrows():
        signals.append(
            SignalEvent(
                symbol=row["symbol"],
                direction=row["direction"],
                timestamp=pd.Timestamp(row["signal_timestamp"]),
                entry=float(row["entry"]),
                stop_loss=float(row["stop_loss"]),
                tp1=float(row["tp1"]),
                tp2=float(row["tp2"]),
                risk_reward=0.0,
                confidence=0,
                confidence_label="",
                atr_15m=0.0,
                reasons=[],
                strategy_name=row["strategy_name"],
                execution_timeframe=str(row["execution_timeframe"]),
                tp1_position_pct=float(row["tp1_position_pct"]),
                tp2_position_pct=float(row["tp2_position_pct"]),
            )
        )
    return signals


def resolve_trade_outcomes(signals: list[SignalEvent]) -> list[TradeResult]:
    """Fetch forward candle data for each signal's symbol/timeframe and resolve outcomes."""
    if not signals:
        return []
    data: dict[str, dict[str, pd.DataFrame]] = {}
    end_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
    for signal in signals:
        timeframe = signal.execution_timeframe
        if signal.symbol in data and timeframe in data[signal.symbol]:
            continue
        start_ms = int(pd.Timestamp(signal.timestamp).timestamp() * 1000)
        df = fetch_klines(signal.symbol, timeframe, start_ms, end_ms)
        data.setdefault(signal.symbol, {})[timeframe] = df
    return simulate_all(signals, data)


def rolling_win_rate(results: list[TradeResult], window: int = ALERT_MIN_TRADES) -> float | None:
    """Return the win rate (%) over the most recent `window` trades, or None if too few."""
    if len(results) < window:
        return None
    recent = sorted(results, key=lambda result: result.signal.timestamp)[-window:]
    _, win_rate, _ = _group_metrics(recent)
    return win_rate


def check_win_rate_alerts(results_by_strategy: dict[str, list[TradeResult]]) -> list[str]:
    """Return warning strings for any strategy whose rolling win rate has dropped too far."""
    warnings: list[str] = []
    for strategy_name, target in STRATEGY_WIN_RATE_TARGETS.items():
        results = results_by_strategy.get(strategy_name, [])
        win_rate = rolling_win_rate(results, ALERT_MIN_TRADES)
        if win_rate is None:
            continue
        drop = target - win_rate
        if drop > ALERT_DROP_THRESHOLD_PP:
            message = (
                f"{strategy_name}: rolling win rate {win_rate:.1f}% is {drop:.1f}pp below "
                f"target {target:.1f}% over the last {ALERT_MIN_TRADES} trades - investigate "
                "before scaling position size."
            )
            warnings.append(message)
            LOGGER.warning(message)
    return warnings


def aggregate_by_symbol_direction(results: list[TradeResult]) -> pd.DataFrame:
    """Return a DataFrame with count/win_rate/total_r per strategy+symbol+direction."""
    rows = []
    groups: dict[tuple[str, str, str], list[TradeResult]] = {}
    for result in results:
        key = (result.signal.strategy_name, result.signal.symbol, result.signal.direction)
        groups.setdefault(key, []).append(result)
    for (strategy_name, symbol, direction), group_results in sorted(groups.items()):
        count, win_rate, avg_r = _group_metrics(group_results)
        rows.append(
            {
                "strategy_name": strategy_name,
                "symbol": symbol,
                "direction": direction,
                "count": count,
                "win_rate": round(win_rate, 1),
                "total_r": round(_total_r(group_results), 2),
                "avg_r": round(avg_r, 2),
            }
        )
    return pd.DataFrame(rows)


def load_backtest_trade_log(path: str | Path) -> pd.DataFrame:
    """Load a backtest trade log CSV (strategy_b_trade_log.csv / strategy_c_trade_log.csv
    schema: symbol, entry_time, entry_price, direction, exit_time, exit_price,
    exit_reason, R_outcome) for use as a recalibration baseline.
    """
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=["symbol", "direction", "R_outcome"])
    return pd.read_csv(path)


def _backtest_log_metrics_by_symbol_direction(trade_log: pd.DataFrame) -> pd.DataFrame:
    if trade_log.empty:
        return pd.DataFrame(columns=["symbol", "direction", "backtest_count", "backtest_win_rate", "backtest_avg_r"])
    rows = []
    for (symbol, direction), group in trade_log.groupby(["symbol", "direction"]):
        count = len(group)
        wins = group["exit_reason"].isin(["TP1", "TP2"]).sum()
        rows.append(
            {
                "symbol": symbol,
                "direction": direction,
                "backtest_count": count,
                "backtest_win_rate": round(wins / count * 100, 1) if count else 0.0,
                "backtest_avg_r": round(group["R_outcome"].astype(float).mean(), 2) if count else 0.0,
            }
        )
    return pd.DataFrame(rows)


def compare_live_vs_backtest(results: list[TradeResult], backtest_log_path: str | Path) -> pd.DataFrame:
    """Recalibration table: live/paper performance per symbol+direction next to the
    original backtest's performance for the same symbol+direction, for human review.
    Never written back to config automatically - see README's rollout policy.
    """
    live_table = aggregate_by_symbol_direction(results)
    backtest_table = _backtest_log_metrics_by_symbol_direction(load_backtest_trade_log(backtest_log_path))
    if live_table.empty:
        return live_table
    merged = live_table.merge(backtest_table, on=["symbol", "direction"], how="left")
    merged["win_rate_delta"] = (merged["win_rate"] - merged["backtest_win_rate"]).round(1)
    return merged


def build_weekly_report(telemetry_path: str | Path) -> tuple[pd.DataFrame, list[str]]:
    """Load telemetry, resolve outcomes, and return (summary_table, alert_warnings)."""
    signals = load_telemetry_signals(telemetry_path)
    results = resolve_trade_outcomes(signals)
    results_by_strategy: dict[str, list[TradeResult]] = {}
    for result in results:
        results_by_strategy.setdefault(result.signal.strategy_name, []).append(result)
    summary = aggregate_by_symbol_direction(results)
    warnings = check_win_rate_alerts(results_by_strategy)
    return summary, warnings


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Weekly live/paper trade telemetry report.")
    parser.add_argument("--telemetry-path", default="logs/live_trade_telemetry.csv")
    parser.add_argument(
        "--backtest-log-path",
        help="Optional backtest trade log (e.g. strategy_b_trade_log.csv) to compare live/paper "
        "performance against, for human recalibration review only.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    signals = load_telemetry_signals(args.telemetry_path)
    results = resolve_trade_outcomes(signals)
    results_by_strategy: dict[str, list[TradeResult]] = {}
    for result in results:
        results_by_strategy.setdefault(result.signal.strategy_name, []).append(result)
    summary = aggregate_by_symbol_direction(results)
    warnings = check_win_rate_alerts(results_by_strategy)

    print("=== WEEKLY TELEMETRY REPORT ===")
    print(f"Telemetry source: {args.telemetry_path}")
    if summary.empty:
        print("No resolved trades yet.")
    else:
        print(summary.to_string(index=False))
    if args.backtest_log_path:
        comparison = compare_live_vs_backtest(results, args.backtest_log_path)
        print("\n--- LIVE/PAPER VS BACKTEST (recalibration review only) ---")
        if comparison.empty:
            print("No resolved trades yet to compare.")
        else:
            print(comparison.to_string(index=False))
    if warnings:
        print("\n--- ALERTS ---")
        for warning in warnings:
            print(f"WARNING: {warning}")
    return 1 if warnings else 0


if __name__ == "__main__":
    raise SystemExit(main())
