"""Compare Strategy A, Strategy B, and combined non-conflicting results."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd

from backtest.data_fetcher import BybitDataError, CacheDataError, fetch_all_symbols, set_data_source
from backtest.report import combined_acceptance, generate_comparison_report, strategy_b_acceptance
from backtest.simulator import TradeResult, simulate_all
from backtest.strategy import compute_indicators, evaluate_signals, load_config
from backtest.strategy_b import compute_indicators_b, evaluate_signals_b


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Strategy A and Strategy B backtests.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--symbols", nargs="+")
    parser.add_argument("--days", type=int)
    parser.add_argument("--api-base-url")
    parser.add_argument("--cache-dir")
    parser.add_argument("--cache-only", action="store_true")
    return parser.parse_args()


def _conflicts_with_strategy_a(result_b: TradeResult, results_a: list[TradeResult]) -> TradeResult | None:
    for result_a in results_a:
        if result_a.signal.symbol != result_b.signal.symbol:
            continue
        if abs(result_b.signal.timestamp - result_a.signal.timestamp) <= pd.Timedelta(hours=1):
            return result_a
    return None


def combine_results(results_a: list[TradeResult], results_b: list[TradeResult]) -> tuple[list[TradeResult], list[tuple]]:
    """Combine results, dropping Strategy B conflicts within 1h on the same symbol."""
    combined = list(results_a)
    conflicts = []
    for result_b in results_b:
        conflict = _conflicts_with_strategy_a(result_b, results_a)
        if conflict is not None:
            conflicts.append((conflict.signal.symbol, conflict.signal.timestamp, result_b.signal.timestamp))
            continue
        combined.append(result_b)
    return sorted(combined, key=lambda result: result.signal.timestamp), conflicts


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    symbols = args.symbols or config["watchlist"]
    days = args.days or int(config["backtest"]["lookback_days"])
    strategy_a_cfg = config["strategy"]
    strategy_b_cfg = config["strategy_b"]
    entry_tf = strategy_a_cfg["entry_timeframe"]
    trend_tf = strategy_a_cfg["trend_timeframe"]
    strategy_b_tf = strategy_b_cfg["timeframe"]
    intervals = sorted({entry_tf, trend_tf, strategy_b_tf})
    set_data_source(
        api_base_url=args.api_base_url or config["backtest"].get("bybit_api_base_url"),
        cache_dir=args.cache_dir or config["backtest"].get("cache_dir"),
    )

    try:
        data = fetch_all_symbols(symbols, intervals, days, cache_only=args.cache_only)
    except (BybitDataError, CacheDataError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    signals_a = []
    signals_b = []
    for symbol in symbols:
        df_15m_a, df_1h = compute_indicators(data[symbol][entry_tf], data[symbol][trend_tf], strategy_a_cfg)
        data[symbol][entry_tf] = df_15m_a
        data[symbol][trend_tf] = df_1h
        signals_a.extend(evaluate_signals(symbol, df_15m_a, df_1h, config["bot"]["cooldown_hours"], strategy_a_cfg))

        df_15m_b = compute_indicators_b(data[symbol][strategy_b_tf], strategy_b_cfg)
        data[symbol][strategy_b_tf] = df_15m_b
        data[symbol]["15"] = df_15m_b
        signals_b.extend(evaluate_signals_b(symbol, df_15m_b, strategy_b_cfg["cooldown_hours"], strategy_b_cfg))

    results_a = simulate_all(signals_a, data)
    results_b = simulate_all(signals_b, data)
    passed_b, failures_b = strategy_b_acceptance(results_b, days)
    combined_results, conflicts = combine_results(results_a, results_b)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    generate_comparison_report(
        results_a,
        results_b,
        combined_results,
        conflicts,
        start.strftime("%Y-%m-%d"),
        end.strftime("%Y-%m-%d"),
        days,
    )
    if not passed_b:
        print(f"Strategy B solo failed; combined recommendation blocked: {'; '.join(failures_b)}", file=sys.stderr)
        return 1
    passed_combined, failures_combined = combined_acceptance(results_a, combined_results, days)
    if not passed_combined:
        print(f"Combined gate failed: {'; '.join(failures_combined)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
