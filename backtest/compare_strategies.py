"""Compare Strategy A, Strategy B, Strategy C, and priority-combined results."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd

from backtest.data_fetcher import BybitDataError, CacheDataError, fetch_all_symbols, set_data_source
from backtest.report import (
    combined_acceptance,
    generate_comparison_report,
    strategy_b_acceptance,
    strategy_c_acceptance,
)
from backtest.simulator import TradeResult, simulate_all
from backtest.strategy import compute_indicators, evaluate_signals, load_config
from backtest.strategy_b import compute_indicators_b, evaluate_signals_b
from backtest.strategy_c import compute_indicators_c, evaluate_signals_c

CONFLICT_WINDOW = pd.Timedelta(hours=1)
PRIORITY = {
    "strategy_c_fvg_breaker_4h": 0,
    "strategy_a_trend_pullback": 1,
    "strategy_b_fvg_breaker_15m": 2,
    "strategy_b_daily_momentum": 3,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Strategy A/B/C backtests.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--symbols", nargs="+")
    parser.add_argument("--days", type=int)
    parser.add_argument("--api-base-url")
    parser.add_argument("--cache-dir")
    parser.add_argument("--cache-only", action="store_true")
    return parser.parse_args()


def _conflicts(candidate: TradeResult, accepted: list[TradeResult]) -> TradeResult | None:
    for result in accepted:
        if result.signal.symbol != candidate.signal.symbol:
            continue
        if abs(candidate.signal.timestamp - result.signal.timestamp) <= CONFLICT_WINDOW:
            return result
    return None


def combine_results(
    results_a: list[TradeResult],
    results_b: list[TradeResult],
    results_c: list[TradeResult] | None = None,
) -> tuple[list[TradeResult], list[tuple]]:
    """Combine results using priority C > A > B on same-symbol conflicts."""
    ordered = sorted(
        [*(results_c or []), *results_a, *results_b],
        key=lambda result: (
            result.signal.timestamp,
            PRIORITY.get(result.signal.strategy_name, 99),
        ),
    )
    accepted: list[TradeResult] = []
    conflicts = []
    for result in ordered:
        conflict = _conflicts(result, accepted)
        if conflict is not None and PRIORITY.get(conflict.signal.strategy_name, 99) <= PRIORITY.get(
            result.signal.strategy_name, 99
        ):
            conflicts.append(
                (
                    result.signal.symbol,
                    conflict.signal.strategy_name,
                    conflict.signal.timestamp,
                    result.signal.strategy_name,
                    result.signal.timestamp,
                )
            )
            continue
        if conflict is not None:
            accepted.remove(conflict)
            conflicts.append(
                (
                    result.signal.symbol,
                    result.signal.strategy_name,
                    result.signal.timestamp,
                    conflict.signal.strategy_name,
                    conflict.signal.timestamp,
                )
            )
        accepted.append(result)
    return sorted(accepted, key=lambda result: result.signal.timestamp), conflicts


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    days = args.days or int(config["backtest"]["lookback_days"])
    strategy_a_cfg = config["strategy"]
    strategy_b_cfg = config["strategy_b"]
    strategy_c_cfg = config["strategy_c"]
    symbols_a = args.symbols or config["watchlist"]
    symbols_b = args.symbols or list(strategy_b_cfg.get("symbols") or config["watchlist"])
    symbols_c = args.symbols or list(strategy_c_cfg.get("symbols") or config["watchlist"])
    set_data_source(
        api_base_url=args.api_base_url or config["backtest"].get("bybit_api_base_url"),
        cache_dir=args.cache_dir or config["backtest"].get("cache_dir"),
    )
    try:
        data: dict[str, dict[str, pd.DataFrame]] = {}
        for symbol in sorted(set(symbols_a) | set(symbols_b) | set(symbols_c)):
            intervals = set()
            if symbol in symbols_a:
                intervals.update([strategy_a_cfg["entry_timeframe"], strategy_a_cfg["trend_timeframe"]])
            if symbol in symbols_b:
                intervals.update([strategy_b_cfg["timeframe_entry"], strategy_b_cfg["timeframe_trend"]])
            if symbol in symbols_c:
                intervals.add(strategy_c_cfg["timeframe_entry"])
            data[symbol] = fetch_all_symbols([symbol], sorted(intervals), days, cache_only=args.cache_only)[symbol]
    except (BybitDataError, CacheDataError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    signals_a = []
    signals_b = []
    signals_c = []
    for symbol in symbols_a:
        df_15m_a, df_1h_a = compute_indicators(
            data[symbol][strategy_a_cfg["entry_timeframe"]],
            data[symbol][strategy_a_cfg["trend_timeframe"]],
            strategy_a_cfg,
        )
        data[symbol][strategy_a_cfg["entry_timeframe"]] = df_15m_a
        data[symbol][strategy_a_cfg["trend_timeframe"]] = df_1h_a
        signals_a.extend(evaluate_signals(symbol, df_15m_a, df_1h_a, config["bot"]["cooldown_hours"], strategy_a_cfg))

    for symbol in symbols_b:
        df_15m_b, df_1h_b = compute_indicators_b(
            data[symbol][strategy_b_cfg["timeframe_entry"]],
            data[symbol][strategy_b_cfg["timeframe_trend"]],
            strategy_b_cfg,
        )
        data[symbol][strategy_b_cfg["timeframe_entry"]] = df_15m_b
        data[symbol][strategy_b_cfg["timeframe_trend"]] = df_1h_b
        data[symbol]["15"] = df_15m_b
        signals_b.extend(evaluate_signals_b(symbol, df_15m_b, df_1h_b, strategy_b_cfg["cooldown_hours"], strategy_b_cfg))

    for symbol in symbols_c:
        df_4h = compute_indicators_c(data[symbol][strategy_c_cfg["timeframe_entry"]], strategy_c_cfg)
        data[symbol][strategy_c_cfg["timeframe_entry"]] = df_4h
        signals_c.extend(evaluate_signals_c(symbol, df_4h, strategy_c_cfg["cooldown_hours"], strategy_c_cfg))

    results_a = simulate_all(signals_a, data)
    results_b = simulate_all(signals_b, data)
    results_c = simulate_all(signals_c, data)
    combined_results, conflicts = combine_results(results_a, results_b, results_c)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    generate_comparison_report(
        results_a,
        results_b,
        results_c,
        combined_results,
        conflicts,
        start.strftime("%Y-%m-%d"),
        end.strftime("%Y-%m-%d"),
        days,
    )
    passed_b, failures_b = strategy_b_acceptance(results_b, days)
    if not passed_b:
        print(f"Strategy B solo failed; combined recommendation blocked: {'; '.join(failures_b)}", file=sys.stderr)
        return 1
    passed_c, failures_c = strategy_c_acceptance(results_c, days)
    if not passed_c:
        print(f"Strategy C solo failed; combined recommendation blocked: {'; '.join(failures_c)}", file=sys.stderr)
        return 1
    passed_combined, failures_combined = combined_acceptance(results_a, combined_results, days)
    if not passed_combined:
        print(f"Combined gate failed: {'; '.join(failures_combined)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
