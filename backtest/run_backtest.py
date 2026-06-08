"""CLI entry point for the Phase 1 backtest engine."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone

import requests

from backtest.data_fetcher import (
    BybitDataError,
    CacheDataError,
    check_data_source,
    fetch_all_symbols,
    set_data_source,
)
from backtest.report import generate_report
from backtest.simulator import simulate_all
from backtest.strategy import compute_indicators, evaluate_signals, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the futures signal strategy backtest.")
    parser.add_argument("--config", default="config.yaml", help="Config YAML path")
    parser.add_argument("--symbols", nargs="+", help="Symbols to backtest, e.g. BTCUSDT ETHUSDT")
    parser.add_argument("--days", type=int, help="Lookback days")
    parser.add_argument("--api-base-url", help="Override Bybit API base URL, e.g. https://api.bybit.com")
    parser.add_argument("--cache-dir", help="Override candle cache directory")
    parser.add_argument("--check-data-source", action="store_true", help="Test Bybit connectivity and exit")
    parser.add_argument("--cache-only", action="store_true", help="Use local CSV cache only; do not call Bybit")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    symbols = args.symbols or config["watchlist"]
    days = args.days or int(config["backtest"]["lookback_days"])
    entry_tf = config["strategy"]["entry_timeframe"]
    trend_tf = config["strategy"]["trend_timeframe"]
    set_data_source(
        api_base_url=args.api_base_url or config["backtest"].get("bybit_api_base_url"),
        cache_dir=args.cache_dir or config["backtest"].get("cache_dir"),
    )

    if args.check_data_source:
        try:
            ok = check_data_source(symbols[0], entry_tf)
        except requests.RequestException as exc:
            print(f"Bybit data source check failed: {exc}", file=sys.stderr)
            print(
                "Workaround: run from another network/VPS, or add CSV cache files under the configured cache dir.",
                file=sys.stderr,
            )
            return 1
        print("Bybit data source check passed." if ok else "Bybit responded but returned no kline rows.")
        return 0 if ok else 1

    try:
        data = fetch_all_symbols(symbols, [entry_tf, trend_tf], days, cache_only=args.cache_only)
    except (BybitDataError, CacheDataError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    all_signals = []
    for symbol in symbols:
        strategy_cfg = config["strategy"]
        df_15m, df_1h = compute_indicators(data[symbol][entry_tf], data[symbol][trend_tf], strategy_cfg)
        data[symbol][entry_tf] = df_15m
        data[symbol][trend_tf] = df_1h
        all_signals.extend(
            evaluate_signals(
                symbol,
                df_15m,
                df_1h,
                config["bot"]["cooldown_hours"],
                strategy_cfg,
            )
        )

    results = simulate_all(all_signals, data)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    generate_report(
        results,
        symbols,
        start.strftime("%Y-%m-%d"),
        end.strftime("%Y-%m-%d"),
    )

    total_r = sum(result.pnl_r for result in results)
    wins = sum(1 for result in results if result.outcome in {"TP1_ONLY", "TP2_HIT"})
    win_rate = (wins / len(results) * 100) if results else 0.0
    if total_r < 0 or win_rate < 40 or len(results) < 30:
        print(
            f"Backtest gate failed: total_r={total_r:.2f}, "
            f"win_rate={win_rate:.1f}%, trades={len(results)}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
