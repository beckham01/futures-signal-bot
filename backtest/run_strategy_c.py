"""CLI entry point for Strategy C backtests."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone

from backtest.data_fetcher import BybitDataError, CacheDataError, fetch_all_symbols, set_data_source
from backtest.report import generate_report_c, strategy_c_acceptance
from backtest.simulator import simulate_all
from backtest.strategy import load_config
from backtest.strategy_c import compute_indicators_c, evaluate_signals_c


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Strategy C 4h FVG/breaker backtest.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--symbols", nargs="+")
    parser.add_argument("--days", type=int)
    parser.add_argument("--api-base-url")
    parser.add_argument("--cache-dir")
    parser.add_argument("--cache-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    cfg = config["strategy_c"]
    symbols = args.symbols or list(cfg.get("symbols") or config["watchlist"])
    days = args.days or int(config["backtest"]["lookback_days"])
    interval = cfg["timeframe_entry"]
    set_data_source(
        api_base_url=args.api_base_url or config["backtest"].get("bybit_api_base_url"),
        cache_dir=args.cache_dir or config["backtest"].get("cache_dir"),
    )
    try:
        data = fetch_all_symbols(symbols, [interval], days, cache_only=args.cache_only)
    except (BybitDataError, CacheDataError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    all_signals = []
    for symbol in symbols:
        df_4h = compute_indicators_c(data[symbol][interval], cfg)
        data[symbol][interval] = df_4h
        all_signals.extend(evaluate_signals_c(symbol, df_4h, cfg["cooldown_hours"], cfg))

    results = simulate_all(all_signals, data)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    generate_report_c(results, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), symbols, days)
    passed, failures = strategy_c_acceptance(results, days)
    if not passed:
        print(f"Strategy C gate failed: {'; '.join(failures)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
