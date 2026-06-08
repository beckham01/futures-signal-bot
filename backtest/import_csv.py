"""Import external candle CSV files into the backtest cache."""

from __future__ import annotations

import argparse
from pathlib import Path

from backtest.data_fetcher import import_klines_csv, set_data_source
from backtest.strategy import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import OHLCV candle CSV into the local backtest cache.")
    parser.add_argument("--symbol", required=True, help="Symbol name, e.g. BTCUSDT")
    parser.add_argument("--interval", required=True, help='Bybit interval notation, e.g. "15" or "60"')
    parser.add_argument("--input", required=True, help="Input CSV path")
    parser.add_argument("--cache-dir", help="Override candle cache directory")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config()
    set_data_source(cache_dir=args.cache_dir or config["backtest"].get("cache_dir"))
    imported = import_klines_csv(Path(args.input), args.symbol.upper(), args.interval)
    start = imported["timestamp"].min()
    end = imported["timestamp"].max()
    print(f"Imported {len(imported)} candles for {args.symbol.upper()} {args.interval}: {start} to {end}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
