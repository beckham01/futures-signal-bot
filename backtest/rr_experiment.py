"""Risk/reward target experiments for existing signals."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path
from typing import Callable

from backtest.data_fetcher import BybitDataError, CacheDataError, fetch_all_symbols, set_data_source
from backtest.simulator import TradeResult, simulate_all
from backtest.strategy import SignalEvent, compute_indicators, evaluate_signals, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest base and extended risk/reward target scenarios.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--symbols", nargs="+")
    parser.add_argument("--days", type=int)
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--api-base-url")
    parser.add_argument("--cache-dir")
    parser.add_argument("--output", default="rr_experiment_results.csv")
    parser.add_argument("--report", default="rr_experiment_report.txt")
    parser.add_argument("--targets", nargs="+", type=float, default=[3.0, 5.0, 7.0, 10.0])
    return parser.parse_args()


def adjusted_signal(signal: SignalEvent, target_rr: float) -> SignalEvent:
    """Return a signal with TP2 moved to a target R multiple."""
    risk = abs(signal.entry - signal.stop_loss)
    if signal.direction == "LONG":
        tp2 = signal.entry + risk * target_rr
    else:
        tp2 = signal.entry - risk * target_rr
    return replace(signal, tp2=tp2, risk_reward=target_rr)


def is_structure_extension_candidate(signal: SignalEvent) -> bool:
    """Conservative extended-target rule based only on information known at signal time."""
    conditions = signal.confidence_conditions
    return (
        signal.confidence >= 85
        and conditions.get("trend_rsi_strong", False)
        and conditions.get("atr_expanding", False)
        and conditions.get("ema55_slope", False)
        and conditions.get("strong_volume", False)
    )


def apply_scenario(
    signals: list[SignalEvent],
    base_rr: float,
    target_rr: float,
    predicate: Callable[[SignalEvent], bool],
) -> list[SignalEvent]:
    """Promote matching signals to target_rr and keep the rest at base_rr."""
    return [adjusted_signal(signal, target_rr if predicate(signal) else base_rr) for signal in signals]


def max_consecutive_losses(results: list[TradeResult]) -> int:
    longest = 0
    current = 0
    for result in sorted(results, key=lambda item: item.signal.timestamp):
        if result.outcome == "STOP_HIT":
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def summarize_results(name: str, target_rr: float, extended_count: int, results: list[TradeResult]) -> dict:
    trades = len(results)
    wins = sum(1 for result in results if result.outcome in {"TP1_ONLY", "TP2_HIT"})
    tp2_hits = sum(1 for result in results if result.outcome == "TP2_HIT")
    tp1_only = sum(1 for result in results if result.outcome == "TP1_ONLY")
    stops = sum(1 for result in results if result.outcome == "STOP_HIT")
    open_trades = sum(1 for result in results if result.outcome == "OPEN")
    total_r = sum(result.pnl_r for result in results)
    return {
        "scenario": name,
        "target_rr": target_rr,
        "extended_signals": extended_count,
        "trades": trades,
        "win_rate": round((wins / trades * 100) if trades else 0.0, 2),
        "tp2_rate": round((tp2_hits / trades * 100) if trades else 0.0, 2),
        "tp2_hits": tp2_hits,
        "tp1_only": tp1_only,
        "stops": stops,
        "open": open_trades,
        "total_r": round(total_r, 3),
        "avg_r": round((total_r / trades) if trades else 0.0, 3),
        "max_consecutive_losses": max_consecutive_losses(results),
    }


def generate_report(rows: list[dict], output_path: str | Path) -> str:
    ranked = sorted(rows, key=lambda row: (row["total_r"], row["avg_r"], row["trades"]), reverse=True)
    lines = [
        "=== RR EXPERIMENT REPORT ===",
        "",
        "Scenarios keep the same entries. They only change TP2 target handling.",
        "Adaptive scenarios keep base 1:3 unless the signal qualifies at entry time.",
        "",
        "--- TOP SCENARIOS ---",
    ]
    for row in ranked[:12]:
        lines.append(
            f"{row['scenario']}: trades={row['trades']} extended={row['extended_signals']} "
            f"win={row['win_rate']:.2f}% tp2={row['tp2_rate']:.2f}% "
            f"avgR={row['avg_r']:.3f} totalR={row['total_r']:.3f} "
            f"maxLoss={row['max_consecutive_losses']}"
        )
    lines.extend(["", "--- ALL SCENARIOS ---"])
    for row in rows:
        lines.append(
            f"{row['scenario']}: target={row['target_rr']:.1f} extended={row['extended_signals']} "
            f"TP2={row['tp2_hits']} TP1={row['tp1_only']} stops={row['stops']} open={row['open']} "
            f"avgR={row['avg_r']:.3f} totalR={row['total_r']:.3f}"
        )
    report = "\n".join(lines)
    Path(output_path).write_text(report, encoding="utf-8")
    return report


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    symbols = args.symbols or config["watchlist"]
    days = args.days or int(config["backtest"]["lookback_days"])
    strategy_cfg = dict(config["strategy"])
    base_rr = float(strategy_cfg["min_risk_reward"])
    entry_tf = strategy_cfg["entry_timeframe"]
    trend_tf = strategy_cfg["trend_timeframe"]
    set_data_source(
        api_base_url=args.api_base_url or config["backtest"].get("bybit_api_base_url"),
        cache_dir=args.cache_dir or config["backtest"].get("cache_dir"),
    )

    try:
        data = fetch_all_symbols(symbols, [entry_tf, trend_tf], days, cache_only=args.cache_only)
    except (BybitDataError, CacheDataError) as exc:
        print(str(exc))
        return 1

    base_signals: list[SignalEvent] = []
    for symbol in symbols:
        df_15m, df_1h = compute_indicators(data[symbol][entry_tf], data[symbol][trend_tf], strategy_cfg)
        data[symbol][entry_tf] = df_15m
        data[symbol][trend_tf] = df_1h
        base_signals.extend(evaluate_signals(symbol, df_15m, df_1h, config["bot"]["cooldown_hours"], strategy_cfg))

    scenarios: list[tuple[str, float, list[SignalEvent]]] = []
    for target_rr in args.targets:
        scenarios.append((f"fixed_{target_rr:.1f}R", target_rr, apply_scenario(base_signals, base_rr, target_rr, lambda _: True)))
        if target_rr > base_rr:
            scenarios.append(
                (
                    f"adaptive_structure_to_{target_rr:.1f}R",
                    target_rr,
                    apply_scenario(base_signals, base_rr, target_rr, is_structure_extension_candidate),
                )
            )
            scenarios.append(
                (
                    f"adaptive_conf95_to_{target_rr:.1f}R",
                    target_rr,
                    apply_scenario(base_signals, base_rr, target_rr, lambda signal: signal.confidence >= 95),
                )
            )

    rows = []
    for name, target_rr, signals in scenarios:
        extended_count = sum(1 for signal in signals if signal.risk_reward > base_rr)
        rows.append(summarize_results(name, target_rr, extended_count, simulate_all(signals, data)))

    output_path = Path(args.output)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    report = generate_report(rows, args.report)
    print(report)
    print(f"\nWrote {args.output} and {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
