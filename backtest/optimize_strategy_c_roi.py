"""ROI-first optimizer for Strategy C."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from backtest.data_fetcher import BybitDataError, CacheDataError, fetch_all_symbols, set_data_source
from backtest.indicators import ema
from backtest.resample import resample_ohlcv
from backtest.roi import RoiMetrics, generate_roi_report, roi_metrics, simulate_roi_all
from backtest.strategy import SignalEvent, load_config
from backtest.strategy_c import compute_indicators_c, evaluate_signals_c


DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "BNBUSDT"]


@dataclass(frozen=True)
class RoiCandidate:
    params: dict[str, Any]
    metrics: RoiMetrics
    accepted: bool
    score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize Strategy C for 5x ROI targets.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--symbols", nargs="+")
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--api-base-url")
    parser.add_argument("--cache-dir")
    parser.add_argument("--output", default="strategy_c_roi_optimization.csv")
    parser.add_argument("--report", default="strategy_c_roi_report.txt")
    parser.add_argument("--max-runs", type=int)
    return parser.parse_args()


def candidate_grid() -> list[dict[str, Any]]:
    candidates = []
    for min_confidence in [80, 90, 100]:
        for cooldown_hours in [24, 48]:
            for fvg_min_gap_atr in [1.0, 1.5]:
                candidates.append(
                    {
                        "timeframe_entry": "240",
                        "min_confidence": min_confidence,
                        "cooldown_hours": cooldown_hours,
                        "fvg_min_gap_atr": fvg_min_gap_atr,
                        "candle_body_min_pct": 0.60,
                        "tp1_risk_multiplier": 1.0,
                        "tp2_risk_multiplier": 2.0,
                        "tp1_position_pct": 0.80,
                        "tp2_position_pct": 0.20,
                    }
                )
    return candidates


def max_bars(hold_days: int) -> int:
    return int((hold_days * 24 * 60) / 240)


def daily_bias_frame(df_4h: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    daily = resample_ohlcv(df_4h, 1440)
    daily["ema21"] = ema(daily["close"], cfg["ema_fast"])
    daily["ema55"] = ema(daily["close"], cfg["ema_slow"])
    daily["available_at"] = pd.to_datetime(daily["timestamp"], utc=True) + pd.Timedelta(days=1)
    return daily


def daily_bias_at(daily: pd.DataFrame, timestamp: pd.Timestamp) -> str | None:
    available = daily[daily["available_at"] <= timestamp].dropna()
    if available.empty:
        return None
    row = available.iloc[-1]
    if float(row["ema21"]) > float(row["ema55"]):
        return "LONG"
    if float(row["ema21"]) < float(row["ema55"]):
        return "SHORT"
    return None


def filter_daily_aligned(signals: list[SignalEvent], daily: pd.DataFrame) -> list[SignalEvent]:
    aligned = []
    for signal in signals:
        if daily_bias_at(daily, signal.timestamp) == signal.direction:
            aligned.append(signal)
    return aligned


def evaluate_candidate(raw: dict[str, dict], symbols: list[str], base_cfg: dict[str, Any], params: dict[str, Any]):
    cfg = dict(base_cfg)
    cfg.update(params)
    eval_data: dict[str, dict] = {}
    signals = []
    for symbol in symbols:
        df_4h = compute_indicators_c(raw[symbol]["240"], cfg)
        daily = daily_bias_frame(raw[symbol]["240"], cfg)
        symbol_signals = filter_daily_aligned(evaluate_signals_c(symbol, df_4h, cfg["cooldown_hours"], cfg), daily)
        for signal in symbol_signals:
            signal.execution_timeframe = "240"
            signal.tp1_position_pct = cfg["tp1_position_pct"]
            signal.tp2_position_pct = cfg["tp2_position_pct"]
        signals.extend(symbol_signals)
        eval_data[symbol] = {"240": df_4h}
    return simulate_roi_all(signals, eval_data, leverage=5, max_bars_by_timeframe={"240": max_bars(10)})


def accepted(metrics: RoiMetrics) -> bool:
    excellent = metrics.roi100_hit_rate >= 65 and metrics.avg_expectancy_r >= 0.4
    return (
        (metrics.roi100_hit_rate >= 70 or excellent)
        and metrics.trades >= 10
        and metrics.total_expectancy_r > 0
        and metrics.max_consecutive_roi100_failures <= 4
    )


def score(metrics: RoiMetrics) -> float:
    if metrics.trades == 0:
        return -9999
    return (
        metrics.roi100_hit_rate * 4
        + metrics.roi200_hit_rate
        + metrics.avg_expectancy_r * 60
        - metrics.max_consecutive_roi100_failures * 8
        + min(metrics.trades, 40) * 0.3
    )


def write_row(path: Path, result: RoiCandidate) -> None:
    row = {"accepted": result.accepted, "score": round(result.score, 4)}
    row.update(result.params)
    row.update({f"metrics_{key}": value for key, value in asdict(result.metrics).items()})
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    symbols = args.symbols or DEFAULT_SYMBOLS
    set_data_source(args.api_base_url or config["backtest"].get("bybit_api_base_url"), args.cache_dir or config["backtest"].get("cache_dir"))
    try:
        raw = fetch_all_symbols(symbols, ["240"], args.days, cache_only=args.cache_only)
    except (BybitDataError, CacheDataError) as exc:
        print(exc)
        return 1
    output = Path(args.output)
    if output.exists():
        output.unlink()
    results = []
    best_roi_results = []
    grid = candidate_grid()
    if args.max_runs:
        grid = grid[: args.max_runs]
    for params in grid:
        roi_results = evaluate_candidate(raw, symbols, config["strategy_c"], params)
        metrics = roi_metrics(roi_results)
        candidate = RoiCandidate(params, metrics, accepted(metrics), score(metrics))
        results.append(candidate)
        write_row(output, candidate)
        if len(results) == 1 or candidate.score >= max(results[:-1], key=lambda item: item.score).score:
            best_roi_results = roi_results
        print(
            f"{params} | trades={metrics.trades} ROI100={metrics.roi100_hit_rate:.1f}% "
            f"ROI200={metrics.roi200_hit_rate:.1f}% exp={metrics.total_expectancy_r:.1f} accepted={candidate.accepted}",
            flush=True,
        )
    ranked = sorted(results, key=lambda item: item.score, reverse=True)
    generate_roi_report(best_roi_results, "Strategy C ROI best candidate", 5, args.report)
    winners = [item for item in ranked if item.accepted]
    if winners:
        config["strategy_c"].update(winners[0].params)
        Path("strategy_c_candidate_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        return 0
    print("No Strategy C ROI candidate passed acceptance gates.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
