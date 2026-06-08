"""Walk-forward strategy optimizer and confidence diagnostics."""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass, asdict
from itertools import product
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from backtest.data_fetcher import BybitDataError, fetch_all_symbols, set_data_source
from backtest.simulator import TradeResult, simulate_trade
from backtest.strategy import SignalEvent, compute_confidence, compute_indicators, load_config

WEAK_SYMBOLS = {"ADAUSDT", "DOGEUSDT", "LINKUSDT", "AVAXUSDT"}
CORE_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "TONUSDT"]
BASELINE_WIN_RATE = 49.0
BASELINE_MAX_LOSSES = 9


@dataclass(frozen=True)
class Metrics:
    trades: int
    win_rate: float
    total_r: float
    avg_r: float
    max_consecutive_losses: int
    top_symbol_trade_share: float


@dataclass(frozen=True)
class ExperimentResult:
    name: str
    symbols: str
    params: dict[str, Any]
    train: Metrics
    validation: Metrics
    accepted: bool
    score: float


@dataclass
class PreparedSymbolData:
    symbol: str
    merged: pd.DataFrame
    candles_15m: pd.DataFrame
    candidates: list["SignalCandidate"]
    candle_timestamps: pd.Series


@dataclass(frozen=True)
class SignalCandidate:
    symbol: str
    direction: str
    timestamp: pd.Timestamp
    entry: float
    atr_15m: float
    distance_to_ema21: float
    prev_rsi_15m: float
    rsi_15m: float
    volume_ratio: float
    price_confirmed: bool
    rsi_1h: float
    ema55_slope: float
    candle_close_quality: bool
    atr_expanding: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize the backtest strategy with walk-forward validation.")
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--symbols", nargs="+", help="Override base watchlist")
    parser.add_argument("--cache-dir", help="Override candle cache directory")
    parser.add_argument("--api-base-url", help="Override Bybit API base URL")
    parser.add_argument("--max-runs", type=int, help="Stop after N experiment combinations")
    parser.add_argument(
        "--symbol-set",
        choices=["all_symbols", "quarantine_weak", "core_liquid"],
        help="Run only one predefined symbol universe",
    )
    parser.add_argument("--output", default="optimize_results.csv")
    return parser.parse_args()


def build_symbol_sets(watchlist: list[str]) -> dict[str, list[str]]:
    quarantine = [symbol for symbol in watchlist if symbol not in WEAK_SYMBOLS]
    core = [symbol for symbol in CORE_SYMBOLS if symbol in watchlist]
    return {
        "all_symbols": watchlist,
        "quarantine_weak": quarantine,
        "core_liquid": core,
    }


def parameter_grid() -> list[dict[str, Any]]:
    keys = [
        "volume_spike_threshold",
        "pullback_atr_tolerance",
        "cooldown_hours",
        "atr_min_pct",
        "atr_max_pct",
        "min_confidence",
    ]
    values = [
        [1.3, 1.5, 1.8],
        [0.35, 0.5, 0.65],
        [4, 8, 12],
        [0.003, 0.005],
        [0.03, 0.04],
        [55, 70, 85],
    ]
    return [dict(zip(keys, combo)) for combo in product(*values)]


def walk_forward_bounds(data: dict[str, dict[str, pd.DataFrame]], symbols: list[str], interval: str, train_ratio: float = 0.7) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    starts = []
    ends = []
    for symbol in symbols:
        timestamps = pd.to_datetime(data[symbol][interval]["timestamp"], utc=True)
        starts.append(timestamps.min())
        ends.append(timestamps.max())
    start = max(starts)
    end = min(ends)
    split = start + (end - start) * train_ratio
    return start, split, end


def prepare_merged_frame(df_15m: pd.DataFrame, df_1h: pd.DataFrame) -> pd.DataFrame:
    """Pre-merge entry candles with the most recent closed 1h candle."""
    entry_frame = df_15m.sort_values("timestamp").reset_index(drop=True).copy()
    trend_frame = df_1h.sort_values("timestamp").reset_index(drop=True).copy()
    trend_frame["available_at"] = pd.to_datetime(trend_frame["timestamp"], utc=True) + pd.Timedelta(hours=1)
    return pd.merge_asof(
        entry_frame,
        trend_frame,
        left_on="timestamp",
        right_on="available_at",
        suffixes=("_15m", "_1h"),
        direction="backward",
    )


def prepare_optimizer_data(
    data: dict[str, dict[str, pd.DataFrame]],
    symbols: list[str],
    entry_tf: str,
    trend_tf: str,
) -> dict[str, PreparedSymbolData]:
    """Build reusable optimizer frames for every symbol."""
    prepared: dict[str, PreparedSymbolData] = {}
    for symbol in symbols:
        candles = data[symbol][entry_tf].sort_values("timestamp").reset_index(drop=True).copy()
        merged = prepare_merged_frame(candles, data[symbol][trend_tf])
        prepared[symbol] = PreparedSymbolData(
            symbol=symbol,
            merged=merged,
            candles_15m=candles,
            candidates=build_signal_candidates(symbol, merged),
            candle_timestamps=pd.to_datetime(candles["timestamp"], utc=True),
        )
    return prepared


def window_data(
    data: dict[str, dict[str, pd.DataFrame]],
    symbols: list[str],
    intervals: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, dict[str, pd.DataFrame]]:
    windowed: dict[str, dict[str, pd.DataFrame]] = {}
    for symbol in symbols:
        windowed[symbol] = {}
        for interval in intervals:
            df = data[symbol][interval]
            timestamps = pd.to_datetime(df["timestamp"], utc=True)
            windowed[symbol][interval] = df[(timestamps >= start) & (timestamps <= end)].reset_index(drop=True).copy()
    return windowed


def _trend_direction_from_row(row: pd.Series) -> str | None:
    close = float(row["close_1h"])
    ema21 = float(row["ema21_1h"])
    ema55 = float(row["ema55_1h"])
    rsi14 = float(row["rsi14_1h"])
    if close > ema21 > ema55 and 45 <= rsi14 <= 75:
        return "LONG"
    if close < ema21 < ema55 and 25 <= rsi14 <= 55:
        return "SHORT"
    return None


def _risk_levels(direction: str, entry: float, atr_15m: float, cfg: dict[str, Any]) -> tuple[float, float, float, float]:
    sl_distance = cfg["sl_atr_multiplier"] * atr_15m
    tp1_distance = cfg["tp1_atr_multiplier"] * atr_15m
    tp2_distance = cfg["tp2_atr_multiplier"] * atr_15m
    if direction == "LONG":
        stop_loss = entry - sl_distance
        tp1 = entry + tp1_distance
        tp2 = entry + tp2_distance
    else:
        stop_loss = entry + sl_distance
        tp1 = entry - tp1_distance
        tp2 = entry - tp2_distance
    return stop_loss, tp1, tp2, abs(tp2 - entry) / abs(entry - stop_loss)


def _quality_close(row: pd.Series, direction: str) -> bool:
    candle_range = float(row["high_15m"]) - float(row["low_15m"])
    if candle_range <= 0:
        return False
    position = (float(row["close_15m"]) - float(row["low_15m"])) / candle_range
    return position >= 0.70 if direction == "LONG" else position <= 0.30


def _reasons(direction: str, volume_ratio: float) -> list[str]:
    arrow = "above" if direction == "LONG" else "below"
    slope = "upward" if direction == "LONG" else "downward"
    return [
        f"EMA21 {arrow} EMA55 on 1h trend filter",
        "15m RSI recovered through trigger threshold",
        f"Volume spike {volume_ratio:.2f}x average",
        f"EMA55 (1h) sloping {slope}",
    ]


def build_signal_candidates(symbol: str, merged: pd.DataFrame) -> list[SignalCandidate]:
    """Build parameter-light signal candidates once for optimizer reuse."""
    candidates: list[SignalCandidate] = []
    required = ["ema21_15m", "rsi14_15m", "atr14_15m", "volume_sma20", "ema21_1h", "ema55_1h", "rsi14_1h"]
    for index in range(1, len(merged)):
        row = merged.iloc[index]
        prev = merged.iloc[index - 1]
        if pd.isna(row.get("timestamp_1h")):
            continue
        if any(pd.isna(row.get(name)) for name in required) or pd.isna(prev.get("rsi14_15m")):
            continue
        direction = _trend_direction_from_row(row)
        if direction is None:
            continue
        entry = float(row["close_15m"])
        ema21_15m = float(row["ema21_15m"])
        candidates.append(
            SignalCandidate(
                symbol=symbol,
                direction=direction,
                timestamp=pd.Timestamp(row["timestamp_15m"]),
                entry=entry,
                atr_15m=float(row["atr14_15m"]),
                distance_to_ema21=abs(entry - ema21_15m),
                prev_rsi_15m=float(prev["rsi14_15m"]),
                rsi_15m=float(row["rsi14_15m"]),
                volume_ratio=float(row["volume_15m"]) / float(row["volume_sma20"]),
                price_confirmed=entry > ema21_15m if direction == "LONG" else entry < ema21_15m,
                rsi_1h=float(row["rsi14_1h"]),
                ema55_slope=float(row["ema55_slope"]) if not pd.isna(row.get("ema55_slope")) else 0.0,
                candle_close_quality=_quality_close(row, direction),
                atr_expanding=bool(row.get("atr_expanding", False)),
            )
        )
    return candidates


def evaluate_prepared_signals(
    symbol: str,
    merged: pd.DataFrame,
    strategy_cfg: dict[str, Any],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> list[SignalEvent]:
    """Evaluate signals against a pre-merged optimizer frame."""
    timestamps = pd.to_datetime(merged["timestamp_15m"], utc=True)
    frame = merged[(timestamps >= start) & (timestamps <= end)]
    signals: list[SignalEvent] = []
    last_signal_by_direction: dict[str, pd.Timestamp] = {}
    last_opposing_signal: dict[str, pd.Timestamp] = {}
    cooldown = pd.Timedelta(hours=int(strategy_cfg["cooldown_hours"]))
    required = ["ema21_15m", "rsi14_15m", "atr14_15m", "volume_sma20", "ema21_1h", "ema55_1h", "rsi14_1h"]

    for index in frame.index:
        if index == 0:
            continue
        row = merged.loc[index]
        prev = merged.loc[index - 1]
        if pd.isna(row.get("timestamp_1h")):
            continue
        if any(pd.isna(row.get(name)) for name in required) or pd.isna(prev.get("rsi14_15m")):
            continue

        direction = _trend_direction_from_row(row)
        if direction is None:
            continue

        timestamp = pd.Timestamp(row["timestamp_15m"])
        last_same_direction = last_signal_by_direction.get(direction)
        if last_same_direction is not None and timestamp - last_same_direction < cooldown:
            continue

        entry = float(row["close_15m"])
        atr_15m = float(row["atr14_15m"])
        ema21_15m = float(row["ema21_15m"])
        volume_ratio = float(row["volume_15m"]) / float(row["volume_sma20"])
        pullback = abs(entry - ema21_15m) <= strategy_cfg["pullback_atr_tolerance"] * atr_15m
        atr_pct = atr_15m / entry if entry else 0

        if direction == "LONG":
            crossed = float(prev["rsi14_15m"]) < strategy_cfg["rsi_long_threshold"] < float(row["rsi14_15m"])
            price_confirmed = entry > ema21_15m
        else:
            crossed = float(prev["rsi14_15m"]) > strategy_cfg["rsi_short_threshold"] > float(row["rsi14_15m"])
            price_confirmed = entry < ema21_15m

        if not (pullback and crossed and volume_ratio > strategy_cfg["volume_spike_threshold"] and price_confirmed):
            continue

        stop_loss, tp1, tp2, risk_reward = _risk_levels(direction, entry, atr_15m, strategy_cfg)
        if risk_reward < strategy_cfg["min_risk_reward"]:
            continue
        if atr_pct > strategy_cfg["atr_max_pct"] or atr_pct < strategy_cfg["atr_min_pct"]:
            continue

        opposing_direction = "SHORT" if direction == "LONG" else "LONG"
        last_opposing = last_opposing_signal.get(opposing_direction)
        no_opposing = last_opposing is None or timestamp - last_opposing >= cooldown
        slope = float(row["ema55_slope"]) if not pd.isna(row.get("ema55_slope")) else 0.0
        confidence_conditions = {
            "trend_rsi_strong": 50 <= float(row["rsi14_1h"]) <= 65 if direction == "LONG" else 35 <= float(row["rsi14_1h"]) <= 50,
            "rsi_clean_cross": abs(float(row["rsi14_15m"]) - float(prev["rsi14_15m"])) >= 2.0,
            "strong_volume": volume_ratio > strategy_cfg["volume_strong_threshold"],
            "ema55_slope": slope > 0 if direction == "LONG" else slope < 0,
            "candle_close_quality": _quality_close(row, direction),
            "atr_expanding": bool(row.get("atr_expanding", False)),
            "no_opposing_signal": no_opposing,
        }
        confidence, label = compute_confidence(confidence_conditions)
        if confidence < strategy_cfg["min_confidence"]:
            continue

        signals.append(
            SignalEvent(
                symbol=symbol,
                direction=direction,
                timestamp=timestamp,
                entry=entry,
                stop_loss=stop_loss,
                tp1=tp1,
                tp2=tp2,
                risk_reward=risk_reward,
                confidence=confidence,
                confidence_label=label,
                atr_15m=atr_15m,
                reasons=_reasons(direction, volume_ratio),
                confidence_conditions=confidence_conditions,
            )
        )
        last_signal_by_direction[direction] = timestamp
        last_opposing_signal[direction] = timestamp

    return signals


def evaluate_signal_candidates(
    candidates: list[SignalCandidate],
    strategy_cfg: dict[str, Any],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> list[SignalEvent]:
    """Evaluate cached candidates for one parameter set."""
    signals: list[SignalEvent] = []
    last_signal_by_direction: dict[str, pd.Timestamp] = {}
    last_opposing_signal: dict[str, pd.Timestamp] = {}
    cooldown = pd.Timedelta(hours=int(strategy_cfg["cooldown_hours"]))

    for candidate in candidates:
        if candidate.timestamp < start or candidate.timestamp > end:
            continue
        last_same_direction = last_signal_by_direction.get(candidate.direction)
        if last_same_direction is not None and candidate.timestamp - last_same_direction < cooldown:
            continue

        pullback = candidate.distance_to_ema21 <= strategy_cfg["pullback_atr_tolerance"] * candidate.atr_15m
        atr_pct = candidate.atr_15m / candidate.entry if candidate.entry else 0
        if candidate.direction == "LONG":
            crossed = candidate.prev_rsi_15m < strategy_cfg["rsi_long_threshold"] < candidate.rsi_15m
        else:
            crossed = candidate.prev_rsi_15m > strategy_cfg["rsi_short_threshold"] > candidate.rsi_15m
        if not (
            pullback
            and crossed
            and candidate.volume_ratio > strategy_cfg["volume_spike_threshold"]
            and candidate.price_confirmed
        ):
            continue

        stop_loss, tp1, tp2, risk_reward = _risk_levels(
            candidate.direction,
            candidate.entry,
            candidate.atr_15m,
            strategy_cfg,
        )
        if risk_reward < strategy_cfg["min_risk_reward"]:
            continue
        if atr_pct > strategy_cfg["atr_max_pct"] or atr_pct < strategy_cfg["atr_min_pct"]:
            continue

        opposing_direction = "SHORT" if candidate.direction == "LONG" else "LONG"
        last_opposing = last_opposing_signal.get(opposing_direction)
        no_opposing = last_opposing is None or candidate.timestamp - last_opposing >= cooldown
        confidence_conditions = {
            "trend_rsi_strong": 50 <= candidate.rsi_1h <= 65 if candidate.direction == "LONG" else 35 <= candidate.rsi_1h <= 50,
            "rsi_clean_cross": abs(candidate.rsi_15m - candidate.prev_rsi_15m) >= 2.0,
            "strong_volume": candidate.volume_ratio > strategy_cfg["volume_strong_threshold"],
            "ema55_slope": candidate.ema55_slope > 0 if candidate.direction == "LONG" else candidate.ema55_slope < 0,
            "candle_close_quality": candidate.candle_close_quality,
            "atr_expanding": candidate.atr_expanding,
            "no_opposing_signal": no_opposing,
        }
        confidence, label = compute_confidence(confidence_conditions)
        if confidence < strategy_cfg["min_confidence"]:
            continue

        signals.append(
            SignalEvent(
                symbol=candidate.symbol,
                direction=candidate.direction,
                timestamp=candidate.timestamp,
                entry=candidate.entry,
                stop_loss=stop_loss,
                tp1=tp1,
                tp2=tp2,
                risk_reward=risk_reward,
                confidence=confidence,
                confidence_label=label,
                atr_15m=candidate.atr_15m,
                reasons=_reasons(candidate.direction, candidate.volume_ratio),
                confidence_conditions=confidence_conditions,
            )
        )
        last_signal_by_direction[candidate.direction] = candidate.timestamp
        last_opposing_signal[candidate.direction] = candidate.timestamp

    return signals


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


def metrics_from_results(results: list[TradeResult]) -> Metrics:
    trades = len(results)
    wins = sum(1 for result in results if result.outcome in {"TP1_ONLY", "TP2_HIT"})
    total_r = sum(result.pnl_r for result in results)
    by_symbol: dict[str, int] = {}
    for result in results:
        by_symbol[result.signal.symbol] = by_symbol.get(result.signal.symbol, 0) + 1
    top_share = max(by_symbol.values()) / trades if trades and by_symbol else 0.0
    return Metrics(
        trades=trades,
        win_rate=(wins / trades * 100) if trades else 0.0,
        total_r=total_r,
        avg_r=(total_r / trades) if trades else 0.0,
        max_consecutive_losses=max_consecutive_losses(results),
        top_symbol_trade_share=top_share,
    )


def simulate_all_fast(
    signals: list[SignalEvent],
    prepared: dict[str, PreparedSymbolData],
) -> list[TradeResult]:
    """Simulate trades using prepared candles and searchsorted timestamp lookup."""
    results: list[TradeResult] = []
    for signal in signals:
        symbol_data = prepared[signal.symbol]
        candles = symbol_data.candles_15m
        start_index = symbol_data.candle_timestamps.searchsorted(signal.timestamp, side="right")
        future = candles.iloc[start_index : start_index + 96]
        results.append(simulate_trade(signal, future))
    return results


def run_prepared_window(
    prepared: dict[str, PreparedSymbolData],
    symbols: list[str],
    strategy_cfg: dict[str, Any],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> list[TradeResult]:
    signals = []
    for symbol in symbols:
        signals.extend(evaluate_signal_candidates(prepared[symbol].candidates, strategy_cfg, start, end))
    return simulate_all_fast(signals, prepared)


def candidate_accepted(metrics: Metrics) -> bool:
    return (
        metrics.win_rate > BASELINE_WIN_RATE
        and metrics.total_r > 0
        and metrics.trades >= 30
        and metrics.avg_r > 0
        and metrics.max_consecutive_losses <= BASELINE_MAX_LOSSES
        and metrics.top_symbol_trade_share <= 0.60
    )


def score_candidate(train: Metrics, validation: Metrics) -> float:
    if validation.trades == 0:
        return -9999.0
    return (
        validation.win_rate
        + validation.avg_r * 20
        + min(validation.trades, 100) * 0.05
        - validation.max_consecutive_losses * 1.5
        - max(validation.top_symbol_trade_share - 0.45, 0) * 20
        + train.avg_r * 5
    )


def flatten_result(result: ExperimentResult) -> dict[str, Any]:
    row = {
        "name": result.name,
        "symbols": result.symbols,
        "accepted": result.accepted,
        "score": round(result.score, 4),
    }
    for key, value in result.params.items():
        row[key] = value
    for prefix, metrics in [("train", result.train), ("validation", result.validation)]:
        for key, value in asdict(metrics).items():
            row[f"{prefix}_{key}"] = round(value, 4) if isinstance(value, float) else value
    return row


def write_results(results: list[ExperimentResult], output_path: str | Path) -> None:
    rows = [flatten_result(result) for result in results]
    if not rows:
        return
    path = Path(output_path)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def append_result(result: ExperimentResult, output_path: str | Path) -> None:
    row = flatten_result(result)
    path = Path(output_path)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def write_recommendation(results: list[ExperimentResult], base_config: dict[str, Any], output_path: str | Path = "recommended_config.yaml") -> None:
    accepted = [result for result in results if result.accepted]
    if not accepted:
        return
    best = sorted(accepted, key=lambda result: result.score, reverse=True)[0]
    recommended = dict(base_config)
    recommended["watchlist"] = best.symbols.split()
    recommended["strategy"] = dict(base_config["strategy"])
    for key, value in best.params.items():
        if key != "cooldown_hours":
            recommended["strategy"][key] = value
    recommended["bot"] = dict(base_config["bot"])
    recommended["bot"]["cooldown_hours"] = int(best.params["cooldown_hours"])
    Path(output_path).write_text(yaml.safe_dump(recommended, sort_keys=False), encoding="utf-8")


def main() -> int:
    started_at = time.perf_counter()
    args = parse_args()
    base_config = load_config()
    watchlist = args.symbols or base_config["watchlist"]
    symbol_sets = build_symbol_sets(watchlist)
    if args.symbol_set:
        symbol_sets = {args.symbol_set: symbol_sets[args.symbol_set]}
    fetch_symbols = sorted({symbol for symbols in symbol_sets.values() for symbol in symbols})
    entry_tf = base_config["strategy"]["entry_timeframe"]
    trend_tf = base_config["strategy"]["trend_timeframe"]
    intervals = [entry_tf, trend_tf]
    set_data_source(
        api_base_url=args.api_base_url or base_config["backtest"].get("bybit_api_base_url"),
        cache_dir=args.cache_dir or base_config["backtest"].get("cache_dir"),
    )

    try:
        data = fetch_all_symbols(fetch_symbols, intervals, args.days)
    except BybitDataError as exc:
        print(str(exc))
        return 1
    base_strategy = dict(base_config["strategy"])
    for symbol in fetch_symbols:
        data[symbol][entry_tf], data[symbol][trend_tf] = compute_indicators(
            data[symbol][entry_tf],
            data[symbol][trend_tf],
            base_strategy,
        )
    prepared = prepare_optimizer_data(data, fetch_symbols, entry_tf, trend_tf)

    all_results: list[ExperimentResult] = []
    run_count = 0
    output_path = Path(args.output)
    if output_path.exists():
        output_path.unlink()
    total_planned = sum(len(parameter_grid()) for symbols in symbol_sets.values() if symbols)
    if args.max_runs:
        total_planned = min(total_planned, args.max_runs)
    grid = parameter_grid()
    for set_name, symbols in symbol_sets.items():
        if not symbols:
            continue
        start, split, end = walk_forward_bounds(data, symbols, entry_tf)
        for params in grid:
            strategy_cfg = dict(base_strategy)
            strategy_cfg.update({key: value for key, value in params.items() if key != "cooldown_hours"})
            strategy_cfg["cooldown_hours"] = params["cooldown_hours"]
            train_results = run_prepared_window(prepared, symbols, strategy_cfg, start, split)
            validation_results = run_prepared_window(prepared, symbols, strategy_cfg, split, end)
            train = metrics_from_results(train_results)
            validation = metrics_from_results(validation_results)
            score = score_candidate(train, validation)
            result = ExperimentResult(
                name=set_name,
                symbols=" ".join(symbols),
                params=params,
                train=train,
                validation=validation,
                accepted=candidate_accepted(validation),
                score=score,
            )
            all_results.append(result)
            append_result(result, args.output)
            run_count += 1
            elapsed = time.perf_counter() - started_at
            rate = elapsed / run_count if run_count else 0.0
            remaining = max(total_planned - run_count, 0) * rate
            print(
                f"[{run_count}/{total_planned}] {set_name} score={score:.2f} "
                f"val_trades={validation.trades} win={validation.win_rate:.1f}% avgR={validation.avg_r:.2f} "
                f"elapsed={elapsed:.1f}s eta={remaining:.1f}s",
                flush=True,
            )
            if args.max_runs and run_count >= args.max_runs:
                break
        if args.max_runs and run_count >= args.max_runs:
            break

    ranked = sorted(all_results, key=lambda result: result.score, reverse=True)
    write_results(ranked, args.output)
    write_recommendation(ranked, base_config)

    print(f"Tested {len(ranked)} experiments. Results written to {args.output}.")
    print("Top 10 candidates:")
    for result in ranked[:10]:
        status = "ACCEPTED" if result.accepted else "review"
        print(
            f"{status} | {result.name} | score={result.score:.2f} | "
            f"val trades={result.validation.trades} win={result.validation.win_rate:.1f}% "
            f"avgR={result.validation.avg_r:.2f} totalR={result.validation.total_r:.2f} "
            f"maxLoss={result.validation.max_consecutive_losses}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
