"""Optimizer for Strategy B parameter tightening."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, asdict
from itertools import product
from pathlib import Path
from typing import Any

import pandas as pd

from backtest.data_fetcher import BybitDataError, CacheDataError, fetch_all_symbols, set_data_source
from backtest.report import _max_consecutive_losses
from backtest.simulator import TradeResult, simulate_all
from backtest.strategy import load_config
from backtest.strategy_b_baseline import (
    STRATEGY_NAME,
    compute_confidence_b,
    compute_indicators_b,
    evaluate_signals_b,
    get_swing_high,
    get_swing_low,
)
from backtest.strategy import SignalEvent


@dataclass(frozen=True)
class StrategyBMetrics:
    trades: int
    trades_per_week: float
    win_rate: float
    total_r: float
    avg_r: float
    max_consecutive_losses: int
    top_symbol_trade_share: float


@dataclass(frozen=True)
class StrategyBExperiment:
    params: dict[str, Any]
    train: StrategyBMetrics
    validation: StrategyBMetrics
    accepted: bool
    score: float


@dataclass(frozen=True)
class StrategyBCandidate:
    symbol: str
    direction: str
    timestamp: pd.Timestamp
    entry: float
    high: float
    low: float
    atr_15m: float
    atr_pct: float
    volume_ratio: float
    rsi14: float
    prev_rsi14: float
    body_ratio: float
    breakout_margin_r: float
    breakout_margins: dict[int, float]
    ema55_trends: dict[int, bool]
    slope_strength: float
    atr_expanding: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize Strategy B tightening parameters.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--symbols", nargs="+")
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--api-base-url")
    parser.add_argument("--cache-dir")
    parser.add_argument("--max-runs", type=int)
    parser.add_argument("--output", default="strategy_b_optimization.csv")
    parser.add_argument("--grid", choices=["default", "strict"], default="strict")
    return parser.parse_args()


def parameter_grid(grid_name: str = "strict") -> list[dict[str, Any]]:
    keys = [
        "volume_spike_threshold",
        "swing_lookback",
        "candle_body_min_pct",
        "breakout_margin_atr",
        "min_confidence",
        "cooldown_hours",
        "atr_max_pct",
        "tp1_risk_multiplier",
        "tp2_risk_multiplier",
    ]
    if grid_name == "default":
        values = [
            [1.5, 1.8, 2.1],
            [12, 16, 24],
            [0.6, 0.7],
            [0.3, 0.6, 1.0],
            [65, 80, 95],
            [4, 8, 12],
            [0.025, 0.035],
            [1.5],
            [2.5, 3.0],
        ]
    else:
        values = [
        [3.0, 4.0],
        [48, 72, 96],
        [0.8, 0.9],
        [1.5, 2.0, 2.5],
        [95, 100],
        [24, 48, 72],
        [0.02, 0.03],
        [1.0, 1.25, 1.5],
        [2.5, 3.0],
        ]
    return [dict(zip(keys, combo)) for combo in product(*values)]


def walk_forward_bounds(data: dict[str, dict[str, pd.DataFrame]], symbols: list[str], interval: str) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    starts = []
    ends = []
    for symbol in symbols:
        timestamps = pd.to_datetime(data[symbol][interval]["timestamp"], utc=True)
        starts.append(timestamps.min())
        ends.append(timestamps.max())
    start = max(starts)
    end = min(ends)
    split = start + (end - start) * 0.7
    return start, split, end


def window_results(results: list[TradeResult], start: pd.Timestamp, end: pd.Timestamp) -> list[TradeResult]:
    return [result for result in results if start <= result.signal.timestamp <= end]


def metrics_from_results(results: list[TradeResult], days: float) -> StrategyBMetrics:
    trades = len(results)
    wins = sum(1 for result in results if result.outcome in {"TP1_ONLY", "TP2_HIT"})
    total_r = sum(result.pnl_r for result in results)
    by_symbol: dict[str, int] = {}
    for result in results:
        by_symbol[result.signal.symbol] = by_symbol.get(result.signal.symbol, 0) + 1
    top_share = max(by_symbol.values()) / trades if trades and by_symbol else 0.0
    return StrategyBMetrics(
        trades=trades,
        trades_per_week=trades / (days / 7) if days else 0.0,
        win_rate=(wins / trades * 100) if trades else 0.0,
        total_r=total_r,
        avg_r=(total_r / trades) if trades else 0.0,
        max_consecutive_losses=_max_consecutive_losses(results),
        top_symbol_trade_share=top_share,
    )


def _body_ratio(row: pd.Series) -> float:
    candle_range = float(row["high"]) - float(row["low"])
    if candle_range <= 0:
        return 0.0
    return abs(float(row["close"]) - float(row["open"])) / candle_range


def build_candidates_for_symbol(symbol: str, df: pd.DataFrame, cfg: dict[str, Any]) -> list[StrategyBCandidate]:
    """Precompute broad Strategy B breakout candidates once per symbol."""
    candidates: list[StrategyBCandidate] = []
    lookbacks = [8, 12, 16, 24, 36, 48, 72, 96]
    max_lookback = max(int(cfg["swing_lookback"]), int(cfg["ema_slope_lookback"]), 3)
    for idx in range(max_lookback, len(df)):
        row = df.iloc[idx]
        prev = df.iloc[idx - 1]
        if any(pd.isna(row.get(name)) for name in ["ema55", "rsi14", "atr14", "volume_sma20"]):
            continue
        close = float(row["close"])
        open_ = float(row["open"])
        ema55_value = float(row["ema55"])
        atr_15m = float(row["atr14"])
        volume_sma = float(row["volume_sma20"])
        if not close or not atr_15m or not volume_sma:
            continue
        volume_ratio = float(row["volume"]) / volume_sma
        atr_pct = atr_15m / close
        body_ratio = _body_ratio(row)
        rsi_value = float(row["rsi14"])
        prev_rsi = float(prev["rsi14"])
        for direction in ["LONG", "SHORT"]:
            breakout_margins = {}
            ema55_trends = {}
            if direction == "LONG":
                if not (close > ema55_value and close > open_ and rsi_value > 55 and rsi_value > prev_rsi):
                    continue
                for lookback in lookbacks:
                    if lookback > idx:
                        continue
                    ema55_trends[lookback] = ema55_value > float(df.iloc[idx - lookback]["ema55"])
                    swing_reference = get_swing_high(df, idx, lookback)
                    breakout_margins[lookback] = (close - swing_reference) / atr_15m
            else:
                if not (close < ema55_value and close < open_ and rsi_value < 45 and rsi_value < prev_rsi):
                    continue
                for lookback in lookbacks:
                    if lookback > idx:
                        continue
                    ema55_trends[lookback] = ema55_value < float(df.iloc[idx - lookback]["ema55"])
                    swing_reference = get_swing_low(df, idx, lookback)
                    breakout_margins[lookback] = (swing_reference - close) / atr_15m
            if max(breakout_margins.values()) <= 0:
                continue
            breakout_margin_r = breakout_margins.get(int(cfg["swing_lookback"]), max(breakout_margins.values()))
            slope_strength = abs((ema55_value - float(df.iloc[idx - 3]["ema55"])) / ema55_value) if ema55_value else 0.0
            candidates.append(
                StrategyBCandidate(
                    symbol=symbol,
                    direction=direction,
                    timestamp=pd.Timestamp(row["timestamp"]),
                    entry=close,
                    high=float(row["high"]),
                    low=float(row["low"]),
                    atr_15m=atr_15m,
                    atr_pct=atr_pct,
                    volume_ratio=volume_ratio,
                    rsi14=rsi_value,
                    prev_rsi14=prev_rsi,
                    body_ratio=body_ratio,
                    breakout_margin_r=breakout_margin_r,
                    breakout_margins=breakout_margins,
                    ema55_trends=ema55_trends,
                    slope_strength=slope_strength,
                    atr_expanding=bool(row.get("atr_expanding", False)),
                )
            )
    return candidates


def prepare_candidate_data(
    data: dict[str, dict[str, pd.DataFrame]],
    symbols: list[str],
    interval: str,
    cfg: dict[str, Any],
) -> tuple[dict[str, list[StrategyBCandidate]], dict[str, dict[str, pd.DataFrame]]]:
    prepared_candidates: dict[str, list[StrategyBCandidate]] = {}
    sim_data: dict[str, dict[str, pd.DataFrame]] = {}
    broad_cfg = dict(cfg)
    broad_cfg["swing_lookback"] = 96
    for symbol in symbols:
        df = compute_indicators_b(data[symbol][interval], broad_cfg)
        prepared_candidates[symbol] = build_candidates_for_symbol(symbol, df, broad_cfg)
        sim_data[symbol] = {"15": df}
    return prepared_candidates, sim_data


def candidate_to_signal(candidate: StrategyBCandidate, cfg: dict[str, Any]) -> SignalEvent | None:
    swing_lookback = int(cfg["swing_lookback"])
    trend_lookback = int(cfg.get("trend_lookback", swing_lookback))
    breakout_margin_r = candidate.breakout_margins.get(swing_lookback, candidate.breakout_margin_r)
    if not candidate.ema55_trends.get(trend_lookback, False):
        return None
    if candidate.volume_ratio <= cfg["volume_spike_threshold"]:
        return None
    if candidate.body_ratio < cfg["candle_body_min_pct"]:
        return None
    if breakout_margin_r <= cfg["breakout_margin_atr"]:
        return None
    if candidate.atr_pct > cfg["atr_max_pct"] or candidate.atr_pct < cfg["atr_min_pct"]:
        return None

    if candidate.direction == "LONG":
        stop_loss = candidate.low - cfg["sl_atr_buffer"] * candidate.atr_15m
        risk = candidate.entry - stop_loss
        tp1 = candidate.entry + cfg["tp1_risk_multiplier"] * risk
        tp2 = candidate.entry + cfg["tp2_risk_multiplier"] * risk
    else:
        stop_loss = candidate.high + cfg["sl_atr_buffer"] * candidate.atr_15m
        risk = stop_loss - candidate.entry
        tp1 = candidate.entry - cfg["tp1_risk_multiplier"] * risk
        tp2 = candidate.entry - cfg["tp2_risk_multiplier"] * risk
    risk_reward = abs(tp2 - candidate.entry) / risk if risk > 0 else 0.0
    if risk_reward < cfg["min_risk_reward"]:
        return None

    conditions = {
        "ema55_slope_strong": candidate.slope_strength > 0.001,
        "strong_volume": candidate.volume_ratio > cfg["volume_strong_threshold"],
        "strong_rsi": candidate.rsi14 > 60 if candidate.direction == "LONG" else candidate.rsi14 < 40,
        "strong_candle_body": candidate.body_ratio >= cfg["candle_body_strong_pct"],
        "clean_breakout": breakout_margin_r > cfg["breakout_margin_atr"],
        "atr_expanding": candidate.atr_expanding,
    }
    confidence, label = compute_confidence_b(conditions)
    if confidence < cfg["min_confidence"]:
        return None
    return SignalEvent(
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
        reasons=[],
        confidence_conditions=conditions,
        target_rr=risk_reward,
        target_note=f"Strategy B target 1:{risk_reward:.2f}",
        strategy_name=STRATEGY_NAME,
    )


def evaluate_candidates_fast(candidates: dict[str, list[StrategyBCandidate]], symbols: list[str], cfg: dict[str, Any]) -> list[SignalEvent]:
    signals: list[SignalEvent] = []
    cooldown = pd.Timedelta(hours=int(cfg["cooldown_hours"]))
    for symbol in symbols:
        last_signal_by_direction: dict[str, pd.Timestamp] = {}
        for candidate in candidates[symbol]:
            last_same = last_signal_by_direction.get(candidate.direction)
            if last_same is not None and candidate.timestamp - last_same < cooldown:
                continue
            signal = candidate_to_signal(candidate, cfg)
            if signal is None:
                continue
            signals.append(signal)
            last_signal_by_direction[candidate.direction] = candidate.timestamp
    return signals


def candidate_accepted(metrics: StrategyBMetrics) -> bool:
    return (
        2.5 <= metrics.trades_per_week <= 9
        and metrics.win_rate >= 40
        and metrics.total_r > 0
        and metrics.avg_r > 0
        and metrics.max_consecutive_losses <= 6
        and metrics.top_symbol_trade_share <= 0.60
    )


def score_candidate(train: StrategyBMetrics, validation: StrategyBMetrics) -> float:
    if validation.trades == 0:
        return -9999.0
    frequency_penalty = abs(validation.trades_per_week - 5.5) * 4
    concentration_penalty = max(validation.top_symbol_trade_share - 0.45, 0) * 30
    return (
        validation.total_r * 2
        + validation.avg_r * 25
        + validation.win_rate
        - validation.max_consecutive_losses * 2
        - frequency_penalty
        - concentration_penalty
        + train.avg_r * 10
    )


def evaluate_params(
    data: dict[str, dict[str, pd.DataFrame]],
    symbols: list[str],
    interval: str,
    base_cfg: dict[str, Any],
    params: dict[str, Any],
    indicator_cache: dict[tuple, dict[str, pd.DataFrame]] | None = None,
) -> list[TradeResult]:
    cfg = dict(base_cfg)
    cfg.update(params)
    indicator_key = (
        cfg["ema_slow"],
        cfg["rsi_period"],
        cfg["atr_period"],
        cfg["volume_sma_period"],
        cfg["ema_slope_lookback"],
    )
    indicator_cache = indicator_cache if indicator_cache is not None else {}
    if indicator_key not in indicator_cache:
        indicator_cache[indicator_key] = {
            symbol: compute_indicators_b(data[symbol][interval], cfg) for symbol in symbols
        }
    signals = []
    eval_data: dict[str, dict[str, pd.DataFrame]] = {}
    for symbol in symbols:
        df = indicator_cache[indicator_key][symbol]
        eval_data[symbol] = {"15": df}
        signals.extend(evaluate_signals_b(symbol, df, int(cfg["cooldown_hours"]), cfg))
    return simulate_all(signals, eval_data)


def flatten_result(result: StrategyBExperiment) -> dict[str, Any]:
    row = {"accepted": result.accepted, "score": round(result.score, 4)}
    row.update(result.params)
    for prefix, metrics in [("train", result.train), ("validation", result.validation)]:
        for key, value in asdict(metrics).items():
            row[f"{prefix}_{key}"] = round(value, 4) if isinstance(value, float) else value
    return row


def append_result(result: StrategyBExperiment, output_path: str | Path) -> None:
    row = flatten_result(result)
    path = Path(output_path)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    symbols = args.symbols or config["watchlist"]
    cfg = config["strategy_b"]
    interval = cfg["timeframe"]
    set_data_source(
        api_base_url=args.api_base_url or config["backtest"].get("bybit_api_base_url"),
        cache_dir=args.cache_dir or config["backtest"].get("cache_dir"),
    )

    try:
        data = fetch_all_symbols(symbols, [interval], args.days, cache_only=args.cache_only)
    except (BybitDataError, CacheDataError) as exc:
        print(str(exc))
        return 1

    start, split, end = walk_forward_bounds(data, symbols, interval)
    train_days = (split - start).total_seconds() / 86400
    validation_days = (end - split).total_seconds() / 86400
    prepared_candidates, sim_data = prepare_candidate_data(data, symbols, interval, cfg)
    output_path = Path(args.output)
    if output_path.exists():
        output_path.unlink()

    results: list[StrategyBExperiment] = []
    grid = parameter_grid(args.grid)
    total = min(len(grid), args.max_runs) if args.max_runs else len(grid)
    for index, params in enumerate(grid[:total], start=1):
        eval_cfg = dict(cfg)
        eval_cfg.update(params)
        signals = evaluate_candidates_fast(prepared_candidates, symbols, eval_cfg)
        all_results = simulate_all(signals, sim_data)
        train = metrics_from_results(window_results(all_results, start, split), train_days)
        validation = metrics_from_results(window_results(all_results, split, end), validation_days)
        score = score_candidate(train, validation)
        result = StrategyBExperiment(
            params=params,
            train=train,
            validation=validation,
            accepted=candidate_accepted(validation),
            score=score,
        )
        results.append(result)
        append_result(result, output_path)
        print(
            f"[{index}/{total}] score={score:.2f} val_tr/wk={validation.trades_per_week:.2f} "
            f"win={validation.win_rate:.1f}% avgR={validation.avg_r:.2f} totalR={validation.total_r:.2f} "
            f"accepted={result.accepted}",
            flush=True,
        )

    ranked = sorted(results, key=lambda item: item.score, reverse=True)
    print("Top 10 Strategy B candidates:")
    for result in ranked[:10]:
        print(
            f"{'ACCEPTED' if result.accepted else 'review'} | score={result.score:.2f} "
            f"val_tr/wk={result.validation.trades_per_week:.2f} win={result.validation.win_rate:.1f}% "
            f"avgR={result.validation.avg_r:.2f} totalR={result.validation.total_r:.2f} "
            f"maxLoss={result.validation.max_consecutive_losses} params={result.params}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
