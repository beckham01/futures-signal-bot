"""ROI-focused trade simulation and reporting."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from backtest.strategy import SignalEvent

ROI_TIERS = [100, 200, 300, 400, 500]


@dataclass
class RoiTradeResult:
    signal: SignalEvent
    max_favorable_roi_pct: float
    max_adverse_roi_pct: float
    reached_tiers: dict[int, bool]
    first_hit_timestamps: dict[int, pd.Timestamp | None]
    stopped: bool
    bars_held: int
    exit_timestamp: pd.Timestamp
    roi_label: str = "NO_ROI_TARGET"


@dataclass(frozen=True)
class RoiMetrics:
    trades: int
    roi100_hit_rate: float
    roi200_hit_rate: float
    total_expectancy_r: float
    avg_expectancy_r: float
    max_consecutive_roi100_failures: int
    avg_bars_to_roi100: float


def required_price_move_pct(target_roi_pct: float, leverage: float) -> float:
    """Return underlying price move percentage required for a leveraged ROI target."""
    if leverage <= 0:
        raise ValueError("leverage must be positive")
    return target_roi_pct / leverage


def roi_target_price(entry: float, direction: str, target_roi_pct: float, leverage: float) -> float:
    """Return price required to hit target ROI."""
    move = required_price_move_pct(target_roi_pct, leverage) / 100
    if direction == "LONG":
        return entry * (1 + move)
    target = entry * (1 - move)
    if target <= 0:
        raise ValueError("short ROI target requires price at or below zero")
    return target


def roi_label_for_tiers(reached_tiers: dict[int, bool]) -> str:
    """Return the highest reached ROI label."""
    if reached_tiers.get(500):
        return "ROI500_EXTREME"
    if reached_tiers.get(400):
        return "ROI400_STRETCH"
    if reached_tiers.get(300):
        return "ROI300_STRETCH"
    if reached_tiers.get(200):
        return "ROI200_STRETCH"
    if reached_tiers.get(100):
        return "ROI100_HIGH_PROB"
    return "NO_ROI_TARGET"


def _roi_from_price_move(entry: float, price: float, direction: str, leverage: float) -> float:
    raw = (price - entry) / entry * 100
    favorable = raw if direction == "LONG" else -raw
    return favorable * leverage


def simulate_roi_trade(
    signal: SignalEvent,
    future_candles: pd.DataFrame,
    leverage: float = 5,
    tiers: list[int] | None = None,
    max_bars: int = 240,
) -> RoiTradeResult:
    """Track ROI tiers reached before stop or max hold."""
    tiers = tiers or ROI_TIERS
    candles = future_candles.sort_values("timestamp").head(max_bars).reset_index(drop=True)
    reached = {tier: False for tier in tiers}
    first_hits: dict[int, pd.Timestamp | None] = {tier: None for tier in tiers}
    max_favorable = 0.0
    max_adverse = 0.0
    stopped = False
    exit_timestamp = signal.timestamp
    bars_held = 0

    for index, candle in candles.iterrows():
        high = float(candle["high"])
        low = float(candle["low"])
        timestamp = pd.Timestamp(candle["timestamp"])
        bars_held = index + 1
        exit_timestamp = timestamp
        favorable_price = high if signal.direction == "LONG" else low
        adverse_price = low if signal.direction == "LONG" else high
        max_favorable = max(max_favorable, _roi_from_price_move(signal.entry, favorable_price, signal.direction, leverage))
        max_adverse = min(max_adverse, _roi_from_price_move(signal.entry, adverse_price, signal.direction, leverage))

        stop_hit = low <= signal.stop_loss if signal.direction == "LONG" else high >= signal.stop_loss
        if stop_hit:
            stopped = True
            break

        for tier in tiers:
            if reached[tier]:
                continue
            try:
                target = roi_target_price(signal.entry, signal.direction, tier, leverage)
            except ValueError:
                continue
            hit = high >= target if signal.direction == "LONG" else low <= target
            if hit:
                reached[tier] = True
                first_hits[tier] = timestamp

    label = roi_label_for_tiers(reached)
    return RoiTradeResult(
        signal=signal,
        max_favorable_roi_pct=round(max_favorable, 2),
        max_adverse_roi_pct=round(max_adverse, 2),
        reached_tiers=reached,
        first_hit_timestamps=first_hits,
        stopped=stopped,
        bars_held=bars_held,
        exit_timestamp=exit_timestamp,
        roi_label=label,
    )


def simulate_roi_all(
    signals: list[SignalEvent],
    data: dict[str, dict[str, pd.DataFrame]],
    leverage: float = 5,
    tiers: list[int] | None = None,
    max_bars_by_timeframe: dict[str, int] | None = None,
) -> list[RoiTradeResult]:
    """Run ROI simulation for all signals."""
    results = []
    max_bars_by_timeframe = max_bars_by_timeframe or {}
    for signal in signals:
        timeframe = getattr(signal, "execution_timeframe", "15")
        candles = data[signal.symbol][timeframe]
        future = candles[pd.to_datetime(candles["timestamp"], utc=True) > signal.timestamp]
        results.append(
            simulate_roi_trade(
                signal,
                future,
                leverage=leverage,
                tiers=tiers,
                max_bars=max_bars_by_timeframe.get(timeframe, 240),
            )
        )
    return results


def roi_metrics(results: list[RoiTradeResult]) -> RoiMetrics:
    trades = len(results)
    roi100_hits = sum(1 for result in results if result.reached_tiers.get(100))
    roi200_hits = sum(1 for result in results if result.reached_tiers.get(200))
    expectancy = sum(1.0 if result.reached_tiers.get(100) else -1.0 if result.stopped else 0.0 for result in results)
    failures = 0
    max_failures = 0
    bars_to_roi100 = []
    for result in sorted(results, key=lambda item: item.signal.timestamp):
        if result.reached_tiers.get(100):
            failures = 0
            hit_ts = result.first_hit_timestamps.get(100)
            if hit_ts is not None:
                bars_to_roi100.append(result.bars_held)
        else:
            failures += 1
            max_failures = max(max_failures, failures)
    return RoiMetrics(
        trades=trades,
        roi100_hit_rate=(roi100_hits / trades * 100) if trades else 0.0,
        roi200_hit_rate=(roi200_hits / trades * 100) if trades else 0.0,
        total_expectancy_r=expectancy,
        avg_expectancy_r=(expectancy / trades) if trades else 0.0,
        max_consecutive_roi100_failures=max_failures,
        avg_bars_to_roi100=(sum(bars_to_roi100) / len(bars_to_roi100)) if bars_to_roi100 else 0.0,
    )


def generate_roi_report(
    results: list[RoiTradeResult],
    strategy_name: str,
    leverage: float,
    output_path: str | Path,
) -> str:
    """Write and print a compact ROI report."""
    metrics = roi_metrics(results)
    tier_counts = {tier: sum(1 for result in results if result.reached_tiers.get(tier)) for tier in ROI_TIERS}
    by_symbol = Counter(result.signal.symbol for result in results)
    failed = [result for result in results if not result.reached_tiers.get(100)]
    lines = [
        f"=== ROI REPORT: {strategy_name} ===",
        f"Leverage: {leverage:.1f}x",
        f"Qualified signals: {metrics.trades}",
        f"+100% ROI hit rate: {metrics.roi100_hit_rate:.1f}%",
        f"+200% ROI hit rate: {metrics.roi200_hit_rate:.1f}%",
        f"Expectancy proxy: {metrics.total_expectancy_r:.2f}R total | {metrics.avg_expectancy_r:.2f}R avg",
        f"Max consecutive ROI100 failures: {metrics.max_consecutive_roi100_failures}",
        f"Average bars to ROI100: {metrics.avg_bars_to_roi100:.1f}",
        "",
        "--- ROI TIERS ---",
    ]
    for tier in ROI_TIERS:
        hit_rate = (tier_counts[tier] / metrics.trades * 100) if metrics.trades else 0.0
        lines.append(f"+{tier}% ROI: {tier_counts[tier]} ({hit_rate:.1f}%)")
    lines.extend(["", "--- BY SYMBOL ---"])
    for symbol, count in by_symbol.most_common():
        symbol_results = [result for result in results if result.signal.symbol == symbol]
        symbol_metrics = roi_metrics(symbol_results)
        lines.append(f"{symbol}: {count} signals | ROI100 {symbol_metrics.roi100_hit_rate:.1f}%")
    lines.extend(["", "--- FAILED ROI100 SETUPS (last 20) ---"])
    for result in failed[-20:]:
        lines.append(
            f"{result.signal.timestamp} | {result.signal.symbol} | {result.signal.direction} | "
            f"MFE ROI {result.max_favorable_roi_pct:.1f}% | MAE ROI {result.max_adverse_roi_pct:.1f}%"
        )
    report = "\n".join(lines)
    Path(output_path).write_text(report + "\n", encoding="utf-8")
    print(report)
    return report
