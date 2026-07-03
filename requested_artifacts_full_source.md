# Requested Full Source Code

## backtest/strategy_b.py
```python
"""Strategy B: 15m FVG + breaker block confluence."""

from __future__ import annotations

from typing import Any

import pandas as pd

from backtest.ict_patterns import detect_breaker_block, detect_fvgs, find_confluence_zone, zones_overlap
from backtest.indicators import atr, ema, ema_slope, rsi, volume_sma
from backtest.strategy import SignalEvent, load_config

STRATEGY_NAME = "strategy_b_fvg_breaker_15m"


def strategy_b_config(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = load_config()["strategy_b"].copy()
    if overrides:
        cfg.update(overrides)
    return cfg


def compute_indicators_b(
    df_15m: pd.DataFrame,
    df_1h: pd.DataFrame | None = None,
    cfg: dict[str, Any] | None = None,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Add Strategy B indicators to 15m candles and optionally 1h trend candles."""
    cfg = cfg or strategy_b_config()
    entry = _with_indicators(df_15m, cfg, include_volume=True)
    if df_1h is None:
        return entry
    trend = _with_indicators(df_1h, cfg, include_volume=False)
    trend["ema55_slope"] = ema_slope(trend["ema55"], 3)
    return entry, trend


def _with_indicators(df: pd.DataFrame, cfg: dict[str, Any], include_volume: bool) -> pd.DataFrame:
    result = df.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True)
    result.sort_values("timestamp", inplace=True)
    result.reset_index(drop=True, inplace=True)
    result["ema21"] = ema(result["close"], cfg["ema_fast"])
    result["ema55"] = ema(result["close"], cfg["ema_slow"])
    result["rsi14"] = rsi(result["close"], cfg["rsi_period"])
    result["atr14"] = atr(result, cfg["atr_period"])
    if include_volume:
        result["volume_sma20"] = volume_sma(result["volume"], cfg["volume_sma_period"])
    return result


def compute_confidence_b(conditions: dict[str, bool]) -> tuple[int, str]:
    weights = {
        "trend_rsi_strong": 20,
        "large_fvg": 20,
        "fresh_breaker": 15,
        "momentum_aligned": 15,
        "strong_rejection_body": 15,
        "volume_above_average": 15,
    }
    score = sum(weight for key, weight in weights.items() if conditions.get(key, False))
    if score >= 80:
        label = "STRONG"
    elif score >= 65:
        label = "HIGH"
    elif score >= 50:
        label = "MODERATE"
    else:
        label = "LOW"
    return int(score), label


def _body_ratio(row: pd.Series) -> float:
    candle_range = float(row["high"]) - float(row["low"])
    if candle_range <= 0:
        return 0.0
    return abs(float(row["close"]) - float(row["open"])) / candle_range


def _closed_trend_frame(df_1h: pd.DataFrame, timeframe: str = "60") -> pd.DataFrame:
    closed = df_1h.copy()
    closed["available_at"] = pd.to_datetime(closed["timestamp"], utc=True) + pd.Timedelta(minutes=int(timeframe))
    return closed


def _trend_bias(row: pd.Series) -> str | None:
    if float(row["ema21"]) > float(row["ema55"]):
        return "LONG"
    if float(row["ema21"]) < float(row["ema55"]):
        return "SHORT"
    return None


def _matching_fvg(fvgs: list[dict], breaker: dict, direction: str) -> dict | None:
    for fvg in fvgs:
        if fvg.get("type") == direction and zones_overlap(fvg["zone"], breaker["zone"]):
            return fvg
    return None


def _risk_levels(direction: str, entry: float, zone: tuple[float, float], atr_value: float, cfg: dict[str, Any]):
    if direction == "LONG":
        stop_loss = zone[0] - cfg["sl_atr_buffer"] * atr_value
        risk = entry - stop_loss
        tp1 = entry + cfg["tp1_risk_multiplier"] * risk
        tp2 = entry + cfg["tp2_risk_multiplier"] * risk
    else:
        stop_loss = zone[1] + cfg["sl_atr_buffer"] * atr_value
        risk = stop_loss - entry
        tp1 = entry - cfg["tp1_risk_multiplier"] * risk
        tp2 = entry - cfg["tp2_risk_multiplier"] * risk
    risk_reward = abs(tp2 - entry) / risk if risk > 0 else 0.0
    return stop_loss, tp1, tp2, risk_reward, risk


def _signal_reasons(direction: str, zone: tuple[float, float], volume_ratio: float) -> list[str]:
    bias = "bullish" if direction == "LONG" else "bearish"
    return [
        f"1h EMA structure is {bias}",
        f"15m FVG overlaps breaker block at {zone[0]:.4f}-{zone[1]:.4f}",
        "Entry candle rejected the confluence zone",
        f"Volume {volume_ratio:.2f}x average",
    ]


def _fast_confluence_by_index(df: pd.DataFrame, cfg: dict[str, Any]) -> list[dict[str, tuple[tuple[float, float], dict, dict]]]:
    highs = df["high"].astype(float).to_list()
    lows = df["low"].astype(float).to_list()
    opens = df["open"].astype(float).to_list()
    closes = df["close"].astype(float).to_list()
    atrs = df["atr14"].astype(float).to_list()
    n = len(df)
    active_fvgs: list[dict] = []
    breakers: dict[str, dict | None] = {"bullish": None, "bearish": None}
    result: list[dict[str, tuple[tuple[float, float], dict, dict]]] = [dict() for _ in range(n)]
    confirm = int(cfg["breaker_confirm_candles"])

    def add_visible_fvg(idx: int) -> None:
        center = idx - 2
        if center < 1 or center + 1 >= n or atrs[center] <= 0:
            return
        min_gap = atrs[center] * float(cfg["fvg_min_gap_atr"])
        if highs[center - 1] < lows[center + 1]:
            zone = (highs[center - 1], lows[center + 1])
            fvg_type = "bullish"
        elif lows[center - 1] > highs[center + 1]:
            zone = (highs[center + 1], lows[center - 1])
            fvg_type = "bearish"
        else:
            return
        if zone[1] - zone[0] >= min_gap:
            active_fvgs.insert(0, {"type": fvg_type, "zone": zone, "formed_at": center, "gap_size": zone[1] - zone[0]})

    def fvg_active(fvg: dict, idx: int) -> bool:
        if idx - int(fvg["formed_at"]) > int(cfg["fvg_max_age_candles"]):
            return False
        low, high = fvg["zone"]
        if fvg["type"] == "bullish":
            return closes[idx] > low
        return closes[idx] < high

    def is_swing_low(j: int) -> bool:
        if j < confirm or j + confirm >= n:
            return False
        return all(lows[k] > lows[j] for k in range(j - confirm, j)) and all(
            lows[k] > lows[j] for k in range(j + 1, j + confirm + 1)
        )

    def is_swing_high(j: int) -> bool:
        if j < confirm or j + confirm >= n:
            return False
        return all(highs[k] < highs[j] for k in range(j - confirm, j)) and all(
            highs[k] < highs[j] for k in range(j + 1, j + confirm + 1)
        )

    def last_opposing(j: int, direction: str) -> int | None:
        for k in range(j - 1, -1, -1):
            if direction == "bullish" and closes[k] < opens[k]:
                return k
            if direction == "bearish" and closes[k] > opens[k]:
                return k
        return None

    def maybe_add_breaker(idx: int, direction: str) -> None:
        j = idx - confirm
        if j < confirm or atrs[j] <= 0:
            return
        if direction == "bullish":
            if not is_swing_low(j) or max(highs[j + 1 : idx + 1]) < lows[j] + atrs[j]:
                return
        else:
            if not is_swing_high(j) or min(lows[j + 1 : idx + 1]) > highs[j] - atrs[j]:
                return
        source = last_opposing(j, direction)
        if source is None:
            return
        breakers[direction] = {
            "type": direction,
            "zone": (lows[source], highs[source]),
            "formed_at": j,
            "source_index": source,
            "mitigated": False,
        }

    def breaker_active(breaker: dict, idx: int, direction: str) -> bool:
        if idx - int(breaker["formed_at"]) > int(cfg["breaker_max_age_candles"]):
            return False
        low, high = breaker["zone"]
        if direction == "bullish":
            return closes[idx] >= low
        return closes[idx] <= high

    for idx in range(n):
        add_visible_fvg(idx)
        active_fvgs = [fvg for fvg in active_fvgs if fvg_active(fvg, idx)]
        maybe_add_breaker(idx, "bullish")
        maybe_add_breaker(idx, "bearish")
        for direction in ["bullish", "bearish"]:
            breaker = breakers[direction]
            if breaker is None or not breaker_active(breaker, idx, direction):
                breakers[direction] = None
                continue
            for fvg in active_fvgs:
                if fvg["type"] == direction and zones_overlap(fvg["zone"], breaker["zone"]):
                    zone = (max(fvg["zone"][0], breaker["zone"][0]), min(fvg["zone"][1], breaker["zone"][1]))
                    result[idx][direction] = (zone, fvg, breaker)
                    break
    return result


def evaluate_signals_b(
    symbol: str,
    df_15m: pd.DataFrame,
    df_1h: pd.DataFrame | None = None,
    cooldown_hours: int = 4,
    cfg: dict[str, Any] | None = None,
) -> list[SignalEvent]:
    """Evaluate 15m FVG/breaker signals using a closed 1h EMA bias."""
    cfg = cfg or strategy_b_config()
    if df_1h is None:
        return []
    if "ema21" not in df_15m.columns or "volume_sma20" not in df_15m.columns:
        df_15m, df_1h = compute_indicators_b(df_15m, df_1h, cfg)

    entry = df_15m.sort_values("timestamp").reset_index(drop=True)
    trend = _closed_trend_frame(df_1h.sort_values("timestamp").reset_index(drop=True), cfg.get("timeframe_trend", "60"))
    merged = pd.merge_asof(
        entry,
        trend,
        left_on="timestamp",
        right_on="available_at",
        suffixes=("_15m", "_1h"),
        direction="backward",
    )
    signals: list[SignalEvent] = []
    last_signal_by_direction: dict[str, pd.Timestamp] = {}
    cooldown = pd.Timedelta(hours=cooldown_hours)
    fast_confluence = _fast_confluence_by_index(entry, cfg) if cfg.get("fast_scan", True) else None

    min_idx = max(8, int(cfg["breaker_confirm_candles"]) * 2 + 2)
    for idx in range(min_idx, len(entry)):
        row = entry.iloc[idx]
        prev = entry.iloc[idx - 1]
        merged_row = merged.iloc[idx]
        required = ["ema21_1h", "ema55_1h", "rsi14_1h", "atr14", "rsi14", "volume_sma20"]
        if any(pd.isna(merged_row.get(name)) for name in ["ema21_1h", "ema55_1h", "rsi14_1h"]):
            continue
        if any(pd.isna(row.get(name)) for name in required[3:]) or pd.isna(prev.get("rsi14")):
            continue
        bias = _trend_bias(pd.Series({"ema21": merged_row["ema21_1h"], "ema55": merged_row["ema55_1h"]}))
        if bias is None:
            continue
        ict_direction = "bullish" if bias == "LONG" else "bearish"
        timestamp = pd.Timestamp(row["timestamp"])
        last_same = last_signal_by_direction.get(bias)
        if last_same is not None and timestamp - last_same < cooldown:
            continue

        atr_value = float(row["atr14"])
        close = float(row["close"])
        atr_pct = atr_value / close if close else 0.0
        if atr_pct < cfg["atr_min_pct"] or atr_pct > cfg["atr_max_pct"]:
            continue
        body_ratio = _body_ratio(row)
        rsi_now = float(row["rsi14"])
        rsi_prev = float(prev["rsi14"])
        if bias == "LONG":
            if not (rsi_now > 45 and rsi_now > rsi_prev and body_ratio >= cfg["candle_body_min_pct"]):
                continue
        else:
            if not (rsi_now < 55 and rsi_now < rsi_prev and body_ratio >= cfg["candle_body_min_pct"]):
                continue
        if fast_confluence is not None:
            confluence = fast_confluence[idx].get(ict_direction)
            if confluence is None:
                continue
            zone, matched_fvg, breaker = confluence
        else:
            fvgs = detect_fvgs(entry, idx, entry["atr14"], cfg["fvg_max_age_candles"], cfg["fvg_min_gap_atr"])
            breaker = detect_breaker_block(
                entry,
                idx,
                entry["atr14"],
                ict_direction,
                cfg["breaker_confirm_candles"],
                cfg["breaker_max_age_candles"],
            )
            zone = find_confluence_zone(breaker, fvgs, ict_direction) if breaker else None
            matched_fvg = _matching_fvg(fvgs, breaker, ict_direction) if breaker else None
        if zone is None or matched_fvg is None:
            continue

        if bias == "LONG":
            trigger = (
                float(row["low"]) <= zone[1]
                and close > zone[1]
                and rsi_now > 45
                and rsi_now > rsi_prev
                and body_ratio >= cfg["candle_body_min_pct"]
            )
        else:
            trigger = (
                float(row["high"]) >= zone[0]
                and close < zone[0]
                and rsi_now < 55
                and rsi_now < rsi_prev
                and body_ratio >= cfg["candle_body_min_pct"]
            )
        if not trigger:
            continue
        stop_loss, tp1, tp2, risk_reward, _ = _risk_levels(bias, close, zone, atr_value, cfg)
        if risk_reward < cfg["min_risk_reward"]:
            continue

        volume_sma = float(row["volume_sma20"])
        volume_ratio = float(row["volume"]) / volume_sma if volume_sma else 0.0
        trend_rsi = float(merged_row["rsi14_1h"])
        confidence_conditions = {
            "trend_rsi_strong": 50 <= trend_rsi <= 65 if bias == "LONG" else 35 <= trend_rsi <= 50,
            "large_fvg": float(matched_fvg["gap_size"]) > 0.5 * atr_value,
            "fresh_breaker": idx - int(breaker["formed_at"]) <= 24,
            "momentum_aligned": rsi_now > 55 if bias == "LONG" else rsi_now < 45,
            "strong_rejection_body": body_ratio >= 0.60,
            "volume_above_average": volume_ratio > 1.0,
        }
        confidence, label = compute_confidence_b(confidence_conditions)
        if confidence < cfg["min_confidence"]:
            continue
        signals.append(
            SignalEvent(
                symbol=symbol,
                direction=bias,
                timestamp=timestamp,
                entry=close,
                stop_loss=stop_loss,
                tp1=tp1,
                tp2=tp2,
                risk_reward=risk_reward,
                confidence=confidence,
                confidence_label=label,
                atr_15m=atr_value,
                reasons=_signal_reasons(bias, zone, volume_ratio),
                confidence_conditions=confidence_conditions,
                target_rr=risk_reward,
                target_note=f"Strategy B target 1:{risk_reward:.2f}",
                strategy_name=STRATEGY_NAME,
                execution_timeframe=cfg["timeframe_entry"],
            )
        )
        last_signal_by_direction[bias] = timestamp
    return signals

``````

## backtest/strategy_c.py
```python
"""Strategy C: 4h FVG + breaker block confluence."""

from __future__ import annotations

from typing import Any

import pandas as pd

from backtest.ict_patterns import detect_breaker_block, detect_fvgs, find_confluence_zone, zones_overlap
from backtest.indicators import atr, ema, ema_slope, rsi
from backtest.strategy import SignalEvent, load_config

STRATEGY_NAME = "strategy_c_fvg_breaker_4h"


def strategy_c_config(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = load_config()["strategy_c"].copy()
    if overrides:
        cfg.update(overrides)
    return cfg


def compute_indicators_c(df_4h: pd.DataFrame, cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    """Add Strategy C indicators to 4h candles."""
    cfg = cfg or strategy_c_config()
    df = df_4h.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    df["ema21"] = ema(df["close"], cfg["ema_fast"])
    df["ema55"] = ema(df["close"], cfg["ema_slow"])
    df["ema21_slope"] = ema_slope(df["ema21"], 3)
    df["ema55_slope"] = ema_slope(df["ema55"], 3)
    df["rsi14"] = rsi(df["close"], cfg["rsi_period"])
    df["atr14"] = atr(df, cfg["atr_period"])
    return df


def compute_confidence_c(conditions: dict[str, bool]) -> tuple[int, str]:
    weights = {
        "ema_slopes_aligned": 25,
        "large_fvg": 20,
        "fresh_breaker": 20,
        "healthy_rsi": 20,
        "strong_rejection_body": 15,
    }
    score = sum(weight for key, weight in weights.items() if conditions.get(key, False))
    if score >= 90:
        label = "STRONG"
    elif score >= 75:
        label = "HIGH"
    elif score >= 60:
        label = "MODERATE"
    else:
        label = "LOW"
    return int(score), label


def _body_ratio(row: pd.Series) -> float:
    candle_range = float(row["high"]) - float(row["low"])
    if candle_range <= 0:
        return 0.0
    return abs(float(row["close"]) - float(row["open"])) / candle_range


def _bias(row: pd.Series) -> str | None:
    close = float(row["close"])
    ema21 = float(row["ema21"])
    ema55 = float(row["ema55"])
    if ema21 > ema55 and close > ema55:
        return "LONG"
    if ema21 < ema55 and close < ema55:
        return "SHORT"
    return None


def _matching_fvg(fvgs: list[dict], breaker: dict, direction: str) -> dict | None:
    for fvg in fvgs:
        if fvg.get("type") == direction and zones_overlap(fvg["zone"], breaker["zone"]):
            return fvg
    return None


def _risk_levels(direction: str, entry: float, zone: tuple[float, float], atr_value: float, cfg: dict[str, Any]):
    if direction == "LONG":
        stop_loss = zone[0] - cfg["sl_atr_buffer"] * atr_value
        risk = entry - stop_loss
        tp1 = entry + cfg["tp1_risk_multiplier"] * risk
        tp2 = entry + cfg["tp2_risk_multiplier"] * risk
    else:
        stop_loss = zone[1] + cfg["sl_atr_buffer"] * atr_value
        risk = stop_loss - entry
        tp1 = entry - cfg["tp1_risk_multiplier"] * risk
        tp2 = entry - cfg["tp2_risk_multiplier"] * risk
    risk_reward = abs(tp2 - entry) / risk if risk > 0 else 0.0
    return stop_loss, tp1, tp2, risk_reward


def _reasons(direction: str, zone: tuple[float, float]) -> list[str]:
    bias = "bullish" if direction == "LONG" else "bearish"
    return [
        f"4h EMA structure is {bias}",
        f"4h FVG overlaps breaker block at {zone[0]:.4f}-{zone[1]:.4f}",
        "4h candle rejected the confluence zone",
        "Higher timeframe setup selected",
    ]


def evaluate_signals_c(
    symbol: str,
    df_4h: pd.DataFrame,
    cooldown_hours: int = 12,
    cfg: dict[str, Any] | None = None,
) -> list[SignalEvent]:
    """Evaluate Strategy C signals for one symbol on 4h candles."""
    cfg = cfg or strategy_c_config()
    if "ema21" not in df_4h.columns or "atr14" not in df_4h.columns:
        df_4h = compute_indicators_c(df_4h, cfg)
    frame = df_4h.sort_values("timestamp").reset_index(drop=True)
    signals: list[SignalEvent] = []
    last_signal_by_direction: dict[str, pd.Timestamp] = {}
    cooldown = pd.Timedelta(hours=cooldown_hours)
    min_idx = max(8, int(cfg["breaker_confirm_candles"]) * 2 + 2)
    for idx in range(min_idx, len(frame)):
        row = frame.iloc[idx]
        required = ["ema21", "ema55", "ema21_slope", "ema55_slope", "rsi14", "atr14"]
        if any(pd.isna(row.get(name)) for name in required):
            continue
        direction = _bias(row)
        if direction is None:
            continue
        timestamp = pd.Timestamp(row["timestamp"])
        last_same = last_signal_by_direction.get(direction)
        if last_same is not None and timestamp - last_same < cooldown:
            continue
        close = float(row["close"])
        atr_value = float(row["atr14"])
        atr_pct = atr_value / close if close else 0.0
        if atr_pct < cfg["atr_min_pct"] or atr_pct > cfg["atr_max_pct"]:
            continue
        rsi_now = float(row["rsi14"])
        body_ratio = _body_ratio(row)
        if direction == "LONG":
            if not (45 < rsi_now < 70 and body_ratio >= cfg["candle_body_min_pct"]):
                continue
        else:
            if not (30 < rsi_now < 55 and body_ratio >= cfg["candle_body_min_pct"]):
                continue
        ict_direction = "bullish" if direction == "LONG" else "bearish"
        fvgs = detect_fvgs(frame, idx, frame["atr14"], cfg["fvg_max_age_candles"], cfg["fvg_min_gap_atr"])
        breaker = detect_breaker_block(
            frame,
            idx,
            frame["atr14"],
            ict_direction,
            cfg["breaker_confirm_candles"],
            cfg["breaker_max_age_candles"],
        )
        zone = find_confluence_zone(breaker, fvgs, ict_direction) if breaker else None
        matched_fvg = _matching_fvg(fvgs, breaker, ict_direction) if breaker else None
        if zone is None or matched_fvg is None:
            continue
        if direction == "LONG":
            trigger = (
                float(row["low"]) <= zone[1]
                and close > zone[0]
                and 45 < rsi_now < 70
                and body_ratio >= cfg["candle_body_min_pct"]
            )
        else:
            trigger = (
                float(row["high"]) >= zone[0]
                and close < zone[1]
                and 30 < rsi_now < 55
                and body_ratio >= cfg["candle_body_min_pct"]
            )
        if not trigger:
            continue
        stop_loss, tp1, tp2, risk_reward = _risk_levels(direction, close, zone, atr_value, cfg)
        if risk_reward < cfg["min_risk_reward"]:
            continue
        confidence_conditions = {
            "ema_slopes_aligned": (
                float(row["ema21_slope"]) > 0 and float(row["ema55_slope"]) > 0
                if direction == "LONG"
                else float(row["ema21_slope"]) < 0 and float(row["ema55_slope"]) < 0
            ),
            "large_fvg": float(matched_fvg["gap_size"]) > 1.0 * atr_value,
            "fresh_breaker": idx - int(breaker["formed_at"]) <= 12,
            "healthy_rsi": 50 <= rsi_now <= 65 if direction == "LONG" else 35 <= rsi_now <= 50,
            "strong_rejection_body": body_ratio >= 0.60,
        }
        confidence, label = compute_confidence_c(confidence_conditions)
        if confidence < cfg["min_confidence"]:
            continue
        signals.append(
            SignalEvent(
                symbol=symbol,
                direction=direction,
                timestamp=timestamp,
                entry=close,
                stop_loss=stop_loss,
                tp1=tp1,
                tp2=tp2,
                risk_reward=risk_reward,
                confidence=confidence,
                confidence_label=label,
                atr_15m=atr_value,
                reasons=_reasons(direction, zone),
                confidence_conditions=confidence_conditions,
                target_rr=risk_reward,
                target_note=f"Strategy C target 1:{risk_reward:.2f}",
                strategy_name=STRATEGY_NAME,
                execution_timeframe=cfg["timeframe_entry"],
                tp1_position_pct=cfg["tp1_position_pct"],
                tp2_position_pct=cfg["tp2_position_pct"],
            )
        )
        last_signal_by_direction[direction] = timestamp
    return signals

``````

## backtest/roi.py
```python
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

``````

## backtest/resample.py
```python
"""OHLCV candle resampling helpers."""

from __future__ import annotations

import pandas as pd


def resample_ohlcv(df: pd.DataFrame, interval_minutes: int) -> pd.DataFrame:
    """Resample a UTC OHLCV frame to a larger minute interval."""
    if df.empty:
        return df.copy()
    frame = df.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame.sort_values("timestamp", inplace=True)
    resampled = (
        frame.set_index("timestamp")
        .resample(f"{interval_minutes}min", label="left", closed="left")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna()
        .reset_index()
    )
    return resampled


def interval_to_minutes(interval: str) -> int:
    """Convert Bybit minute interval notation to integer minutes."""
    return int(interval)

``````

## backtest/optimize_strategy_b_roi.py
```python
"""ROI-first optimizer for Strategy B."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from backtest.data_fetcher import BybitDataError, CacheDataError, fetch_all_symbols, set_data_source
from backtest.resample import resample_ohlcv
from backtest.roi import RoiMetrics, generate_roi_report, roi_metrics, simulate_roi_all
from backtest.strategy import load_config
from backtest.strategy_b import compute_indicators_b, evaluate_signals_b


DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "DOGEUSDT", "ATOMUSDT", "INJUSDT"]


@dataclass(frozen=True)
class RoiCandidate:
    params: dict[str, Any]
    metrics: RoiMetrics
    accepted: bool
    score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize Strategy B for 5x ROI targets.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--symbols", nargs="+")
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--api-base-url")
    parser.add_argument("--cache-dir")
    parser.add_argument("--output", default="strategy_b_roi_optimization.csv")
    parser.add_argument("--report", default="strategy_b_roi_report.txt")
    parser.add_argument("--max-runs", type=int)
    return parser.parse_args()


def candidate_grid() -> list[dict[str, Any]]:
    candidates = []
    for entry_tf in ["30", "60"]:
        for min_confidence in [75, 85, 95]:
            for cooldown_hours in [24, 48]:
                for fvg_min_gap_atr in [0.5, 0.8]:
                    candidates.append(
                        {
                            "timeframe_entry": entry_tf,
                            "timeframe_trend": "240",
                            "min_confidence": min_confidence,
                            "cooldown_hours": cooldown_hours,
                            "fvg_min_gap_atr": fvg_min_gap_atr,
                            "candle_body_min_pct": 0.60,
                            "tp1_risk_multiplier": 1.0,
                            "tp2_risk_multiplier": 2.0,
                            "tp1_position_pct": 0.70,
                            "tp2_position_pct": 0.30,
                        }
                    )
    return candidates


def max_bars(timeframe: str, hold_days: int) -> int:
    return int((hold_days * 24 * 60) / int(timeframe))


def prepare_frames(raw: dict[str, dict], symbol: str, entry_tf: str) -> tuple:
    entry = resample_ohlcv(raw[symbol]["15"], int(entry_tf)) if entry_tf == "30" else raw[symbol]["60"]
    trend = resample_ohlcv(raw[symbol]["60"], 240)
    return entry, trend


def evaluate_candidate(raw: dict[str, dict], symbols: list[str], base_cfg: dict[str, Any], params: dict[str, Any]):
    cfg = dict(base_cfg)
    cfg.update(params)
    eval_data: dict[str, dict] = {}
    signals = []
    for symbol in symbols:
        entry, trend = prepare_frames(raw, symbol, cfg["timeframe_entry"])
        entry_ind, trend_ind = compute_indicators_b(entry, trend, cfg)
        eval_data[symbol] = {cfg["timeframe_entry"]: entry_ind}
        symbol_signals = evaluate_signals_b(symbol, entry_ind, trend_ind, cfg["cooldown_hours"], cfg)
        for signal in symbol_signals:
            signal.execution_timeframe = cfg["timeframe_entry"]
            signal.tp1_position_pct = cfg["tp1_position_pct"]
            signal.tp2_position_pct = cfg["tp2_position_pct"]
        signals.extend(symbol_signals)
    roi_results = simulate_roi_all(
        signals,
        eval_data,
        leverage=5,
        max_bars_by_timeframe={cfg["timeframe_entry"]: max_bars(cfg["timeframe_entry"], 5)},
    )
    return roi_results


def accepted(metrics: RoiMetrics) -> bool:
    return (
        metrics.roi100_hit_rate >= 60
        and metrics.trades >= 20
        and metrics.total_expectancy_r > 0
        and metrics.max_consecutive_roi100_failures <= 5
    )


def score(metrics: RoiMetrics) -> float:
    if metrics.trades == 0:
        return -9999
    return (
        metrics.roi100_hit_rate * 3
        + metrics.roi200_hit_rate
        + metrics.avg_expectancy_r * 50
        - metrics.max_consecutive_roi100_failures * 4
        + min(metrics.trades, 80) * 0.2
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
        raw = fetch_all_symbols(symbols, ["15", "60"], args.days, cache_only=args.cache_only)
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
        roi_results = evaluate_candidate(raw, symbols, config["strategy_b"], params)
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
    generate_roi_report(best_roi_results, "Strategy B ROI best candidate", 5, args.report)
    winners = [item for item in ranked if item.accepted]
    if winners:
        config["strategy_b"].update(winners[0].params)
        Path("strategy_b_candidate_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        return 0
    print("No Strategy B ROI candidate passed acceptance gates.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

``````

## backtest/optimize_strategy_c_roi.py
```python
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

``````

## backtest/simulator.py
```python
"""Trade simulation engine for generated SignalEvent objects."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backtest.strategy import SignalEvent


@dataclass
class TradeResult:
    signal: SignalEvent
    outcome: str
    pnl_r: float
    bars_held: int
    exit_price: float
    exit_timestamp: pd.Timestamp


def reward_r(entry: float, target: float, risk: float) -> float:
    """Return target distance in R units."""
    return abs(target - entry) / risk if risk else 0.0


def partial_tp1_r(signal: SignalEvent, risk: float) -> float:
    """Configured TP1 position exits; the remainder is protected at breakeven."""
    return round(signal.tp1_position_pct * reward_r(signal.entry, signal.tp1, risk), 3)


def blended_tp2_r(signal: SignalEvent, risk: float) -> float:
    """Configured portions exit at TP1 and TP2."""
    return round(
        signal.tp1_position_pct * reward_r(signal.entry, signal.tp1, risk)
        + signal.tp2_position_pct * reward_r(signal.entry, signal.tp2, risk),
        3,
    )


def simulate_trade(signal: SignalEvent, future_candles: pd.DataFrame) -> TradeResult:
    """Walk forward through future candles and resolve a trade."""
    candles = future_candles.sort_values("timestamp").head(96).reset_index(drop=True)
    risk = abs(signal.entry - signal.stop_loss)
    tp1_only_r = partial_tp1_r(signal, risk)
    tp2_hit_r = blended_tp2_r(signal, risk)
    tp1_hit = False
    breakeven_stop = signal.entry

    for index, candle in candles.iterrows():
        high = float(candle["high"])
        low = float(candle["low"])
        timestamp = pd.Timestamp(candle["timestamp"])
        bars_held = index + 1

        if signal.direction == "LONG":
            if not tp1_hit:
                if low <= signal.stop_loss:
                    return TradeResult(signal, "STOP_HIT", -1.0, bars_held, signal.stop_loss, timestamp)
                if high >= signal.tp1:
                    tp1_hit = True
                    if high >= signal.tp2:
                        return TradeResult(signal, "TP2_HIT", tp2_hit_r, bars_held, signal.tp2, timestamp)
            else:
                if low <= breakeven_stop:
                    return TradeResult(signal, "TP1_ONLY", tp1_only_r, bars_held, breakeven_stop, timestamp)
                if high >= signal.tp2:
                    return TradeResult(signal, "TP2_HIT", tp2_hit_r, bars_held, signal.tp2, timestamp)
        else:
            if not tp1_hit:
                if high >= signal.stop_loss:
                    return TradeResult(signal, "STOP_HIT", -1.0, bars_held, signal.stop_loss, timestamp)
                if low <= signal.tp1:
                    tp1_hit = True
                    if low <= signal.tp2:
                        return TradeResult(signal, "TP2_HIT", tp2_hit_r, bars_held, signal.tp2, timestamp)
            else:
                if high >= breakeven_stop:
                    return TradeResult(signal, "TP1_ONLY", tp1_only_r, bars_held, breakeven_stop, timestamp)
                if low <= signal.tp2:
                    return TradeResult(signal, "TP2_HIT", tp2_hit_r, bars_held, signal.tp2, timestamp)

    if candles.empty:
        return TradeResult(signal, "OPEN", 0.0, 0, signal.entry, signal.timestamp)
    last = candles.iloc[-1]
    return TradeResult(signal, "OPEN", tp1_only_r if tp1_hit else 0.0, len(candles), float(last["close"]), pd.Timestamp(last["timestamp"]))


def simulate_all(
    signals: list[SignalEvent],
    data: dict[str, dict[str, pd.DataFrame]],
) -> list[TradeResult]:
    """Run trade simulation for all signals."""
    results: list[TradeResult] = []
    for signal in signals:
        timeframe = getattr(signal, "execution_timeframe", "15")
        df = data[signal.symbol][timeframe]
        future = df[pd.to_datetime(df["timestamp"], utc=True) > signal.timestamp]
        results.append(simulate_trade(signal, future))
    return results

``````

