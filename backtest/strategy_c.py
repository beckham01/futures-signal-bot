"""Strategy C: 4h FVG + breaker block confluence."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from backtest.ict_patterns import detect_breaker_block, detect_fvgs, find_confluence_zone, zones_overlap
from backtest.indicators import atr, ema, ema_slope, rsi
from backtest.strategy import SignalEvent, load_config
from config import strategy_filters

LOGGER = logging.getLogger(__name__)
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


def _passes_strategy_filter(symbol: str, direction: str, timestamp: pd.Timestamp) -> bool:
    normalized_symbol = symbol.upper()
    normalized_direction = direction.upper()
    whitelist = {item.upper() for item in strategy_filters.STRATEGY_C_WHITELIST}
    directions_allowed = {item.upper() for item in strategy_filters.STRATEGY_C_DIRECTIONS_ALLOWED}
    if normalized_symbol not in whitelist:
        LOGGER.debug(
            "Signal rejected by strategy filter",
            extra={
                "strategy_name": STRATEGY_NAME,
                "symbol": normalized_symbol,
                "direction": normalized_direction,
                "timestamp": timestamp,
                "reason": "symbol_not_whitelisted",
            },
        )
        return False
    if normalized_direction not in directions_allowed:
        LOGGER.debug(
            "Signal rejected by strategy filter",
            extra={
                "strategy_name": STRATEGY_NAME,
                "symbol": normalized_symbol,
                "direction": normalized_direction,
                "timestamp": timestamp,
                "reason": "direction_filtered",
            },
        )
        return False
    return True


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
        if not _passes_strategy_filter(symbol, direction, timestamp):
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
