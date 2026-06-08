"""Signal generation logic for the futures pullback strategy."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from backtest.indicators import atr, ema, ema_slope, rsi, volume_sma


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


@dataclass
class SignalEvent:
    symbol: str
    direction: str
    timestamp: pd.Timestamp
    entry: float
    stop_loss: float
    tp1: float
    tp2: float
    risk_reward: float
    confidence: int
    confidence_label: str
    atr_15m: float
    reasons: list[str]
    confidence_conditions: dict[str, bool] = field(default_factory=dict)
    target_rr: float | None = None
    target_note: str = ""
    strategy_name: str = "strategy_a_trend_pullback"


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def strategy_config(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = load_config()["strategy"]
    if overrides:
        cfg.update(overrides)
    return cfg


def compute_indicators(
    df_15m: pd.DataFrame,
    df_1h: pd.DataFrame,
    cfg: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add all required indicator columns in place and return both frames."""
    cfg = cfg or strategy_config()
    for df in (df_15m, df_1h):
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df.sort_values("timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)
        df["ema21"] = ema(df["close"], cfg["ema_fast"])
        df["ema55"] = ema(df["close"], cfg["ema_slow"])
        df["rsi14"] = rsi(df["close"], cfg["rsi_period"])
        df["atr14"] = atr(df, cfg["atr_period"])
    df_15m["volume_sma20"] = volume_sma(df_15m["volume"], cfg["volume_sma_period"])
    df_1h["ema55_slope"] = ema_slope(df_1h["ema55"], 3)
    df_15m["atr_expanding"] = df_15m["atr14"] > df_15m["atr14"].shift(3)
    return df_15m, df_1h


def compute_confidence(conditions: dict[str, bool]) -> tuple[int, str]:
    """Apply confidence weights and return score and label."""
    weights = {
        "trend_rsi_strong": 20,
        "rsi_clean_cross": 15,
        "strong_volume": 15,
        "ema55_slope": 15,
        "candle_close_quality": 10,
        "atr_expanding": 10,
        "no_opposing_signal": 15,
    }
    score = sum(weight for key, weight in weights.items() if conditions.get(key, False))
    if score >= 85:
        label = "STRONG"
    elif score >= 70:
        label = "HIGH"
    else:
        label = "MODERATE"
    return int(score), label


def _as_float(row: pd.Series, name: str) -> float:
    return float(row[name])


def _trend_direction(row_1h: pd.Series) -> str | None:
    close = _as_float(row_1h, "close")
    ema21 = _as_float(row_1h, "ema21")
    ema55 = _as_float(row_1h, "ema55")
    rsi14 = _as_float(row_1h, "rsi14")
    if close > ema21 > ema55 and 45 <= rsi14 <= 75:
        return "LONG"
    if close < ema21 < ema55 and 25 <= rsi14 <= 55:
        return "SHORT"
    return None


def _closed_1h_frame(df_1h: pd.DataFrame) -> pd.DataFrame:
    closed = df_1h.copy()
    closed["available_at"] = pd.to_datetime(closed["timestamp"], utc=True) + pd.Timedelta(hours=1)
    return closed


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
    risk_reward = abs(tp2 - entry) / abs(entry - stop_loss)
    return stop_loss, tp1, tp2, risk_reward


def _apply_extended_target(
    direction: str,
    entry: float,
    stop_loss: float,
    tp2: float,
    risk_reward: float,
    confidence: int,
    cfg: dict[str, Any],
) -> tuple[float, float, str]:
    """Optionally extend TP2 when configured signal quality allows it."""
    extended_cfg = cfg.get("extended_targets", {})
    if not extended_cfg.get("enabled", False):
        return tp2, risk_reward, f"Base target 1:{risk_reward:.2f}"

    base_rr = float(extended_cfg.get("base_rr", cfg.get("min_risk_reward", risk_reward)))
    target_rr = base_rr
    note = f"Base target 1:{base_rr:.2f}"
    min_score = int(extended_cfg.get("high_confidence_min_score", 95))
    high_confidence_rr = float(extended_cfg.get("high_confidence_rr", base_rr))
    if confidence >= min_score and high_confidence_rr > base_rr:
        target_rr = high_confidence_rr
        note = f"Extended target 1:{target_rr:.2f} because confidence >= {min_score}"

    risk = abs(entry - stop_loss)
    if direction == "LONG":
        adjusted_tp2 = entry + risk * target_rr
    else:
        adjusted_tp2 = entry - risk * target_rr
    return adjusted_tp2, target_rr, note


def _quality_close(row: pd.Series, direction: str) -> bool:
    high = _as_float(row, "high")
    low = _as_float(row, "low")
    close = _as_float(row, "close")
    candle_range = high - low
    if candle_range <= 0:
        return False
    position = (close - low) / candle_range
    return position >= 0.70 if direction == "LONG" else position <= 0.30


def _build_reasons(direction: str, row_15m: pd.Series, row_1h: pd.Series, volume_ratio: float) -> list[str]:
    arrow = "above" if direction == "LONG" else "below"
    slope = "upward" if direction == "LONG" else "downward"
    return [
        f"EMA21 {arrow} EMA55 on 1h trend filter",
        f"15m RSI recovered through trigger threshold",
        f"Volume spike {volume_ratio:.2f}x average",
        f"EMA55 (1h) sloping {slope}",
    ]


def evaluate_signals(
    symbol: str,
    df_15m: pd.DataFrame,
    df_1h: pd.DataFrame,
    cooldown_hours: int = 4,
    cfg: dict[str, Any] | None = None,
) -> list[SignalEvent]:
    """Evaluate all qualifying signals for one symbol."""
    cfg = cfg or strategy_config()
    if "ema21" not in df_15m.columns or "ema21" not in df_1h.columns:
        df_15m, df_1h = compute_indicators(df_15m, df_1h, cfg)

    entry_frame = df_15m.sort_values("timestamp").reset_index(drop=True).copy()
    trend_frame = _closed_1h_frame(df_1h.sort_values("timestamp").reset_index(drop=True))
    merged = pd.merge_asof(
        entry_frame,
        trend_frame,
        left_on="timestamp",
        right_on="available_at",
        suffixes=("_15m", "_1h"),
        direction="backward",
    )

    signals: list[SignalEvent] = []
    last_signal_by_direction: dict[str, pd.Timestamp] = {}
    last_opposing_signal: dict[str, pd.Timestamp] = {}
    cooldown = pd.Timedelta(hours=cooldown_hours)

    for i, row in merged.iterrows():
        if i == 0 or pd.isna(row.get("timestamp_1h")):
            continue
        prev = merged.iloc[i - 1]
        required = ["ema21_15m", "rsi14_15m", "atr14_15m", "volume_sma20", "ema21_1h", "ema55_1h", "rsi14_1h"]
        if any(pd.isna(row.get(name)) for name in required) or pd.isna(prev.get("rsi14_15m")):
            continue

        trend_row = pd.Series(
            {
                "close": row["close_1h"],
                "ema21": row["ema21_1h"],
                "ema55": row["ema55_1h"],
                "rsi14": row["rsi14_1h"],
            }
        )
        direction = _trend_direction(trend_row)
        if direction is None:
            continue

        timestamp = pd.Timestamp(row["timestamp_15m"])
        last_same_direction = last_signal_by_direction.get(direction)
        if last_same_direction is not None and timestamp - last_same_direction < cooldown:
            continue

        entry = _as_float(row, "close_15m")
        atr_15m = _as_float(row, "atr14_15m")
        ema21_15m = _as_float(row, "ema21_15m")
        volume_ratio = _as_float(row, "volume_15m") / _as_float(row, "volume_sma20")
        pullback = abs(entry - ema21_15m) <= cfg["pullback_atr_tolerance"] * atr_15m
        atr_pct = atr_15m / entry if entry else 0

        if direction == "LONG":
            crossed = _as_float(prev, "rsi14_15m") < cfg["rsi_long_threshold"] < _as_float(row, "rsi14_15m")
            price_confirmed = entry > ema21_15m
        else:
            crossed = _as_float(prev, "rsi14_15m") > cfg["rsi_short_threshold"] > _as_float(row, "rsi14_15m")
            price_confirmed = entry < ema21_15m

        if not (pullback and crossed and volume_ratio > cfg["volume_spike_threshold"] and price_confirmed):
            continue

        stop_loss, tp1, tp2, risk_reward = _risk_levels(direction, entry, atr_15m, cfg)
        if risk_reward < cfg["min_risk_reward"]:
            continue
        if atr_pct > cfg["atr_max_pct"] or atr_pct < cfg["atr_min_pct"]:
            continue

        opposing_direction = "SHORT" if direction == "LONG" else "LONG"
        last_opposing = last_opposing_signal.get(opposing_direction)
        no_opposing = last_opposing is None or timestamp - last_opposing >= cooldown
        slope = _as_float(row, "ema55_slope") if not pd.isna(row.get("ema55_slope")) else 0.0
        confidence_conditions = {
            "trend_rsi_strong": 50 <= _as_float(row, "rsi14_1h") <= 65 if direction == "LONG" else 35 <= _as_float(row, "rsi14_1h") <= 50,
            "rsi_clean_cross": abs(_as_float(row, "rsi14_15m") - _as_float(prev, "rsi14_15m")) >= 2.0,
            "strong_volume": volume_ratio > cfg["volume_strong_threshold"],
            "ema55_slope": slope > 0 if direction == "LONG" else slope < 0,
            "candle_close_quality": _quality_close(pd.Series({"high": row["high_15m"], "low": row["low_15m"], "close": row["close_15m"]}), direction),
            "atr_expanding": bool(row.get("atr_expanding", False)),
            "no_opposing_signal": no_opposing,
        }
        confidence, label = compute_confidence(confidence_conditions)
        if confidence < cfg["min_confidence"]:
            continue
        tp2, risk_reward, target_note = _apply_extended_target(
            direction,
            entry,
            stop_loss,
            tp2,
            risk_reward,
            confidence,
            cfg,
        )

        signal = SignalEvent(
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
            reasons=_build_reasons(direction, row, trend_row, volume_ratio),
            confidence_conditions=confidence_conditions,
            target_rr=risk_reward,
            target_note=target_note,
        )
        signals.append(signal)
        last_signal_by_direction[direction] = timestamp
        last_opposing_signal[direction] = timestamp

    return signals
