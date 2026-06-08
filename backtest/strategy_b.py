"""Strategy B: Daily Momentum Continuation."""

from __future__ import annotations

from typing import Any

import pandas as pd

from backtest.indicators import atr, ema, ema_slope, rsi, volume_sma
from backtest.strategy import SignalEvent, load_config

STRATEGY_NAME = "strategy_b_daily_momentum"


def strategy_b_config(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = load_config()["strategy_b"]
    if overrides:
        cfg.update(overrides)
    return cfg


def compute_indicators_b(df_15m: pd.DataFrame, cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    """Add Strategy B indicator columns to one 15m candle frame."""
    cfg = cfg or strategy_b_config()
    df = df_15m.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    df["ema55"] = ema(df["close"], cfg["ema_slow"])
    df["ema55_slope"] = ema_slope(df["ema55"], cfg["ema_slope_lookback"])
    df["rsi14"] = rsi(df["close"], cfg["rsi_period"])
    df["atr14"] = atr(df, cfg["atr_period"])
    df["volume_sma20"] = volume_sma(df["volume"], cfg["volume_sma_period"])
    df["atr_expanding"] = df["atr14"] > df["atr14"].shift(3)
    return df


def get_swing_high(df_15m: pd.DataFrame, idx: int, lookback: int = 8) -> float:
    """Return highest high of the previous lookback candles before idx."""
    return float(df_15m.iloc[idx - lookback : idx]["high"].max())


def get_swing_low(df_15m: pd.DataFrame, idx: int, lookback: int = 8) -> float:
    """Return lowest low of the previous lookback candles before idx."""
    return float(df_15m.iloc[idx - lookback : idx]["low"].min())


def compute_confidence_b(conditions: dict[str, bool]) -> tuple[int, str]:
    weights = {
        "ema55_slope_strong": 20,
        "strong_volume": 20,
        "strong_rsi": 15,
        "strong_candle_body": 15,
        "clean_breakout": 15,
        "atr_expanding": 15,
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


def _risk_levels(direction: str, row: pd.Series, cfg: dict[str, Any]) -> tuple[float, float, float, float]:
    entry = float(row["close"])
    atr_15m = float(row["atr14"])
    if direction == "LONG":
        stop_loss = float(row["low"]) - cfg["sl_atr_buffer"] * atr_15m
        risk = entry - stop_loss
        tp1 = entry + cfg["tp1_risk_multiplier"] * risk
        tp2 = entry + cfg["tp2_risk_multiplier"] * risk
    else:
        stop_loss = float(row["high"]) + cfg["sl_atr_buffer"] * atr_15m
        risk = stop_loss - entry
        tp1 = entry - cfg["tp1_risk_multiplier"] * risk
        tp2 = entry - cfg["tp2_risk_multiplier"] * risk
    risk_reward = abs(tp2 - entry) / risk if risk > 0 else 0.0
    return stop_loss, tp1, tp2, risk_reward


def _reasons(direction: str, volume_ratio: float, breakout_margin_r: float) -> list[str]:
    breakout = "swing high" if direction == "LONG" else "swing low"
    return [
        f"15m price broke recent {breakout}",
        f"Volume {volume_ratio:.2f}x average",
        "RSI momentum confirms breakout",
        f"Breakout margin {breakout_margin_r:.2f} ATR",
    ]


def evaluate_signals_b(
    symbol: str,
    df_15m: pd.DataFrame,
    cooldown_hours: int = 4,
    cfg: dict[str, Any] | None = None,
) -> list[SignalEvent]:
    """Evaluate Strategy B signals for one symbol using 15m candles only."""
    cfg = cfg or strategy_b_config()
    if "ema55" not in df_15m.columns or "volume_sma20" not in df_15m.columns:
        df_15m = compute_indicators_b(df_15m, cfg)

    signals: list[SignalEvent] = []
    last_signal_by_direction: dict[str, pd.Timestamp] = {}
    cooldown = pd.Timedelta(hours=cooldown_hours)
    lookback = int(cfg["swing_lookback"])
    trend_lookback = int(cfg.get("trend_lookback", cfg["ema_slope_lookback"]))
    required = ["ema55", "rsi14", "atr14", "volume_sma20", "ema55_slope"]

    for idx in range(max(lookback, trend_lookback, cfg["ema_slope_lookback"], 3), len(df_15m)):
        row = df_15m.iloc[idx]
        prev = df_15m.iloc[idx - 1]
        if any(pd.isna(row.get(name)) for name in required) or pd.isna(prev.get("rsi14")):
            continue
        timestamp = pd.Timestamp(row["timestamp"])
        close = float(row["close"])
        open_ = float(row["open"])
        high = float(row["high"])
        low = float(row["low"])
        ema55_value = float(row["ema55"])
        ema55_prev = float(df_15m.iloc[idx - trend_lookback]["ema55"])
        atr_15m = float(row["atr14"])
        volume_sma = float(row["volume_sma20"])
        volume_ratio = float(row["volume"]) / volume_sma if volume_sma else 0.0
        atr_pct = atr_15m / close if close else 0.0
        body_ratio = _body_ratio(row)
        swing_high = get_swing_high(df_15m, idx, lookback)
        swing_low = get_swing_low(df_15m, idx, lookback)

        directions = []
        if (
            close > ema55_value
            and ema55_value > ema55_prev
            and close > swing_high
            and volume_ratio > cfg["volume_spike_threshold"]
            and float(row["rsi14"]) > 55
            and float(row["rsi14"]) > float(prev["rsi14"])
            and close > open_
            and body_ratio >= cfg["candle_body_min_pct"]
        ):
            directions.append("LONG")
        if (
            close < ema55_value
            and ema55_value < ema55_prev
            and close < swing_low
            and volume_ratio > cfg["volume_spike_threshold"]
            and float(row["rsi14"]) < 45
            and float(row["rsi14"]) < float(prev["rsi14"])
            and close < open_
            and body_ratio >= cfg["candle_body_min_pct"]
        ):
            directions.append("SHORT")

        if not directions or atr_pct > cfg["atr_max_pct"] or atr_pct < cfg["atr_min_pct"]:
            continue

        for direction in directions:
            last_same_direction = last_signal_by_direction.get(direction)
            if last_same_direction is not None and timestamp - last_same_direction < cooldown:
                continue
            stop_loss, tp1, tp2, risk_reward = _risk_levels(direction, row, cfg)
            if risk_reward < cfg["min_risk_reward"]:
                continue

            breakout_distance = (close - swing_high) if direction == "LONG" else (swing_low - close)
            breakout_margin_r = breakout_distance / atr_15m if atr_15m else 0.0
            slope_strength = abs((ema55_value - float(df_15m.iloc[idx - 3]["ema55"])) / ema55_value) if ema55_value else 0.0
            confidence_conditions = {
                "ema55_slope_strong": slope_strength > 0.001,
                "strong_volume": volume_ratio > cfg["volume_strong_threshold"],
                "strong_rsi": float(row["rsi14"]) > 60 if direction == "LONG" else float(row["rsi14"]) < 40,
                "strong_candle_body": body_ratio >= cfg["candle_body_strong_pct"],
                "clean_breakout": breakout_margin_r > cfg["breakout_margin_atr"],
                "atr_expanding": bool(row.get("atr_expanding", False)),
            }
            confidence, label = compute_confidence_b(confidence_conditions)
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
                    atr_15m=atr_15m,
                    reasons=_reasons(direction, volume_ratio, breakout_margin_r),
                    confidence_conditions=confidence_conditions,
                    target_rr=risk_reward,
                    target_note=f"Strategy B target 1:{risk_reward:.2f}",
                    strategy_name=STRATEGY_NAME,
                )
            )
            last_signal_by_direction[direction] = timestamp

    return signals
