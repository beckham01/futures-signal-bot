"""Strategy B: 15m FVG + breaker block confluence."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from backtest.ict_patterns import detect_breaker_block, detect_fvgs, find_confluence_zone, zones_overlap
from backtest.indicators import atr, ema, ema_slope, rsi, volume_sma
from backtest.regime import is_btc_regime_trending
from backtest.strategy import SignalEvent, load_config
from config import strategy_filters

LOGGER = logging.getLogger(__name__)
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


def _passes_strategy_filter(symbol: str, direction: str, timestamp: pd.Timestamp) -> bool:
    normalized_symbol = symbol.upper()
    normalized_direction = direction.upper()
    whitelist = {item.upper() for item in strategy_filters.STRATEGY_B_WHITELIST}
    directions_allowed = {item.upper() for item in strategy_filters.STRATEGY_B_DIRECTIONS_ALLOWED}
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


def _passes_regime_filter(
    symbol: str,
    direction: str,
    timestamp: pd.Timestamp,
    fetch_hourly=None,
) -> bool:
    """Strategy B only: reject signals unless BTC is in a 'net-trending' regime."""
    if is_btc_regime_trending(timestamp, fetch_hourly=fetch_hourly):
        return True
    LOGGER.debug(
        "Signal rejected by strategy filter",
        extra={
            "strategy_name": STRATEGY_NAME,
            "symbol": symbol.upper(),
            "direction": direction.upper(),
            "timestamp": timestamp,
            "reason": "regime_chop",
        },
    )
    return False


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

    # Only invoked by is_btc_regime_trending on an actual cache miss (i.e. the
    # first time a given day is seen). Ignores the narrow start_ms/end_ms it is
    # called with and instead returns one memoized fetch spanning the full
    # entry-data range. This avoids repeatedly overwriting the shared
    # BTCUSDT_60 cache file with narrow windows (data_fetcher's cache write
    # replaces the whole file rather than merging).
    btc_hourly_cache: dict[str, pd.DataFrame] = {}

    def _fetch_btc_hourly_for_regime(symbol_: str, interval_: str, start_ms_: int, end_ms_: int) -> pd.DataFrame:
        if "df" not in btc_hourly_cache:
            from backtest.data_fetcher import fetch_klines

            start_ms = int((entry["timestamp"].min() - pd.Timedelta(days=12)).timestamp() * 1000)
            end_ms = int(entry["timestamp"].max().timestamp() * 1000)
            btc_hourly_cache["df"] = fetch_klines("BTCUSDT", "60", start_ms, end_ms)
        return btc_hourly_cache["df"]

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
        if not _passes_strategy_filter(symbol, bias, timestamp):
            continue
        if not _passes_regime_filter(symbol, bias, timestamp, fetch_hourly=_fetch_btc_hourly_for_regime):
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
