"""Shared ICT pattern detection helpers for FVG and breaker block strategies.

All functions are backward-looking relative to eval_idx. Pattern confirmation
never uses candles that would not have existed when eval_idx was evaluated.
"""

from __future__ import annotations

import pandas as pd


Zone = tuple[float, float]


def zones_overlap(zone_a: Zone, zone_b: Zone) -> bool:
    """Return True when two inclusive price zones share any range."""
    return zone_a[0] <= zone_b[1] and zone_b[0] <= zone_a[1]


def overlap_zone(zone_a: Zone, zone_b: Zone) -> Zone:
    """Return the overlapping range of two zones."""
    return (max(zone_a[0], zone_b[0]), min(zone_a[1], zone_b[1]))


def is_price_in_zone(price: float, zone: Zone) -> bool:
    """Return True when price is inside zone bounds."""
    return zone[0] <= price <= zone[1]


def _fvg_filled(fvg: dict, df: pd.DataFrame, eval_idx: int) -> bool:
    start = int(fvg["formed_at"]) + 2
    if start > eval_idx:
        return False
    zone_low, zone_high = fvg["zone"]
    closes = df.iloc[start : eval_idx + 1]["close"].astype(float)
    if fvg["type"] == "bullish":
        return bool((closes <= zone_low).any())
    return bool((closes >= zone_high).any())


def detect_fvgs(
    df: pd.DataFrame,
    eval_idx: int,
    atr: pd.Series,
    max_age_candles: int = 48,
    min_gap_atr_multiplier: float = 0.2,
) -> list[dict]:
    """Return active unfilled FVGs visible at eval_idx, newest first."""
    if eval_idx < 3 or df.empty:
        return []
    candles = df
    fvgs: list[dict] = []
    max_center = min(eval_idx - 2, len(candles) - 2)
    min_center = max(1, eval_idx - max_age_candles)
    for center in range(max_center, min_center - 1, -1):
        if pd.isna(atr.iloc[center]) or float(atr.iloc[center]) <= 0:
            continue
        prev = candles.iloc[center - 1]
        nxt = candles.iloc[center + 1]
        min_gap = float(atr.iloc[center]) * min_gap_atr_multiplier
        fvg_type = None
        zone: Zone | None = None
        if float(prev["high"]) < float(nxt["low"]):
            zone = (float(prev["high"]), float(nxt["low"]))
            fvg_type = "bullish"
        elif float(prev["low"]) > float(nxt["high"]):
            zone = (float(nxt["high"]), float(prev["low"]))
            fvg_type = "bearish"
        if zone is None or zone[1] - zone[0] < min_gap:
            continue
        fvg = {
            "type": fvg_type,
            "zone": zone,
            "formed_at": center,
            "filled": False,
            "gap_size": zone[1] - zone[0],
        }
        fvg["filled"] = _fvg_filled(fvg, candles, eval_idx)
        if not fvg["filled"]:
            fvgs.append(fvg)
    return fvgs


def mark_filled_fvgs(fvgs: list[dict], df: pd.DataFrame, eval_idx: int) -> list[dict]:
    """Return FVG copies with filled status updated through eval_idx."""
    candles = df
    marked = []
    for fvg in fvgs:
        updated = dict(fvg)
        updated["filled"] = _fvg_filled(updated, candles, eval_idx)
        marked.append(updated)
    return marked


def _is_swing_low(df: pd.DataFrame, idx: int, confirm: int) -> bool:
    low = float(df.iloc[idx]["low"])
    left = df.iloc[idx - confirm : idx]["low"].astype(float)
    right = df.iloc[idx + 1 : idx + confirm + 1]["low"].astype(float)
    return bool((left > low).all() and (right > low).all())


def _is_swing_high(df: pd.DataFrame, idx: int, confirm: int) -> bool:
    high = float(df.iloc[idx]["high"])
    left = df.iloc[idx - confirm : idx]["high"].astype(float)
    right = df.iloc[idx + 1 : idx + confirm + 1]["high"].astype(float)
    return bool((left < high).all() and (right < high).all())


def _breaker_mitigated(direction: str, zone: Zone, df: pd.DataFrame, start: int, eval_idx: int) -> bool:
    if start > eval_idx:
        return False
    closes = df.iloc[start : eval_idx + 1]["close"].astype(float)
    if direction == "bullish":
        return bool((closes < zone[0]).any())
    return bool((closes > zone[1]).any())


def detect_breaker_block(
    df: pd.DataFrame,
    eval_idx: int,
    atr: pd.Series,
    direction: str,
    swing_confirm_candles: int = 3,
    max_age_candles: int = 96,
) -> dict | None:
    """Return the newest valid unmitigated breaker block for direction."""
    if direction not in {"bullish", "bearish"}:
        raise ValueError("direction must be 'bullish' or 'bearish'")
    if eval_idx < swing_confirm_candles * 2 + 1 or df.empty:
        return None
    candles = df
    latest_candidate = min(eval_idx - swing_confirm_candles, len(candles) - swing_confirm_candles - 1)
    earliest_candidate = max(swing_confirm_candles, eval_idx - max_age_candles)
    for swing_idx in range(latest_candidate, earliest_candidate - 1, -1):
        if pd.isna(atr.iloc[swing_idx]) or float(atr.iloc[swing_idx]) <= 0:
            continue
        if direction == "bullish":
            if not _is_swing_low(candles, swing_idx, swing_confirm_candles):
                continue
            moved_away = candles.iloc[swing_idx + 1 : eval_idx + 1]["high"].astype(float).max() >= (
                float(candles.iloc[swing_idx]["low"]) + float(atr.iloc[swing_idx])
            )
            candle_filter = candles.iloc[:swing_idx]
            opposing = candle_filter[candle_filter["close"].astype(float) < candle_filter["open"].astype(float)]
        else:
            if not _is_swing_high(candles, swing_idx, swing_confirm_candles):
                continue
            moved_away = candles.iloc[swing_idx + 1 : eval_idx + 1]["low"].astype(float).min() <= (
                float(candles.iloc[swing_idx]["high"]) - float(atr.iloc[swing_idx])
            )
            candle_filter = candles.iloc[:swing_idx]
            opposing = candle_filter[candle_filter["close"].astype(float) > candle_filter["open"].astype(float)]
        if not moved_away or opposing.empty:
            continue
        block_idx = int(opposing.index[-1])
        block = candles.iloc[block_idx]
        zone = (float(block["low"]), float(block["high"]))
        if _breaker_mitigated(direction, zone, candles, swing_idx + 1, eval_idx):
            continue
        return {
            "type": direction,
            "zone": zone,
            "formed_at": swing_idx,
            "source_index": block_idx,
            "mitigated": False,
        }
    return None


def find_confluence_zone(breaker: dict, fvgs: list[dict], direction: str) -> Zone | None:
    """Return the first overlap between breaker and a matching-direction FVG."""
    if not breaker:
        return None
    for fvg in fvgs:
        if fvg.get("type") != direction:
            continue
        if zones_overlap(breaker["zone"], fvg["zone"]):
            return overlap_zone(breaker["zone"], fvg["zone"])
    return None
