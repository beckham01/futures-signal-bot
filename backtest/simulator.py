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
